"""Termination probe: measures how often a post-trained checkpoint finishes its reasoning (paper §5.2.1, §6.1, §6.4).

A response is closed (the paper's finished answer) once it writes "</think>"; the reasoning parser splits the text at
the last "</think>" into reasoning and answer (eval/official_reasoning_parser.py). A response that never closes has no
answer and fails. A loop is reasoning that keeps repeating the same text (eval/loop.py; paper §6.1).

Input: the checkpoint's --chat export (inference/export_hf.py: bf16 weights, fp32 output layer, end-of-sequence = BOS).
vLLM samples each prompt of two fixed sets once, offline:

- evalplus: the first HumanEval+ tasks as EvalPlus's OpenAI backend sends them (one user message: codegen.py's
  instruction prefix, then the task prompt stripped in a python fence), rendered by the export's chat template as
  `vllm serve` renders them for scripts/eval_posttrained.sh;
- val: held-out SFT problems (corpus/posttrain/build.py), drawn without replacement with default_rng(SEED); each
  prompt is BOS followed by the context ids the SFT trained on (<split>.context: the prompt and THINK).

Sampling is the sampled mode of scripts/eval_posttrained.sh (SAMPLING), seed SEED + the prompt's index. Generation
runs in rounds of CHUNK tokens with prefix caching (`generate`); each round's seed is offset by SEED_STRIDE, so only
the first CHUNK tokens equal a one-shot generation's. A response ends with finish reason "stop" when it emits BOS,
"length" at its token limit, or "loop" when the loop rule stops it after a round.

The probe (`main`, then `score`) samples PROMPTS prompts per set with up to the context length minus the prompt as new
tokens. Per set (`summary`): five exclusive outcomes (closed and ended; closed at the limit; unclosed at the limit;
unclosed and loop-stopped; unclosed and ended), reasoning-length percentiles p50/p90/p95 (an unclosed response counts
as all reasoning), the median answer length of closed responses, among closed answers the share holding a fenced code
block and the share ending with BOS, the mean `repetition` of the responses that never closed, and for evalplus the
pass on EvalPlus's base and plus tests over all responses and over closed ones (evalplus.evaluate's
check_correctness on the answer as EvalPlus's codegen sanitizes it; an unclosed answer is empty). `score` runs in a
CPU process apart from the engine: in a process forked from it, EvalPlus's 4 GiB address-space limit fails large
allocations.

The temperature sensitivity (`sensitivity`, paper §5.2.1) samples SENSITIVITY_PROMPTS prompts per set at SAMPLING and
at temperature 1.0, each reasoning stopped at END_THINK or at SENSITIVITY_TOKENS new tokens, and counts per set the
closed responses and the unclosed ones that loop (`repetition` >= LOOP).

    python -m eval.official_termination <model dir> <generated json>
    python -m eval.official_termination --score <generated json> <out json>
    python -m eval.official_termination --sensitivity <model dir> <out json>
"""

import json
import os
import re
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from corpus.layout import TOKEN_DTYPE, split_path
from corpus.posttrain.sft import END_THINK, NAME
from corpus.tokenizer import get_tokenizer
from eval.loop import CHUNK, END, LOOP, REPEAT_TAIL, repetition

PROMPTS, SENSITIVITY_PROMPTS, SENSITIVITY_TOKENS = 128, 64, 16384
SEED, SEED_STRIDE = 0, 1_000_003
SAMPLING = {"temperature": 0.6, "top_p": 0.95}
TEMPERATURES = {"protocol": SAMPLING, "t1.0": SAMPLING | {"temperature": 1.0}}
# EvalPlus's chat instruction (evalplus/codegen.py:222, a local of run_codegen)
INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves the following problem in a markdown "
                      "code block:")
FENCE = re.compile(r"```[^\n]*\n.*?```", re.S)
OUTCOMES = ("closed_ended", "closed_limit", "unclosed_limit", "unclosed_loop_stopped", "unclosed_ended")


def evalplus_prompts(template, n: int) -> list[dict]:
    from evalplus.data import get_human_eval_plus

    tasks = list(get_human_eval_plus().items())[:n]
    return [{"id": task_id, "ids": template.encode(template.apply_chat_template(
        [{"role": "user", "content": f"{INSTRUCTION_PREFIX}\n```python\n{task['prompt'].strip()}\n```"}],
        add_generation_prompt=True, tokenize=False), add_special_tokens=False)} for task_id, task in tasks]


def val_prompts(n: int) -> list[dict]:
    tok = get_tokenizer()
    path = split_path(NAME, "val", tok)
    tokens = np.memmap(path.with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r")
    offsets = np.memmap(path.with_suffix(".idx"), dtype=np.int64, mode="r")
    contexts = np.fromfile(path.with_suffix(".context"), dtype=np.int64)
    return [{"id": i, "ids": [tok.bos_token_id, *tokens[offsets[i]:offsets[i] + contexts[i]].tolist()]}
            for i in sorted(np.random.default_rng(SEED).choice(len(contexts), size=n, replace=False).tolist())]


def prompt_sets(llm, n: int) -> dict[str, list[dict]]:
    return {"evalplus": evalplus_prompts(llm.get_tokenizer(), n), "val": val_prompts(n)}


def engine(model: Path):
    """vLLM's offline engine at the export's context, seeded, with prefix caching (the rounds of `generate`), on 0.85
    of the GPU's memory. The share sizes the KV cache, and so the batching: the probe's record ran at this share, the
    temperature-sensitivity record at a smaller one beside a training job, so `sensitivity` reproduces that record in
    distribution, not token for token."""
    from vllm import LLM

    context = json.loads((model / "config.json").read_text())["max_position_embeddings"]
    return LLM(str(model), trust_remote_code=True, dtype="bfloat16", max_model_len=context, seed=SEED,
               enable_prefix_caching=True, gpu_memory_utilization=0.85), context


def response(ids: list[int], text: str, finish: str, bos: int, decode) -> dict:
    """One response's ending: tokens (the closing BOS not counted), closure, the reasoning through the last END."""
    ended = finish == "stop"  # the only stop condition: eos (BOS)
    n = len(ids) - (ids[-1:] == [bos])
    closures = text.count(END)
    lo, hi = 1, n  # the shortest prefix holding every END (decode(ids[:k]).count(END) only grows with k)
    while closures and lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if decode(ids[:mid]).count(END) < closures else (lo, mid)
    reasoning = lo if closures else n
    answer = text.rpartition(END)[2].strip() if closures else ""
    outcome = ("closed" if closures else "unclosed") + (
        "_ended" if ended else "_loop_stopped" if finish == "loop" and not closures else "_limit")
    return {"tokens": n, "finish": finish, "outcome": outcome, "closed": closures > 0,
            "ended": ended, "reasoning_tokens": reasoning, "answer_tokens": n - reasoning,
            "fenced": bool(FENCE.search(answer)), "answer": answer,
            "repetition": None if closures else repetition(ids[:n])}


def generate(llm, prompts: list[list[int]], sampling: list[dict], seeds: list[int], limits: list[int],
             stop: str | None = None) -> list[tuple[list[int], str, str]]:
    """Each prompt's (generated ids, their text, finish), sampled in CHUNK-token rounds, each round's prompt the
    prompt and every token so far (prefix caching reuses its KV): finish "stop" at eos (BOS) or at `stop` (text cut
    after its first occurrence), "length" at the prompt's limit, "loop" for a sequence holding neither `stop` nor END
    whose last REPEAT_TAIL tokens have `repetition` >= LOOP after a round. Round r is seeded seed + r * SEED_STRIDE,
    so the first CHUNK tokens equal a one-shot generation's and the rest differ from it (each request restarts its
    seeded generator): chunked and one-shot generations are equal in distribution, not token for token."""
    from vllm import SamplingParams

    tok = get_tokenizer()
    generated, finish = [[] for _ in prompts], [None] * len(prompts)
    active, chunk = list(range(len(prompts))), 0
    while active:
        outputs = llm.generate(
            [{"prompt_token_ids": prompts[k] + generated[k]} for k in active],
            [SamplingParams(**sampling[k], max_tokens=min(CHUNK, limits[k] - len(generated[k])),
                            seed=seeds[k] + chunk * SEED_STRIDE,
                            **({"stop": [stop], "include_stop_str_in_output": True} if stop else {}))
             for k in active], use_tqdm=False)
        remaining = []
        for k, output in zip(active, outputs, strict=True):
            (completion,) = output.outputs
            generated[k] += completion.token_ids
            text = tok.enc.decode([t for t in generated[k] if t != tok.bos_token_id])
            if completion.finish_reason == "stop" or (stop and stop in text):  # `stop` across a round's boundary
                finish[k] = "stop"
            elif len(generated[k]) >= limits[k]:
                finish[k] = "length"
            elif len(generated[k]) >= REPEAT_TAIL and END not in text and repetition(generated[k]) >= LOOP:
                finish[k] = "loop"
            else:
                remaining.append(k)
        active, chunk = remaining, chunk + 1
    out = []
    for ids, f in zip(generated, finish, strict=True):
        text = tok.enc.decode([t for t in ids if t != tok.bos_token_id])
        if stop and stop in text:
            text = text[:text.index(stop) + len(stop)]
        out.append((ids, text, f))
    return out


def summary(rows: list[dict]) -> dict:
    closed = [r for r in rows if r["closed"]]
    unclosed = [r for r in rows if not r["closed"]]
    counts = Counter(r["outcome"] for r in rows)
    out = {"responses": len(rows), "outcomes": {k: counts[k] for k in OUTCOMES},
           "closed": len(closed) / len(rows), "at_length_limit": sum(not r["ended"] for r in rows) / len(rows),
           "reasoning_tokens_p50_p90_p95": np.percentile([r["reasoning_tokens"] for r in rows], [50, 90, 95]).tolist(),
           "answer_tokens_p50_closed": float(np.median([r["answer_tokens"] for r in closed])) if closed else None,
           "fenced_of_closed": sum(r["fenced"] for r in closed) / len(closed) if closed else None,
           "ended_of_closed": sum(r["ended"] for r in closed) / len(closed) if closed else None,
           "repetition_mean_unclosed": statistics.fmean(r["repetition"] for r in unclosed) if unclosed else None}
    if all("base" in r for r in rows):  # the evalplus set
        out["evalplus_pass"] = {k: sum(r[k] for r in rows) / len(rows) for k in ("base", "plus")}
        out["evalplus_pass_closed"] = {k: sum(r[k] for r in closed) / len(closed) if closed else None
                                       for k in ("base", "plus")}
    return out


def evalplus_check(rows: list[dict]) -> None:
    """Each evalplus response's base and plus pass (EvalPlus's codegen sanitize, then evaluate's check_correctness)."""
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.sanitize import sanitize

    problems = get_human_eval_plus()
    expected = get_groundtruth(problems, get_human_eval_plus_hash(), [])
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:  # every core of the CPU set
        futures = [pool.submit(check_correctness, "humaneval", 0, problems[r["id"]],
                               sanitize(r["answer"], entrypoint=problems[r["id"]]["entry_point"]),
                               expected[r["id"]], False, True) for r in rows]
        for r, future in zip(rows, futures, strict=True):
            result = future.result()
            r["base"] = result["base"][0] == "pass"
            r["plus"] = r["base"] and result["plus"][0] == "pass"


def main(model: Path, out: Path) -> None:
    """The probe's generation (the GPU process): the responses, unscored, to `out`; `score` scores them."""
    llm, context = engine(model)
    tok = get_tokenizer()
    flat = [(name, p) for name, every in prompt_sets(llm, PROMPTS).items() for p in every]
    outputs = generate(llm, [p["ids"] for _, p in flat], [SAMPLING] * len(flat),
                       [SEED + i for i in range(len(flat))], [context - len(p["ids"]) for _, p in flat])
    responses = {"evalplus": [], "val": []}
    for (name, p), (ids, text, finish) in zip(flat, outputs, strict=True):
        responses[name].append({"id": p["id"], "prompt_tokens": len(p["ids"])}
                               | response(ids, text, finish, tok.bos_token_id, tok.enc.decode))
    out.write_text(json.dumps({"model": str(model), "context": context, "sampling": SAMPLING | {"seed": SEED},
                               "prompts_per_set": PROMPTS, "chunk": CHUNK, "loop_stop": LOOP,
                               "responses": responses}, indent=1) + "\n")


def score(generated: Path, out: Path) -> None:
    """The probe's scoring (a CPU process): the evalplus answers' pass, then the summaries."""
    report = json.loads(generated.read_text())
    rows = report.pop("responses")
    evalplus_check(rows["evalplus"])
    report["sets"] = {name: summary(r) for name, r in rows.items()} | {"all": summary(sum(rows.values(), []))}
    report["responses"] = rows  # last, after the summaries
    out.write_text(json.dumps(report, indent=1) + "\n")


def sensitivity(model: Path, out: Path) -> None:
    """The temperature sensitivity: per temperature and set, the closed responses and the unclosed ones that loop."""
    llm, _ = engine(model)
    tok = get_tokenizer()
    flat = [(name, p) for name, every in prompt_sets(llm, SENSITIVITY_PROMPTS).items() for p in every]
    runs = [(config, i, name, p) for config in TEMPERATURES for i, (name, p) in enumerate(flat)]
    outputs = generate(llm, [p["ids"] for *_, p in runs], [TEMPERATURES[config] for config, *_ in runs],
                       [SEED + i for _, i, *_ in runs], [SENSITIVITY_TOKENS] * len(runs), stop=END_THINK)
    rows = {config: {"evalplus": [], "val": []} for config in TEMPERATURES}
    for (config, _, name, p), (generated, text, finish) in zip(runs, outputs, strict=True):
        ids = [t for t in generated if t != tok.bos_token_id]
        closed = text.endswith(END_THINK)
        rows[config][name].append({"id": p["id"], "closed": closed, "finish": finish, "tokens": len(ids),
                                   "repetition": None if closed else repetition(ids)})
    out.write_text(json.dumps({
        "model": str(model), "prompts_per_set": SENSITIVITY_PROMPTS, "reasoning_max_tokens": SENSITIVITY_TOKENS,
        "configs": TEMPERATURES,
        "summary": {config: {name: {"responses": len(rs), "closed": sum(r["closed"] for r in rs),
                                    "unclosed_loops": sum(not r["closed"] and r["repetition"] >= LOOP for r in rs)}
                             for name, rs in sets.items()} for config, sets in rows.items()},
        "responses": rows}, indent=1) + "\n")


if __name__ == "__main__":
    command = {"--score": score, "--sensitivity": sensitivity}.get(sys.argv[1])
    if command is None:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        command(Path(sys.argv[2]), Path(sys.argv[3]))
