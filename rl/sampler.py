"""
The RL sampler and the KodCode validation (paper §5.2.2, §6.1). Both generate responses with one vLLM engine and test
each finished response at once in a process pool (rl/verify.py `judge`), so the tests run while generation continues.
The sampler runs in the vLLM image beside the trainer (rl/trainer.py, training image) on the same GPU; the engine
takes UTILIZATION of the GPU's memory, leaving the trainer room for its packed 32K-token rows.

Rollouts (group sampling): PROMPTS train-pool problems (rl/prompts.py usable rows whose context plus MAX_TOKENS fits
the model's context, in a SEED permutation of the pool; batch k is the k-th slice) x GROUP responses each, Lite PPO's
batch (arXiv:2508.08221 §3: 128 prompts x 8, 1,024 responses). Sampling is at temperature 1.0 and top-p 1.0, so
responses follow the trainer's distribution exactly (DAPO samples at 1.0, arXiv:2503.14476 §4.1; ProRL found
temperature 0.6 less stable than 1.2, arXiv:2507.12507 §5.1). The response length limit is MAX_TOKENS: in the run's
first batch (results/runs/sft/onpolicy-7676/rl/train.jsonl, step 0) the SFT model's responses at temperature 1.0
averaged 7,598 tokens and 4.3% (44 of 1,024) reached it; a limit of 8,192 would sit just above the mean length
(truncation above 10-15% destabilizes training, arXiv:2510.13786 App. A.15). Request k of a call has seed base + k.

Loop stopping: every CHECK generated tokens after the first CYCLE_TAIL, a response caught by `stopped` (an exact
cycle, or the evaluation's loop rule, and no END) is aborted with finish "loop". Loops hold 68.9% of the SFT model's
validation tokens (results/runs/sft/onpolicy-7676/eval-model/val: 7.07M of 10.25M); stopping each exact cycle at its
first detectable point removes 23.0% of the tokens and 38.6% of the KV-cache reads. A loop scores 0 either way, since
an exact cycle repeats to the limit.

Reward: 1 when the answer passes all unit tests of its problem, else 0 (no closed reasoning, a failing answer, or a
loop: an exact cycle never ends, so it counts as a failure, not a truncation), plus DAPO's soft overlong penalty
(`overlong`, arXiv:2503.14476 §3.4, Eq. 13; "4,096 tokens as the soft punish cache", §4.1). A problem whose
wrong-program stub passes has trivial tests; its group is marked and not trained on.

KodCode validation (paper §6.1, `--validate`): VALIDATION_PROMPTS problems of the val pool held out from the train
pool (rl/prompts.py `held_out`, `sample` with seed SEED, contexts that fit as above), VALIDATION_SAMPLES responses
each at the evaluation's sampling (eval/official_termination.py SAMPLING: temperature 0.6, top-p 0.95), MAX_TOKENS,
seeds from SEED, the same loop stopping. pass@1 is over all responses; passes on trivial problems are not counted.
It runs on the SFT export and on the trainer's final policy. Every RL batch has new problems, so the batch pass rate
already measures the policy on unseen problems; the validation compares the final policy with the SFT model.

Exchange with the trainer under X (each file written under a temporary name, then renamed):
  X/policy-<k>/   the policy after k updates in the export format; loaded into the engine
  X/batch-<k>.pkl the rollouts of policy k: {"groups": [{"prompt_id", "prompt" (ids), "trivial", "samples": [{"ids"
                  (generated; a loop's up to its check), "finish", "verdict", "reward"}]}], "stats"}
  X/done          written by the trainer after its last update; the sampler then exits

    python -m rl.sampler <model dir> <exchange dir>
    python -m rl.sampler --validate <model dir> <out json>
"""
import argparse
import itertools
import json
import os
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from corpus.parallel import SPAWN
from corpus.tokenizer import get_tokenizer
from eval.loop import END, looping
from eval.official_termination import SAMPLING as VALIDATION_SAMPLING
from rl.prompts import context_ids, held_out, read_pool, sample, usable
from rl.verify import check, judge, stub

PROMPTS, GROUP, SAMPLING, SEED = 128, 8, {"temperature": 1.0, "top_p": 1.0}, 0
# The response length limit (see the module docstring) and DAPO's soft punishment interval below it
MAX_TOKENS, OVERLONG = 16384, 4096
VALIDATION_PROMPTS, VALIDATION_SAMPLES = 512, 4
# Request seeds of batch k: BATCH_SEED + k * PROMPTS * GROUP + request index, disjoint from the validation's 0..2047
BATCH_SEED = 10_000_000
# The loop stop: the check interval in generated tokens (a divisor of eval/loop.py CHUNK, so the evaluation rule's
# checks fall on it), and for `cycle` the longest period tried and the tail that must repeat exactly at the period
CHECK, MAX_PERIOD, CYCLE_TAIL = 128, 2048, 1024
# The engine draws 660 W at 100% utilization on a batch's 1,024 requests; half of the GPU memory leaves the trainer
# room for its packed rows
UTILIZATION = 0.5
# Cores the test workers leave to the engine process, this driver and the trainer
RESERVED_CORES = 5


def cyclic(ids: list[int]) -> bool:
    """Whether a response ends in an exact cycle: its last CYCLE_TAIL tokens repeat at a period p <= MAX_PERIOD
    (ids[i] == ids[i - p]). Only the periods at which the last token recurs can fit, so only those are tried, in
    increasing order."""
    a = np.asarray(ids)
    n = len(a)
    lo = max(n - 1 - MAX_PERIOD, 0)
    for p in (n - 1 - lo - np.flatnonzero(a[lo:n - 1] == a[n - 1]))[::-1].tolist():
        if n < CYCLE_TAIL + p:
            return False
        if np.array_equal(a[n - CYCLE_TAIL:], a[n - CYCLE_TAIL - p:n - p]):
            return True
    return False


def stopped(ids: list[int], decode) -> bool:
    """Whether loop stopping ends a response at these ids (at least CYCLE_TAIL + CHECK of them): it has not closed its
    reasoning (no END), and either it ends in an exact cycle or it meets the evaluation's loop rule (eval/loop.py
    `looping`)."""
    return (cyclic(ids) or looping(ids, decode) is not None) and END not in decode(ids)


def overlong(n: int) -> float:
    """DAPO's soft overlong penalty for an n-token response: 0 up to MAX_TOKENS - OVERLONG tokens, then
    ((MAX_TOKENS - OVERLONG) - n) / OVERLONG, reaching -1 at MAX_TOKENS."""
    return max(-1.0, min(0.0, (MAX_TOKENS - OVERLONG - n) / OVERLONG))


def problems(model: Path, pool: list[dict]) -> tuple[int, list[dict]]:
    """The model's context and the usable rows of `pool` whose BOS, context and MAX_TOKENS fit in it."""
    context = json.loads((model / "config.json").read_text())["max_position_embeddings"]
    return context, [row for row in pool if usable(row) and 1 + row["context_tokens"] + MAX_TOKENS <= context]


def load_engine(model: Path, context: int):
    from vllm import LLM

    return LLM(str(model), trust_remote_code=True, dtype="bfloat16", max_model_len=context, seed=SEED,
               enable_prefix_caching=True, gpu_memory_utilization=UTILIZATION, max_num_seqs=PROMPTS * GROUP)


def rollouts(llm, judges: ProcessPoolExecutor, rows: list[dict], samples: int, sampling: dict,
             seed: int) -> tuple[list[dict], dict]:
    """Samples `samples` responses per row (request k seeded seed + k; a group's requests share the prompt's KV cache
    through prefix caching), stops loops, tests and rewards each response. Returns the groups and generation
    statistics."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    decode = get_tokenizer().enc.decode
    prompts = [context_ids(row) for row in rows]
    stubs = [judges.submit(check, (stub(row), row["test"], row["kind"])) for row in rows]
    requests = [g for g in range(len(rows)) for _ in range(samples)]
    engine = llm.llm_engine
    for k, g in enumerate(requests):
        engine.add_request(str(k), {"prompt_token_ids": prompts[g]},
                           SamplingParams(**sampling, seed=seed + k, max_tokens=MAX_TOKENS,
                                          output_kind=RequestOutputKind.DELTA, detokenize=False))
    ids, finish, verdicts = [[] for _ in requests], [None] * len(requests), [None] * len(requests)
    due = [CYCLE_TAIL + CHECK] * len(requests)

    def done(k: int, how: str) -> None:
        row = rows[requests[k]]
        finish[k] = how
        verdicts[k] = judges.submit(judge, (ids[k], row["names"], row["test"], row["kind"]))

    start = time.time()
    while engine.has_unfinished_requests():
        for o in engine.step():
            k = int(o.request_id)
            if finish[k] is not None:  # aborted
                continue
            ids[k] += o.outputs[0].token_ids
            if o.finished:
                done(k, "stop" if o.outputs[0].finish_reason == "stop" else "length")
            elif len(ids[k]) >= due[k]:
                m = len(ids[k]) // CHECK * CHECK
                due[k] = m + CHECK
                if stopped(ids[k][:m], decode):
                    engine.abort_request([str(k)])
                    ids[k] = ids[k][:m]
                    done(k, "loop")
    generated = time.time() - start
    groups = [{"prompt_id": row["id"], "prompt": prompts[g], "trivial": stubs[g].result() == "pass", "samples": []}
              for g, row in enumerate(rows)]
    for k, g in enumerate(requests):
        verdict = verdicts[k].result()
        groups[g]["samples"].append({"ids": ids[k], "finish": finish[k], "verdict": verdict,
                                     "reward": (verdict == "pass") + overlong(len(ids[k]))})
    return groups, {"generated_tokens": sum(map(len, ids)), "generation_seconds": generated,
                    "finishes": dict(Counter(finish))}


def test_workers() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(len(os.sched_getaffinity(0)) - RESERVED_CORES, mp_context=SPAWN)


def validate(model: Path, out: Path) -> None:
    """Writes the KodCode validation of `model` to `out`: pass@1, the verdict and finish counts, the mean response
    length, the held-out counts (paper §2.3) and each response's verdict."""
    val = read_pool("val")
    kept = held_out(val, read_pool("train"))
    context, pool = problems(model, kept)
    rows = sample(pool, VALIDATION_PROMPTS, SEED)
    llm = load_engine(model, context)
    with test_workers() as judges:
        groups, stats = rollouts(llm, judges, rows, VALIDATION_SAMPLES, VALIDATION_SAMPLING, SEED)
    responses = [{"prompt_id": g["prompt_id"], "sample": j, "finish": s["finish"], "tokens": len(s["ids"]),
                  "verdict": "pass_trivial" if g["trivial"] and s["verdict"] == "pass" else s["verdict"]}
                 for g in groups for j, s in enumerate(g["samples"])]
    verdicts = Counter(r["verdict"] for r in responses)
    report = {"pass@1": verdicts["pass"] / len(responses), "verdicts": dict(verdicts),
              "mean_tokens": stats["generated_tokens"] / len(responses),
              "held_out": {"rows": len(val), "kept": len(kept), "usable": sum(map(usable, kept))}} | stats
    out.write_text(json.dumps(report | {"responses": responses}) + "\n")
    print(json.dumps(report), flush=True)


def main(model: Path, exchange: Path) -> None:
    context, pool = problems(model, read_pool("train"))
    order = np.random.RandomState(SEED).permutation(len(pool))
    llm = load_engine(model, context)
    with test_workers() as judges:
        for k in itertools.count():
            policy = exchange / f"policy-{k}"
            while not policy.exists():
                if (exchange / "done").exists():
                    return
                time.sleep(0.2)
            llm.collective_rpc("reload_weights", kwargs={"weights_path": str(policy)})
            llm.reset_prefix_cache()
            rows = [pool[i] for i in order[k * PROMPTS:(k + 1) * PROMPTS]]
            groups, stats = rollouts(llm, judges, rows, GROUP, SAMPLING, BATCH_SEED + k * PROMPTS * GROUP)
            tmp = exchange / f"batch-{k}.pkl.tmp"
            tmp.write_bytes(pickle.dumps({"groups": groups, "stats": stats}, protocol=5))
            tmp.rename(exchange / f"batch-{k}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path, help="the exchange directory, or with --validate the output json")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    (validate if args.validate else main)(args.model, args.out)
