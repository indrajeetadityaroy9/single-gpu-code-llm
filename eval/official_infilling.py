"""HumanEval infilling (arXiv 2207.14255) generation for scripts/eval_pretrained.sh: openai/human-eval-infilling ships
the problems and the judge (evaluate_functional_correctness) but no generator, so vLLM generates the samples here from
the export, and the package's CLI judges them: single-line, multi-line and random-span, greedy, one sample a task, in
the prefix-suffix-middle order (PSM).

Prompts are corpus.tokenizer.fim_prompt's PSM token ids, the layout the packer trains (paper §3.3): prefix and
suffix encoded apart, BOS first. A middle ends at BOS (the model's end of text, the paper's EOT) or another special
token, at most `budget` tokens, the benchmark's longest reference middle (the paper names no budget). Samples are
written as the CLI reads them, {"task_id", "completion"} per line.

    python -m eval.official_infilling <model dir> <out dir>
"""

import functools
import json
import sys
from pathlib import Path

from human_eval_infilling.data import read_problems

from corpus.tokenizer import fim_prompt, get_tokenizer

BENCHMARKS = ("single-line", "multi-line", "random-span")


@functools.cache
def budget(name: str) -> int:
    """The benchmark's longest reference middle in tokens."""
    encode = get_tokenizer().enc.encode_ordinary
    return max(len(encode(p["canonical_solution"])) for p in read_problems(name).values())


def main(model: str, out: Path) -> None:
    from vllm import LLM, SamplingParams, TokensPrompt

    tokenizer = get_tokenizer()
    stops = sorted(tokenizer.special_tokens.values())
    llm = LLM(model, trust_remote_code=True, dtype="bfloat16")
    for name in BENCHMARKS:
        problems = list(read_problems(name).values())
        prompts = [fim_prompt(tokenizer, p["prompt"], p["suffix"]) for p in problems]
        outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts],
                               SamplingParams(temperature=0.0, max_tokens=budget(name), stop_token_ids=stops))
        samples = [{"task_id": p["task_id"], "completion": o.outputs[0].text}
                   for p, o in zip(problems, outputs, strict=True)]
        (out / f"{name}-psm.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
