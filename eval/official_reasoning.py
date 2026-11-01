"""Summarizes the EvalPlus results of a post-trained (SFT or RL) checkpoint into summary.json (paper §6.1, §6.3, §6.4).

Input: <dir>/evalplus/, written by scripts/eval_posttrained.sh into its output directory <dir>. It holds, per
benchmark and decoding mode (greedy: 1 sample, temperature 0; sampled: 20 samples at eval/official_termination.py
SAMPLING's temperature, top-p 0.95), the samples' endings (<samples>.endings.jsonl from eval/official_evalplus.py) and
their verdicts (evalplus.evaluate's <samples>.eval_results.json). A sample is matched to its verdict by its problem and
its sanitized solution text.

Output per benchmark and mode:
- the share of finished answers (responses that write "</think>") and the finish reasons;
- the number of responses ended by the server's loop stop (eval/official_loop_stop.py; "loop_stops", found from the
  token ids);
- pass@1 over all samples, and pass@1 over the finished answers only (pooled: each problem weighted by its number of
  finished answers).
"base" scores on the original HumanEval or MBPP tests; "plus" also requires EvalPlus's extra tests to pass.

    python -m eval.official_reasoning <dir>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from eval.official_termination import SAMPLING

MODES = {"greedy": "sgc_openai_temp_0.0", "sampled": f"sgc_openai_temp_{SAMPLING['temperature']}"}
BENCHMARKS = {"humaneval_plus": "humaneval", "mbpp_plus": "mbpp"}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def samples(stem: Path) -> list[dict]:
    """Each sample's ending and verdicts, in the samples file's order (stem: the samples file less ".jsonl")."""
    rows, endings = jsonl(Path(f"{stem}.jsonl")), jsonl(Path(f"{stem}.endings.jsonl"))
    verdicts = defaultdict(list)  # (problem, solution) -> its verdicts, in evaluate's order
    for task_id, results in json.loads(Path(f"{stem}.eval_results.json").read_text())["eval"].items():
        for r in results:
            verdicts[task_id, r["solution"]].append(r)
    out = []
    for row, ending in zip(rows, endings, strict=True):
        r = verdicts[row["task_id"], row["solution"]].pop(0)
        out.append(ending | {"base": r["base_status"] == "pass",
                            "plus": r["base_status"] == "pass" and r["plus_status"] == "pass"})
    return out


def cell(rows: list[dict]) -> dict:
    closed = [r for r in rows if r["closed"]]
    reasons = defaultdict(int)
    for r in rows:
        reasons[r["finish_reason"]] += 1
    return {"problems": len({r["task_id"] for r in rows}), "samples": len(rows), "closed": len(closed) / len(rows),
            "finish_reasons": dict(reasons), "loop_stopped": sum(r["loop_stopped"] for r in rows),
            "pass@1": {k: sum(r[k] for r in rows) / len(rows) for k in ("base", "plus")},
            "pass@1_closed": {k: sum(r[k] for r in closed) / len(closed) if closed else None for k in ("base", "plus")}}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    summary = {benchmark: {mode: cell(samples(out / "evalplus" / dataset / identifier))
                           for mode, identifier in MODES.items()}
               for benchmark, dataset in BENCHMARKS.items()}
    summary["loop_stops"] = {
        "evalplus": sum(c["loop_stopped"] for modes in summary.values() for c in modes.values())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
