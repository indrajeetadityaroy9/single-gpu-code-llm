"""Collects the official tools' outputs for one pretrained checkpoint into one record (paper §6.1, §6.2).

Input: the output directory of scripts/eval_pretrained.sh. Output: <dir>/summary.json, a record
in codegen and fim sections. Each problem's pass comes from the tool's own verdict:

| Record | Tool output | Pass |
| codegen.{humaneval,mbpp}_plus.greedy | evalplus.evaluate's `<samples>.eval_results.json` | base_status, plus_status |
| codegen.{humaneval,mbpp}_cpp.{greedy,sampled} | MultiPL-E main.py's `<problem>.results.json.gz` | pass_k.py's: "OK", exit 0 |
| codegen.mbpp_3shot.greedy | lm-evaluation-harness `--log_samples` of mbpp_lf (eval/official_lm_eval/) | its `pass_at_1` |
| fim.<benchmark>/psm.greedy | human-eval-infilling's `<benchmark>-psm.jsonl_results.jsonl` (eval/official_infilling.py) | its `passed` |

Each cell holds pass@1 with its interval (metrics.pass_cell: mean, standard error, and a Wilson interval when there is
one sample per problem). Infilling records cluster their intervals by HumanEval problem (task ids
"<Benchmark>/HumanEval/<i>/L<j>" or ".../L<j>_L<k>" share "HumanEval/<i>"), since one problem yields several infilling
tasks. evalplus_average averages the greedy EvalPlus records. An infilling record also counts its middles that reached
the token limit ("at_budget"); every other middle ended at BOS or another special token. Those middles' tokens are
re-encoded with encode_ordinary to count them.

    python -m eval.official <dir>
"""

import gzip
import json
import sys
from pathlib import Path

import metrics
from corpus.tokenizer import get_tokenizer
from eval.official_infilling import BENCHMARKS, budget


def evalplus(out: Path, dataset: str) -> dict:
    (results,) = (out / "evalplus" / dataset).glob("*.eval_results.json")
    tasks = json.loads(results.read_text())["eval"]
    return {"ids": list(tasks), "base_status": [[r["base_status"] for r in rs] for rs in tasks.values()],
            "plus_status": [[r["plus_status"] for r in rs] for rs in tasks.values()]}


def multipl_e(directory: Path) -> dict:
    """pass_k.py for_file's count: status "OK" and exit code 0; problems in the dataset's order (their index)."""
    files = sorted(directory.glob("*.results.json.gz"), key=lambda p: int(p.name.split("_")[1]))
    problems = [json.loads(gzip.decompress(p.read_bytes())) for p in files]
    return {"ids": [p["name"] for p in problems],
            "status": [["OK" if r["status"] == "OK" and r["exit_code"] == 0 else r["status"] for r in p["results"]]
                       for p in problems]}


def mbpp_3shot(out: Path) -> dict:
    (samples,) = (out / "lm-eval").rglob("samples_mbpp_lf_*.jsonl")
    rows = sorted((json.loads(line) for line in samples.read_text().splitlines()), key=lambda r: r["doc"]["task_id"])
    return {"ids": [str(r["doc"]["task_id"]) for r in rows],
            "status": [["passed" if r["pass_at_1"] == 1 else "failed"] for r in rows]}


def infilling(path: Path, limit: int) -> dict:
    """One judged greedy sample file as {"ids", "result", "completions"}, tasks in file order, and
    the number of middles whose re-encoded tokens reach the budget `limit`."""
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    enc = get_tokenizer().enc.encode_ordinary
    return {"ids": [s["task_id"] for s in samples], "result": [s["result"] for s in samples],
            "completions": [s["completion"] for s in samples],
            "at_budget": sum(len(enc(s["completion"])) >= limit for s in samples), "max_new_tokens": limit}


def record(out: Path) -> dict:
    codegen = {"humaneval_plus": {"greedy": evalplus(out, "humaneval")},
               "mbpp_plus": {"greedy": evalplus(out, "mbpp")},
               **{f"{d}_cpp": {mode: multipl_e(out / "multipl-e" / mode / f"{d}-cpp") for mode in ("greedy", "sampled")}
                  for d in ("humaneval", "mbpp")},
               "mbpp_3shot": {"greedy": mbpp_3shot(out)}}
    for benchmark, modes in codegen.items():
        for r in modes.values():  # the cells: EvalPlus {column: {"pass@1": cell}}, else {"pass@1": cell}
            columns = metrics.pass_columns(benchmark, r)
            r["scores"] = ({c: {"pass@1": metrics.pass_cell(p)} for c, p in columns.items()} if "base_status" in r
                           else {"pass@1": metrics.pass_cell(columns["pass@1"])})
    fims = {}
    for name in BENCHMARKS:
        r = infilling(out / "infilling" / f"{name}-psm.jsonl_results.jsonl", budget(name))
        # clusters: each task's HumanEval problem
        r["scores"] = {"pass@1": metrics.summary([p for (p,) in metrics.pass_columns(name, r)["pass@1"]],
                                                 ["/".join(i.split("/")[1:3]) for i in r["ids"]])}
        fims[f"{name}/psm"] = {"greedy": r}
    return {"codegen": codegen, "fim": fims,
            "evalplus_average": {"greedy": metrics.evalplus_average(codegen["humaneval_plus"]["greedy"],
                                                                   codegen["mbpp_plus"]["greedy"])},
            "tools": json.loads((out / "tools.json").read_text())}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    (out / "summary.json").write_text(json.dumps(record(out), indent=1) + "\n")
