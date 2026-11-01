"""
Metrics and statistics, without torch (paper §5.1, §6.1). The run log RUNS/<run>/metrics.jsonl (append); model FLOPs
and the peak that model FLOPs utilization divides by (MFU, arXiv 2204.02311 App. B); the least-squares parabola of the
learning-rate calibrations; and pass@1, the fraction of problems whose program passes all tests (arXiv 2107.03374
§2.1), with standard errors and 95% intervals (normal, clustered, or Wilson for one program per problem; arXiv
2411.00640).
"""
import json
import statistics
import time
from collections import Counter
from collections.abc import Sequence

import numpy as np
import statsmodels.api as sm
from numpy.typing import ArrayLike
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from paths import RUNS


def append(run_name: str, record: dict) -> None:
    line = json.dumps({"run_name": run_name, "time": time.time(), **record})
    with (RUNS / run_name / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def summary(scores: ArrayLike, clusters: ArrayLike | None = None) -> dict:
    """{"mean", "se", "interval", "n", "clusters"} of question scores by OLS on a constant (arXiv 2411.00640).
    Unclustered: Eq. 1; clustered: Liang-Zeger CR0 without small-sample correction; 95% normal interval (Eq. 3)."""
    s = np.asarray(scores, dtype=np.float64)
    mean = float(s.mean())
    groups = None if clusters is None else np.unique(np.asarray(clusters), return_inverse=True)[1]
    out = {"n": int(s.size), "clusters": None if groups is None else int(groups.max() + 1), "mean": mean}
    if np.ptp(s) == 0:
        return out | {"se": 0.0, "interval": (mean, mean)}
    model = sm.OLS(s, np.ones_like(s))
    fit = (model.fit(use_t=False) if groups is None else
           model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": False}, use_t=False))
    low, high = fit.conf_int()[0]
    return out | {"se": float(fit.bse[0]), "interval": (float(low), float(high))}


def pass_cell(passed: Sequence[Sequence[bool]]) -> dict:
    """A pass@1 table cell: summary of each problem's mean pass over its samples (arXiv 2411.00640 §3.1: a question's
    K answers averaged first, the error over the question means by Eq. 1), and, when every problem has one sample,
    "wilson": the Wilson 95% score interval of the passes (statsmodels proportion_confint), inside [0, 1] and not
    degenerate at 0 or all passes, where summary's normal interval is."""
    cell = summary([statistics.fmean(p) for p in passed])
    if all(len(p) == 1 for p in passed):
        low, high = proportion_confint(sum(p[0] for p in passed), len(passed), alpha=0.05, method="wilson")
        cell["wilson"] = (float(low), float(high))
    return cell


def evalplus_passed(record: dict) -> tuple[list[list[bool]], list[list[bool]]]:
    """An EvalPlus record's base and plus pass outcomes [problem][sample]: base status "pass"; plus, both "pass"."""
    base = [[s == "pass" for s in ss] for ss in record["base_status"]]
    plus = [[s == t == "pass" for s, t in zip(ss, ts, strict=True)]
            for ss, ts in zip(record["base_status"], record["plus_status"], strict=True)]
    return base, plus


def evalplus_average(humaneval: dict, mbpp: dict) -> dict:
    """Qwen3's EvalPlus score (arXiv 2505.09388 §3.3: the average of HumanEval, MBPP, HumanEval+ and MBPP+) of one
    mode's humaneval_plus and mbpp_plus records, as a summary cell. With a task's score t = (base + plus) / 2 of its
    pass@1 (samples averaged first, arXiv 2411.00640 §3.1) the score is ½(mean_HE t + mean_MBPP t): a task's base and
    plus columns are one cluster, the benchmarks independent, so se² = ¼(se²_HE + se²_MBPP), each by summary over the
    task scores (Eq. 1); 95% normal interval (Eq. 3). n counts tasks."""
    cells = []
    for record in (humaneval, mbpp):
        base, plus = evalplus_passed(record)
        scores = [(statistics.fmean(b) + statistics.fmean(p)) / 2 for b, p in zip(base, plus, strict=True)]
        cells.append(summary(scores))
    mean = (cells[0]["mean"] + cells[1]["mean"]) / 2
    se = float(np.hypot(cells[0]["se"], cells[1]["se"]) / 2)
    half = float(norm.ppf(0.975)) * se
    return {"n": cells[0]["n"] + cells[1]["n"], "clusters": None, "mean": mean, "se": se,
            "interval": (mean - half, mean + half)}


def pass_columns(benchmark: str, record: dict) -> dict[str, list[list[bool]]]:
    """A codegen or fim record's pass outcomes per score column, [problem][sample]: EvalPlus base and plus (columns
    humaneval and humaneval_plus, or mbpp and mbpp_plus); else "pass@1" of status "OK" (MultiPL-E) or "passed" (lm-eval
    MBPP), or of result "passed" (human_eval_infilling)."""
    if "base_status" in record:
        return dict(zip((benchmark.removesuffix("_plus"), benchmark), evalplus_passed(record), strict=True))
    if "status" in record:
        return {"pass@1": [[s in ("OK", "passed") for s in ss] for ss in record["status"]]}
    return {"pass@1": [[r == "passed"] for r in record["result"]]}


# The GPU's dense BF16 peak: NVIDIA's H100 SXM5 figure (H100 Tensor Core GPU Architecture whitepaper: 989.4 TFLOP/s,
# 1,978.9 with sparsity; 132 SMs x 4,096 FLOPs per clock x 1.83 GHz), the GH200's GPU being an H100 of 132 SMs. MFU
# (train/loop.py) is the model FLOPs done per second over it.
PEAK_FLOPS = 989.4e12


def executed_model_flops(lengths: np.ndarray, *, matmul_params: int, hidden_size: int, windows: Sequence[int]) -> int:
    """Training FLOPs a micro-batch of segment `lengths` executes: 6 N per position + 12 hidden_size per attended key
    per layer (QK^T and PV, 4 hidden_size per key forward, twice backward; FA4's score recomputation not counted). A
    segment of length L attends Σ_p min(p, window + 1), p = 1..L, keys causally in a layer of that window."""
    lengths = np.asarray(lengths, dtype=np.int64)
    keys = 0
    for window, count in Counter(windows).items():
        m = np.minimum(lengths, window + 1)
        keys += count * int((m * (m + 1) // 2 + (lengths - m) * (window + 1)).sum())
    return 6 * matmul_params * int(lengths.sum()) + 12 * hidden_size * keys


def parabola_vertex(x: Sequence[float], y: Sequence[float]) -> float:
    """The vertex -b / (2a) of the least-squares parabola a x² + b x + c (numpy polyfit)."""
    a, b, _ = np.polyfit(x, y, 2)
    return float(-b / (2 * a))
