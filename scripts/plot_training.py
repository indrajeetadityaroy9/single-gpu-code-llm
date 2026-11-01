"""
Draw the paper's Figures 2-4 from the pretraining run's own records, as vector PDFs sized for the ICML column
(3.25 in) and text (6.75 in) widths.

- fig_residual (Figure 2, paper §4): the residual scalars lambda_r and lambda_0 of every layer over training. Dots
  mark the layers with value embeddings.
- fig_stability (Figure 3, paper §5.1): the gradient norm with the gradient-clipping threshold, the log-sum-exp of
  the soft-capped logits with its bound c + ln V, and each layer's largest QK-norm scale.
- fig_loss (Figure 4, paper §5.1): the training loss (window means) and the learning-rate schedule.

Inputs: metrics.jsonl (the `run` record's plan, one `train` record per step; train/loop.py) and diagnostics-step.npy,
whose column names are in diagnostics.json (train/diagnostics.py). Text is a Times-metric serif at 8 pt with fonts
embedded as TrueType (no Type 3); per-step traces are rasterized; the PDFs carry no timestamp.

    scripts/container.sh python scripts/plot_training.py <run dir>    # writes figures/fig_{loss,stability,residual}.pdf
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import blended_transform_factory

from model.config import FINAL_LOGIT_SOFTCAPPING, NUM_HIDDEN_LAYERS, VOCAB_SIZE
from model.modeling import has_value_embedding
from train.optim import MAX_GRAD_NORM

# 1-based layers with value embeddings
VE_LAYERS = [i + 1 for i in range(NUM_HIDDEN_LAYERS) if has_value_embedding(i, num_hidden_layers=NUM_HIDDEN_LAYERS)]
COLUMN, TEXT = 3.25, 6.75  # ICML widths, inches
ACCENT = "#D55E00"  # Okabe-Ito vermilion: reference lines
SCALARS = ("loss", "learning_rate", "grad_norm", "max_logsumexp", "mean_square_logsumexp")
OUT = Path(__file__).resolve().parents[1] / "figures"  # the repository's figures/, where the PDFs are written

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Times", "Nimbus Roman"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 200,
})


def load(run: Path) -> dict:
    """Per-step arrays of the run's train records, its tokens per step, the step of the switch to the repository-level
    stage, and the per-step diagnostics."""
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    (plan,) = [r["plan"] for r in records if r["kind"] == "run"]
    a = np.array([[r[k] for k in SCALARS] for r in records if r["kind"] == "train"], dtype=np.float64)
    steps = len(a)
    names = json.loads((run / "diagnostics.json").read_text())
    return {k: a[:, i] for i, k in enumerate(SCALARS)} | {
        "steps": steps, "batch": plan["batch_size"], "stage": plan["repo_level_start"],
        "columns": names["diagnostics-step.npy"], "diagnostics": np.load(run / "diagnostics-step.npy", mmap_mode="r")}


def tokens(d, step) -> np.ndarray:
    """Training tokens (billions) seen after `step`."""
    return (np.asarray(step) + 1) * d["batch"] / 1e9


def windows(y: np.ndarray, w: int, start: int, stop: int, reduce=np.mean):
    """Apply `reduce` to y over [start, stop) in windows of w steps; a last window shorter than w/2 is dropped.
    Returns each window's middle step and its value."""
    mids, vals = [], []
    for a in range(start, stop, w):
        b = min(a + w, stop)
        if b - a < w // 2:
            break
        mids.append((a + b - 1) / 2)
        vals.append(reduce(y[a:b]))
    return np.array(mids), np.array(vals)


def mark_stage(ax, d, label=True):
    """Shade the repository-level (32K) stage."""
    x0, x1 = tokens(d, d["stage"] - 1), tokens(d, d["steps"] - 1)
    ax.axvspan(x0, x1, color="0.92", lw=0, zorder=0)
    if label:
        ax.text(x1, 1.0, "32K", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=6.5, color="0.35")


def tag(ax, letter: str, x: float):
    ax.text(x, 1.02, f"({letter})", transform=ax.transAxes, fontweight="bold")


def save(fig, name: str):
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})  # no timestamp: identical bytes on a rerun
    plt.close(fig)


def fig_loss(d):
    """Training loss (means over 500 steps, 250 in the 32K stage, from step 500) and the learning rate."""
    fig, ax = plt.subplots(figsize=(COLUMN, 2.0))
    mark_stage(ax, d)
    for a, b, w in ((500, d["stage"], 500), (d["stage"], d["steps"], 250)):
        m, v = windows(d["loss"], w, a, b)
        ax.plot(tokens(d, m), v, color="k", lw=1.1)
    ax.set_ylim(0.84, 1.40)
    ax.set_xlim(0, tokens(d, d["steps"] - 1))
    ax.set_xlabel("training tokens (B)")
    ax.set_ylabel("training loss (nats)")
    lr = ax.twinx()
    lr.spines["right"].set_visible(True)
    lr.plot(tokens(d, np.arange(d["steps"])), d["learning_rate"] * 1e3, color="0.45", ls="--", lw=0.9)
    lr.set_ylim(0, 2.6)
    lr.set_ylabel(r"learning rate ($10^{-3}$)", color="0.35")
    lr.tick_params(axis="y", colors="0.35")
    save(fig, "fig_loss")


def fig_stability(d):
    """Gradient norm, log-sum-exp of the soft-capped logits, and each layer's largest QK-norm scale."""
    x = tokens(d, np.arange(d["steps"]))
    fig, axs = plt.subplots(1, 3, figsize=(TEXT, 1.9), gridspec_kw={"wspace": 0.42})
    a, b, c = axs
    a.plot(x, d["grad_norm"], color="0.72", lw=0.3, rasterized=True)
    m, v = windows(d["grad_norm"], 250, 0, d["steps"], np.median)
    a.plot(tokens(d, m), v, color="k", lw=0.9)
    a.axhline(MAX_GRAD_NORM, color=ACCENT, ls="--", lw=0.8)
    a.text(0.98, MAX_GRAD_NORM * 1.12, "clipping threshold", transform=a.get_yaxis_transform(), ha="right",
           fontsize=6.5, color=ACCENT)
    a.set_yscale("log")
    a.set_ylim(0.05, 5)
    a.set_ylabel("gradient norm")
    b.plot(x, d["max_logsumexp"], color="0.72", lw=0.3, rasterized=True, label="maximum")
    m, v = windows(np.sqrt(d["mean_square_logsumexp"]), 250, 0, d["steps"])
    b.plot(tokens(d, m), v, color="k", lw=0.9, label="root mean square")
    bound = FINAL_LOGIT_SOFTCAPPING + math.log(VOCAB_SIZE)
    b.axhline(bound, color=ACCENT, ls="--", lw=0.8)
    b.text(0.98, bound - 0.5, r"bound $c + \ln V$", transform=b.get_yaxis_transform(), ha="right", va="top",
           fontsize=6.5, color=ACCENT)
    b.set_ylim(0, 27)
    b.set_ylabel("log-sum-exp of the logits")
    b.legend(loc="lower right", handlelength=1.2)
    sub = np.arange(0, d["steps"], 50)
    rows = np.asarray(d["diagnostics"][sub])
    cmap = plt.get_cmap("viridis")
    for layer in range(NUM_HIDDEN_LAYERS):
        q = d["columns"].index(f"gain_max/layers.{layer}.self_attn.q_norm_weight")
        k = d["columns"].index(f"gain_max/layers.{layer}.self_attn.k_norm_weight")
        c.plot(tokens(d, sub), np.maximum(rows[:, q], rows[:, k]), color=cmap(layer / (NUM_HIDDEN_LAYERS - 1)),
               lw=0.6)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, NUM_HIDDEN_LAYERS)), ax=c, pad=0.02,
                      fraction=0.06, ticks=[1, NUM_HIDDEN_LAYERS // 2, NUM_HIDDEN_LAYERS])
    cb.set_label("layer", labelpad=1)
    cb.outline.set_linewidth(0.4)
    c.set_ylabel("largest QK-norm scale")
    for ax, letter in zip(axs, "abc"):
        mark_stage(ax, d, label=False)
        ax.set_xlabel("training tokens (B)")
        ax.set_xlim(0, tokens(d, d["steps"] - 1))
        tag(ax, letter, -0.27)
    save(fig, "fig_stability")


def fig_residual(d):
    """The residual scalars lambda_r and lambda_0 of every layer over training (every 50th step). White is the
    initial value (lambda_r = 1, lambda_0 = 0), and each side of it has its own linear color scale. Dots mark the
    layers with value embeddings."""
    sub = np.arange(0, d["steps"], 50)
    rows = np.asarray(d["diagnostics"][sub])
    fig, axs = plt.subplots(1, 2, figsize=(TEXT, 2.0), gridspec_kw={"wspace": 0.42})
    extent = [tokens(d, sub[0]), tokens(d, sub[-1]), 0.5, NUM_HIDDEN_LAYERS + 0.5]
    for ax, key, label, center, cmap, letter in ((axs[0], "resid_lambdas", r"$\lambda_r$", 1.0, "PuOr_r", "a"),
                                                  (axs[1], "x0_lambdas", r"$\lambda_0$", 0.0, "RdBu_r", "b")):
        z = rows[:, [d["columns"].index(f"{key}/{i}") for i in range(NUM_HIDDEN_LAYERS)]].T
        norm = TwoSlopeNorm(vcenter=center, vmin=min(z.min(), center - 1e-3), vmax=max(z.max(), center + 1e-3))
        im = ax.imshow(z, aspect="auto", origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest",
                       rasterized=True)
        cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.05)
        cb.set_label(label, labelpad=2)
        cb.outline.set_linewidth(0.4)
        ax.axvline(tokens(d, d["stage"] - 1), color="k", ls=":", lw=0.7)
        ax.set_yticks([1, *range(4, NUM_HIDDEN_LAYERS + 1, 4)])
        ax.tick_params(axis="y", length=0, pad=7)
        ax.plot(np.full(len(VE_LAYERS), -0.022), VE_LAYERS, "o", color="k", ms=1.8, clip_on=False,
                transform=blended_transform_factory(ax.transAxes, ax.transData))
        ax.set_ylabel("layer")
        ax.set_xlabel("training tokens (B)")
        tag(ax, letter, -0.13)
    save(fig, "fig_residual")


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("run", type=Path)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load(args.run)
    fig_loss(d)
    fig_stability(d)
    fig_residual(d)


if __name__ == "__main__":
    main()
