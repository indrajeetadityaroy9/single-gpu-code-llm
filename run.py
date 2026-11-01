"""
The paper's training experiments, each defined once below: python run.py <experiment>, in this order.

- peak_lr: peak learning-rate calibration of pretraining (paper §5.1, "Batch size and learning rate");
- pretrain: pretraining of the base model (paper §5.1);
- sft_lr: peak learning-rate calibration of SFT (paper §5.2.1, "Learning rate");
- sft: supervised fine-tuning of the base model (paper §5.2.1).

Each run trains the 522M model (d24) into RUNS/<run name>/ (train/loop.py). A calibration writes its record to
results/, from which the later experiments take their learning rates.
"""
import json
import math
import multiprocessing
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import metrics
from corpus.tokenizer import get_tokenizer
from eval.bpb import evaluate, evaluation_path
from model.modeling import meta_model
from paths import RESULTS, RUNS
from train.checkpoint import checkpoint_path
from train.loop import train
from train.schedule import RunPlan, plan, sft_plan
from train.stages import max_tokens_per_param

PEAK_LR_RECORD = RESULTS / "pretrain" / "peak_lr_d24.jsonl"
SFT_LR_RECORD = RESULTS / "posttrain" / "sft_lr.jsonl"


def fresh(target: Callable[..., None], **kwargs) -> None:
    """target(**kwargs) in a new process (its own CUDA context, compiled graphs and garbage collector state); its
    exception is raised here."""
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        pool.submit(target, **kwargs).result()


def peak_lr() -> None:
    """DeepSeek LLM's law learning rate (train/schedule.py plan) was fit for AdamW with a step-wise schedule, so the
    model is trained at 1, 2 and 4 times it on 5 tokens per parameter, the longest run the time budget allows, and each
    run's final checkpoint is scored by its code bits per byte (eval/bpb.py). Record: one line per run."""
    lines = []
    for scale in (1.0, 2.0, 4.0):
        name, run = f"peak_lr-{scale}", plan(model=meta_model(), tokens_per_param=5.0, learning_rate_scale=scale)
        fresh(train, run=run, name=name, checkpoints=frozenset({run.max_steps}), validations=frozenset(), base=None)
        fresh(evaluate, run_name=name, step=run.max_steps)
        bpb = json.loads(evaluation_path(name, run.max_steps).read_text(encoding="utf-8"))["bpb"]
        lines.append({"run_name": name, "config": {"learning_rate_scale": scale}, "measures": {"code": bpb}})
    PEAK_LR_RECORD.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def lr_multiplier() -> float:
    """The pretraining run's multiplier of the law's learning rate: the minimum of the least-squares parabola through
    peak_lr's code bits per byte against u = ln(multiplier) (as arXiv 2409.19913 §3.1); the law carries the change in
    compute to the full run."""
    lines = [json.loads(line) for line in PEAK_LR_RECORD.read_text(encoding="utf-8").splitlines()]
    vertex = metrics.parabola_vertex([math.log(line["config"]["learning_rate_scale"]) for line in lines],
                                     [line["measures"]["code"] for line in lines])
    return math.exp(vertex)


def pretraining() -> RunPlan:
    """The pretraining plan: the most tokens per parameter at which no source is overdrawn (train/stages.py), at the
    calibrated multiplier of the law's learning rate."""
    return plan(model=meta_model(), tokens_per_param=max_tokens_per_param(get_tokenizer()),
                learning_rate_scale=lr_multiplier())


def pretrain() -> None:
    """The base model on the pretraining plan from initialization; its final weights are saved (the start of SFT)."""
    run = pretraining()
    train(run=run, name="pretrain", checkpoints=frozenset({run.max_steps}), validations=frozenset(), base=None)


def sft_lr() -> None:
    """Complete schedules (warmup and linear decay) at 1/10, 1/3 and 1 times the pretraining peak learning rate over
    the same first 2% of the SFT training rows (0.02 epochs, with the base and data order of sft), each scored by its
    held-out loss after its last step. Record: one line per run, with its val records."""
    base = pretraining()
    lines = []
    for scale in (0.1, 1 / 3, 1.0):
        name = f"sft_lr-{scale}"
        run = sft_plan(epochs=0.02, batch_size=base.batch_size, learning_rate=scale * base.learning_rate)
        fresh(train, run=run, name=name, checkpoints=frozenset(), validations=frozenset({run.max_steps}),
              base=checkpoint_path("pretrain", base.max_steps))
        records = map(json.loads, (RUNS / name / "metrics.jsonl").read_text(encoding="utf-8").splitlines())
        lines.append({"run_name": name, "config": {"learning_rate_scale": scale},
                      "records": [r for r in records if r["kind"] == "val"]})
    SFT_LR_RECORD.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def sft_lr_scale(steps: int) -> float:
    """The multiplier of the pretraining peak learning rate for an SFT run of `steps`: the minimum of the parabola
    through sft_lr's final held-out losses against u = ln(multiplier), times (steps / sft_lr's steps)^-0.30. At a fixed
    batch size the optimal learning rate falls with the training length as steps^-0.30 (arXiv 2409.19913 §4.1,
    Llama-1; 0.32 in its §6 Eq. 5). Applying it to SFT is an assumption."""
    lines = [json.loads(line) for line in SFT_LR_RECORD.read_text(encoding="utf-8").splitlines()]
    finals = [max((r for r in line["records"] if r["kind"] == "val"), key=lambda r: r["step"]) for line in lines]
    vertex = metrics.parabola_vertex([math.log(line["config"]["learning_rate_scale"]) for line in lines],
                                     [r["loss"] for r in finals])
    (calibration_steps,) = {r["step"] for r in finals}
    return math.exp(vertex) * (steps / calibration_steps) ** -0.30


def sft() -> None:
    """2 epochs over the SFT data (corpus/posttrain/build.py; Nemotron-Cascade 2's single SFT stage peaks at about 1.5
    epochs, arXiv 2603.19220 §3.1.1) from pretrain's final checkpoint, at its batch size and sft_lr_scale times its
    peak learning rate, the held-out loss measured and the weights saved every 1,000 steps and after the last."""
    base = pretraining()
    run = sft_plan(epochs=2.0, batch_size=base.batch_size, learning_rate=base.learning_rate)
    run = replace(run, learning_rate=sft_lr_scale(run.max_steps) * base.learning_rate)
    steps = frozenset({*range(1000, run.max_steps, 1000), run.max_steps})
    train(run=run, name="sft", checkpoints=steps, validations=steps | {0},
          base=checkpoint_path("pretrain", base.max_steps))


if __name__ == "__main__":
    {"peak_lr": peak_lr, "pretrain": pretrain, "sft_lr": sft_lr, "sft": sft}[sys.argv[1]]()
