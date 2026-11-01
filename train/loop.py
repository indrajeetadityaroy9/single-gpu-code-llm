"""
The training loop for pretraining (paper §5.1) and SFT (paper §5.2.1). Pretraining minimizes the next-token loss
over the 8,192-token file-level stage, then the 32,768-token repository-level stage; packed rows contain
fill-in-the-middle examples in PSM order (DeepSeek-Coder, arXiv 2401.14196 §3.1; paper §3.3). SFT starts from a
pretrained checkpoint and trains on the response tokens, measuring the loss on held-out rows at intervals. Outputs, in
RUNS/<run_name>/: checkpoints, a JSON-lines record per step (metrics.append) and the per-tensor diagnostics of
train/diagnostics.py.
"""

import gc
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch

import metrics
from corpus.loader import IGNORE_INDEX
from corpus.tokenizer import get_tokenizer
from model.modeling import accumulated_weights, meta_model, num_matmul_params
from model.ops.cross_entropy import step_shares
from paths import RUNS
from train.checkpoint import checkpoint_path
from train.diagnostics import Diagnostics
from train.optim import optimizers, update
from train.schedule import RunPlan
from train.stages import stage_batches, val_batches

# the seed of the weights' initialization and of the data order
SEED = 0


def train(*, run: RunPlan, name: str, checkpoints: frozenset[int], validations: frozenset[int],
          base: Path | None) -> None:
    """Train the base model (model/modeling.py meta_model) on CUDA by plan `run` as run `name`, in a new directory
    RUNS/<name>/.

    With `base` (a checkpoint path), training starts from its weights with fresh optimizer state; otherwise from
    initialization. The weights are saved after each step count in `checkpoints`, and the held-out loss of the last
    stage (validate) is recorded after each step count in `validations`."""
    (RUNS / name).mkdir()
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    model = meta_model()
    model.to_empty(device=device)
    model.init_weights(seed=SEED)
    # fp32 weight gradients the GEMMs accumulate into across micro-batches (model/ops/linear.py); update zeroes them
    accumulated = accumulated_weights(model)
    for p in accumulated:
        p.grad = torch.zeros_like(p)
    adamw, muon = optimizers(model, lr=run.learning_rate, weight_decay=run.weight_decay)
    if base is not None:  # its weights only, memory-mapped
        model.load_state_dict(torch.load(base, map_location="cpu", mmap=True)["model"])

    stage = run.stage_at(0)
    model.set_stage(stage)
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    read = partial(stage_batches, tokenizer, run=run, seed=SEED, device=device)
    train_iter = read(stage=stage)
    batches = [next(train_iter) for _ in range(run.gradient_accumulation_steps)]
    diagnostics = Diagnostics(model, adamw, muon, run_name=name, max_steps=run.max_steps)
    matmul_params, hidden_size = num_matmul_params(model), model.config.hidden_size
    metrics.append(name, {"kind": "run", "step": 0, "plan": asdict(run)})

    # A step's values reach the host without blocking and are logged after the next step launches (no per-step drain);
    # pending: the last step's log() arguments
    pending, previous_end = None, torch.cuda.Event(enable_timing=True)
    previous_end.record()

    def log(step: int, learning_rate: float, model_flops: int, values: torch.Tensor, measured: bool,
            start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        """Once a step's values reach the host: write its side-file rows and its train record."""
        end.synchronize()
        values = values.numpy()
        dt = start.elapsed_time(end) / 1000
        metrics.append(name, {"kind": "train", "step": step, "loss": float(values[0]), "learning_rate": learning_rate,
                              "grad_norm": float(values[1]), "step_seconds": dt,
                              "tokens_per_second": run.batch_size / dt, "mfu": model_flops / dt / metrics.PEAK_FLOPS}
                       | diagnostics.record(step, values, measured))

    # the last stage's val split, read whole (SFT's held-out prompts)
    val_losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    val_rows = val_batches(tokenizer, stage=run.last_stage, device=device) if validations else None

    @torch.no_grad()
    def validate(step: int) -> None:
        """The mean loss over the val split's labels (an SFT split's: each response and its closing BOS), recorded."""
        total = torch.zeros((), dtype=torch.float64, device=device)
        labeled = torch.zeros((), dtype=torch.int64, device=device)
        for input_ids, labels, cu_seqlens, segments, _ in val_rows:
            total += val_losses(input_ids, labels, cu_seqlens, segments).sum(dtype=torch.float64)
            labeled += (labels != IGNORE_INDEX).sum()
        metrics.append(name, {"kind": "val", "step": step, "loss": total.item() / labeled.item(),
                              "labels": labeled.item()})

    for step in range(run.max_steps + 1):
        if step in validations:  # the next step's time starts after it
            validate(step)
            previous_end = torch.cuda.Event(enable_timing=True)
            previous_end.record()
        if step in checkpoints:  # the weights after `step` steps, at the context they trained in
            torch.save({"model": model.state_dict(), "config": asdict(model.config)}, checkpoint_path(name, step))
            gc.collect()
        if step == run.max_steps:
            break
        if run.stage_at(step) is not stage:
            # repo-level stage: new context and compiled graph; its batches were read the step before
            stage = run.stage_at(step)
            model.set_stage(stage)
            compiled = torch.compile(model, dynamic=False, fullgraph=True)

        # the next step's micro-batches are read during this step's backward (the repo-level stage's if it starts it)
        reading = step + 1 < run.max_steps
        if reading and run.stage_at(step + 1) is not stage:
            train_iter = read(stage=run.stage_at(step + 1))
        learning_rate = run.learning_rate_at(step)
        for group in (*adamw.param_groups, *muon.param_groups):
            group["lr"] = learning_rate
        diagnostics.begin_step(step)
        losses, upcoming, model_flops = [], [], 0
        for (input_ids, labels, cu_seqlens, segments, lengths), share in zip(
                batches, step_shares([b[1] for b in batches]), strict=True):
            loss, _, lse = compiled(input_ids, labels, cu_seqlens, segments)
            loss = loss * share
            losses.append(loss.detach())
            diagnostics.micro_batch(labels, lse)
            loss.backward()
            model_flops += metrics.executed_model_flops(lengths, matmul_params=matmul_params, hidden_size=hidden_size,
                                                        windows=model.window_sizes)
            if reading:
                upcoming.append(next(train_iter))
        grad_norm = update(model, (adamw, muon), accumulated)
        model.refresh_gemm_weights()
        values, measured = diagnostics.end_step(step, torch.stack(losses).sum(), grad_norm)
        batches = upcoming
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if pending is not None:
            log(*pending)
        pending, previous_end = (step, learning_rate, model_flops, values, measured, previous_end, end), end
        if step == 0:
            # collector frozen and off after the first step, collecting only after saves
            gc.collect()
            gc.freeze()
            gc.disable()
    log(*pending)
    diagnostics.flush()
