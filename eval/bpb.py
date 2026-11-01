"""
Likelihood evaluation of a pretraining checkpoint: validation bits per byte on code (the summed next-token loss in
bits over the UTF-8 bytes of the targets; arXiv 2101.00027 §3.1), the measure the peak learning-rate calibration
compares its runs by (paper §5.1). `evaluate` writes RUNS/<run>/eval-<step>.json.

Each code corpus of the file-level stage is scored on a systematic sample of its packed validation rows (arXiv
2207.14255 §2) at the stage's context; the corpora are pooled by the stage's training row shares. Targets of 0 bytes
(such as special tokens) are excluded from both the loss sum and the byte count.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from corpus.layout import packed_rows, stage_rows
from corpus.loader import IGNORE_INDEX, PretrainingBatches
from corpus.mixture import FILE_LEVEL_STAGE
from corpus.tokenizer import get_tokenizer
from model.modeling import CausalLM
from paths import RUNS
from train.checkpoint import checkpoint_path, load_model

# Validation tokens per corpus (not ablated here), or the whole split if smaller
VAL_TOKENS = 80 * 2**19


@torch.no_grad()
def code_bpb(model: CausalLM) -> dict:
    """{"bpb", "corpora": {corpus: {"bits", "bytes", "rows", "weight"}}} of the file-level stage's code corpora, by the
    compiled CausalLM.losses. cu_seqlens is padded to one segment count for every corpus (one compiled shape)."""
    stage, tokenizer, device = FILE_LEVEL_STAGE, get_tokenizer(), torch.device("cuda")
    losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    token_bytes = torch.from_numpy(tokenizer.token_lengths).to(device=device, dtype=torch.int32)
    # VAL_TOKENS' worth of rows in whole micro-batches; systematic, as the packer orders each shard's rows longest first
    most = -(-VAL_TOKENS // (stage.micro_batch_size * stage.seq_len)) * stage.micro_batch_size
    loaders, sampled = {}, {}
    for name in stage.code:
        num_rows = packed_rows(name, "val", tokenizer, stage.seq_len)
        n = sampled[name] = min(num_rows, most)
        order = np.concatenate([np.arange(n, dtype=np.int64) * num_rows // n,
                                np.full(-n % stage.micro_batch_size, -1, dtype=np.int64)])
        loaders[name] = PretrainingBatches({name: 1.0}, "val", tokenizer, {name: order}, seq_len=stage.seq_len,
                                           micro_batch_size=stage.micro_batch_size, device=device)
    num_segments = max(loader.num_segments for loader in loaders.values())
    rows = stage_rows(tokenizer, stage)
    weights = stage.shares(rows, sum(rows.values()))
    corpora = {}
    for name, loader in loaders.items():
        total = torch.zeros(2, dtype=torch.float64, device=device)
        for x, y, cu_seqlens, segments, _ in loader:
            cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[-1:].expand(num_segments + 1 - cu_seqlens.numel())])
            y = y.flatten()
            nbytes = torch.where(y == IGNORE_INDEX, 0, token_bytes[y.clamp_min(0)])
            nats = torch.where(nbytes > 0, losses(x, y.view_as(x), cu_seqlens, segments), 0.0)
            total += torch.stack([nats.double().sum(), nbytes.double().sum()])
        nats, nbytes = total.tolist()
        corpora[name] = {"bits": nats / math.log(2), "bytes": nbytes, "rows": sampled[name], "weight": weights[name]}
    # val rows drawn with the training row weights w: Σ w bits / rows over Σ w bytes / rows
    bpb = (sum(c["weight"] * c["bits"] / c["rows"] for c in corpora.values())
           / sum(c["weight"] * c["bytes"] / c["rows"] for c in corpora.values()))
    return {"bpb": bpb, "corpora": corpora}


def evaluation_path(run_name: str, step: int) -> Path:
    return RUNS / run_name / f"eval-{step}.json"


def evaluate(run_name: str, step: int) -> None:
    """code_bpb of a run's checkpoint at step, written to evaluation_path(run_name, step)."""
    model = load_model(checkpoint_path(run_name, step), device=torch.device("cuda"))
    evaluation_path(run_name, step).write_text(json.dumps({"run_name": run_name, "step": step} | code_bpb(model)))
