"""
The training data of each stage (paper §3.4). Each source's packed rows are read in a seeded random order and
interleaved by the stage's mixture weights (corpus/mixture.py). The module also finds the training length: the most
tokens per parameter at which no source is drawn beyond its prepared rows, so no document is read twice (paper §5.1,
"Training length and stages"). For SFT it reads the held-out rows.
"""

import bisect
from collections.abc import Iterator

import numpy as np
import torch

from corpus.layout import packed_rows, stage_rows
from corpus.loader import PretrainingBatches
from corpus.mixture import FILE_LEVEL_STAGE, MICRO_BATCH_TOKENS, REPO_LEVEL_STAGE, Stage, weights
from corpus.tokenizer import Tokenizer
from model.modeling import meta_model, num_non_embedding_params
from train.schedule import RunPlan, plan


def drawn(stage: Stage, run: RunPlan) -> int:
    """The rows the run draws in `stage`: its steps there times micro-batches per step times rows per micro-batch."""
    steps = run.max_steps - run.repo_level_start if stage is run.last_stage else run.repo_level_start
    return steps * run.gradient_accumulation_steps * stage.micro_batch_size


def stage_batches(tokenizer: Tokenizer, *, stage: Stage, run: RunPlan, seed: int, device: torch.device
                  ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]]:
    """The micro-batches the run draws in `stage`, by the stage's mixture weights.

    Each source's rows are permuted with numpy's RandomState(seed), whose stream is fixed across numpy versions. A
    source drawn beyond its rows (SFT for more than one epoch) is read on through further epochs; epoch e uses a fresh
    permutation seeded with seed + e, so no row repeats within an epoch."""
    rows = stage_rows(tokenizer, stage)
    mixture, draws, _ = weights(stage, rows, drawn(stage, run))
    orders = {corpus: np.concatenate([np.random.RandomState(seed + e).permutation(rows[corpus])
                                      for e in range(max(1, -(-draws[corpus] // rows[corpus])))])
              for corpus in mixture}
    return iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=stage.seq_len,
                                   micro_batch_size=stage.micro_batch_size, device=device))


def val_batches(tokenizer: Tokenizer, *, stage: Stage, device: torch.device) -> PretrainingBatches:
    """Every row of the val split of a single-source stage (SFT_STAGE's held-out rows), in order; the last
    micro-batch is filled with blank rows."""
    (corpus,) = stage.sources
    rows = packed_rows(corpus, "val", tokenizer, stage.seq_len)
    order = np.concatenate([np.arange(rows), np.full(-rows % stage.micro_batch_size, -1)])
    return PretrainingBatches({corpus: 1.0}, "val", tokenizer, {corpus: order}, seq_len=stage.seq_len,
                              micro_batch_size=stage.micro_batch_size, device=device)


def max_tokens_per_param(tokenizer: Tokenizer) -> float:
    """The largest tokens per parameter, in whole micro-batches, at which no source is overdrawn, found by bisection.
    Overdrawing is not monotone in the token count: gradient_accumulation_steps rises in steps with it, so the trained
    tokens can dip. The result fits, but a larger count that also fits may exist."""
    model = meta_model()
    params = num_non_embedding_params(model)
    rows = [(stage, stage_rows(tokenizer, stage)) for stage in (FILE_LEVEL_STAGE, REPO_LEVEL_STAGE)]

    def tokens_per_param(n: int) -> float:
        return n * MICRO_BATCH_TOKENS / params

    def overdrawn(n: int) -> bool:
        # the steps and stages do not depend on the learning rate
        run = plan(model=model, tokens_per_param=tokens_per_param(n), learning_rate_scale=1.0)
        return any(weights(stage, held, drawn(stage, run))[2] for stage, held in rows)

    n = bisect.bisect_left(range(sum(rows[0][1].values()) * FILE_LEVEL_STAGE.seq_len // MICRO_BATCH_TOKENS + 1), True,
                           key=overdrawn)
    return tokens_per_param(n - 1)
