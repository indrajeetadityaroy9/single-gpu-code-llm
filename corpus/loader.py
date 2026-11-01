"""
The training data loader (paper §3.2, §3.4): packed sequences from a run's mixture of sources, interleaved
deterministically by the mixture weights and assembled into micro-batches with document masking (cu_seqlens).
"""

import itertools
from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import torch

from corpus.layout import PIECE_COLUMNS, TOKEN_DTYPE, fim_packed, packed_path, split_path
from corpus.mixture import mixture_rows
from corpus.posttrain import sft
from corpus.tokenizer import Tokenizer

# Label of a position excluded from the loss (padding, a BOS that is no target); as an input it becomes BOS
IGNORE_INDEX = -1


class PackedSplit(NamedTuple):
    """A packed split, memory-mapped; fim is None when the split has no FIM examples (layout.fim_packed; an empty
    file cannot be memory-mapped); contexts holds each document's leading tokens that are excluded from the loss (the
    SFT prompts, corpus/posttrain/build.py), None for every other corpus."""
    tokens: np.memmap
    offsets: np.memmap
    pieces: np.memmap
    rows: np.memmap
    fim: np.memmap | None
    contexts: np.memmap | None


def packed(corpus: str, split: str, tokenizer: Tokenizer, seq_len: int) -> PackedSplit:
    """The packed split of a source at sequences of seq_len + 1 tokens (pack.py), memory-mapped in place."""
    path, packing = split_path(corpus, split, tokenizer), packed_path(corpus, split, tokenizer, seq_len)
    return PackedSplit(np.memmap(path.with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r"),
                       np.memmap(path.with_suffix(".idx"), dtype=np.int64, mode="r"),
                       np.memmap(packing.with_suffix(".pieces"), dtype=np.int64, mode="r").reshape(-1, PIECE_COLUMNS),
                       np.memmap(packing.with_suffix(".rows"), dtype=np.int64, mode="r"),
                       np.memmap(packing.with_suffix(".fim"), dtype=TOKEN_DTYPE, mode="r")
                       if fim_packed(corpus, split) else None,
                       np.memmap(path.with_suffix(".context"), dtype=np.int64, mode="r")
                       if corpus == sft.NAME else None)


class PretrainingBatches:
    """Re-iterable micro-batches (input_ids int32, labels int64, cu_seqlens int32 on the device; the segment count as
    a 0-dim int32 tensor and the segment lengths int64 on the host) of a mixture {source: weight}; a piece packed as a
    FIM example (in a FIM-packed train split) is read as that example. cu_seqlens has a fixed num_segments + 1
    entries, unused ones repeating the last offset. Iteration stops, dropping the partial micro-batch, when a source's
    order is exhausted."""

    def __init__(self, mixture: dict[str, float], split: str, tokenizer: Tokenizer, orders: dict[str, np.ndarray], *,
                 seq_len: int, micro_batch_size: int, device: torch.device | str) -> None:
        self.names, self.weights = list(mixture), np.array(list(mixture.values()))
        self.packed = {name: packed(name, split, tokenizer, seq_len) for name in self.names}
        self.orders = [orders[name] for name in self.names]
        self.seq_len, self.micro_batch_size, self.device = seq_len, micro_batch_size, device
        self.bos = tokenizer.bos_token_id
        # micro_batch_size x (the most pieces in a sequence + 1 for the padding)
        self.num_segments = micro_batch_size * (
            max(int(np.diff(packing.rows).max()) for packing in self.packed.values()) + 1)
        self.contexts = any(packing.contexts is not None for packing in self.packed.values())

    def assemble(self, out: np.ndarray, context: np.ndarray, name: str, row: int) -> list[int]:
        """Write a source's packed sequence into out (padding stays in place): each piece after a BOS (IGNORE_INDEX
        when the previous piece's BOS is not a target), and mark in context (one sequence of labels) each document's
        context tokens. Returns each piece's start in the sequence, then the end of the last piece."""
        tokens, offsets, pieces, rows, fim, contexts = self.packed[name]
        row_pieces = pieces[rows[row]:rows[row + 1]]
        examples = row_pieces[:, 3] > row_pieces[:, 2]
        # a piece that ends its document (offsets) has its next BOS as a target, as does every FIM example
        targeted = (examples | (offsets[np.searchsorted(offsets, row_pieces[:, 1])] == row_pieces[:, 1])).tolist()
        examples = examples.tolist()
        documents = np.searchsorted(offsets, row_pieces[:, 0]).tolist()
        at, starts = 0, []
        for k, (start, end, f0, f1) in enumerate(row_pieces.tolist()):
            starts.append(at)
            out[at] = self.bos if k == 0 or targeted[k - 1] else IGNORE_INDEX
            part = fim[f0:f1] if examples[k] else tokens[start:end]
            out[at + 1:at + 1 + len(part)] = part
            if contexts is not None and offsets[documents[k]] == start:  # the labels from the BOS on
                context[at:at + contexts[documents[k]]] = True
            at += 1 + len(part)
        if targeted[-1]:  # write the last piece's target BOS
            out[at] = self.bos
        return [*starts, at]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]]:
        orders = [iter(order) for order in self.orders]
        for position in itertools.count(0, self.micro_batch_size):
            corpora = mixture_rows(self.weights, start=position, count=self.micro_batch_size)
            batch = np.full((self.micro_batch_size, self.seq_len + 1), IGNORE_INDEX, dtype=np.int32)
            context = np.zeros((self.micro_batch_size, self.seq_len), dtype=bool)
            bounds = []
            for r, c in enumerate(corpora):
                row = next(orders[c], None)
                if row is None:  # a source's order is used up
                    return
                starts = self.assemble(batch[r], context[r], self.names[c], row) if row >= 0 else [0]
                # segment starts: the pieces, then the padding if the input_ids hold any
                bounds += [r * self.seq_len + start for start in starts if start < self.seq_len]
            # page-locked, so the host-to-device copy is asynchronous
            batch = torch.from_numpy(batch).pin_memory().to(self.device, non_blocking=True)
            cu = np.full(self.num_segments + 1, self.micro_batch_size * self.seq_len, dtype=np.int32)
            cu[:len(bounds)] = bounds
            cu_seqlens = torch.from_numpy(cu).pin_memory().to(self.device, non_blocking=True)
            input_ids, labels = batch[:, :-1], batch[:, 1:].to(torch.int64)
            if self.contexts:
                labels.masked_fill_(torch.from_numpy(context).pin_memory().to(self.device, non_blocking=True),
                                    IGNORE_INDEX)
            yield (input_ids.masked_fill(input_ids == IGNORE_INDEX, self.bos), labels, cu_seqlens,
                   torch.tensor(len(bounds), dtype=torch.int32), np.diff(cu[:len(bounds) + 1]).astype(np.int64))
