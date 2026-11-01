"""
Tokenizer training (paper §3.1): byte-level BPE whose merges are chosen by SA-BPE's score F·ln R (arXiv 2604.14053
§4.1), within the pre-tokens of SPLIT_PATTERN, on a systematic sample of the file-level stage's curated train splits
in proportion to each source's share of the stage's bytes. R counts repositories (repository files and functions) or
documents (every other source).

python -m corpus.train_tokenizer
"""
import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from corpus import mixture
from corpus.parallel import SPAWN
from corpus.pretrain.sources import curated_shards, text_bytes, texts
from corpus.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN, TOKENIZER, get_tokenizer, tiktoken_file
from model.config import VOCAB_SIZE
from native import corpus_crate

# The reference tokenizer's sample size, so the vocabulary is trained on as much text
SAMPLE_BYTES = 13.38e9


@functools.cache
def shard_bytes(source: str) -> list[int]:
    """The text bytes of each shard of the source's curated train split."""
    return [text_bytes(pq.read_metadata(path), "content") for path in curated_shards(source, "train")]


def byte_shares() -> dict[str, float]:
    """Each file-level source's share of the stage's bytes: its token share (Stage.shares) times its bytes per token."""
    sources, per_token = mixture.FILE_LEVEL_STAGE.sources, mixture.BYTES_PER_TOKEN
    shares = mixture.FILE_LEVEL_STAGE.shares({s: sum(shard_bytes(s)) / per_token[s] for s in sources},
                                             mixture.FILE_LEVEL_STAGE.tokens)
    stage_bytes = sum(shares[s] * per_token[s] for s in sources)
    return {s: shares[s] * per_token[s] / stage_bytes for s in sources}


def sample_row_group(task: tuple[str, Path, int, int, float]) -> tuple[list[str], list[str]]:
    """Row group g of a curated train shard: (texts, repository units) of the rows the systematic sample takes, row i
    when floor((i + 1) * fraction) > floor(i * fraction), `start` being the source's rows before the group."""
    source, path, g, start, fraction = task
    group = pq.ParquetFile(path).read_row_group(g)
    rows = np.arange(start, start + group.num_rows)
    batch = group.take(np.flatnonzero(np.floor((rows + 1) * fraction) > np.floor(rows * fraction)))
    if "repo_name" in batch.column_names:
        units = [name.lower() for name in batch["repo_name"].to_pylist()]
    else:
        units = batch["id"].to_pylist()
    return texts(source, batch), units


def sample() -> tuple[list[str], list[int]]:
    """The training sample: texts and their dense repository-unit ids (R in SA-BPE's score)."""
    tasks = []
    for s, share in byte_shares().items():
        fraction, at = share * SAMPLE_BYTES / sum(shard_bytes(s)), 0
        for path in curated_shards(s, "train"):
            md = pq.read_metadata(path)
            for g in range(md.num_row_groups):
                tasks.append((s, path, g, at, fraction))
                at += md.row_group(g).num_rows
    docs, units, ids = [], [], {}
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        for batch, names in pool.map(sample_row_group, tasks):
            docs += batch
            units += [ids.setdefault(name, len(ids)) for name in names]
    return docs, units


def main() -> None:
    """Train VOCAB_SIZE - len(SPECIAL_TOKENS) - 256 merges (merge i makes token 256 + i) on the sample and write them
    to TOKENIZER as tokenizer.tiktoken (the 256 bytes, then each merge's) and tokenizer.pattern."""
    docs, units = sample()
    merges = corpus_crate().train(docs, units, pattern=SPLIT_PATTERN,
                                  n_merges=VOCAB_SIZE - len(SPECIAL_TOKENS) - 256)
    mergeable = [bytes([i]) for i in range(256)]
    for left, right in merges:
        mergeable.append(mergeable[left] + mergeable[right])
    TOKENIZER.mkdir()
    (TOKENIZER / "tokenizer.tiktoken").write_text(tiktoken_file(mergeable), newline="")
    (TOKENIZER / "tokenizer.pattern").write_text(SPLIT_PATTERN, newline="")
    print(f"{len(merges):,} merges on {len(docs):,} documents of {max(units) + 1:,} units, "
          f"data key {get_tokenizer().data_key}", flush=True)


if __name__ == "__main__":
    main()
