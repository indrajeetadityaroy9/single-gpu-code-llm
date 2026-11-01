"""
Tokenization (paper §3.1, §2.1 "Repository-level documents"): every curated source tokenized with the corpus's
tokenizer into <split>.bin, .idx and .json per source and split, plus the repository-level stage's per-language
datasets of whole repositories (mixture.REPOSITORIES). The output does not depend on the worker count.

python -m corpus.tokenize
"""

import itertools
import json
import os
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from corpus import mixture
from corpus.layout import TOKEN_DTYPE, shard_bounds, split_path
from corpus.parallel import MEMORY, SPAWN, ordered
from corpus.pretrain.sources import (
    SOURCES,
    curated_dir,
    curated_documents,
    curated_shards,
    repository_samples,
    text_bytes,
    texts,
)
from corpus.tokenizer import get_tokenizer

Documents = tuple[np.ndarray, np.ndarray]  # tokens, document lengths
# Peak memory per partition text byte: ru_maxrss of a spawned worker, the larger of the two languages' (C++)
PEAK = 8.18


def encode(documents: list[str], mapper: Callable) -> Documents:
    """(tokens, document lengths) of documents, encoded longest first through `mapper` (map, or a thread pool's map);
    special-token text is encoded as ordinary text."""
    enc = get_tokenizer().enc
    order = sorted(range(len(documents)), key=lambda i: -len(documents[i]))
    ids = [np.zeros(0, TOKEN_DTYPE)] * len(documents)
    for i, tokens in zip(order, mapper(
            lambda i: enc.encode_to_numpy(documents[i], disallowed_special=()).astype(TOKEN_DTYPE), order),
            strict=True):
        ids[i] = tokens
    lengths = np.array([len(doc) for doc in ids], dtype=np.int64)
    return np.concatenate(ids), lengths


def row_group_documents(task: tuple[str, Path, int]) -> Documents:
    """The documents of one curated row group (curated_documents)."""
    source, path, g = task
    return encode(curated_documents(source, pq.ParquetFile(path).read_row_group(g)), map)


def language_sources(language: str) -> list[str]:
    """A language's repository sources, in mixture.REPOSITORY_SOURCES order."""
    return [s for s in mixture.REPOSITORY_SOURCES if SOURCES[s].language == language]


def repository_documents(task: tuple[str, str, int]) -> Documents:
    """The repository-level documents (whole repositories) of one partition of one language's repository sources."""
    language, split, part = task
    sources = language_sources(language)
    schema = pa.schema([("repo_name", pa.string()), ("relative_path", pa.string()), ("content", pa.large_string())])
    table = pa.concat_tables(
        table.append_column("rank", pa.array(np.full(table.num_rows, rank)))
        for rank, table in enumerate(pq.read_table(curated_dir(source, split) / f"{part}.parquet",
                                                   columns=schema.names).cast(schema) for source in sources))
    table, ends, samples = repository_samples(table, ("relative_path", "rank"), language, one_per_path=True)
    files = texts(sources[0], table)
    del table
    files = [files[i] for sample in samples for i in sample]  # each repository's files in sample order
    documents = ["".join(files[a:b]) for a, b in itertools.pairwise([0, *ends])]
    del files
    with ThreadPoolExecutor(os.cpu_count()) as threads:
        return encode(documents, threads.map)


def tokenize_split(corpus: str, split: str, shards: list[int], results: Iterable[Documents]) -> None:
    """Write a source's split as <split>.bin, .idx and .json; shards[k] results belong to curated shard k."""
    path = split_path(corpus, split, get_tokenizer())
    print(f"Tokenizing {len(shards)} {split} shards ({sum(shards):,} tasks) of '{corpus}' into {path}.bin", flush=True)
    per_shard, offset, results = [], 0, iter(results)
    with open(path.with_suffix(".bin"), "wb") as bin_file, open(path.with_suffix(".idx"), "wb") as idx_file:
        idx_file.write(np.zeros(1, np.int64))
        for tasks in shards:
            documents = 0
            for tokens, lengths in itertools.islice(results, tasks):
                bin_file.write(tokens)
                idx_file.write(offset + np.cumsum(lengths))
                offset += len(tokens)
                documents += len(lengths)
            per_shard.append({"documents": documents})
    path.with_suffix(".json").write_text(json.dumps({"files": per_shard}))


def tokenize_source(source: str, split: str, pool: Executor) -> None:
    """A source's curated split, one row group per task."""
    tasks = [[(source, path, g) for g in range(pq.read_metadata(path).num_row_groups)]
             for path in curated_shards(source, split)]
    tokenize_split(source, split, [len(shard) for shard in tasks],
                   pool.map(row_group_documents, [task for shard in tasks for task in shard]))  # in task order


def shard_tokens(corpus: str, split: str) -> np.ndarray:
    """The tokens of each curated shard of a tokenized split."""
    tokenizer = get_tokenizer()
    offsets = np.memmap(split_path(corpus, split, tokenizer).with_suffix(".idx"), dtype=np.int64, mode="r")
    return np.diff(offsets[shard_bounds(corpus, split, tokenizer)])


def train_parts() -> list[int]:
    """The repository sources' train partitions, in order."""
    return [int(path.stem) for path in curated_shards(mixture.REPOSITORY_SOURCES[0], "train")]


def tokenize_repo_level_stage(need: float) -> None:
    """The repository-level datasets of val partition 0 and of the fewest trailing train partitions holding `need`
    tokens, tokenized newest partition first; as many tasks run at once as MEMORY holds at PEAK times the largest
    partition's text bytes."""
    parts, languages = train_parts(), list(mixture.REPOSITORIES)
    tasks = [(language, "val", 0) for language in languages]
    tasks += [(language, "train", part) for part in reversed(parts) for language in languages]
    largest = max(sum(text_bytes(pq.read_metadata(curated_dir(s, split) / f"{part}.parquet"), "content")
                      for s in language_sources(language))
                  for language, split, part in tasks)
    window = int(MEMORY / (PEAK * largest))
    held, total = {}, 0
    with ProcessPoolExecutor(window, mp_context=SPAWN) as pool:
        results = ordered(pool, repository_documents, tasks, window)
        for (language, split, part), documents in zip(tasks, results, strict=False):
            held[language, split, part] = documents
            total += len(documents[0]) if split == "train" else 0
            if split == "train" and language == languages[-1] and total >= need:
                break
        results.close()
    chosen = parts[-(len(held) // len(languages) - 1):]
    for language, corpus in mixture.REPOSITORIES.items():
        split_path(corpus, "val", get_tokenizer()).parent.mkdir(parents=True)
        tokenize_split(corpus, "val", [1], [held.pop((language, "val", 0))])
        tokenize_split(corpus, "train", [1] * len(chosen), (held.pop((language, "train", part)) for part in chosen))
        print(f"{corpus}: train partitions {chosen[0]}..{chosen[-1]}, "
              f"{shard_tokens(corpus, 'train').sum():,} tokens", flush=True)


def repository_parts() -> list[int]:
    """The train partitions held by the tokenized repository-level datasets."""
    return train_parts()[-len(shard_tokens(mixture.REPOSITORIES["Python"], "train")):]


def main() -> None:
    """Tokenize every source's splits, then the repository-level stage's datasets."""
    jobs = list(itertools.product(SOURCES, ("val", "train")))
    with ProcessPoolExecutor(mp_context=SPAWN) as pool, ThreadPoolExecutor(len(jobs)) as threads:
        for corpus in SOURCES:
            split_path(corpus, "val", get_tokenizer()).parent.mkdir(parents=True)
        list(threads.map(lambda job: tokenize_source(job[0], job[1], pool), jobs))
    # the repository-level stage's repository tokens: CODE_SHARE of its tokens less its supply sources'
    tokenize_repo_level_stage(mixture.CODE_SHARE * mixture.REPO_LEVEL_STAGE.tokens
                              - sum(shard_tokens(s, "train").sum() for s in mixture.REPO_LEVEL_STAGE.supply))


if __name__ == "__main__":
    main()
