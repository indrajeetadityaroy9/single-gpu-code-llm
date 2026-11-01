"""
Curation of the pretraining corpus (paper §2.1): reads each source's standardized Parquet files and writes its curated
train and validation shards. The steps are the Python 3 syntax check with repair, exact deduplication across all
sources, the train/validation split, and sharding. Every output is a deterministic function of the standardized files.

The split hashes each document's split group (its repository, Codeforces problem, or the document itself) with
md5_number(group) % SPLIT; residue 0 is validation, so 1% of the groups are held out.

python -m corpus.pretrain.curate
"""

import ast
import functools
import hashlib
import itertools
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fissix import refactor

from corpus.parallel import MEMORY, SPAWN, ordered
from corpus.pretrain.schema import COMPRESSION, REPOSITORY_FILE, SAMPLE_FILE, SPLIT, sha1
from corpus.pretrain.sources import (
    SOURCES,
    checked_path,
    curated_dir,
    footer_rows,
    repository_samples,
    sample_starts,
    standardized_path,
)
from native import corpus_crate
from paths import CORPUS

EMPTY = hashlib.sha1(b"").hexdigest()
# A repository source's row groups hold at least this many files (except the last) and only whole samples, so a
# reader never splits a sample. A sample is a set of files of one repository connected by imports (see
# sources.repository_samples). Fixed: the published dataset (publish.DATASET) was built with it
SAMPLE_ROWS = 65_536
PLACEHOLDER = re.compile(r"<(EMAIL|KEY|NAME|PASSWORD)>")
LEADING = re.compile(r"^[ \t]+", re.MULTILINE)
# Recursion limit for fissix's recursive 2to3 (ast.parse does not use it); fixed: the published dataset used it
RECURSION = 6_000


def residue(group: str) -> str:
    """DuckDB expression for a group's residue, md5_number(group) % SPLIT; residue 0 is the validation split."""
    return f"md5_number({group}) % {SPLIT}"


def database() -> duckdb.DuckDBPyConnection:
    """A DuckDB connection limited to MEMORY, spilling to CORPUS/spill, without insertion order (parallel.MEMORY)."""
    return duckdb.connect(config={"memory_limit": f"{MEMORY}B", "temp_directory": str(CORPUS / "spill"),
                                  "preserve_insertion_order": False})


# ---- 1. Python 3 syntax check and repair

def parses(text: str) -> bool:
    """Whether ast.parse accepts text. Every exception it raises counts as a rejection, and none depends on the host:
    ValueError on f"{x:{y=}}" (python/cpython#121130), MemoryError when the parser's stack overflows ("Python source
    too complex to parse"), and RecursionError when the syntax tree nests deeper than CPython's fixed C recursion
    limit (independent of RECURSION)."""
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


@functools.cache
def fixers() -> refactor.RefactoringTool:
    """fissix's 2to3 refactoring tool with every fixer of fissix.fixes, built once per process."""
    return refactor.RefactoringTool(refactor.get_fixers_from_package("fissix.fixes"))


def two_to_three(text: str) -> str | None:
    """text converted from Python 2 to Python 3 by every fixer of fissix, or None when fissix cannot parse it."""
    try:
        return str(fixers().refactor_string(text if text.endswith("\n") else text + "\n", "<file>"))
    except Exception:  # fissix raises its grammar's parse and tokenize errors, and others, on text it cannot read
        return None


def bare_placeholders(text: str) -> str:
    """Anonymization placeholders such as <EMAIL> replaced by the plain word (EMAIL)."""
    return PLACEHOLDER.sub(lambda m: m.group(1), text)


def expanded_tabs(text: str) -> str:
    """Tabs in leading indentation expanded to 8-column tab stops."""
    return LEADING.sub(lambda m: m.group(0).expandtabs(8), text)


def repaired(text: str) -> str | None:
    """text if it parses; else the smallest repair that parses, trying subsets of the repair steps that change the
    text by size and then in step order; else None."""
    if parses(text):
        return text
    steps = [step for step in (bare_placeholders, expanded_tabs) if step(text) != text] + [two_to_three]
    for size in range(1, len(steps) + 1):
        for subset in itertools.combinations(steps, size):
            result: str | None = text
            for step in subset:
                result = step(result)
                if result is None:  # fissix could not parse it
                    break
            if result is not None and parses(result):
                return result
    return None


def check_row_group(task: tuple[str, int, int]) -> pa.Table:
    """Row group g of a Python source's standardized file k: the files that parse after repair, ids recomputed."""
    source, k, g = task
    sys.setrecursionlimit(RECURSION)
    schema = SOURCES[source].category.schema
    table = pq.ParquetFile(standardized_path(source, k)).read_row_group(g)
    texts = pa.array([repaired(text) for text in table["content"].to_pylist()], pa.string())
    content = texts.drop_null()
    table = table.filter(texts.is_valid())
    table = table.set_column(schema.get_field_index("content"), schema.field("content"), content)
    return table.set_column(schema.get_field_index("id"), schema.field("id"), sha1(content))


def check(source: str, files: int, pool: ProcessPoolExecutor) -> None:
    """Write checked_path for each of a Python source's standardized files. As many row groups run at once as MEMORY
    holds copies of the largest one."""
    metadata = [pq.read_metadata(standardized_path(source, k)) for k in range(files)]
    tasks = [(source, k, g) for k, md in enumerate(metadata) for g in range(md.num_row_groups)]
    largest = max(md.row_group(g).total_byte_size for md in metadata for g in range(md.num_row_groups))
    checked_path(source, 0).parent.mkdir(parents=True)
    writers: dict[int, pq.ParquetWriter] = {}
    for (_, k, g), table in zip(tasks, ordered(pool, check_row_group, tasks, MEMORY // largest), strict=True):
        if g == 0:
            writers[k] = pq.ParquetWriter(checked_path(source, k), SOURCES[source].category.schema,
                                          compression=COMPRESSION)
        writers[k].write_table(table)
        if g == metadata[k].num_row_groups - 1:
            writers.pop(k).close()


# ---- 2. Exact deduplication

def duplicates(paths: dict[str, list[Path]]) -> dict[str, pa.Table]:
    """{source: (file, row) of each row to drop}: every row but the first with its id (first in the order of paths),
    and every empty text."""
    union = " UNION ALL ".join(
        f"SELECT id, {i} AS source, file_index AS file, file_row_number AS row "
        f"FROM read_parquet({[str(p) for p in files]}, file_row_number = true)"
        for i, files in enumerate(paths.values()))
    with database() as con:
        drops = con.sql(f"SELECT source, file, row FROM (SELECT id, source, file, row, row_number() OVER "
                        f"(PARTITION BY id ORDER BY source, file, row) AS n FROM ({union})) "
                        f"WHERE n > 1 OR id = '{EMPTY}'").to_arrow_table()
    return {source: drops.filter(pc.equal(drops["source"], i)).select(["file", "row"])
            for i, source in enumerate(paths)}


# ---- 3 and 4. Train/validation split and shards

def shuffle_dir(source: str) -> Path:
    return CORPUS / "shuffle" / source


def undropped(paths: list[Path]) -> str:
    """A DuckDB FROM clause over the rows of paths that are not listed in the registered table `drops`, aliased n."""
    return (f"read_parquet({[str(p) for p in paths]}, file_row_number = true) AS n "
            f"ANTI JOIN drops AS d ON d.file = n.file_index AND d.row = n.file_row_number")


def shuffle(source: str, paths: list[Path], drops: pa.Table) -> None:
    """Write a repository source's kept rows as Hive partitions shuffle/<source>/part=<residue>/."""
    shuffle_dir(source).parent.mkdir(exist_ok=True)
    with database() as con:
        con.register("drops", drops)
        con.execute(f"COPY (SELECT n.repo_name, n.relative_path, n.content, n.id, {residue(REPOSITORY_FILE.group)} "
                    f"AS part FROM {undropped(paths)}) "
                    f"TO '{shuffle_dir(source)}' (FORMAT parquet, COMPRESSION {COMPRESSION}, PARTITION_BY (part))")


def curate_partition(task: tuple[str, int]) -> None:
    """Write one partition as a curated shard (partition 0 is validation), in sample order, with row groups of at
    least SAMPLE_ROWS files made of whole samples."""
    source, part = task
    table = pq.read_table(shuffle_dir(source) / f"part={part}",
                          schema=pa.schema([("repo_name", pa.string()), ("relative_path", pa.string()),
                                            ("content", pa.large_string()), ("id", pa.string())]))
    table, _, groups = repository_samples(table, ("relative_path", "id"), SOURCES[source].language, one_per_path=False)
    order, samples = np.concatenate(groups), np.repeat(np.arange(len(groups)), [len(group) for group in groups])
    cuts = [0]
    for start in sample_starts(samples):
        if start - cuts[-1] >= SAMPLE_ROWS:
            cuts.append(start)
    columns = table.select(["repo_name", "relative_path", "content"])
    split = "val" if part == 0 else "train"
    with pq.ParquetWriter(curated_dir(source, split) / f"{part}.parquet", SAMPLE_FILE,
                          compression=COMPRESSION) as writer:
        for a, b in itertools.pairwise([*cuts, len(order)]):
            sample = columns.take(order[a:b]).append_column("sample", pa.array(samples[a:b]))
            writer.write_table(sample.cast(SAMPLE_FILE), row_group_size=b - a)


def shard_documents(source: str, paths: list[Path], drops: pa.Table) -> None:
    """Write a non-repository source's kept rows: validation in one shard, train in id order over SPLIT shards."""
    category = SOURCES[source].category
    columns = ", ".join(f"n.{name}" for name in category.schema.names)
    names = ", ".join(category.schema.names)
    with database() as con:
        con.register("drops", drops)
        con.execute(f"CREATE TABLE kept AS SELECT {columns}, {residue(category.group)} = 0 AS val "
                    f"FROM {undropped(paths)}")
        con.execute(f"COPY (SELECT {names} FROM kept WHERE val ORDER BY id) "
                    f"TO '{curated_dir(source, 'val')}/0.parquet' (FORMAT parquet, COMPRESSION {COMPRESSION})")
        con.execute(f"CREATE TABLE train AS SELECT {names}, ntile({SPLIT}) OVER (ORDER BY id) - 1 AS shard "
                    f"FROM kept WHERE NOT val")
        for k in range(SPLIT):
            con.execute(f"COPY (SELECT {names} FROM train WHERE shard = {k} ORDER BY id) "
                        f"TO '{curated_dir(source, 'train')}/{k}.parquet' (FORMAT parquet, COMPRESSION {COMPRESSION})")


def main() -> None:
    """Curate every source from its standardized files (for a Python source, its checked files)."""
    corpus_crate()  # built once here, before the workers load it
    files = {source: len(list(standardized_path(source, 0).parent.iterdir())) for source in SOURCES}
    python = [source for source, spec in SOURCES.items() if spec.category.code and spec.language == "Python"]
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        for source in python:
            check(source, files[source], pool)
    paths = {source: [(checked_path if source in python else standardized_path)(source, k)
                      for k in range(files[source])] for source in SOURCES}
    drops = duplicates(paths)
    for source, spec in SOURCES.items():
        print(f"{source}: {footer_rows(paths[source]):,} rows, "
              f"{drops[source].num_rows:,} duplicates", flush=True)
        for split in ("train", "val"):
            curated_dir(source, split).mkdir(parents=True)
        if spec.category == REPOSITORY_FILE:
            shuffle(source, paths[source], drops[source])
            # MEMORY // (2 * largest) workers leave each worker 4 times the largest partition's size of memory
            # (MEMORY is half of the machine's) [arithmetic]. A worker holds three copies (the read table, its sorted
            # copy, and the contents as Python strings for topological_sort); the peak is not measured [inference]
            largest = max(sum(md.row_group(g).total_byte_size for md in map(pq.read_metadata, part.iterdir())
                              for g in range(md.num_row_groups)) for part in shuffle_dir(source).iterdir())
            with ProcessPoolExecutor(min(os.cpu_count(), MEMORY // (2 * largest)), mp_context=SPAWN) as pool:
                list(pool.map(curate_partition, [(source, part) for part in range(SPLIT)]))
            shutil.rmtree(shuffle_dir(source))
        else:
            shard_documents(source, paths[source], drops[source])


if __name__ == "__main__":
    main()
