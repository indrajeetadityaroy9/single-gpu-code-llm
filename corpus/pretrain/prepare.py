"""
Acquisition of the pretraining sources (paper §2.1): downloads each source's Hugging Face Hub files, reading only the
columns the standardizer needs, and writes one standardized Parquet file per Hub file in the schema of the source's
category. Text is normalized to LF line endings without byte-order marks. Stack-Edu lists files by blob id only, so
their text is fetched from Software Heritage's public S3 bucket (as The Stack v2 dataset card describes).

python -m corpus.pretrain.prepare
"""

import asyncio
import fnmatch
import functools
import gzip
import hashlib
import itertools
import resource
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import obstore
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download, list_repo_files
from obstore.store import S3Store

from corpus import mixture
from corpus.parallel import SPAWN
from corpus.pretrain.schema import COMPRESSION, CONVERSATION, FUNCTION, REPOSITORY_FILE, SPLIT, sha1
from corpus.pretrain.sources import (
    EDU,
    SOURCES,
    Source,
    footer_rows,
    raw_dir,
    standardized_path,
    text_bytes,
)

# Files downloaded at once: the highest throughput of 1, 2, 4 and 8
FETCHES = 4


@functools.cache
def hub_files(source: str) -> list[str]:
    """The source's Hub files that its globs match, sorted."""
    spec = SOURCES[source]
    return sorted(f for f in list_repo_files(spec.repo, repo_type="dataset")
                  if any(fnmatch.fnmatchcase(f, glob) for glob in spec.files.values()))


@functools.cache
def file_bytes(source: str) -> dict[str, int]:
    """{Hub file: its text bytes, from its Parquet footer} of every Hub file of the source."""
    spec, files = SOURCES[source], hub_files(source)

    def footer_bytes(file: str) -> int:
        with HfFileSystem().open(f"datasets/{spec.repo}/{file}", "rb") as remote:
            return text_bytes(pq.ParquetFile(remote).metadata, spec.text)

    with ThreadPoolExecutor(FETCHES) as pool:
        return dict(zip(files, pool.map(footer_bytes, files), strict=True))


def extent(source: str) -> list[str]:
    """The Hub files to download. For a fully used source, all of them, sorted. For a source of which only part is
    used (mixture.PART), the fewest files, in the SHA-1 order of their paths, whose text bytes cover its draw with
    the margin mixture.SPREAD and the validation share's correction SPLIT / (SPLIT - 1)."""
    if source not in mixture.PART:
        return hub_files(source)
    group = mixture.PART[source]
    tokens = sum(sum(file_bytes(s).values()) / mixture.BYTES_PER_TOKEN[s]
                 for s, g in mixture.PART.items() if g == group)
    need = (mixture.draw(group) / tokens * (1 + mixture.SPREAD) * SPLIT / (SPLIT - 1)
            * sum(file_bytes(source).values()))
    order = sorted(hub_files(source), key=lambda f: hashlib.sha1(f.encode()).digest())
    held = itertools.accumulate(file_bytes(source)[f] for f in order)
    return order[:next(k + 1 for k, total in enumerate(held) if total >= need)]


@functools.cache
def swh_store() -> S3Store:
    """This process's client of Software Heritage's bucket; raises the file-descriptor limit to its hard limit."""
    resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)
    return S3Store("softwareheritage", region="us-east-1", skip_signature=True)


def blob_texts(task: tuple[Path, int]) -> pa.Table:
    """A Stack-Edu metadata row group with each blob's text (null if missing), all requests in flight at once."""
    meta, group = task
    rows = pq.ParquetFile(meta).read_row_group(group)
    store = swh_store()

    async def blob(blob_id: str, encoding: str) -> str | None:
        try:
            response = await obstore.get_async(store, "content/" + blob_id)
            return gzip.decompress(await response.bytes_async()).decode(encoding)
        except (FileNotFoundError, UnicodeDecodeError):  # not in the bucket, or not text in its src_encoding
            return None

    async def fetched() -> list[str | None]:
        return await asyncio.gather(*map(blob, rows["blob_id"].to_pylist(), rows["src_encoding"].to_pylist()))

    return rows.append_column("text", pa.array(asyncio.run(fetched()), pa.string()))


def fetch(source: str, file: str) -> Path:
    """Download one of the source's Hub files and return its local path."""
    spec = SOURCES[source]
    path = raw_dir(source) / file
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec.repo != EDU:
        read = columns(spec)
        with HfFileSystem().open(f"datasets/{spec.repo}/{file}", "rb") as remote:
            hub = pq.ParquetFile(remote)
            with pq.ParquetWriter(path, pa.schema([hub.schema_arrow.field(c) for c in read]),
                                  compression=COMPRESSION) as writer:
                for g in range(hub.num_row_groups):
                    writer.write_table(hub.read_row_group(g, columns=read))
    else:
        meta = hf_hub_download(spec.repo, file, repo_type="dataset")
        with (ProcessPoolExecutor(mp_context=SPAWN) as pool,
              pq.ParquetWriter(path, pq.read_schema(meta).append(pa.field("text", pa.string())),
                               compression=COMPRESSION) as writer):
            for table in pool.map(blob_texts, [(meta, g) for g in range(pq.ParquetFile(meta).num_row_groups)]):
                writer.write_table(table)
    return path


def lines(texts: pa.ChunkedArray) -> pa.ChunkedArray:
    """texts with CR LF and a lone CR converted to LF and every leading U+FEFF (byte-order mark) removed (Python
    Language Reference §2.1.2, §2.1.4; C++23 [lex.phases] phases 1-2)."""
    texts = pc.replace_substring_regex(texts, "^\ufeff+", "")
    return pc.replace_substring(pc.replace_substring(texts, "\r\n", "\n"), "\r", "\n")


def document(spec: Source, content: pa.ChunkedArray, fields: dict[str, pa.ChunkedArray]) -> pa.Table:
    """A table in the source's category schema: content normalized (lines), its SHA-1 as id, the given fields and
    the language."""
    content = lines(content)
    fields = fields | {"id": sha1(content), "content": content}
    if "language" in spec.category.schema.names:
        fields["language"] = pa.repeat(spec.language, len(content))
    return pa.table({name: fields[name] for name in spec.category.schema.names}, schema=spec.category.schema)


def repository_file(t: pa.Table, spec: Source) -> pa.Table:
    """A repository file with its repository and path."""
    return document(spec, t["content"], {"repo_name": t["repo_name"], "relative_path": t["relative_path"]})


def stack_edu(t: pa.Table, spec: Source) -> pa.Table:
    """A Stack-Edu file, its path without a leading "/"."""
    return document(spec, t["text"], {"repo_name": t["repo_name"],
                                      "relative_path": pc.utf8_ltrim(t["path"], characters="/")})


def conversation(t: pa.Table, spec: Source) -> pa.Table:
    """A finished Codeforces-CoTs trace (finish_reason "stop"): the prompt followed by the generation, and the
    problem id."""
    t = t.filter(pc.equal(t["finish_reason"], "stop"))
    return document(spec, pc.binary_join_element_wise(t["prompt"], t[spec.text], ""), {"problem_id": t["id"]})


def function(t: pa.Table, spec: Source) -> pa.Table:
    """A CodeSearchNet (CodeXGLUE) function, code with its docstring, and its repository."""
    return document(spec, t["code"], {"repo_name": t["repo"]})


def text_only(t: pa.Table, spec: Source) -> pa.Table:
    """A document that is only its text."""
    return document(spec, t[spec.text], {})


def standardizer(spec: Source) -> Callable[[pa.Table, Source], pa.Table]:
    """The function that converts a source's Hub rows into its category schema."""
    if spec.repo == EDU:
        return stack_edu
    return {REPOSITORY_FILE: repository_file, CONVERSATION: conversation, FUNCTION: function}.get(
        spec.category, text_only)


def columns(spec: Source) -> list[str]:
    """The Hub columns a source's standardizer reads (Stack-Edu's metadata is downloaded whole)."""
    return {repository_file: ["repo_name", "relative_path", "content"],
            conversation: ["id", "prompt", "generation", "finish_reason"], function: ["repo", "code"],
            text_only: [spec.text]}[standardizer(spec)]


def standardize(task: tuple[str, int, Path]) -> None:
    """Convert one downloaded file into its standardized file: the rows with text, in the source's category schema."""
    source, k, path = task
    spec = SOURCES[source]
    standardized = standardizer(spec)
    pf = pq.ParquetFile(path)
    with pq.ParquetWriter(standardized_path(source, k), spec.category.schema, compression=COMPRESSION) as writer:
        for g in range(pf.num_row_groups):
            raw = pf.read_row_group(g)
            writer.write_table(standardized(raw.filter(pc.is_valid(raw[spec.text])), spec))


def prepare(source: str, files: list[str]) -> None:
    """Download and standardize the source's files into a new standardized directory."""
    spec = SOURCES[source]
    standardized_path(source, 0).parent.mkdir(parents=True)
    # one Stack-Edu file at a time: each already runs a process per CPU with every blob of a row group in flight
    with ThreadPoolExecutor(1 if spec.repo == EDU else FETCHES) as pool:
        paths = list(pool.map(functools.partial(fetch, source), files))
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        list(pool.map(standardize, [(source, k, path) for k, path in enumerate(paths)]))
    rows = footer_rows(standardized_path(source, k) for k in range(len(files)))
    print(f"{source}: {len(files)} of {len(hub_files(source))} files of {spec.repo}, {rows:,} rows", flush=True)


def main() -> None:
    for source in SOURCES:
        prepare(source, extent(source))


if __name__ == "__main__":
    main()
