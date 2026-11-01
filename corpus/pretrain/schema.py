"""
Document categories of the pretraining corpus (paper §2.1). Each category has a Parquet schema of non-null strings,
a split group (the unit kept on one side of the train/validation split), and a flag for whether its content is code.
"""

import hashlib
from dataclasses import dataclass

import pyarrow as pa

# 1% of split groups are held out for validation; also the number of train shards per non-repository source (ntile)
# and of repository partitions
SPLIT = 100
# The compression codec of the published dataset
COMPRESSION = "zstd"


def strings(*names: str) -> list[pa.Field]:
    """Non-null string fields with the given names."""
    return [pa.field(name, pa.string(), nullable=False) for name in names]


def sha1(texts: pa.Array | pa.ChunkedArray) -> pa.Array:
    """The SHA-1 hex digest of each text, used as the document id and for exact deduplication."""
    return pa.array([hashlib.sha1(b).hexdigest() for b in texts.cast(pa.large_binary()).to_pylist()], pa.string())


@dataclass(frozen=True)
class Category:
    name: str  # keeps categories with equal schemas, groups and flags distinct
    schema: pa.Schema
    group: str  # the split group: a DuckDB expression over the schema's columns
    code: bool


REPOSITORY_FILE = Category("repository file", pa.schema(strings("id", "repo_name", "relative_path", "content",
                                                                "language")), "lower(repo_name)", code=True)
EXERCISE = Category("exercise", pa.schema(strings("id", "content", "language")), "id", code=False)
SYNTHETIC_FILE = Category("synthetic file", pa.schema(strings("id", "content", "language")), "id", code=True)
FUNCTION = Category("function", pa.schema(strings("id", "repo_name", "content", "language")), "lower(repo_name)",
                    code=True)
CONVERSATION = Category("conversation", pa.schema(strings("id", "problem_id", "content")), "problem_id", code=False)
NOTEBOOK = Category("notebook", pa.schema(strings("id", "content", "language")), "id", code=False)
TEXT = Category("text", pa.schema(strings("id", "content")), "id", code=False)

# A curated repository shard's schema: `sample` numbers the import-connected sets of files (sources.repository_samples)
SAMPLE_FILE = pa.schema([*strings("repo_name", "relative_path", "content"),
                         pa.field("sample", pa.int64(), nullable=False)])
