"""Native libraries (the CUDA kernels and the corpus's Rust crate as a Python module), built on first use and cached
under NATIVE; pointer and stream build the kernels' ctypes arguments. Only the CUDA paths import torch, so the corpus
workers can load the crate without it."""

import ctypes
import functools
import hashlib
import importlib.util
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from model.config import HEAD_DIM, HIDDEN_SIZE
from paths import NATIVE

if TYPE_CHECKING:
    import torch

KERNELS = Path(__file__).resolve().parent / "model" / "kernels"
CORPUS_CRATE = Path(__file__).resolve().parent / "corpus" / "crate"


def native_library(name: str, command: list[str], sources: list[Path], build: Callable[[Path], None]) -> Path:
    """NATIVE/<name>-<sha256(command, sources)>.so, built by build(file) in a temporary directory unique to the build
    (removed on exit, a failed build's output with it) and renamed atomically, so concurrent first builds (any
    process, any container) never share a file."""
    digest = hashlib.sha256(" ".join(command).encode())
    for source in sources:
        digest.update(source.read_bytes())
    path = NATIVE / f"{name}-{digest.hexdigest()}.so"
    if not path.exists():
        with tempfile.TemporaryDirectory(dir=NATIVE) as tmp:
            built = Path(tmp) / path.name
            build(built)
            built.replace(path)
    return path


def pointer(t: "torch.Tensor") -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def stream() -> ctypes.c_void_p:
    """The current CUDA stream, which the kernels launch on."""
    import torch

    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


@functools.cache
def cuda_library(name: str) -> ctypes.CDLL:
    """model/kernels/<name>.cu built by nvcc for the GH200 (sm_90) with the model's HEAD_DIM and HIDDEN_SIZE, keyed by
    the command, the file and the shared header; loaded once per process."""
    source = KERNELS / f"{name}.cu"
    command = ["nvcc", "-arch=sm_90", f"-DHEAD_DIM={HEAD_DIM}", f"-DHIDDEN_SIZE={HIDDEN_SIZE}", "-Xcompiler", "-fPIC",
               "-shared", "-o"]

    def build(path: Path) -> None:
        subprocess.run([*command, path, source], check=True)

    return ctypes.CDLL(native_library(name, command, [source, KERNELS / "common.cuh"], build))


@functools.cache
def corpus_crate() -> ModuleType:
    """The pyo3 module of the corpus's Rust crate (corpus/crate), named as its Cargo.toml package, built by cargo into
    NATIVE/crate-target, keyed by the command and the crate's files; imported once per process."""
    name = tomllib.loads((CORPUS_CRATE / "Cargo.toml").read_text())["package"]["name"]
    target = NATIVE / f"{CORPUS_CRATE.name}-target"
    command = ["cargo", "build", "--release", "--locked", "--manifest-path", str(CORPUS_CRATE / "Cargo.toml"),
               "--target-dir", str(target)]

    def build(path: Path) -> None:
        subprocess.run(command, check=True)
        shutil.copyfile(target / "release" / f"lib{name}.so", path)

    sources = [*sorted(CORPUS_CRATE.glob("*.rs")), CORPUS_CRATE / "Cargo.toml", CORPUS_CRATE / "Cargo.lock"]
    path = native_library(CORPUS_CRATE.name, command, sources, build)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
