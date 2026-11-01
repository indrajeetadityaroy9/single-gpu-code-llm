"""
Every location the code reads or writes, all inside the checkout. Run as a script (the image runs it before every
command, docker/Dockerfile), it creates the directories below and the cache directories the environment names; code
never creates them.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CORPUS = ARTIFACTS / "corpus"
RUNS = ARTIFACTS / "runs"
CACHE = ARTIFACTS / "cache"
# native libraries and their build directories (native.py)
NATIVE = CACHE / "native"
# tracked records (run.py): the learning-rate calibrations' records, read by the later experiments
RESULTS = ROOT / "results"
# the compiler, kernel and library caches docker/Dockerfile places under CACHE
CACHE_VARIABLES = ("XDG_CACHE_HOME", "HF_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "QUACK_CACHE_DIR",
                   "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", "CARGO_HOME")

if __name__ == "__main__":
    for directory in (CORPUS, RUNS, NATIVE, *(Path(os.environ[n]) for n in CACHE_VARIABLES)):
        directory.mkdir(parents=True, exist_ok=True)
