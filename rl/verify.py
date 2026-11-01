"""
Unit-test verification of a response's answer: the RL reward and the validation verdicts (paper §2.3, §5.2.2). The
answer's code (`code`) is its last fenced block tagged python that defines one of the names the tests import (else
its last python block, since a later block is often a usage example or a test), else its last untagged block,
dedented; a response without code fails. Instruct and complete problems: the code is written as solution.py beside
KodCode's pytest file, prefixed with `from solution import *` (some test files use the solution's names without
importing them), and passes when pytest exits 0 within PYTEST_TIMEOUT. Online-judge problems: the code runs as a
program per stdin case within CASE_TIMEOUT each and passes when every stdout equals the expected one line by line,
ignoring trailing whitespace and trailing blank lines. A problem whose tests a wrong program (`stub`: its names
returning None, or no output) passes has trivial tests; its passes are 'pass_trivial', not successes.

Sandboxing: each run is a fresh process in its own session and temporary directory, under resource limits (address
space MEMORY, file size FILE_SIZE, CPU seconds equal to its timeout, no core files), with single-threaded BLAS; its
process group is killed at the end. Workers are spawned, so no parent state (CUDA) is inherited. Run in a container
without network: a CPU-only one, or the RL sampler's (rl/sampler.py, via `judge`).

Harness check on the pools' own R1 answers (each passed KodCode's tests in KodCode's environment), `--known-good`:
2,995 of 3,000 train and 1,499 of 1,500 val rows pass; of the 6 others, 2 name the function differently from the
tests, 2 have tests that depend on the environment (the home directory; bs4, not installed) and 2 are failing
answers. One train row (Package_3833_I) passes in some runs, giving 2,996. Stubs pass 2 of 3,000 and 3 of 1,500
(tests of plotting or printing side effects).

    python -m rl.verify --known-good <split> <n>
"""
import argparse
import ast
import contextlib
import json
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from corpus.parallel import SPAWN
from corpus.tokenizer import get_tokenizer
from eval.loop import END
from rl.prompts import read_pool, sample

PYTEST_TIMEOUT, CASE_TIMEOUT = 30, 10  # seconds
MEMORY, FILE_SIZE = 16 << 30, 64 << 20  # bytes; torch's import alone reserves more than 4 GiB of address space
TEST_PREFIX = "from solution import *\n"
FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
PYTHON_TAGS = ("python", "py", "python3")
ENV = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LC_ALL": "C.UTF-8", "MPLBACKEND": "Agg"}


def code(answer: str, names: list[str]) -> str | None:
    """The answer's last fenced block tagged python that defines (def or class) one of `names`, else its last python
    block, else its last untagged block; dedented (a fence inside a list item carries the item's indentation); None
    if it has none."""
    blocks = FENCE.findall(answer)
    python = [body for tag, body in blocks if tag.strip().lower() in PYTHON_TAGS]
    defining = [body for body in python
                if any(re.search(rf"^\s*(def|class)\s+{re.escape(n)}\b", body, re.M) for n in names)]
    found = defining or python or [body for tag, body in blocks if not tag.strip()]
    return textwrap.dedent(found[-1]) if found else None


def limits(seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_SIZE, FILE_SIZE))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))


def run(command: list[str], cwd: str, timeout: int, stdin: str = "") -> tuple[int | None, str]:
    """(exit code or None at the timeout, stdout) of command in its own session; its process group killed after."""
    with tempfile.TemporaryFile("w+") as inp, tempfile.TemporaryFile("w+") as out:
        inp.write(stdin)
        inp.seek(0)
        process = subprocess.Popen(command, cwd=cwd, stdin=inp, stdout=out, stderr=subprocess.DEVNULL, text=True,
                                   env=ENV | {"HOME": cwd, "TMPDIR": cwd}, start_new_session=True,
                                   preexec_fn=lambda: limits(timeout))
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = None
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        out.seek(0)
        return status, out.read()


def lines(text: str) -> list[str]:
    out = [line.rstrip() for line in text.split("\n")]
    while out and not out[-1]:
        out.pop()
    return out


def check(task: tuple[str | None, str, str]) -> str:
    """'pass', 'fail', 'timeout' or 'no_code' for (code, test, kind)."""
    program, test, kind = task
    if program is None:
        return "no_code"
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "solution.py").write_text(program)
        if kind == "pytest":
            Path(directory, "test_solution.py").write_text(TEST_PREFIX + test)
            status, _ = run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                             "test_solution.py"], directory, PYTEST_TIMEOUT)
            return "timeout" if status is None else "pass" if status == 0 else "fail"
        cases = ast.literal_eval(test)
        for stdin, expected in zip(cases["stdin"], cases["stdout"], strict=True):
            status, out = run([sys.executable, "solution.py"], directory, CASE_TIMEOUT, stdin)
            if status is None:
                return "timeout"
            if status != 0 or lines(out) != lines(expected):
                return "fail"
        return "pass"


def judge(task: tuple[list[int], list[str], str, str]) -> str:
    """The verdict for one sampled response (rl/sampler.py), from its generated ids and its problem's names, tests and
    kind. 'not_closed' if the decoded text (BOS removed) has no END; otherwise `check` on the code in the answer, the
    text after the last END."""
    ids, names, test, kind = task
    tok = get_tokenizer()
    text = tok.enc.decode([i for i in ids if i != tok.bos_token_id])
    if END not in text:
        return "not_closed"
    return check((code(text.rpartition(END)[2].strip(), names), test, kind))


def stub(row: dict) -> str:
    """A wrong program for a row: each name its tests import returning None (pytest), or one printing nothing."""
    return "".join(f"def {name}(*args, **kwargs):\n    return None\n" for name in row["names"]) or "pass\n"


def known_good(split: str, n: int) -> dict:
    """Checks the harness on the pool's own R1 answers, which passed KodCode's tests in KodCode's environment: n rows
    drawn with rl.prompts.sample, their answers' verdicts, and the rows whose wrong-program stub passes (tests that
    accept anything). One worker per core of the CPU set."""
    rows = sample(read_pool(split), n, seed=1)
    with ProcessPoolExecutor(len(os.sched_getaffinity(0)), mp_context=SPAWN) as pool:
        verdicts = list(pool.map(check, [(code(r["r1_answer"], r["names"]), r["test"], r["kind"]) for r in rows],
                                 chunksize=4))
        stubs = list(pool.map(check, [(stub(r), r["test"], r["kind"]) for r in rows], chunksize=4))
    return {"split": split, "rows": n, "passed": verdicts.count("pass"),
            "failures": [{"id": r["id"], "verdict": v} for r, v in zip(rows, verdicts, strict=True) if v != "pass"],
            "stub_passes": [r["id"] for r, v in zip(rows, stubs, strict=True) if v == "pass"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--known-good", nargs=2, metavar=("SPLIT", "N"), required=True)
    split, n = parser.parse_args().known_good
    print(json.dumps(known_good(split, int(n)), indent=1))
