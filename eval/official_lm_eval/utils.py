"""mbpp_lf.yaml's functions: lm-evaluation-harness's tasks/mbpp/utils.py with the prompt's fields (statement, code,
asserts) line-ending-normalized (eval/line_endings.py); its pass_at_1 unchanged."""
import importlib.util
from pathlib import Path

import lm_eval

from eval.line_endings import lf

_spec = importlib.util.spec_from_file_location("lm_eval_mbpp_utils",
                                               Path(lm_eval.__file__).parent / "tasks" / "mbpp" / "utils.py")
mbpp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mbpp)
pass_at_1 = mbpp.pass_at_1


def normalized(doc: dict) -> dict:
    return doc | {"text": lf(doc["text"]), "code": lf(doc["code"]), "test_list": [lf(t) for t in doc["test_list"]]}


def process_docs(dataset):
    return dataset.map(normalized)


def list_fewshot_samples() -> list[dict]:
    return [normalized(doc) for doc in mbpp.list_fewshot_samples()]
