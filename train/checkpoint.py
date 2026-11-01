"""
Checkpoints: RUNS/<run_name>/checkpoint-<step>.pt holds the FP32 weights ("model") after `step` training steps and the
ModelConfig ("config") at the context they trained in (train/loop.py saves them).
"""

from pathlib import Path

import torch

from model.modeling import CausalLM, ModelConfig
from paths import RUNS


def checkpoint_path(run_name: str, step: int) -> Path:
    return RUNS / run_name / f"checkpoint-{step}.pt"


def load_model(path: Path, *, device: torch.device) -> CausalLM:
    """A saved model at the context and ModelConfig (output-logit cap included) its weights trained in."""
    checkpoint = torch.load(path, map_location=device)
    with torch.device("meta"):
        model = CausalLM(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], assign=True)
    model.init_buffers()
    return model
