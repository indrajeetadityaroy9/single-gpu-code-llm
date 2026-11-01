"""Export a training checkpoint as a Hugging Face model directory, which transformers and vLLM load (paper §6.1).

Input: a training checkpoint (checkpoint-<step>.pt, written by train/checkpoint.py). Output: model.safetensors,
config.json, generation_config.json, the tokenizer files and a copy of inference/modeling_sgc.py, which transformers
and vLLM's Transformers backend load with trust_remote_code (the config's auto_map points at it).

With --chat (an SFT or RL model, paper §5.2), the tokenizer gets a chat template that reproduces the post-training
prompt format: BOS, the messages' contents joined by "\\n\\n" and stripped of surrounding whitespace, then
"\\n\\n<think>\\n" (THINK), so the model always starts by reasoning. This is the same text that SFT trains on
(corpus/posttrain/build.py render encodes the stripped prompt and THINK as one string). A system message is not a
separate role: it is prepended to the user message, separated by "\\n\\n". The generation config then sets the
sampling defaults to the evaluation protocol's (eval/official_termination.py SAMPLING: temperature 0.6, top-p 0.95,
the reasoning-model defaults of DeepSeek-R1, arXiv:2501.12948 §3).
Without --chat (a pretrained model), the generation config has no sampling defaults.

    python -m inference.export_hf <checkpoint.pt> <out dir> [--chat]
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import processors
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter

from corpus.posttrain.sft import THINK
from corpus.tokenizer import BOS, SPECIAL_TOKENS, TOKENIZER, get_tokenizer
from eval.official_termination import SAMPLING
from inference.modeling_sgc import SgcPreTrainedModel
from model.config import HEAD_DIM, ROPE_THETA, VOCAB_SIZE
from model.modeling import ModelConfig, has_value_embedding, layer_windows
from model.ops.linear import EPS

# The post-training prompt format (see the module docstring). Jinja's trim equals Python's str.strip. THINK is written
# as a Jinja string literal because Jinja drops a template's own trailing newline (keep_trailing_newline is off).
CHAT_TEMPLATE = ("{{ bos_token }}{{ messages | map(attribute='content') | join('\\n\\n') | trim }}"
                 f"{{{{ {json.dumps(THINK)} }}}}")


def hf_weights(model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a CausalLM state dict to Hugging Face names (prefix "model.", except lm_head) and cast it for
    model.safetensors. Weights the training forward reads in bf16 (matmul weights, the token and value embeddings)
    are rounded to nearest bf16, as training's own bf16 copies are; the parameters listed in
    _keep_in_fp32_modules_strict (residual scalars, QK-norm scales, value-embedding gates) stay fp32. One known
    difference: the compiled training forward computes x0 as the RMSNorm of the fp32 embedding row (Inductor drops
    the bf16 cast), while the export stores the token embedding in bf16."""
    fp32 = set(SgcPreTrainedModel._keep_in_fp32_modules_strict)
    return {(k if k == "lm_head.weight" else f"model.{k}"):
            v.detach().to(torch.float32 if fp32 & set(k.split(".")) else torch.bfloat16).cpu().contiguous()
            for k, v in model.items()}


def export(checkpoint: Path, out: Path, *, chat: bool) -> None:
    out.mkdir(parents=True)
    state = torch.load(checkpoint, map_location="cpu", mmap=True)
    config = ModelConfig(**state["config"])
    n, C, T = config.num_hidden_layers, config.hidden_size, config.max_position_embeddings
    save_file(hf_weights(state["model"]), out / "model.safetensors", metadata={"format": "pt"})

    tokenizer = get_tokenizer()
    windows = layer_windows(config)
    (short,) = {w for w in windows if w < T}
    shutil.copy(Path(__file__).with_name("modeling_sgc.py"), out / "modeling_sgc.py")
    (out / "config.json").write_text(json.dumps({
        "architectures": ["SgcForCausalLM"], "model_type": "sgc",
        "auto_map": {"AutoConfig": "modeling_sgc.SgcConfig", "AutoModel": "modeling_sgc.SgcModel",
                     "AutoModelForCausalLM": "modeling_sgc.SgcForCausalLM"},
        "vocab_size": VOCAB_SIZE, "hidden_size": C,
        "intermediate_size": state["model"]["layers.0.mlp.up_proj.weight"].shape[0], "num_hidden_layers": n,
        "num_attention_heads": C // HEAD_DIM, "num_key_value_heads": C // HEAD_DIM, "head_dim": HEAD_DIM,
        "max_position_embeddings": T, "rope_parameters": {"rope_type": "default", "rope_theta": float(ROPE_THETA)},
        # transformers' window counts the query: S keys before it + 1
        "sliding_window": short + 1,
        "layer_types": ["sliding_attention" if w < T else "full_attention" for w in windows],
        "value_embedding_layers": [i for i in range(n) if has_value_embedding(i, num_hidden_layers=n)],
        "final_logit_softcapping": config.final_logit_softcapping, "rms_norm_eps": EPS,
        # vLLM's head (LogitsProcessor._apply_head) in fp32, as the evaluation computes it; transformers ignores it
        "head_dtype": "float32", "tie_word_embeddings": False, "dtype": "bfloat16",
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.bos_token_id,
    }, indent=2) + "\n")
    (out / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.bos_token_id}
        | (SAMPLING if chat else {}), indent=2) + "\n")

    fast = TikTokenConverter(vocab_file=str(TOKENIZER / "tokenizer.tiktoken"), pattern=tokenizer.pattern,
                             extra_special_tokens=list(SPECIAL_TOKENS)).converted()
    fast.post_processor = processors.TemplateProcessing(single=f"{BOS} $A",
                                                       special_tokens=[(BOS, tokenizer.bos_token_id)])
    PreTrainedTokenizerFast(tokenizer_object=fast, bos_token=BOS, eos_token=BOS, model_max_length=T,
                            chat_template=CHAT_TEMPLATE if chat else None).save_pretrained(out)


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]), chat="--chat" in sys.argv[3:])
