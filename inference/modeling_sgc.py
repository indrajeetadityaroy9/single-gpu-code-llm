"""The sgc model for Hugging Face transformers. It reads the weights and config.json that inference/export_hf.py
writes and computes the same next-token logits as model/modeling.py's CausalLM. It implements the architecture of
paper §4, and vLLM serves it for the evaluation of paper §6.1 (loaded through auto_map with trust_remote_code).

The architecture (paper §4) is a decoder-only transformer with pre-norm RMSNorm without learned scales and no biases;
the input is x0 = RMSNorm(token embedding), and before each layer the residual stream is mixed as
x <- lambda_r x + lambda_0 x0 (two learned scalars per layer). Attention uses multi-head attention (6 heads of 128)
with QK-norm (RMSNorm with a learned per-dimension scale) before RoPE (base 1e6); three of every four layers attend
to a sliding window of the previous 512 tokens, the others (and the last layer) to the whole context. Every second
layer adds a gated value embedding to the attention values. The MLP is ReLU(x W1)^2 W2, and the output logits, from
a separate (untied) output layer, are soft-capped as c tanh(z / c) with c = 15.

The module layout and names follow transformers' Qwen3 (modeling_qwen3.py), so tools that expect that layout work.
Two details match the training kernels rather than library code: RoPE rotates by -theta, i.e. the sign convention of
arXiv:2104.09864 §3.2.2 as model/kernels/qk_prologue.cu implements it, so transformers' rotary helpers are not used;
and every operation the training forward computes in fp32 is computed in fp32 here too.
"""

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class SgcConfig(PreTrainedConfig):
    """The config.json written by inference/export_hf.py, which sets every field explicitly."""
    model_type = "sgc"


def norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm without a learned scale (model.modeling.norm), statistics in fp32."""
    return F.rms_norm(x, (x.size(-1),), eps=eps)


def rotary(positions: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE angles as (cos, sin), each (*positions.shape, head_dim / 2) in fp32. The angle of position t and pair i
    is t * theta^(-2i/D), computed in fp64 as CausalLM.init_buffers does."""
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float64, device=positions.device) / head_dim
    freqs = positions.double()[..., None] * theta**-exponent
    return freqs.cos().float(), freqs.sin().float()


def qk_norm_rope(x: torch.Tensor, gain: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 eps: float) -> torch.Tensor:
    """QK-norm then RoPE on x (..., H, D): RMSNorm with the learned per-dimension scale `gain`, then rotation by
    -theta, all in fp32 and rounded to x's dtype once (as qk_rotate in model/kernels/qk_prologue.cu);
    cos, sin (..., D/2)."""
    m = x.float()
    m = m * torch.rsqrt(m.square().mean(-1, keepdim=True) + eps) * gain
    m1, m2 = m.chunk(2, dim=-1)
    c, s = cos[..., None, :], sin[..., None, :]
    return torch.cat([m1 * c + m2 * s, m2 * c - m1 * s], dim=-1).to(x.dtype)


class Gate(nn.Module):
    """The value-embedding gate's weight W (H, C): the gate per head is 2 sigmoid(W x_norm), between 0 and 2.
    It is a plain parameter rather than an nn.Linear so that vLLM's Transformers backend keeps it in fp32: vLLM gives
    its linear layers' weights a weight_loader, and _keep_in_fp32_modules_strict does not apply to those."""

    def __init__(self, C: int, H: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(H, C))


class SgcAttention(nn.Module):
    """Multi-head attention with QK-norm, RoPE, an optional sliding window and an optional value embedding."""
    def __init__(self, config: SgcConfig, layer_idx: int) -> None:
        super().__init__()
        C, D, H = config.hidden_size, config.head_dim, config.num_attention_heads
        self.config, self.layer_idx, self.head_dim = config, layer_idx, D
        self.scaling, self.is_causal = D**-0.5, True
        self.q_proj = nn.Linear(C, H * D, bias=False)
        self.k_proj = nn.Linear(C, H * D, bias=False)
        self.v_proj = nn.Linear(C, H * D, bias=False)
        self.o_proj = nn.Linear(H * D, C, bias=False)
        self.q_norm_weight = nn.Parameter(torch.ones(D))
        self.k_norm_weight = nn.Parameter(torch.ones(D))
        self.ve_gate = Gate(C, H) if layer_idx in config.value_embedding_layers else None
        self.sliding_window = (config.sliding_window if config.layer_types[layer_idx] == "sliding_attention"
                               else None)

    def forward(self, n, value_embeds, position_embeddings, attention_mask, past_key_values, **kwargs):
        """n (B, T, C) the normalized residual stream; value_embeds (B, T, C) this layer's value embedding or None."""
        shape = (*n.shape[:-1], -1, self.head_dim)
        q, k, v = (proj(n).view(shape) for proj in (self.q_proj, self.k_proj, self.v_proj))
        if value_embeds is not None:  # value embedding (model/kernels/ve_gate.cu): v + 2 sigmoid(W n) ve per head, fp32
            gate = 2 * torch.sigmoid(F.linear(n.float(), self.ve_gate.weight))
            v = (v.float() + gate[..., None] * value_embeds.view(v.shape).float()).to(v.dtype)
        eps, (cos, sin) = self.config.rms_norm_eps, position_embeddings
        q = qk_norm_rope(q, self.q_norm_weight, cos, sin, eps).transpose(1, 2)
        k = qk_norm_rope(k, self.k_norm_weight, cos, sin, eps).transpose(1, 2)
        v = v.transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        attention = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        out, _ = attention(self, q, k, v, attention_mask, scaling=self.scaling, sliding_window=self.sliding_window,
                           **kwargs)
        return self.o_proj(out.reshape(*shape[:-2], -1).contiguous())


class SgcMLP(nn.Module):
    """The MLP: squared ReLU, ReLU(x W_up)^2 W_down, hidden size 4C."""
    def __init__(self, config: SgcConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(x)).square())


class SgcDecoderLayer(nn.Module):
    """One pre-norm layer: x + attention(RMSNorm(x)), then + MLP(RMSNorm(x)). The residual mix is in SgcModel."""
    def __init__(self, config: SgcConfig, layer_idx: int) -> None:
        super().__init__()
        self.eps = config.rms_norm_eps
        self.self_attn = SgcAttention(config, layer_idx)
        self.mlp = SgcMLP(config)

    def forward(self, h, value_embeds, position_embeddings, attention_mask, past_key_values, **kwargs):
        h = h + self.self_attn(norm(h, self.eps), value_embeds, position_embeddings, attention_mask, past_key_values,
                               **kwargs)
        return h + self.mlp(norm(h, self.eps))


class SgcPreTrainedModel(PreTrainedModel):
    config_class = SgcConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _supports_attention_backend = True  # vLLM's Transformers backend runs models that declare it
    # the parameters the training forward reads in fp32; export_hf.py writes these in fp32 and the rest in bf16
    _keep_in_fp32_modules_strict: ClassVar[list[str]] = ["resid_lambdas", "x0_lambdas", "q_norm_weight",
                                                         "k_norm_weight", "ve_gate"]


class SgcModel(SgcPreTrainedModel):
    """The decoder: token embedding, value-embedding tables, the layers and the residual-mix scalars
    (resid_lambdas = lambda_r, x0_lambdas = lambda_0); returns the final RMSNorm of the residual stream."""
    def __init__(self, config: SgcConfig) -> None:
        super().__init__(config)
        n, V, C = config.num_hidden_layers, config.vocab_size, config.hidden_size
        self.embed_tokens = nn.Embedding(V, C)
        self.layers = nn.ModuleList([SgcDecoderLayer(config, i) for i in range(n)])
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(V, C) for i in config.value_embedding_layers})
        self.resid_lambdas = nn.Parameter(torch.ones(n))
        self.x0_lambdas = nn.Parameter(torch.zeros(n))
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs) -> BaseModelOutputWithPast:
        """input_ids is always required (the value embeddings look them up); inputs_embeds, when given, must be
        their token embeddings."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + seen)[None]
        if not isinstance(masks := attention_mask, dict):
            mask_kwargs = {"config": self.config, "inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
                           "past_key_values": past_key_values, "position_ids": position_ids}
            masks = {"full_attention": create_causal_mask(**mask_kwargs),
                     "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs)}
        # computed per call: a registered buffer would be cast to the model's dtype
        position_embeddings = rotary(position_ids, self.config.head_dim, self.config.rope_parameters["rope_theta"])
        eps = self.config.rms_norm_eps
        h = x0 = norm(inputs_embeds, eps)
        for i, layer in enumerate(self.layers):
            # residual mix x <- lambda_r x + lambda_0 x0 (CausalLM.hidden), in fp32 and rounded once, as Inductor does
            h = (self.resid_lambdas[i] * h.float() + self.x0_lambdas[i] * x0.float()).to(h.dtype)
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            h = layer(h, ve, position_embeddings, masks[self.config.layer_types[i]], past_key_values, **kwargs)
        return BaseModelOutputWithPast(last_hidden_state=norm(h, eps),
                                       past_key_values=past_key_values if use_cache else None)


class SgcForCausalLM(SgcPreTrainedModel, GenerationMixin):
    """SgcModel plus the untied output layer and the logit soft-cap."""
    def __init__(self, config: SgcConfig) -> None:
        super().__init__(config)
        self.model = SgcModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, logits_to_keep=0, **kwargs) -> CausalLMOutputWithPast:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden = out.last_hidden_state[:, keep]
        # as in the evaluation's head (model/ops/cross_entropy.py lm_losses): the bf16 matmul with fp32 output (a
        # product of two bf16 values is exact in fp32), then the soft-cap in fp32
        logits = F.linear(hidden.float(), self.lm_head.weight.float())
        c = self.config.final_logit_softcapping
        return CausalLMOutputWithPast(logits=c * torch.tanh(logits / c), past_key_values=out.past_key_values)
