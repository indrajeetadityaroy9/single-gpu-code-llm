"""The sgc decoder-only transformer (paper §4): pre-norm RMSNorm layers (arXiv 2002.04745 §3.1), a residual stream
rescaled and mixed with the normalized token embedding before each layer (arXiv 2402.02622 §3), QK-norm with learned
per-dimension gains (arXiv 2309.14322 §3.2.5), RoPE, local (sliding-window) attention in three of every four layers,
gated value embeddings in every second layer (arXiv 2601.22887 §4.1), a squared-ReLU MLP and output logits soft-capped
as c tanh(z / c). Parameters are fp32; activations are bf16."""

import zlib
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from corpus.mixture import FILE_LEVEL_STAGE, Stage
from model.config import FINAL_LOGIT_SOFTCAPPING, HEAD_DIM, HIDDEN_SIZE, NUM_HIDDEN_LAYERS, ROPE_THETA, VOCAB_SIZE
from model.ops.cross_entropy import lm_loss, lm_losses
from model.ops.flash_attn import attention
from model.ops.linear import EPS, qkv_proj, relu2_mlp


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    max_position_embeddings: int
    num_hidden_layers: int
    hidden_size: int
    # Soft-capping c, applied in every loss (cross_entropy.py)
    final_logit_softcapping: float


def has_value_embedding(layer_idx: int, *, num_hidden_layers: int) -> bool:
    """Whether layer layer_idx adds a value embedding: every second layer, including the last
    (arXiv 2601.22887 §4.1; paper §4)."""
    return layer_idx % 2 == (num_hidden_layers - 1) % 2


def layer_windows(config: ModelConfig) -> list[int]:
    """Attention window per layer (number of earlier keys each query sees): global attention over the whole context
    in every fourth layer and the last, local attention over the previous 512 tokens elsewhere
    (arXiv 2004.05150 §3.1; window size after arXiv 2503.19786 §5.2; paper §4)."""
    n = config.num_hidden_layers
    return [config.max_position_embeddings if i % 4 == 3 or i == n - 1 else 512 for i in range(n)]


def norm(x: Tensor) -> Tensor:
    """RMSNorm without a learned scale; statistics in fp32, epsilon EPS (FLT_EPSILON, as in qk_prologue.cu)."""
    return F.rms_norm(x, (x.size(-1),), eps=EPS)


class GemmLinear(nn.Linear):
    """Bias-free linear layer whose GEMM ops read the bf16 copy weight_bf16 and accumulate dW into the fp32
    weight.grad in place."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)
        self.register_buffer("weight_bf16", torch.empty_like(self.weight, dtype=torch.bfloat16), persistent=False)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        C = config.hidden_size
        self.q_proj = nn.Linear(C, C, bias=False)
        self.k_proj = nn.Linear(C, C, bias=False)
        self.v_proj = nn.Linear(C, C, bias=False)
        self.register_buffer("qkv_bf16", self.q_proj.weight.new_empty((3 * C, C), dtype=torch.bfloat16),
                             persistent=False)
        self.o_proj = GemmLinear(C, C)
        self.q_norm_weight = nn.Parameter(torch.empty(HEAD_DIM))
        self.k_norm_weight = nn.Parameter(torch.empty(HEAD_DIM))
        self.ve_gate = (nn.Linear(C, C // HEAD_DIM, bias=False)
                        if has_value_embedding(layer_idx, num_hidden_layers=config.num_hidden_layers) else None)

    def qkv_weights(self) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
        return self.q_proj.weight, self.k_proj.weight, self.v_proj.weight

    def forward(self, hidden_states: Tensor, residual: Tensor, input_ids: Tensor, ve_table: Tensor | None, *,
                cos_sin: tuple[Tensor, Tensor], window_size: int, cu_seqlens: Tensor,
                segments: Tensor) -> tuple[Tensor, Tensor]:
        """Return (residual + o_proj(attention(hidden_states)), the RMSNorm rstd of each row of that sum);
        hidden_states is the normalized residual stream."""
        qkv = qkv_proj(hidden_states, self.qkv_bf16, *(w.grad for w in self.qkv_weights()))
        # Value-embedding gate 2 sigmoid(x w) in (0, 2), 1 at zero init (paper §4). The sorted token ids let the
        # backward accumulate each table row's gradient in token order.
        ve = ((hidden_states, input_ids, ve_table, self.ve_gate.weight, *torch.sort(input_ids.flatten()))
              if ve_table is not None else (None,) * 6)
        return attention(qkv, self.q_norm_weight, self.k_norm_weight, *cos_sin, window_size, cu_seqlens, segments,
                         self.o_proj.weight_bf16, self.o_proj.weight.grad, residual, *ve)[:2]


class MLP(nn.Module):
    """Squared-ReLU MLP with a residual connection, h + down_proj(relu(up_proj(norm(h)))^2), given the stream h and
    the rstd of its rows; the norm, activation and residual add run in GEMM epilogues (paper §4)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        C = config.hidden_size
        self.up_proj = GemmLinear(C, 4 * C)
        self.down_proj = GemmLinear(4 * C, C)

    def forward(self, h: Tensor, rstd: Tensor) -> Tensor:
        return relu2_mlp(h, rstd, self.up_proj.weight_bf16, self.down_proj.weight_bf16, self.up_proj.weight.grad,
                         self.down_proj.weight.grad)[0]


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, hidden_states: Tensor, input_ids: Tensor, ve_table: Tensor | None, *,
                cos_sin: tuple[Tensor, Tensor], window_size: int, cu_seqlens: Tensor, segments: Tensor) -> Tensor:
        h, rstd = self.self_attn(norm(hidden_states), hidden_states, input_ids, ve_table, cos_sin=cos_sin,
                                 window_size=window_size, cu_seqlens=cu_seqlens, segments=segments)
        return self.mlp(h, rstd)


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        """Allocate shapes only (on the meta device); init_weights sets the values."""
        super().__init__()
        self.config = config
        self.window_sizes = layer_windows(config)
        C, n = config.hidden_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, C)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(n)])
        self.lm_head = nn.Linear(C, VOCAB_SIZE, bias=False)
        self.resid_lambdas = nn.Parameter(torch.empty(n))
        self.x0_lambdas = nn.Parameter(torch.empty(n))
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(VOCAB_SIZE, C) for i in range(n)
                                           if has_value_embedding(i, num_hidden_layers=n)})
        for name in ("cos", "sin"):
            self.register_buffer(name, None, persistent=False)
        self.register_load_state_dict_post_hook(lambda module, _: module.refresh_gemm_weights())

    @torch.no_grad()
    def init_weights(self, *, seed: int) -> None:
        """Uniform init in ±sqrt(3/C) ("Efficient BackProp" §4.6). Zero init for o_proj, down_proj, lm_head
        (arXiv 1901.09321 §3), the value-embedding gates and x0_lambdas; ones for resid_lambdas and the QK-norm
        gains. Each tensor gets its own generator seeded with seed * 2^32 + crc32(name)."""
        bound = (3 / self.config.hidden_size) ** 0.5
        zero = ("self_attn.o_proj.weight", "mlp.down_proj.weight", "lm_head.weight", "self_attn.ve_gate.weight",
                "x0_lambdas")
        one = ("q_norm_weight", "k_norm_weight", "resid_lambdas")
        for name, t in self.named_parameters():
            if name.endswith(zero):
                t.zero_()
            elif name.endswith(one):
                t.fill_(1)
            else:
                t.uniform_(-bound, bound,
                           generator=torch.Generator(t.device).manual_seed(seed * 2**32 + zlib.crc32(name.encode())))
        self.init_buffers()
        self.refresh_gemm_weights()

    @torch.no_grad()
    def refresh_gemm_weights(self) -> None:
        """Copy the fp32 weights into their bf16 GEMM copies in place (a compiled graph holds the copies); call after
        init, load and every optimizer step. A copy is reallocated only when a load moved its weight to another
        device."""
        linears = [m for m in self.modules() if isinstance(m, GemmLinear)]
        attentions = [m for m in self.modules() if isinstance(m, Attention)]
        for m in linears:
            if m.weight_bf16.device != m.weight.device:
                m.weight_bf16 = torch.empty_like(m.weight, dtype=torch.bfloat16)
        for m in attentions:
            if m.qkv_bf16.device != m.q_proj.weight.device:
                m.qkv_bf16 = torch.empty_like(m.qkv_bf16, device=m.q_proj.weight.device)
        torch._foreach_copy_([m.weight_bf16 for m in linears] + [b for m in attentions for b in m.qkv_bf16.chunk(3)],
                             [m.weight for m in linears] + [w for m in attentions for w in m.qkv_weights()])

    def init_buffers(self) -> None:
        """Build the fp32 RoPE tables (T, HEAD_DIM/2) of angles t ROPE_THETA^(-2i/HEAD_DIM) (arXiv 2104.09864 §3.2.2).
        Angles are computed in fp64 and rounded once, since an fp32 angle can be off by up to t 2^-24."""
        device = self.embed_tokens.weight.device
        exponent = torch.arange(0, HEAD_DIM, 2, dtype=torch.float64, device=device) / HEAD_DIM
        t = torch.arange(self.config.max_position_embeddings, dtype=torch.float64, device=device)
        freqs = torch.outer(t, ROPE_THETA ** -exponent)
        self.cos, self.sin = freqs.cos().float(), freqs.sin().float()

    def set_stage(self, stage: Stage) -> None:
        """Switch to a training stage's context length: recompute the windows and RoPE tables; weights unchanged."""
        self.config = replace(self.config, max_position_embeddings=stage.seq_len)
        self.window_sizes = layer_windows(self.config)
        self.init_buffers()

    def hidden(self, input_ids: Tensor, cu_seqlens: Tensor, segments: Tensor) -> Tensor:
        """Final normalized hidden states (B*T, C) for input_ids (B, T) int32. cu_seqlens holds the int32 boundaries
        of the packed document chunks, of which the first segments + 1 are used (segments: 0-dim int32 on the host)."""
        hidden_states = x0 = norm(F.embedding(input_ids, self.embed_tokens.weight).to(torch.bfloat16))
        for i, layer in enumerate(self.layers):
            # Residual scalars x <- resid_lambda x + x0_lambda x0 (paper §4). The 0-dim fp32 scalars times bf16 let
            # Inductor reduce their gradients in fp32 without fp32 copies of the mixed stream.
            hidden_states = self.resid_lambdas[i] * hidden_states + self.x0_lambdas[i] * x0
            ve_table = self.value_embeds[str(i)].weight if str(i) in self.value_embeds else None
            hidden_states = layer(hidden_states, input_ids, ve_table, cos_sin=(self.cos, self.sin),
                                  window_size=self.window_sizes[i], cu_seqlens=cu_seqlens, segments=segments)
        return norm(hidden_states).flatten(0, 1)

    def forward(self, input_ids: Tensor, labels: Tensor, cu_seqlens: Tensor,
                segments: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the mean loss over non-padding labels, the per-position loss (B*T,) (0 at padding) and the
        per-position logsumexp, all fp32; the last two carry no gradient."""
        loss, _, _, losses, lse = lm_loss(self.hidden(input_ids, cu_seqlens, segments), self.lm_head.weight,
                                          labels.flatten(), self.config.final_logit_softcapping)
        return loss, losses, lse

    def losses(self, input_ids: Tensor, labels: Tensor, cu_seqlens: Tensor, segments: Tensor) -> Tensor:
        """Per-position next-token loss (B*T,) fp32, 0 at padding labels."""
        return lm_losses(self.hidden(input_ids, cu_seqlens, segments), self.lm_head.weight, labels.flatten(),
                         self.config.final_logit_softcapping)


def meta_model() -> CausalLM:
    """The base model (model/config.py) on the meta device at the file-level stage's context length."""
    config = ModelConfig(max_position_embeddings=FILE_LEVEL_STAGE.seq_len, num_hidden_layers=NUM_HIDDEN_LAYERS,
                         hidden_size=HIDDEN_SIZE, final_logit_softcapping=FINAL_LOGIT_SOFTCAPPING)
    with torch.device("meta"):
        return CausalLM(config)


def num_non_embedding_params(model: CausalLM) -> int:
    """Non-embedding parameters: the decoder layers plus lm_head, the count the token budget (tokens per parameter)
    scales with. Embedding tables are excluded (arXiv 2001.08361 §1.3); the untied output layer is counted
    (arXiv 2406.19146 §3.2)."""
    return sum(p.numel() for p in model.layers.parameters()) + model.lm_head.weight.numel()


def accumulated_weights(model: CausalLM) -> list[nn.Parameter]:
    """Weights whose fp32 .grad the GEMM ops accumulate into in place; a training backward needs .grad allocated."""
    return [w for m in model.modules()
            for w in (m.qkv_weights() if isinstance(m, Attention)
                      else (m.weight,) if isinstance(m, GemmLinear) else ())]


def num_matmul_params(model: CausalLM) -> int:
    """PaLM's N for model FLOPs: every 2-D weight the forward multiplies by (embedding tables are lookups)."""
    return sum(p.numel() for m in model.modules() if not isinstance(m, nn.Embedding)
               for p in m.parameters(recurse=False) if p.ndim == 2)


def non_embedding_flops_per_token(model: CausalLM) -> int:
    """DeepSeek LLM's non-embedding FLOPs per token, M = 72 n_layer d_model^2 + 12 n_layer d_model l_seq
    (arXiv 2401.02954 §3.2 Eq. 2); local-attention layers are counted as global."""
    n, d = model.config.num_hidden_layers, model.config.hidden_size
    return 72 * n * d * d + 12 * n * d * model.config.max_position_embeddings
