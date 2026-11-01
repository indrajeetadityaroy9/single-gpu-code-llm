"""The model's fixed hyperparameters (paper §4), shared by the tokenizer, the model and its CUDA kernels (no torch
import)."""

# Vocabulary size including special tokens: 2^15 (arXiv 2402.01035 §3.2; arXiv 2402.19173 §6.2; paper §3.1)
VOCAB_SIZE = 2**15
# Attention head size, chosen independently of the number of heads (arXiv 2002.07028 Theorem 1)
HEAD_DIM = 128
# Number of layers of the base model (d24)
NUM_HIDDEN_LAYERS = 24
# Hidden size by the shape rule of MobileLLM-350M's deep-and-narrow 32 layers x 960 (arXiv 2402.14905 Table 9;
# paper §4): 960 / 32 per layer, rounded up to a multiple of the head size (768 at d24)
HIDDEN_SIZE = -(-NUM_HIDDEN_LAYERS * (960 // 32) // HEAD_DIM) * HEAD_DIM
# RoPE base in both training stages: above the lower bound 6.4e5 for a 32K context (arXiv 2405.14591 Table 2; paper §4)
ROPE_THETA = 1_000_000
# Output logit soft-capping c: logits become c tanh(z / c) (arXiv 1611.09940 App. A.2; arXiv 2408.00118 §2; paper §4)
FINAL_LOGIT_SOFTCAPPING = 15.0
