# RL image "sgc-rl" (paper §5.2.2): the evaluation image (docker/Dockerfile.eval: vLLM 0.32.0, torch, transformers,
# EvalPlus) plus matplotlib, which some KodCode tests import, installed under the same constraints. Used by the RL
# sampler and the KodCode validation (rl/sampler.py; scripts/post_sft.sh, scripts/eval_kodcode.sh).
FROM sgc-eval
RUN uv pip install -c /opt/base-constraints.txt matplotlib==3.11.2
