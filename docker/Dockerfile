# Training image "sgc" for the GH200 (sm_90). The base, NGC PyTorch, provides torch, Triton, CUDA, nvcc and numpy;
# these are not in uv.lock. uv.lock's packages are installed in a virtual environment on top of it.
# Base images are pinned by manifest digest and git dependencies by commit. The image holds no code or data:
# scripts/container.sh mounts the checkout at WORKDIR, and the entrypoint runs paths.py, which creates artifacts/,
# before every command.

FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /workspace/sgc

# Rust toolchain (pinned version, minimal profile) for the Rust crates, which are built on first use; CARGO_HOME is in
# artifacts/cache (below). rustup-init 1.29.2 for aarch64, checked against its published SHA-256.
ENV RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:$PATH
RUN curl -sSfo /tmp/rustup-init \
      https://static.rust-lang.org/rustup/archive/1.29.2/aarch64-unknown-linux-gnu/rustup-init && \
    echo "a9c168dbea81a84671c6b7a5c1795cb0eb699020110d107ddfca427bd797e697  /tmp/rustup-init" | sha256sum -c - && \
    chmod +x /tmp/rustup-init && \
    CARGO_HOME=/usr/local/cargo /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.99.1 && \
    rm /tmp/rustup-init
# The base image's `flash_attn` package (FA2 with an FA4 beta) would shadow the flash_attn.cute version uv.lock pins
RUN python3 -m pip uninstall -y flash-attn flash-attn-4

# Dependency layer, rebuilt only when pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --system-site-packages --python /usr/bin/python3 $VIRTUAL_ENV && \
    uv sync --frozen --active

# The problems of openai/human-eval-infilling, read by corpus/decontaminate.py through read_problems. The checkout is
# put on sys.path with a .pth file, since read_problems reads <package>/../data. HumanEval-RandomSpan.jsonl.gz, the
# name data.py:13 reads, is linked to the shipped file.
RUN git clone https://github.com/openai/human-eval-infilling /opt/human-eval-infilling && \
    git -C /opt/human-eval-infilling checkout 88062ff9859c875d04db115b698ed4b0f0395170 && \
    ln -s HumanEval-RandomSpanInfilling.jsonl.gz /opt/human-eval-infilling/data/HumanEval-RandomSpan.jsonl.gz && \
    echo /opt/human-eval-infilling > "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/human_eval_infilling.pth"

# TF32 off: the NGC image sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which runs fp32 cuBLAS matmuls in TF32. One OpenMP
# thread, since parallelism is by process. Compiler, kernel and user caches live in the mounted artifacts/cache
# (paths.py CACHE) and are reused across containers.
ENV XDG_CACHE_HOME=/workspace/sgc/artifacts/cache
ENV HF_HOME=$XDG_CACHE_HOME/huggingface HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/inductor TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton \
    QUACK_CACHE_DIR=$XDG_CACHE_HOME/quack FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$XDG_CACHE_HOME/flash_attn CARGO_HOME=$XDG_CACHE_HOME/cargo \
    OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# NGC's entrypoint sources /opt/nvidia/entrypoint.d before running the command: this step creates the artifact and
# cache directories (paths.py)
RUN echo 'python /workspace/sgc/paths.py || exit' > /opt/nvidia/entrypoint.d/90-sgc-paths.sh
