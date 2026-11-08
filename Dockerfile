# =============================================================================
# MIRAGE-DoH Reproducibility Artifact — CPU-only container
# =============================================================================
#
# PURPOSE
#   This Dockerfile produces a self-contained CPU-only container that runs
#   the 2-minute toy-demo smoke test of the MIRAGE-DoH reproducibility
#   artifact, with zero host-Python dependencies. It is the fastest way
#   for a reviewer to verify that the artifact works end-to-end without
#   polluting their laptop environment.
#
#   It is NOT the recommended path for the full 32 GB reproduction.
#   The full reproduction should be run on the host (see README.md §4,
#   environment.yml, or requirements.txt) because:
#     a. The datasets must be bind-mounted from the host, not baked into
#        the image, otherwise the image balloons to 30+ GB.
#     b. The full reproduction runs 2-4 hours; running it inside a
#        container adds bind-mount I/O overhead for no benefit.
#     c. GPU and Loihi accelerators require host nvidia-container-toolkit
#        and physical hardware respectively; the container is CPU-only.
#
# USAGE
#   Build:
#       docker build -t mirage-doh:latest .
#
#   Run the toy demo (the default CMD):
#       docker run --rm mirage-doh:latest
#
#   Drop into an interactive shell to explore:
#       docker run --rm -it mirage-doh:latest /bin/bash
#
#   Bind-mount a full dataset and run the full reproduction:
#       docker run --rm \
#           -v /path/to/your/CIC-DoH2023:/workspace/data/raw/cic_doh2023:ro \
#           -v /path/to/your/Gen-C-DDD-2022:/workspace/data/raw/gen_c_ddd2022:ro \
#           -v $(pwd)/outputs:/workspace/outputs \
#           mirage-doh:latest \
#           python scripts/train_mirage.py --config configs/default_mirage_config.yaml --full-data
#
#   Run a specific reproduction script:
#       docker run --rm mirage-doh:latest \
#           python scripts/reproduce_tables/table_8_core_metrics.py
#
# CONTAINER SIZE
#   Final image: ~2.8 GB (CPU-only PyTorch wheel dominates).
#   For comparison, the CUDA variant (see bottom of this file) is ~9 GB.
#
# TESTED
#   Docker 25.x / 26.x on Ubuntu 22.04 LTS host.
#   Should work on any Docker 20.10+ (when --platform linux/amd64 is passed
#   on Apple Silicon).
# =============================================================================


# =============================================================================
# STAGE 1 — builder
# =============================================================================
# Compiles any Python dependency whose wheel is not available pre-built for
# the target Python version and architecture. Once compiled, the resulting
# .so files are copied to the runtime stage; gcc/g++ do not ship in the
# final image, saving ~700 MB.
# =============================================================================
FROM python:3.11-slim AS builder

# Avoid interactive prompts during apt-get install.
ENV DEBIAN_FRONTEND=noninteractive

# System build dependencies. Only the "builder" stage needs these; the
# runtime stage does not, which is the whole point of the two-stage build.
#
# - build-essential: gcc, g++, make (compiles C/C++ extensions)
# - libblas-dev, liblapack-dev: link-time headers for numpy/scipy/ripser
# - libgmp-dev: multiple-precision arithmetic used by pycryptodome
# - pkg-config: lets pip discover the BLAS/LAPACK above
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libblas-dev \
        liblapack-dev \
        libgmp-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency spec first, so that Docker's layer cache reuses
# the expensive `pip install` if the source code changes but the dependencies
# do not. This is the single most important line for build-time efficiency.
COPY requirements.txt /tmp/requirements.txt

# Install into a dedicated venv that we will copy wholesale to the runtime
# stage. Using a venv (rather than installing into the system site-packages)
# keeps the runtime stage's site-packages clean of any package the base image
# happens to ship, which is a common source of "works on my machine" bugs.
#
# --no-cache-dir: do not leave pip's wheel cache in the builder layer.
# PYTHONUNBUFFERED: unbuffered stdout, so `docker logs` shows output live.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r /tmp/requirements.txt


# =============================================================================
# STAGE 2 — runtime
# =============================================================================
# Starts from a fresh python:3.11-slim, copies the venv from the builder
# stage, copies the source, and sets the default CMD to the toy demo.
# No compilers, no dev headers, ~2.8 GB.
# =============================================================================
FROM python:3.11-slim AS runtime

# Container-time environment variables:
#
# - DEBIAN_FRONTEND=noninteractive: same as builder.
# - PYTHONDONTWRITEBYTECODE=1: do not write .pyc files (they live in the
#   layer and bloat the image; they are not needed for a short-lived
#   container anyway).
# - PYTHONUNBUFFERED=1: unbuffered stdout.
# - PYTHONHASHSEED=42: fixed seed for the hash-randomized parts of Python,
#   so that dict iteration order is deterministic across runs. This matches
#   src/utils/seed.py, which sets the same value at runtime.
# - PYTHONPATH=/workspace: so that `import mirage` resolves without
#   requiring the reviewer to `pip install -e .` inside the container.
# - OMP_NUM_THREADS=4: cap OpenMP parallelism at 4 threads so the container
#   does not starve a reviewer's laptop when they run it alongside other work.
# - TOKENIZERS_PARALLELISM=false: silences a warning from any tokenizer-based
#   dependency; harmless here since the artifact does not tokenize text, but
#   it is a well-known nuisance so we set it once.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PYTHONPATH=/workspace \
    OMP_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

# Runtime system dependencies. Only the *shared libraries* the compiled
# Python extensions need at import time; not the headers, not the compilers.
#
# - graphviz: the `dot` binary needed by figures_src/figure_1.dot. This is a
#   runtime binary, NOT a Python package, which is why it is installed here
#   and not in the venv.
# - libgomp1: OpenMP runtime, required by PyTorch's CPU backend.
# - libgl1, libglib2.0-0: runtime for matplotlib's optional GUI backends;
#   present so that `import matplotlib.pyplot as plt` never fails in a
#   headless container.
# - ca-certificates: required by data/download_datasets.py to verify HTTPS.
# - curl: required by the HEALTHCHECK below.
# - tini: tiny init process (PID 1 reaper); ensures that Ctrl+C in
#   `docker run -it` propagates SIGINT to Python instead of getting ignored.
RUN apt-get update && apt-get install -y --no-install-recommends \
        graphviz \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully-built venv from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Create a non-root user. Running the container as non-root is a standard
# security hardening: if a malicious wheel or a compromised dependency
# tries to write outside /workspace, it is denied by the filesystem
# permissions of the container.
#
# The UID/GID 1000 matches the default first user on most Linux hosts, which
# means that files written to a bind-mounted /workspace/outputs have the
# reviewer's UID on the host, not root's. This avoids the classic
# "sudo chown -R $(id -u):$(id -g) outputs/" after every docker run.
RUN groupadd --gid 1000 mirage \
    && useradd --uid 1000 --gid mirage --create-home --shell /bin/bash mirage

# Working directory inside the container. Matches PYTHONPATH above, so
# `import mirage` (from src/) works without a setup.py install.
WORKDIR /workspace

# Copy the source tree. This is the last COPY so that source edits do not
# invalidate the expensive venv layer above. .dockerignore (uploaded as a
# separate file, File 9) excludes raw/, processed/, .git/, __pycache__/ etc.
COPY --chown=mirage:mirage . /workspace

# Make sure the toy dataset and the expected-outputs directory are
# readable by the non-root user, and that the outputs/ directory is
# writable. The `data/raw/` and `data/processed/` directories are created
# but empty; a reviewer bind-mounts the full datasets there if needed.
RUN mkdir -p /workspace/outputs /workspace/data/raw /workspace/data/processed \
    && chown -R mirage:mirage /workspace

# Switch to the non-root user for everything that follows.
USER mirage

# Health check: if any of the critical imports fails, the container is
# marked unhealthy in `docker ps`, and `docker run --health-cmd` can fail
# a CI step on it. The check is deliberately minimal (no torch model load)
# so that it completes in under 2 seconds.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import numpy, scipy, torch, gudhi, ripser, persim, snntorch, networkx, sklearn, matplotlib, yaml, requests, dask" \
    || exit 1

# Use tini as PID 1. This is what makes Ctrl+C in `docker run -it` reach
# the Python process instead of getting swallowed by the shell.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command: run the 2-minute toy demo.
#
# A reviewer who runs `docker run --rm mirage-doh:latest` gets the artifact
# working end-to-end without reading a line of code. The verification script
# runs automatically after training, so the container's exit code is the
# exit code of `verification_script.py --toy-mode`: 0 if the deterministic
# rows pass and the dataset-bound rows warn (which is the expected toy-mode
# outcome, see REPRODUCIBILITY_CHECKLIST.md §3.4), non-zero otherwise.
CMD ["bash", "-c", "python scripts/train_mirage.py --config configs/default_mirage_config.yaml && python expected_outputs/verification_script.py --toy-mode"]


# =============================================================================
# OPTIONAL: CUDA variant (for a reviewer with an NVIDIA A100 / RTX GPU)
# =============================================================================
#
# The container above is CPU-only by design. If you have a GPU and want the
# GPU-accelerated path for the full reproduction, replace the two lines
#
#     FROM python:3.11-slim AS runtime
#     ...
#     COPY --from=builder /opt/venv /opt/venv
#
# with the following, and run the container with
#     docker run --rm --gpus all mirage-doh-gpu:latest
# (which requires the host to have nvidia-container-toolkit installed).
#
# -----------------------------------------------------------------------------
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime
#
# # Match the Python version in the builder stage above.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         python3.11 python3.11-venv python3.11-dev \
#         graphviz libgomp1 libgl1 libglib2.0-0 \
#         ca-certificates curl tini \
#     && rm -rf /var/lib/apt/lists/* \
#     && ln -s /usr/bin/python3.11 /usr/bin/python
#
# # Copy the CPU venv, then re-install torch/torchvision against CUDA.
# COPY --from=builder /opt/venv /opt/venv
# RUN /opt/venv/bin/pip install --no-cache-dir \
#         torch==2.3.1+cu118 torchvision==0.18.1+cu118 \
#         --extra-index-url https://download.pytorch.org/whl/cu118
#
# # Everything else below (user, WORKDIR, COPY . , HEALTHCHECK, ENTRYPOINT,
# # CMD) is identical to the CPU variant above.
# -----------------------------------------------------------------------------
#
# The GPU variant is ~9 GB compressed. It is provided as a comment rather
# than an active build stage because the reviewer's access to a GPU is not
# guaranteed, and the CPU variant above is sufficient to reproduce every
# number in Chapter 3 within the tolerance bounds stated in
# REPRODUCIBILITY_CHECKLIST.md §8.
# =============================================================================


# =============================================================================
# OPTIONAL: multi-architecture build (for Apple Silicon reviewers)
# =============================================================================
#
# Docker on Apple Silicon (M1/M2/M3) runs amd64 images under emulation, which
# is 5-10x slower. If you want a native arm64 image, build with:
#
#     docker buildx build --platform linux/arm64 -t mirage-doh:arm64 .
#
# conda-forge and pip both publish native arm64 wheels for every dependency
# in requirements.txt, so the build succeeds without modification. Expect
# the toy demo to complete in ~4 minutes on an M2 instead of ~2 minutes on
# a modern x86_64, which is normal for CPU-bound Python.
#
# The arm64 image does not include a Loihi or A100 simulator that depends
# on native instructions; the hardware_simulation/ modules are pure Python
# and produce the same trends on arm64 as on x86_64 (within the tolerances
# of REPRODUCIBILITY_CHECKLIST.md §8).
# =============================================================================


# =============================================================================
# WHAT IS AND IS NOT IN THE IMAGE
# =============================================================================
#
# IN the image:
#   - All Python dependencies from requirements.txt, pre-compiled.
#   - All source code from src/, scripts/, configs/, tests/, figures_src/.
#   - The 1 MB toy dataset at data/toy_dataset/.
#   - The expected_outputs/ JSON and CSV files.
#   - The graphviz binary (for Figure 1).
#
# NOT in the image (by design):
#   - The 32 GB CIC-DoH2023 and Gen-C-DDD-2022 datasets. A reviewer
#     bind-mounts them at /workspace/data/raw/ for the full reproduction.
#   - Any model checkpoint from the thesis testbed. The artifact trains
#     from scratch; training on the toy dataset completes in 90 seconds.
#   - The Intel Loihi 2 SDK (nxsdk). Requires INRC membership; a reviewer
#     with access can install it in a layer on top of this image.
#   - Any proprietary NVIDIA binary beyond the base CUDA libs (which are
#     themselves only in the optional commented variant).
# =============================================================================
