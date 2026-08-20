FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# adf-core-python is installed from GitHub, so git is required while syncing.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.14-slim

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY main.py ./
COPY config/ ./config/
COPY precompute/ ./precompute/
COPY src/ ./src/

ENTRYPOINT ["/app/.venv/bin/python", "main.py"]
