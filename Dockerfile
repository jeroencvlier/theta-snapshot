# Build stage
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/

# Change to app directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create venv and install dependencies
RUN uv venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install using uv sync which handles git dependencies better
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Copy the project files
COPY . .

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies and copy uv
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application files first (needed for uv sync)
COPY . .

# Create fresh venv in runtime and install from lock file
RUN uv venv && \
    uv sync --frozen

# Set environment variables
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Use uv run for the default command
# CMD ["uv", "run", "python", "-m", "predict"]
CMD ["/bin/bash"]