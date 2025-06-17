FROM python:3.11-slim

WORKDIR /app

# uv
COPY --from=ghcr.io/astral-sh/uv:0.4.15 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

ENV PYTHONPATH=/app

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

ENV PATH="$PATH:/app/.venv/bin"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "src.main:app"]