# Test web app (Streamlit). It only calls the API: no model, no preprocessing.
# Build from the repository root: docker build -f app/Dockerfile -t gbg-app .
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS base

# Install the dependencies of the `app` group only (streamlit, httpx).
FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group app --no-install-project

# Run the web app without the installation tools.
FROM base AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    API_URL=http://api:8000

WORKDIR /app

RUN useradd --create-home --uid 10001 app

COPY --from=builder /opt/venv /opt/venv
COPY .streamlit ./.streamlit
COPY app ./app

USER app

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
