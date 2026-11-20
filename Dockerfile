# The image: everything the service needs to RUN, and nothing it does not.
# Secrets (the DeepSeek key) are NOT here. The host passes them in when the container starts.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# Run as a normal user, not root, so a bug in the service cannot touch the whole container.
# (uid 1000 is what Hugging Face Spaces expects.)
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
RUN mkdir -p /app/.cache && chown appuser:appuser /app/.cache

# --- 1. Libraries. Own layer: rebuilt only when the requirements change, not on every code edit.
# The default PyTorch build carries GPU libraries worth several GB that a CPU host never uses.
# The CPU-only build is a fraction of that size.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-service.txt ./
COPY demo_assets/constraints.txt ./constraints.txt
RUN pip install -r requirements-service.txt -c constraints.txt

# --- 2. The embedding model's weights, downloaded NOW (build time) into the image, so the
# container never has to download anything when it starts.
USER appuser
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# --- 3. Our code and the demo data. Changes most often, so it comes last (keeps rebuilds fast).
COPY --chown=appuser:appuser agent_demo.py api.py demo_dbs.py embedder.py exemplars.py limits.py runtime_tools.py safe_sql.py ./
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser demo_assets ./demo_assets

ENV SPIDER_DB_PATH=/app/demo_assets/databases \
    SCHEMA_PKL_PATH=/app/demo_assets/spider_schema_lookup.pkl

# Fail the BUILD, not the first visitor, if the model is not really inside the image:
# this loads it with network access to the model hub switched off.
RUN HF_HUB_OFFLINE=1 python -c "from embedder import warm_up; warm_up(); print('model loads offline: ok')"
ENV HF_HUB_OFFLINE=1

# One process, many threads (the daily counters live in memory; see limits.py).
# The host may set PORT; 7860 is the default Hugging Face Spaces looks for.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api:build_app --factory --host 0.0.0.0 --port ${PORT:-7860}"]
