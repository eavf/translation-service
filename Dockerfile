FROM python:3.11-slim

# sentencepiece may compile from source on uncommon platforms; build-essential
# covers that case without adding unnecessary weight to common x86_64 images.
RUN apt-get -o Acquire::Retries=5 update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch CPU-only ──────────────────────────────────────────────────────────
# Using the dedicated CPU wheel index guarantees that no CUDA runtime
# (~2 GB) is ever pulled, which is important on a Synology NAS with limited
# disk and no GPU. This step is separated so Docker can cache the heavy layer.
RUN pip install --no-cache-dir --retries 10 --timeout 120 \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

# ── Application dependencies ──────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY app/ ./app/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /models/huggingface \
    && chmod -R 777 /models

# ── Hugging Face model cache ──────────────────────────────────────────────────
# Mapped to ./models on the host via docker-compose.yml so the model
# survives container recreation without re-downloading (~300 MB for opus-mt-sk-fr).
# Synology: make sure the NAS volume backing ./models has at least 2 GB free
# to accommodate future model changes.
# HF_HOME is the canonical variable since transformers 4.22; TRANSFORMERS_CACHE
# is omitted because it was deprecated and logs a warning at >=4.36.
ENV HF_HOME=/models/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

EXPOSE 8088

# start_period is generous (300 s) because the very first run downloads the
# model from Hugging Face before serving any requests.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8088/health')" \
        || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088"]
