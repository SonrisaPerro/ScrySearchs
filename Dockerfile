FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf-cache

# CPU-only torch first: the default PyPI wheel bundles several GB of CUDA libraries this server can't use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the CLIP model and the pinned search index into the image so startup downloads nothing.
COPY assets.py index_version.txt ./
RUN python -c "import assets; assets.ensure_assets(); from sentence_transformers import SentenceTransformer; SentenceTransformer(assets.MODEL_NAME)"
# Everything is baked in now; never contact the Hugging Face Hub at startup.
ENV HF_HUB_OFFLINE=1

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
