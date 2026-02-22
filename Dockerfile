# AlignLayer scoring server
# Runs on ARM64 (Pi 4/5) or amd64. CPU-only torch — no CUDA/MPS needed.
#
# Build:
#   docker build -t alignlayer-scorer .
#
# Run (corpus mounted from host so it survives image rebuilds):
#   docker run -d \
#     --name alignlayer-scorer \
#     --restart unless-stopped \
#     -p 8000:8000 \
#     -v /path/to/scores-cache.jsonl:/data/scores-cache.jsonl:ro \
#     alignlayer-scorer
#
# Override checkpoint or corpus:
#   docker run ... \
#     -e ALIGNLAYER_CHECKPOINT=/app/model/checkpoints/best-v5.pt \
#     -e ALIGNLAYER_CORPUS=/data/scores-cache.jsonl \
#     alignlayer-scorer

FROM python:3.11-slim

WORKDIR /app

# Install torch CPU wheel first (large, cache this layer separately)
# ARM64 and amd64 wheels both available from PyPI for torch 2.x
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model code and checkpoint
COPY model/siamese.py   model/siamese.py
COPY model/serve.py     model/serve.py
COPY model/checkpoints/ model/checkpoints/

# Corpus is expected at /data/scores-cache.jsonl (mount from host)
# Fall back to empty path so the container starts even without a mount
ENV ALIGNLAYER_CHECKPOINT=model/checkpoints/best.pt
ENV ALIGNLAYER_CORPUS=/data/scores-cache.jsonl
ENV ALIGNLAYER_K=5
ENV ALIGNLAYER_THRESHOLD=0.55
ENV PORT=8000

EXPOSE 8000

CMD ["python", "model/serve.py"]
