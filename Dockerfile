# PaperPilot — container for local use or Hugging Face Spaces (Docker SDK).
FROM python:3.12-slim

# libgomp1 provides the OpenMP runtime that faiss/torch CPU wheels link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as a non-root user (uid 1000). Match that so the
# model cache we populate at build time is owned by the runtime user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    # 1.5B is slow on a free CPU Space; ship the snappier 0.5B model.
    PAPERPILOT_LLM=Qwen/Qwen2.5-0.5B-Instruct
WORKDIR $HOME/app

# Install CPU-only torch first (much smaller than the default CUDA build), then
# the rest. torch>=2.2.0 in requirements is already satisfied, so it's not
# re-fetched from PyPI.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# App code + prebuilt FAISS index (so no ingestion is needed at runtime).
COPY --chown=user . .

# Pre-download both models into the image cache so the first user request is
# fast and works even if the Hub is rate-limited at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" && \
    python -c "import os; from transformers import AutoModelForCausalLM, AutoTokenizer; \
m=os.environ['PAPERPILOT_LLM']; AutoTokenizer.from_pretrained(m); \
AutoModelForCausalLM.from_pretrained(m)"

EXPOSE 7860
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
