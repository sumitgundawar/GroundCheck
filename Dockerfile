FROM python:3.12-slim

# non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Ensure the cache directories are writable by the non-root user.
RUN mkdir -p /home/user/.cache/huggingface /home/user/.cache/sentence-transformers \
    && chown -R user /home/user/.cache /app

USER user

# Build the FAISS index at image build time so startup is instant and no runtime
# download is needed, then run the offline evaluation so the summary ships baked in.
RUN python scripts/build_index.py && python scripts/run_eval.py

# Listen on $PORT when the host provides one (Render), defaulting to 7860
# (Hugging Face Spaces). Shell form so the variable expands at runtime.
ENV PORT=7860
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
