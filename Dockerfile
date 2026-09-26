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
# PyTorch for the CPU by default, which keeps the image a few GB smaller. For
# training on NVIDIA GPUs build with
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Pinned. These were installed unpinned, so two builds a month apart shipped
# different versions of the library the imaging models run on -- and the
# evaluation summary baked into this image was produced by whichever one the
# build happened to get.
ARG TORCH_VERSION=2.14.0
ARG TORCHVISION_VERSION=0.29.0
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" \
       "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "psycopg[binary]==3.*" "pymysql==1.*"

COPY --chown=user . .

# Ensure the cache directories are writable by the non-root user.
# /data is where deployments mount their volume (deploy/docker-compose.yml). A
# new named volume copies this folder's ownership, so the app can write to it.
RUN mkdir -p /home/user/.cache/huggingface /home/user/.cache/sentence-transformers /data \
    && chown -R user /home/user/.cache /app /data

USER user

# Build the search index at image build time so startup is instant and no runtime
# download is needed, then run the offline evaluation so the summary ships baked in.
RUN python scripts/build_index.py && python scripts/run_eval.py

# Listen on $PORT when the host provides one (Render), defaulting to 7860
# (Hugging Face Spaces). Shell form so the variable expands at runtime.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
