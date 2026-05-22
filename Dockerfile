# Yard Monitor — Docker image
# CPU build. For GPU, base on nvidia/cuda and install torch-cuda.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Native deps that OpenCV / EasyOCR need
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-fetch the YOLO model so first run isn't slow inside the container
RUN python scripts/download_models.py || true

EXPOSE 8000
CMD ["python", "main.py"]
