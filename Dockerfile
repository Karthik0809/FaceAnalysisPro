FROM python:3.10-slim

WORKDIR /app

# System libs for OpenCV headless + MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgl1-mesa-glx git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch — saves ~1.5 GB vs the full CUDA build
RUN pip install --no-cache-dir \
    "torch==2.2.0+cpu" \
    "torchvision==0.17.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

# CLIP — required by YOLO-World (install before ultralytics)
RUN pip install --no-cache-dir "git+https://github.com/ultralytics/CLIP.git"

# Remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces uses 7860; local default is 8000
ENV PORT=7860
EXPOSE 7860

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
