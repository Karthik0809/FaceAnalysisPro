FROM python:3.10-slim

WORKDIR /app

# System libs required by OpenCV + MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgl1-mesa-glx git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install CLIP dependency for YOLO-World
RUN pip install --no-cache-dir "git+https://github.com/ultralytics/CLIP.git"

COPY . .

# Hugging Face Spaces uses port 7860; local default is 8000
# Set PORT env var to override
ENV PORT=7860

EXPOSE 7860

# Note: Webcam live stream requires a physical camera (local use only).
# All REST endpoints (analyze-image, register-face, etc.) work on cloud.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
