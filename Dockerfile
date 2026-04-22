FROM python:3.10-slim

WORKDIR /app

# System libs: OpenCV headless + MediaPipe + C++ compiler for chromadb/hnswlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgl1-mesa-glx \
    git build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch — much smaller than CUDA build (~200 MB vs ~2 GB)
RUN pip install --no-cache-dir \
    "torch==2.2.0+cpu" \
    "torchvision==0.17.0+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    || echo "PyTorch CPU unavailable — emotion MLP disabled, DeepFace fallback active"

# CLIP for YOLO-World (install before ultralytics)
RUN pip install --no-cache-dir "git+https://github.com/ultralytics/CLIP.git"

# CV + data science layer
RUN pip install --no-cache-dir \
    "mediapipe==0.10.14" \
    "opencv-python-headless>=4.9.0" \
    "numpy>=1.24.0,<2.0" \
    "Pillow>=10.0.0" \
    "scikit-learn>=1.4.0" \
    "tqdm>=4.66.0"

# DeepFace + TensorFlow (largest dependency group)
RUN pip install --no-cache-dir \
    "tensorflow-cpu>=2.16.0,<2.19.0" \
    "deepface>=0.0.89"

# Vector DB
RUN pip install --no-cache-dir "chromadb>=0.5.0,<1.0.0"

# Object detection
RUN pip install --no-cache-dir "ultralytics>=8.2.0"

# API server + misc (starlette version is pinned by FastAPI automatically)
RUN pip install --no-cache-dir \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.27.0" \
    "python-multipart>=0.0.9" \
    "jinja2>=3.1.3" \
    "ollama>=0.3.0"

COPY . .

# HuggingFace Spaces uses 7860; local default is 8000
ENV PORT=7860
EXPOSE 7860

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
