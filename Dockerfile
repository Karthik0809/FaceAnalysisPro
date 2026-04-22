FROM python:3.10-slim

WORKDIR /app

# System libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgl1-mesa-glx \
    build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

# Step A — lightweight API + image libs
RUN pip install --no-cache-dir \
    "numpy>=1.24.0,<2.0" \
    "Pillow>=10.0.0" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.27.0" \
    "python-multipart>=0.0.9" \
    "jinja2>=3.1.3" \
    "tqdm>=4.66.0" \
    "ollama>=0.3.0"

# Step B — OpenCV headless (pinned <4.11 because 4.11+ requires numpy>=2, conflicting with TF)
RUN pip install --no-cache-dir "opencv-python-headless==4.10.0.84"

# Step C — MediaPipe (pinned; later versions removed solutions API)
RUN pip install --no-cache-dir "mediapipe==0.10.14"

# Step D — scikit-learn
RUN pip install --no-cache-dir "scikit-learn>=1.4.0"

# Step E — ChromaDB vector store
RUN pip install --no-cache-dir "chromadb>=0.5.0,<1.0.0"

# Step F — TensorFlow CPU (explicit to avoid 2 GB CUDA build)
RUN pip install --no-cache-dir "tensorflow-cpu>=2.16.0,<2.19.0"

# Step G — DeepFace without opencv-python dep (headless already installed)
# --no-deps skips deepface's opencv-python pull; install its other deps manually
RUN pip install --no-cache-dir --no-deps "deepface>=0.0.89" && \
    pip install --no-cache-dir \
        "mtcnn>=0.1.0" \
        "retina-face>=0.0.1" \
        "gdown>=4.0.0" \
        "pandas>=1.3.0" \
        "matplotlib>=3.2.2" \
        "requests>=2.23.0" \
        "keras" \
        "tf-keras" \
        "fire" \
        "flask" \
        "flask-cors" \
        "python-dotenv" \
        "lightdsa" \
        "lightphe"

COPY . .

ENV PORT=7860
EXPOSE 7860
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
