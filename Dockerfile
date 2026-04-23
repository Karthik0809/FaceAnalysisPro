FROM python:3.10-slim

WORKDIR /app

# System libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgl1 \
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

# Step B+C — MediaPipe installs opencv-contrib-python (needs numpy>=2) which
# conflicts with TF 2.15's numpy<2 pin. Install mediapipe first, then
# force-reinstall headless OpenCV so cv2 comes from the numpy-1.x-compatible build.
RUN pip install --no-cache-dir "mediapipe==0.10.14" && \
    pip install --no-cache-dir --force-reinstall "opencv-python-headless==4.10.0.84"

# Step D — scikit-learn
RUN pip install --no-cache-dir "scikit-learn>=1.4.0"

# Step F — DeepFace + TensorFlow CPU
# protobuf must stay <4 for mediapipe==0.10.14; tensorflow-cpu<2.16 is the
# last series that accepts protobuf 3.x. retina-face/mtcnn installed with
# --no-deps so they cannot upgrade tensorflow or protobuf.
RUN pip install --no-cache-dir \
        "protobuf>=4.25.3,<5.0.0" \
        "tensorflow-cpu>=2.13.0,<2.16.0" && \
    pip install --no-cache-dir --no-deps "deepface>=0.0.89" && \
    pip install --no-cache-dir --no-deps \
        "mtcnn>=0.1.0" \
        "retina-face>=0.0.1" && \
    pip install --no-cache-dir \
        "gdown>=4.0.0" \
        "pandas>=1.3.0" \
        "matplotlib>=3.2.2" \
        "requests>=2.23.0" \
        "gunicorn>=20.1.0" \
        "fire" \
        "flask" \
        "flask-cors" \
        "python-dotenv" \
        "lightdsa" \
        "lightphe"

# mediapipe pulls in jax 0.6.x, but tensorflow-cpu 2.15 pins ml_dtypes==0.3.2
# which is too old for jax 0.6 (needs float8_e3m4). jax is not used by our app —
# uninstalling it lets TF's wrapped "except ImportError" handle it cleanly.
RUN pip uninstall -y jax jaxlib || true

COPY . .

ENV PORT=7860
EXPOSE 7860
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
