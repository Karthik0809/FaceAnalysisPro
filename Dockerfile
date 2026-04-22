FROM python:3.10-slim

WORKDIR /app

# System libs required by OpenCV + MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Webcam passthrough required for live stream: docker run --device=/dev/video0 ...
# For static-image analysis only, no device flag needed.
CMD ["python", "main.py"]
