---
title: FaceAnalysis Pro
sdk: docker
app_port: 7860
license: mit
---

# FaceAnalysis Pro

A production-quality real-time computer vision system combining face analysis, full-body pose estimation, hand tracking, and object detection — streamed live to a clean browser dashboard. Everything runs locally with no cloud APIs required.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-orange) ![YOLO](https://img.shields.io/badge/YOLO--World-X-red)

---

## What It Does

Point a webcam at yourself and the system simultaneously:

- Tracks **468 facial landmarks** with a live mesh overlay
- Draws a **33-keypoint glowing body skeleton** that moves with you in real time
- Draws **21 hand landmarks** per hand with full finger-bone connections
- Classifies **7 emotions** (happy, sad, angry, surprise, fear, disgust, neutral) via DeepFace
- Estimates **head pose** (pitch / yaw / roll) with 3D axis arrows using solvePnP
- Tracks **eye gaze** (left / center / right) via iris landmark ratios
- Detects **blinks** and computes **PERCLOS** drowsiness score over a 90-frame rolling window
- Classifies **posture** (upright / slouching / tilted / lean left / lean right)
- Detects **60+ object classes** using YOLO-World open-vocabulary detection (pens, phones, wallets, watches, etc.)
- **Recognizes registered faces** using Facenet512 cosine similarity (99.6% LFW benchmark)
- Generates **AI session reports** via local Ollama LLM (rule-based fallback if Ollama not running)
- Stores face embeddings in **ChromaDB** and runs **DBSCAN clustering** for re-identification
- Records sessions as **AVI video** files
- Serves a **REST API** with auto-generated Swagger UI

---

## Architecture

```
Webcam (OpenCV)
    │
    ├─► MediaPipe Face Mesh (468 landmarks)        ← main thread, every frame
    │       ├─ Head Pose  (solvePnP + Rodrigues)
    │       ├─ Eye metrics (EAR, PERCLOS, iris gaze)
    │       ├─ Smile / Talking (MAR, mouth-width ratio)
    │       └─ Emotion (DeepFace, 7 classes)
    │
    ├─► MediaPipe Hands (21 landmarks × 2 hands)  ← main thread, every frame
    │       └─ Full finger skeleton overlay
    │
    ├─► MediaPipe Body Pose (33 keypoints)         ← main thread, every frame
    │       └─ Posture classifier (shoulder tilt, lateral lean)
    │
    ├─► YOLO-World Object Detection                ← background thread, every 5 frames
    │       └─ 60+ custom classes, open-vocabulary (pen, phone, wallet, watch…)
    │
    └─► Background Analysis Thread
            ├─ MediaPipe Face Detection (bbox)
            ├─ DeepFace: Facenet512 recognition + age + gender
            └─ ChromaDB: embed → upsert face vector
```

---

## Feature Overview

| Category | Feature | Method |
|---|---|---|
| Face | 468-point mesh overlay | MediaPipe FaceMesh |
| Face | Emotion (7 classes) | DeepFace |
| Face | Age & gender | DeepFace |
| Face | Face recognition | Facenet512 + cosine similarity (threshold 0.68) |
| Face | Head pose (P/Y/R) | solvePnP + Rodrigues decomposition |
| Eyes | Blink detection | Eye Aspect Ratio (EAR) state machine |
| Eyes | Drowsiness (PERCLOS) | 90-frame rolling closure ratio |
| Eyes | Gaze direction | Iris landmark ratio (L/C/R) |
| Mouth | Smile detection | Mouth-width / face-width ratio |
| Mouth | Talking detection | Mouth Aspect Ratio (MAR) |
| Hands | Full finger skeleton | MediaPipe Hands (21 landmarks, HAND_CONNECTIONS) |
| Body | Full skeleton overlay | MediaPipe Pose (33 KP, glowing lines) |
| Body | Posture classification | Shoulder tilt + lateral lean + torso height |
| Scene | Object detection | YOLO-World X (open-vocabulary, 60+ custom classes) |
| Memory | Cross-session face store | ChromaDB (cosine similarity space) |
| Memory | Face clustering | DBSCAN on stored Facenet512 embeddings |
| AI | Session report | Ollama local LLM (rule-based fallback) |
| AI | Live commentary | Ollama or rule-based fallback |
| Infra | Live stream | FastAPI WebSocket, base64 MJPEG |
| Infra | Session logging | SQLite |
| Infra | Video recording | OpenCV VideoWriter (XVID AVI) |
| Infra | REST API | FastAPI with auto-generated /docs |
| Infra | Containerization | Docker |

---

## Tech Stack

| Layer | Library / Tool | Version |
|---|---|---|
| Computer Vision | OpenCV | ≥ 4.9 |
| Landmark tracking | MediaPipe | 0.10.14 |
| Object detection | YOLO-World X (ultralytics) | ≥ 8.2 |
| Face recognition | DeepFace (Facenet512) | ≥ 0.0.89 |
| Vector DB | ChromaDB | ≥ 0.5 |
| Local LLM | Ollama (llama3.2, optional) | ≥ 0.3 |
| Clustering | scikit-learn DBSCAN | ≥ 1.4 |
| API server | FastAPI + Uvicorn | ≥ 0.110 |
| Database | SQLite (built-in) | — |

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/FaceAnalysisPro.git
cd FaceAnalysisPro

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **First run:** DeepFace models (~500 MB) and YOLO-World weights (~140 MB) download automatically.

### 3. (Optional) Set up Ollama for AI reports

```bash
# Install from https://ollama.com/download
ollama pull llama3.2
```

If Ollama is not running, the system automatically uses a rule-based reporter — no crash, no error.

### 4. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** → click **▶ Start**

### Docker

```bash
docker build -t face-analysis .
# With webcam:
docker run -p 8000:8000 --device=/dev/video0 face-analysis
# Static image analysis only (no webcam):
docker run -p 8000:8000 face-analysis
```

---

## Usage

### Live Stream
Click **▶ Start** on the dashboard. The webcam feed appears with all overlays active — face mesh, hand skeleton, body skeleton, YOLO bounding boxes, head pose arrows, and all metric badges updating in real time.

### Register a Known Face
Go to the **🛠️ Tools** tab → **Register a Face**: enter a name and upload a clear frontal photo. The system averages 4 augmented embeddings (original + horizontal flip + brightness variants) for robust recognition.

### Analyze a Static Image
Go to **🛠️ Tools** → **Analyze an Image**: upload any photo and get the full face analysis with an annotated result image — works without a webcam.

### Search the Vector Database
In **🛠️ Tools** → **Analyze an Image** → click **🔎 Search DB**: finds the most visually similar faces stored across all sessions using ChromaDB cosine similarity.

### AI Session Report
In **🛠️ Tools** → **✨ Generate Report**: produces a natural-language summary of the current session.

### Video Recording
Click **⏺ Record** in the header to start/stop. Videos saved to `recordings/` as XVID AVI.

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard UI |
| POST | `/analyze-image` | Upload image → full analysis JSON + annotated image |
| POST | `/register-face` | Register a new known identity |
| GET | `/known-faces` | List registered identities |
| POST | `/search-face` | Find similar faces in ChromaDB |
| GET | `/report` | Generate AI session report |
| GET | `/commentary` | Live one-sentence frame description |
| GET | `/cluster-faces` | DBSCAN clustering of stored embeddings |
| GET | `/history` | SQLite detection log |
| GET | `/stats` | Aggregated session statistics |
| POST | `/recording/start` | Start video recording |
| POST | `/recording/stop` | Stop and save recording |
| GET | `/docs` | Auto-generated Swagger UI |

---

## Project Structure

```
FaceAnalysisPro/
├── main.py                    # FastAPI app, all endpoints, WebSocket stream
├── analyzer.py                # Core CV pipeline (face mesh, hands, pose, YOLO, metrics)
├── expression_model.py        # PyTorch MLP wrapper
├── train_expression_model.py  # Standalone MLP training script
├── llm_reporter.py            # Ollama LLM + ChromaDB + rule-based fallback
├── db.py                      # SQLite session logging
├── templates/
│   └── index.html             # Single-page dashboard (tabbed, clean dark UI)
├── known_faces/               # Drop face images here for recognition
├── models/                    # Trained MLP checkpoint
├── chroma_db/                 # Persistent ChromaDB face vector store
├── recordings/                # Saved session videos (AVI)
├── yolov8x-worldv2.pt         # YOLO-World weights (auto-downloaded)
├── requirements.txt
└── Dockerfile
```

---

## Key Design Decisions

**Why YOLO-World instead of standard YOLOv8?**
Standard YOLOv8 (COCO 80 classes) frequently misclassifies common desk objects — a pen becomes a "toothbrush", a wallet becomes a "book". YOLO-World uses CLIP open-vocabulary matching with exactly the class names we define (pen, wallet, wristwatch, lip balm, etc.).

**Why Facenet512 with threshold 0.68?**
A lower threshold (0.50) causes false positives. At 0.68, the system only confirms a match when genuinely confident. The reference embedding is an average of 4 augmented variants (flip + brightness) for robustness across lighting and angle variation.

**Why MediaPipe 0.10.14 pinned?**
Versions ≥ 0.10.15 removed the `solutions` API. Version 0.10.14 is the last stable release with `mp.solutions.face_mesh`, `mp.solutions.hands`, and `mp.solutions.pose`.

---

## Resume Bullets

> Built a real-time multi-modal behavioral analysis system combining MediaPipe (468-pt face mesh + 21-pt hand skeleton + 33-pt body pose), DeepFace emotion/age/gender analysis, Facenet512 face recognition, and YOLO-World open-vocabulary object detection into a single live-streamed FastAPI WebSocket pipeline

> Engineered a CLIP-based open-vocabulary object detection pipeline (YOLO-World X) with 60+ custom classes and class-name tuning to eliminate COCO misclassifications; improved face recognition precision by averaging multi-augmented Facenet512 embeddings with a calibrated cosine similarity threshold

> Designed a local-first Gen AI layer using Ollama (llama3.2) and ChromaDB for session reporting and cross-session face similarity search; implemented DBSCAN clustering on 512-dim Facenet512 embeddings for unsupervised visitor re-identification
