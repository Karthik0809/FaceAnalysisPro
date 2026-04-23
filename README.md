---
title: PerceptAI
emoji: 👁
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# PerceptAI

**Live Demo:** [karthikmulugu08-perceptai.hf.space](https://karthikmulugu08-perceptai.hf.space)

A real-time computer vision system combining face analysis, full-body pose estimation, hand tracking, and object detection — streamed live to a browser dashboard via WebSocket.

---

## Features

| Category | What it does |
|---|---|
| Face | 468-point mesh, emotion (7 classes), age, gender, Facenet512 recognition |
| Eyes | Blink detection, PERCLOS drowsiness, iris gaze (L/C/R) |
| Mouth | Smile detection, talking detection (MAR) |
| Head | Pitch / yaw / roll via solvePnP + Rodrigues |
| Hands | 21-landmark finger skeleton per hand |
| Body | 33-keypoint skeleton, posture classification |
| Scene | 60+ object classes via YOLO-World open-vocabulary detection |
| Memory | Cross-session face store (ChromaDB), DBSCAN clustering |
| AI | Session reports + live commentary via Ollama (rule-based fallback) |
| Infra | FastAPI WebSocket stream, SQLite logging, video recording |

---

## Tech Stack

| Layer | Tool | Version |
|---|---|---|
| Computer Vision | OpenCV (headless) | 4.10 |
| Landmark Tracking | MediaPipe | 0.10.14 |
| Object Detection | YOLO-World X | ≥ 8.2 |
| Face Recognition | DeepFace (Facenet512) | ≥ 0.0.89 |
| Deep Learning | TensorFlow CPU | 2.15 |
| Vector DB | ChromaDB | ≥ 0.5 |
| Local LLM | Ollama (llama3.2, optional) | ≥ 0.3 |
| Clustering | scikit-learn DBSCAN | ≥ 1.4 |
| API Server | FastAPI + Uvicorn | ≥ 0.110 |
| Database | SQLite | built-in |
| Deployment | Docker on HuggingFace Spaces | — |

---

## Local Setup

```bash
git clone https://github.com/Karthik0809/PerceptAI.git
cd PerceptAI
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** and click **▶ Start**.

> DeepFace models (~500 MB) download automatically on first run.

### Optional: Ollama for AI reports

```bash
ollama pull llama3.2
```

If Ollama is not running the system falls back to rule-based reports automatically.

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard UI |
| POST | `/analyze-image` | Upload image → full analysis + annotated result |
| POST | `/register-face` | Register a known identity |
| GET | `/known-faces` | List registered identities |
| POST | `/search-face` | Find similar faces in vector DB |
| GET | `/report` | Generate AI session report |
| GET | `/commentary` | Live one-sentence frame description |
| GET | `/cluster-faces` | DBSCAN clustering of stored embeddings |
| GET | `/history` | SQLite detection log |
| GET | `/stats` | Aggregated session statistics |
| GET | `/docs` | Auto-generated Swagger UI |

---

## Project Structure

```
PerceptAI/
├── main.py                 # FastAPI app, endpoints, WebSocket stream
├── analyzer.py             # Core CV pipeline (face mesh, hands, pose, YOLO)
├── llm_reporter.py         # Ollama LLM + ChromaDB + rule-based fallback
├── db.py                   # SQLite session logging
├── expression_model.py     # PyTorch MLP wrapper (optional)
├── train_expression_model.py
├── templates/index.html    # Single-page dashboard
├── known_faces/            # Drop face images here for recognition
├── requirements.txt
└── Dockerfile
```
