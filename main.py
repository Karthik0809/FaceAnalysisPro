import asyncio
import base64
import os
import shutil
import time
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from analyzer import EMOTION_COLORS, FaceAnalyzer
from db import SESSION_ID, get_history, get_stats, init_db, log_detection

app       = FastAPI(title="Real-Time Face Analysis System", version="3.0.0")
templates = Jinja2Templates(directory="templates")
analyzer  = FaceAnalyzer(known_faces_dir="known_faces")
init_db()

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class FaceResult(BaseModel):
    id:               int
    name:             str
    emotion:          str
    emotion_scores:   dict[str, float] = {}
    emotion_history:  list[str]        = []
    emotion_source:   str              = "deepface"
    age:              Any
    gender:           str
    confidence:       float
    pitch:            float
    yaw:              float
    roll:             float
    gaze:             str
    attention:        str
    ear:              float
    perclos:          float
    blinks:           int
    smile:            bool
    talking:          bool
    smile_ratio:      float
    mar:              float
    posture:          str              = "UPRIGHT"

class AnalyzeResponse(BaseModel):
    face_count:      int
    faces:           list[FaceResult]
    annotated_image: str
    posture:         str              = "UPRIGHT"
    objects:         list[dict]       = []

# ── Recording state ───────────────────────────────────────────────────────────

_rec: dict = {"active": False, "writer": None, "path": None, "started": None}

# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ── Face registration ─────────────────────────────────────────────────────────

@app.post("/register-face")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    ext  = os.path.splitext(file.filename or "face.jpg")[1] or ".jpg"
    path = os.path.join("known_faces", f"{name}{ext}")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    cache = os.path.join("known_faces", "representations_facenet512.pkl")
    if os.path.exists(cache):
        os.remove(cache)
    analyzer.reload_known_faces()
    return JSONResponse({"status": "ok", "message": f"Registered '{name}' successfully."})


@app.get("/known-faces")
async def list_known_faces():
    names = [
        os.path.splitext(fn)[0]
        for fn in os.listdir("known_faces")
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    return {"names": names}

# ── Static image analysis ─────────────────────────────────────────────────────

@app.post("/analyze-image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Upload any image → full face analysis JSON + base64-annotated image."""
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Could not decode image."}, status_code=400)
    processed, metadata = analyzer.process(img)
    _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return AnalyzeResponse(
        face_count=metadata["face_count"],
        faces=metadata["faces"],
        annotated_image=base64.b64encode(buf).decode(),
        posture=metadata.get("posture", "UPRIGHT"),
        objects=metadata.get("objects", []),
    )

# ── Face search via ChromaDB ──────────────────────────────────────────────────

@app.post("/search-face")
async def search_face(file: UploadFile = File(...), top_k: int = 5):
    """
    Upload a face photo → find the most similar faces in the vector DB.
    Uses Facenet512 embeddings + ChromaDB cosine similarity search.
    """
    from llm_reporter import search_similar_faces, chroma_face_count

    if chroma_face_count() == 0:
        return JSONResponse({"message": "Vector DB is empty. Run the live stream first to populate it.", "results": []})

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Could not decode image."}, status_code=400)

    from deepface import DeepFace
    try:
        rep = DeepFace.represent(img_path=img, model_name="Facenet512",
                                 enforce_detection=False, detector_backend="retinaface")
        emb = rep[0]["embedding"]
    except Exception as e:
        return JSONResponse({"error": f"Embedding failed: {e}"}, status_code=422)

    results = search_similar_faces(emb, top_k=top_k)
    return {"query_indexed": True, "results": results, "db_size": chroma_face_count()}

# ── LLM: session report ───────────────────────────────────────────────────────

@app.get("/report")
async def session_report():
    """
    Generate a natural-language session analysis report using Claude.
    Aggregates SQLite stats + current frame faces and sends to Claude Sonnet.
    """
    from llm_reporter import generate_session_report, OLLAMA_OK

    stats = get_stats()
    with analyzer._analysis_lock:
        recent_faces = [
            {k: v for k, v in f.items() if k != "embedding"}
            for f in analyzer._cached_analysis
        ]

    report = generate_session_report(stats, recent_faces)
    return {
        "report":      report,
        "ollama_used": OLLAMA_OK,
        "stats":       stats,
    }

# ── LLM: live frame commentary ────────────────────────────────────────────────

@app.get("/commentary")
async def live_commentary():
    """
    One-sentence live description of what's happening in the current frame,
    generated by Claude Haiku for low latency.
    """
    from llm_reporter import generate_live_commentary, OLLAMA_OK

    with analyzer._analysis_lock:
        faces = [
            {k: v for k, v in f.items() if k != "embedding"}
            for f in analyzer._cached_analysis
        ]

    commentary = generate_live_commentary(faces)
    return {"commentary": commentary, "ollama_used": OLLAMA_OK, "face_count": len(faces)}

# ── Face clustering via DBSCAN on ChromaDB embeddings ────────────────────────

@app.get("/cluster-faces")
async def cluster_faces(eps: float = 0.45, min_samples: int = 2):
    """
    DBSCAN on all session face embeddings stored in ChromaDB.
    Groups similar-looking faces — useful for re-identifying unknown visitors.
    """
    from collections import Counter
    from llm_reporter import get_all_embeddings_for_clustering, chroma_face_count
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize

    if chroma_face_count() < 2:
        return {"message": "Not enough faces in vector DB yet.", "clusters": []}

    embs, metas = get_all_embeddings_for_clustering()
    if len(embs) < 2:
        return {"message": "Not enough embeddings.", "clusters": []}

    embs_norm = normalize(embs)
    labels    = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embs_norm)

    clusters: dict[int, list] = {}
    for label, meta in zip(labels, metas):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(meta)

    result = []
    for cid, members in clusters.items():
        names   = [m["name"] for m in members]
        counts  = Counter(names)
        emotions = Counter(m["emotion"] for m in members)
        result.append({
            "cluster_id":     cid,
            "size":           len(members),
            "dominant_name":  counts.most_common(1)[0][0],
            "name_counts":    dict(counts),
            "emotion_counts": dict(emotions),
        })

    return {
        "db_size":       chroma_face_count(),
        "clusters_found": len(result),
        "noise_points":  int((labels == -1).sum()),
        "clusters":      sorted(result, key=lambda x: -x["size"]),
    }

# ── History & stats ───────────────────────────────────────────────────────────

@app.get("/history")
async def history(limit: int = 100, session: str | None = None):
    return {"session_id": SESSION_ID, "records": get_history(limit, session)}


@app.get("/stats")
async def stats():
    return get_stats()

# ── Recording ─────────────────────────────────────────────────────────────────

@app.post("/recording/start")
async def start_recording(width: int = 1280, height: int = 720, fps: int = 20):
    if _rec["active"]:
        return {"status": "already_recording", "path": _rec["path"]}
    os.makedirs("recordings", exist_ok=True)
    path = f"recordings/session_{int(time.time())}.avi"
    _rec.update(
        writer=cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)),
        active=True, path=path, started=time.time(),
    )
    return {"status": "recording", "path": path}


@app.post("/recording/stop")
async def stop_recording():
    if not _rec["active"]:
        return {"status": "not_recording"}
    _rec["writer"].release()
    duration = round(time.time() - _rec["started"], 1)
    path = _rec["path"]
    _rec.update(active=False, writer=None, path=None, started=None)
    return {"status": "saved", "path": path, "duration_sec": duration}


@app.get("/recording/status")
async def recording_status():
    if _rec["active"]:
        return {"active": True, "path": _rec["path"],
                "elapsed_sec": round(time.time() - _rec["started"], 1)}
    return {"active": False}

# ── WebSocket live stream ─────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket):
    """
    Browser-camera mode: client captures frames via getUserMedia and sends them
    as base64 JPEG. Server processes each frame and returns the annotated result.
    Works on any deployment (local, cloud) — no server camera needed.
    """
    await websocket.accept()
    frame_num = 0
    try:
        while True:
            # Receive raw camera frame from browser
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            except asyncio.TimeoutError:
                continue

            raw_b64 = data.get("frame")
            if not raw_b64:
                continue

            raw   = base64.b64decode(raw_b64)
            arr   = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            processed, metadata = analyzer.process(frame)
            frame_num += 1

            if _rec["active"] and _rec["writer"]:
                _rec["writer"].write(processed)

            if frame_num % 30 == 0:
                for face in metadata["faces"]:
                    log_detection({k: v for k, v in face.items() if k != "embedding"})

            _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
            metadata["faces"] = [
                {k: v for k, v in f.items() if k != "embedding"}
                for f in metadata["faces"]
            ]
            await websocket.send_json({
                "frame":    base64.b64encode(buf).decode(),
                "metadata": metadata,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
