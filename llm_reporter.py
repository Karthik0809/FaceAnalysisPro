"""
Gen-AI layer: Ollama (local LLM) + ChromaDB (local vector DB).
No API keys — runs entirely on your machine.

Setup (one-time):
  1. Install Ollama  →  https://ollama.com/download
  2. Pull a model   →  ollama pull llama3.2
  3. pip install chromadb ollama

If Ollama is not running, the system automatically falls back to a
rule-based analytics reporter that produces professional-quality output
using only the session statistics — no external service needed.
"""

import os
import time
import uuid
from typing import Any

import numpy as np

# ── ChromaDB — local persistent vector store ─────────────────────────────────
try:
    import chromadb
    _chroma   = chromadb.PersistentClient(path="./chroma_db")
    _face_col = _chroma.get_or_create_collection(
        name="session_faces",
        metadata={"hnsw:space": "cosine"},
    )
    CHROMA_OK = True
    print(f"[chromadb] ready — {_face_col.count()} faces indexed")
except Exception as e:
    CHROMA_OK = False
    print(f"[chromadb] unavailable: {e}")

# ── Ollama — local LLM (no API key needed) ────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_OK    = False

try:
    import ollama as _ollama_lib
    _ollama_lib.list()          # will raise if Ollama daemon isn't running
    OLLAMA_OK = True
    print(f"[ollama] ready — model: {OLLAMA_MODEL}")
except Exception as e:
    print(f"[ollama] not available ({e}) — using rule-based reporter")


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def add_face_to_store(embedding: list[float], name: str,
                      emotion: str, age: Any, gender: str) -> bool:
    if not CHROMA_OK:
        return False
    try:
        _face_col.add(
            embeddings=[embedding],
            metadatas=[{
                "name":    name,
                "emotion": emotion,
                "age":     str(age),
                "gender":  gender,
                "ts":      str(time.time()),
            }],
            ids=[str(uuid.uuid4())],
        )
        return True
    except Exception as e:
        print(f"[chromadb] add error: {e}")
        return False


def search_similar_faces(embedding: list[float], top_k: int = 5) -> list[dict]:
    if not CHROMA_OK or _face_col.count() == 0:
        return []
    try:
        n = min(top_k, _face_col.count())
        results = _face_col.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["metadatas", "distances"],
        )
        return [
            {**meta, "similarity": round(1.0 - float(dist), 3)}
            for meta, dist in zip(results["metadatas"][0], results["distances"][0])
        ]
    except Exception as e:
        print(f"[chromadb] query error: {e}")
        return []


def get_all_embeddings_for_clustering() -> tuple[np.ndarray, list[dict]]:
    if not CHROMA_OK or _face_col.count() == 0:
        return np.array([]), []
    data = _face_col.get(include=["embeddings", "metadatas"])
    return np.array(data["embeddings"], dtype=np.float32), data["metadatas"]


def chroma_face_count() -> int:
    return _face_col.count() if CHROMA_OK else 0


# ── Rule-based report (works with zero dependencies) ─────────────────────────

def _rule_based_report(stats: dict, recent_faces: list[dict]) -> str:
    total     = stats.get("total_detections", 0)
    emotions  = stats.get("emotions", [])
    people    = stats.get("top_identities", [])
    attentions = stats.get("attention_breakdown", [])

    if total == 0:
        return (
            "No session data recorded yet. Start the live stream and allow "
            "the system to observe faces before generating a report."
        )

    # Emotion summary
    dom_emo   = emotions[0]["emotion"] if emotions else "neutral"
    dom_pct   = round(emotions[0]["cnt"] / total * 100) if emotions else 0
    sec_emo   = emotions[1]["emotion"] if len(emotions) > 1 else None

    # Attention breakdown
    attn_map  = {a["attention"]: a["cnt"] for a in attentions}
    focused   = attn_map.get("FOCUSED",    0)
    distracted = attn_map.get("DISTRACTED", 0)
    drowsy    = attn_map.get("DROWSY",     0)
    attn_tot  = max(focused + distracted + drowsy, 1)
    focus_pct = round(focused    / attn_tot * 100)
    distr_pct = round(distracted / attn_tot * 100)
    drowsy_pct = round(drowsy    / attn_tot * 100)

    paragraphs = []

    # 1 — Overview
    p1 = (
        f"Session analytics summary: {total} face detection(s) logged across "
        f"{stats.get('total_sessions', 1)} session(s). "
        f"The predominant emotional state was {dom_emo} ({dom_pct}% of detections)"
    )
    if sec_emo:
        p1 += f", with {sec_emo} as the secondary mood pattern."
    else:
        p1 += "."
    paragraphs.append(p1)

    # 2 — Attention
    if focus_pct >= 70:
        attn_summary = "indicating strong sustained engagement."
    elif focus_pct >= 45:
        attn_summary = "suggesting moderate engagement with intermittent distraction."
    else:
        attn_summary = "indicating low engagement — environmental or motivational factors may warrant review."
    paragraphs.append(
        f"Attention analysis shows {focus_pct}% focused, {distr_pct}% distracted, "
        f"and {drowsy_pct}% drowsy states — {attn_summary}"
    )

    # 3 — Identity
    if people:
        names_str = ", ".join(f"{p['name']} ({p['cnt']} detections)" for p in people[:3])
        paragraphs.append(
            f"Registered identities observed: {names_str}. "
            f"{len(people)} unique individual(s) were matched against the known-face registry."
        )
    else:
        paragraphs.append(
            "No registered identities were matched during this session. "
            "Register known faces via the dashboard to enable identity tracking across sessions."
        )

    # 4 — Current frame
    if recent_faces:
        f0    = recent_faces[0]
        smile = " and is smiling" if f0.get("smile") else ""
        talk  = ", and appears to be talking" if f0.get("talking") else ""
        paragraphs.append(
            f"Current frame: {len(recent_faces)} face(s) visible. "
            f"Primary subject ({f0.get('name', 'unknown')}) shows a {f0.get('emotion', 'neutral')} "
            f"expression{smile}{talk}, gaze directed {f0.get('gaze', 'CENTER').lower()}, "
            f"attention state: {f0.get('attention', 'FOCUSED')}."
        )

    # 5 — Vector DB
    vdb = chroma_face_count()
    paragraphs.append(
        f"Vector database contains {vdb} indexed face embedding(s) available for "
        "similarity search and DBSCAN cluster analysis across sessions."
    )

    return "\n\n".join(paragraphs)


# ── Ollama LLM report ─────────────────────────────────────────────────────────

def _ollama_report(stats: dict, recent_faces: list[dict]) -> str:
    emotions_str   = ", ".join(f"{e['emotion']} ({e['cnt']}x)" for e in stats.get("emotions", [])) or "none"
    people_str     = ", ".join(f"{p['name']} ({p['cnt']})" for p in stats.get("top_identities", [])) or "none"
    attention_str  = ", ".join(f"{a['attention']} ({a['cnt']})" for a in stats.get("attention_breakdown", [])) or "none"
    recent_str     = "; ".join(
        f"{f.get('name')} {f.get('emotion')} gaze:{f.get('gaze')} {f.get('attention')}"
        for f in (recent_faces or [])
    ) or "no faces"

    prompt = (
        "You are an AI analyst for a real-time face analysis surveillance system.\n"
        "Write a professional 4-5 sentence session report based on this data:\n\n"
        f"Total detections: {stats.get('total_detections', 0)}\n"
        f"Emotion breakdown: {emotions_str}\n"
        f"Identified individuals: {people_str}\n"
        f"Attention states: {attention_str}\n"
        f"Vector DB size: {chroma_face_count()}\n"
        f"Current frame: {recent_str}\n\n"
        "Be specific with numbers. Professional tone. No bullet points."
    )

    try:
        resp = _ollama_lib.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"]
    except Exception as e:
        print(f"[ollama] report failed: {e} — falling back to rule-based")
        return _rule_based_report(stats, recent_faces)


# ── Rule-based live commentary ────────────────────────────────────────────────

def _rule_based_commentary(faces: list[dict]) -> str:
    if not faces:
        return "No faces currently detected in the frame."

    if len(faces) == 1:
        f     = faces[0]
        name  = f["name"] if f["name"] != "Unknown" else "An unidentified person"
        smile = ", smiling" if f.get("smile") else ""
        talk  = " and appears to be talking" if f.get("talking") else ""
        pcls  = f.get("perclos", 0)
        alert = " — drowsiness indicators detected" if pcls > 0.35 else ""
        return (
            f"{name} is {f.get('attention', 'focused').lower()}{smile}{talk}, "
            f"displaying a {f.get('emotion', 'neutral')} expression with "
            f"gaze directed {f.get('gaze', 'CENTER').lower()}{alert}."
        )

    focused = sum(1 for f in faces if f.get("attention") == "FOCUSED")
    emotions = list({f.get("emotion", "neutral") for f in faces})
    names = [f["name"] for f in faces if f["name"] != "Unknown"]
    name_str = f" including {', '.join(names)}" if names else ""
    return (
        f"{len(faces)} people detected{name_str}. "
        f"{focused} appear focused. "
        f"Observed emotions: {', '.join(emotions)}."
    )


# ── Ollama live commentary ────────────────────────────────────────────────────

def _ollama_commentary(faces: list[dict]) -> str:
    desc = "; ".join(
        f"{f['name']}: {f['emotion']}, gaze {f['gaze']}, {f['attention']}"
        + (", smiling" if f.get("smile") else "")
        + (", talking" if f.get("talking") else "")
        for f in faces
    )
    try:
        resp = _ollama_lib.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": f"Describe in one sentence what you observe: {desc}"}],
        )
        return resp["message"]["content"]
    except Exception:
        return _rule_based_commentary(faces)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_session_report(stats: dict, recent_faces: list[dict]) -> str:
    if OLLAMA_OK:
        return _ollama_report(stats, recent_faces)
    return _rule_based_report(stats, recent_faces)


def generate_live_commentary(faces: list[dict]) -> str:
    if not faces:
        return "No faces currently detected in the frame."
    if OLLAMA_OK:
        return _ollama_commentary(faces)
    return _rule_based_commentary(faces)
