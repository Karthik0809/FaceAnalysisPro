import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime

DB_PATH = "session_log.db"
SESSION_ID = str(uuid.uuid4())[:8].upper()


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                session_id  TEXT    NOT NULL,
                face_index  INTEGER,
                name        TEXT,
                emotion     TEXT,
                age         TEXT,
                gender      TEXT,
                pitch       REAL,
                yaw         REAL,
                roll        REAL,
                gaze        TEXT,
                attention   TEXT,
                ear         REAL,
                blinks      INTEGER
            )
        """)


def log_detection(face: dict, session_id: str = SESSION_ID):
    with _conn() as c:
        c.execute("""
            INSERT INTO detections
              (timestamp, session_id, face_index, name, emotion, age, gender,
               pitch, yaw, roll, gaze, attention, ear, blinks)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.utcnow().isoformat(),
            session_id,
            face.get("id"),
            face.get("name"),
            face.get("emotion"),
            str(face.get("age", "")),
            face.get("gender"),
            face.get("pitch"),
            face.get("yaw"),
            face.get("roll"),
            face.get("gaze"),
            face.get("attention"),
            face.get("ear"),
            face.get("blinks"),
        ))


def get_history(limit: int = 100, session_id: str | None = None) -> list[dict]:
    with _conn() as c:
        if session_id:
            rows = c.execute(
                "SELECT * FROM detections WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        sessions = c.execute("SELECT COUNT(DISTINCT session_id) FROM detections").fetchone()[0]
        emotions = c.execute("""
            SELECT emotion, COUNT(*) AS cnt FROM detections
            WHERE emotion IS NOT NULL GROUP BY emotion ORDER BY cnt DESC
        """).fetchall()
        people = c.execute("""
            SELECT name, COUNT(*) AS cnt FROM detections
            WHERE name != 'Unknown' AND name IS NOT NULL
            GROUP BY name ORDER BY cnt DESC LIMIT 10
        """).fetchall()
        attentions = c.execute("""
            SELECT attention, COUNT(*) AS cnt FROM detections
            WHERE attention IS NOT NULL GROUP BY attention ORDER BY cnt DESC
        """).fetchall()
    return {
        "total_detections": total,
        "total_sessions": sessions,
        "current_session": SESSION_ID,
        "emotions": [dict(r) for r in emotions],
        "top_identities": [dict(r) for r in people],
        "attention_breakdown": [dict(r) for r in attentions],
    }
