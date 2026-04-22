import math
import os
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace


def _to_python(obj):
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ── Optional: custom PyTorch expression model ─────────────────────────────────
try:
    from expression_model import ExpressionMLP
    _expr_model = ExpressionMLP()
except Exception:
    _expr_model = None

# ── Optional: ChromaDB face store ─────────────────────────────────────────────
try:
    from llm_reporter import add_face_to_store as _chroma_add
    CHROMA_OK = True
except Exception:
    CHROMA_OK = False
    def _chroma_add(*_, **__): return False

# ── Optional: YOLO-World open-vocabulary object detection ────────────────────
# YOLO-World uses CLIP similarity → confidence scores are lower than regular YOLO;
# use threshold 0.20–0.25 (not 0.5+).
# NOTE: "smartphone" removed — CLIP maps any flat dark rectangle to it, causing
# wallets and watches to score as smartphone.  Use specific phone terms instead.
_WORLD_CLASSES = [
    # Writing instruments
    "pen", "ballpoint pen", "pencil", "marker", "highlighter", "fountain pen",
    # Phones — specific phrases, NOT generic "smartphone"
    "mobile phone", "cell phone",
    # Computer peripherals
    "computer mouse", "wireless mouse", "optical mouse",
    "keyboard", "laptop computer", "tablet computer",
    # Watches — "wristwatch" includes wrist context; CLIP scores high for wrist+dial
    "wristwatch", "analog wristwatch", "digital wristwatch",
    "Casio watch", "sport watch",
    # Audio
    "earphones", "headphones", "earbuds", "AirPods", "wireless earbuds",
    # Other electronics
    "remote control", "TV remote", "smartwatch", "fitness tracker",
    "charger", "power bank", "USB drive", "camera", "calculator",
    # Wallets — descriptive phrases beat plain "wallet" against phone confusion
    "leather wallet", "bifold wallet", "money wallet",
    "purse", "handbag", "clutch bag", "backpack", "bag",
    # Accessories
    "keys", "keychain", "glasses", "sunglasses", "ring", "bracelet",
    # Beauty / personal care
    "lipstick", "lip balm", "lip gloss", "mascara", "makeup compact",
    "perfume bottle", "deodorant stick", "nail polish", "foundation bottle",
    # Stationery / office
    "book", "notebook", "notepad", "paper", "folder", "scissors",
    "ruler", "eraser", "stapler", "tape", "sticky notes", "envelope",
    # Food / drink
    "cup", "mug", "water bottle", "plastic bottle", "bottle",
    "coffee cup", "snack", "apple", "banana", "food",
    # Other
    "hat", "cap", "umbrella", "coin", "credit card",
    "toothbrush", "medicine bottle", "toy",
]

try:
    from ultralytics import YOLOWorld as _YOLO_cls
    _yolo = _YOLO_cls("yolov8x-worldv2.pt")   # open-vocabulary, best accuracy
    _yolo.set_classes(_WORLD_CLASSES)
    YOLO_OK = True
    print(f"[yolo] YOLO-World ready — {len(_WORLD_CLASSES)} custom classes")
except Exception as e:
    YOLO_OK = False
    _yolo   = None
    print(f"[yolo] not available: {e}")

# ── Face mesh landmark indices ────────────────────────────────────────────────
_POSE_IDX = [4, 152, 263, 33, 287, 57]
_MODEL_3D = np.array([
    [   0.0,    0.0,    0.0],
    [   0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)

_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]
_LEFT_IRIS, _RIGHT_IRIS = 468, 473
_L_EYE_H = (362, 263)
_R_EYE_H = (33,  133)

_MOUTH_L, _MOUTH_R     = 61, 291
_UPPER_LIP, _LOWER_LIP = 13, 14
_EYE_L_OUT, _EYE_R_OUT = 33, 263

EAR_THRESH     = 0.20
GAZE_LO        = 0.37
GAZE_HI        = 0.63
PERCLOS_WINDOW = 90
PERCLOS_ALERT  = 0.35
SMILE_THRESH   = 0.44
MAR_THRESH     = 0.15

EMOTION_COLORS = {
    "happy":    (0,   255, 150),
    "sad":      (255, 100,  60),
    "angry":    (0,    60, 255),
    "surprise": (0,   200, 255),
    "fear":     (180,  60, 255),
    "disgust":  (60,  255, 200),
    "neutral":  (160, 160, 160),
}

# ── Body pose skeleton — upper body only (webcam desk view) ──────────────────
# Hips/legs removed: never fully visible from a seated webcam angle
_SKEL = [
    (11, 12, (0,   255, 150)),   # shoulder bar    — neon green
    (11, 13, (255, 140,   0)),   # left upper-arm  — orange
    (13, 15, (255, 210,   0)),   # left forearm
    (12, 14, (255, 140,   0)),   # right upper-arm
    (14, 16, (255, 210,   0)),   # right forearm
]

# YOLO-World uses custom class list — no built-in "person" class to suppress
_YOLO_SKIP: set[int] = set()


class FaceAnalyzer:
    def __init__(self, known_faces_dir: str = "known_faces"):
        self.known_faces_dir = known_faces_dir
        os.makedirs(known_faces_dir, exist_ok=True)

        _mp = mp.solutions
        self._mp_mesh   = _mp.face_mesh
        self._mp_det    = _mp.face_detection
        self._mp_draw   = _mp.drawing_utils
        self._mp_styles = _mp.drawing_styles

        self.face_mesh = self._mp_mesh.FaceMesh(
            max_num_faces=5, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._bg_det = self._mp_det.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        # Body pose — full frame, main thread
        self.pose = _mp.pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Hand tracking — finger joints, main thread
        self.hands = _mp.hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._mp_hands = _mp.hands

        self.known_embeddings: list[np.ndarray] = []
        self.known_names:      list[str]        = []
        self._load_known_faces()

        # Shared state
        self._latest_frame: np.ndarray | None = None
        self._frame_lock    = threading.Lock()
        self._cached_analysis: list[dict] = []
        self._analysis_lock = threading.Lock()
        self._running       = True

        # Per-slot rolling state (up to 5 faces)
        self._ear_below   = [False]                          * 5
        self.blink_count  = [0]                              * 5
        self._ear_history = [deque(maxlen=PERCLOS_WINDOW)    for _ in range(5)]
        self._emo_history = [deque(maxlen=20)                for _ in range(5)]

        # FPS
        self._frame_times: list[float] = []
        self.fps = 0

        # Pose + YOLO state
        self._frame_count   = 0
        self._yolo_cache:   list[tuple] = []        # (x1,y1,x2,y2,label,conf)
        self._posture_str   = "UPRIGHT"
        self._latest_yolo_frame: np.ndarray | None = None
        self._yolo_frame_lock = threading.Lock()

        # Age smoothing (rolling average to reduce DeepFace jitter)
        self._age_history = [deque(maxlen=10) for _ in range(5)]

        self._worker = threading.Thread(target=self._analysis_worker, daemon=True)
        self._worker.start()
        if YOLO_OK:
            self._yolo_worker = threading.Thread(target=self._yolo_analysis_worker, daemon=True)
            self._yolo_worker.start()

    # ── Known-face management ─────────────────────────────────────────────────

    def _load_known_faces(self):
        self.known_embeddings.clear()
        self.known_names.clear()
        for fn in os.listdir(self.known_faces_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            name = os.path.splitext(fn)[0]
            path = os.path.join(self.known_faces_dir, fn)
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                # Build augmented variants: original + H-flip + brightness ±15%
                variants = [
                    img,
                    cv2.flip(img, 1),
                    cv2.convertScaleAbs(img, alpha=1.15, beta=0),
                    cv2.convertScaleAbs(img, alpha=0.85, beta=0),
                ]
                embs = []
                for v in variants:
                    try:
                        r = DeepFace.represent(
                            img_path=v, model_name="Facenet512",
                            enforce_detection=False, detector_backend="retinaface",
                        )
                        if r:
                            embs.append(np.array(r[0]["embedding"]))
                    except Exception:
                        pass
                if embs:
                    avg = np.mean(embs, axis=0)
                    avg /= (np.linalg.norm(avg) + 1e-8)
                    self.known_embeddings.append(avg)
                    self.known_names.append(name)
                    print(f"[recognizer] loaded: {name} ({len(embs)}/4 augmented embeddings averaged)")
            except Exception as e:
                print(f"[recognizer] skip {fn}: {e}")

    def reload_known_faces(self):
        self._load_known_faces()

    # ── Background analysis worker ────────────────────────────────────────────

    def _analysis_worker(self):
        while self._running:
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.04)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            det = self._bg_det.process(rgb)

            boxes = []
            if det.detections:
                for d in det.detections:
                    bb = d.location_data.relative_bounding_box
                    x  = max(0, int(bb.xmin * w))
                    y  = max(0, int(bb.ymin * h))
                    bw = min(int(bb.width  * w), w - x)
                    bh = min(int(bb.height * h), h - y)
                    if bw > 15 and bh > 15:
                        boxes.append((x, y, bw, bh))

            results = self._analyze_faces(frame, boxes) if boxes else []
            with self._analysis_lock:
                self._cached_analysis = results

    def _yolo_analysis_worker(self):
        while self._running:
            with self._yolo_frame_lock:
                frame = self._latest_yolo_frame
                self._latest_yolo_frame = None
            if frame is None:
                time.sleep(0.08)
                continue
            try:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yres = _yolo(rgb, verbose=False, iou=0.45)[0]
                cache = []
                for box in yres.boxes:
                    cls = int(box.cls[0])
                    if cls in _YOLO_SKIP:
                        continue
                    conf = float(box.conf[0])
                    if conf < 0.22:   # YOLO-World CLIP scores are lower than regular YOLO
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cache.append((x1, y1, x2, y2, yres.names[cls], conf))
                self._yolo_cache = cache
            except Exception:
                pass

    def _analyze_faces(self, frame: np.ndarray, boxes: list) -> list:
        results = []
        for x, y, bw, bh in boxes:
            mg   = int(min(bw, bh) * 0.15)
            face = frame[max(0, y-mg):min(frame.shape[0], y+bh+mg),
                         max(0, x-mg):min(frame.shape[1], x+bw+mg)]
            if face.size == 0:
                continue

            data: dict = dict(bbox=(x, y, bw, bh), name="Unknown",
                              emotion="neutral", emotion_scores={},
                              age="?", gender="?", confidence=0.0,
                              embedding=None)

            try:
                actions = ["age", "gender"] if (_expr_model and _expr_model.available) else ["emotion", "age", "gender"]
                a = DeepFace.analyze(
                    img_path=face, actions=actions,
                    enforce_detection=False, detector_backend="skip", silent=True,
                )
                if isinstance(a, list): a = a[0]
                data["age"]    = int(round(float(a.get("age", 0))))
                data["gender"] = a.get("dominant_gender", "?")
                if not (_expr_model and _expr_model.available):
                    data["emotion"]        = a.get("dominant_emotion", "neutral")
                    data["emotion_scores"] = a.get("emotion", {})
            except Exception:
                pass

            try:
                rep = DeepFace.represent(
                    img_path=face, model_name="Facenet512",
                    enforce_detection=False, detector_backend="skip",
                )
                if rep:
                    emb  = np.array(rep[0]["embedding"])
                    data["embedding"] = emb.tolist()
                    norm = np.linalg.norm(emb)

                    if self.known_embeddings:
                        sims = [float(np.dot(emb, k) / (norm * np.linalg.norm(k) + 1e-8))
                                for k in self.known_embeddings]
                        best = int(np.argmax(sims))
                        if sims[best] > 0.68:   # raised from 0.50 → higher precision
                            data["name"]       = self.known_names[best]
                            data["confidence"] = round(sims[best], 2)

                    _chroma_add(
                        data["embedding"], data["name"],
                        data["emotion"], data["age"], data["gender"],
                    )
            except Exception:
                pass

            results.append(data)
        return results

    # ── Head-pose estimation ──────────────────────────────────────────────────

    def _head_pose(self, frame: np.ndarray, lm, w: int, h: int) -> tuple[float, float, float]:
        pts2d = np.array(
            [[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in _POSE_IDX],
            dtype=np.float64,
        )
        focal = w
        cam   = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
        dist  = np.zeros((4, 1))

        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_3D, pts2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2( rmat[2, 1], rmat[2, 2]))
            yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll  = math.degrees(math.atan2( rmat[1, 0], rmat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll  = 0.0

        nose  = tuple(pts2d[0].astype(int))
        axis  = np.float32([[90, 0, 0], [0, -90, 0], [0, 0, -90]])
        ap, _ = cv2.projectPoints(axis, rvec, tvec, cam, dist)
        cv2.arrowedLine(frame, nose, tuple(ap[0].ravel().astype(int)), (0,  80, 255), 2, tipLength=0.25)
        cv2.arrowedLine(frame, nose, tuple(ap[1].ravel().astype(int)), (0, 255,  80), 2, tipLength=0.25)
        cv2.arrowedLine(frame, nose, tuple(ap[2].ravel().astype(int)), (255, 120,  0), 2, tipLength=0.25)

        return pitch, yaw, roll

    # ── Eye metrics: EAR + PERCLOS + gaze ────────────────────────────────────

    @staticmethod
    def _ear(lm, indices: list, w: int, h: int) -> float:
        pts = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in indices])
        v1  = np.linalg.norm(pts[1] - pts[5])
        v2  = np.linalg.norm(pts[2] - pts[4])
        hz  = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * hz + 1e-6)

    def _eye_metrics(self, lm, w: int, h: int, slot: int) -> tuple[str, float, int, float]:
        left_ear  = self._ear(lm, _LEFT_EYE,  w, h)
        right_ear = self._ear(lm, _RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESH:
            self._ear_below[slot] = True
        elif self._ear_below[slot]:
            self.blink_count[slot] += 1
            self._ear_below[slot] = False

        self._ear_history[slot].append(avg_ear)
        closed  = sum(1 for e in self._ear_history[slot] if e < EAR_THRESH)
        perclos = closed / max(len(self._ear_history[slot]), 1)

        gaze = "CENTER"
        if len(lm.landmark) > _RIGHT_IRIS:
            li_x = lm.landmark[_LEFT_IRIS].x
            ri_x = lm.landmark[_RIGHT_IRIS].x
            ll, lr = lm.landmark[_L_EYE_H[0]].x, lm.landmark[_L_EYE_H[1]].x
            rl, rr = lm.landmark[_R_EYE_H[0]].x, lm.landmark[_R_EYE_H[1]].x
            avg_r = ((li_x - ll) / (lr - ll + 1e-6) + (ri_x - rl) / (rr - rl + 1e-6)) / 2
            if   avg_r < GAZE_LO: gaze = "LEFT"
            elif avg_r > GAZE_HI: gaze = "RIGHT"

        return gaze, avg_ear, self.blink_count[slot], round(perclos, 3)

    # ── Expression detection ──────────────────────────────────────────────────

    @staticmethod
    def _expressions(lm, w: int, h: int) -> tuple[bool, bool, float, float]:
        def pt(i): return np.array([lm.landmark[i].x * w, lm.landmark[i].y * h])
        mouth_w = np.linalg.norm(pt(_MOUTH_R)   - pt(_MOUTH_L))
        face_w  = np.linalg.norm(pt(_EYE_R_OUT) - pt(_EYE_L_OUT))
        smile_r = mouth_w / (face_w + 1e-6)
        mouth_h = np.linalg.norm(pt(_LOWER_LIP) - pt(_UPPER_LIP))
        mar     = mouth_h / (mouth_w + 1e-6)
        return bool(smile_r > SMILE_THRESH), bool(mar > MAR_THRESH), round(float(smile_r), 3), round(float(mar), 3)

    # ── Body pose skeleton ────────────────────────────────────────────────────

    def _draw_pose_skeleton(self, frame: np.ndarray, landmarks, w: int, h: int):
        pts: dict[int, tuple[int, int]] = {}
        for a, b, _ in _SKEL:
            for idx in (a, b):
                if idx not in pts:
                    lm = landmarks.landmark[idx]
                    if lm.visibility > 0.65:
                        pts[idx] = (int(lm.x * w), int(lm.y * h))

        max_seg = w * 0.55   # skip connections longer than 55% of frame width

        # Pass 1 — subtle glow (narrow, dim)
        for a, b, color in _SKEL:
            if a in pts and b in pts:
                if math.hypot(pts[a][0]-pts[b][0], pts[a][1]-pts[b][1]) > max_seg:
                    continue
                dim = tuple(max(0, c // 5) for c in color)
                cv2.line(frame, pts[a], pts[b], dim, 6, cv2.LINE_AA)
        # Pass 2 — main line
        for a, b, color in _SKEL:
            if a in pts and b in pts:
                if math.hypot(pts[a][0]-pts[b][0], pts[a][1]-pts[b][1]) > max_seg:
                    continue
                cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        # Pass 3 — joint dots
        for pt in pts.values():
            cv2.circle(frame, pt, 4, (255, 255, 200), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 6, (180, 180, 100),  1, cv2.LINE_AA)

    @staticmethod
    def _posture_metrics(landmarks, w: int, h: int) -> str:
        def vis(idx):
            lm = landmarks.landmark[idx]
            return (lm.x * w, lm.y * h) if lm.visibility > 0.65 else None

        ls, rs = vis(11), vis(12)   # shoulders (always visible from webcam)
        nose   = vis(0)

        if not (ls and rs):
            return "UPRIGHT"

        # Shoulder tilt — uneven shoulders
        tilt = abs(ls[1] - rs[1]) / h

        # Lateral lean — nose left/right of shoulder center
        sc_x = (ls[0] + rs[0]) / 2
        lean = ((nose[0] - sc_x) / w) if nose else 0

        # Slouch — shoulders too high relative to frame (hunched forward)
        sc_y_norm = (ls[1] + rs[1]) / 2 / h  # 0=top, 1=bottom

        if tilt  > 0.07:  return "TILTED"
        if lean  < -0.08: return "LEAN_R"
        if lean  >  0.08: return "LEAN_L"
        if sc_y_norm < 0.25: return "SLOUCHING"
        return "UPRIGHT"

    # ── Match mesh face → cached analysis ────────────────────────────────────

    @staticmethod
    def _match(mesh_bbox: tuple, analyses: list) -> dict:
        mx, my, mw, mh = mesh_bbox
        mc = (mx + mw / 2, my + mh / 2)
        best, best_d = {}, float("inf")
        for a in analyses:
            ax, ay, aw, ah = a["bbox"]
            d = math.hypot(mc[0] - ax - aw/2, mc[1] - ay - ah/2)
            if d < best_d:
                best_d, best = d, a
        return best if best_d < 200 else {}

    # ── Drowsiness alert overlay ──────────────────────────────────────────────

    @staticmethod
    def _draw_alert(frame: np.ndarray, msg: str):
        h, w = frame.shape[:2]
        alpha  = 0.5 + 0.5 * math.sin(time.time() * 6)
        border = int(8 * alpha) + 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 220), border)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (0, h - 52), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, f"  ! {msg}", (12, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2)

    # ── Main process loop ─────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        now = time.time()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
        self.fps = len(self._frame_times)

        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_count += 1

        with self._frame_lock:
            self._latest_frame = frame.copy()

        with self._analysis_lock:
            analyses = list(self._cached_analysis)

        # ── Body pose (every frame — fast ~10 ms) ────────────────────────────
        pose_result = self.pose.process(rgb)
        if pose_result.pose_landmarks:
            self._posture_str = self._posture_metrics(pose_result.pose_landmarks, w, h)
            self._draw_pose_skeleton(frame, pose_result.pose_landmarks, w, h)
        posture_now = self._posture_str

        # ── Hand landmarks — finger joints (every frame) ─────────────────────
        hand_results = self.hands.process(rgb)
        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame, hand_lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=3),
                    self._mp_draw.DrawingSpec(color=(0, 180, 150), thickness=2),
                )

        # ── Face mesh (every frame) ───────────────────────────────────────────
        mesh = self.face_mesh.process(rgb)
        faces_json: list[dict] = []
        any_drowsy = False

        if mesh.multi_face_landmarks:
            for i, lm in enumerate(mesh.multi_face_landmarks):
                self._mp_draw.draw_landmarks(
                    frame, lm, self._mp_mesh.FACEMESH_TESSELATION,
                    None, self._mp_styles.get_default_face_mesh_tesselation_style())
                self._mp_draw.draw_landmarks(
                    frame, lm, self._mp_mesh.FACEMESH_CONTOURS,
                    None, self._mp_styles.get_default_face_mesh_contours_style())
                self._mp_draw.draw_landmarks(
                    frame, lm, self._mp_mesh.FACEMESH_IRISES,
                    None, self._mp_styles.get_default_face_mesh_iris_connections_style())

                xs  = [l.x * w for l in lm.landmark]
                ys  = [l.y * h for l in lm.landmark]
                bx  = max(0, int(min(xs)));  by  = max(0, int(min(ys)))
                bw_ = min(int(max(xs) - min(xs)), w - bx)
                bh_ = min(int(max(ys) - min(ys)), h - by)
                slot = min(i, 4)

                pitch, yaw, roll = self._head_pose(frame, lm, w, h)
                gaze, ear, blinks, perclos = self._eye_metrics(lm, w, h, slot)
                smile, talking, smile_r, mar = self._expressions(lm, w, h)

                an = self._match((bx, by, bw_, bh_), analyses)
                if _expr_model and _expr_model.available:
                    emo, em_sc = _expr_model.predict(lm)
                else:
                    emo   = an.get("emotion", "...")
                    em_sc = an.get("emotion_scores", {})

                name = an.get("name",    "Unknown")
                raw_age = an.get("age", None)
                if raw_age not in (None, "?"):
                    self._age_history[slot].append(int(raw_age))
                age  = int(round(sum(self._age_history[slot]) / len(self._age_history[slot]))) if self._age_history[slot] else "?"
                gen  = an.get("gender",  "?")
                conf = an.get("confidence", 0.0)
                emb  = an.get("embedding", None)

                if emo not in ("...", ""):
                    self._emo_history[slot].append(emo)
                emo_hist = list(self._emo_history[slot])

                # Attention (now also incorporates posture slouching)
                if perclos > PERCLOS_ALERT or ear < EAR_THRESH:
                    attention = "DROWSY";     any_drowsy = True
                elif gaze != "CENTER":
                    attention = "DISTRACTED"
                else:
                    attention = "FOCUSED"

                # Corner-bracket bounding box
                color = (0, 255, 150)
                cl    = 22
                for px, py, dx, dy in [(bx,by,1,1),(bx+bw_,by,-1,1),(bx,by+bh_,1,-1),(bx+bw_,by+bh_,-1,-1)]:
                    cv2.line(frame, (px, py), (px + dx * cl, py), color, 2)
                    cv2.line(frame, (px, py), (px, py + dy * cl), color, 2)

                scan_y = by + int((time.time() * 80) % max(bh_, 1))
                cv2.line(frame, (bx, scan_y), (bx + bw_, scan_y), (0, 255, 100), 1)

                # PERCLOS bar
                bar_x  = bx + bw_ + 6
                filled = int(bh_ * perclos)
                pcolor = (0, 0, 220) if perclos > PERCLOS_ALERT else (0, 200, 80)
                cv2.rectangle(frame, (bar_x, by), (bar_x + 5, by + bh_), (30, 30, 30), -1)
                cv2.rectangle(frame, (bar_x, by + bh_ - filled), (bar_x + 5, by + bh_), pcolor, -1)

                # Labels
                font   = cv2.FONT_HERSHEY_SIMPLEX
                ly     = by - 14 if by > 60 else by + bh_ + 24
                attn_c = {"FOCUSED":(0,255,150),"DISTRACTED":(0,180,255),"DROWSY":(0,60,255)}
                gaze_c = {"CENTER":(0,255,150),"LEFT":(0,180,255),"RIGHT":(0,180,255)}
                post_c = {"UPRIGHT":(0,255,150),"SLOUCHING":(0,60,255),
                          "TILTED":(0,180,255),"LEAN_L":(0,200,255),"LEAN_R":(0,200,255)}
                src_tag = "[ML]" if (_expr_model and _expr_model.available) else "[DF]"

                cv2.putText(frame, f"#{i+1} {name}",  (bx, ly),      font, 0.60, color, 2)
                cv2.putText(frame, f"{emo} {src_tag} | {age}yr | {gen}", (bx, ly+18), font, 0.40, (160,255,210), 1)
                cv2.putText(frame, f"P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f}", (bx, ly+33), font, 0.38, (100,200,255), 1)
                cv2.putText(frame, f"GAZE:{gaze}", (bx,     ly+47), font, 0.38, gaze_c.get(gaze, color), 1)
                cv2.putText(frame, attention,      (bx+100, ly+47), font, 0.38, attn_c.get(attention, color), 1)
                cv2.putText(frame, f"PCLS:{perclos:.0%}", (bx+190, ly+47), font, 0.38, pcolor, 1)
                cv2.putText(frame, "SMILE" if smile   else "smile", (bx,    ly+61), font, 0.36, (0,255,150) if smile   else (60,60,60), 1)
                cv2.putText(frame, "TALK"  if talking else "talk",  (bx+68, ly+61), font, 0.36, (0,200,255) if talking else (60,60,60), 1)
                cv2.putText(frame, posture_now, (bx+140, ly+61), font, 0.36, post_c.get(posture_now, color), 1)

                faces_json.append({
                    "id":             i + 1,
                    "name":           name,
                    "emotion":        emo,
                    "emotion_scores": {k: float(v) for k, v in em_sc.items()},
                    "emotion_history": emo_hist,
                    "emotion_source": "custom_mlp" if (_expr_model and _expr_model.available) else "deepface",
                    "age":            int(age) if str(age).lstrip('-').isdigit() else age,
                    "gender":         gen,
                    "confidence":     float(conf),
                    "embedding":      emb,
                    "pitch":          round(float(pitch), 1),
                    "yaw":            round(float(yaw),   1),
                    "roll":           round(float(roll),  1),
                    "gaze":           gaze,
                    "attention":      attention,
                    "ear":            round(float(ear),     3),
                    "perclos":        round(float(perclos), 3),
                    "blinks":         int(blinks),
                    "smile":          bool(smile),
                    "talking":        bool(talking),
                    "smile_ratio":    float(smile_r),
                    "mar":            float(mar),
                    "posture":        posture_now,
                })

        if any_drowsy:
            self._draw_alert(frame, "DROWSINESS DETECTED — PERCLOS THRESHOLD EXCEEDED")

        # ── YOLO object detection (background thread, feed every 8 frames) ──
        if YOLO_OK and self._frame_count % 5 == 0:
            with self._yolo_frame_lock:
                self._latest_yolo_frame = frame.copy()

        yolo_objects: list[dict] = []
        for x1, y1, x2, y2, label, conf in self._yolo_cache:
            # Outer glow box
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (60, 100, 0), 2)
            # Main box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
            # Label pill background
            txt = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 180, 200), -1)
            cv2.putText(frame, txt, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            yolo_objects.append({"label": label, "confidence": round(conf, 2),
                                 "bbox": [x1, y1, x2, y2]})

        # ── Posture overlay badge (top-right corner) ──────────────────────────
        if pose_result.pose_landmarks:
            post_c = {"UPRIGHT":(0,255,150),"SLOUCHING":(0,60,255),
                      "TILTED":(0,180,255),"LEAN_L":(0,200,255),"LEAN_R":(0,200,255)}
            pc = post_c.get(posture_now, (160, 160, 160))
            cv2.rectangle(frame, (w - 150, 50), (w - 6, 78), (0, 18, 10), -1)
            cv2.rectangle(frame, (w - 150, 50), (w - 6, 78), pc, 1)
            cv2.putText(frame, f"POSTURE: {posture_now}", (w - 147, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, pc, 1)

        # ── HUD bar ───────────────────────────────────────────────────────────
        src = "CUSTOM MLP + DeepFace" if (_expr_model and _expr_model.available) else "DeepFace"
        chroma_str = f" | VDB:{__import__('llm_reporter').chroma_face_count()}" if CHROMA_OK else ""
        yolo_str   = f" | OBJ:{len(yolo_objects)}" if YOLO_OK else ""

        cv2.rectangle(frame, (0, 0), (w, 42), (0, 18, 10), -1)
        cv2.putText(frame, f"FACE ANALYSIS [{src}]", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 150), 2)
        info = f"FPS {self.fps}  FACES {len(faces_json)}{chroma_str}{yolo_str}"
        (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(frame, info, (w - tw - 12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 255, 200), 1)

        return frame, _to_python({
            "fps":        self.fps,
            "face_count": len(faces_json),
            "faces":      faces_json,
            "posture":    posture_now,
            "objects":    yolo_objects,
        })

    def stop(self):
        self._running = False
