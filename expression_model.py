"""
Custom PyTorch MLP trained on MediaPipe 468-point facial landmarks.
468 landmarks × 3 coords (x,y,z) = 1404-dimensional feature vector.

Used by the analyzer for fast, lightweight emotion classification
that runs in the main thread every frame (unlike DeepFace which is slow).
"""

import os

import numpy as np

MODEL_PATH = "models/expression_mlp.pth"
EMOTIONS   = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
INPUT_DIM  = 1404  # 468 × 3


def extract_landmark_features(face_landmarks) -> np.ndarray:
    """
    Convert MediaPipe face landmarks to a scale- and position-invariant
    1404-dim feature vector suitable for the MLP.
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[:468]],
        dtype=np.float32,
    )
    # Normalize x,y to [0,1] relative to the face bounding box
    min_xy = coords[:, :2].min(axis=0)
    max_xy = coords[:, :2].max(axis=0)
    scale  = (max_xy - min_xy).max() + 1e-6
    coords[:, 0] = (coords[:, 0] - min_xy[0]) / scale
    coords[:, 1] = (coords[:, 1] - min_xy[1]) / scale
    # z is already relative in MediaPipe output
    return coords.flatten()


class ExpressionMLP:
    """Lazy-loads torch only when the model file exists."""

    def __init__(self, model_path: str = MODEL_PATH):
        self._model  = None
        self._labels = EMOTIONS
        self.available = False

        if not os.path.exists(model_path):
            print(f"[expression_model] no model at {model_path} — run train_expression_model.py first")
            return

        try:
            import torch
            import torch.nn as nn

            class _MLP(nn.Module):
                def __init__(self, n_in, n_cls):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.30),
                        nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.20),
                        nn.Linear(256, 128),  nn.GELU(),
                        nn.Linear(128, n_cls),
                    )
                def forward(self, x): return self.net(x)

            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            self._labels = ckpt.get("labels", EMOTIONS)
            net = _MLP(INPUT_DIM, len(self._labels))
            net.load_state_dict(ckpt["model"])
            net.eval()
            self._model  = net
            self._torch  = torch
            self.available = True
            print(f"[expression_model] loaded — classes: {self._labels}")

        except Exception as e:
            print(f"[expression_model] load failed: {e}")

    def predict(self, face_landmarks) -> tuple[str, dict[str, float]]:
        """Returns (dominant_emotion, scores_dict). Call only if self.available."""
        feats  = extract_landmark_features(face_landmarks)
        x      = self._torch.tensor(feats, dtype=self._torch.float32).unsqueeze(0)
        with self._torch.no_grad():
            probs = self._torch.softmax(self._model(x), dim=1).squeeze().numpy()
        scores   = {lbl: float(p) for lbl, p in zip(self._labels, probs)}
        dominant = max(scores, key=scores.get)
        return dominant, scores
