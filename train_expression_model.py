"""
Train the custom landmark-based expression classifier.

Expected dataset layout (same as FER2013 / RAF-DB folder structure):
  data/
    train/
      angry/   img1.jpg img2.jpg ...
      disgust/ ...
      fear/    ...
      happy/   ...
      neutral/ ...
      sad/     ...
      surprise/...
    val/
      angry/   ...
      ...

Recommended free datasets:
  - FER2013     : https://www.kaggle.com/datasets/msambare/fer2013
  - RAF-DB      : http://www.whdeng.cn/RAF/model1.html
  - AffectNet   : http://mohammadmahoor.com/affectnet/

Usage:
  pip install torch torchvision tqdm scikit-learn mediapipe opencv-python
  python train_expression_model.py --data_dir data --epochs 60

The trained model is saved to models/expression_mlp.pth and is automatically
loaded by the analyzer at startup.
"""

import argparse
import os
import sys

import cv2
import numpy as np

try:
    import mediapipe as mp
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nRun: pip install torch mediapipe tqdm scikit-learn")

from expression_model import INPUT_DIM, MODEL_PATH, extract_landmark_features


# ── Model definition (mirrors expression_model.py) ───────────────────────────

class _MLP(nn.Module):
    def __init__(self, n_cls: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, 128),       nn.GELU(),
            nn.Linear(128, n_cls),
        )
    def forward(self, x): return self.net(x)


# ── Landmark extraction ───────────────────────────────────────────────────────

def extract_dataset(root: str, split: str) -> tuple[np.ndarray, list[str], list[str]]:
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        sys.exit(f"Directory not found: {split_dir}")

    labels = sorted(
        d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
    )
    if not labels:
        sys.exit(f"No class subdirectories found in {split_dir}")

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.4,
    )

    X, y = [], []
    skipped = 0

    for label in labels:
        label_dir = os.path.join(split_dir, label)
        files = [f for f in os.listdir(label_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  [{split}/{label}] {len(files)} images")

        for fn in tqdm(files, desc=f"{split}/{label}", leave=False):
            img = cv2.imread(os.path.join(label_dir, fn))
            if img is None:
                skipped += 1
                continue
            rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            if not result.multi_face_landmarks:
                skipped += 1
                continue
            X.append(extract_landmark_features(result.multi_face_landmarks[0]))
            y.append(label)

    mesh.close()
    print(f"  Extracted {len(X)} samples ({skipped} skipped — no face detected)")
    return np.array(X, dtype=np.float32), y, labels


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_dir: str, epochs: int, lr: float, batch_size: int, device_str: str):
    os.makedirs("models", exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\nExtracting TRAIN landmarks...")
    X_tr, y_tr_raw, labels = extract_dataset(data_dir, "train")
    print("\nExtracting VAL landmarks...")
    X_vl, y_vl_raw, _      = extract_dataset(data_dir, "val")

    le    = LabelEncoder().fit(labels)
    y_tr  = le.transform(y_tr_raw)
    y_vl  = le.transform(y_vl_raw)

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long))
    vl_ds = TensorDataset(torch.tensor(X_vl), torch.tensor(y_vl, dtype=torch.long))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    vl_dl = DataLoader(vl_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model   = _MLP(n_cls=len(labels)).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTraining  classes={labels}  samples={len(X_tr)}  val={len(X_vl)}\n")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        sched.step()

        # ── Validate ──
        model.eval()
        correct = total = 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for Xb, yb in vl_dl:
                Xb = Xb.to(device)
                preds = model(Xb).argmax(dim=1).cpu()
                correct += (preds == yb).sum().item()
                total   += len(yb)
                all_preds.extend(preds.tolist())
                all_true.extend(yb.tolist())

        val_acc = correct / total
        print(f"Epoch {epoch:03d}/{epochs}  loss={tr_loss/len(tr_dl):.4f}  val_acc={val_acc:.3f}", end="")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "labels": list(le.classes_)}, MODEL_PATH)
            print("  ✓ saved", end="")
        print()

    print(f"\nBest val accuracy: {best_acc:.3f}")
    print(f"Model saved to:    {MODEL_PATH}\n")

    # Final classification report
    print(classification_report(all_true, all_preds, target_names=list(le.classes_)))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train landmark-based expression MLP")
    p.add_argument("--data_dir",   default="data",  help="Root dir with train/ and val/")
    p.add_argument("--epochs",     default=60,  type=int)
    p.add_argument("--lr",         default=1e-3, type=float)
    p.add_argument("--batch_size", default=64,  type=int)
    p.add_argument("--device",     default="cuda", help="cuda or cpu")
    args = p.parse_args()
    train(args.data_dir, args.epochs, args.lr, args.batch_size, args.device)
