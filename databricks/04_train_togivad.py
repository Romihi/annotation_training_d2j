# Databricks notebook source
# MAGIC %md
# MAGIC # TogiVAD-Nano（軌道語彙分類）モデルの学習
# MAGIC
# MAGIC 走行データの **将来軌道ラベル**（`togivad/future_traj`）から軌道語彙を
# MAGIC 構築し、単一フレーム画像 → 軌道語彙インデックスの分類として学習します。
# MAGIC （アプリの TogivadTrainingManager のクラウド版・自己完結実装）
# MAGIC
# MAGIC **前提**: アプリのマップビューで「軌道ラベルを計算して保存」を実行し、
# MAGIC catalog に `togivad/future_traj`（各フレーム H×2 の ego 座標系軌道）が
# MAGIC 保存されていること。無い場合はセルがエラーで停止し、手順を案内します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定

%pip install torch torchvision scikit-learn

# COMMAND ----------

# パイプライン実行用パラメータ（手動実行時は下のデフォルト値が使われます）
dbutils.widgets.text("data_path", "")
dbutils.widgets.text("epochs", "20")
dbutils.widgets.text("batch_size", "32")
dbutils.widgets.text("learning_rate", "0.0003")
dbutils.widgets.text("vocab_k", "128")
dbutils.widgets.text("image_column", "cam/image_array")
dbutils.widgets.text("traj_column", "togivad/future_traj")

def _get_param(name, default):
    v = dbutils.widgets.get(name)
    return v if v else default

DATA_PATH = _get_param(
    "data_path",
    "/Volumes/workspace/default/annotation_data/annotation_20251201_001802")
EPOCHS = int(_get_param("epochs", "20"))
BATCH_SIZE = int(_get_param("batch_size", "32"))
LEARNING_RATE = float(_get_param("learning_rate", "0.0003"))
VOCAB_K = int(_get_param("vocab_k", "128"))
IMAGE_COLUMN = _get_param("image_column", "cam/image_array")
TRAJ_COLUMN = _get_param("traj_column", "togivad/future_traj")
TRAIN_RATIO = 0.8
IMG_H, IMG_W = 120, 160

print(f"データパス: {DATA_PATH}")
print(f"エポック: {EPOCHS} / バッチ: {BATCH_SIZE} / lr: {LEARNING_RATE}")
print(f"語彙数K: {VOCAB_K}")
print(f"画像カラム: {IMAGE_COLUMN} / 軌道ラベルカラム: {TRAJ_COLUMN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの読み込み

# COMMAND ----------

import os
import json
import numpy as np

def load_annotations(data_path):
    annotations = []
    catalog_files = sorted([
        f for f in os.listdir(data_path)
        if f.endswith('.catalog') and not f.endswith('.catalog_manifest')
    ])
    for catalog_file in catalog_files:
        with open(os.path.join(data_path, catalog_file), 'r') as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line.strip()))
    return annotations

records = load_annotations(DATA_PATH)
print(f"レコード数: {len(records)}")

# 将来軌道ラベルを持つレコードだけを対象にする
def _parse_traj(v):
    """togivad/future_traj を (H,2) の np.float32 に整形。無効なら None。"""
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 2:
        return arr
    return None

samples = []  # [(image_name, traj(H,2))]
horizon = None
for r in records:
    img_name = r.get(IMAGE_COLUMN)
    traj = _parse_traj(r.get(TRAJ_COLUMN))
    if img_name is None or traj is None:
        continue
    if horizon is None:
        horizon = traj.shape[0]
    if traj.shape[0] != horizon:      # 点数が揃わないものは除外
        continue
    samples.append((img_name, traj))

if not samples:
    raise ValueError(
        f"'{TRAJ_COLUMN}' を持つフレームがありません。\n"
        "アプリのマップビューで『軌道ラベルを計算して保存』を実行して、\n"
        "catalog に将来軌道ラベルを保存してから再度アップロードしてください。")

print(f"軌道ラベルを持つサンプル数: {len(samples)} / horizon(点数): {horizon}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 軌道語彙の構築（k-means）

# COMMAND ----------

from sklearn.cluster import KMeans

trajs = np.stack([t for _, t in samples])            # (N, H, 2)
flat = trajs.reshape(len(trajs), -1)                 # (N, H*2)
k = min(VOCAB_K, len(samples))                       # サンプル数未満に丸める
print(f"k-means: N={len(flat)} K={k}")
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(flat)
vocab = kmeans.cluster_centers_.reshape(k, horizon, 2).astype(np.float32)  # (K,H,2)
labels_all = kmeans.labels_.astype(np.int64)         # 各サンプルの語彙index
print(f"語彙 shape: {vocab.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PyTorch Dataset / Model

# COMMAND ----------

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

images_dir = os.path.join(DATA_PATH, "images")

class TogivadClsDataset(Dataset):
    def __init__(self, items, labels):
        self.items = items      # [(img_name, traj)]
        self.labels = labels    # np.int64[N]
        self.transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name, _ = self.items[idx]
        img = Image.open(os.path.join(images_dir, img_name)).convert('RGB')
        return self.transform(img), int(self.labels[idx])

class TogivadClsModel(nn.Module):
    """単一フレーム画像 → 軌道語彙 K クラス分類"""
    def __init__(self, num_classes, backbone='resnet18', pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))

# train/val split
np.random.seed(42)
idx = np.random.permutation(len(samples))
n_train = int(len(samples) * TRAIN_RATIO)
tr_idx, va_idx = idx[:n_train], idx[n_train:]

train_ds = TogivadClsDataset([samples[i] for i in tr_idx], labels_all[tr_idx])
val_ds = TogivadClsDataset([samples[i] for i in va_idx], labels_all[va_idx])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"学習: {len(train_ds)} / 検証: {len(val_ds)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TogivadClsModel(num_classes=k).to(device)
print(f"デバイス: {device} / パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 学習（MLflow記録）

# COMMAND ----------

import mlflow

user_email = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().userName().get()
)
experiment_path = f"/Users/{user_email}/annotation_training_d2j/togivad_models"
mlflow.set_experiment(experiment_path)

vocab_t = torch.from_numpy(vocab).to(device)          # (K,H,2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

with mlflow.start_run(log_system_metrics=True):
    mlflow.log_params({
        "model_type": "togivad",
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "vocab_k": k,
        "horizon": horizon, "image_column": IMAGE_COLUMN,
        "train_samples": len(train_ds), "val_samples": len(val_ds),
    })

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(len(train_loader), 1)

        model.eval()
        va_loss = top1 = cnt = 0.0
        ade_sum = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                va_loss += criterion(logits, lbls).item()
                pred = logits.argmax(1)
                top1 += (pred == lbls).sum().item()
                # ADE: 予測語彙軌道 vs 正解語彙軌道（ラベルは語彙indexなので近似指標）
                ade_sum += torch.linalg.norm(
                    vocab_t[pred] - vocab_t[lbls], dim=2).mean(1).sum().item()
                cnt += lbls.shape[0]
        va_loss /= max(len(val_loader), 1)
        val_top1 = top1 / max(cnt, 1)
        val_ade = ade_sum / max(cnt, 1)

        mlflow.log_metrics({"train_loss": tr_loss, "val_loss": va_loss,
                            "val_top1": val_top1, "val_ade_m": val_ade}, step=epoch)
        print(f"Epoch {epoch+1}/{EPOCHS} - train_CE={tr_loss:.4f} "
              f"val_CE={va_loss:.4f} top1={val_top1:.1%} ADE={val_ade:.3f}m")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({
                "state_dict": model.state_dict(),
                "vocab": vocab,
                "model_type": "togivad",
                "config": {"backbone": "resnet18", "input_size": (IMG_H, IMG_W),
                           "vocab_k": k, "horizon": horizon,
                           "image_column": IMAGE_COLUMN},
            }, "/tmp/best_togivad.pth")

    mlflow.log_artifact("/tmp/best_togivad.pth")
    print(f"\n学習完了! Best Val CE: {best_val_loss:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのエクスポート（Volumes）

# COMMAND ----------

from datetime import datetime, timezone
ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
# 保存先はデータのVolumesルート直下 models/（アプリの取り込みと同じ場所）
# 命名はアプリのローカルTogiVAD学習と同じ規則: "togivad_{ts}_{n}cam_k{K}.pth"
# （単一画像カラム学習なので 1cam）
_volumes_root = os.path.dirname(DATA_PATH.rstrip('/'))
model_save_path = os.path.join(
    _volumes_root, "models", f"togivad_{ts}_1cam_k{k}.pth")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

import shutil
shutil.copy("/tmp/best_togivad.pth", model_save_path)
print(f"モデルを保存しました: {model_save_path}")
