# Databricks notebook source
# MAGIC %md
# MAGIC # モデルの学習
# MAGIC
# MAGIC アノテーションデータを使用してステアリング・スロットル予測モデルを学習します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定
# MAGIC
# MAGIC %pip install torch torchvision

# COMMAND ----------

# パイプライン実行用パラメータ（手動実行時は下のデフォルト値が使われます）
dbutils.widgets.text("data_path", "")

# 展開済みデータのパス
_data_path_param = dbutils.widgets.get("data_path")
DATA_PATH = _data_path_param if _data_path_param else "/Volumes/workspace/default/annotation_data/annotation_20251201_001802"

# 学習設定
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8

# 使用する画像カラム（cam/image_array, cam0/image_array など）
IMAGE_COLUMN = "cam/image_array"

print(f"データパス: {DATA_PATH}")
print(f"バッチサイズ: {BATCH_SIZE}")
print(f"エポック数: {EPOCHS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの読み込み

# COMMAND ----------

import os
import json
import numpy as np
from PIL import Image

def load_annotations(data_path):
    """カタログファイルからアノテーションを読み込む"""
    annotations = []
    catalog_files = sorted([
        f for f in os.listdir(data_path)
        if f.endswith('.catalog') and not f.endswith('.catalog_manifest')
    ])

    for catalog_file in catalog_files:
        catalog_path = os.path.join(data_path, catalog_file)
        with open(catalog_path, 'r') as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line.strip()))

    return annotations

# widgetのdata_pathが無効な場合は、最新のannotationディレクトリを自動選択
if not os.path.exists(DATA_PATH):
    annotation_dir = "/Volumes/workspace/default/annotation_data"
    subdirs = [d for d in os.listdir(annotation_dir)
               if os.path.isdir(os.path.join(annotation_dir, d)) and d.startswith('annotation_')]
    if subdirs:
        latest_annotation = sorted(subdirs)[-1]
        DATA_PATH = os.path.join(annotation_dir, latest_annotation)
        print(f"Using latest annotation data: {DATA_PATH}")
    else:
        raise FileNotFoundError(f"No annotation data found in {annotation_dir}")

annotations = load_annotations(DATA_PATH)
print(f"アノテーション数: {len(annotations)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PyTorch Dataset の作成

# COMMAND ----------

# MAGIC %pip install torch torchvision
# MAGIC import torch
# MAGIC
# MAGIC from torch.utils.data import Dataset, DataLoader
# MAGIC from torchvision import transforms
# MAGIC
# MAGIC class DonkeyDataset(Dataset):
# MAGIC     """Donkeycar形式のデータセット"""
# MAGIC
# MAGIC     def __init__(self, annotations, images_dir, image_column, transform=None):
# MAGIC         self.annotations = annotations
# MAGIC         self.images_dir = images_dir
# MAGIC         self.image_column = image_column
# MAGIC         self.transform = transform or transforms.Compose([
# MAGIC             transforms.Resize((120, 160)),
# MAGIC             transforms.ToTensor(),
# MAGIC             transforms.Normalize(mean=[0.485, 0.456, 0.406],
# MAGIC                                std=[0.229, 0.224, 0.225])
# MAGIC         ])
# MAGIC
# MAGIC     def __len__(self):
# MAGIC         return len(self.annotations)
# MAGIC
# MAGIC     def __getitem__(self, idx):
# MAGIC         record = self.annotations[idx]
# MAGIC
# MAGIC         # 画像を読み込み
# MAGIC         img_name = record[self.image_column]
# MAGIC         img_path = os.path.join(self.images_dir, img_name)
# MAGIC         image = Image.open(img_path).convert('RGB')
# MAGIC
# MAGIC         if self.transform:
# MAGIC             image = self.transform(image)
# MAGIC
# MAGIC         # ラベル（angle, throttle）
# MAGIC         angle = record['user/angle']
# MAGIC         throttle = record['user/throttle']
# MAGIC         label = torch.tensor([angle, throttle], dtype=torch.float32)
# MAGIC
# MAGIC         return image, label
# MAGIC
# MAGIC # データセットを作成
# MAGIC images_dir = os.path.join(DATA_PATH, "images")
# MAGIC
# MAGIC # データを分割
# MAGIC np.random.seed(42)
# MAGIC indices = np.random.permutation(len(annotations))
# MAGIC train_size = int(len(annotations) * TRAIN_RATIO)
# MAGIC
# MAGIC train_annotations = [annotations[i] for i in indices[:train_size]]
# MAGIC val_annotations = [annotations[i] for i in indices[train_size:]]
# MAGIC
# MAGIC print(f"学習データ数: {len(train_annotations)}")
# MAGIC print(f"検証データ数: {len(val_annotations)}")
# MAGIC
# MAGIC # データセットとDataLoaderを作成
# MAGIC train_dataset = DonkeyDataset(train_annotations, images_dir, IMAGE_COLUMN)
# MAGIC val_dataset = DonkeyDataset(val_annotations, images_dir, IMAGE_COLUMN)
# MAGIC
# MAGIC train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
# MAGIC val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルの定義

# COMMAND ----------

import torch.nn as nn
import torchvision.models as models

class DonkeyModel(nn.Module):
    """ステアリング・スロットル予測モデル"""

    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()

        # バックボーンを読み込み
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 出力層（angle, throttle）
        self.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Tanh()  # 出力を-1〜1に制限
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.fc(features)
        return output

# モデルを作成
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DonkeyModel(backbone='resnet18').to(device)
print(f"デバイス: {device}")
print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 学習

# COMMAND ----------

import mlflow
import mlflow.pytorch
import getpass

# MLflow実験を設定（ユーザー名を自動取得してパスを生成）
user_email = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().userName().get()
)
experiment_path = f"/Users/{user_email}/annotation_training_d2j/autonomous_driving_models"
mlflow.set_experiment(experiment_path)


# 損失関数とオプティマイザ
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 学習ループ
# log_system_metrics=True (MLflow 3): 学習中のGPU/CPU/メモリ使用率を自動記録
# （GPUメトリクスには nvidia-ml-py、CPU/メモリには psutil が必要）
with mlflow.start_run(log_system_metrics=True):
    # パラメータを記録
    mlflow.log_params({
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "backbone": "resnet18",
        "train_samples": len(train_annotations),
        "val_samples": len(val_annotations)
    })

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # 学習
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # メトリクスを記録
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # ベストモデルを保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/tmp/best_model.pt")

    # モデルを記録（MLflow 3: artifact_path は非推奨のため name= を使用）
    # ベスト重みをロードしてから記録
    model.load_state_dict(torch.load("/tmp/best_model.pt"))
    model.eval()

    # signature と input_example を付与（モデルサービング/検証で利用可能）
    from mlflow.models.signature import infer_signature
    sample_images, _ = next(iter(val_loader))
    sample_images = sample_images.to(device)
    with torch.no_grad():
        sample_output = model(sample_images)
    input_example = sample_images[:1].cpu().numpy()
    signature = infer_signature(
        sample_images.cpu().numpy(),
        sample_output.cpu().numpy()
    )

    mlflow.pytorch.log_model(
        model,
        name="model",
        signature=signature,
        input_example=input_example,
    )
    mlflow.log_artifact("/tmp/best_model.pt")

    print(f"\n学習完了! Best Val Loss: {best_val_loss:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 推論テスト

# COMMAND ----------

# ベストモデルを読み込み
model.load_state_dict(torch.load("/tmp/best_model.pt"))
model.eval()

# サンプル画像で推論
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i, (image, label) in enumerate(val_loader):
    if i >= 10:
        break

    image = image[0:1].to(device)
    label = label[0].numpy()

    with torch.no_grad():
        pred = model(image).cpu().numpy()[0]

    # 画像を表示
    img = image[0].cpu().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    axes[i].imshow(img)
    axes[i].set_title(f"GT: ({label[0]:.2f}, {label[1]:.2f})\nPred: ({pred[0]:.2f}, {pred[1]:.2f})")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのエクスポート
# MAGIC
# MAGIC 学習したモデルをVolumesに保存します。

# COMMAND ----------

# モデルを保存（データのVolumesルート直下 models/ = アプリの取り込みと同じ場所）
# 命名はアプリのローカル学習と同じ規則: "{model_type}_{timestamp}.pth"
# （拡張子 .pth・プレフィックスにmodel_typeを付けることで、アプリの自動運転
#  モデル一覧にそのまま認識される）
from datetime import datetime, timezone
_ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
_volumes_root = os.path.dirname(DATA_PATH.rstrip('/'))
model_save_path = os.path.join(_volumes_root, "models", f"donkeycar_{_ts}.pth")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'backbone': 'resnet18',
        'input_size': (120, 160),
        'output_size': 2
    }
}, model_save_path)

print(f"モデルを保存しました: {model_save_path}")
