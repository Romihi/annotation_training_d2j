# Databricks notebook source
# MAGIC %md
# MAGIC # モデルの学習
# MAGIC
# MAGIC アノテーションデータを使用してステアリング・スロットル予測モデルを学習します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定

%pip install torch torchvision

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

annotations = load_annotations(DATA_PATH)
print(f"アノテーション数: {len(annotations)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PyTorch Dataset の作成

# COMMAND ----------

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DonkeyDataset(Dataset):
    """Donkeycar形式のデータセット"""

    def __init__(self, annotations, images_dir, image_column, transform=None):
        self.annotations = annotations
        self.images_dir = images_dir
        self.image_column = image_column
        self.transform = transform or transforms.Compose([
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        record = self.annotations[idx]

        # 画像を読み込み
        img_name = record[self.image_column]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # ラベル（angle, throttle）
        angle = record['user/angle']
        throttle = record['user/throttle']
        label = torch.tensor([angle, throttle], dtype=torch.float32)

        return image, label

# データセットを作成
images_dir = os.path.join(DATA_PATH, "images")

# データを分割
np.random.seed(42)
indices = np.random.permutation(len(annotations))
train_size = int(len(annotations) * TRAIN_RATIO)

train_annotations = [annotations[i] for i in indices[:train_size]]
val_annotations = [annotations[i] for i in indices[train_size:]]

print(f"学習データ数: {len(train_annotations)}")
print(f"検証データ数: {len(val_annotations)}")

# データセットとDataLoaderを作成
train_dataset = DonkeyDataset(train_annotations, images_dir, IMAGE_COLUMN)
val_dataset = DonkeyDataset(val_annotations, images_dir, IMAGE_COLUMN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
with mlflow.start_run():
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

    # モデルを記録
    mlflow.pytorch.log_model(model, "model")
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

# モデルを保存
model_save_path = "/Volumes/workspace/default/annotation_data/models/donkey_model.pt"
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
