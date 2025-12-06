# Databricks notebook source
# MAGIC %md
# MAGIC # アノテーションデータの読み込み
# MAGIC
# MAGIC 展開されたアノテーションデータをPythonで読み込み、確認します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定

# COMMAND ----------

# 展開済みデータのパス
DATA_PATH = "/Volumes/workspace/default/annotation_data/annotation_20251201_001802"

print(f"データパス: {DATA_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## カタログファイルからアノテーションを読み込み

# COMMAND ----------

import os
import json

def load_annotations(data_path):
    """カタログファイルからアノテーションを読み込む"""
    annotations = []

    # カタログファイルを取得（catalog_0.catalog, catalog_1.catalog, ...）
    catalog_files = sorted([
        f for f in os.listdir(data_path)
        if f.endswith('.catalog') and not f.endswith('.catalog_manifest')
    ])

    print(f"カタログファイル数: {len(catalog_files)}")

    for catalog_file in catalog_files:
        catalog_path = os.path.join(data_path, catalog_file)
        with open(catalog_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    annotations.append(record)

    return annotations

# アノテーションを読み込み
annotations = load_annotations(DATA_PATH)
print(f"アノテーション数: {len(annotations)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## アノテーションデータの構造確認

# COMMAND ----------

# 最初のレコードを確認
sample = annotations[0]
print("サンプルレコード:")
print(json.dumps(sample, indent=2, ensure_ascii=False))

# COMMAND ----------

# カラム一覧
print("カラム一覧:")
for key in sample.keys():
    print(f"  {key}: {type(sample[key]).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas DataFrameに変換

# COMMAND ----------

import pandas as pd

# DataFrameに変換
df = pd.DataFrame(annotations)

# 基本統計
print(f"レコード数: {len(df)}")
print(f"カラム数: {len(df.columns)}")
print("\nカラム一覧:")
print(df.columns.tolist())

# COMMAND ----------

# 数値カラムの統計
numeric_cols = ['user/angle', 'user/throttle']
if 'pilot/angle' in df.columns:
    numeric_cols.extend(['pilot/angle', 'pilot/throttle'])

print("数値データの統計:")
df[numeric_cols].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark DataFrameに変換

# COMMAND ----------

# Spark DataFrameに変換
spark_df = spark.createDataFrame(df)
spark_df.printSchema()

# COMMAND ----------

# 表示
display(spark_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 画像ファイルの確認

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

images_dir = os.path.join(DATA_PATH, "images")

# 最初のアノテーションの画像を表示
sample = annotations[0]

# 画像カラムを探す
image_cols = [col for col in sample.keys() if 'image_array' in col]
print(f"画像カラム: {image_cols}")

# 画像を表示
fig, axes = plt.subplots(1, len(image_cols), figsize=(5*len(image_cols), 5))
if len(image_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, image_cols):
    img_name = sample[col]
    img_path = os.path.join(images_dir, img_name)

    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f"{col}\n{img_name}")
        ax.axis('off')
    else:
        ax.set_title(f"{col}\nNot found")
        ax.axis('off')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## アノテーション分布の可視化

# COMMAND ----------

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Angle分布
axes[0].hist(df['user/angle'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Angle')
axes[0].set_ylabel('Count')
axes[0].set_title('Angle Distribution')
axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)

# Throttle分布
axes[1].hist(df['user/throttle'], bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Throttle')
axes[1].set_ylabel('Count')
axes[1].set_title('Throttle Distribution')
axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 次のステップ
# MAGIC
# MAGIC - `03_train_model.py` - このデータを使ってモデルを学習
