# Databricks notebook source
# MAGIC %md
# MAGIC # アノテーションデータの展開
# MAGIC
# MAGIC アノテーションツールから転送されたZIPファイルを展開します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定

# COMMAND ----------

# パイプライン実行用パラメータ（手動実行時は下のデフォルト値が使われます）
dbutils.widgets.text("zip_path", "")

# ZIPファイルのパス（転送時に表示されたパスを指定）
_zip_path_param = dbutils.widgets.get("zip_path")
ZIP_PATH = _zip_path_param if _zip_path_param else "/Volumes/workspace/default/annotation_data/annotation_20251201_001802.zip"

# 展開先のパス（ZIPファイル名から.zipを除いたパス）
EXTRACT_PATH = ZIP_PATH.replace(".zip", "")

print(f"ZIPファイル: {ZIP_PATH}")
print(f"展開先: {EXTRACT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ZIPファイルの展開

# COMMAND ----------

import zipfile
import os

# 展開
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    zf.extractall(EXTRACT_PATH)

print("展開完了!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 展開結果の確認

# COMMAND ----------

# ファイル一覧
print("展開されたファイル:")
for f in os.listdir(EXTRACT_PATH):
    full_path = os.path.join(EXTRACT_PATH, f)
    if os.path.isdir(full_path):
        file_count = len(os.listdir(full_path))
        print(f"  {f}/ ({file_count} files)")
    else:
        size_kb = os.path.getsize(full_path) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")

# COMMAND ----------

# 画像数の確認
images_dir = os.path.join(EXTRACT_PATH, "images")
if os.path.exists(images_dir):
    image_files = os.listdir(images_dir)
    print(f"画像数: {len(image_files)}")
    print(f"\n最初の10ファイル:")
    for f in sorted(image_files)[:10]:
        print(f"  {f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## マニフェストの確認

# COMMAND ----------

import json

manifest_path = os.path.join(EXTRACT_PATH, "manifest.json")
with open(manifest_path, 'r') as f:
    lines = f.readlines()

print("manifest.json の内容:")
print("-" * 50)

# 各行をパースして表示
labels = ["カラム名", "データ型", "追加設定", "セッション情報", "カタログ情報"]
for i, line in enumerate(lines):
    data = json.loads(line.strip())
    print(f"\n[{labels[i] if i < len(labels) else i}]")
    print(json.dumps(data, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 次のステップ
# MAGIC
# MAGIC 展開が完了したら、以下のノートブックでデータを活用できます：
# MAGIC
# MAGIC - `02_load_annotations.py` - アノテーションデータの読み込み
# MAGIC - `03_train_model.py` - モデルの学習

# COMMAND ----------

# パイプライン実行時に展開先パスを次のノートブックに渡す
dbutils.notebook.exit(EXTRACT_PATH)
