# Databricks連携ガイド

このドキュメントでは、アノテーションツールとDatabricksワークスペースの連携方法について説明します。

## 概要

Databricks連携を有効にすると、学習結果をローカルとDatabricksの両方に記録できます（ローカル併用記録）。

```
学習実行
    │
    ├─→ ローカル (mlruns/) ← 常に記録
    │
    └─→ Databricks MLflow  ← DATABRICKS_ENABLED=true の場合
```

## セットアップ

### 1. 必要なパッケージ

```bash
pip install databricks-sdk mlflow
```

### 2. Databricksパーソナルアクセストークン（PAT）の取得

本ツールでは **Personal Access Token (PAT)** を使用します。Fine-grained tokenなど他のタイプのトークンは使用できません。

1. Databricksワークスペースにログイン
2. 右上のユーザーアイコン → **Settings**
3. **Developer** → **Access tokens**
4. **Generate new token** をクリック
5. **Comment** に用途を入力（例: `annotation_tool`）
6. **Lifetime** にトークンの有効期限を設定（空欄で無期限）
7. **Generate** をクリック
8. 表示されたトークン（`dapi` で始まる文字列）をコピー（**一度しか表示されません**）

> **重要**: トークンには十分な権限（スコープ）が必要です。ワークスペース管理者がトークンの権限を制限している場合、以下のスコープを付与してください：
>
> | スコープ | 用途 |
> |---|---|
> | **workspace** | ワークスペース内のディレクトリ・実験の作成・読み取り |
> | **mlflow** | MLflow実験・Runの作成・記録 |
> | **files** | アーティファクト（モデルファイル）のアップロード・同期 |
>
> 権限不足のエラーが出る場合は、ワークスペース管理者に相談してください。

### 3. 環境変数の設定

セキュリティのため、全ての認証情報は環境変数で設定します。

#### Windows (PowerShell)

```powershell
$env:DATABRICKS_ENABLED = "true"
$env:DATABRICKS_HOST = "https://your-workspace.cloud.databricks.com"
$env:DATABRICKS_TOKEN = "dapi..."
$env:DATABRICKS_EXPERIMENT_PREFIX = "/Users/your-email@example.com/annotation_training_d2j"
```

#### Windows (コマンドプロンプト)

```cmd
set DATABRICKS_ENABLED=true
set DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
set DATABRICKS_TOKEN=dapi...
set DATABRICKS_EXPERIMENT_PREFIX=/Users/your-email@example.com/annotation_training_d2j
```

#### Linux/Mac

```bash
export DATABRICKS_ENABLED="true"
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_EXPERIMENT_PREFIX="/Users/your-email@example.com/annotation_training_d2j"
```

### 4. 永続的な設定

#### Windows

システムの環境変数に追加するか、PowerShellプロファイルに設定を追加：

```powershell
# $PROFILE を編集
notepad $PROFILE

# 以下を追加
$env:DATABRICKS_ENABLED = "true"
$env:DATABRICKS_HOST = "https://your-workspace.cloud.databricks.com"
$env:DATABRICKS_TOKEN = "dapi..."
```

#### Linux/Mac

`~/.bashrc` または `~/.zshrc` に追加：

```bash
export DATABRICKS_ENABLED="true"
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
```

## 環境変数一覧

| 環境変数 | 必須 | 説明 | 例 |
|----------|------|------|-----|
| `DATABRICKS_ENABLED` | Yes | 連携の有効/無効 | `true` / `false` |
| `DATABRICKS_HOST` | Yes | ワークスペースURL | `https://dbc-xxx.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | Yes | アクセストークン | `dapi...` |
| `DATABRICKS_EXPERIMENT_PREFIX` | No | 実験パス | `/Users/email/annotation_training_d2j` |
| `DATABRICKS_CATALOG` | No | Unity Catalogカタログ名 | `main` |
| `DATABRICKS_SCHEMA` | No | Unity Catalogスキーマ名 | `default` |

## 使い方

### GUIでの操作

1. 環境変数を設定してからアプリを起動
2. 左パネルの「モデル管理」セクションで「Databricks連携」チェックボックスをON
3. 状態ラベルで接続状態を確認
   - `✓ Databricks+ローカル併用` - 正常に接続
   - `✗ Databricks: 未接続` - 接続失敗
   - `ローカルMLflow使用中` - Databricks無効

### 設定の確認

「設定」ボタンをクリックすると：
- 現在の環境変数の状態を確認
- 設定テンプレートをクリップボードにコピー
- READMEを開く

## 記録される実験

以下の実験がDatabricksワークスペースに作成されます：

| 実験名 | 説明 |
|--------|------|
| `autonomous_driving_models` | 自動運転モデル（ステアリング・スロットル予測） |
| `position_estimation_models` | 位置推定モデル（分類） |
| `waypoint_regression_models` | ウェイポイント回帰モデル |
| `yolo_detection_models` | YOLO物体検出モデル |
| `yolo_segmentation_models` | YOLOセグメンテーションモデル |

## 記録される情報

各学習実行で以下の情報が記録されます：

### パラメータ
- モデルタイプ、アーキテクチャ
- エポック数、バッチサイズ、学習率
- オーグメンテーション設定
- データセット情報

### メトリクス
- 学習/検証損失
- 精度（分類タスクの場合）
- mAP（物体検出の場合）

### アーティファクト
- 学習済みモデルファイル（.pt）
- 設定ファイル

## トラブルシューティング

### 接続できない場合

1. **環境変数の確認**
   ```powershell
   # PowerShellで確認
   echo $env:DATABRICKS_ENABLED
   echo $env:DATABRICKS_HOST
   echo $env:DATABRICKS_TOKEN
   ```

2. **トークンの確認**
   - トークンが **Personal Access Token (PAT)** であることを確認（`dapi` で始まる）
   - トークンが有効か確認（期限切れの場合は再生成）
   - トークンに十分な権限（workspace, mlflow スコープ）があるか確認

3. **ホストURLの確認**
   - `https://` で始まっているか
   - 末尾にスラッシュがないか

### ローカルのみに記録される場合

Databricksへの接続に失敗した場合、自動的にローカルのみに記録されます。
コンソールにエラーメッセージが表示されるので確認してください。

### エラーメッセージ

| メッセージ | 対処法 |
|-----------|--------|
| `Invalid access token` | トークンを再生成 |
| `does not have required scopes: workspace` | トークンの権限不足。PATを再生成するか、管理者にworkspaceスコープの付与を依頼 |
| `Reading Databricks credential configuration failed` | 認証情報が正しく設定されていない。環境変数を確認 |
| `環境変数 DATABRICKS_HOST が設定されていません` | 環境変数を設定 |
| `環境変数 DATABRICKS_TOKEN が設定されていません` | 環境変数を設定 |

## Databricksでの確認方法

1. Databricksワークスペースにログイン
2. 左メニューから **Experiments** を選択
3. 実験パス（例: `/Users/your-email/experiments/autonomous_driving_models`）を開く
4. 各Runの詳細を確認

または、アプリの「MLflow/Databricksを開く」ボタンをクリックすると、
自動的にブラウザでDatabricksのMLflow UIが開きます。

## ファイル構成

```
annotation_training_d2j/
├── config_databricks.py      # Databricks設定（環境変数から読み込み）
├── managers/
│   └── mlflow_manager.py     # MLflow管理（Databricks対応）
├── mlruns/                   # ローカルMLflow記録
└── README_DATABRICKS.md      # このファイル
```

## セキュリティに関する注意

- **認証情報をコードにハードコードしないでください**
- **config_databricks.py にトークンを直接書かないでください**
- 環境変数またはシークレット管理サービスを使用してください
- トークンは定期的にローテーションすることを推奨します
- `.env` ファイルを使用する場合は `.gitignore` に追加してください

## 注意事項

- Databricksへの記録が失敗しても、ローカルには必ず記録されます
- 大きなモデルファイルのアップロードには時間がかかる場合があります
- ネットワーク接続が不安定な環境では、ローカルのみモードを推奨します
- 環境変数を変更した場合は、アプリを再起動してください

---

# アノテーションデータの転送

## 概要

アノテーションツールで作成したデータをDatabricks Unity Catalog Volumesに転送できます。

```
アノテーションツール
    │
    ├─→ エクスポート（Donkeycar形式）
    │
    ├─→ ZIP圧縮
    │
    └─→ Databricks Volumes にアップロード
            │
            └─→ /Volumes/{catalog}/{schema}/{volume}/annotation_YYYYMMDD_HHMMSS.zip
```

## 転送先の設定

### 環境変数

```powershell
# PowerShell
$env:DATABRICKS_VOLUMES_PATH = "/Volumes/workspace/default/annotation_data"
```

```cmd
# コマンドプロンプト
set DATABRICKS_VOLUMES_PATH=/Volumes/workspace/default/annotation_data
```

### Databricksでの準備

転送前にDatabricksでVolumesを作成しておく必要があります：

1. Databricksワークスペースにログイン
2. **カタログ** → **workspace**（または使用するカタログ）
3. **default**（または使用するスキーマ）を選択
4. **Create** → **Volume** をクリック
5. Volume名を入力（例: `annotation_data`）
6. **Create** をクリック

## 転送方法

1. アノテーションツールでアノテーションを作成
2. 左パネルの「Databricks」セクションで「転送」ボタンをクリック
3. ZIPファイル名を入力（デフォルト: `annotation_YYYYMMDD_HHMMSS`）
4. 確認ダイアログで「はい」をクリック
5. 転送完了を待つ

### 転送される内容

```
annotation_YYYYMMDD_HHMMSS.zip
├── catalog_0.catalog           # アノテーションデータ（JSON Lines）
├── catalog_0.catalog_manifest  # カタログマニフェスト
├── manifest.json               # 全体マニフェスト
└── images/                     # 画像ファイル
    ├── 0_cam_image_array_.jpg
    ├── 0_cam0_image_array_.jpg
    └── ...
```

## Databricksでのデータ活用

### サンプルノートブック

`databricks/` フォルダに以下のサンプルノートブックがあります：

| ファイル | 説明 |
|----------|------|
| `01_extract_annotations.py` | ZIPファイルの展開 |
| `02_load_annotations.py` | アノテーションデータの読み込みと可視化 |
| `03_train_model.py` | PyTorchでのモデル学習 |

### 使い方

1. サンプルノートブックをDatabricksにインポート
2. `ZIP_PATH` や `DATA_PATH` を実際のパスに変更
3. セルを順番に実行

### ZIPファイルの展開（基本）

```python
import zipfile

zip_path = "/Volumes/workspace/default/annotation_data/annotation_20251201_001802.zip"
extract_path = "/Volumes/workspace/default/annotation_data/annotation_20251201_001802"

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(extract_path)

print("展開完了!")
```

### アノテーションの読み込み

```python
import os
import json

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

annotations = load_annotations(extract_path)
print(f"アノテーション数: {len(annotations)}")
```

### 画像の表示

```python
from PIL import Image
import os

images_dir = os.path.join(extract_path, "images")
sample = annotations[0]

# 画像を読み込み
img_name = sample['cam/image_array']
img_path = os.path.join(images_dir, img_name)
img = Image.open(img_path)

display(img)
print(f"Angle: {sample['user/angle']:.3f}")
print(f"Throttle: {sample['user/throttle']:.3f}")
```

## トラブルシューティング

### 転送先パスが存在しない

```
エラー: パスが存在しません: /Volumes/workspace/default/annotation_data
```

**対処法**: Databricksで先にVolumesを作成してください（上記「Databricksでの準備」参照）

### タイムアウトエラー

```
エラー: Timed out after 0:05:00
```

**対処法**:
- ネットワーク接続を確認
- ファイルサイズが大きい場合は、アノテーション数を減らして再試行
- 時間帯を変えて再試行（ネットワーク混雑の可能性）

### 環境変数一覧（転送関連）

| 環境変数 | 必須 | 説明 | デフォルト値 |
|----------|------|------|-------------|
| `DATABRICKS_VOLUMES_PATH` | No | 転送先Volumesパス | `/Volumes/workspace/default/annotation_data` |

## ファイル構成

```
annotation_training_d2j/
├── config_databricks.py          # Databricks設定
├── utils/
│   └── databricks_transfer.py    # 転送処理
├── databricks/                   # Databricksサンプルノートブック
│   ├── 01_extract_annotations.py # ZIPの展開
│   ├── 02_load_annotations.py    # データ読み込み
│   └── 03_train_model.py         # モデル学習
├── mlruns/                       # ローカルMLflow記録
└── README_DATABRICKS.md          # このファイル
```
