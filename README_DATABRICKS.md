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

### 2. Databricksアクセストークンの取得

1. Databricksワークスペースにログイン
2. 右上のユーザーアイコン → **Settings**
3. **Developer** → **Access tokens**
4. **Generate new token** をクリック
5. トークンをコピー（一度しか表示されません）

### 3. 環境変数の設定

セキュリティのため、全ての認証情報は環境変数で設定します。

#### Windows (PowerShell)

```powershell
$env:DATABRICKS_ENABLED = "true"
$env:DATABRICKS_HOST = "https://your-workspace.cloud.databricks.com"
$env:DATABRICKS_TOKEN = "dapi..."
$env:DATABRICKS_EXPERIMENT_PREFIX = "/Users/your-email@example.com/experiments"
```

#### Windows (コマンドプロンプト)

```cmd
set DATABRICKS_ENABLED=true
set DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
set DATABRICKS_TOKEN=dapi...
set DATABRICKS_EXPERIMENT_PREFIX=/Users/your-email@example.com/experiments
```

#### Linux/Mac

```bash
export DATABRICKS_ENABLED="true"
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_EXPERIMENT_PREFIX="/Users/your-email@example.com/experiments"
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
   - トークンが有効か確認（期限切れの場合は再生成）
   - トークンに適切な権限があるか確認

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
