# Google Colab連携機能

アノテーションデータをGoogle Driveに転送し、Google Colabで学習を実行する機能です。

## 目次

- [セットアップ](#セットアップ)
- [使い方](#使い方)
- [学習ノートブック](#学習ノートブック)
- [利用可能なモデル](#利用可能なモデル)
- [トラブルシューティング](#トラブルシューティング)

---

## セットアップ

### 1. Google Cloud Console でOAuth認証情報を作成

1. [Google Cloud Console](https://console.cloud.google.com/) にアクセス
2. プロジェクトを作成または選択
3. 左メニュー「APIとサービス」→「ライブラリ」
4. 「Google Drive API」を検索して有効化
5. 左メニュー「APIとサービス」→「認証情報」
6. 「認証情報を作成」→「OAuthクライアントID」
7. **アプリケーションの種類: 「ウェブアプリケーション」を選択**
8. 承認済みのリダイレクトURIに追加:
   ```
   http://localhost:8080/
   ```
9. 「作成」をクリック
10. JSONをダウンロード → `client_secrets.json` としてプロジェクトルートに保存

### 2. テストユーザーの追加（公開前の場合）

OAuth同意画面が「テスト」モードの場合:

1. Google Cloud Console →「APIとサービス」→「OAuth同意画面」
2. 「テストユーザー」セクションで「+ ADD USERS」
3. 使用するGoogleアカウントのメールアドレスを追加

### 3. 環境変数の設定

```bash
# Windows (PowerShell)
$env:COLAB_ENABLED = "true"
$env:GOOGLE_CLIENT_SECRETS = "C:\path\to\client_secrets.json"

# Windows (コマンドプロンプト)
set COLAB_ENABLED=true
set GOOGLE_CLIENT_SECRETS=C:\path\to\client_secrets.json

# Linux/macOS
export COLAB_ENABLED=true
export GOOGLE_CLIENT_SECRETS=/path/to/client_secrets.json
```

または、`client_secrets.json` をプロジェクトルートに配置すれば自動検出されます。

---

## 使い方

### アプリからの転送

1. アプリを起動し、アノテーション作業を完了
2. 画面下部の「Colab連携」セクションで「有効」チェックボックスをオン
3. 「接続テスト」ボタンでGoogle認証を確認
4. 「転送」ボタンをクリック
5. 転送オプションを設定:
   - フォルダ名（デフォルト: `annotation_data`）
   - ZIPファイル名
   - 学習ノートブック生成の有無
6. 「転送開始」をクリック

### 転送フロー

```
[転送ボタン] → オプションダイアログ → 進捗ダイアログ
     ↓
[Stage 1] Donkey形式でエクスポート (0-15%)
     ↓
[Stage 2] ZIP圧縮 (15-45%)
     ↓
[Stage 3] Google Driveにアップロード (45-80%)
     ↓
[Stage 4] 学習ノートブック生成・アップロード (80-100%)
     ↓
[完了] ブラウザでColabを開く（オプション）
```

### 学習済みモデルのダウンロード

Colabで学習したモデルをローカルに取得できます。

1. 「取得」ボタンをクリック
2. Google Drive上のモデル一覧が表示される
3. ダウンロードするモデルを選択
4. 保存先フォルダを選択（デフォルト: `models/`）
5. 「OK」でダウンロード開始
6. ダウンロード完了後、そのまま読み込むか選択

**対応ファイル形式**:
- `.pt` (PyTorchモデル)
- `.onnx` (ONNX形式)

---

## 学習ノートブック

### VSCodeからColabに接続して使用

1. VSCodeで `colab/train_model.ipynb` を開く
2. 右上のカーネル選択から「既存のJupyterサーバーに接続」
3. Colabのランタイムに接続
4. セルを順番に実行

### ノートブックの設定項目（セル6）

```python
# データ設定
FOLDER_NAME = "annotation_data"      # Google Driveのフォルダ名
DATA_FILE_NAME = "annotation_xxx.zip" # ZIPファイル名

# モデル設定
MODEL_NAME = "resnet18"              # 使用するモデル

# ハイパーパラメータ
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8
INPUT_SIZE = (120, 160)              # 入力画像サイズ
EARLY_STOPPING_PATIENCE = 5
```

### ノートブックの構成

| セクション | 内容 |
|-----------|------|
| 1-2 | Google Driveマウント |
| 3-4 | ライブラリインストール、設定 |
| 5-6 | データ展開・読み込み |
| 7 | Dataset/DataLoader作成 |
| 8 | モデル定義・作成 |
| 9 | 学習ループ（Early Stopping付き） |
| 10-11 | 結果可視化 |
| 12 | モデル保存（Google Driveに保存） |
| 13 | ONNXエクスポート（オプション） |

---

## 利用可能なモデル

| モデル名 | 説明 | パラメータ数 | 推奨用途 |
|---------|------|-------------|---------|
| `donkeycar` | Donkeycar標準モデル | ~100K | 軽量・高速推論 |
| `resnet18` | ResNet18ベース | ~11M | バランス型 |
| `mobilevit_xxs` | MobileViT超軽量 | ~1.3M | エッジデバイス |
| `mobilenetv3_small_100` | MobileNetV3 Small | ~2.5M | モバイル向け |
| `mobilenetv4_conv_small` | MobileNetV4 Small | ~3.8M | 最新軽量モデル |
| `efficientnet_b0` | EfficientNet B0 | ~5.3M | 効率重視 |
| `efficientnetv2_s` | EfficientNetV2 Small | ~21M | 高精度 |
| `edgenext_xx_small` | EdgeNeXt超軽量 | ~1.3M | エッジ最適化 |
| `efficientformer_l1` | EfficientFormer L1 | ~12M | Transformer系 |

---

## トラブルシューティング

### 403: access_denied エラー

**原因**: OAuth同意画面がテストモードで、テストユーザーに登録されていない

**解決方法**:
1. Google Cloud Console →「OAuth同意画面」
2. 「テストユーザー」に自分のメールアドレスを追加

### "no code found in redirect" エラー

**原因**: OAuthクライアントの種類が「デスクトップアプリ」になっている

**解決方法**:
1. OAuthクライアントIDを削除
2. 「ウェブアプリケーション」タイプで新規作成
3. リダイレクトURIに `http://localhost:8080/` を追加
4. 新しい `client_secrets.json` をダウンロード

### 接続テストでタイムアウト

**原因**: ブラウザでの認証が60秒以内に完了しなかった

**解決方法**:
- 再度「接続テスト」をクリック
- ブラウザで素早くGoogleアカウントを選択・許可

### データファイルが見つからない

Colabノートブックで「データファイルが見つかりません」エラーが出る場合:

1. Google Driveの `マイドライブ/{FOLDER_NAME}/` を確認
2. ZIPファイルが正しくアップロードされているか確認
3. ノートブックの `DATA_FILE_NAME` が正しいか確認

### モデル学習が遅い

1. Colabでランタイムタイプを「GPU」に変更
   - 「ランタイム」→「ランタイムのタイプを変更」→「T4 GPU」
2. `BATCH_SIZE` を増やす（メモリが許す範囲で）

---

## ファイル構成

```
project/
├── client_secrets.json          # OAuth認証情報（.gitignore済み）
├── config_colab.py              # Colab設定
├── utils/
│   └── colab_transfer.py        # 転送マネージャー
├── colab/
│   ├── train_model.ipynb        # 学習ノートブック（VSCode用）
│   └── train_model_template.py  # ノートブックテンプレート
└── README_COLAB.md              # このファイル
```

---

## 依存ライブラリ

```
pydrive2>=1.19.0
google-auth>=2.0.0
google-auth-oauthlib>=1.0.0
PyYAML>=6.0
```

インストール:
```bash
pip install pydrive2 google-auth google-auth-oauthlib PyYAML
```
