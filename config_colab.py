# config_colab.py
"""
Google Colab連携設定

全ての認証情報は環境変数またはclient_secrets.jsonから読み込みます。
セキュリティのため、このファイルに認証情報を直接記載しないでください。

必要な環境変数/ファイル:
  COLAB_ENABLED          - 連携の有効/無効 (true/false)
  GOOGLE_CLIENT_SECRETS  - OAuth2クライアントシークレットJSONファイルパス

オプションの環境変数:
  COLAB_DRIVE_FOLDER_ID   - Google Drive保存先フォルダID（省略時は自動作成）
  COLAB_DRIVE_FOLDER_NAME - Google Drive保存先フォルダ名（デフォルト: annotation_data）
"""

import os
from pathlib import Path

# ===========================================
# 環境変数から設定を読み込み
# ===========================================

def _get_bool_env(key: str, default: bool = False) -> bool:
    """環境変数からブール値を取得"""
    value = os.environ.get(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default

# Colab連携の有効/無効
COLAB_ENABLED = _get_bool_env("COLAB_ENABLED", False)

# Google OAuth2設定
GOOGLE_CLIENT_SECRETS = os.environ.get(
    "GOOGLE_CLIENT_SECRETS",
    str(Path(__file__).parent / "client_secrets.json")
)

# 認証トークン保存パス
GOOGLE_CREDENTIALS_PATH = os.environ.get(
    "GOOGLE_CREDENTIALS_PATH",
    str(Path(__file__).parent / ".google_credentials.json")
)

# Google Drive設定
COLAB_DRIVE_FOLDER_ID = os.environ.get("COLAB_DRIVE_FOLDER_ID", "")
COLAB_DRIVE_FOLDER_NAME = os.environ.get("COLAB_DRIVE_FOLDER_NAME", "annotation_data")

# ===========================================
# 設定の検証
# ===========================================

def validate_colab_config() -> list:
    """Colab設定の検証を行う"""
    errors = []

    if COLAB_ENABLED:
        if not GOOGLE_CLIENT_SECRETS:
            errors.append("環境変数 GOOGLE_CLIENT_SECRETS が設定されていません")
        elif not os.path.exists(GOOGLE_CLIENT_SECRETS):
            errors.append(f"クライアントシークレットファイルが見つかりません: {GOOGLE_CLIENT_SECRETS}")

    return errors

def get_colab_status() -> dict:
    """Colab接続状態のサマリーを返す"""
    if not COLAB_ENABLED:
        return {
            "enabled": False,
            "status": "無効",
            "message": "Google Colab連携は無効です\n\n"
                      "有効にするには環境変数を設定してください:\n"
                      "  COLAB_ENABLED=true\n"
                      "  GOOGLE_CLIENT_SECRETS=path/to/client_secrets.json"
        }

    errors = validate_colab_config()
    if errors:
        return {
            "enabled": True,
            "status": "設定エラー",
            "message": "\n".join(errors)
        }

    # 認証済みかどうかを確認
    authenticated = os.path.exists(GOOGLE_CREDENTIALS_PATH)

    return {
        "enabled": True,
        "status": "認証済み" if authenticated else "未認証",
        "authenticated": authenticated,
        "drive_folder": COLAB_DRIVE_FOLDER_NAME,
        "message": f"Google Drive フォルダ: {COLAB_DRIVE_FOLDER_NAME}"
                   + ("\n認証済み" if authenticated else "\n要認証（初回転送時にブラウザ認証）")
    }

def print_config_status():
    """設定状態をコンソールに出力（デバッグ用）"""
    status = get_colab_status()
    print("=" * 50)
    print("Google Colab設定状態")
    print("=" * 50)
    print(f"有効: {status['enabled']}")
    print(f"状態: {status['status']}")
    print(f"メッセージ: {status['message']}")
    if status['enabled'] and status['status'] in ["認証済み", "未認証"]:
        print(f"Driveフォルダ: {status.get('drive_folder', 'N/A')}")
    print("=" * 50)

# ===========================================
# 環境変数設定のヘルパー（開発用）
# ===========================================

def get_env_template() -> str:
    """環境変数設定のテンプレートを返す"""
    return """
# Google Colab設定用環境変数

# Windows (PowerShell):
$env:COLAB_ENABLED = "true"
$env:GOOGLE_CLIENT_SECRETS = "C:\\path\\to\\client_secrets.json"
$env:COLAB_DRIVE_FOLDER_NAME = "annotation_data"

# Windows (コマンドプロンプト):
set COLAB_ENABLED=true
set GOOGLE_CLIENT_SECRETS=C:\\path\\to\\client_secrets.json
set COLAB_DRIVE_FOLDER_NAME=annotation_data

# Linux/Mac:
export COLAB_ENABLED="true"
export GOOGLE_CLIENT_SECRETS="/path/to/client_secrets.json"
export COLAB_DRIVE_FOLDER_NAME="annotation_data"

# .env ファイル形式:
COLAB_ENABLED=true
GOOGLE_CLIENT_SECRETS=/path/to/client_secrets.json
COLAB_DRIVE_FOLDER_NAME=annotation_data
"""

def get_oauth_setup_guide() -> str:
    """OAuth設定ガイドを返す"""
    return """
================================================================================
Google Cloud Console での OAuth設定手順
================================================================================

1. Google Cloud Console にアクセス
   https://console.cloud.google.com/

2. プロジェクトを作成または選択
   - 画面上部のプロジェクト選択メニューから「新しいプロジェクト」
   - プロジェクト名を入力して作成

3. Google Drive API を有効化
   - 左メニューから「APIとサービス」→「ライブラリ」
   - 「Google Drive API」を検索
   - 「有効にする」をクリック

4. OAuth同意画面を設定
   - 「APIとサービス」→「OAuth同意画面」
   - ユーザータイプ: 「外部」を選択（個人使用の場合）
   - アプリ名、メールアドレスを入力
   - スコープの追加は不要（後で自動設定される）
   - テストユーザーに自分のメールアドレスを追加

5. OAuth クライアントIDを作成
   - 「APIとサービス」→「認証情報」
   - 「認証情報を作成」→「OAuthクライアントID」
   - アプリケーションの種類: 「デスクトップアプリ」
   - 名前を入力して「作成」

6. client_secrets.json をダウンロード
   - 作成したクライアントIDの右側にあるダウンロードアイコンをクリック
   - 「JSONをダウンロード」
   - ファイル名を「client_secrets.json」に変更して保存

7. 環境変数を設定
   COLAB_ENABLED=true
   GOOGLE_CLIENT_SECRETS=保存したclient_secrets.jsonのパス

================================================================================
注意事項
================================================================================

- client_secrets.json は秘密情報です。Gitにコミットしないでください
- .gitignore に client_secrets.json を追加することを推奨します
- 初回転送時にブラウザでGoogleアカウントの認証が求められます
- 認証情報は .google_credentials.json に保存され、2回目以降は自動的に使用されます

================================================================================
"""
