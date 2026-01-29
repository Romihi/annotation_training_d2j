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
from translations import get_text

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
            errors.append(get_text('msg_env_client_secrets_not_set'))
        elif not os.path.exists(GOOGLE_CLIENT_SECRETS):
            errors.append(get_text('msg_env_client_secrets_not_found', GOOGLE_CLIENT_SECRETS))

    return errors

def get_colab_status() -> dict:
    """Colab接続状態のサマリーを返す"""
    if not COLAB_ENABLED:
        return {
            "enabled": False,
            "status": get_text('status_disabled'),
            "message": get_text('msg_colab_disabled')
        }

    errors = validate_colab_config()
    if errors:
        return {
            "enabled": True,
            "status": get_text('status_config_error'),
            "message": "\n".join(errors)
        }

    # 認証済みかどうかを確認
    authenticated = os.path.exists(GOOGLE_CREDENTIALS_PATH)

    return {
        "enabled": True,
        "status": get_text('status_authenticated') if authenticated else get_text('status_not_authenticated'),
        "authenticated": authenticated,
        "drive_folder": COLAB_DRIVE_FOLDER_NAME,
        "message": get_text('msg_colab_workspace', COLAB_DRIVE_FOLDER_NAME)
                   + (get_text('msg_colab_authenticated') if authenticated else get_text('msg_colab_auth_required'))
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
    return get_text('msg_colab_env_template')

def get_oauth_setup_guide() -> str:
    """OAuth設定ガイドを返す"""
    return get_text('msg_oauth_setup_guide_full')
