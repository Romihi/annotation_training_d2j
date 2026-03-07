# config_databricks.py
"""
Databricks連携設定

全ての認証情報は環境変数から読み込みます。
セキュリティのため、このファイルに認証情報を直接記載しないでください。

必要な環境変数:
  DATABRICKS_ENABLED  - 連携の有効/無効 (true/false)
  DATABRICKS_HOST     - ワークスペースURL
  DATABRICKS_TOKEN    - パーソナルアクセストークン

オプションの環境変数:
  DATABRICKS_EXPERIMENT_PREFIX - 実験パス（デフォルト: /Shared/annotation_tool_experiments）
"""

import os
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

# Databricks連携の有効/無効
DATABRICKS_ENABLED = _get_bool_env("DATABRICKS_ENABLED", False)

# Databricks認証設定（環境変数から取得）
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")

# MLflow実験パス
DATABRICKS_EXPERIMENT_PREFIX = os.environ.get(
    "DATABRICKS_EXPERIMENT_PREFIX",
    "/Shared/annotation_tool_experiments"
)

# ===========================================
# 将来の拡張用設定（環境変数から取得）
# ===========================================

# モデルレジストリのカタログ名（Unity Catalog使用時）
DATABRICKS_MODEL_REGISTRY_CATALOG = os.environ.get("DATABRICKS_CATALOG", "main")
DATABRICKS_MODEL_REGISTRY_SCHEMA = os.environ.get("DATABRICKS_SCHEMA", "default")

# Databricksクラスター設定（自動学習パイプライン用）
DATABRICKS_CLUSTER_ID = os.environ.get("DATABRICKS_CLUSTER_ID", "")

# ノートブックのワークスペースパス
DATABRICKS_NOTEBOOK_PATH = os.environ.get(
    "DATABRICKS_NOTEBOOK_PATH",
    "/Workspace/Users/{user}/annotation_training_d2j/databricks"
)

# Unity Catalog Volumes パス（データ保存用）
# 形式: /Volumes/{catalog}/{schema}/{volume_name}
# 例: /Volumes/workspace/default/annotation_data
DATABRICKS_VOLUMES_PATH = os.environ.get(
    "DATABRICKS_VOLUMES_PATH",
    "/Volumes/workspace/default/annotation_data"
)

# ===========================================
# 設定の検証
# ===========================================

def validate_databricks_config():
    """Databricks設定の検証を行う"""
    errors = []

    if DATABRICKS_ENABLED:
        if not DATABRICKS_HOST:
            errors.append(get_text('msg_env_host_not_set'))
        elif not DATABRICKS_HOST.startswith("https://"):
            errors.append(get_text('msg_env_host_https_required'))

        if not DATABRICKS_TOKEN:
            errors.append(get_text('msg_env_token_not_set'))

    return errors

def get_databricks_status():
    """Databricks接続状態のサマリーを返す"""
    if not DATABRICKS_ENABLED:
        return {
            "enabled": False,
            "status": get_text('status_disabled'),
            "message": get_text('msg_databricks_disabled')
        }

    errors = validate_databricks_config()
    if errors:
        return {
            "enabled": True,
            "status": get_text('status_config_error'),
            "message": "\n".join(errors)
        }

    # トークンの一部をマスク表示
    masked_token = "****" + DATABRICKS_TOKEN[-4:] if len(DATABRICKS_TOKEN) > 4 else "****"

    return {
        "enabled": True,
        "status": get_text('status_configured'),
        "host": DATABRICKS_HOST,
        "token_masked": masked_token,
        "experiment_prefix": DATABRICKS_EXPERIMENT_PREFIX,
        "message": get_text('msg_databricks_workspace', DATABRICKS_HOST)
    }

def print_config_status():
    """設定状態をコンソールに出力（デバッグ用）"""
    status = get_databricks_status()
    print("=" * 50)
    print("Databricks設定状態")
    print("=" * 50)
    print(f"有効: {status['enabled']}")
    print(f"状態: {status['status']}")
    print(f"メッセージ: {status['message']}")
    if status['enabled'] and status['status'] == "設定済み":
        print(f"ホスト: {status['host']}")
        print(f"トークン: {status['token_masked']}")
        print(f"実験パス: {status['experiment_prefix']}")
    print("=" * 50)

# ===========================================
# 環境変数設定のヘルパー（開発用）
# ===========================================

def get_env_template():
    """環境変数設定のテンプレートを返す"""
    return get_text('msg_databricks_env_template')
