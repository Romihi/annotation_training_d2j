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
# アプリ内保存の設定ストアを os.environ に反映
# （環境変数を読む前に呼ぶ。ストアの非空値が環境変数を上書きする）
# ===========================================
try:
    from databricks import settings_store as _settings_store
    _settings_store.load_and_apply()
except Exception as _e:  # ストアが無い/壊れていても環境変数だけで動作可能
    _settings_store = None
    print(f"[config_databricks] 設定ストアの読み込みをスキップ: {_e}")

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

# ===========================================
# 設定の再読込・保存（アプリ内編集用・再起動不要）
# ===========================================

def reload():
    """設定ストア→os.environ を再適用し、モジュール変数を再読込する。

    アプリ内で設定を変更・保存した直後に呼ぶことで、アプリを再起動せずに
    新しい設定（HOST/TOKEN/VOLUMES/CLUSTER等）を反映できる。
    モジュール属性として参照している側（config_databricks.DATABRICKS_HOST 等）
    は最新値を得られる（トップレベルで from ... import した名前は更新されない
    点に注意。接続処理は config_databricks.X の形で参照すること）。
    """
    global DATABRICKS_ENABLED, DATABRICKS_HOST, DATABRICKS_TOKEN
    global DATABRICKS_EXPERIMENT_PREFIX, DATABRICKS_MODEL_REGISTRY_CATALOG
    global DATABRICKS_MODEL_REGISTRY_SCHEMA, DATABRICKS_CLUSTER_ID
    global DATABRICKS_NOTEBOOK_PATH, DATABRICKS_VOLUMES_PATH

    if _settings_store is not None:
        try:
            _settings_store.load_and_apply()
        except Exception as e:
            print(f"[config_databricks] reload時のストア適用に失敗: {e}")

    DATABRICKS_ENABLED = _get_bool_env("DATABRICKS_ENABLED", False)
    DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
    DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
    DATABRICKS_EXPERIMENT_PREFIX = os.environ.get(
        "DATABRICKS_EXPERIMENT_PREFIX", "/Shared/annotation_tool_experiments")
    DATABRICKS_MODEL_REGISTRY_CATALOG = os.environ.get("DATABRICKS_CATALOG", "main")
    DATABRICKS_MODEL_REGISTRY_SCHEMA = os.environ.get("DATABRICKS_SCHEMA", "default")
    DATABRICKS_CLUSTER_ID = os.environ.get("DATABRICKS_CLUSTER_ID", "")
    DATABRICKS_NOTEBOOK_PATH = os.environ.get(
        "DATABRICKS_NOTEBOOK_PATH",
        "/Workspace/Users/{user}/annotation_training_d2j/databricks")
    DATABRICKS_VOLUMES_PATH = os.environ.get(
        "DATABRICKS_VOLUMES_PATH", "/Volumes/workspace/default/annotation_data")


def save_settings(new_values: dict) -> str:
    """アプリ内で編集した設定を永続化し、即時反映する。

    Args:
        new_values: {環境変数名: 値} の辞書（settings_store.SETTINGS_KEYS のキー）

    Returns:
        保存先ファイルパス（ストアが無い場合は空文字）
    """
    path = ""
    if _settings_store is not None:
        path = _settings_store.save(new_values)
    else:
        # ストアが使えない場合でも当該セッションには反映する
        for k, v in new_values.items():
            if v is not None and str(v) != "":
                os.environ[k] = str(v)
    reload()
    return path


def get_all_settings() -> dict:
    """現在の全設定値を辞書で返す（設定ダイアログの初期表示用）"""
    return {
        "DATABRICKS_ENABLED": "true" if DATABRICKS_ENABLED else "false",
        "DATABRICKS_HOST": DATABRICKS_HOST,
        "DATABRICKS_TOKEN": DATABRICKS_TOKEN,
        "DATABRICKS_EXPERIMENT_PREFIX": DATABRICKS_EXPERIMENT_PREFIX,
        "DATABRICKS_VOLUMES_PATH": DATABRICKS_VOLUMES_PATH,
        "DATABRICKS_CLUSTER_ID": DATABRICKS_CLUSTER_ID,
        "DATABRICKS_NOTEBOOK_PATH": DATABRICKS_NOTEBOOK_PATH,
        "DATABRICKS_CATALOG": DATABRICKS_MODEL_REGISTRY_CATALOG,
        "DATABRICKS_SCHEMA": DATABRICKS_MODEL_REGISTRY_SCHEMA,
    }
