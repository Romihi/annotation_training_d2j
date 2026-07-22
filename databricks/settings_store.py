# settings_store.py
"""
Databricks設定のローカル永続化ストア

アプリ内の設定ダイアログで編集した値を JSON ファイルに保存し、
起動時に読み込んで os.environ に反映する。これにより環境変数を
毎回シェルで設定したり、変更のたびにアプリを再起動する必要がなくなる。

precedence（優先順位）:
    ストアに非空の値がある場合、それを os.environ に上書き適用する
    （＝アプリ内で最後に保存した設定が正）。
    ストアに値が無いキーはシェルの環境変数がそのまま使われる。

セキュリティ:
    このファイルにはトークンが平文で保存される。リポジトリには
    コミットしないこと（.gitignore に databricks_settings.local.json を追加済み）。
"""

import json
import os

# 設定ファイルのパス（databricks/ の 1つ上 = アプリルート直下）
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(_APP_DIR, "databricks_settings.local.json")

# 永続化するキー（すべて DATABRICKS_ プレフィックスの環境変数名と一致させる）
SETTINGS_KEYS = [
    "DATABRICKS_ENABLED",
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_EXPERIMENT_PREFIX",
    "DATABRICKS_VOLUMES_PATH",
    "DATABRICKS_CLUSTER_ID",
    "DATABRICKS_NOTEBOOK_PATH",
    "DATABRICKS_CATALOG",
    "DATABRICKS_SCHEMA",
]


def load() -> dict:
    """設定ファイルを読み込む（存在しない/壊れている場合は空辞書）"""
    if not os.path.exists(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # 既知キーのみ返す（未知キーは無視）
            return {k: v for k, v in data.items() if k in SETTINGS_KEYS}
        return {}
    except Exception as e:
        print(f"[settings_store] 読み込み失敗: {e}")
        return {}


def apply_to_environ(settings: dict) -> None:
    """設定値を os.environ に反映（非空の値のみ上書き）"""
    for key in SETTINGS_KEYS:
        value = settings.get(key)
        if value is None:
            continue
        value = str(value)
        if value != "":
            os.environ[key] = value


def save(settings: dict) -> str:
    """設定を保存（既存値とマージ）し、os.environ にも即時反映

    Args:
        settings: 保存する {キー: 値} 。SETTINGS_KEYS 以外は無視。
                  値が空文字のキーは「削除」として扱い、保存対象から外す。

    Returns:
        保存先ファイルパス
    """
    current = load()
    for key in SETTINGS_KEYS:
        if key not in settings:
            continue
        value = settings[key]
        if value is None or str(value) == "":
            current.pop(key, None)          # 空はストアから削除
        else:
            current[key] = str(value)

    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
        # 認証情報を含むため、可能ならオーナー読み書きのみに制限（Windowsでは無視される）
        try:
            os.chmod(SETTINGS_PATH, 0o600)
        except Exception:
            pass
    except Exception as e:
        print(f"[settings_store] 保存失敗: {e}")

    apply_to_environ(current)
    return SETTINGS_PATH


def load_and_apply() -> dict:
    """設定ファイルを読み込み、os.environ に反映して返す（起動時に呼ぶ）"""
    settings = load()
    apply_to_environ(settings)
    return settings


def has_saved_settings() -> bool:
    """アプリ内で保存された設定ファイルが存在するか"""
    return os.path.exists(SETTINGS_PATH)
