"""mlflow を Python 3.14 で動かすためのホットパッチ。

mlflow 3.14 の `mlflow/assistant/skill_installer.py` は
`from importlib.abc import Traversable` を実行するが、Python 3.14 では
`Traversable` が `importlib.abc` から削除されている（`importlib.resources.abc` へ移動）。
このため `mlflow ui`（FastAPIサーバ）起動時に ImportError で停止する。

本スクリプトはその import を新旧Python両対応の形に書き換える（冪等）。
mlflow を再インストールした後に1回実行すれば再適用される。

使い方:
    # venv の python で実行すること
    venv\\Scripts\\python tools/patch_mlflow_py314.py
"""
import os
import sys

OLD = "from importlib.abc import Traversable"
NEW = (
    "try:\n"
    "    # Python 3.12+ : Traversable は importlib.resources.abc へ移動（3.14でimportlib.abcから削除）\n"
    "    from importlib.resources.abc import Traversable\n"
    "except ImportError:\n"
    "    from importlib.abc import Traversable"
)


def main():
    try:
        import mlflow
    except ImportError:
        print("mlflow が見つかりません。venv の python で実行してください。")
        sys.exit(1)

    target = os.path.join(
        os.path.dirname(mlflow.__file__), "assistant", "skill_installer.py"
    )
    if not os.path.exists(target):
        print(f"対象ファイルがありません（このmlflowバージョンでは不要かも）: {target}")
        return

    with open(target, "r", encoding="utf-8") as f:
        src = f.read()

    if "from importlib.resources.abc import Traversable" in src:
        print(f"既にパッチ済みです: {target}")
        return

    if OLD not in src:
        print(f"対象のimport行が見つかりません（バージョンが異なる可能性）: {target}")
        return

    src = src.replace(OLD, NEW, 1)
    with open(target, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"パッチ適用完了: {target}")
    print("mlflow version:", mlflow.__version__, "| Python:", sys.version.split()[0])


if __name__ == "__main__":
    main()
