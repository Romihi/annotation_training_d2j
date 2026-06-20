"""既存のファイルストア(file://)MLflowデータを sqlite バックエンドへ移行するツール。

MLflow 3 ではファイルストアが非推奨(2026年2月〜)になったため、
ローカル記録を sqlite:///<dir>/mlflow.db + file://<dir>/mlartifacts へ移行する。

使い方:
    # 既定(config.mlflow_dir)を対象に移行
    python tools/migrate_mlflow_to_sqlite.py

    # 任意のディレクトリ(旧ファイルストアのルート)を指定
    python tools/migrate_mlflow_to_sqlite.py --dir "C:/path/to/data_folder"

    # 実行内容だけ確認(書き込みなし)
    python tools/migrate_mlflow_to_sqlite.py --dry-run

移行はパラメータ/タグ/メトリクス(履歴含む)/アーティファクトを再記録する。
run_id は新規発行され、元の run_id は original_run_id タグに保存する。
冪等性: 同名Runが既にsqlite側に存在する場合はスキップする。
"""
import argparse
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


def _normalize(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _file_uri(path: str) -> str:
    norm = _normalize(path)
    return f"file:///{norm}" if sys.platform.startswith("win") else f"file://{norm}"


def _sqlite_uri(base_dir: str) -> str:
    return f"sqlite:///{_normalize(base_dir)}/mlflow.db"


def _artifact_root(base_dir: str) -> str:
    return f"{_file_uri(base_dir)}/mlartifacts"


def migrate(base_dir: str, dry_run: bool = False) -> dict:
    src_uri = _file_uri(base_dir)
    dst_uri = _sqlite_uri(base_dir)
    art_root = _artifact_root(base_dir)

    print(f"移行元 (file store): {src_uri}")
    print(f"移行先 (sqlite)    : {dst_uri}")
    print(f"アーティファクト    : {art_root}")
    if dry_run:
        print("[dry-run] 書き込みは行いません")

    result = {"experiments": 0, "migrated": 0, "skipped": 0, "failed": 0}

    # --- 移行元の実験・Runを収集 ---
    mlflow.set_tracking_uri(src_uri)
    src_client = MlflowClient(tracking_uri=src_uri)
    experiments = src_client.search_experiments()

    collected = []  # (exp_name, run_object)
    for exp in experiments:
        if exp.name == "Default":
            continue
        runs = src_client.search_runs([exp.experiment_id])
        for run in runs:
            collected.append((exp.name, run))
        result["experiments"] += 1
        print(f"  実験 {exp.name}: {len(runs)} runs")

    if not collected:
        print("移行対象のRunがありません")
        return result

    # --- sqlite側に再記録 ---
    for exp_name, run in collected:
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        try:
            mlflow.set_tracking_uri(dst_uri)
            dst_client = MlflowClient(tracking_uri=dst_uri)

            # 実験を用意(なければ作成)
            dst_exp = dst_client.get_experiment_by_name(exp_name)
            if dst_exp is None:
                if dry_run:
                    print(f"  [dry-run] 実験作成: {exp_name}")
                else:
                    dst_client.create_experiment(
                        exp_name, artifact_location=f"{art_root}/{exp_name}"
                    )
                    dst_exp = dst_client.get_experiment_by_name(exp_name)

            # 既存の同名Runはスキップ
            if dst_exp is not None:
                existing = dst_client.search_runs(
                    [dst_exp.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{run_name}'",
                )
                if existing:
                    print(f"  スキップ(既存): {run_name}")
                    result["skipped"] += 1
                    continue

            if dry_run:
                print(f"  [dry-run] 移行: {run_name}")
                result["migrated"] += 1
                continue

            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=run_name):
                # パラメータ
                for k, v in run.data.params.items():
                    mlflow.log_param(k, v)
                # メトリクス(履歴含む)
                for k in run.data.metrics:
                    history = src_client.get_metric_history(run.info.run_id, k)
                    for mh in history:
                        mlflow.log_metric(k, mh.value, step=mh.step, timestamp=mh.timestamp)
                # タグ(システムタグ以外)
                for k, v in run.data.tags.items():
                    if not k.startswith("mlflow."):
                        mlflow.set_tag(k, v)
                mlflow.set_tag("migrated_from_filestore", "true")
                mlflow.set_tag("original_run_id", run.info.run_id)

                # アーティファクト
                try:
                    artifacts = src_client.list_artifacts(run.info.run_id)
                    for art in artifacts:
                        local_path = src_client.download_artifacts(run.info.run_id, art.path)
                        if os.path.isdir(local_path):
                            mlflow.log_artifacts(local_path, art.path)
                        else:
                            mlflow.log_artifact(local_path, os.path.dirname(art.path) or None)
                except Exception as ae:
                    print(f"    アーティファクト移行警告 ({run_name}): {ae}")

            print(f"  移行完了: {run_name}")
            result["migrated"] += 1

        except Exception as e:
            print(f"  移行失敗 ({run_name}): {e}")
            result["failed"] += 1

    return result


def main():
    parser = argparse.ArgumentParser(description="MLflowファイルストア→sqlite移行")
    parser.add_argument("--dir", default=None, help="旧ファイルストアのルートディレクトリ")
    parser.add_argument("--dry-run", action="store_true", help="書き込みせず内容のみ表示")
    args = parser.parse_args()

    base_dir = args.dir
    if base_dir is None:
        # プロジェクトルートを import パスに追加して config を読む
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import mlflow_dir
        base_dir = mlflow_dir

    if not os.path.isdir(base_dir):
        print(f"ディレクトリが存在しません: {base_dir}")
        sys.exit(1)

    result = migrate(base_dir, dry_run=args.dry_run)
    print("\n=== 移行結果 ===")
    print(f"  実験数  : {result['experiments']}")
    print(f"  移行    : {result['migrated']}")
    print(f"  スキップ: {result['skipped']}")
    print(f"  失敗    : {result['failed']}")


if __name__ == "__main__":
    main()
