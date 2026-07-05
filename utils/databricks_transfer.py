"""
Databricksへのデータ転送ユーティリティ

アノテーションデータをエクスポート → ZIP圧縮 → Databricks Unity Catalog Volumesにアップロード
"""

import os
import zipfile
import tempfile
import shutil
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.files import FilesAPI

from utils.export_utils import export_to_donkey
from databricks import config_databricks


def debug_print(message: str):
    """デバッグ出力（即座にフラッシュ）"""
    print(f"[DatabricksTransfer] {message}")
    sys.stdout.flush()


class DatabricksTransferManager:
    """Databricksへのデータ転送を管理するクラス"""

    def __init__(self, volumes_path: str = None):
        """
        Args:
            volumes_path: Unity Catalog Volumesのパス
                         例: /Volumes/main/default/annotation_data
                         Noneの場合はconfig_databricks.DATABRICKS_VOLUMES_PATHを使用
        """
        self.volumes_path = volumes_path or config_databricks.DATABRICKS_VOLUMES_PATH
        self._client = None

    @property
    def client(self) -> WorkspaceClient:
        """WorkspaceClientを取得（遅延初期化）"""
        if self._client is None:
            self._client = WorkspaceClient(
                host=config_databricks.DATABRICKS_HOST,
                token=config_databricks.DATABRICKS_TOKEN
            )
        return self._client

    def test_connection(self) -> tuple:
        """接続テスト

        Returns:
            (success: bool, message: str)
        """
        debug_print("接続テスト開始...")
        try:
            # ワークスペースの情報を取得して接続確認
            current_user = self.client.current_user.me()
            debug_print(f"接続成功: {current_user.user_name}")
            return True, f"接続成功: {current_user.user_name}"
        except Exception as e:
            debug_print(f"接続失敗: {str(e)}")
            return False, f"接続失敗: {str(e)}"

    def check_volumes_path(self) -> tuple:
        """Volumesパスの存在確認

        Returns:
            (exists: bool, message: str)
        """
        debug_print(f"Volumesパス確認: {self.volumes_path}")
        try:
            # パスの存在確認
            items = list(self.client.files.list_directory_contents(self.volumes_path))
            debug_print(f"Volumesパス存在確認OK: {len(items)}個のアイテム")
            return True, f"パスが存在します（{len(items)}個のファイル/フォルダ）"
        except Exception as e:
            error_msg = str(e)
            debug_print(f"Volumesパス確認エラー: {error_msg}")
            if "NOT_FOUND" in error_msg or "404" in error_msg:
                return False, f"パスが存在しません: {self.volumes_path}\n\nDatabricksでVolumesを作成してください。"
            elif "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                return False, f"アクセス権限がありません: {self.volumes_path}"
            else:
                return False, f"確認エラー: {error_msg}"

    def create_volumes_directory(self) -> tuple:
        """Volumesディレクトリを作成（可能な場合）

        Returns:
            (success: bool, message: str)
        """
        debug_print(f"Volumesディレクトリ作成試行: {self.volumes_path}")
        try:
            # ディレクトリ作成を試みる
            self.client.files.create_directory(self.volumes_path)
            debug_print("Volumesディレクトリ作成成功")
            return True, "ディレクトリを作成しました"
        except Exception as e:
            error_msg = str(e)
            debug_print(f"Volumesディレクトリ作成エラー: {error_msg}")
            return False, f"ディレクトリ作成に失敗: {error_msg}"

    def create_zip(
        self,
        source_dir: str,
        zip_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> bool:
        """ディレクトリをZIP圧縮

        Args:
            source_dir: 圧縮対象ディレクトリ
            zip_path: 出力ZIPファイルパス
            progress_callback: 進捗コールバック(current, total, filename)
            cancel_check: キャンセル確認コールバック（Trueでキャンセル）

        Returns:
            成功した場合True
        """
        debug_print(f"ZIP圧縮開始: {source_dir} -> {zip_path}")

        # ファイル一覧を取得
        all_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                all_files.append((file_path, arcname))

        total_files = len(all_files)
        debug_print(f"圧縮対象ファイル数: {total_files}")

        try:
            # ZIP_STOREDを使用（画像は既に圧縮済みなので再圧縮しない）
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
                for i, (file_path, arcname) in enumerate(all_files):
                    # キャンセルチェック
                    if cancel_check and cancel_check():
                        debug_print("ZIP圧縮がキャンセルされました")
                        # 途中のZIPファイルを削除
                        zf.close()
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                        return False

                    # ファイルを追加
                    zf.write(file_path, arcname)

                    # 進捗コールバック（100ファイルごとにデバッグ出力）
                    if progress_callback:
                        progress_callback(i + 1, total_files, arcname)

                    if (i + 1) % 100 == 0 or (i + 1) == total_files:
                        debug_print(f"ZIP圧縮進捗: {i + 1}/{total_files} ({(i + 1) * 100 // total_files}%)")

            zip_size = os.path.getsize(zip_path)
            debug_print(f"ZIP圧縮完了: {zip_size / (1024*1024):.2f} MB")
            return True

        except Exception as e:
            debug_print(f"ZIP圧縮エラー: {e}")
            # エラー時はZIPファイルを削除
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """ファイルをDatabricksにアップロード

        Args:
            local_path: ローカルファイルパス
            remote_path: リモートパス（Volumes内の相対パス）
            progress_callback: 進捗コールバック(uploaded_bytes, total_bytes)

        Returns:
            成功した場合True
        """
        # フルパスを構築
        full_remote_path = f"{self.volumes_path}/{remote_path}"

        file_size = os.path.getsize(local_path)
        debug_print(f"アップロード開始: {local_path} ({file_size / (1024*1024):.2f} MB)")
        debug_print(f"転送先: {full_remote_path}")

        try:
            # ファイルをバイナリストリームとしてアップロード
            debug_print("ファイルストリームでアップロード中...")
            with open(local_path, 'rb') as f:
                # Files APIでストリームアップロード
                self.client.files.upload(full_remote_path, f, overwrite=True)
            debug_print("アップロード完了")

            # 完了コールバック
            if progress_callback:
                progress_callback(file_size, file_size)

            return True

        except Exception as e:
            debug_print(f"アップロードエラー: {e}")
            raise

    def transfer_annotations(
        self,
        annotations: Dict[Union[str, int], Dict[str, Any]],
        inference_results: Optional[Dict[Union[str, int], Dict[str, Any]]] = None,
        image_map: Optional[Dict[int, Dict[str, str]]] = None,
        variant_keys: Optional[Dict[str, str]] = None,
        zip_name: Optional[str] = None,
        deleted_indexes: Optional[List[int]] = None,
        diff_vectors: Optional[Dict[Union[str, int], Dict[str, Any]]] = None,
        waypoint_annotations: Optional[Dict[Union[str, int], List[tuple]]] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> dict:
        """アノテーションをエクスポート → ZIP → アップロード

        Args:
            annotations: アノテーション辞書
            inference_results: 推論結果辞書
            image_map: インデックスごとの画像パスマップ
            variant_keys: 画像バリアントのキー名
            zip_name: ZIPファイル名（Noneの場合は自動生成）
            deleted_indexes: 削除されたインデックス
            diff_vectors: 差分ベクトル
            waypoint_annotations: ウェイポイントアノテーション
            progress_callback: 進捗コールバック(stage, current, total, message)
                stage: 'export', 'zip', 'upload'
            cancel_check: キャンセル確認コールバック

        Returns:
            {
                'success': bool,
                'zip_size': int,
                'remote_path': str,
                'annotation_count': int,
                'error': str (失敗時のみ)
            }
        """
        debug_print("========== 転送処理開始 ==========")
        debug_print(f"アノテーション数: {len(annotations)}")

        # ZIPファイル名を生成
        if zip_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"annotation_{timestamp}.zip"

        # .zipで終わっていなければ追加
        if not zip_name.endswith('.zip'):
            zip_name += '.zip'

        debug_print(f"ZIPファイル名: {zip_name}")

        # 一時ディレクトリを作成
        temp_dir = tempfile.mkdtemp(prefix="databricks_transfer_")
        export_dir = os.path.join(temp_dir, "export")
        zip_path = os.path.join(temp_dir, zip_name)
        debug_print(f"一時ディレクトリ: {temp_dir}")

        try:
            # ステージ1: エクスポート
            debug_print("--- ステージ1: エクスポート ---")
            if progress_callback:
                progress_callback('export', 0, 100, 'アノテーションをエクスポート中...')

            if cancel_check and cancel_check():
                debug_print("キャンセルされました（エクスポート前）")
                return {'success': False, 'error': 'キャンセルされました'}

            # export_to_donkeyでエクスポート
            debug_print("export_to_donkey 呼び出し中...")
            manifest_path = export_to_donkey(
                folder_path=export_dir,
                annotations=annotations,
                inference_results=inference_results,
                deleted_indexes=deleted_indexes,
                image_map=image_map,
                variant_keys=variant_keys,
                diff_vectors=diff_vectors,
                waypoint_annotations=waypoint_annotations
            )
            debug_print(f"export_to_donkey 完了: {manifest_path}")

            if not manifest_path:
                debug_print("エクスポート失敗: manifest_path が None")
                return {'success': False, 'error': 'エクスポートに失敗しました'}

            if progress_callback:
                progress_callback('export', 100, 100, 'エクスポート完了')

            # ステージ2: ZIP圧縮
            debug_print("--- ステージ2: ZIP圧縮 ---")
            if cancel_check and cancel_check():
                debug_print("キャンセルされました（ZIP圧縮前）")
                return {'success': False, 'error': 'キャンセルされました'}

            def zip_progress(current, total, filename):
                if progress_callback:
                    progress_callback('zip', current, total, f'圧縮中: {filename}')

            success = self.create_zip(
                source_dir=export_dir,
                zip_path=zip_path,
                progress_callback=zip_progress,
                cancel_check=cancel_check
            )

            if not success:
                debug_print("ZIP圧縮がキャンセルされました")
                return {'success': False, 'error': 'ZIP圧縮がキャンセルされました'}

            zip_size = os.path.getsize(zip_path)
            debug_print(f"ZIPファイルサイズ: {zip_size / (1024*1024):.2f} MB")

            # ステージ3: アップロード
            debug_print("--- ステージ3: アップロード ---")
            if cancel_check and cancel_check():
                debug_print("キャンセルされました（アップロード前）")
                return {'success': False, 'error': 'キャンセルされました'}

            if progress_callback:
                progress_callback('upload', 0, zip_size, 'アップロード中...')

            def upload_progress(uploaded, total):
                if progress_callback:
                    progress_callback('upload', uploaded, total,
                                     f'アップロード中: {uploaded // (1024*1024)} MB / {total // (1024*1024)} MB')

            self.upload_file(
                local_path=zip_path,
                remote_path=zip_name,
                progress_callback=upload_progress
            )

            remote_full_path = f"{self.volumes_path}/{zip_name}"

            debug_print("========== 転送完了 ==========")
            debug_print(f"転送先: {remote_full_path}")

            return {
                'success': True,
                'zip_size': zip_size,
                'remote_path': remote_full_path,
                'annotation_count': len(annotations)
            }

        except Exception as e:
            debug_print(f"転送エラー: {e}")
            import traceback
            debug_print(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }

        finally:
            # クリーンアップ: 一時ディレクトリを削除
            debug_print(f"一時ディレクトリを削除: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                debug_print("一時ディレクトリ削除完了")
            except Exception as e:
                debug_print(f"一時ディレクトリの削除に失敗: {e}")

    def submit_training_workflow(self, zip_remote_path: str, notebook_base_path: str, cluster_id: str) -> dict:
        """
        Databricks Runs Submit APIで3ノートブックをチェーン実行

        Args:
            zip_remote_path: アップロード済みZIPファイルのリモートパス
            notebook_base_path: ノートブックのワークスペースベースパス
            cluster_id: 既存クラスターID

        Returns:
            {"run_id": int, "run_url": str}

        Raises:
            Exception: ジョブ送信に失敗した場合
        """
        from databricks.sdk.service.jobs import SubmitTask, NotebookTask, TaskDependency

        extract_path = zip_remote_path.replace(".zip", "")

        debug_print(f"ワークフロー送信: zip={zip_remote_path}, notebooks={notebook_base_path}, cluster={cluster_id}")

        wait = self.client.jobs.submit(
            run_name=f"auto_train_{os.path.basename(zip_remote_path)}",
            tasks=[
                SubmitTask(
                    task_key="extract",
                    existing_cluster_id=cluster_id,
                    notebook_task=NotebookTask(
                        notebook_path=f"{notebook_base_path}/01_extract_annotations",
                        base_parameters={"zip_path": zip_remote_path}
                    )
                ),
                SubmitTask(
                    task_key="load",
                    depends_on=[TaskDependency(task_key="extract")],
                    existing_cluster_id=cluster_id,
                    notebook_task=NotebookTask(
                        notebook_path=f"{notebook_base_path}/02_load_annotations",
                        base_parameters={"data_path": extract_path}
                    )
                ),
                SubmitTask(
                    task_key="train",
                    depends_on=[TaskDependency(task_key="load")],
                    existing_cluster_id=cluster_id,
                    notebook_task=NotebookTask(
                        notebook_path=f"{notebook_base_path}/03_train_model",
                        base_parameters={"data_path": extract_path}
                    )
                ),
            ]
        )

        run_id = wait.response.run_id
        run_url = f"{config_databricks.DATABRICKS_HOST}#job/{run_id}"
        debug_print(f"ワークフロー送信完了: run_id={run_id}, url={run_url}")
        return {"run_id": run_id, "run_url": run_url}

    def list_remote_files(self, remote_path: str = "") -> List[dict]:
        """リモートのファイル一覧を取得

        Args:
            remote_path: Volumes内の相対パス（空文字でルート）

        Returns:
            ファイル情報のリスト
            [{'name': str, 'path': str, 'size': int, 'is_dir': bool}, ...]
        """
        full_path = f"{self.volumes_path}/{remote_path}".rstrip('/')

        try:
            items = self.client.files.list_directory_contents(full_path)
            result = []
            for item in items:
                result.append({
                    'name': item.name,
                    'path': item.path,
                    'size': getattr(item, 'file_size', 0) or 0,
                    'is_dir': item.is_directory
                })
            return result
        except Exception as e:
            print(f"ファイル一覧取得エラー: {e}")
            return []

    def delete_remote_file(self, remote_path: str) -> bool:
        """リモートファイルを削除

        Args:
            remote_path: Volumes内の相対パスまたはフルパス

        Returns:
            成功した場合True
        """
        # フルパスでない場合は構築
        if not remote_path.startswith('/Volumes'):
            full_path = f"{self.volumes_path}/{remote_path}"
        else:
            full_path = remote_path

        try:
            self.client.files.delete(full_path)
            return True
        except Exception as e:
            print(f"ファイル削除エラー: {e}")
            return False
