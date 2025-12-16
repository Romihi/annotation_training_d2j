"""
Google Colabへのデータ転送ユーティリティ

アノテーションデータをエクスポート → ZIP圧縮 → Google Driveにアップロード → Colabノートブック生成
"""

import os
import zipfile
import tempfile
import shutil
import sys
import webbrowser
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path

from utils.export_utils import export_to_donkey
import config_colab


def debug_print(message: str):
    """デバッグ出力（即座にフラッシュ）"""
    print(f"[ColabTransfer] {message}")
    sys.stdout.flush()


class ColabTransferManager:
    """Google Colabへのデータ転送を管理するクラス"""

    def __init__(self, folder_name: str = None):
        """
        Args:
            folder_name: Google Drive上のフォルダ名
                        Noneの場合はconfig_colab.COLAB_DRIVE_FOLDER_NAMEを使用
        """
        self.folder_name = folder_name or config_colab.COLAB_DRIVE_FOLDER_NAME
        self._gauth = None
        self._drive = None
        self._folder_id = config_colab.COLAB_DRIVE_FOLDER_ID or None

    def _ensure_pydrive2(self):
        """PyDrive2がインストールされているか確認"""
        try:
            from pydrive2.auth import GoogleAuth
            from pydrive2.drive import GoogleDrive
            return True
        except ImportError:
            raise ImportError(
                "PyDrive2がインストールされていません。\n"
                "以下のコマンドでインストールしてください:\n"
                "  pip install pydrive2 google-auth google-auth-oauthlib"
            )

    @property
    def gauth(self):
        """GoogleAuth を取得（遅延初期化）"""
        if self._gauth is None:
            self._ensure_pydrive2()
            from pydrive2.auth import GoogleAuth

            # client_secrets.jsonを読み込んで形式を確認
            client_config = self._load_client_secrets()

            # settings.yamlを動的に作成
            settings_content = {
                "client_config_backend": "settings",
                "client_config": client_config,
                "save_credentials": True,
                "save_credentials_backend": "file",
                "save_credentials_file": config_colab.GOOGLE_CREDENTIALS_PATH,
                "get_refresh_token": True,
                "oauth_scope": ["https://www.googleapis.com/auth/drive"],
            }

            # 一時的なsettings.yamlを作成
            settings_dir = Path(config_colab.GOOGLE_CLIENT_SECRETS).parent
            settings_path = settings_dir / "pydrive_settings.yaml"

            import yaml
            with open(settings_path, 'w', encoding='utf-8') as f:
                yaml.dump(settings_content, f, default_flow_style=False)

            # settings.yamlのパスを指定してGoogleAuthを初期化
            self._gauth = GoogleAuth(settings_file=str(settings_path))

        return self._gauth

    def _load_client_secrets(self) -> dict:
        """client_secrets.jsonを読み込み、PyDrive2形式に変換"""
        with open(config_colab.GOOGLE_CLIENT_SECRETS, 'r', encoding='utf-8') as f:
            secrets = json.load(f)

        # ウェブアプリケーション形式（"web"キー）かデスクトップ形式（"installed"キー）かを判定
        if "web" in secrets:
            config = secrets["web"]
            debug_print("ウェブアプリケーション形式のclient_secrets.jsonを検出")
        elif "installed" in secrets:
            config = secrets["installed"]
            debug_print("デスクトップアプリ形式のclient_secrets.jsonを検出")
        else:
            raise ValueError("client_secrets.jsonの形式が不正です。'web'または'installed'キーが必要です。")

        # PyDrive2が期待する形式に変換
        return {
            "client_id": config.get("client_id"),
            "client_secret": config.get("client_secret"),
            "auth_uri": config.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": config.get("token_uri", "https://oauth2.googleapis.com/token"),
            "redirect_uris": config.get("redirect_uris", ["http://localhost:8080/"])
        }

    @property
    def drive(self):
        """GoogleDrive を取得（遅延初期化）"""
        if self._drive is None:
            self._ensure_pydrive2()
            from pydrive2.drive import GoogleDrive

            self._authenticate()
            self._drive = GoogleDrive(self.gauth)
        return self._drive

    def _authenticate(self, timeout: int = 60) -> bool:
        """Google認証を実行

        Args:
            timeout: ブラウザ認証のタイムアウト秒数（デフォルト60秒）

        Returns:
            認証成功した場合True

        Raises:
            TimeoutError: 認証がタイムアウトした場合
            Exception: その他の認証エラー
        """
        debug_print("Google認証開始...")

        credentials_path = config_colab.GOOGLE_CREDENTIALS_PATH

        # 保存済みの認証情報があれば読み込み
        if os.path.exists(credentials_path):
            debug_print("保存済み認証情報を読み込み中...")
            try:
                self.gauth.LoadCredentialsFile(credentials_path)
            except Exception as e:
                debug_print(f"認証情報の読み込みに失敗: {e}")
                self.gauth.credentials = None

        if self.gauth.credentials is None:
            # 新規認証（ブラウザを開く）- タイムアウト付き
            debug_print(f"ブラウザでGoogle認証を開始... (タイムアウト: {timeout}秒)")
            self._browser_auth_with_timeout(timeout)
        elif self.gauth.access_token_expired:
            # トークン更新
            debug_print("アクセストークンを更新中...")
            try:
                self.gauth.Refresh()
            except Exception as e:
                debug_print(f"トークン更新に失敗、再認証します: {e}")
                self._browser_auth_with_timeout(timeout)
        else:
            # 既存の認証情報を使用
            debug_print("既存の認証情報を使用")
            self.gauth.Authorize()

        # 認証情報を保存
        self.gauth.SaveCredentialsFile(credentials_path)
        debug_print("認証完了")

        return True

    def _browser_auth_with_timeout(self, timeout: int):
        """タイムアウト付きブラウザ認証

        Args:
            timeout: タイムアウト秒数

        Raises:
            TimeoutError: 認証がタイムアウトした場合
        """
        import threading

        auth_result = {'success': False, 'error': None}

        def auth_thread():
            try:
                # まずLocalWebserverAuthを試す
                self.gauth.LocalWebserverAuth()
                auth_result['success'] = True
            except Exception as e:
                auth_result['error'] = e

        thread = threading.Thread(target=auth_thread, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # タイムアウト - サーバーを停止させる試み
            debug_print(f"認証がタイムアウトしました ({timeout}秒)")
            raise TimeoutError(
                f"Google認証がタイムアウトしました（{timeout}秒）。\n"
                "ブラウザで認証を完了してください。\n"
                "再度お試しください。"
            )

        if auth_result['error']:
            error_msg = str(auth_result['error'])
            # リダイレクトエラーの場合、手動認証を案内
            if 'code' in error_msg.lower() or 'redirect' in error_msg.lower():
                raise Exception(
                    f"リダイレクト認証に失敗しました。\n\n"
                    "Google Cloud Consoleで以下のリダイレクトURIを追加してください:\n"
                    "  http://localhost:8080/\n"
                    "  http://localhost:8090/\n"
                    "  http://localhost:8888/\n\n"
                    f"元のエラー: {error_msg}"
                )
            raise auth_result['error']

        if not auth_result['success']:
            raise Exception("認証が完了しませんでした")

    def test_connection(self) -> tuple:
        """接続テスト

        Returns:
            (success: bool, message: str)
        """
        debug_print("接続テスト開始...")
        try:
            # Google Driveにアクセスしてルートフォルダを取得
            file_list = self.drive.ListFile({
                'q': "'root' in parents and trashed=false",
                'maxResults': 1
            }).GetList()
            debug_print("接続成功")
            return True, "Google Driveへの接続に成功しました"
        except Exception as e:
            debug_print(f"接続失敗: {str(e)}")
            return False, f"接続失敗: {str(e)}"

    def get_or_create_folder(self) -> str:
        """転送先フォルダを取得または作成

        Returns:
            フォルダID
        """
        if self._folder_id:
            debug_print(f"既存のフォルダIDを使用: {self._folder_id}")
            return self._folder_id

        debug_print(f"フォルダを検索: {self.folder_name}")

        # フォルダを検索
        query = f"title='{self.folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        file_list = self.drive.ListFile({'q': query}).GetList()

        if file_list:
            self._folder_id = file_list[0]['id']
            debug_print(f"既存フォルダを発見: {self._folder_id}")
            return self._folder_id

        # フォルダを作成
        debug_print(f"フォルダを作成: {self.folder_name}")
        folder_metadata = {
            'title': self.folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = self.drive.CreateFile(folder_metadata)
        folder.Upload()
        self._folder_id = folder['id']
        debug_print(f"フォルダ作成完了: {self._folder_id}")

        return self._folder_id

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
            # 進捗更新の頻度を最適化（50ファイルごと、または最後）
            update_interval = max(50, total_files // 20)  # 最低50ファイルごと、または全体の5%ごと

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
                for i, (file_path, arcname) in enumerate(all_files):
                    # キャンセルチェック（50ファイルごとに確認）
                    if cancel_check and (i % 50 == 0) and cancel_check():
                        debug_print("ZIP圧縮がキャンセルされました")
                        zf.close()
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                        return False

                    # ファイルを追加
                    zf.write(file_path, arcname)

                    # 進捗コールバック（頻度を最適化）
                    current = i + 1
                    if progress_callback and (current % update_interval == 0 or current == total_files):
                        progress_callback(current, total_files, f"{current}/{total_files} ファイル")

                    if current % 100 == 0 or current == total_files:
                        debug_print(f"ZIP圧縮進捗: {current}/{total_files} ({current * 100 // total_files}%)")

            zip_size = os.path.getsize(zip_path)
            debug_print(f"ZIP圧縮完了: {zip_size / (1024*1024):.2f} MB")
            return True

        except Exception as e:
            debug_print(f"ZIP圧縮エラー: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise

    def upload_file(
        self,
        local_path: str,
        remote_name: str,
        folder_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """ファイルをGoogle Driveにアップロード

        Args:
            local_path: ローカルファイルパス
            remote_name: Google Drive上のファイル名
            folder_id: 親フォルダID
            progress_callback: 進捗コールバック(uploaded_bytes, total_bytes)

        Returns:
            アップロードしたファイルのID
        """
        file_size = os.path.getsize(local_path)
        debug_print(f"アップロード開始: {local_path} ({file_size / (1024*1024):.2f} MB)")

        try:
            # Google Driveファイルを作成
            gfile = self.drive.CreateFile({
                'title': remote_name,
                'parents': [{'id': folder_id}]
            })

            # ファイルをアップロード
            gfile.SetContentFile(local_path)
            gfile.Upload()

            file_id = gfile['id']
            debug_print(f"アップロード完了: {file_id}")

            # 完了コールバック
            if progress_callback:
                progress_callback(file_size, file_size)

            return file_id

        except Exception as e:
            debug_print(f"アップロードエラー: {e}")
            raise

    def generate_colab_notebook(
        self,
        data_file_id: str,
        data_file_name: str,
        output_path: str
    ) -> str:
        """Colabノートブックを生成

        Args:
            data_file_id: Google Drive上のデータファイルID
            data_file_name: データファイル名（ZIP）
            output_path: ノートブック出力パス

        Returns:
            生成したノートブックパス
        """
        debug_print(f"Colabノートブック生成中...")

        # train_model.ipynbを直接使用
        notebook_path = Path(__file__).parent.parent / "colab" / "train_model.ipynb"

        if notebook_path.exists():
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_json = json.load(f)

            # 設定セルを探してデータパスを更新
            for cell in notebook_json.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        source_text = ''.join(source)
                    else:
                        source_text = source

                    # 設定セル（FOLDER_NAMEとDATA_FILE_NAMEを含むセル）を検出
                    if 'FOLDER_NAME' in source_text and 'DATA_FILE_NAME' in source_text:
                        # 設定値を更新
                        new_source = source_text.replace(
                            'FOLDER_NAME = "annotation_data"',
                            f'FOLDER_NAME = "{self.folder_name}"'
                        )
                        # DATA_FILE_NAMEの行を置換（パターンマッチ）
                        import re
                        new_source = re.sub(
                            r'DATA_FILE_NAME = "[^"]*"',
                            f'DATA_FILE_NAME = "{data_file_name}"',
                            new_source
                        )
                        cell['source'] = new_source.split('\n')
                        cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                         for i, line in enumerate(cell['source'])]
                        debug_print(f"設定セルを更新: FOLDER_NAME={self.folder_name}, DATA_FILE_NAME={data_file_name}")
                        break
        else:
            # ipynbがない場合はエラー
            debug_print(f"警告: train_model.ipynbが見つかりません: {notebook_path}")
            raise FileNotFoundError(f"train_model.ipynbが見つかりません: {notebook_path}")

        # 出力
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_json, f, ensure_ascii=False, indent=2)

        debug_print(f"Colabノートブック生成完了: {output_path}")
        return output_path

    def upload_notebook_to_drive(self, notebook_path: str, folder_id: str) -> str:
        """ノートブックをGoogle Driveにアップロード

        Returns:
            アップロードしたファイルのColab URL
        """
        notebook_name = os.path.basename(notebook_path)

        gfile = self.drive.CreateFile({
            'title': notebook_name,
            'parents': [{'id': folder_id}],
            'mimeType': 'application/vnd.google.colaboratory'
        })
        gfile.SetContentFile(notebook_path)
        gfile.Upload()

        file_id = gfile['id']
        colab_url = f"https://colab.research.google.com/drive/{file_id}"

        debug_print(f"ノートブックアップロード完了: {colab_url}")
        return colab_url

    def _copy_common_modules(self, dest_dir: str):
        """共通モジュールをエクスポートディレクトリにコピー

        Args:
            dest_dir: コピー先ディレクトリ
        """
        # プロジェクトルートディレクトリ
        project_root = Path(__file__).parent.parent

        # コピーするファイル一覧
        files_to_copy = [
            "model_catalog.py",   # モデル定義
            "model_info.py",      # モデル情報（精度、パラメータ数等）
            "model_training.py",  # 学習ユーティリティ（EarlyStopping等）
        ]

        for filename in files_to_copy:
            src_path = project_root / filename
            if src_path.exists():
                dst_path = os.path.join(dest_dir, filename)
                shutil.copy2(str(src_path), dst_path)
                debug_print(f"共通モジュールをコピー: {filename}")
            else:
                debug_print(f"警告: {filename} が見つかりません: {src_path}")

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
        generate_notebook: bool = True,
        open_colab: bool = True,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> dict:
        """アノテーションをエクスポート → ZIP → アップロード → Notebook生成

        Args:
            annotations: アノテーション辞書
            inference_results: 推論結果辞書
            image_map: インデックスごとの画像パスマップ
            variant_keys: 画像バリアントのキー名
            zip_name: ZIPファイル名（Noneの場合は自動生成）
            deleted_indexes: 削除されたインデックス
            diff_vectors: 差分ベクトル
            waypoint_annotations: ウェイポイントアノテーション
            generate_notebook: Colabノートブックを生成するかどうか
            open_colab: 転送後にColabを開くかどうか
            progress_callback: 進捗コールバック(stage, current, total, message)
                stage: 'export', 'zip', 'upload', 'notebook'
            cancel_check: キャンセル確認コールバック

        Returns:
            {
                'success': bool,
                'zip_size': int,
                'file_id': str,
                'colab_url': str (generate_notebookがTrueの場合),
                'annotation_count': int,
                'error': str (失敗時のみ)
            }
        """
        debug_print("========== Colab転送処理開始 ==========")
        debug_print(f"アノテーション数: {len(annotations)}")

        # ZIPファイル名を生成
        if zip_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"annotation_{timestamp}.zip"

        if not zip_name.endswith('.zip'):
            zip_name += '.zip'

        debug_print(f"ZIPファイル名: {zip_name}")

        # 一時ディレクトリを作成
        temp_dir = tempfile.mkdtemp(prefix="colab_transfer_")
        export_dir = os.path.join(temp_dir, "export")
        zip_path = os.path.join(temp_dir, zip_name)
        debug_print(f"一時ディレクトリ: {temp_dir}")

        try:
            # ステージ1: エクスポート
            debug_print("--- ステージ1: エクスポート ---")
            if progress_callback:
                progress_callback('export', 0, 100, 'アノテーションをエクスポート中...')

            if cancel_check and cancel_check():
                return {'success': False, 'error': 'キャンセルされました'}

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

            if not manifest_path:
                return {'success': False, 'error': 'エクスポートに失敗しました'}

            # 共通コードファイルをエクスポートディレクトリにコピー
            self._copy_common_modules(export_dir)

            if progress_callback:
                progress_callback('export', 100, 100, 'エクスポート完了')

            # ステージ2: ZIP圧縮
            debug_print("--- ステージ2: ZIP圧縮 ---")
            if cancel_check and cancel_check():
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
                return {'success': False, 'error': 'ZIP圧縮がキャンセルされました'}

            zip_size = os.path.getsize(zip_path)

            # ステージ3: Google Driveにアップロード
            debug_print("--- ステージ3: アップロード ---")
            if cancel_check and cancel_check():
                return {'success': False, 'error': 'キャンセルされました'}

            if progress_callback:
                progress_callback('upload', 0, zip_size, 'Google Driveにアップロード中...')

            # フォルダを取得または作成
            folder_id = self.get_or_create_folder()

            def upload_progress(uploaded, total):
                if progress_callback:
                    progress_callback('upload', uploaded, total,
                                     f'アップロード中: {uploaded // (1024*1024)} MB / {total // (1024*1024)} MB')

            file_id = self.upload_file(
                local_path=zip_path,
                remote_name=zip_name,
                folder_id=folder_id,
                progress_callback=upload_progress
            )

            result = {
                'success': True,
                'zip_size': zip_size,
                'file_id': file_id,
                'folder_id': folder_id,
                'annotation_count': len(annotations)
            }

            # ステージ4: Notebookを生成
            if generate_notebook:
                debug_print("--- ステージ4: Notebook生成 ---")
                if progress_callback:
                    progress_callback('notebook', 0, 100, 'Colabノートブックを生成中...')

                notebook_name = zip_name.replace('.zip', '_training.ipynb')
                notebook_path = os.path.join(temp_dir, notebook_name)

                self.generate_colab_notebook(
                    data_file_id=file_id,
                    data_file_name=zip_name,
                    output_path=notebook_path
                )

                # NotebookをGoogle Driveにアップロード
                colab_url = self.upload_notebook_to_drive(notebook_path, folder_id)
                result['colab_url'] = colab_url

                if progress_callback:
                    progress_callback('notebook', 100, 100, 'Notebook生成完了')

                # Colabを開く
                if open_colab:
                    debug_print(f"Colabを開きます: {colab_url}")
                    webbrowser.open(colab_url)

            debug_print("========== Colab転送完了 ==========")
            return result

        except Exception as e:
            debug_print(f"転送エラー: {e}")
            import traceback
            debug_print(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }

        finally:
            # クリーンアップ
            debug_print(f"一時ディレクトリを削除: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                debug_print(f"一時ディレクトリの削除に失敗: {e}")

    def list_remote_files(self, folder_id: str = None, file_type: str = None) -> List[dict]:
        """Google Drive上のファイル一覧を取得

        Args:
            folder_id: フォルダID（Noneの場合はデフォルトフォルダ）
            file_type: ファイルタイプでフィルタ（'model', 'zip', 'notebook', None=すべて）

        Returns:
            ファイル情報のリスト
        """
        if folder_id is None:
            folder_id = self.get_or_create_folder()

        query = f"'{folder_id}' in parents and trashed=false"
        file_list = self.drive.ListFile({'q': query}).GetList()

        result = []
        for f in file_list:
            file_info = {
                'id': f['id'],
                'name': f['title'],
                'size': int(f.get('fileSize', 0)),
                'mimeType': f['mimeType'],
                'webViewLink': f.get('webViewLink', ''),
                'createdDate': f.get('createdDate', '')
            }

            # ファイルタイプフィルタ
            if file_type is None:
                result.append(file_info)
            elif file_type == 'model' and (f['title'].endswith('.pt') or f['title'].endswith('.pth') or f['title'].endswith('.onnx')):
                result.append(file_info)
            elif file_type == 'zip' and f['title'].endswith('.zip'):
                result.append(file_info)
            elif file_type == 'notebook' and f['title'].endswith('.ipynb'):
                result.append(file_info)

        # 作成日時でソート（新しい順）
        result.sort(key=lambda x: x['createdDate'], reverse=True)
        return result

    def download_file(
        self,
        file_id: str,
        local_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """Google Driveからファイルをダウンロード

        Args:
            file_id: ダウンロードするファイルのID
            local_path: ローカル保存先パス
            progress_callback: 進捗コールバック(downloaded_bytes, total_bytes)

        Returns:
            ダウンロードしたファイルのパス
        """
        debug_print(f"ダウンロード開始: {file_id} -> {local_path}")

        try:
            # ファイルを取得
            gfile = self.drive.CreateFile({'id': file_id})
            gfile.FetchMetadata()

            file_name = gfile['title']
            file_size = int(gfile.get('fileSize', 0))
            debug_print(f"ファイル名: {file_name}, サイズ: {file_size / (1024*1024):.2f} MB")

            # ダウンロード先ディレクトリを作成
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # ファイルをダウンロード
            gfile.GetContentFile(local_path)

            debug_print(f"ダウンロード完了: {local_path}")

            # 完了コールバック
            if progress_callback:
                actual_size = os.path.getsize(local_path)
                progress_callback(actual_size, actual_size)

            return local_path

        except Exception as e:
            debug_print(f"ダウンロードエラー: {e}")
            raise

    def download_model(
        self,
        model_name: str = None,
        local_dir: str = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Optional[str]:
        """学習済みモデルをダウンロード

        Args:
            model_name: モデルファイル名（Noneの場合は最新のモデルを取得）
            local_dir: ローカル保存先ディレクトリ（Noneの場合はmodels/）
            progress_callback: 進捗コールバック

        Returns:
            ダウンロードしたモデルのパス（見つからない場合はNone）
        """
        debug_print("モデルダウンロード開始...")

        # モデルファイル一覧を取得
        models = self.list_remote_files(file_type='model')

        if not models:
            debug_print("Google Driveにモデルファイルが見つかりません")
            return None

        debug_print(f"利用可能なモデル: {len(models)}件")
        for m in models[:5]:  # 最新5件を表示
            debug_print(f"  - {m['name']} ({m['size'] / (1024*1024):.2f} MB)")

        # ダウンロード対象を決定
        if model_name:
            target = next((m for m in models if m['name'] == model_name), None)
            if not target:
                debug_print(f"モデルが見つかりません: {model_name}")
                return None
        else:
            # 最新のモデルを取得
            target = models[0]
            debug_print(f"最新モデルを選択: {target['name']}")

        # ローカル保存先
        if local_dir is None:
            local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

        local_path = os.path.join(local_dir, target['name'])

        # ダウンロード実行
        return self.download_file(target['id'], local_path, progress_callback)

    def list_models(self) -> List[dict]:
        """Google Drive上のモデルファイル一覧を取得

        Returns:
            モデルファイル情報のリスト（新しい順）
        """
        return self.list_remote_files(file_type='model')

    def find_mlruns_folder(self, folder_id: str = None) -> Optional[str]:
        """Google Drive上のmlrunsフォルダを検索

        Args:
            folder_id: 親フォルダID（Noneの場合はデフォルトフォルダ）

        Returns:
            mlrunsフォルダのID（見つからない場合はNone）
        """
        if folder_id is None:
            folder_id = self.get_or_create_folder()

        query = f"title='mlruns' and mimeType='application/vnd.google-apps.folder' and '{folder_id}' in parents and trashed=false"
        file_list = self.drive.ListFile({'q': query}).GetList()

        if file_list:
            debug_print(f"mlrunsフォルダを発見: {file_list[0]['id']}")
            return file_list[0]['id']

        debug_print("mlrunsフォルダが見つかりません")
        return None

    def _collect_files_recursive(
        self,
        folder_id: str,
        local_dir: str,
        files_list: List[tuple]
    ):
        """フォルダ内のファイル一覧を再帰的に収集（ダウンロードはしない）

        Args:
            folder_id: Google DriveフォルダID
            local_dir: ローカル保存先ディレクトリ
            files_list: ファイル情報を追加するリスト [(file_id, local_path, file_size), ...]
        """
        # ローカルディレクトリを作成
        os.makedirs(local_dir, exist_ok=True)

        # フォルダ内のファイル一覧を取得
        query = f"'{folder_id}' in parents and trashed=false"
        items = self.drive.ListFile({'q': query}).GetList()

        for item in items:
            item_name = item['title']
            local_path = os.path.join(local_dir, item_name)

            if item['mimeType'] == 'application/vnd.google-apps.folder':
                # サブフォルダを再帰的にスキャン
                self._collect_files_recursive(item['id'], local_path, files_list)
            else:
                # ファイル情報を追加（サイズも含める）
                file_size = int(item.get('fileSize', 0))
                files_list.append((item['id'], local_path, file_size))

    def download_folder_recursive(
        self,
        folder_id: str,
        local_dir: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        skip_existing: bool = True,
        compare_dir: str = None
    ) -> dict:
        """フォルダをダウンロード（最適化版）

        Args:
            folder_id: Google DriveフォルダID
            local_dir: ローカル保存先ディレクトリ
            progress_callback: 進捗コールバック(filename, current, total)
            cancel_check: キャンセル確認コールバック
            skip_existing: 既存ファイルをスキップするかどうか
            compare_dir: スキップ判定に使う比較用ディレクトリ（Noneの場合はlocal_dir）

        Returns:
            dict: {
                'success': bool,
                'downloaded': int,  # ダウンロードしたファイル数
                'skipped': int,     # スキップしたファイル数
                'total': int        # 総ファイル数
            }
        """
        debug_print(f"フォルダダウンロード開始: {folder_id} -> {local_dir}")

        # まずファイル一覧を収集（フォルダ構造のスキャン）
        debug_print("ファイル一覧を収集中...")
        files_to_download = []
        self._collect_files_recursive(folder_id, local_dir, files_to_download)

        total_files = len(files_to_download)
        debug_print(f"ダウンロード対象ファイル数: {total_files}")

        if total_files == 0:
            debug_print("ダウンロードするファイルがありません")
            return {'success': True, 'downloaded': 0, 'skipped': 0, 'total': 0}

        downloaded_count = 0
        skipped_count = 0

        # 逐次ダウンロード（PyDrive2はスレッドセーフでないため）
        # ただし進捗表示は10ファイルごとに抑制
        for i, (file_id, local_path, remote_size) in enumerate(files_to_download):
            current = i + 1

            # キャンセルチェック（10ファイルごと）
            if cancel_check and (current % 10 == 0) and cancel_check():
                debug_print("ダウンロードがキャンセルされました")
                return {'success': False, 'downloaded': downloaded_count, 'skipped': skipped_count, 'total': total_files}

            # 既存ファイルのスキップチェック
            if skip_existing:
                # 比較用ディレクトリが指定されている場合は、相対パスを計算して比較
                if compare_dir:
                    rel_path = os.path.relpath(local_path, local_dir)
                    compare_path = os.path.join(compare_dir, rel_path)
                else:
                    compare_path = local_path

                if os.path.exists(compare_path):
                    compare_size = os.path.getsize(compare_path)
                    if compare_size == remote_size:
                        # ファイルサイズが一致 → スキップ
                        skipped_count += 1
                        continue

            try:
                gfile = self.drive.CreateFile({'id': file_id})
                gfile.GetContentFile(local_path)
                downloaded_count += 1
            except Exception as e:
                debug_print(f"ファイルダウンロードエラー: {os.path.basename(local_path)} - {e}")
                # 個別ファイルのエラーは警告として続行
                continue

            # 進捗コールバック（10ファイルごと、または最後）
            if progress_callback and (current % 10 == 0 or current == total_files):
                progress_callback(f"{current}/{total_files} ファイル", current, total_files)

            # コンソール出力も抑制（20%ごと）
            if current == total_files or (total_files >= 10 and current % max(1, total_files // 5) == 0):
                debug_print(f"ダウンロード進捗: {current}/{total_files} ({current * 100 // total_files}%)")

        # 結果サマリー
        if skipped_count > 0:
            debug_print(f"スキップ: {skipped_count}件 (既存ファイル)")
        debug_print(f"フォルダダウンロード完了: {local_dir} (ダウンロード: {downloaded_count}, スキップ: {skipped_count})")

        return {'success': True, 'downloaded': downloaded_count, 'skipped': skipped_count, 'total': total_files}

    def download_mlruns(
        self,
        local_dir: str = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        skip_existing: bool = True,
        compare_with_local: str = None
    ) -> Optional[dict]:
        """Google Driveからmlrunsフォルダをダウンロード

        Args:
            local_dir: ローカル保存先ディレクトリ（Noneの場合は一時ディレクトリ）
            progress_callback: 進捗コールバック(filename, current, total)
            cancel_check: キャンセル確認コールバック
            skip_existing: 既存ファイルをスキップするかどうか
            compare_with_local: ローカルのmlrunsディレクトリパス（スキップ判定に使用）

        Returns:
            dict: {
                'path': str,        # mlrunsフォルダのパス
                'downloaded': int,  # ダウンロードしたファイル数
                'skipped': int,     # スキップしたファイル数
                'total': int        # 総ファイル数
            }
            見つからない場合はNone
        """
        debug_print("mlrunsダウンロード開始...")

        # mlrunsフォルダを検索
        mlruns_folder_id = self.find_mlruns_folder()
        if not mlruns_folder_id:
            debug_print("mlrunsフォルダが見つかりません")
            return None

        # ローカル保存先
        if local_dir is None:
            local_dir = tempfile.mkdtemp(prefix="mlruns_download_")

        mlruns_local_path = os.path.join(local_dir, "mlruns")

        # 再帰的にダウンロード
        result = self.download_folder_recursive(
            mlruns_folder_id,
            mlruns_local_path,
            progress_callback,
            cancel_check,
            skip_existing,
            compare_dir=compare_with_local
        )

        if result['success']:
            debug_print(f"mlrunsダウンロード完了: {mlruns_local_path}")
            if result['skipped'] > 0 and result['downloaded'] == 0:
                debug_print(f"全ファイルが既にダウンロード済みです ({result['skipped']}件)")
            return {
                'path': mlruns_local_path,
                'downloaded': result['downloaded'],
                'skipped': result['skipped'],
                'total': result['total']
            }
        else:
            return None

    def list_mlruns_experiments(self) -> List[dict]:
        """Google Drive上のmlruns内の実験一覧を取得

        Returns:
            実験情報のリスト
            [{'id': folder_id, 'name': experiment_name, 'runs_count': int}, ...]
        """
        mlruns_folder_id = self.find_mlruns_folder()
        if not mlruns_folder_id:
            return []

        # mlruns直下のフォルダ（実験）を取得
        query = f"'{mlruns_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        experiments = self.drive.ListFile({'q': query}).GetList()

        result = []
        for exp in experiments:
            exp_id = exp['title']

            # 実験名を取得（meta.yamlから読むのは複雑なのでフォルダ名を使用）
            # 実験フォルダ内のrun数をカウント
            runs_query = f"'{exp['id']}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            runs = self.drive.ListFile({'q': runs_query}).GetList()

            result.append({
                'folder_id': exp['id'],
                'experiment_id': exp_id,
                'runs_count': len(runs)
            })

        debug_print(f"実験数: {len(result)}")
        for exp in result:
            debug_print(f"  実験ID: {exp['experiment_id']}, runs: {exp['runs_count']}")

        return result
