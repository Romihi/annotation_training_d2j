import os
import sys
import subprocess
import mlflow
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox, QApplication
import torch
from enum import Enum

# Databricks設定をインポート
try:
    from config_databricks import (
        DATABRICKS_ENABLED,
        DATABRICKS_HOST,
        DATABRICKS_TOKEN,
        DATABRICKS_EXPERIMENT_PREFIX,
        validate_databricks_config,
        get_databricks_status
    )
    DATABRICKS_CONFIG_AVAILABLE = True
except ImportError:
    DATABRICKS_CONFIG_AVAILABLE = False
    DATABRICKS_ENABLED = False

class ModelType(Enum):
    """モデルタイプの定義"""
    AUTONOMOUS_DRIVING = "autonomous_driving"
    POSITION_ESTIMATION = "position_estimation"
    WAYPOINT_REGRESSION = "waypoint_regression"
    YOLO_DETECTION = "yolo_detection"
    YOLO_SEGMENTATION = "yolo_segmentation"

class MLflowManager:
    """実験別MLflow統合管理クラス（Databricks対応・ローカル併用記録）"""

    # 実験名の定義
    EXPERIMENT_NAMES = {
        ModelType.AUTONOMOUS_DRIVING: "autonomous_driving_models",
        ModelType.POSITION_ESTIMATION: "position_estimation_models",
        ModelType.WAYPOINT_REGRESSION: "waypoint_regression_models",
        ModelType.YOLO_DETECTION: "yolo_detection_models",
        ModelType.YOLO_SEGMENTATION: "yolo_segmentation_models"
    }

    def __init__(self, folder_path=None, use_databricks=None):
        self.folder_path = folder_path
        self.tracking_uri = None
        self.local_tracking_uri = None  # ローカル用URI（常に保持）
        self.current_experiment = None
        self.is_initialized = False

        # Databricks使用フラグ（Noneの場合はconfig_databricks.pyの設定を使用）
        if use_databricks is None:
            self.use_databricks = DATABRICKS_ENABLED if DATABRICKS_CONFIG_AVAILABLE else False
        else:
            self.use_databricks = use_databricks

        self._databricks_connected = False
        self._local_initialized = False

    def set_databricks_mode(self, enabled: bool):
        """Databricksモードを切り替える（再初期化が必要）"""
        if self.use_databricks != enabled:
            self.use_databricks = enabled
            self.is_initialized = False
            self._databricks_connected = False
            print(f"Databricksモード: {'有効' if enabled else '無効'}")

    def get_backend_info(self) -> dict:
        """現在のバックエンド情報を取得"""
        if self.use_databricks and self._databricks_connected:
            return {
                "type": "databricks+local",
                "host": DATABRICKS_HOST if DATABRICKS_CONFIG_AVAILABLE else "",
                "tracking_uri": self.tracking_uri,
                "local_tracking_uri": self.local_tracking_uri,
                "status": "接続済み（ローカル併用）"
            }
        elif self.use_databricks and not self._databricks_connected:
            return {
                "type": "databricks",
                "host": DATABRICKS_HOST if DATABRICKS_CONFIG_AVAILABLE else "",
                "tracking_uri": None,
                "status": "未接続"
            }
        else:
            return {
                "type": "local",
                "host": "localhost",
                "tracking_uri": self.local_tracking_uri,
                "status": "ローカル"
            }

    def _initialize_databricks(self, parent_widget=None):
        """Databricks MLflowの初期化"""
        if not DATABRICKS_CONFIG_AVAILABLE:
            print("警告: config_databricks.py が見つかりません")
            return False

        # 設定の検証
        errors = validate_databricks_config()
        if errors:
            print(f"Databricks設定エラー: {errors}")
            return False

        try:
            # Databricks認証情報を環境変数に設定
            os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
            os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

            # MLflowのトラッキングURIをDatabricksに設定
            self.tracking_uri = "databricks"
            mlflow.set_tracking_uri(self.tracking_uri)

            # 親ディレクトリの存在確認と作成
            missing_dirs = self._check_databricks_directories()
            if missing_dirs:
                # ユーザーに確認
                if not self._ask_create_directories(missing_dirs, parent_widget):
                    print("ユーザーがディレクトリ作成をキャンセルしました")
                    return False

                # ディレクトリを作成
                if not self._create_databricks_directories(missing_dirs):
                    print("Databricksディレクトリの作成に失敗しました")
                    return False

            # 実験を作成（Databricksワークスペース内）
            for model_type, experiment_name in self.EXPERIMENT_NAMES.items():
                # Databricksでは実験パスを使用
                experiment_path = f"{DATABRICKS_EXPERIMENT_PREFIX}/{experiment_name}"
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_path)
                    if experiment is None:
                        mlflow.create_experiment(experiment_path)
                        print(f"Databricks実験を作成: {experiment_path}")
                except Exception as exp_error:
                    print(f"実験作成警告 ({experiment_path}): {exp_error}")

            self._databricks_connected = True
            self.is_initialized = True
            print(f"Databricks MLflow初期化成功: {DATABRICKS_HOST}")
            return True

        except Exception as e:
            print(f"Databricks MLflow初期化エラー: {e}")
            self._databricks_connected = False
            return False

    def _check_databricks_directories(self):
        """Databricksの親ディレクトリが存在するか確認"""
        missing_dirs = set()

        for model_type, experiment_name in self.EXPERIMENT_NAMES.items():
            experiment_path = f"{DATABRICKS_EXPERIMENT_PREFIX}/{experiment_name}"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_path)
                if experiment is None:
                    # 実験が存在しない場合、作成を試みてエラーを確認
                    try:
                        mlflow.create_experiment(experiment_path)
                        # 成功したら削除（後で正式に作成する）
                        # Note: MLflowでは実験の削除は soft delete のため、そのまま残す
                    except Exception as create_error:
                        error_msg = str(create_error)
                        if "RESOURCE_DOES_NOT_EXIST" in error_msg and "Parent directory" in error_msg:
                            # 親ディレクトリが存在しない
                            missing_dirs.add(DATABRICKS_EXPERIMENT_PREFIX)
            except Exception as e:
                error_msg = str(e)
                if "RESOURCE_DOES_NOT_EXIST" in error_msg:
                    missing_dirs.add(DATABRICKS_EXPERIMENT_PREFIX)

        return list(missing_dirs)

    def _ask_create_directories(self, missing_dirs, parent_widget=None):
        """ユーザーにディレクトリ作成を確認"""
        if not missing_dirs:
            return True

        # 親ウィジェットがない場合はアプリケーションのアクティブウィンドウを使用
        if parent_widget is None:
            parent_widget = QApplication.activeWindow()

        dirs_text = "\n".join(f"  - {d}" for d in missing_dirs)
        message = (
            f"Databricksワークスペースに以下のディレクトリが存在しません:\n\n"
            f"{dirs_text}\n\n"
            f"ディレクトリを作成しますか？"
        )

        reply = QMessageBox.question(
            parent_widget,
            "Databricksディレクトリ作成",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        return reply == QMessageBox.Yes

    def _create_databricks_directories(self, directories):
        """Databricks Workspace APIを使用してディレクトリを作成"""
        try:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.service.workspace import ImportFormat

            # WorkspaceClientを初期化（環境変数から認証情報を取得）
            w = WorkspaceClient(
                host=DATABRICKS_HOST,
                token=DATABRICKS_TOKEN
            )

            for dir_path in directories:
                try:
                    # Workspace APIでディレクトリを作成
                    # mkdirs は再帰的にディレクトリを作成する
                    w.workspace.mkdirs(dir_path)
                    print(f"Databricksディレクトリを作成しました: {dir_path}")
                except Exception as e:
                    print(f"ディレクトリ作成エラー ({dir_path}): {e}")
                    return False

            return True

        except ImportError:
            print("databricks-sdk がインストールされていません。pip install databricks-sdk を実行してください。")
            # SDK がない場合はメッセージを表示
            parent_widget = QApplication.activeWindow()
            if parent_widget:
                QMessageBox.warning(
                    parent_widget,
                    "Databricks SDK未インストール",
                    "Databricks SDKがインストールされていないため、ディレクトリを自動作成できません。\n\n"
                    "以下のコマンドでインストールしてください:\n"
                    "pip install databricks-sdk\n\n"
                    "または、Databricksワークスペースで手動でディレクトリを作成してください:\n"
                    f"{', '.join(directories)}"
                )
            return False
        except Exception as e:
            print(f"Databricksディレクトリ作成エラー: {e}")
            return False

    def _initialize_local(self, folder_path=None):
        """ローカルMLflowの初期化"""
        if folder_path:
            self.folder_path = folder_path

        if not self.folder_path:
            print("警告: 画像フォルダが設定されていません。MLflowの初期化ができません。")
            return False

        try:
            # MLflow用のディレクトリを作成
            mlflow_dir = self.folder_path
            os.makedirs(mlflow_dir, exist_ok=True)

            # パスの正規化
            normalized_path = os.path.normpath(mlflow_dir).replace('\\', '/')

            # Windows環境での正しいURI形式を構築
            if sys.platform.startswith('win'):
                self.local_tracking_uri = f"file:///{normalized_path}"
            else:
                self.local_tracking_uri = f"file://{normalized_path}"

            # メインのtracking_uriも設定
            self.tracking_uri = self.local_tracking_uri

            print(f"ローカルMLflowトラッキングURI: {self.local_tracking_uri}")
            mlflow.set_tracking_uri(self.local_tracking_uri)

            # 全ての実験を作成
            for model_type, experiment_name in self.EXPERIMENT_NAMES.items():
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
                    print(f"ローカル実験を作成: {experiment_name}")

            self._local_initialized = True
            print(f"ローカルMLflow初期化成功: {self.local_tracking_uri}")
            return True

        except Exception as e:
            print(f"ローカルMLflow初期化エラー: {e}")
            return False

    def initialize(self, folder_path=None, parent_widget=None):
        """MLflowの初期化と設定を行う（ローカル併用記録対応）

        Args:
            folder_path: MLflowのトラッキングディレクトリ
            parent_widget: ダイアログ表示用の親ウィジェット（Databricksディレクトリ作成確認用）
        """

        if folder_path:
            self.folder_path = folder_path

        # 常にローカルを初期化
        local_result = self._initialize_local(folder_path)

        # Databricksモードの場合は追加で初期化
        if self.use_databricks:
            databricks_result = self._initialize_databricks(parent_widget)
            if not databricks_result:
                print("Databricks接続に失敗しました。ローカルのみに記録します。")
                self._databricks_connected = False

        # ローカルに戻しておく
        if self.local_tracking_uri:
            mlflow.set_tracking_uri(self.local_tracking_uri)

        self.is_initialized = local_result
        return local_result
    
    def set_experiment(self, model_type: ModelType, target: str = "local"):
        """指定されたモデルタイプの実験を設定

        Args:
            model_type: モデルタイプ
            target: "local" または "databricks"
        """
        if not self.is_initialized:
            if not self.initialize():
                return False

        experiment_name = self.EXPERIMENT_NAMES[model_type]

        if target == "databricks" and self.use_databricks and self._databricks_connected:
            # Databricks用
            experiment_path = f"{DATABRICKS_EXPERIMENT_PREFIX}/{experiment_name}"
            mlflow.set_tracking_uri("databricks")
        else:
            # ローカル用
            experiment_path = experiment_name
            mlflow.set_tracking_uri(self.local_tracking_uri)

        try:
            mlflow.set_experiment(experiment_path)
            self.current_experiment = experiment_path

            # 環境変数も設定（YOLO用）
            current_uri = "databricks" if target == "databricks" else self.local_tracking_uri
            os.environ["MLFLOW_TRACKING_URI"] = current_uri
            os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_path

            print(f"実験を設定 ({target}): {experiment_path}")
            return True
        except Exception as e:
            print(f"実験設定エラー ({target}): {e}")
            return False

    def _log_run_to_target(self, target: str, model_type: ModelType, run_name: str,
                           params: dict, run_metrics: dict, tags: dict,
                           dataset_info: dict, metrics: dict, model_path: str):
        """指定されたターゲット（local/databricks）にログを記録"""
        try:
            if not self.set_experiment(model_type, target):
                return False

            with mlflow.start_run(run_name=run_name):
                # タグを設定
                mlflow.set_tags(tags)

                # パラメータをログ（Noneは除外）
                for key, value in params.items():
                    if value is not None:
                        mlflow.log_param(key, value)

                # データセット情報をログ
                for key, value in dataset_info.items():
                    if key == "input_shape" and isinstance(value, (tuple, list)):
                        mlflow.log_param(f"dataset_image_dims", f"{value[0]}x{value[1]}")
                    elif key == "location_mapping" and isinstance(value, dict):
                        mlflow.log_param("dataset_location_mapping", str(value))
                    elif key == "unique_locations" and isinstance(value, list):
                        mlflow.log_param("dataset_unique_locations", ','.join(map(str, value)))
                    elif key == "classes" and isinstance(value, list):
                        mlflow.log_param("dataset_classes", ','.join(map(str, value)))
                    else:
                        mlflow.log_param(f"dataset_{key}", value)

                # メトリクスをログ
                for key, value in run_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)

                # 学習曲線をログ
                if "train_losses" in metrics and "val_losses" in metrics:
                    for epoch, (train_loss, val_loss) in enumerate(zip(metrics["train_losses"], metrics["val_losses"])):
                        mlflow.log_metric("train_loss", train_loss, step=epoch)
                        mlflow.log_metric("val_loss", val_loss, step=epoch)

                # 精度曲線をログ（利用可能な場合）
                if "train_accuracies" in metrics and "val_accuracies" in metrics:
                    for epoch, (train_acc, val_acc) in enumerate(zip(metrics["train_accuracies"], metrics["val_accuracies"])):
                        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

                # モデルファイルを記録
                if model_path and os.path.exists(model_path):
                    if sys.platform.startswith('win'):
                        model_path = os.path.normpath(model_path)
                    mlflow.log_artifact(model_path, "model")

            print(f"モデルを{target}に記録しました: {run_name}")
            return True

        except Exception as e:
            print(f"{target}への記録エラー: {e}")
            return False

    def _log_with_local_fallback(self, model_type: ModelType, run_name: str,
                                  params: dict, run_metrics: dict, tags: dict,
                                  dataset_info: dict, metrics: dict, model_path: str):
        """ローカルに記録し、Databricks有効時は追加でDatabricksにも記録"""
        # まずローカルに記録（必須）
        local_success = self._log_run_to_target(
            "local", model_type, run_name, params, run_metrics, tags,
            dataset_info, metrics, model_path
        )

        # Databricks有効時は追加で記録
        databricks_success = False
        if self.use_databricks and self._databricks_connected:
            databricks_success = self._log_run_to_target(
                "databricks", model_type, run_name, params, run_metrics, tags,
                dataset_info, metrics, model_path
            )
            if databricks_success:
                print(f"Databricksにも記録しました: {run_name}")
            else:
                print(f"Databricksへの記録に失敗しましたが、ローカルには記録済みです")

        # ローカルに戻す
        mlflow.set_tracking_uri(self.local_tracking_uri)

        return local_success
    
    def open_ui(self, parent_widget=None, model_type: ModelType = None):
        """MLflow UIを開く"""

        if not self.is_initialized:
            if not self.initialize():
                if parent_widget:
                    QMessageBox.warning(parent_widget, "エラー", "MLflowの初期化に失敗しました。")
                return

        # Databricksモードの場合はブラウザでDatabricksワークスペースを開く
        if self.use_databricks and self._databricks_connected:
            self._open_databricks_ui(parent_widget, model_type)
            return

        try:
            # 特定の実験を指定した場合、その実験にフォーカス
            experiment_filter = ""
            if model_type:
                experiment_name = self.EXPERIMENT_NAMES[model_type]
                # MLflow UIでは実験IDでフィルタリング
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    experiment_filter = f" --default-artifact-root {experiment.artifact_location}"

            # 環境に応じてコマンドを構築
            if sys.platform.startswith('win'):  # Windows
                cmd = f'start cmd /k "mlflow ui --backend-store-uri {self.tracking_uri}{experiment_filter}"'
                print(f"実行コマンド: {cmd}")
                subprocess.Popen(cmd, shell=True)
            else:  # Mac/Linux
                cmd = f'mlflow ui --backend-store-uri {self.tracking_uri}{experiment_filter}'
                subprocess.Popen(cmd, shell=True)

            if parent_widget:
                message = "MLflow UIを起動しました。ブラウザで http://localhost:5000 にアクセスして実験結果を確認できます。"
                if model_type:
                    message += f"\n\n現在の実験: {self.EXPERIMENT_NAMES[model_type]}"
                message += "\n\nUIを終了するには、コマンドウィンドウを閉じてください。"

                QMessageBox.information(parent_widget, "MLflow UI", message)

        except Exception as e:
            error_msg = str(e)
            print(f"MLflow UI起動エラー: {error_msg}")

            if parent_widget:
                QMessageBox.critical(
                    parent_widget,
                    "エラー",
                    f"MLflow UIの起動に失敗しました: {error_msg}\n\n"
                    "MLflowがインストールされているか確認してください: pip install mlflow"
                )

    def _open_databricks_ui(self, parent_widget=None, model_type: ModelType = None):
        """DatabricksのMLflow UIをブラウザで開く"""
        import webbrowser

        try:
            # DatabricksワークスペースのMLflow実験URLを構築
            base_url = DATABRICKS_HOST.rstrip('/')

            if model_type:
                experiment_name = self.EXPERIMENT_NAMES[model_type]
                experiment_path = f"{DATABRICKS_EXPERIMENT_PREFIX}/{experiment_name}"
                # 実験IDを取得してURLを構築
                experiment = mlflow.get_experiment_by_name(experiment_path)
                if experiment:
                    url = f"{base_url}/#mlflow/experiments/{experiment.experiment_id}"
                else:
                    url = f"{base_url}/#mlflow/experiments"
            else:
                url = f"{base_url}/#mlflow/experiments"

            print(f"Databricks MLflow UIを開く: {url}")
            webbrowser.open(url)

            if parent_widget:
                message = f"Databricks MLflow UIをブラウザで開きました。\n\nURL: {url}"
                if model_type:
                    message += f"\n\n実験: {self.EXPERIMENT_NAMES[model_type]}"
                QMessageBox.information(parent_widget, "Databricks MLflow", message)

        except Exception as e:
            error_msg = str(e)
            print(f"Databricks UI起動エラー: {error_msg}")

            if parent_widget:
                QMessageBox.critical(
                    parent_widget,
                    "エラー",
                    f"Databricks MLflow UIを開けませんでした: {error_msg}"
                )
    
    def log_autonomous_driving_model(self, model_path, training_params, metrics, dataset_info):
        """自動運転モデルの学習結果を記録（ローカル併用記録対応）"""

        # 基本パラメータ
        params = {
            "framework": "pytorch",
            "model_type": training_params.get("model_type", "autonomous_driving"),
            "data_folder": training_params.get("data_folder", "unknown"),
            "epochs": training_params.get("num_epochs", 0),
            "completed_epochs": training_params.get("completed_epochs", 0),
            "learning_rate": training_params.get("learning_rate", 0.001),
            "batch_size": training_params.get("batch_size", 32),
            "early_stopping": "enabled" if training_params.get("use_early_stopping", False) else "disabled",
            "patience": training_params.get("patience", 0),
            "initial_weights": training_params.get("initial_weights", "pretrained"),
            "pretrained_model_name": training_params.get("pretrained_model_name", None),
            "sampling_strategy": training_params.get("sampling_strategy", "all"),
            "augmentation_enabled": training_params.get("augmentation_enabled", False)
        }

        # コメントがあれば追加
        if training_params.get("comment"):
            params["comment"] = training_params["comment"]

        # オーグメンテーション詳細パラメータ
        aug_params = training_params.get("augmentation_params", {})
        if aug_params.get("enabled", False):
            params.update({
                "aug_flip": aug_params.get("use_flip", False),
                "aug_flip_prob": aug_params.get("flip_prob", 0.0),
                "aug_color": aug_params.get("use_color", False),
                "aug_brightness": aug_params.get("brightness", 0.0),
                "aug_contrast": aug_params.get("contrast", 0.0),
                "aug_saturation": aug_params.get("saturation", 0.0),
                "aug_geometry": aug_params.get("use_geometry", False),
                "aug_rotation": aug_params.get("rotation_degrees", 0),
                "aug_translate": aug_params.get("translate_ratio", 0.0),
                "aug_erase": aug_params.get("use_erase", False),
                "aug_erase_prob": aug_params.get("erase_prob", 0.0)
            })

        # 自動運転特有のメトリクス
        run_metrics = {
            "best_val_loss": metrics.get("best_val_loss", 0.0),
            "final_train_loss": metrics.get("final_train_loss", 0.0),
            "final_val_loss": metrics.get("final_val_loss", 0.0)
        }

        # 自動運転特有のメトリクス（利用可能な場合）
        if "steering_accuracy" in metrics:
            run_metrics["steering_accuracy"] = metrics["steering_accuracy"]
        if "throttle_accuracy" in metrics:
            run_metrics["throttle_accuracy"] = metrics["throttle_accuracy"]
        if "steering_mae" in metrics:
            run_metrics["steering_mae"] = metrics["steering_mae"]
        if "throttle_mae" in metrics:
            run_metrics["throttle_mae"] = metrics["throttle_mae"]

        # タグ
        tags = {
            "model_category": "autonomous_driving",
            "task_type": "regression",
            "framework": "pytorch",
            "status": metrics.get("status", "completed"),
            "training_environment": training_params.get("training_environment", "local")  # local, colab, databricks
        }

        # MLflow実行名（カスタムモデル名が指定されていればそれを使用）
        custom_name = training_params.get('model_name', '')
        if custom_name:
            run_name = custom_name
        else:
            run_name = f"autonomous_driving_{training_params.get('model_type', 'unknown')}_{dataset_info.get('used_samples', 0)}samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ローカル併用記録を使用
        return self._log_with_local_fallback(
            ModelType.AUTONOMOUS_DRIVING, run_name, params, run_metrics, tags,
            dataset_info, metrics, model_path
        )
    
    def log_position_estimation_model(self, model_path, training_params, metrics, dataset_info):
        """位置推論モデルの学習結果を記録（ローカル併用記録対応）"""

        # 基本パラメータ
        params = {
            "framework": "pytorch",
            "model_type": training_params.get("model_type", "position_estimation"),
            "data_folder": training_params.get("data_folder", "unknown"),
            "task_type": "classification",  # 位置推論は分類タスク
            "epochs": training_params.get("num_epochs", 0),
            "completed_epochs": training_params.get("completed_epochs", 0),
            "learning_rate": training_params.get("learning_rate", 0.001),
            "batch_size": training_params.get("batch_size", 32),
            "early_stopping": "enabled" if training_params.get("use_early_stopping", False) else "disabled",
            "patience": training_params.get("patience", 0),
            "augmentation_enabled": training_params.get("augmentation_enabled", False),
            "coordinate_system": training_params.get("coordinate_system", "classification"),
            "estimation_method": training_params.get("estimation_method", "cnn_classification"),
            "fixed_classes": training_params.get("fixed_classes", 8),
            "actual_classes": training_params.get("actual_classes", 0)
        }

        # コメントがあれば追加
        if training_params.get("comment"):
            params["comment"] = training_params["comment"]

        # 位置推論特有のメトリクス
        run_metrics = {
            "best_val_loss": metrics.get("best_val_loss", 0.0),
            "best_val_acc": metrics.get("best_val_acc", 0.0),
            "final_train_loss": metrics.get("final_train_loss", 0.0),
            "final_val_loss": metrics.get("final_val_loss", 0.0),
            "final_train_acc": metrics.get("final_train_acc", 0.0),
            "final_val_acc": metrics.get("final_val_acc", 0.0)
        }

        # 分類精度関連のメトリクス（利用可能な場合）
        if "position_error_mean" in metrics:
            run_metrics["position_error_mean"] = metrics["position_error_mean"]
        if "position_error_std" in metrics:
            run_metrics["position_error_std"] = metrics["position_error_std"]
        if "convergence_rate" in metrics:
            run_metrics["convergence_rate"] = metrics["convergence_rate"]

        # タグ
        tags = {
            "model_category": "position_estimation",
            "task_type": "classification",
            "framework": "pytorch",
            "status": metrics.get("status", "completed"),
            "coordinate_type": training_params.get("coordinate_system", "classification"),
            "training_environment": training_params.get("training_environment", "local")  # local, colab, databricks
        }

        # MLflow実行名（カスタムモデル名が指定されていればそれを使用）
        custom_name = training_params.get('model_name', '')
        if custom_name:
            run_name = custom_name
        else:
            run_name = f"position_estimation_{training_params.get('model_type', 'unknown')}_{dataset_info.get('used_samples', 0)}samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ローカル併用記録を使用
        return self._log_with_local_fallback(
            ModelType.POSITION_ESTIMATION, run_name, params, run_metrics, tags,
            dataset_info, metrics, model_path
        )

    def log_yolo_model(self, model_type, results, training_params, dataset_info):
        """YOLO検出モデルの学習結果を記録（ローカル併用記録対応）"""

        params = {
            "framework": "yolo",
            "model_type": model_type,
            "data_folder": training_params.get("data_folder", "unknown"),
            "epochs": training_params.get("epochs", 0),
            "batch_size": training_params.get("batch_size", 16),
            "img_size": training_params.get("img_size", 640),
            "learning_rate": training_params.get("learning_rate", 0.001),
            "augmentation_enabled": training_params.get("augmentation_enabled", False),
            "mosaic": training_params.get("mosaic", 0.0),
            "fliplr": training_params.get("fliplr", 0.0)
        }

        # コメントがあれば追加
        if training_params.get("comment"):
            params["comment"] = training_params["comment"]

        # YOLO特有のメトリクス
        run_metrics = {}
        if hasattr(results, 'maps') and results.maps is not None:
            if isinstance(results.maps, (list, tuple)) and len(results.maps) > 0:
                run_metrics["mAP_50"] = float(results.maps[0]) if results.maps[0] is not None else 0.0
                if len(results.maps) > 1:
                    run_metrics["mAP_50_95"] = float(results.maps[1]) if results.maps[1] is not None else 0.0

        if hasattr(results, 'box') and results.box is not None:
            if hasattr(results.box, 'map'):
                run_metrics["box_mAP"] = float(results.box.map)
            if hasattr(results.box, 'map50'):
                run_metrics["box_mAP_50"] = float(results.box.map50)

        tags = {
            "model_category": "object_detection",
            "framework": "yolo",
            "task_type": "detection",
            "training_environment": training_params.get("training_environment", "local")  # local, colab, databricks
        }

        run_name = f"yolo_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # YOLOの重みファイルパスを取得
        weights_path = None
        if hasattr(results, 'save_dir'):
            weights_dir = os.path.join(results.save_dir, "weights")
            best_path = os.path.join(weights_dir, "best.pt")
            if os.path.exists(best_path):
                weights_path = best_path

        # ローカル併用記録を使用
        return self._log_yolo_with_local_fallback(
            ModelType.YOLO_DETECTION, run_name, params, run_metrics, tags,
            dataset_info, results, weights_path
        )

    def _log_yolo_with_local_fallback(self, model_type: ModelType, run_name: str,
                                       params: dict, run_metrics: dict, tags: dict,
                                       dataset_info: dict, results, weights_path: str):
        """YOLO用のローカル併用記録"""

        def log_yolo_run(target: str):
            try:
                if not self.set_experiment(model_type, target):
                    return False

                with mlflow.start_run(run_name=run_name):
                    mlflow.set_tags(tags)

                    for key, value in params.items():
                        if value is not None:
                            mlflow.log_param(key, value)

                    for key, value in dataset_info.items():
                        mlflow.log_param(f"dataset_{key}", value)

                    for key, value in run_metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)

                    # YOLOの結果ファイルを記録
                    if hasattr(results, 'save_dir'):
                        weights_dir = os.path.join(results.save_dir, "weights")
                        if os.path.exists(weights_dir):
                            for weight_file in os.listdir(weights_dir):
                                weight_path = os.path.join(weights_dir, weight_file)
                                if os.path.isfile(weight_path):
                                    mlflow.log_artifact(weight_path, "weights")

                print(f"YOLOモデルを{target}に記録しました: {run_name}")
                return True
            except Exception as e:
                print(f"YOLO {target}記録エラー: {e}")
                return False

        # ローカルに記録
        local_success = log_yolo_run("local")

        # Databricks有効時は追加で記録
        if self.use_databricks and self._databricks_connected:
            databricks_success = log_yolo_run("databricks")
            if databricks_success:
                print(f"Databricksにも記録しました: {run_name}")

        # ローカルに戻す
        mlflow.set_tracking_uri(self.local_tracking_uri)

        return local_success
    
    # def log_yolo_segmentation_model(self, model_path, training_params, metrics, dataset_info):
    #     """セグメンテーションモデルの学習結果を記録"""
        
    #     if not self.set_experiment(ModelType.YOLO_SEGMENTATION):
    #         return False
        
    #     params = {
    #         "framework": "pytorch",
    #         "model_type": "yolo_segmentation",
    #         "architecture": training_params.get("architecture", "unet"),
    #         "epochs": training_params.get("num_epochs", 0),
    #         "learning_rate": training_params.get("learning_rate", 0.001),
    #         "batch_size": training_params.get("batch_size", 32),
    #         "num_classes": training_params.get("num_classes", 2),
    #         "loss_function": training_params.get("loss_function", "dice_loss")
    #     }
        
    #     # セグメンテーション特有のメトリクス
    #     run_metrics = {
    #         "dice_coefficient": metrics.get("dice_coefficient", 0.0),
    #         "iou_score": metrics.get("iou_score", 0.0),
    #         "pixel_accuracy": metrics.get("pixel_accuracy", 0.0),
    #         "mean_iou": metrics.get("mean_iou", 0.0),
    #         "final_loss": metrics.get("final_loss", 0.0),
    #         "validation_loss": metrics.get("validation_loss", 0.0)
    #     }
        
    #     tags = {
    #         "model_category": "yolo_segmentation",
    #         "task_type": "yolo_segmentation",
    #         "architecture": training_params.get("architecture", "unet")
    #     }
        
    #     return self._log_run("yolo_segmentation", params, run_metrics, tags, model_path, dataset_info)
    
    def log_yolo_segmentation_model(self, model_path, training_params, metrics, dataset_info):
        """YOLOセグメンテーションモデルの学習結果を記録（ローカル併用記録対応）"""

        # 基本パラメータ
        params = {
            "framework": training_params.get("framework", "yolo"),
            "model_type": training_params.get("model_type", "yolo_segmentation"),
            "data_folder": training_params.get("data_folder", "unknown"),
            "architecture": training_params.get("architecture", "yolo_segmentation"),
            "epochs": training_params.get("epochs", 0),
            "batch_size": training_params.get("batch_size", 16),
            "img_size": training_params.get("img_size", 640),
            "learning_rate": training_params.get("learning_rate", 0.001),
            "patience": training_params.get("patience", 0),
            "initial_weights": training_params.get("initial_weights", "pretrained"),
            "augmentation_enabled": training_params.get("augmentation_enabled", False),
            "num_classes": training_params.get("num_classes", dataset_info.get("num_classes", 0)),
            "loss_function": training_params.get("loss_function", "yolo_segmentation_loss"),
            "task_type": "segmentation"
        }

        # コメントがあれば追加
        if training_params.get("comment"):
            params["comment"] = training_params["comment"]

        # オーグメンテーションパラメータ（有効な場合のみ記録）
        if training_params.get("augmentation_enabled", False):
            params.update({
                "aug_mosaic": training_params.get("mosaic", 0.0),
                "aug_fliplr": training_params.get("fliplr", 0.0),
                "aug_hsv_h": training_params.get("hsv_h", 0.0),
                "aug_hsv_s": training_params.get("hsv_s", 0.0),
                "aug_hsv_v": training_params.get("hsv_v", 0.0),
                "aug_translate": training_params.get("translate", 0.0),
                "aug_scale": training_params.get("scale", 0.0),
                "aug_erasing": training_params.get("erasing", 0.0)
            })

        # YOLOセグメンテーション特有のメトリクス
        run_metrics = {
            "box_mAP": metrics.get("box_mAP", 0.0),
            "mask_mAP": metrics.get("mask_mAP", 0.0),
            "final_loss": metrics.get("final_loss", 0.0)
        }

        # 追加のセグメンテーションメトリクス（利用可能な場合）
        if "dice_coefficient" in metrics:
            run_metrics["dice_coefficient"] = metrics["dice_coefficient"]
        if "iou_score" in metrics:
            run_metrics["iou_score"] = metrics["iou_score"]
        if "pixel_accuracy" in metrics:
            run_metrics["pixel_accuracy"] = metrics["pixel_accuracy"]
        if "mean_iou" in metrics:
            run_metrics["mean_iou"] = metrics["mean_iou"]
        if "validation_loss" in metrics:
            run_metrics["validation_loss"] = metrics["validation_loss"]

        # YOLOの結果から追加メトリクスを取得（利用可能な場合）
        if "mAP_50" in metrics:
            run_metrics["mAP_50"] = metrics["mAP_50"]
        if "mAP_50_95" in metrics:
            run_metrics["mAP_50_95"] = metrics["mAP_50_95"]

        # タグ
        tags = {
            "model_category": "yolo_segmentation",
            "task_type": "segmentation",
            "framework": "yolo",
            "architecture": training_params.get("architecture", "yolo_segmentation"),
            "status": "completed",
            "training_environment": training_params.get("training_environment", "local")  # local, colab, databricks
        }

        # MLflow実行名（カスタムモデル名が指定されていればそれを使用）
        custom_name = training_params.get('model_name', '')
        if custom_name:
            run_name = custom_name
        else:
            run_name = f"yolo_segmentation_{training_params.get('model_type', 'yolo')}_{dataset_info.get('train_samples', 0)}samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ローカル併用記録を使用
        return self._log_with_local_fallback(
            ModelType.YOLO_SEGMENTATION, run_name, params, run_metrics, tags,
            dataset_info, metrics, model_path
        )

    def _log_run(self, model_category, params, metrics, tags, model_path, dataset_info):
        """共通のMLflow記録処理"""
        
        run_name = f"{model_category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name):
                # タグを設定
                mlflow.set_tags(tags)
                
                # パラメータをログ（Noneは除外）
                for key, value in params.items():
                    if value is not None:
                        mlflow.log_param(key, value)
                
                # データセット情報をログ
                for key, value in dataset_info.items():
                    mlflow.log_param(f"dataset_{key}", value)
                
                # メトリクスをログ
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # モデルファイルを記録
                if model_path and os.path.exists(model_path):
                    if sys.platform.startswith('win'):
                        model_path = os.path.normpath(model_path)
                    mlflow.log_artifact(model_path, "model")
            
            print(f"{model_category}モデルをMLflowに記録しました: {run_name}")
            return True
            
        except Exception as e:
            print(f"{model_category} MLflow記録エラー: {e}")
            return False
    
    def get_experiment_runs(self, model_type: ModelType):
        """指定されたモデルタイプの実験Run一覧を取得"""
        if not self.is_initialized:
            if not self.initialize():
                return []
        
        try:
            experiment_name = self.EXPERIMENT_NAMES[model_type]
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                return runs
            return []
        except Exception as e:
            print(f"Run取得エラー: {e}")
            return []
    
    def compare_models_by_type(self, parent_widget, model_type: ModelType):
        """指定されたモデルタイプの実験結果を比較"""
        self.open_ui(parent_widget, model_type)

    def log_waypoint_regression_model(self, model_path, training_params, metrics, dataset_info):
        """ウェイポイント回帰モデルの学習結果を記録（ローカル併用記録対応）"""

        # 基本パラメータ
        params = {
            "framework": "pytorch",
            "model_type": training_params.get("model_type", "waypoint_regression"),
            "data_folder": training_params.get("data_folder", "unknown"),
            "task_type": "regression",
            "epochs": training_params.get("num_epochs", 0),
            "completed_epochs": training_params.get("completed_epochs", 0),
            "learning_rate": training_params.get("learning_rate", 0.001),
            "batch_size": training_params.get("batch_size", 8),
            "early_stopping": "enabled" if training_params.get("use_early_stopping", False) else "disabled",
            "patience": training_params.get("patience", 0),
            "augmentation_enabled": training_params.get("use_augmentation", False),
            "num_waypoints": training_params.get("num_waypoints", 4),
            "output_format": "xy_coordinates",
            "coordinate_system": "continuous"
        }

        # コメントがあれば追加
        if training_params.get("comment"):
            params["comment"] = training_params["comment"]

        # ウェイポイント回帰特有のメトリクス
        run_metrics = {
            "best_val_loss": metrics.get("best_val_loss", 0.0),
            "final_train_loss": metrics.get("final_train_loss", 0.0),
            "final_val_loss": metrics.get("final_val_loss", 0.0),
            "total_training_time": metrics.get("total_training_time", 0.0),
            "avg_epoch_time": metrics.get("avg_epoch_time", 0.0),
            "completed_epochs": metrics.get("completed_epochs", 0)
        }

        # タグ
        tags = {
            "model_category": "waypoint_regression",
            "task_type": "regression",
            "framework": "pytorch",
            "status": metrics.get("status", "completed"),
            "waypoint_count": str(training_params.get("num_waypoints", 4)),
            "training_environment": training_params.get("training_environment", "local")  # local, colab, databricks
        }

        # データセット情報の追加
        if dataset_info:
            params.update({
                "train_samples": dataset_info.get("train_samples", 0),
                "val_samples": dataset_info.get("val_samples", 0),
                "total_samples": dataset_info.get("total_annotations", 0),
                "used_samples": dataset_info.get("used_samples", 0)
            })

        # MLflow実行名（カスタムモデル名が指定されていればそれを使用）
        custom_name = training_params.get('model_name', '')
        if custom_name:
            run_name = custom_name
        else:
            run_name = f"waypoint_{training_params.get('model_type', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ローカル併用記録を使用
        success = self._log_with_local_fallback(
            ModelType.WAYPOINT_REGRESSION, run_name, params, run_metrics, tags,
            dataset_info if dataset_info else {}, metrics, model_path
        )

        if success:
            return {"status": "success", "run_name": run_name}
        else:
            return {"status": "error", "message": "記録に失敗しました"}

    def _get_default_mlflow_uri(self):
        """デフォルトのmlrunsディレクトリURIを取得"""
        try:
            from config import mlflow_dir
            normalized_path = os.path.normpath(mlflow_dir).replace('\\', '/')
            if sys.platform.startswith('win'):
                return f"file:///{normalized_path}"
            else:
                return f"file://{normalized_path}"
        except ImportError:
            return self.local_tracking_uri

    def sync_local_to_databricks(self, parent_widget=None, progress_callback=None, cancel_check=None, delete_orphaned=False):
        """ローカルのMLflow記録をDatabricksに同期する

        Args:
            parent_widget: 進捗ダイアログ表示用の親ウィジェット
            progress_callback: 進捗コールバック関数 (current, total, message)
            cancel_check: キャンセル確認関数（Trueを返すとキャンセル）
            delete_orphaned: Trueの場合、ローカルに存在しないDatabricks上のRunを削除

        Returns:
            dict: 同期結果 {"synced": int, "skipped": int, "failed": int, "deleted": int, "errors": list, "cancelled": bool}
        """
        if not self.use_databricks:
            return {"synced": 0, "skipped": 0, "failed": 0, "deleted": 0, "errors": ["Databricksモードが無効です"], "cancelled": False}

        # Databricks接続を確認
        if not self._databricks_connected:
            if not self._initialize_databricks(parent_widget):
                return {"synced": 0, "skipped": 0, "failed": 0, "deleted": 0, "errors": ["Databricksに接続できません"], "cancelled": False}

        result = {"synced": 0, "skipped": 0, "failed": 0, "deleted": 0, "errors": [], "cancelled": False}

        # デフォルトのmlrunsディレクトリを使用
        mlruns_uri = self._get_default_mlflow_uri()

        try:
            # ローカルのトラッキングURIに切り替え
            mlflow.set_tracking_uri(mlruns_uri)
            print(f"同期元: {mlruns_uri}")

            runs_to_sync = []
            local_run_names = {}  # {experiment_name: set(run_names)}

            # 各実験のRunを収集
            for model_type, experiment_name in self.EXPERIMENT_NAMES.items():
                local_run_names[experiment_name] = set()
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment:
                        runs = mlflow.search_runs(
                            experiment_ids=[experiment.experiment_id],
                            order_by=["start_time DESC"]
                        )
                        print(f"実験 {experiment_name}: {len(runs)} runs")
                        for _, run in runs.iterrows():
                            run_name = run.get('tags.mlflow.runName', '')
                            if run_name:
                                local_run_names[experiment_name].add(run_name)
                            runs_to_sync.append({
                                "model_type": model_type,
                                "experiment_name": experiment_name,
                                "run": run
                            })
                except Exception as e:
                    print(f"ローカル実験取得エラー ({experiment_name}): {e}")

            total_runs = len(runs_to_sync)

            # 各Runを同期
            for i, run_info in enumerate(runs_to_sync):
                # キャンセルチェック
                if cancel_check and cancel_check():
                    print("同期がキャンセルされました")
                    result["cancelled"] = True
                    break

                if progress_callback:
                    progress_callback(i + 1, total_runs + (1 if delete_orphaned else 0),
                                     f"同期中: {run_info['run'].get('tags.mlflow.runName', 'unknown')}")

                sync_result = self._sync_run_to_databricks(
                    run_info["model_type"],
                    run_info["experiment_name"],
                    run_info["run"],
                    mlruns_uri
                )

                if sync_result == "synced":
                    result["synced"] += 1
                elif sync_result == "skipped":
                    result["skipped"] += 1
                else:
                    result["failed"] += 1
                    result["errors"].append(f"{run_info['run'].get('tags.mlflow.runName', 'unknown')}: {sync_result}")

            # 削除同期（ローカルに存在しないDatabricks上のRunを削除）
            if delete_orphaned and not result["cancelled"]:
                if progress_callback:
                    progress_callback(total_runs, total_runs + 1, "不要なRunを削除中...")

                delete_result = self._delete_orphaned_databricks_runs(local_run_names, cancel_check)
                result["deleted"] = delete_result.get("deleted", 0)
                result["errors"].extend(delete_result.get("errors", []))

            if total_runs == 0 and not delete_orphaned:
                result["message"] = "同期するRunがありません"

            return result

        except Exception as e:
            result["errors"].append(str(e))
            return result
        finally:
            # ローカルに戻す
            if self.local_tracking_uri:
                mlflow.set_tracking_uri(self.local_tracking_uri)

    def _delete_orphaned_databricks_runs(self, local_run_names, cancel_check=None):
        """ローカルに存在しないDatabricks上のRunを削除

        Args:
            local_run_names: {experiment_name: set(run_names)} ローカルに存在するRun名
            cancel_check: キャンセル確認関数

        Returns:
            dict: {"deleted": int, "errors": list}
        """
        result = {"deleted": 0, "errors": []}

        try:
            mlflow.set_tracking_uri("databricks")
            client = mlflow.tracking.MlflowClient()

            for experiment_name, local_names in local_run_names.items():
                if cancel_check and cancel_check():
                    break

                experiment_path = f"{DATABRICKS_EXPERIMENT_PREFIX}/{experiment_name}"
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_path)
                    if not experiment:
                        continue

                    # Databricks上のRunを取得
                    databricks_runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id]
                    )

                    for _, db_run in databricks_runs.iterrows():
                        if cancel_check and cancel_check():
                            break

                        run_name = db_run.get('tags.mlflow.runName', '')
                        run_id = db_run.get('run_id', '')

                        # ローカルに存在しないRunを削除
                        if run_name and run_name not in local_names:
                            try:
                                client.delete_run(run_id)
                                result["deleted"] += 1
                                print(f"削除: {run_name} (実験: {experiment_name})")
                            except Exception as del_error:
                                result["errors"].append(f"削除失敗 {run_name}: {del_error}")

                except Exception as e:
                    result["errors"].append(f"実験 {experiment_name} の処理エラー: {e}")

        except Exception as e:
            result["errors"].append(f"削除同期エラー: {e}")

        return result

    def get_orphaned_runs_count(self):
        """ローカルに存在しないDatabricks上のRun数を取得"""
        if not self.use_databricks or not self._databricks_connected:
            return 0

        mlruns_uri = self._get_default_mlflow_uri()
        orphaned_count = 0

        try:
            # ローカルのRun名を収集
            mlflow.set_tracking_uri(mlruns_uri)
            local_run_names = {}

            for model_type, experiment_name in self.EXPERIMENT_NAMES.items():
                local_run_names[experiment_name] = set()
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment:
                        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                        for _, run in runs.iterrows():
                            run_name = run.get('tags.mlflow.runName', '')
                            if run_name:
                                local_run_names[experiment_name].add(run_name)
                except Exception:
                    pass

            # Databricks上のRunと比較
            mlflow.set_tracking_uri("databricks")

            for experiment_name, local_names in local_run_names.items():
                experiment_path = f"{DATABRICKS_EXPERIMENT_PREFIX}/{experiment_name}"
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_path)
                    if experiment:
                        databricks_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                        for _, db_run in databricks_runs.iterrows():
                            run_name = db_run.get('tags.mlflow.runName', '')
                            if run_name and run_name not in local_names:
                                orphaned_count += 1
                except Exception:
                    pass

        except Exception as e:
            print(f"孤立Run数取得エラー: {e}")
        finally:
            if self.local_tracking_uri:
                mlflow.set_tracking_uri(self.local_tracking_uri)

        return orphaned_count

    def _sync_run_to_databricks(self, model_type: ModelType, experiment_name: str, local_run, mlruns_uri=None):
        """個別のRunをDatabricksに同期"""
        try:
            run_name = local_run.get("tags.mlflow.runName", "")
            run_id = local_run.get("run_id", "")
            # ローカルURIを設定
            local_uri = mlruns_uri or self.local_tracking_uri

            if not run_name:
                return "skipped"

            # Databricksに切り替え
            mlflow.set_tracking_uri("databricks")
            experiment_path = f"{DATABRICKS_EXPERIMENT_PREFIX}/{experiment_name}"

            try:
                mlflow.set_experiment(experiment_path)
            except Exception as exp_error:
                print(f"Databricks実験設定エラー: {exp_error}")
                return f"実験設定エラー: {exp_error}"

            # 既存のRunを確認（同名のRunがあればスキップ）
            try:
                existing_runs = mlflow.search_runs(
                    experiment_names=[experiment_path],
                    filter_string=f"tags.mlflow.runName = '{run_name}'"
                )
                if len(existing_runs) > 0:
                    print(f"スキップ（既存）: {run_name}")
                    return "skipped"
            except Exception:
                pass  # 検索失敗は無視して続行

            # 新しいRunを作成
            with mlflow.start_run(run_name=run_name):
                # パラメータをコピー
                for col in local_run.index:
                    if col.startswith("params."):
                        param_name = col.replace("params.", "")
                        value = local_run[col]
                        if value is not None and str(value) != "nan":
                            try:
                                mlflow.log_param(param_name, value)
                            except Exception:
                                pass

                # メトリクスをコピー
                for col in local_run.index:
                    if col.startswith("metrics."):
                        metric_name = col.replace("metrics.", "")
                        value = local_run[col]
                        if value is not None and str(value) != "nan":
                            try:
                                mlflow.log_metric(metric_name, float(value))
                            except Exception:
                                pass

                # タグをコピー（システムタグ以外）
                for col in local_run.index:
                    if col.startswith("tags.") and not col.startswith("tags.mlflow."):
                        tag_name = col.replace("tags.", "")
                        value = local_run[col]
                        if value is not None and str(value) != "nan":
                            try:
                                mlflow.set_tag(tag_name, value)
                            except Exception:
                                pass

                # 同期元情報をタグとして追加
                mlflow.set_tag("synced_from_local", "true")
                mlflow.set_tag("original_run_id", run_id)
                mlflow.set_tag("sync_timestamp", datetime.now().isoformat())

                # アーティファクトをコピー（ローカルに戻してからパスを取得）
                mlflow.set_tracking_uri(local_uri)
                try:
                    local_client = mlflow.tracking.MlflowClient()
                    artifacts = local_client.list_artifacts(run_id)

                    for artifact in artifacts:
                        artifact_path = local_client.download_artifacts(run_id, artifact.path)
                        # Databricksに切り替えてアップロード
                        mlflow.set_tracking_uri("databricks")
                        if os.path.isdir(artifact_path):
                            mlflow.log_artifacts(artifact_path, artifact.path)
                        else:
                            mlflow.log_artifact(artifact_path, os.path.dirname(artifact.path) or None)
                except Exception as artifact_error:
                    print(f"アーティファクト同期警告: {artifact_error}")

            print(f"同期完了: {run_name}")
            return "synced"

        except Exception as e:
            print(f"Run同期エラー ({run_name}): {e}")
            return str(e)

    def get_sync_status(self):
        """ローカルとDatabricksの同期状態を取得"""
        status = {
            "local_runs": 0,
            "databricks_runs": 0,
            "unsynced_runs": 0,
            "experiments": {}
        }

        # デフォルトのmlrunsディレクトリを使用
        mlruns_uri = self._get_default_mlflow_uri()

        try:
            # ローカルのRun数をカウント
            mlflow.set_tracking_uri(mlruns_uri)
            print(f"同期状態確認: {mlruns_uri}")
            for model_type, experiment_name in self.EXPERIMENT_NAMES.items():
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment:
                        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                        local_count = len(runs)
                        status["local_runs"] += local_count
                        status["experiments"][experiment_name] = {"local": local_count, "databricks": 0}
                        print(f"  {experiment_name}: {local_count} runs")
                except Exception as e:
                    print(f"  {experiment_name}: エラー - {e}")

            # Databricks有効時はDatabricksのRun数もカウント
            if self.use_databricks and self._databricks_connected:
                mlflow.set_tracking_uri("databricks")
                for model_type, experiment_name in self.EXPERIMENT_NAMES.items():
                    experiment_path = f"{DATABRICKS_EXPERIMENT_PREFIX}/{experiment_name}"
                    try:
                        experiment = mlflow.get_experiment_by_name(experiment_path)
                        if experiment:
                            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                            databricks_count = len(runs)
                            status["databricks_runs"] += databricks_count
                            if experiment_name in status["experiments"]:
                                status["experiments"][experiment_name]["databricks"] = databricks_count
                    except Exception:
                        pass

            # 未同期のRun数を計算（簡易版：ローカル - Databricks）
            status["unsynced_runs"] = max(0, status["local_runs"] - status["databricks_runs"])

        except Exception as e:
            print(f"同期状態取得エラー: {e}")
        finally:
            if self.local_tracking_uri:
                mlflow.set_tracking_uri(self.local_tracking_uri)

        return status