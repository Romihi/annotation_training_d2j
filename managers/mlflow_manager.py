import os
import sys
import subprocess
import mlflow
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox
import torch
from enum import Enum

class ModelType(Enum):
    """モデルタイプの定義"""
    AUTONOMOUS_DRIVING = "autonomous_driving"
    POSITION_ESTIMATION = "position_estimation"
    YOLO_DETECTION = "yolo_detection"
    YOLO_SEGMENTATION = "yolo_segmentation"

class MLflowManager:
    """実験別MLflow統合管理クラス"""
    
    # 実験名の定義
    EXPERIMENT_NAMES = {
        ModelType.AUTONOMOUS_DRIVING: "autonomous_driving_models",
        ModelType.POSITION_ESTIMATION: "position_estimation_models", 
        ModelType.YOLO_DETECTION: "yolo_detection_models",
        ModelType.YOLO_SEGMENTATION: "yolo_segmentation_models"
    }
    
    def __init__(self, folder_path=None):
        self.folder_path = folder_path
        self.tracking_uri = None
        self.current_experiment = None
        self.is_initialized = False
        
    def initialize(self, folder_path=None):
        """MLflowの初期化と設定を行う"""
        
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
                self.tracking_uri = f"file:///{normalized_path}"
            else:
                self.tracking_uri = f"file://{normalized_path}"
            
            print(f"MLflowトラッキングURI: {self.tracking_uri}")
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # 全ての実験を作成
            for model_type, experiment_name in self.EXPERIMENT_NAMES.items():
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
                    print(f"実験を作成: {experiment_name}")
            
            self.is_initialized = True
            print(f"MLflow初期化成功: {self.tracking_uri}")
            return True
            
        except Exception as e:
            print(f"MLflow初期化エラー: {e}")
            return False
    
    def set_experiment(self, model_type: ModelType):
        """指定されたモデルタイプの実験を設定"""
        if not self.is_initialized:
            if not self.initialize():
                return False
        
        experiment_name = self.EXPERIMENT_NAMES[model_type]
        try:
            mlflow.set_experiment(experiment_name)
            self.current_experiment = experiment_name
            
            # 環境変数も設定（YOLO用）
            os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
            os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
            
            print(f"実験を設定: {experiment_name}")
            return True
        except Exception as e:
            print(f"実験設定エラー: {e}")
            return False
    
    def open_ui(self, parent_widget=None, model_type: ModelType = None):
        """MLflow UIを開く"""
        
        if not self.is_initialized:
            if not self.initialize():
                if parent_widget:
                    QMessageBox.warning(parent_widget, "エラー", "MLflowの初期化に失敗しました。")
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
    
    def log_autonomous_driving_model(self, model_path, training_params, metrics, dataset_info):
        """自動運転モデルの学習結果を記録"""
        
        if not self.set_experiment(ModelType.AUTONOMOUS_DRIVING):
            return False
        
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
            "sampling_strategy": training_params.get("sampling_strategy", "all"),
            "augmentation_enabled": training_params.get("augmentation_enabled", False)
        }
        
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
            "status": metrics.get("status", "completed")
        }
        
        # MLflow実行名
        run_name = f"autonomous_driving_{training_params.get('model_type', 'unknown')}_{dataset_info.get('used_samples', 0)}samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name):
                # タグを設定
                mlflow.set_tags(tags)
                
                # パラメータをログ
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # データセット情報をログ
                for key, value in dataset_info.items():
                    if key == "input_shape" and isinstance(value, (tuple, list)):
                        # タプルは文字列に変換
                        mlflow.log_param(f"dataset_image_dims", f"{value[0]}x{value[1]}")
                    else:
                        mlflow.log_param(f"dataset_{key}", value)
                
                # メトリクスをログ
                for key, value in run_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # 学習曲線をログ（利用可能な場合）
                if "train_losses" in metrics and "val_losses" in metrics:
                    train_losses = metrics["train_losses"]
                    val_losses = metrics["val_losses"]
                    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                        mlflow.log_metric("train_loss", train_loss, step=epoch)
                        mlflow.log_metric("val_loss", val_loss, step=epoch)
                
                # モデルファイルを記録
                if model_path and os.path.exists(model_path):
                    if sys.platform.startswith('win'):
                        model_path = os.path.normpath(model_path)
                    mlflow.log_artifact(model_path, "model")
            
            print(f"自動運転モデルをMLflowに記録しました: {run_name}")
            return True
            
        except Exception as e:
            print(f"自動運転MLflow記録エラー: {e}")
            return False
    
    def log_position_estimation_model(self, model_path, training_params, metrics, dataset_info):
        """位置推論モデルの学習結果を記録"""
        
        if not self.set_experiment(ModelType.POSITION_ESTIMATION):
            return False
        
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
            "coordinate_type": training_params.get("coordinate_system", "classification")
        }
        
        # MLflow実行名
        run_name = f"position_estimation_{training_params.get('model_type', 'unknown')}_{dataset_info.get('used_samples', 0)}samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name):
                # タグを設定
                mlflow.set_tags(tags)
                
                # パラメータをログ
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # データセット情報をログ
                for key, value in dataset_info.items():
                    if key == "input_shape" and isinstance(value, (tuple, list)):
                        # タプルは文字列に変換
                        mlflow.log_param(f"dataset_image_dims", f"{value[0]}x{value[1]}")
                    elif key == "location_mapping" and isinstance(value, dict):
                        # 位置マッピング情報を文字列に変換
                        mlflow.log_param("dataset_location_mapping", str(value))
                    elif key == "unique_locations" and isinstance(value, list):
                        # ユニークな位置リストを文字列に変換
                        mlflow.log_param("dataset_unique_locations", ','.join(map(str, value)))
                    else:
                        mlflow.log_param(f"dataset_{key}", value)
                
                # メトリクスをログ
                for key, value in run_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # 学習曲線をログ（利用可能な場合）
                if "train_losses" in metrics and "val_losses" in metrics:
                    train_losses = metrics["train_losses"]
                    val_losses = metrics["val_losses"]
                    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                        mlflow.log_metric("train_loss", train_loss, step=epoch)
                        mlflow.log_metric("val_loss", val_loss, step=epoch)
                
                # 精度曲線をログ（利用可能な場合）
                if "train_accuracies" in metrics and "val_accuracies" in metrics:
                    train_accuracies = metrics["train_accuracies"]
                    val_accuracies = metrics["val_accuracies"]
                    for epoch, (train_acc, val_acc) in enumerate(zip(train_accuracies, val_accuracies)):
                        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                
                # モデルファイルを記録
                if model_path and os.path.exists(model_path):
                    if sys.platform.startswith('win'):
                        model_path = os.path.normpath(model_path)
                    mlflow.log_artifact(model_path, "model")
            
            print(f"位置推論モデルをMLflowに記録しました: {run_name}")
            return True
            
        except Exception as e:
            print(f"位置推論MLflow記録エラー: {e}")
            return False

    def log_yolo_model(self, model_type, results, training_params, dataset_info):
        """YOLO検出モデルの学習結果を記録"""
        
        if not self.set_experiment(ModelType.YOLO_DETECTION):
            return False
        
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
            "task_type": "detection"
        }
        
        run_name = f"yolo_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags(tags)
                
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                for key, value in dataset_info.items():
                    mlflow.log_param(f"dataset_{key}", value)
                
                for key, value in run_metrics.items():
                    mlflow.log_metric(key, value)
                
                # YOLOの結果ファイルを記録
                if hasattr(results, 'save_dir'):
                    weights_dir = os.path.join(results.save_dir, "weights")
                    if os.path.exists(weights_dir):
                        for weight_file in os.listdir(weights_dir):
                            weight_path = os.path.join(weights_dir, weight_file)
                            if os.path.isfile(weight_path):
                                mlflow.log_artifact(weight_path, "weights")
            
            return True
        except Exception as e:
            print(f"YOLO MLflow記録エラー: {e}")
            return False
    
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
        """YOLOセグメンテーションモデルの学習結果を記録"""
        
        if not self.set_experiment(ModelType.YOLO_SEGMENTATION):
            return False
        
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
            "status": "completed"
        }
        
        # MLflow実行名
        run_name = f"yolo_segmentation_{training_params.get('model_type', 'yolo')}_{dataset_info.get('train_samples', 0)}samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name):
                # タグを設定
                mlflow.set_tags(tags)
                
                # パラメータをログ
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # データセット情報をログ
                for key, value in dataset_info.items():
                    if key == "classes" and isinstance(value, list):
                        # クラスリストを文字列に変換
                        mlflow.log_param("dataset_classes", ','.join(map(str, value)))
                    elif key == "task_type":
                        # タスクタイプを記録
                        mlflow.log_param("dataset_task_type", value)
                    else:
                        mlflow.log_param(f"dataset_{key}", value)
                
                # メトリクスをログ
                for key, value in run_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # 学習曲線をログ（利用可能な場合）
                if "train_losses" in metrics and isinstance(metrics["train_losses"], list):
                    train_losses = metrics["train_losses"]
                    for epoch, loss in enumerate(train_losses):
                        mlflow.log_metric("train_loss", loss, step=epoch)
                
                if "val_losses" in metrics and isinstance(metrics["val_losses"], list):
                    val_losses = metrics["val_losses"]
                    for epoch, loss in enumerate(val_losses):
                        mlflow.log_metric("val_loss", loss, step=epoch)
                
                # mAP曲線をログ（利用可能な場合）
                if "box_mAP_history" in metrics and isinstance(metrics["box_mAP_history"], list):
                    box_mAP_history = metrics["box_mAP_history"]
                    for epoch, mAP in enumerate(box_mAP_history):
                        mlflow.log_metric("box_mAP_epoch", mAP, step=epoch)
                
                if "mask_mAP_history" in metrics and isinstance(metrics["mask_mAP_history"], list):
                    mask_mAP_history = metrics["mask_mAP_history"]
                    for epoch, mAP in enumerate(mask_mAP_history):
                        mlflow.log_metric("mask_mAP_epoch", mAP, step=epoch)
                
                # モデルファイルを記録
                if model_path and os.path.exists(model_path):
                    if sys.platform.startswith('win'):
                        model_path = os.path.normpath(model_path)
                    mlflow.log_artifact(model_path, "model")
                
                # 追加のアーティファクト（設定ファイルなど）
                if "yaml_file" in dataset_info and os.path.exists(dataset_info["yaml_file"]):
                    mlflow.log_artifact(dataset_info["yaml_file"], "config")
            
            print(f"YOLOセグメンテーションモデルをMLflowに記録しました: {run_name}")
            return True
            
        except Exception as e:
            print(f"YOLOセグメンテーションMLflow記録エラー: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _log_run(self, model_category, params, metrics, tags, model_path, dataset_info):
        """共通のMLflow記録処理"""
        
        run_name = f"{model_category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name):
                # タグを設定
                mlflow.set_tags(tags)
                
                # パラメータをログ
                for key, value in params.items():
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