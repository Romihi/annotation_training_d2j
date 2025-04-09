import os
import torch
import json
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QFileDialog, QComboBox, 
                             QSpinBox, QCheckBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

class ObjectDetectionAnnotationTool:
    def __init__(self, main_window):
        self.main_window = main_window
        self.current_annotations = {}
        self.detection_mode = False
        self.selected_classes = []
        self.current_model = None

    def toggle_object_detection_mode(self, state):
        """切り替えオブジェクト検知モード"""
        self.detection_mode = state
        if state and not self.current_model:
            # デフォルトモデルをロード
            try:
                self.current_model = YOLO('yolov8n.pt')
            except Exception as e:
                print(f"モデルのロードに失敗: {e}")
                return False
        return True

    def load_detection_model(self):
        """カスタムYOLOモデルをロードするダイアログ"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, 
            "YOLOモデルを選択", 
            "", 
            "PyTorch モデル (*.pt);;すべてのファイル (*)"
        )
        if file_path:
            try:
                self.current_model = YOLO(file_path)
                return True
            except Exception as e:
                print(f"モデルのロードに失敗: {e}")
                return False

    def run_object_detection(self, image_path):
        """指定された画像に対してオブジェクト検知を実行"""
        if not self.current_model or not self.detection_mode:
            return []

        try:
            results = self.current_model(image_path)
            annotations = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # クラス情報
                    cls = int(box.cls[0])
                    class_name = self.current_model.names[cls]
                    
                    # バウンディングボックス情報（正規化座標）
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    
                    annotations.append({
                        'class': class_name,
                        'class_id': cls,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(box.conf[0])
                    })
            
            return annotations
        except Exception as e:
            print(f"物体検知中にエラー: {e}")
            return []

    def train_yolo_model(self):
        """YOLO物体検知モデルの学習用ダイアログ"""
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("YOLO物体検知モデル学習")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        
        # データセットパス選択
        dataset_layout = QHBoxLayout()
        dataset_label = QLabel("データセットパス:")
        self.dataset_path_input = QLineEdit()
        dataset_browse_button = QPushButton("参照...")
        dataset_browse_button.clicked.connect(self.browse_dataset)
        
        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(self.dataset_path_input)
        dataset_layout.addWidget(dataset_browse_button)
        layout.addLayout(dataset_layout)
        
        # モデル選択
        model_layout = QHBoxLayout()
        model_label = QLabel("ベースモデル:")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolov8n.pt", 
            "yolov8s.pt", 
            "yolov8m.pt", 
            "yolov8l.pt", 
            "yolov8x.pt"
        ])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # トレーニングパラメータ
        epochs_layout = QHBoxLayout()
        epochs_label = QLabel("学習エポック数:")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # データ拡張オプション
        augmentation_check = QCheckBox("データ拡張を有効化")
        augmentation_check.setChecked(True)
        layout.addWidget(augmentation_check)
        
        # クラス選択（オプション）
        class_group = QButtonGroup()
        all_classes_radio = QRadioButton("全クラスを学習")
        custom_classes_radio = QRadioButton("カスタムクラスを選択")
        all_classes_radio.setChecked(True)
        
        layout.addWidget(all_classes_radio)
        layout.addWidget(custom_classes_radio)
        
        # 学習実行ボタン
        train_button = QPushButton("モデル学習開始")
        train_button.clicked.connect(self.start_yolo_training)
        layout.addWidget(train_button)
        
        dialog.exec_()

    def browse_dataset(self):
        """YOLOデータセットのパスを選択"""
        folder_path = QFileDialog.getExistingDirectory(
            self.main_window, 
            "YOLOデータセットフォルダを選択"
        )
        if folder_path:
            self.dataset_path_input.setText(folder_path)

    def start_yolo_training(self):
        """YOLO学習の実際の実行"""
        dataset_path = self.dataset_path_input.text()
        model_name = self.model_combo.currentText()
        epochs = self.epochs_spin.value()
        
        if not os.path.exists(dataset_path):
            print("有効なデータセットパスを選択してください")
            return
        
        try:
            model = YOLO(model_name)
            results = model.train(
                data=dataset_path, 
                epochs=epochs, 
                imgsz=640, 
                augment=True
            )
            
            print("学習完了:", results)
            
        except Exception as e:
            print(f"学習中にエラーが発生: {e}")

    def render_object_annotations(self, image_path, annotations):
        """物体検知アノテーションを画像にレンダリング"""
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        
        for annotation in annotations:
            bbox = annotation['bbox']
            class_name = annotation['class']
            
            # 座標を画像サイズに変換
            x1, y1, x2, y2 = [
                int(bbox[0] * width), 
                int(bbox[1] * height), 
                int(bbox[2] * width), 
                int(bbox[3] * height)
            ]
            
            # 枠線を描画
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ラベルを描画
            label = f"{class_name}"
            cv2.putText(
                img, label, (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
        
        return img

