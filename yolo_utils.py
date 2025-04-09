# yolo_utils.py
"""YOLOv8を使用した物体検知のユーティリティ関数"""
import os
import json
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
import cv2
from pathlib import Path
from PyQt5.QtWidgets import (QLabel)
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PyQt5.QtCore import Qt, QRect

# Ultralytics YOLOv8 インポート
try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics がインストールされていません。pip install ultralytics でインストールしてください。")

# 物体カテゴリ定義
DEFAULT_CLASSES = ["traffic_cone", "person", "car", "bicycle", "motorcycle", "truck", "bus", "stop_sign", "parking_meter"]

def get_yolo_model(model_path=None, pretrained=True):
    """
    YOLOモデルを読み込む
    Args:
        model_path: カスタムモデルのパス
        pretrained: 事前学習済みモデルを使用するかどうか
    Returns:
        YOLOモデル
    """
    if model_path and os.path.exists(model_path):
        try:
            model = YOLO(model_path)
            print(f"カスタムYOLOモデルを読み込みました: {model_path}")
            return model
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            
    if pretrained:
        # 事前学習済みのyolov8nを使用
        model = YOLO("yolov8n.pt")
        print("事前学習済みYOLOv8nモデルを読み込みました")
        return model
    
    return None

def detect_objects(image_path, model=None, conf_threshold=0.25):
    """
    画像内の物体を検出する
    Args:
        image_path: 画像のパス
        model: YOLOモデル
        conf_threshold: 信頼度のしきい値
    Returns:
        検出結果のリスト [{'class': クラス名, 'bbox': [x1, y1, x2, y2], 'confidence': 信頼度}, ...]
    """
    if model is None:
        model = get_yolo_model()
    
    if not os.path.exists(image_path):
        print(f"画像が見つかりません: {image_path}")
        return []
    
    try:
        # 画像を読み込み
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # 推論実行
        results = model(image_path, conf=conf_threshold)[0]
        
        # 結果を整形
        detections = []
        for i, det in enumerate(results.boxes.data):
            x1, y1, x2, y2, conf, cls = det.tolist()
            
            # クラスIDからクラス名を取得
            class_name = results.names[int(cls)]
            
            # 結果を追加
            detections.append({
                'class': class_name,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf)
            })
        
        return detections
    
    except Exception as e:
        print(f"検出エラー: {e}")
        return []

def train_yolo_model(dataset_dir, epochs=50, batch_size=16, img_size=640, save_dir=None, pretrained=True):
    """
    YOLOv8モデルを学習する
    Args:
        dataset_dir: データセットディレクトリ (YOLO形式)
        epochs: エポック数
        batch_size: バッチサイズ
        img_size: 画像サイズ
        save_dir: モデル保存ディレクトリ
        pretrained: 事前学習済みモデルを使用するかどうか
    Returns:
        学習済みモデルのパス
    """
    try:
        # 設定ファイルパスを確認
        yaml_path = os.path.join(dataset_dir, 'data.yaml')
        if not os.path.exists(yaml_path):
            print(f"設定ファイルが見つかりません: {yaml_path}")
            return None
        
        # モデルを初期化
        if pretrained:
            model = YOLO('yolov8n.pt')
            print("事前学習済みYOLOv8nモデルから学習を開始します")
        else:
            model = YOLO('yolov8n.yaml')
            print("YOLOv8nモデルをスクラッチから学習します")
        
        # トレーニングパラメータ
        params = {
            'data': yaml_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'patience': 10,  # Early stopping patience
            'device': 0 if torch.cuda.is_available() else 'cpu',
        }
        
        # 保存先が指定されている場合
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            params['project'] = save_dir
            params['name'] = f'yolov8n_custom_{time.strftime("%Y%m%d_%H%M%S")}'
        
        # トレーニング実行
        results = model.train(**params)
        
        # 最良のモデルのパスを取得
        best_model_path = results.best
        
        print(f"トレーニング完了! モデル保存先: {best_model_path}")
        return best_model_path
    
    except Exception as e:
        print(f"トレーニングエラー: {e}")
        return None

def convert_to_yolo_format(annotations, image_dir, output_dir, classes=None):
    """
    アノテーションをYOLO形式に変換する
    Args:
        annotations: アノテーション辞書 {画像パス: {クラス名: [[x1,y1,x2,y2], ...], ...}}
        image_dir: 入力画像ディレクトリ
        output_dir: 出力ディレクトリ
        classes: クラスリスト
    Returns:
        変換されたデータセットのディレクトリパス
    """
    if classes is None:
        classes = DEFAULT_CLASSES
    
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 画像とラベルを分割するディレクトリ
    train_images_dir = os.path.join(images_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')
    train_labels_dir = os.path.join(labels_dir, 'train')
    val_labels_dir = os.path.join(labels_dir, 'val')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # アノテーションを変換
    image_files = list(annotations.keys())
    np.random.shuffle(image_files)
    
    # 訓練/検証分割 (80/20)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # データ変換
    for img_path in train_files:
        _convert_single_annotation(img_path, annotations[img_path], train_images_dir, train_labels_dir, classes)
    
    for img_path in val_files:
        _convert_single_annotation(img_path, annotations[img_path], val_images_dir, val_labels_dir, classes)
    
    # YAML設定ファイル作成
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': os.path.join('images', 'train'),
        'val': os.path.join('images', 'val'),
        'nc': len(classes),
        'names': classes
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml_content_str = ""
        for key, value in yaml_content.items():
            if key == 'names':
                yaml_content_str += f"{key}: {value}\n"
            else:
                yaml_content_str += f"{key}: {value}\n"
        f.write(yaml_content_str)
    
    print(f"YOLOデータセットの変換が完了しました: {output_dir}")
    print(f"トレーニングデータ: {len(train_files)}個, 検証データ: {len(val_files)}個")
    
    return output_dir

def _convert_single_annotation(img_path, annotations, output_images_dir, output_labels_dir, classes):
    """
    1枚の画像のアノテーションをYOLO形式に変換する
    Args:
        img_path: 画像パス
        annotations: アノテーション {クラス名: [[x1,y1,x2,y2], ...], ...}
        output_images_dir: 出力画像ディレクトリ
        output_labels_dir: 出力ラベルディレクトリ
        classes: クラスリスト
    """
    try:
        # 画像ファイル名取得
        img_filename = os.path.basename(img_path)
        img_name = os.path.splitext(img_filename)[0]
        
        # 画像をコピー
        import shutil
        output_img_path = os.path.join(output_images_dir, img_filename)
        shutil.copy2(img_path, output_img_path)
        
        # 画像サイズ取得
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # ラベルファイル作成
        label_file = os.path.join(output_labels_dir, f"{img_name}.txt")
        
        with open(label_file, 'w') as f:
            for class_name, boxes in annotations.items():
                if class_name not in classes:
                    continue
                
                class_id = classes.index(class_name)
                
                for box in boxes:
                    x1, y1, x2, y2 = box
                    
                    # YOLO形式に変換 (中心x, 中心y, 幅, 高さ) - 正規化
                    center_x = ((x1 + x2) / 2) / img_width
                    center_y = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # ラベル行を書き込み: <class_id> <center_x> <center_y> <width> <height>
                    f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
    
    except Exception as e:
        print(f"変換エラー ({img_path}): {e}")

def batch_detect_objects(image_paths, model=None, conf_threshold=0.25, progress_callback=None):
    """
    複数の画像で物体検出を実行する
    Args:
        image_paths: 画像パスのリスト
        model: YOLOモデル
        conf_threshold: 信頼度のしきい値
        progress_callback: 進捗コールバック関数
    Returns:
        検出結果の辞書 {画像パス: 検出結果, ...}
    """
    if model is None:
        model = get_yolo_model()
    
    results = {}
    total = len(image_paths)
    
    for i, img_path in enumerate(image_paths):
        if progress_callback:
            if not progress_callback(i, total, f"画像 {i+1}/{total} を処理中: {os.path.basename(img_path)}"):
                break
                
        detections = detect_objects(img_path, model, conf_threshold)
        results[img_path] = detections
    
    return results

def draw_detection_preview(image_path, detections, output_path=None):
    """
    検出結果のプレビュー画像を生成する
    Args:
        image_path: 元画像のパス
        detections: 検出結果のリスト
        output_path: 出力画像のパス
    Returns:
        描画された画像 (PIL.Image)
    """
    try:
        # 画像を開く
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # 各検出結果を描画
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # 色を決定（クラスによって変える）
            from hashlib import md5
            color_hash = int(md5(class_name.encode()).hexdigest(), 16) % 0xFFFFFF
            r = (color_hash & 0xFF0000) >> 16
            g = (color_hash & 0x00FF00) >> 8
            b = color_hash & 0x0000FF
            color = (r, g, b)
            
            # バウンディングボックスを描画
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # ラベルを描画
            label = f"{class_name} {confidence:.2f}"
            label_size = draw.textlength(label, font=None)
            draw.rectangle([x1, y1, x1 + label_size, y1 + 15], fill=color)
            draw.text([x1, y1], label, fill=(255, 255, 255))
        
        # 出力パスが指定されている場合は保存
        if output_path:
            img.save(output_path)
        
        return img
    
    except Exception as e:
        print(f"描画エラー: {e}")
        return None

# --- ここから追加するイメージラベルクラス ---

class ObjectDetectionImageLabel(QLabel):
    """物体検知用の画像ラベルクラス"""
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1000, 800)
        
        self.current_class = "traffic_cone"  # デフォルトクラス
        self.zoom_factor = 2.5  # 拡大率
        
        # 物体検知用の変数
        self.start_point = None
        self.current_point = None
        self.drawing = False
        self.boxes = []  # 描画したバウンディングボックスを保存
        self.selected_box = None  # 選択中のボックス
        self.dragging = False
        self.drag_start = None
        self.drag_corner = None  # リサイズ中のコーナー
        
        # 検出結果表示用
        self.detections = []
        self.show_detections = True
    
    def set_detections(self, detections):
        """検出結果を設定する"""
        self.detections = detections
        self.update()
    
    def set_current_class(self, class_name):
        """現在のクラスを設定する"""
        self.current_class = class_name
    
    def get_boxes(self):
        """描画したバウンディングボックスを取得する"""
        return self.boxes
    
    def clear_boxes(self):
        """バウンディングボックスをクリアする"""
        self.boxes = []
        self.selected_box = None
        self.update()
    
    def set_boxes(self, boxes):
        """バウンディングボックスを設定する"""
        self.boxes = boxes
        self.selected_box = None
        self.update()
        
    def mousePressEvent(self, event):
        if not self.pixmap() or not self.main_window:
            return
        
        # クリック位置を取得
        pos = event.pos()
        
        # 元の画像サイズ
        pix_width = self.pixmap().width()
        pix_height = self.pixmap().height()
        
        # ズーム係数を使用して拡大後のサイズを計算
        scaled_width = int(pix_width * self.zoom_factor)
        scaled_height = int(pix_height * self.zoom_factor)
        
        # 表示領域の計算
        x = (self.width() - scaled_width) // 2
        y = (self.height() - scaled_height) // 2
        target_rect = QRect(x, y, scaled_width, scaled_height)
        
        # クリック位置が画像内かチェック
        if not target_rect.contains(pos):
            return
        
        # 画像内の相対位置を計算
        rel_x = (pos.x() - target_rect.x()) / target_rect.width()
        rel_y = (pos.y() - target_rect.y()) / target_rect.height()
        
        # 元の画像の座標に変換
        orig_x = int(rel_x * pix_width)
        orig_y = int(rel_y * pix_height)
        
        # 既存のボックスの選択/操作をチェック
        for i, box in enumerate(self.boxes):
            box_class, (x1, y1, x2, y2) = box
            
            # ボックスの内部または境界上か判定
            is_inside = x1 <= orig_x <= x2 and y1 <= orig_y <= y2
            
            # コーナーの近くかチェック (リサイズ用)
            corner_size = 10
            corner_points = [
                ("tl", x1, y1), ("tr", x2, y1),
                ("bl", x1, y2), ("br", x2, y2)
            ]
            
            near_corner = None
            for corner_id, cx, cy in corner_points:
                if abs(orig_x - cx) <= corner_size and abs(orig_y - cy) <= corner_size:
                    near_corner = corner_id
                    break
            
            if near_corner:
                # コーナーをドラッグ開始（リサイズ）
                self.selected_box = i
                self.dragging = True
                self.drag_corner = near_corner
                self.drag_start = (orig_x, orig_y)
                self.update()
                return
            elif is_inside:
                # ボックス内部をクリック（選択または移動）
                if event.button() == Qt.LeftButton:
                    self.selected_box = i
                    self.dragging = True
                    self.drag_corner = None
                    self.drag_start = (orig_x, orig_y)
                    self.update()
                    return
                elif event.button() == Qt.RightButton and self.selected_box == i:
                    # 右クリックで選択中のボックスを削除
                    self.boxes.pop(i)
                    self.selected_box = None
                    self.update()
                    return
        
        # 新しいボックスの描画開始
        if event.button() == Qt.LeftButton:
            self.start_point = (orig_x, orig_y)
            self.current_point = (orig_x, orig_y)
            self.drawing = True
            self.selected_box = None
            self.update()
    
    def mouseMoveEvent(self, event):
        if not self.pixmap() or not self.main_window:
            return
        
        # 元の画像サイズ
        pix_width = self.pixmap().width()
        pix_height = self.pixmap().height()
        
        # ズーム係数を使用して拡大後のサイズを計算
        scaled_width = int(pix_width * self.zoom_factor)
        scaled_height = int(pix_height * self.zoom_factor)
        
        # 表示領域の計算
        x = (self.width() - scaled_width) // 2
        y = (self.height() - scaled_height) // 2
        target_rect = QRect(x, y, scaled_width, scaled_height)
        
        # クリック位置が画像内かチェック
        pos = event.pos()
        if not target_rect.contains(pos):
            return
        
        # 画像内の相対位置を計算
        rel_x = (pos.x() - target_rect.x()) / target_rect.width()
        rel_y = (pos.y() - target_rect.y()) / target_rect.height()
        
        # 元の画像の座標に変換
        orig_x = int(rel_x * pix_width)
        orig_y = int(rel_y * pix_height)
        
        # 範囲内に制限
        orig_x = max(0, min(orig_x, pix_width))
        orig_y = max(0, min(orig_y, pix_height))
        
        if self.drawing:
            # 描画中は現在位置を更新
            self.current_point = (orig_x, orig_y)
            self.update()
        elif self.dragging and self.selected_box is not None:
            # ドラッグ中はボックスを移動/リサイズ
            if self.drag_corner:
                # コーナーをドラッグしてリサイズ
                class_name, (x1, y1, x2, y2) = self.boxes[self.selected_box]
                
                if self.drag_corner == "tl":
                    x1, y1 = orig_x, orig_y
                elif self.drag_corner == "tr":
                    x2, y1 = orig_x, orig_y
                elif self.drag_corner == "bl":
                    x1, y2 = orig_x, orig_y
                elif self.drag_corner == "br":
                    x2, y2 = orig_x, orig_y
                
                # x1 < x2, y1 < y2 を保証
                if x1 > x2:
                    x1, x2 = x2, x1
                    if self.drag_corner in ["tl", "bl"]:
                        self.drag_corner = "tr" if self.drag_corner == "tl" else "br"
                    else:
                        self.drag_corner = "tl" if self.drag_corner == "tr" else "bl"
                
                if y1 > y2:
                    y1, y2 = y2, y1
                    if self.drag_corner in ["tl", "tr"]:
                        self.drag_corner = "bl" if self.drag_corner == "tl" else "br"
                    else:
                        self.drag_corner = "tl" if self.drag_corner == "bl" else "tr"
                
                self.boxes[self.selected_box] = (class_name, (x1, y1, x2, y2))
            else:
                # ボックス全体を移動
                class_name, (x1, y1, x2, y2) = self.boxes[self.selected_box]
                
                # 移動量を計算
                delta_x = orig_x - self.drag_start[0]
                delta_y = orig_y - self.drag_start[1]
                
                # 新しい座標を計算
                new_x1 = x1 + delta_x
                new_y1 = y1 + delta_y
                new_x2 = x2 + delta_x
                new_y2 = y2 + delta_y
                
                # 画像内に収まるように調整
                if new_x1 < 0:
                    new_x2 -= new_x1
                    new_x1 = 0
                elif new_x2 > pix_width:
                    new_x1 -= (new_x2 - pix_width)
                    new_x2 = pix_width
                
                if new_y1 < 0:
                    new_y2 -= new_y1
                    new_y1 = 0
                elif new_y2 > pix_height:
                    new_y1 -= (new_y2 - pix_height)
                    new_y2 = pix_height
                
                self.boxes[self.selected_box] = (class_name, (new_x1, new_y1, new_x2, new_y2))
                self.drag_start = (orig_x, orig_y)
            
            self.update()
    
    def mouseReleaseEvent(self, event):
        if not self.pixmap() or not self.main_window:
            return
        
        if self.drawing:
            # 描画終了
            self.drawing = False
            
            # 新しいボックスを追加（最小サイズチェック）
            if self.start_point and self.current_point:
                x1 = min(self.start_point[0], self.current_point[0])
                y1 = min(self.start_point[1], self.current_point[1])
                x2 = max(self.start_point[0], self.current_point[0])
                y2 = max(self.start_point[1], self.current_point[1])
                
                # 最小サイズチェック (10x10ピクセル以上)
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.boxes.append((self.current_class, (x1, y1, x2, y2)))
                    
                    # コールバック呼び出し（アノテーション更新通知）
                    if self.main_window and hasattr(self.main_window, 'handle_object_annotation'):
                        self.main_window.handle_object_annotation()
            
            self.start_point = None
            self.current_point = None
        
        if self.dragging:
            # ドラッグ終了
            self.dragging = False
            self.drag_start = None
            self.drag_corner = None
            
            # コールバック呼び出し（アノテーション更新通知）
            if self.main_window and hasattr(self.main_window, 'handle_object_annotation'):
                self.main_window.handle_object_annotation()
        
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if not self.pixmap():
            painter = QPainter(self)
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "フォルダを選択し、読込ボタンを押してください")
            painter.end()
            return
        
        painter = QPainter(self)
        
        # 元の画像のサイズ
        pix_width = self.pixmap().width()
        pix_height = self.pixmap().height()
        
        # ズーム係数を使用して拡大後のサイズを計算
        scaled_width = int(pix_width * self.zoom_factor)
        scaled_height = int(pix_height * self.zoom_factor)
        
        # 中央に配置するための座標計算
        x = (self.width() - scaled_width) // 2
        y = (self.height() - scaled_height) // 2
        
        # 画像を拡大して描画
        target_rect = QRect(x, y, scaled_width, scaled_height)
        painter.drawPixmap(target_rect, self.pixmap())
        
        # 検出結果を描画
        if self.show_detections and self.detections:
            for det in self.detections:
                box = det['bbox']
                x1, y1, x2, y2 = box
                class_name = det['class']
                confidence = det['confidence']
                
                # 表示座標に変換
                x1_scaled = x + int(x1 * scaled_width / pix_width)
                y1_scaled = y + int(y1 * scaled_height / pix_height)
                x2_scaled = x + int(x2 * scaled_width / pix_width)
                y2_scaled = y + int(y2 * scaled_height / pix_height)
                
                # クラス別の色を決定
                from hashlib import md5
                color_hash = int(md5(class_name.encode()).hexdigest(), 16)
                r = (color_hash & 0xFF0000) >> 16
                g = (color_hash & 0x00FF00) >> 8
                b = color_hash & 0x0000FF
                color = QColor(r, g, b)
                
                # バウンディングボックスを描画
                painter.setPen(QPen(color, 2))
                painter.drawRect(x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled)
                
                # ラベルを描画
                label = f"{class_name} {confidence:.2f}"
                painter.setFont(QFont("Arial", 8))
                text_width = painter.fontMetrics().horizontalAdvance(label)
                
                # ラベル背景
                painter.fillRect(x1_scaled, y1_scaled - 18, text_width + 4, 18, color)
                
                # ラベルテキスト
                painter.setPen(QPen(Qt.white))
                painter.drawText(x1_scaled + 2, y1_scaled - 5, label)
        
        # 現在の描画中のボックスを描画
        if self.drawing and self.start_point and self.current_point:
            x1 = min(self.start_point[0], self.current_point[0])
            y1 = min(self.start_point[1], self.current_point[1])
            x2 = max(self.start_point[0], self.current_point[0])
            y2 = max(self.start_point[1], self.current_point[1])
            
            # 表示座標に変換
            x1_scaled = x + int(x1 * scaled_width / pix_width)
            y1_scaled = y + int(y1 * scaled_height / pix_height)
            x2_scaled = x + int(x2 * scaled_width / pix_width)
            y2_scaled = y + int(y2 * scaled_height / pix_height)
            
            # 描画中のボックスは点線で表示
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            painter.drawRect(x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled)
            
            # クラス名を表示
            painter.setFont(QFont("Arial", 8))
            painter.setPen(QPen(QColor(255, 255, 0)))
            painter.drawText(x1_scaled + 2, y1_scaled - 5, self.current_class)
        
        # 既存のバウンディングボックスを描画
        for i, box in enumerate(self.boxes):
            class_name, (x1, y1, x2, y2) = box
            
            # 表示座標に変換
            x1_scaled = x + int(x1 * scaled_width / pix_width)
            y1_scaled = y + int(y1 * scaled_height / pix_height)
            x2_scaled = x + int(x2 * scaled_width / pix_width)
            y2_scaled = y + int(y2 * scaled_height / pix_height)
            
            # クラス別の色を決定
            from hashlib import md5
            color_hash = int(md5(class_name.encode()).hexdigest(), 16)
            r = (color_hash & 0xFF0000) >> 16
            g = (color_hash & 0x00FF00) >> 8
            b = color_hash & 0x0000FF
            color = QColor(r, g, b)
            
            # 選択中のボックスは太線で表示
            if i == self.selected_box:
                painter.setPen(QPen(color, 3))
            else:
                painter.setPen(QPen(color, 2))
            
            painter.drawRect(x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled)
            
            # ラベルを描画
            painter.setFont(QFont("Arial", 8))
            
            # ラベル背景
            painter.fillRect(x1_scaled, y1_scaled - 18, 
                            painter.fontMetrics().horizontalAdvance(class_name) + 4, 18, color)
            
            # ラベルテキスト
            painter.setPen(QPen(Qt.white))
            painter.drawText(x1_scaled + 2, y1_scaled - 5, class_name)
            
            # 選択中のボックスにはコーナーポイントを表示
            if i == self.selected_box:
                corner_size = 6
                corner_points = [
                    (x1_scaled, y1_scaled),  # 左上
                    (x2_scaled, y1_scaled),  # 右上
                    (x1_scaled, y2_scaled),  # 左下
                    (x2_scaled, y2_scaled)   # 右下
                ]
                
                painter.setBrush(QBrush(Qt.white))
                for cx, cy in corner_points:
                    painter.drawRect(cx - corner_size // 2, cy - corner_size // 2, 
                                    corner_size, corner_size)
        
        painter.end()
