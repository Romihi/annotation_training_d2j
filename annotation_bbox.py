# annotation.py
""" AIミニカーで取得した画像に対して教師データを作成、モデルを学習するツール"""
import sys
import os
import re
import json
import shutil
import time
import random
import subprocess
import inspect
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # GUIバックエンドを使用しない設定
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import io

import torch
torch.set_num_threads(2)  # スレッド数を制限
# マルチプロセッシングのコンテキストが設定されていない場合のみ設定する
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # すでに設定されている場合は無視
# メモリ管理の最適化
torch.cuda.empty_cache()

from ultralytics import YOLO
from ultralytics import settings

import mlflow
import mlflow.pytorch

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QMessageBox,
                            QScrollArea, QGridLayout, QFrame, QLineEdit, QProgressDialog,
                            QCheckBox, QSpinBox, QComboBox, QSlider, QInputDialog, 
                            QDoubleSpinBox, QGraphicsOpacityEffect, QDialog, QDialogButtonBox,
                            QGroupBox, QRadioButton, QTabWidget, QSizePolicy,QButtonGroup,
                            QListView, QTreeView, QAbstractItemView)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QBrush, QFont
from PyQt5.QtCore import Qt, QRect, QPoint, QTimer, QEvent

from PIL import Image, ImageDraw

# カスタムモジュールのインポート
from model_catalog import get_model, list_available_models
from inference_utils import batch_inference
from exports_file import export_to_donkey, export_to_jetracer, export_to_video, export_to_video_multi_source
from exports_file import export_to_yolo
from model_training import train_model, create_datasets
from model_training import train_location_model
from model_training import create_location_datasets
from model_training import prepare_location_data_from_annotations
from location_model_manager import LocationModelManager
from model_training import generate_augmentation_samples
# TODO:ボタンのスタイルを実装、他のUIの移植については後ほど検討
from styles import get_location_color, apply_style, set_theme, get_current_theme, PRIMARY_STYLE, MODEL_STYLE, TRAINING_STYLE, EXPORT_STYLE, SPECIAL_STYLE, DESTRUCTIVE_STYLE, NAV_STYLE

try:
    from enhanced_annotations import apply_enhanced_annotations_display
    print("物体検知アノテーション表示拡張モジュールを読み込みました")
    use_yolo = True
except ImportError:
    print("警告: enhanced_annotations.pyが見つかりません。物体検知アノテーション表示拡張は無効です")
    use_yolo = False


import traceback
def exception_hook(exc_type, exc_value, exc_traceback):
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_hook


# グローバル設定変数
## アプリケーション関連のパス設定
APP_DIR_PATH = os.path.dirname(os.path.abspath(__file__))  # スクリプトのあるディレクトリを基準
SESSION_DIR_NAME = "sessions"  # セッション情報の保存フォルダ名
MODELS_DIR_NAME = "models"     # モデル保存用のフォルダ名
ANNOTATION_DIR_NAME = "annotation"  # アノテーション関連データフォルダ名
DATA_DONKEY_DIR_NAME = "data_donkey"  # Donkeycar形式データ保存用フォルダ名
DATA_JETRACER_DIR_NAME = "data_jetracer"  # Jetracer形式データ保存用フォルダ名
DATA_YOLO_DIR_NAME = "data_yolo"

# TODO:後ほど再チェック
## 保存先フォルダ関係
### アノテーション関連
annotation_folder = os.path.join(APP_DIR_PATH, ANNOTATION_DIR_NAME)
os.makedirs(annotation_folder, exist_ok=True)
donkey_dataset_dir = os.path.join(annotation_folder, DATA_DONKEY_DIR_NAME)
os.makedirs(donkey_dataset_dir, exist_ok=True)
jetracer_dataset_dir = os.path.join(annotation_folder, DATA_JETRACER_DIR_NAME)
os.makedirs(jetracer_dataset_dir, exist_ok=True)
yolo_dataset_dir = os.path.join(APP_DIR_PATH, DATA_YOLO_DIR_NAME)
os.makedirs(yolo_dataset_dir, exist_ok=True)
video_folder = os.path.join(annotation_folder, "video")
os.makedirs(video_folder, exist_ok=True)


### モデル関連
models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
os.makedirs(models_dir, exist_ok=True)
mlflow_dir = os.path.join(APP_DIR_PATH, "mlruns")
os.makedirs(mlflow_dir, exist_ok=True)
# パスの正規化 - すべてのバックスラッシュをフォワードスラッシュに変換
normalized_path = mlflow_dir.replace("\\", "/")

### アプリ関連
session_dir = os.path.join(APP_DIR_PATH, SESSION_DIR_NAME)
os.makedirs(session_dir, exist_ok=True)

class ImageLabel(QLabel):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1000, 800)
        self.annotation_point = None
        self.show_grid = True  # グリッド表示フラグ
        self.grid_size = 10    # グリッドの分割数
        self.inference_point = None  # 推論結果の表示用ポイント
        self.show_inference = False  # 推論表示フラグ
        self.zoom_factor = 2.5  # デフォルトの拡大率（250%）
        self.is_deleted = False  # 削除状態フラグ

        # バウンディングボックス関連
        self.bbox_start = None  # バウンディングボックスの開始点
        self.bbox_end = None    # バウンディングボックスの終了点
        self.bboxes = []        # 作成済みのバウンディングボックス
        self.current_class = 0  # 現在選択中のクラスインデックス
        self.is_drawing_bbox = False  # バウンディングボックス描画中フラグ
        self.selected_bbox_index = None  # 選択されたバウンディングボックスのインデックス
        self.is_moving_bbox = False      # バウンディングボックス移動中フラグ
        self.move_start_pos = None       # 移動開始位置
        self.setMouseTracking(True)  # マウスの移動を追跡
        self.hovering_bbox_index = None  # ホバー中のバウンディングボックスのインデックス

        # 修飾キーの状態
        self.key_b_pressed = False  # bキーが押されているかどうか
        self.setFocusPolicy(Qt.StrongFocus)  # キーボードフォーカスを受け取れるように
        
    def keyPressEvent(self, event):
        # bキーが押されたらフラグを設定
        if event.key() == Qt.Key_B:
            self.key_b_pressed = True

        # エラーを修正 - main_image_view の属性をチェック
        elif event.key() in [Qt.Key_Delete, Qt.Key_Backspace] and hasattr(self, 'main_image_view') and self.main_image_view.selected_bbox_index is not None:
            # ここで main_image_view の selected_bbox_index を使用
            print(f"削除キーが押されました。選択されたインデックス: {self.main_image_view.selected_bbox_index}")
            current_img_path = self.images[self.current_index]
            if current_img_path in self.bbox_annotations:
                bboxes = self.bbox_annotations[current_img_path]
                if 0 <= self.main_image_view.selected_bbox_index < len(bboxes):
                    # 選択されたバウンディングボックスを削除
                    del bboxes[self.main_image_view.selected_bbox_index]
                    # インデックスをリセット
                    self.main_image_view.selected_bbox_index = None
                    # 再描画
                    self.main_image_view.update()
                    
                    # バウンディングボックスの統計情報を更新
                    self.update_bbox_stats()
                    
                    print("バウンディングボックスを削除しました")
                    return  # イベント処理を終了
        
        # 親クラスのキーイベント処理を呼び出す
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        # bキーが離されたらフラグをクリア
        if event.key() == Qt.Key_B:
            self.key_b_pressed = False
        super().keyReleaseEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_moving_bbox:
            # 移動が完了したのでフラグをリセット
            self.is_moving_bbox = False
            self.move_start_pos = None
            
            # 移動完了メッセージ表示
            if self.selected_bbox_index is not None:
                current_img_path = self.main_window.images[self.main_window.current_index]
                if current_img_path in self.main_window.bbox_annotations:
                    bboxes = self.main_window.bbox_annotations[current_img_path]
                    if 0 <= self.selected_bbox_index < len(bboxes):
                        bbox = bboxes[self.selected_bbox_index]
                        class_name = bbox.get('class', 'unknown')
                        # ステータスバーに移動完了メッセージを表示
                        if hasattr(self.main_window, 'statusBar'):
                            x1 = bbox['x1']
                            y1 = bbox['y1']
                            x2 = bbox['x2']
                            y2 = bbox['y2']
                            width = x2 - x1
                            height = y2 - y1
                            self.main_window.statusBar().showMessage(
                                f"'{class_name}' バウンディングボックスを移動しました "
                                f"[位置: ({x1:.2f}, {y1:.2f}), サイズ: {width:.2f}x{height:.2f}]", 
                                3000
                            )
            
            # 通常のカーソルに戻す
            self.setCursor(Qt.ArrowCursor)
            self.update()
        elif self.is_drawing_bbox and self.pixmap() and self.bbox_start and self.bbox_end:
            # バウンディングボックスの確定処理
            if abs(self.bbox_end.x() - self.bbox_start.x()) > 10 and abs(self.bbox_end.y() - self.bbox_start.y()) > 10:
                class_name = self.main_window.select_object_class()
                if class_name:
                    bbox = {
                        'x1': min(self.bbox_start.x(), self.bbox_end.x()) / self.pixmap().width(),
                        'y1': min(self.bbox_start.y(), self.bbox_end.y()) / self.pixmap().height(),
                        'x2': max(self.bbox_start.x(), self.bbox_end.x()) / self.pixmap().width(),
                        'y2': max(self.bbox_start.y(), self.bbox_end.y()) / self.pixmap().height(),
                        'class': class_name
                    }
                    self.main_window.add_bbox_annotation(bbox)
            
            self.is_drawing_bbox = False
            self.bbox_start = None
            self.bbox_end = None
            # 通常のカーソルに戻す
            self.setCursor(Qt.ArrowCursor)
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
        
        # 削除済みの場合は赤い枠を表示
        if self.is_deleted:
            painter.setPen(QPen(QColor(255, 85, 85), 6))  # 赤い枠線
            border_rect = QRect(x-6, y-6, scaled_width+12, scaled_height+12)
            painter.drawRect(border_rect)
            
            # 削除済みバッジを表示
            badge_rect = QRect(x - 100, y, 80, 40)
            painter.fillRect(badge_rect, QColor(255, 85, 85))
            painter.setPen(QPen(Qt.white, 2))
            painter.setFont(QFont("Arial", 12, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, "削除済み")
        # 画像の位置情報があれば、その色で枠を描画
        elif self.main_window and hasattr(self.main_window, 'current_location') and self.main_window.current_location is not None:
            loc_value = self.main_window.current_location
            loc_color = get_location_color(loc_value)
            
            # 太い枠線を描画
            painter.setPen(QPen(loc_color, 6))
            border_rect = QRect(x-3, y-3, scaled_width+6, scaled_height+6)
            painter.drawRect(border_rect)
            
            # 位置番号を左側に表示する枠を描画
            badge_size = 40
            badge_rect = QRect(x - badge_size - 10, y, badge_size, badge_size)
            painter.fillRect(badge_rect, loc_color)
            painter.setPen(QPen(Qt.white, 2))
            painter.setFont(QFont("Arial", 16, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, str(loc_value))
        
        # 画像を拡大して描画
        target_rect = QRect(x, y, scaled_width, scaled_height)
        painter.drawPixmap(target_rect, self.pixmap())
        
        # グリッド表示
        if self.show_grid:
            painter.setPen(QPen(QColor(100, 100, 100, 100), 1))  # 半透明グレー
            
            # 横線（X座標のグリッド）
            step_x = target_rect.width() / self.grid_size
            for i in range(1, self.grid_size):
                x_pos = target_rect.x() + i * step_x
                painter.drawLine(int(x_pos), target_rect.y(), int(x_pos), target_rect.y() + target_rect.height())
                
            # 縦線（Y座標のグリッド）
            step_y = target_rect.height() / self.grid_size
            for i in range(1, self.grid_size):
                y_pos = target_rect.y() + i * step_y
                painter.drawLine(target_rect.x(), int(y_pos), target_rect.x() + target_rect.width(), int(y_pos))
            
            # 中央の十字線（より目立たせる）
            painter.setPen(QPen(QColor(200, 200, 200, 150), 2))
            mid_x = target_rect.x() + target_rect.width() / 2
            mid_y = target_rect.y() + target_rect.height() / 2
            painter.drawLine(int(mid_x), target_rect.y(), int(mid_x), target_rect.y() + target_rect.height())
            painter.drawLine(target_rect.x(), int(mid_y), target_rect.x() + target_rect.width(), int(mid_y))

            # 目盛り表示
            # フォント設定
            painter.setFont(QFont("Arial", 10))
            
            # 軸の色を設定
            painter.setPen(QPen(QColor(80, 80, 80, 200), 1))
            
            # X軸の目盛り表示（上側の水平線）
            painter.drawText(target_rect.x() - 25, target_rect.y() - 5, "-1")  # 左端 (-1)
            painter.drawText(target_rect.x() + target_rect.width() + 5, target_rect.y() - 5, "1")  # 右端 (1)
            
            # 中間の目盛り (X軸)
            for i in range(1, self.grid_size):
                value = -1 + (2.0 * i / self.grid_size)
                x_pos = target_rect.x() + i * (target_rect.width() / self.grid_size)
                # 0の場合は特別に表示
                if abs(value) < 0.1:  # ほぼ0
                    painter.drawText(int(x_pos) - 5, target_rect.y() - 5, "0")
                elif i % 2 == 0:  # 偶数の目盛りのみ値を表示
                    painter.drawText(int(x_pos) - 15, target_rect.y() - 5, f"{value:.1f}")
            
            # Y軸の目盛り表示（左側の垂直線）
            painter.drawText(target_rect.x() - 35, target_rect.y() + 15, "-1")  # 上端 (-1)
            painter.drawText(target_rect.x() - 35, target_rect.y() + target_rect.height(), "1")  # 下端 (1)
            
            # 中間の目盛り (Y軸)
            for i in range(1, self.grid_size):
                value = -1 + (2.0 * i / self.grid_size)
                y_pos = target_rect.y() + i * (target_rect.height() / self.grid_size)
                # 0の場合は特別に表示
                if abs(value) < 0.1:  # ほぼ0
                    painter.drawText(target_rect.x() - 35, int(y_pos) + 5, "0")
                elif i % 2 == 0:  # 偶数の目盛りのみ値を表示
                    painter.drawText(target_rect.x() - 35, int(y_pos) + 5, f"{value:.1f}")


        # 削除済みでない場合のみアノテーションや推論を表示
        if not self.is_deleted:
            # 常にバウンディングボックスを表示（モードに関わらず）
            if self.main_window and hasattr(self.main_window, 'bbox_annotations'):
                current_img_path = self.main_window.images[self.main_window.current_index]
                if current_img_path in self.main_window.bbox_annotations:
                    bboxes = self.main_window.bbox_annotations[current_img_path]
                    
                    for i, bbox in enumerate(bboxes):
                        # クラスに応じた色を設定
                        class_name = bbox.get('class', 'unknown')
                        class_colors = {
                            'car': QColor(255, 0, 0, 180),     # 赤
                            'person': QColor(0, 255, 0, 180),  # 緑
                            'sign': QColor(0, 0, 255, 180),    # 青
                            'cone': QColor(255, 255, 0, 180),  # 黄
                            'unknown': QColor(128, 128, 128, 180) # グレー
                        }
                        color = class_colors.get(class_name, QColor(255, 0, 0, 180))
                        
                        # 選択またはホバーされているバウンディングボックスかどうかで線の太さを変更
                        is_selected = i == self.selected_bbox_index
                        is_hovered = i == self.hovering_bbox_index
                        
                        pen_width = 3 if is_selected else (2.5 if is_hovered else 2)
                        pen_style = Qt.DashLine if is_selected else (Qt.DashDotLine if is_hovered else Qt.SolidLine)
                        
                        # 正規化された座標を画面座標に変換
                        x1 = int(target_rect.x() + bbox['x1'] * target_rect.width())
                        y1 = int(target_rect.y() + bbox['y1'] * target_rect.height())
                        x2 = int(target_rect.x() + bbox['x2'] * target_rect.width())
                        y2 = int(target_rect.y() + bbox['y2'] * target_rect.height())
                        
                        # バウンディングボックスを描画
                        painter.setPen(QPen(color, pen_width, pen_style))
                        
                        # ホバー中のバウンディングボックスは半透明の塗りつぶしを追加
                        if is_hovered or is_selected:
                            highlight_color = QColor(color)
                            highlight_color.setAlpha(40)  # 非常に透明に
                            painter.setBrush(QBrush(highlight_color))
                        else:
                            painter.setBrush(QBrush())  # 透明ブラシ
                        
                        painter.drawRect(QRect(x1, y1, x2-x1, y2-y1))
                        
                        # 選択されているバウンディングボックスには角にハンドルを表示
                        if is_selected:
                            handle_size = 6
                            painter.setBrush(QBrush(color))
                            painter.drawRect(QRect(x1-handle_size//2, y1-handle_size//2, handle_size, handle_size))
                            painter.drawRect(QRect(x2-handle_size//2, y1-handle_size//2, handle_size, handle_size))
                            painter.drawRect(QRect(x1-handle_size//2, y2-handle_size//2, handle_size, handle_size))
                            painter.drawRect(QRect(x2-handle_size//2, y2-handle_size//2, handle_size, handle_size))
                        
                        # ラベルテキストを作成（信頼度情報がある場合は追加）
                        label_text = class_name
                        if 'confidence' in bbox:
                            label_text += f" {bbox['confidence']:.2f}"
                        
                        # クラスラベルの背景を描画
                        label_rect = QRect(x1, y1-20, len(label_text)*8+10, 20)
                        painter.fillRect(label_rect, color)
                        
                        # クラス名を描画
                        painter.setPen(QPen(Qt.white, 1))
                        painter.setFont(QFont("Arial", 10, QFont.Bold))
                        painter.drawText(label_rect, Qt.AlignCenter, label_text)
                
                # 描画中のバウンディングボックスがあれば表示
                if self.is_drawing_bbox and self.bbox_start and self.bbox_end:
                    # バウンディングボックスの座標を計算
                    start_rel_x = self.bbox_start.x() / pix_width
                    start_rel_y = self.bbox_start.y() / pix_height
                    end_rel_x = self.bbox_end.x() / pix_width
                    end_rel_y = self.bbox_end.y() / pix_height
                    
                    start_x = int(target_rect.x() + start_rel_x * target_rect.width())
                    start_y = int(target_rect.y() + start_rel_y * target_rect.height())
                    end_x = int(target_rect.x() + end_rel_x * target_rect.width())
                    end_y = int(target_rect.y() + end_rel_y * target_rect.height())
                    
                    # 半透明の黄色でドラッグ中のボックスを描画
                    painter.setPen(QPen(QColor(255, 255, 0, 180), 2, Qt.DashLine))
                    painter.setBrush(QBrush(QColor(255, 255, 0, 40)))
                    painter.drawRect(QRect(
                        min(start_x, end_x),
                        min(start_y, end_y),
                        abs(end_x - start_x),
                        abs(end_y - start_y)
                    ))
                
                # アノテーションポイントの描画（運転制御アノテーション）
                if self.annotation_point:
                    rel_x = self.annotation_point.x() / self.pixmap().width()
                    rel_y = self.annotation_point.y() / self.pixmap().height()
                    
                    scaled_x = int(target_rect.x() + rel_x * target_rect.width())
                    scaled_y = int(target_rect.y() + rel_y * target_rect.height())
                    
                    # 赤い円の描画 - より大きく太く
                    painter.setPen(QPen(QColor(255, 0, 0), 4))  # 太さを4に増加
                    circle_size = 15  # 円のサイズを大きく(元は10)
                    painter.drawEllipse(scaled_x - circle_size, scaled_y - circle_size, circle_size*2, circle_size*2)
                
                # 推論ポイントの描画
                if self.show_inference and self.inference_point:
                    rel_x = self.inference_point.x() / self.pixmap().width()
                    rel_y = self.inference_point.y() / self.pixmap().height()
                    
                    scaled_x = int(target_rect.x() + rel_x * target_rect.width())
                    scaled_y = int(target_rect.y() + rel_y * target_rect.height())
                    
                    # 青い円の描画 - より大きく太く
                    painter.setPen(QPen(QColor(0, 0, 255), 4))  # 太さを4に増加
                    circle_size = 15  # 円のサイズを大きく(元は10)
                    painter.drawEllipse(scaled_x - circle_size, scaled_y - circle_size, circle_size*2, circle_size*2)
            
                # ここから物体検知推論結果表示の追加部分
                # 推論結果表示チェックがオンで、detection_inference_resultsデータがある場合に表示
                if (not self.is_deleted and 
                    self.main_window and 
                    hasattr(self.main_window, 'show_detection_inference') and 
                    self.main_window.show_detection_inference and
                    hasattr(self.main_window, 'detection_inference_results')):
                    
                    current_img_path = self.main_window.images[self.main_window.current_index]
                    if current_img_path in self.main_window.detection_inference_results:
                        inference_bboxes = self.main_window.detection_inference_results[current_img_path]
                        
                        for i, bbox in enumerate(inference_bboxes):
                            # クラスに応じた色を設定 (推論結果は別の透明度で表示)
                            class_name = bbox.get('class', 'unknown')
                            class_colors = {
                                'car': QColor(255, 0, 0, 120),     # 赤 (半透明)
                                'person': QColor(0, 255, 0, 120),  # 緑 (半透明)
                                'sign': QColor(0, 0, 255, 120),    # 青 (半透明)
                                'cone': QColor(255, 255, 0, 120),  # 黄 (半透明)
                                'unknown': QColor(128, 128, 128, 120) # グレー (半透明)
                            }
                            color = class_colors.get(class_name, QColor(255, 0, 0, 120))
                            
                            # 推論結果は点線で表示
                            pen_width = 2
                            pen_style = Qt.DashLine
                            
                            # 正規化された座標を画面座標に変換
                            x1 = int(target_rect.x() + bbox['x1'] * target_rect.width())
                            y1 = int(target_rect.y() + bbox['y1'] * target_rect.height())
                            x2 = int(target_rect.x() + bbox['x2'] * target_rect.width())
                            y2 = int(target_rect.y() + bbox['y2'] * target_rect.height())
                            
                            # バウンディングボックスを描画
                            painter.setPen(QPen(color, pen_width, pen_style))
                            painter.drawRect(QRect(x1, y1, x2-x1, y2-y1))
                            
                            # ラベルテキストを作成（信頼度情報がある場合は追加）
                            label_text = f"推論:{class_name}"
                            if 'confidence' in bbox:
                                label_text += f" {bbox['confidence']:.2f}"
                            
                            # クラスラベルの背景を描画
                            label_rect = QRect(x1, y1-20, len(label_text)*8+10, 20)
                            painter.fillRect(label_rect, color)
                            
                            # クラス名を描画
                            painter.setPen(QPen(Qt.white, 1))
                            painter.setFont(QFont("Arial", 10, QFont.Bold))
                            painter.drawText(label_rect, Qt.AlignCenter, label_text)

            # 削除済みの場合は半透明の赤オーバーレイを表示
            if self.is_deleted:
                painter.setOpacity(0.25)  # 75%透明
                painter.fillRect(target_rect, QColor(255, 0, 0))
                
                # 中央に削除済みテキストを表示
                painter.setOpacity(1.0)  # 不透明に戻す
                painter.setPen(QPen(Qt.white, 2))
                painter.setFont(QFont("Arial", 24, QFont.Bold))
                
                painter.drawText(
                    target_rect, 
                    Qt.AlignCenter, 
                    "削除済み\nクリックで再アノテーション"
                )
            
            painter.end()

    def mousePressEvent(self, event):
        if self.pixmap() and self.main_window:
            # クリック位置を取得
            pos = event.pos()
            
            # 元の画像のサイズ
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
            
            # 現在のモードに基づいて処理
            if hasattr(self.main_window, 'current_mode') and self.main_window.current_mode == 1:
                # 物体検知アノテーションモード
                current_img_path = self.main_window.images[self.main_window.current_index]
                
                # 既存のバウンディングボックスを選択するかチェック
                if hasattr(self.main_window, 'bbox_annotations') and current_img_path in self.main_window.bbox_annotations:
                    bboxes = self.main_window.bbox_annotations[current_img_path]
                    
                    # 各バウンディングボックスについて、クリック位置が内部にあるかチェック
                    for i, bbox in enumerate(bboxes):
                        # バウンディングボックスの座標を計算
                        x1 = int(bbox['x1'] * pix_width)
                        y1 = int(bbox['y1'] * pix_height)
                        x2 = int(bbox['x2'] * pix_width)
                        y2 = int(bbox['y2'] * pix_height)
                        
                        # クリック位置がバウンディングボックス内にあるか
                        if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                            # 選択済みのボックスをクリックした場合
                            if self.selected_bbox_index == i:
                                # 選択解除が必要かどうかを判断（例：シフトキーが押されているなど）
                                if event.modifiers() & Qt.ShiftModifier:
                                    self.selected_bbox_index = None
                                    self.update()
                                    if hasattr(self.main_window, 'statusBar'):
                                        self.main_window.statusBar().showMessage("バウンディングボックスの選択を解除しました", 3000)
                                    return
                            
                            # 新規選択の場合
                            self.selected_bbox_index = i
                            self.is_moving_bbox = True
                            self.move_start_pos = QPoint(orig_x, orig_y)
                            
                            # ステータスバーにメッセージ表示
                            if hasattr(self.main_window, 'statusBar'):
                                class_name = bbox.get('class', 'unknown')
                                self.main_window.statusBar().showMessage(f"'{class_name}' バウンディングボックスを選択しました", 3000)
                            
                            self.update()  # 再描画
                            return
                    
                    # どのバウンディングボックスにも含まれない場合、新規描画開始
                    self.selected_bbox_index = None
                    self.bbox_start = QPoint(orig_x, orig_y)
                    self.is_drawing_bbox = True
                    self.bbox_end = self.bbox_start  # 初期点で初期化
                    self.update()  # 再描画
                else:
                    # バウンディングボックスがない場合、新規描画開始
                    self.selected_bbox_index = None
                    self.bbox_start = QPoint(orig_x, orig_y)
                    self.is_drawing_bbox = True
                    self.bbox_end = self.bbox_start  # 初期点で初期化
                    self.update()  # 再描画
            else:
                # 自動運転アノテーションモード
                self.annotation_point = QPoint(orig_x, orig_y)
                
                # メインウィンドウに通知
                self.main_window.handle_annotation(orig_x, orig_y)
                
                # アノテーション後に自動的に次の画像に進む（スキップ枚数考慮）
                if hasattr(self.main_window, 'skip_images_on_click') and self.main_window.skip_images_on_click.isChecked():
                    skip_count = self.main_window.skip_count_spin.value()
                    self.main_window.skip_images(skip_count)
                else:
                    self.main_window.skip_images(1)  # デフォルトは1枚

    def leaveEvent(self, event):
        """マウスがウィジェットから離れた時の処理"""
        self.setCursor(Qt.ArrowCursor)  # 通常のカーソルに戻す
        self.hovering_bbox_index = None
        super().leaveEvent(event)

    def check_bbox_hover(self, pos):
        """マウス位置がバウンディングボックス上にあるかチェック"""
        if not self.pixmap() or not hasattr(self.main_window, 'current_mode'):
            return None
        
        # 物体検知モードでない場合は処理しない
        if self.main_window.current_mode != 1:
            return None
        
        # 元の画像のサイズ
        pix_width = self.pixmap().width()
        pix_height = self.pixmap().height()
        
        # ズーム係数を使用して拡大後のサイズを計算
        scaled_width = int(pix_width * self.zoom_factor)
        scaled_height = int(pix_height * self.zoom_factor)
        
        # 表示領域の計算
        x = (self.width() - scaled_width) // 2
        y = (self.height() - scaled_height) // 2
        target_rect = QRect(x, y, scaled_width, scaled_height)
        
        # マウス位置が画像内かチェック
        if not target_rect.contains(pos):
            return None
        
        # 画像内の相対位置を計算
        rel_x = (pos.x() - target_rect.x()) / target_rect.width()
        rel_y = (pos.y() - target_rect.y()) / target_rect.height()
        
        # 元の画像の座標に変換
        orig_x = int(rel_x * pix_width)
        orig_y = int(rel_y * pix_height)
        
        # 現在の画像のバウンディングボックスをチェック
        current_img_path = self.main_window.images[self.main_window.current_index]
        if current_img_path in self.main_window.bbox_annotations:
            bboxes = self.main_window.bbox_annotations[current_img_path]
            
            # 各バウンディングボックスについて、マウス位置が内部にあるかチェック
            for i, bbox in enumerate(bboxes):
                # バウンディングボックスの座標を計算
                x1 = int(bbox['x1'] * pix_width)
                y1 = int(bbox['y1'] * pix_height)
                x2 = int(bbox['x2'] * pix_width)
                y2 = int(bbox['y2'] * pix_height)
                
                # マウス位置がバウンディングボックス内にあるか
                if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                    return i
        
        return None

    def mouseMoveEvent(self, event):
        """マウス移動時の処理 - ホバー効果とカーソル変更を行う"""
        # まずホバーされているバウンディングボックスをチェック（バウンディングボックス移動中以外）
        if not self.is_moving_bbox and not self.is_drawing_bbox:
            hover_index = self.check_bbox_hover(event.pos())
            
            # ホバー状態が変わった場合は再描画
            if hover_index != self.hovering_bbox_index:
                self.hovering_bbox_index = hover_index
                
                # カーソルを更新
                if hover_index is not None:
                    self.setCursor(Qt.OpenHandCursor)  # バウンディングボックス上では手の形
                else:
                    self.setCursor(Qt.ArrowCursor)  # 通常は矢印
                
                self.update()  # 再描画
        
        # 物体検知モードでドラッグ中はカーソルを変更
        if self.is_moving_bbox:
            self.setCursor(Qt.ClosedHandCursor)  # つかんでいる状態
        elif self.is_drawing_bbox:
            self.setCursor(Qt.CrossCursor)  # 描画中は十字
        
        # 既存の移動/描画処理
        if self.pixmap() and (self.is_drawing_bbox or self.is_moving_bbox) and (self.bbox_start or self.is_moving_bbox):
            # クリック位置を取得
            pos = event.pos()
            
            # 元の画像のサイズ
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
                # 画像外に出た場合は画像の端に制限
                constrained_x = max(target_rect.left(), min(pos.x(), target_rect.right()))
                constrained_y = max(target_rect.top(), min(pos.y(), target_rect.bottom()))
                pos = QPoint(constrained_x, constrained_y)
            
            # 画像内の相対位置を計算
            rel_x = (pos.x() - target_rect.x()) / target_rect.width()
            rel_y = (pos.y() - target_rect.y()) / target_rect.height()
            
            # 元の画像の座標に変換
            orig_x = int(rel_x * pix_width)
            orig_y = int(rel_y * pix_height)
            
            if self.is_moving_bbox and self.selected_bbox_index is not None:
                # バウンディングボックスの移動処理
                current_img_path = self.main_window.images[self.main_window.current_index]
                if current_img_path in self.main_window.bbox_annotations:
                    bboxes = self.main_window.bbox_annotations[current_img_path]
                    if 0 <= self.selected_bbox_index < len(bboxes):
                        # 移動距離を計算
                        dx = orig_x - self.move_start_pos.x()
                        dy = orig_y - self.move_start_pos.y()
                        
                        # 更新されたバウンディングボックス座標を計算
                        bbox = bboxes[self.selected_bbox_index]
                        
                        # 画像の端を超えないように制限
                        new_x1 = max(0, min(bbox['x1'] * pix_width + dx, pix_width - 10)) / pix_width
                        new_y1 = max(0, min(bbox['y1'] * pix_height + dy, pix_height - 10)) / pix_height
                        new_x2 = max(10/pix_width, min(bbox['x2'] * pix_width + dx, pix_width)) / pix_width
                        new_y2 = max(10/pix_height, min(bbox['y2'] * pix_height + dy, pix_height)) / pix_height
                        
                        # バウンディングボックスを更新
                        bboxes[self.selected_bbox_index]['x1'] = new_x1
                        bboxes[self.selected_bbox_index]['y1'] = new_y1
                        bboxes[self.selected_bbox_index]['x2'] = new_x2
                        bboxes[self.selected_bbox_index]['y2'] = new_y2
                        
                        # 移動開始位置を更新
                        self.move_start_pos = QPoint(orig_x, orig_y)
                        
                        # ステータスバーに情報表示
                        if hasattr(self.main_window, 'statusBar'):
                            class_name = bbox.get('class', 'unknown')
                            self.main_window.statusBar().showMessage(f"'{class_name}' バウンディングボックスを移動中... [x1={new_x1:.2f}, y1={new_y1:.2f}, x2={new_x2:.2f}, y2={new_y2:.2f}]", 500)
            elif self.is_drawing_bbox:
                # 新規バウンディングボックスの描画処理
                self.bbox_end = QPoint(orig_x, orig_y)
                
                # サイズ情報をステータスバーに表示
                if hasattr(self.main_window, 'statusBar'):
                    width = abs(self.bbox_end.x() - self.bbox_start.x())
                    height = abs(self.bbox_end.y() - self.bbox_start.y())
                    self.main_window.statusBar().showMessage(f"新規バウンディングボックス作成中... 幅: {width}px, 高さ: {height}px", 500)
            
            self.update()  # 画面を更新

class ThumbnailWidget(QWidget):
    def __init__(self, parent=None, img_path="", index=0, is_selected=False, 
                 annotation=None, on_click=None, location_value=None, is_deleted=False):
        super().__init__(parent)
        self.img_path = img_path
        self.index = index
        self.on_click = on_click
        self.is_selected = is_selected
        self.annotation = annotation  # アノテーション情報
        self.location_value = location_value  # 変更: 辞書ではなく直接位置情報の値を受け取る
        self.is_deleted = is_deleted  # 削除済みフラグ
        
        # サムネイル全体のサイズも調整
        self.setMinimumWidth(210)
        self.setMinimumHeight(170)  # 高さを少し小さく
        
        # メインレイアウト（水平レイアウト）
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)  # マージンをなくす
        self.layout.setSpacing(1)  # 最小限のスペーシング
        
        # 左側の情報パネル
        info_panel = QWidget()
        info_panel.setFixedWidth(70)  # 情報パネル幅
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(1)

        # ラベル
        info_panel_label = QLabel("情報パネル:")
        info_panel_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(info_panel_label)

        # インデックス番号
        self.idx_label = QLabel(f"{index + 1}")
        self.idx_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.idx_label)
        
        # 削除済みバッジ（削除されている場合）
        if is_deleted:
            deleted_badge = QLabel("削除済")
            deleted_badge.setAlignment(Qt.AlignCenter)
            deleted_badge.setStyleSheet("""
                background-color: #FF5555;
                color: white;
                font-weight: bold;
                border-radius: 10px;
                min-width: 20px;
                min-height: 20px;
                padding: 1px;
            """)
            info_layout.addWidget(deleted_badge)
        # アノテーション情報
        elif annotation and not is_deleted:
            angle_label = QLabel(f"A: {annotation.get('angle', 0):.2f}")
            angle_label.setStyleSheet("color: #FF6666; font-size: 12px;font-weight: bold;")
            info_layout.addWidget(angle_label)
            
            throttle_label = QLabel(f"T: {annotation.get('throttle', 0):.2f}")
            throttle_label.setStyleSheet("color: #FF6666; font-size: 12px;font-weight: bold;")
            info_layout.addWidget(throttle_label)

            # 位置情報バッジ（位置情報がある場合）
            # 変更: 辞書からの参照ではなく、直接location_valueを使用
            if location_value is not None:
                loc_color = get_location_color(location_value)
                
                loc_badge = QLabel(str(location_value))
                loc_badge.setAlignment(Qt.AlignCenter)
                loc_badge.setStyleSheet(f"""
                    background-color: {loc_color.name()};
                    color: white;
                    font-weight: bold;
                    border-radius: 10px;
                    min-width: 20px;
                    min-height: 20px;
                    padding: 1px;
                """)
                info_layout.addWidget(loc_badge)
                    

        # 分布グラフ用ラベルを追加 (info_layoutに追加)
        self.distribution_label = QLabel("アノテーションがありません")
        self.distribution_label.setAlignment(Qt.AlignCenter)
        self.distribution_label.setMinimumHeight(300)  # 高さを確保
        self.distribution_label.setStyleSheet("background-color: #f5f5f5; border: 1px solid #dddddd;")
        info_layout.addWidget(self.distribution_label)

        # 残りのスペースを埋めるスペーサー
        info_layout.addStretch()
        
        # 左側の情報パネルをメインレイアウトに追加
        self.layout.addWidget(info_panel)
        
        # 右側の画像パネル
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)  # スペーシングをなくす
        
        # ファイル名ラベルを画像の上部に配置
        filename = os.path.basename(img_path)
        if len(filename) > 20:  # ファイル名が長い場合は切り詰める
            filename = filename[:18] + "..."
        
        name_label = QLabel(filename)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("font-size: 12px; color: #444444; background-color: #f8f8f8;font-weight: bold;")
        name_label.setFixedHeight(10)  # 高さを最小限に
        image_layout.addWidget(name_label)
        
        # 画像コンテナ（枠を付けるための外側のコンテナ）
        image_container = QFrame()
        
        # ボーダーのスタイル設定
        border_style = ""
        if is_selected:
            border_style = "border: 2px solid red;"
        elif is_deleted:
            border_style = "border: 2px solid #FF5555;"  # 削除済みは赤い枠線
        elif location_value is not None:
            loc_color = get_location_color(location_value)
            border_style = f"border: 2px solid {loc_color.name()};"
        elif annotation:
            border_style = "border: 2px solid #FF9966;"  # アノテーションのみはオレンジ系
        else:
            border_style = "border: 1px solid #dddddd;"
        
        # 画像コンテナのレイアウト - マージンを完全に削除
        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)  # 余白なし
        image_container_layout.setSpacing(0)  # スペーシングなし
        
        # 画像ラベル
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(150, 140)  # 幅を少し広げる
        
        # 削除済みの場合は半透明になるスタイルを追加
        if is_deleted:
            self.img_label.setStyleSheet(f"{border_style} padding: 0px; opacity: 0.5;")
        else:
            self.img_label.setStyleSheet(f"{border_style} padding: 0px;")
            
        image_container_layout.addWidget(self.img_label)
        
        # 画像コンテナをイメージパネルに追加
        image_layout.addWidget(image_container)
        
        # 画像パネルをメインレイアウトに追加
        self.layout.addWidget(image_panel)
        
        # 画像を読み込む
        self.load_image(img_path)
        
        # ウィジェット全体の枠線はなし
        self.setStyleSheet("border: none;")

    def mousePressEvent(self, event):
        # クリック時にon_click関数を呼び出す
        if self.on_click and event.button() == Qt.LeftButton:
            self.on_click(self.index)

    def load_image(self, img_path):
        if not os.path.exists(img_path):
            return
        
        try:
            # アノテーションがある場合は、丸を描画したイメージを作成
            if self.annotation and not self.is_deleted:
                # PILで画像を開く
                pil_img = Image.open(img_path)
                
                # アノテーションの座標を取得
                x, y = self.annotation["x"], self.annotation["y"]
                
                # PILのDrawオブジェクトを作成し、丸を描画
                draw = ImageDraw.Draw(pil_img)
                circle_size = 15  # サムネイル用の丸のサイズを大きく
                draw.ellipse((x-circle_size, y-circle_size, x+circle_size, y+circle_size), outline='red', width=4)
                
                # PILイメージをQImageに変換
                pil_img = pil_img.convert("RGBA")
                data = pil_img.tobytes("raw", "RGBA")
                qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
                
                # QImageをQPixmapに変換
                pixmap = QPixmap.fromImage(qimg)
            else:
                # アノテーションがない場合は通常通り画像を読み込む
                pixmap = QPixmap(img_path)
            
            if not pixmap.isNull():
                pixmap = pixmap.scaled(170, 170, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # サイズを調整
                self.img_label.setPixmap(pixmap)
                
                # 削除済みの場合は半透明にする追加の処理
                if self.is_deleted:
                    self.setGraphicsEffect(QGraphicsOpacityEffect(opacity=0.5))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

class ImageAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize state
        self.folder_path = ""
        self.folder_paths = []       
        self.image_folders = []      
        self.images = []
        self.current_index = 0
        self.available_variants = {}
        self.annotations = {}
        self.annotation_history = []
        self.annotated_count = 0
        # 現在のアノテーションモード（0=自動運転、1=物体検知）
        self.current_mode = 0
        self.last_selected_bbox_class = None  # 前回選択した物体検知クラス
        self.last_bbox = None  # 前回作成したバウンディングボックスの情報
        self.last_bboxes = []  # 前回の画像の全てのバウンディングボックスを保存するリスト（新規追加）
        self.auto_apply_last_bbox = False  # 前回のバウンディングボックスを自動適用するかどうか

        # 削除インデックス
        self.deleted_indexes = []

        # manifest.jsonのパスを保存する変数
        self.last_manifest_path = None

        self.info_panel_width = 280  # 基本の幅
        self.info_panel_margin = 20  # パネル周りの余白（左右合計）

        # 位置情報関連の初期化
        self.location_buttons = []  # 位置情報ボタンのリスト
        self.current_location = None  # 現在選択されている位置情報
        self.location_annotations = {}  # 画像ごとの位置情報アノテーション
        
        # アノテーションのタイムスタンプを保存する辞書
        self.annotation_timestamps = {}
        
        # 推論結果のキャッシュ
        self.inference_results = {}

        # YOLO関連の初期化を追加
        self.yolo_model = None  # YOLOモデルのインスタンス
        self.yolo_confidence_threshold = 0.6  # デフォルトの信頼度閾値
        # 既存の初期化コードに追加
        self.bbox_annotations = {}  # 物体検知アノテーション用
        self.class_names = ["car", "person", "sign", "cone"]  # デフォルトクラス

        # Setup UI
        self.init_ui()

        # Update UI
        self.update_stats()
        self.display_current_image()
        self.update_gallery()
        #self.update_distribution_graph()  

        if hasattr(self, 'prev_multi_button') and hasattr(self, 'next_multi_button'):
            self.update_skip_button_labels(10)  # デフォルト値は10

        self.add_session_check_to_init_ui()

        QApplication.instance().installEventFilter(self)

    def init_ui(self):
        self.setWindowTitle("画像アノテーションツール")
        self.setGeometry(100, 100, 1600, 900)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)
        main_layout.addWidget(left_panel)

        # Folder selection
        folder_label = QLabel("データ読込（imagesフォルダを持つ親フォルダ選択:")
        left_layout.addWidget(folder_label)
        
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("フォルダパスを入力または参照ボタンで複数選択可能")
        self.folder_input.textChanged.connect(self.on_folder_path_changed)
        folder_layout.addWidget(self.folder_input)
        
        browse_button = QPushButton("参照...")
        browse_button.clicked.connect(self.browse_folder)
        folder_layout.addWidget(browse_button)
        apply_style(browse_button, 'primary')
        left_layout.addLayout(folder_layout)
        
        load_button_layout = QHBoxLayout()

        self.load_button = QPushButton("画像を読込")
        self.load_button.clicked.connect(self.load_images)
        self.load_button.setEnabled(False)  # 初期状態は無効
        apply_style(self.load_button, 'primary')
        load_button_layout.addWidget(self.load_button)

        # アノテーションデータ読み込みボタンを追加
        self.load_annotation_button = QPushButton("アノテーションデータを読込")
        self.load_annotation_button.clicked.connect(self.load_annotations)
        self.load_annotation_button.setEnabled(False)  # 初期状態は無効
        apply_style(self.load_annotation_button, 'primary')
        load_button_layout.addWidget(self.load_annotation_button)

        left_layout.addLayout(load_button_layout)

        # Stats
        self.stats_label = QLabel("アノテーション済み: 0 / 0")
        left_layout.addWidget(self.stats_label)
                
        # --- 統計ラベル直後にキー選択用ラジオボタン群を追加 ---
        variants = self.available_variants
        variant_box = QGroupBox("画像ソース")
        # variant_box = QGroupBox("画像ソース:idx_sensor_image_array_.jpg（学習に用いられます）")
        variant_layout = QHBoxLayout(variant_box)
        # 排他制御用のボタングループ
        self.variant_button_group = QButtonGroup(self)
        self.variant_button_group.setExclusive(True)
        for var in variants:
            rb = QRadioButton(var)
            variant_layout.addWidget(rb)
            self.variant_button_group.addButton(rb)
            if var == self.current_variant:
                rb.setChecked(True)
            # 切替時、チェックされた変化のみ受け取る
            rb.toggled.connect(lambda checked, v=var: self.on_variant_changed(v) if checked else None)
        left_layout.addWidget(variant_box)

        # エクスポートセクション
        save_label = QLabel("保存:")
        save_label.setStyleSheet("font-weight: bold;")  # 太文字にするスタイルを追加
        left_layout.addWidget(save_label)

        export_layout = QHBoxLayout()
        
        # donkey保存
        donkey_btn = QPushButton("Donkey形式")
        donkey_btn.clicked.connect(self.export_to_donkey)
        apply_style(donkey_btn, 'export')
        export_layout.addWidget(donkey_btn)

        # jetracer保存保存
        jetracer_btn = QPushButton("Jetracr形式")
        jetracer_btn.clicked.connect(self.export_to_jetracer)
        apply_style(jetracer_btn, 'export')
        export_layout.addWidget(jetracer_btn)

        # YOLOフォーマット保存ボタンを追加
        yolo_btn = QPushButton("YOLO形式")
        yolo_btn.clicked.connect(self.export_to_yolo)
        apply_style(yolo_btn, 'export')
        export_layout.addWidget(yolo_btn)

        left_layout.addLayout(export_layout)

        ## 動画作成ボタン
        create_video_button = QPushButton("アノテーション動画作成")
        create_video_button.clicked.connect(self.create_annotation_video)
        left_layout.addWidget(create_video_button)
        

        # 自動運転コンテナ
        self.pilot_container = QWidget()
        pilot_layout = QVBoxLayout(self.pilot_container)

        # 学習モード
        pilot_label = QLabel("自動運転:")
        pilot_label.setStyleSheet("font-weight: bold;")  # 太文字にするスタイルを追加
        pilot_layout.addWidget(pilot_label)

        # 学習方法選択
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("走行モデル選択:"))
        self.auto_method_combo = QComboBox()

        # 利用可能なモデルのリストを取得
        available_models = list_available_models()

        # コンボボックスのアイテムをモデルリストで初期化
        self.auto_method_combo.addItems(available_models)

        self.auto_method_combo.currentIndexChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.auto_method_combo)
        pilot_layout.addLayout(method_layout)

        # モデル選択コンボボックス（1行使用）
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)  # 幅を広げて見やすく
        self.model_combo.setStyleSheet("combobox-popup: 0;")  # ドロップダウンリストの高さを自動調整
        pilot_layout.addWidget(self.model_combo)

        # モデル操作ボタン（更新と読み込み - 横並び）
        model_buttons_layout = QHBoxLayout()

        # モデルリスト更新ボタン
        self.model_refresh_button = QPushButton("モデル一覧更新")
        self.model_refresh_button.clicked.connect(self.refresh_model_list)
        model_buttons_layout.addWidget(self.model_refresh_button)

        # モデル明示的読み込みボタン
        self.model_load_button = QPushButton("モデル読込")
        self.model_load_button.setToolTip("modelsフォルダのモデルを読込む")
        self.model_load_button.clicked.connect(self.load_selected_model)
        apply_style(self.model_load_button, 'model')
        model_buttons_layout.addWidget(self.model_load_button)

        pilot_layout.addLayout(model_buttons_layout)

        # モデル学習ボタン（自動運転用）
        train_model_button = QPushButton("学習・保存")
        train_model_button.clicked.connect(self.train_and_save_model)
        apply_style(train_model_button, 'training')
        pilot_layout.addWidget(train_model_button)  

        # 推論結果表示オプション
        inference_layout = QHBoxLayout()
        self.inference_checkbox = QCheckBox("推論結果表示（青丸）")
        self.inference_checkbox.setChecked(False)
        self.inference_checkbox.stateChanged.connect(self.toggle_inference_display)
        inference_layout.addWidget(self.inference_checkbox)


        # 一括推論実行ボタンを追加
        batch_inference_button = QPushButton("全画像の推論実行")
        batch_inference_button.setToolTip("全ての画像に対して推論を実行します")
        batch_inference_button.clicked.connect(self.run_batch_inference)
        inference_layout.addWidget(batch_inference_button)

        pilot_layout.addLayout(inference_layout)

        left_layout.addWidget(self.pilot_container)
                    
        # 物体検知推論結果表示フラグの初期化
        self.show_detection_inference = False

        # 物体検知推論結果格納用の辞書を初期化
        self.detection_inference_results = {}

        # 物体検知推論結果表示用ラベルを作成
        self.detection_inference_info_label = QLabel("")
        self.detection_inference_info_label.setStyleSheet("color: blue;")
        self.detection_inference_info_label.setWordWrap(True)

        # 推論実行ボタン
        inference_button_layout = QHBoxLayout()

        # オートアノテーションボタン
        auto_annotate_button = QPushButton("オートアノテーション実行")
        auto_annotate_button.clicked.connect(self.auto_annotate)
        apply_style(auto_annotate_button, 'special')
        inference_button_layout.addWidget(auto_annotate_button)

        left_layout.addLayout(inference_button_layout)


        # 物体検知設定コンテナ
        self.object_detection_container = QWidget()
        obj_detection_layout = QVBoxLayout(self.object_detection_container)

        ### 位置推論モデル追加
        self.add_location_model_section()


        # ラベル
        obj_detection_label = QLabel("物体検知:")
        obj_detection_label.setStyleSheet("font-weight: bold")
        obj_detection_layout.addWidget(obj_detection_label)

        # YOLOアノテーション読み込みボタン
        load_yolo_btn = QPushButton("YOLOアノテーション読込")
        load_yolo_btn.clicked.connect(self.load_yolo_annotations)
        apply_style(load_yolo_btn, 'primary')
        obj_detection_layout.addWidget(load_yolo_btn)

        # クラス設定
        classes_layout = QHBoxLayout()
        classes_layout.addWidget(QLabel("検知クラス:"))
        self.classes_input = QLineEdit("car,person,sign,cone")
        self.classes_input.setPlaceholderText("カンマ区切りでクラス名を入力")
        classes_layout.addWidget(self.classes_input)
        obj_detection_layout.addLayout(classes_layout)

        # モデルタイプ選択
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("YOLOモデル:"))
        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        self.yolo_model_combo.currentIndexChanged.connect(self.on_yolo_model_type_changed)
        model_type_layout.addWidget(self.yolo_model_combo)
        obj_detection_layout.addLayout(model_type_layout)

        # 学習済みYOLOモデル選択コンボボックス
        self.yolo_saved_model_combo = QComboBox()
        self.yolo_saved_model_combo.setMinimumWidth(180)
        self.yolo_saved_model_combo.setStyleSheet("combobox-popup: 0;")
        obj_detection_layout.addWidget(self.yolo_saved_model_combo)

        # 5. モデル操作ボタン（更新と読み込み - 横並び）
        yolo_model_buttons_layout = QHBoxLayout()

        # モデルリスト更新ボタン
        self.yolo_refresh_button = QPushButton("モデル一覧更新")
        self.yolo_refresh_button.clicked.connect(self.refresh_yolo_model_list)
        yolo_model_buttons_layout.addWidget(self.yolo_refresh_button)

        # モデル読み込みボタン
        self.yolo_load_button = QPushButton("モデル読込")
        self.yolo_load_button.setToolTip("modelsフォルダのモデルを読込む")
        self.yolo_load_button.clicked.connect(self.load_yolo_model)
        apply_style(self.yolo_load_button, 'model')
        yolo_model_buttons_layout.addWidget(self.yolo_load_button)

        obj_detection_layout.addLayout(yolo_model_buttons_layout)

        # 物体検知モデル学習ボタン
        train_yolo_button = QPushButton("YOLOモデルを学習・保存")
        train_yolo_button.clicked.connect(self.train_and_save_yolo_model)
        apply_style(train_yolo_button, 'training')
        obj_detection_layout.addWidget(train_yolo_button)

        # 物体検知推論結果表示オプション
        detection_inference_layout = QHBoxLayout()
        self.detection_inference_checkbox = QCheckBox("物体検知推論結果表示")
        self.detection_inference_checkbox.setChecked(False)
        self.detection_inference_checkbox.stateChanged.connect(self.toggle_detection_inference_display)
        detection_inference_layout.addWidget(self.detection_inference_checkbox)
        obj_detection_layout.addLayout(detection_inference_layout)
        
        # 物体検知コンテナ追加
        left_layout.addWidget(self.object_detection_container)

        # --- MLflow関連ボタンを追加 ---
        mlflow_layout = QVBoxLayout()

        mlflow_label = QLabel("MLflow:")
        mlflow_label.setStyleSheet("font-weight: bold;")  # 太文字にするスタイルを追加
        mlflow_layout.addWidget(mlflow_label)

        # MLflow比較ボタン
        mlflow_compare_button = QPushButton("モデルのパラメータと性能を比較")
        apply_style(mlflow_compare_button, 'special')
        mlflow_compare_button.clicked.connect(self.compare_models_mlflow)
        mlflow_layout.addWidget(mlflow_compare_button)

        left_layout.addLayout(mlflow_layout)

        self.on_method_changed(self.auto_method_combo.currentIndex())

        # Current image info
        left_layout.addWidget(QLabel(""))  # Spacer
        self.current_image_label = QLabel("画像が選択されていません")
                        
        # ステータスバー
        self.statusBar().showMessage("Bキーを押しながらクリックすると、いつでもバウンディングボックスを作成できます。Deleteキーで選択したボックスを削除できます。", 10000)

        # 最後にスペーサーを追加
        left_layout.addStretch()
        
        # Right panel for images
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel)
        
        # メイン画像と位置情報パネルを横に並べるレイアウト - 1:4:1の比率に変更
        main_panel_layout = QHBoxLayout()
        
        # 1. 左側の情報パネル（アノテーション情報表示用）- 追加
        info_panel = QWidget()
        info_panel.setObjectName("info_panel")  # スタイルシート適用用
        info_panel.setStyleSheet("#info_panel { background-color: rgba(0, 0, 0, 0.1); border-radius: 5px; }")
        info_layout = QVBoxLayout(info_panel)
        info_layout.setSpacing(8)  # スペーシングを調整
        
        # 情報パネルの内容
        self.current_image_info = QLabel("画像情報")
        self.current_image_info.setStyleSheet("color: #333333; font-weight: bold;")
        info_layout.addWidget(self.current_image_info)
        
        self.annotation_info_label = QLabel("")
        self.annotation_info_label.setWordWrap(True)  # テキスト折り返し
        info_layout.addWidget(self.annotation_info_label)
        
        self.inference_info_label = QLabel("")
        self.inference_info_label.setWordWrap(True)
        self.inference_info_label.setStyleSheet("color: blue;")
        info_layout.addWidget(self.inference_info_label)

        # 物体検知推論結果表示ラベル
        self.detection_inference_info_label = QLabel("")
        self.detection_inference_info_label.setWordWrap(True)
        self.detection_inference_info_label.setStyleSheet("color: green;")  # 緑色で表示して区別
        info_layout.addWidget(self.detection_inference_info_label)
        
        # 空白を下に追加
        # info_layout.addStretch()

        # 上部のウィジェットと分布グラフの間にスペーサーを入れる
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        info_layout.addWidget(spacer)

        # 分布タイトルラベル
        graph_title = QLabel("運転アノテーション分布")
        graph_title.setStyleSheet("font-weight: bold; color: #333333;")
        graph_title.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(graph_title)

        # 分布グラフ用ラベル - 固定サイズで配置
        self.distribution_label = QLabel()
        self.distribution_label.setAlignment(Qt.AlignCenter)
        self.distribution_label.setFixedHeight(400)  # 高さを固定
        self.distribution_label.setStyleSheet("background-color: #f8f8f8; border: 1px solid #dddddd; border-radius: 4px;")

        # 初期表示テキストの設定
        no_data_font = QFont()
        no_data_font.setPointSize(10)
        self.distribution_label.setFont(no_data_font)
        self.distribution_label.setText("アノテーションがありません")

        info_layout.addWidget(self.distribution_label)

        # パネルのサイズ設定を記録（グラフ作成時に使用）
        self.info_panel_width = 200  # 基本の幅
        self.info_panel_margin = 0  # パネル周りの余白（左右合計）

        # パネルの最小幅を設定
        info_panel.setMinimumWidth(self.info_panel_width)
        
        # パネルのサイズ設定
        # info_panel.setMinimumWidth(200)  # 最小幅
        main_panel_layout.addWidget(info_panel, 1)  # 比率1
        
        # 2. 中央の画像パネル - 既存のmain_image_containerをそのまま利用
        # メインイメージの周りにマージンを調整 - 左側マージンを0に変更（情報パネルを別ウィジェットにしたため）
        main_image_container = QVBoxLayout()
        main_image_container.setContentsMargins(0, 0, 0, 0)  # マージンを0に変更
        
        self.main_image_view = ImageLabel(main_window=self)
        self.main_image_view.setMinimumSize(800, 600)
        main_image_container.addWidget(self.main_image_view)
        
        # ナビゲーションコントロールをメイン画像の下に配置
        nav_container = QWidget()
        nav_container_layout = QVBoxLayout(nav_container)
        
        # スライダーの配置
        slider_layout = QHBoxLayout()
        slider_label = QLabel("画像シーク:")
        slider_layout.addWidget(slider_label)
        
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.setMaximum(0)  # 初期値は0（画像が読み込まれたら更新）
        self.image_slider.setValue(0)
        self.image_slider.setTickPosition(QSlider.TicksBelow)
        self.image_slider.setTickInterval(10)
        self.image_slider.valueChanged.connect(self.slider_changed)
        slider_layout.addWidget(self.image_slider)
        
        self.slider_value_label = QLabel("0/0")
        slider_layout.addWidget(self.slider_value_label)
        
        nav_container_layout.addLayout(slider_layout)
        
        # ナビゲーションボタンの配置
        nav_layout = QHBoxLayout()
        
        self.prev_multi_button = QPushButton("◀◀")  # 早戻しマーク
        self.prev_multi_button.clicked.connect(lambda: self.skip_images(-self.skip_count_spin.value()))
        nav_layout.addWidget(self.prev_multi_button)
        
        prev_button = QPushButton("◀ 前へ")
        prev_button.clicked.connect(lambda: self.skip_images(-1))
        nav_layout.addWidget(prev_button)
        
        next_button = QPushButton("次へ ▶")
        next_button.clicked.connect(lambda: self.skip_images(1))
        nav_layout.addWidget(next_button)
        
        self.next_multi_button = QPushButton("▶▶")  # 早送りマーク
        self.next_multi_button.clicked.connect(lambda: self.skip_images(self.skip_count_spin.value()))
        nav_layout.addWidget(self.next_multi_button)
        
        nav_container_layout.addLayout(nav_layout)
        
        # 再生ボタンの配置
        play_layout = QHBoxLayout()
        
        play_layout.addWidget(QLabel("再生:"))
        reverse_play_button = QPushButton("◀️")
        reverse_play_button.clicked.connect(self.play_reverse)
        play_layout.addWidget(reverse_play_button)
        
        play_button = QPushButton("▶️")
        play_button.clicked.connect(self.play_forward)
        play_layout.addWidget(play_button)
        
        nav_container_layout.addLayout(play_layout)

        # 削除機能を追加 - 1. 現在のアノテーションを削除するボタン
        delete_layout = QHBoxLayout()
        delete_layout.addWidget(QLabel("削除/復元:"))

        delete_current_button = QPushButton("現在のアノテーション削除")
        delete_current_button.clicked.connect(self.delete_current_annotation)
        apply_style(delete_current_button, "destructive")
        delete_layout.addWidget(delete_current_button)

        # 復元ボタンを追加
        restore_button = QPushButton("削除状態を復元")
        restore_button.clicked.connect(self.restore_deleted_annotation)
        delete_layout.addWidget(restore_button)

        # 全ての削除状態を復元するボタンを追加
        restore_all_button = QPushButton("全ての削除状態を復元")
        restore_all_button.clicked.connect(self.restore_all_deleted_annotations)
        delete_layout.addWidget(restore_all_button)

        nav_container_layout.addLayout(delete_layout)

        # 削除機能を追加 - 2. クリップ機能（範囲指定削除）- ここを修正
        clip_layout = QHBoxLayout()
        clip_layout.addWidget(QLabel("クリップ範囲:"))

        # クリップ開始位置入力と「現在位置を設定」ボタン
        start_layout = QHBoxLayout()
        self.clip_start_spin = QSpinBox()
        self.clip_start_spin.setRange(0, 99999)
        self.clip_start_spin.setValue(0)
        start_layout.addWidget(self.clip_start_spin)

        self.set_start_button = QPushButton("現在位置")
        self.set_start_button.clicked.connect(self.set_clip_start_to_current)
        self.set_start_button.setToolTip("現在のインデックスを開始位置に設定")
        start_layout.addWidget(self.set_start_button)
        clip_layout.addLayout(start_layout)

        clip_layout.addWidget(QLabel("から"))

        # クリップ終了位置入力と「現在位置を設定」ボタン
        end_layout = QHBoxLayout()
        self.clip_end_spin = QSpinBox()
        self.clip_end_spin.setRange(0, 99999)
        self.clip_end_spin.setValue(0)
        end_layout.addWidget(self.clip_end_spin)

        self.set_end_button = QPushButton("現在位置")
        self.set_end_button.clicked.connect(self.set_clip_end_to_current)
        self.set_end_button.setToolTip("現在のインデックスを終了位置に設定")
        end_layout.addWidget(self.set_end_button)
        clip_layout.addLayout(end_layout)

        clip_button = QPushButton("範囲削除")
        clip_button.clicked.connect(self.delete_clip_range)
        apply_style(clip_button, "destructive")
        clip_layout.addWidget(clip_button)

        nav_container_layout.addLayout(clip_layout)


        # ナビゲーションコンテナをメイン画像コンテナに追加
        main_image_container.addWidget(nav_container)
        
        # 中央パネルをメインパネルに追加 - 比率4に設定
        main_panel_layout.addLayout(main_image_container, 4)
        
        # 3. 右側の位置情報パネル - 既存のright_layoutをそのまま利用、比率1に設定
        location_panel = QWidget()
        location_layout = QVBoxLayout(location_panel)
        location_layout.setSpacing(5)
        
        #m
        # テーマ切り替えエリア
        theme_layout = QHBoxLayout()
        theme_layout.setSpacing(5)

        theme_label = QLabel("テーマ:")
        theme_label.setStyleSheet("font-weight: bold;")
        theme_layout.addWidget(theme_label)

        # テーマ切り替えボタン
        self.light_theme_button = QPushButton("ライト")
        self.light_theme_button.setCheckable(True)
        self.light_theme_button.setChecked(True)  # デフォルトはライトテーマ
        self.light_theme_button.clicked.connect(lambda: self.toggle_theme("light"))
        apply_style(self.light_theme_button, 'primary')  # スタイル適用
        theme_layout.addWidget(self.light_theme_button)

        self.dark_theme_button = QPushButton("ダーク")
        self.dark_theme_button.setCheckable(True)
        self.dark_theme_button.setChecked(False)
        self.dark_theme_button.clicked.connect(lambda: self.toggle_theme("dark"))
        apply_style(self.dark_theme_button, 'primary')  # スタイル適用
        theme_layout.addWidget(self.dark_theme_button)

        # グループ化してボタンを排他的にする
        self.theme_button_group = QButtonGroup(self)
        self.theme_button_group.addButton(self.light_theme_button)
        self.theme_button_group.addButton(self.dark_theme_button)
        self.theme_button_group.setExclusive(True)

        # レイアウトに追加
        location_layout.addLayout(theme_layout)
        

        mode_layout_label = QLabel("アノテーションモード:")
        mode_layout_label.setStyleSheet("font-weight: bold;")
        location_layout.addWidget(mode_layout_label)

        # アノテーションモード切替ボタン
        mode_layout = QHBoxLayout()
        
        self.auto_mode_button = QPushButton("自動運転")
        self.auto_mode_button.setCheckable(True)
        self.auto_mode_button.setChecked(True)  # デフォルトは自動運転モード
        self.auto_mode_button.clicked.connect(self.toggle_annotation_mode)
        self.auto_mode_button.setStyleSheet("""
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)
        mode_layout.addWidget(self.auto_mode_button)

        self.detection_mode_button = QPushButton("物体検知")
        self.detection_mode_button.setCheckable(True)
        self.detection_mode_button.setChecked(False)  # 初期状態では未選択
        self.detection_mode_button.clicked.connect(self.toggle_annotation_mode)
        self.detection_mode_button.setStyleSheet("""
            QPushButton:checked {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
            }
        """)
        mode_layout.addWidget(self.detection_mode_button)

        location_layout.addLayout(mode_layout)

        # 現在のモードを表すヒントラベル
        self.mode_hint_label = QLabel("※Bキーを押すとモードが切り替わります")
        self.mode_hint_label.setStyleSheet("color: #666; font-style: italic;")
        location_layout.addWidget(self.mode_hint_label)

        # 前回のバウンディングボックスを自動適用するチェックボックス
        self.apply_last_bbox_checkbox = QCheckBox("前回のバウンディングボックスを適用")
        self.apply_last_bbox_checkbox.setChecked(False)
        self.apply_last_bbox_checkbox.setToolTip("前回作成したバウンディングボックスを現在の画像にも適用します")
        self.apply_last_bbox_checkbox.stateChanged.connect(self.toggle_auto_apply_bbox)
        location_layout.addWidget(self.apply_last_bbox_checkbox)

        # スキップ枚数設定
        skip_layout = QHBoxLayout()
        self.skip_images_on_click = QCheckBox("クリック時自動スキップ枚数")
        self.skip_images_on_click.setChecked(True)  # デフォルトでオン
        skip_layout.addWidget(self.skip_images_on_click)
        self.skip_count_spin = QSpinBox()
        self.skip_count_spin.setRange(1, 1000)
        self.skip_count_spin.setValue(10)  # デフォルト値は10
        self.skip_count_spin.valueChanged.connect(self.update_skip_button_labels)
        skip_layout.addWidget(self.skip_count_spin)

        location_layout.addLayout(skip_layout)

        location_label = QLabel("コースの位置情報:")
        location_label.setStyleSheet("font-weight: bold;")
        location_layout.addWidget(location_label)

        # 位置情報の自動適用チェックボックス
        self.apply_location_checkbox = QCheckBox("前回の位置情報を適用")
        self.apply_location_checkbox.setChecked(False)
        self.apply_location_checkbox.setToolTip("前回選択した位置情報を現在の画像にも適用します")
        self.apply_location_checkbox.stateChanged.connect(self.toggle_auto_apply_location)
        location_layout.addWidget(self.apply_location_checkbox)
        
        # 位置情報の選択肢を管理するレイアウト
        self.location_buttons_layout = QVBoxLayout()
        location_layout.addLayout(self.location_buttons_layout)
        
        # 位置情報の追加ボタン
        add_location_layout = QHBoxLayout()
        self.new_location_input = QSpinBox()
        self.new_location_input.setRange(0, 100)
        self.new_location_input.setValue(8)  # 初期値を8に設定（8個作成後）
        add_location_layout.addWidget(self.new_location_input)
        
        add_location_button = QPushButton("位置情報を追加")
        add_location_button.clicked.connect(self.add_location_button)
        add_location_layout.addWidget(add_location_button)
        location_layout.addLayout(add_location_layout)
        
        # 現在の位置情報表示ラベル
        self.current_location_label = QLabel("現在の位置情報: なし")
        location_layout.addWidget(self.current_location_label)
        
        # スペーサーを追加して上部に配置
        location_layout.addStretch()
        
        # 位置情報パネルをメインパネルに追加
        main_panel_layout.addWidget(location_panel, 1)  # 比率1に設定
        
        # メインパネルをレイアウトに追加
        right_layout.addLayout(main_panel_layout)

        # Gallery
        gallery_label = QLabel("ギャラリー:")
        right_layout.addWidget(gallery_label)
        
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_layout.setSpacing(2)
        
        gallery_scroll = QScrollArea()
        gallery_scroll.setWidgetResizable(True)
        gallery_scroll.setWidget(self.gallery_widget)
        gallery_scroll.setMinimumHeight(200)
        right_layout.addWidget(gallery_scroll)
        
        # 位置情報ボタンを初期化（8個作成）
        self.init_location_buttons()
        
        # キーボードイベント用のフォーカス設定
        self.setFocusPolicy(Qt.StrongFocus)

        # 削除インデックス
        self.deleted_indexes = []

        # 現在の位置情報を初期化（明示的に None に設定）
        self.current_location = None
        self.auto_apply_location = False  # 位置情報の自動適用フラグ

        # 初期状態の設定
        self.current_mode = 0  # 自動運転モード
        self.auto_mode_button.setChecked(True)
        self.detection_mode_button.setChecked(False)

        # 物体検知アノテーションの表示追加
        try:
            apply_enhanced_annotations_display(self)
        except Exception as e:
            print(f"物体検知アノテーション表示拡張の適用に失敗しました: {e}")

        self.update_theme_button_styles()  # 初期スタイルを適用

    # 初期化/ファイル読込
    def save_session_info(self):
        """現在の作業セッション情報を保存する"""
        try:            
            # 保存する情報
            session_info = {
                "last_folder_path": self.folder_path if hasattr(self, 'folder_path') else "",
                "last_folder_paths": self.folder_paths if hasattr(self, 'folder_paths') else [],
                "last_model_arch": self.auto_method_combo.currentText() if hasattr(self, 'auto_method_combo') else "",
                "last_model_name": self.model_combo.currentText() if hasattr(self, 'model_combo') else "",
                "timestamp": int(time.time())
            }
            
            # ファイルに保存
            session_file = os.path.join(session_dir, "session.json")
            with open(session_file, 'w') as f:
                json.dump(session_info, f)
                
            print(f"セッション情報を保存しました: {session_file}")
        except Exception as e:
            print(f"セッション情報の保存に失敗: {e}")

    def load_session_info(self):
        """保存されたセッション情報を読み込む"""
        try:
            # セッション情報ファイルのパス
            session_dir = os.path.join(APP_DIR_PATH, SESSION_DIR_NAME)
            session_file = os.path.join(session_dir, "session.json")
            
            # ファイルが存在しない場合は空の情報を返す
            if not os.path.exists(session_file):
                return {}
            
            # ファイルから読み込み
            with open(session_file, 'r') as f:
                session_info = json.load(f)
                
            print(f"セッション情報を読み込みました: {session_file}")
            return session_info
        except Exception as e:
            print(f"セッション情報の読み込みに失敗: {e}")
            return {}

    def update_distribution_graph(self):
        """アノテーションの角度とスロットル値の分布を縦並びのヒストグラムで表示"""
        
        if not self.annotations:
            # アノテーションがない場合は空のグラフを表示
            self.distribution_label.clear()
            self.distribution_label.setText("アノテーションがありません")
            return
        
        # 既存のアノテーションからangleとthrottleの値を抽出
        angles = []
        throttles = []
        
        for img_path, anno in self.annotations.items():
            if 'angle' in anno and 'throttle' in anno:
                angles.append(anno['angle'])
                throttles.append(anno['throttle'])
        
        # データがない場合は終了
        if not angles or not throttles:
            self.distribution_label.clear()
            self.distribution_label.setText("有効なアノテーションがありません")
            return
        
        # グラフ作成
        try:
            # 情報パネルの幅に合わせてグラフサイズを調整
            panel_width = self.info_panel_width - self.info_panel_margin
            # 縦に2つのグラフを配置（縦長）
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(panel_width/100, 4))
            
            # カラーマップとヒストグラムの設定
            cm = plt.get_cmap('viridis')
            bins = 20
            
            # angle分布をヒストグラムで表示
            n1, bins1, patches1 = ax1.hist(angles, bins=bins, alpha=0.7, color='skyblue')
            # 度数に応じた色付け
            bin_centers1 = 0.5 * (bins1[:-1] + bins1[1:])
            col1 = bin_centers1 - min(bin_centers1)
            col1 = col1 / max(col1) if max(col1) > 0 else col1
            for c, p in zip(col1, patches1):
                plt.setp(p, 'facecolor', cm(c))
            
            # スタイル設定
            ax1.set_title('Angle dist', fontsize=10)
            # ax1.set_xlabel('Angle値 (-1.0～1.0)', fontsize=8)
            # ax1.set_ylabel('頻度', fontsize=8)
            ax1.tick_params(axis='both', which='major', labelsize=7)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-1.05, 1.05)
                        
            # throttle分布をヒストグラムで表示
            n2, bins2, patches2 = ax2.hist(throttles, bins=bins, alpha=0.7, color='salmon')
            # 度数に応じた色付け
            bin_centers2 = 0.5 * (bins2[:-1] + bins2[1:])
            col2 = bin_centers2 - min(bin_centers2)
            col2 = col2 / max(col2) if max(col2) > 0 else col2
            for c, p in zip(col2, patches2):
                plt.setp(p, 'facecolor', cm(c))
            
            # スタイル設定
            ax2.set_title('Throttle dist', fontsize=10)
            # ax2.set_xlabel('Throttle値 (-1.0～1.0)', fontsize=8)
            # ax2.set_ylabel('頻度', fontsize=8)
            ax2.tick_params(axis='both', which='major', labelsize=7)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-1.05, 1.05)
                        
            # レイアウト調整
            plt.tight_layout(pad=1.0)
            
            # メモリ上にグラフを保存
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # QImageに変換してQPixmapに設定
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            
            # ラベルにセット
            self.distribution_label.setPixmap(pixmap)
            self.distribution_label.setScaledContents(True)
            
            # 後始末
            plt.close(fig)
            
        except Exception as e:
            print(f"分布グラフ作成中にエラー: {str(e)}")
            traceback.print_exc()
            self.distribution_label.setText(f"グラフ作成エラー: {str(e)}")

    def eventFilter(self, obj, event):
        # キーイベントを処理
        if event.type() == QEvent.KeyPress and hasattr(self, 'main_image_view') and self.main_image_view.selected_bbox_index is not None:
            key = event.key()
            if key in [Qt.Key_Delete, Qt.Key_Backspace]:
                # 削除キーが押された場合、選択されたバウンディングボックスを削除
                self.delete_selected_bbox()
                return True  # イベントを消費
        
        # 親クラスのイベントフィルタを呼び出す
        return super().eventFilter(obj, event)

    def delete_selected_bbox(self):
        """選択されたバウンディングボックスを削除する"""
        if not self.images:
            return
        
        current_img_path = self.images[self.current_index]
        selected_index = self.main_image_view.selected_bbox_index
        
        if current_img_path in self.bbox_annotations and selected_index is not None:
            bboxes = self.bbox_annotations[current_img_path]
            if 0 <= selected_index < len(bboxes):
                # 確認ダイアログ
                bbox = bboxes[selected_index]
                class_name = bbox.get('class', 'unknown')
                
                reply = QMessageBox.question(
                    self, 
                    "バウンディングボックス削除", 
                    f"選択された '{class_name}' のバウンディングボックスを削除しますか？",
                    QMessageBox.Yes | QMessageBox.No, 
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # バウンディングボックスを削除
                    del bboxes[selected_index]
                    # 選択インデックスをリセット
                    self.main_image_view.selected_bbox_index = None
                    # 画面更新
                    self.main_image_view.update()
                    # 統計情報更新
                    self.update_bbox_stats()

    def update_bbox_stats(self):
        """
        Update statistics for bounding box annotations
        """
        # Count the number of bounding box annotations
        bbox_count = len(self.bbox_annotations) if hasattr(self, 'bbox_annotations') else 0
        
        # Update the stats label to show bounding box count
        if hasattr(self, 'stats_label'):
            self.stats_label.setText(f"Bounding Boxes: {bbox_count} / {len(self.images)}")
            
    def add_detection_inference_controls(self):
        """物体検知推論表示コントロールを追加"""
        # 推論結果表示オプションを配置する既存のレイアウトを探す
        inference_layout = None
        
        # 既存の推論結果表示オプションの後に追加
        if hasattr(self, 'inference_checkbox'):
            inference_parent = self.inference_checkbox.parent()
            if inference_parent:
                # 親ウィジェットからレイアウトを取得
                parent_layout = inference_parent.layout()
                
                # 親レイアウトが見つかった場合、同じ階層に新しいレイアウトを追加
                if parent_layout:
                    # 物体検知推論結果表示チェックボックス
                    detection_inference_layout = QHBoxLayout()
                    self.detection_inference_checkbox = QCheckBox("物体検知推論結果表示")
                    self.detection_inference_checkbox.setChecked(False)
                    self.detection_inference_checkbox.stateChanged.connect(self.toggle_detection_inference_display)
                    detection_inference_layout.addWidget(self.detection_inference_checkbox)
                    
                    # レイアウトに追加
                    parent_layout.addLayout(detection_inference_layout)
        
        # 物体検知推論結果表示フラグの初期化
        self.show_detection_inference = False
        
        # 物体検知推論結果格納用の辞書を初期化
        self.detection_inference_results = {}

    def toggle_detection_inference_display(self, state):
        """物体検知推論表示の切り替え"""
        show_inference = (state == Qt.Checked)
        print(f"物体検知推論表示切替: {show_inference} (state={state}, Qt.Checked={Qt.Checked})")

        self.show_detection_inference = show_inference

        # 画面更新
        self.update_detection_info_panel()        
        self.main_image_view.update()
        
        # 表示状態をステータスバーに反映
        if show_inference:
            self.statusBar().showMessage("物体検知推論結果表示をオンにしました", 3000)
        else:
            self.statusBar().showMessage("物体検知推論結果表示をオフにしました", 3000)
    
    def toggle_location_inference_display(self, state):
        """位置推論表示の切り替え"""
        show_inference = (state == Qt.Checked)
        print(f"位置推論表示切替: {show_inference} (state={state}, Qt.Checked={Qt.Checked})")

        self.show_location_inference = show_inference

        # 画面更新
        self.update_location_info_panel()
        self.main_image_view.update()
        
        # 表示状態をステータスバーに反映
        if show_inference:
            self.statusBar().showMessage("位置推論結果表示をオンにしました", 3000)
        else:
            self.statusBar().showMessage("位置推論結果表示をオフにしました", 3000)

    # mlflow関連
    def initialize_mlflow(self):
        """MLflowの初期化と設定を行う - Windows環境対応（修正版）"""
        
        # MLflowのトラッキングサーバーの設定
        if not hasattr(self, 'folder_path') or not self.folder_path:
            QMessageBox.warning(self, "警告", "画像フォルダが設定されていません。MLflowの初期化ができません。")
            return False
        
        try:            
            # Windows環境での正しいURI形式を構築
            # Windows: file:///C:/path/to/dir (スラッシュ3つ)
            # Unix: file:/path/to/dir (スラッシュ2つ)
            if sys.platform.startswith('win'):
                tracking_uri = f"file:///{normalized_path}"
            else:
                tracking_uri = f"file://{normalized_path}"
            
            print(f"トラッキングURI: {tracking_uri}")
            mlflow.set_tracking_uri(tracking_uri)
            
            # 実験名を設定
            experiment_name = "minicar_model_training"
            
            # 実験が存在するか確認し、なければ作成
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            
            # 実験を設定
            mlflow.set_experiment(experiment_name)
            
            self.mlflow_tracking_uri = tracking_uri
            print(f"MLflow初期化成功: {tracking_uri}")
            return True
            
        except Exception as e:
            print(f"MLflow初期化エラー: {e}")
            QMessageBox.warning(
                self, 
                "MLflow初期化エラー", 
                f"MLflowの初期化中にエラーが発生しました: {str(e)}\n"
                "この機能を使用するにはMLflowをインストールしてください: pip install mlflow"
            )
            return False

    def open_mlflow_ui(self):
        """MLflow UIを開く - Windows環境対応（パス正規化）"""
        
        if not hasattr(self, 'mlflow_tracking_uri'):
            if not self.initialize_mlflow():
                return
        
        try:
            # トラッキングURIを取得（すでに正規化されているはず）
            tracking_uri = self.mlflow_tracking_uri
            
            # 環境に応じてコマンドを構築
            if sys.platform.startswith('win'):  # Windows
                # Windows環境でのMLflow UI起動コマンド
                # cmd /k でコマンド実行後もウィンドウを残す
                # クォーテーションの処理に注意
                cmd = f'start cmd /k "mlflow ui --backend-store-uri {tracking_uri}"'
                print(f"実行コマンド: {cmd}")
                subprocess.Popen(cmd, shell=True)
            else:  # Mac/Linux
                # Unix系環境でのMLflow UI起動コマンド
                cmd = f'mlflow ui --backend-store-uri {tracking_uri}'
                subprocess.Popen(cmd, shell=True)
            
            QMessageBox.information(
                self, 
                "MLflow UI", 
                "MLflow UIを起動しました。ブラウザで http://localhost:5000 にアクセスして実験結果を確認できます。\n"
                "UIを終了するには、コマンドウィンドウを閉じてください。"
            )
        except Exception as e:
            error_msg = str(e)
            print(f"MLflow UI起動エラー: {error_msg}")
            
            # エラーメッセージに応じたヒントを提供
            hint = ""
            if "No such file or directory" in error_msg:
                hint = "\n\nヒント: MLflow CLIがインストールされていない可能性があります。"
            elif "mlflow: command not found" in error_msg:
                hint = "\n\nヒント: MLflowがPATHに含まれていないか、インストールされていない可能性があります。"
            
            QMessageBox.critical(
                self, 
                "エラー", 
                f"MLflow UIの起動に失敗しました: {error_msg}{hint}\n\n"
                "MLflowがインストールされているか確認してください: pip install mlflow"
            )

    def log_model_to_mlflow(self, model_path, model_type, training_params, metrics, dataset_info):
        """モデル情報をMLflowに記録する - Windows環境対応"""
        
        # MLflowが初期化されていない場合は初期化
        if not hasattr(self, 'mlflow_tracking_uri'):
            if not self.initialize_mlflow():
                return False
        
        # 学習に使用したモデルパラメータ
        params = {
            "model_type": model_type,
            "epochs": training_params.get("num_epochs", 0),
            "completed_epochs": training_params.get("completed_epochs", 0),
            "learning_rate": training_params.get("learning_rate", 0.001),
            "batch_size": training_params.get("batch_size", 32),
            "early_stopping": "enabled" if training_params.get("use_early_stopping", False) else "disabled",
            "patience": training_params.get("patience", 0),
            "weight_decay": training_params.get("weight_decay", 1e-4),
            # 修正: training_paramsから直接オーグメンテーション情報を取得
            "augmentation": "enabled" if training_params.get("augmentation_enabled", False) else "disabled",
            "sampling_strategy": training_params.get("sampling_strategy", "all"),
            "initial_weights": training_params.get("initial_weights", "pretrained"),   
            "augmentation": "enabled" if training_params.get("augmentation_enabled", False) else "disabled",
            "sampling_strategy": training_params.get("sampling_strategy", "all"),
            "initial_weights": training_params.get("initial_weights", "pretrained")
        }
        
        # データセット情報
        dataset_params = {
            "total_annotations": dataset_info.get("total_annotations", 0),
            "used_samples": dataset_info.get("used_samples", 0),
            "train_samples": dataset_info.get("train_samples", 0),
            "val_samples": dataset_info.get("val_samples", 0),
            "image_height": dataset_info.get("input_shape", [0, 0])[0],
            "image_width": dataset_info.get("input_shape", [0, 0])[1]
        }
        
        # メトリクス
        run_metrics = {
            "best_val_loss": metrics.get("best_val_loss", 0.0),
            "final_train_loss": metrics.get("final_train_loss", 0.0),
            "final_val_loss": metrics.get("final_val_loss", 0.0)
        }
        
        # 実行時タグ（検索用）
        tags = {
            "model_type": model_type,
            "status": "early_stopped" if training_params.get("early_stopped", False) else "completed"
        }
        
        # MLflowのRun開始
        run_name = f"{model_type}_{dataset_info.get('used_samples', 0)}samples_{training_params.get('completed_epochs', 0)}epochs"
        
        try:
            with mlflow.start_run(run_name=run_name):
                # タグを設定
                mlflow.set_tags(tags)
                
                # パラメータをログ
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # データセット情報をログ
                for key, value in dataset_params.items():
                    mlflow.log_param(f"dataset_{key}", value)
                
                # メトリクスをログ
                for key, value in run_metrics.items():
                    mlflow.log_metric(key, value)
                
                # 学習曲線をログ（損失の推移）
                if "train_losses" in metrics and "val_losses" in metrics:
                    train_losses = metrics["train_losses"]
                    val_losses = metrics["val_losses"]
                    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                        mlflow.log_metric("train_loss", train_loss, step=epoch)
                        mlflow.log_metric("val_loss", val_loss, step=epoch)
                
                try:
                    # モデルファイルをアーティファクトとして保存
                    # Windows環境ではパスの形式に注意
                    if sys.platform.startswith('win'):
                        # Windows環境ではパスの区切り文字をチェック
                        model_path = os.path.normpath(model_path)
                    
                    mlflow.log_artifact(model_path, "model")
                    
                    # PyTorchモデル自体も保存（再利用のため）
                    try:
                        model = get_model(model_type, pretrained=False)
                        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                        
                        mlflow.pytorch.log_model(model, "pytorch_model")
                    except Exception as e:
                        print(f"PyTorchモデル保存中にエラーが発生: {e}")
                    
                except Exception as e:
                    print(f"モデルアーティファクト保存中にエラーが発生: {e}")
            
            return True
        except Exception as e:
            print(f"MLflow記録エラー: {e}")
            return False
        
    def compare_models_mlflow(self):
        """MLflowで記録されたモデルを比較する"""
        if not hasattr(self, 'mlflow_tracking_uri'):
            if not self.initialize_mlflow():
                return
        
        # MLflow UIを開いて比較してもらう
        self.open_mlflow_ui()

    def closeEvent(self, event):
        """アプリケーション終了時にセッション情報を保存する"""
        # セッション情報を保存
        self.save_session_info()
        
        # 親クラスのcloseEventを呼び出して通常の終了処理を行う
        super().closeEvent(event)

    # location関連

    def update_location_button_counts(self):
        """各位置ボタンのアノテーション数を更新する"""
        if not hasattr(self, 'location_buttons') or not self.location_buttons:
            return

        # 位置情報ごとのアノテーション数をカウント
        location_counts = {}
        for img_path, anno in self.annotations.items():
            if 'loc' in anno:
                loc_value = anno['loc']
                location_counts[loc_value] = location_counts.get(loc_value, 0) + 1

        # 各ボタンのカウント表示を更新
        for button in self.location_buttons:
            loc_value = button.property("location_value")
            count = location_counts.get(loc_value, 0)
            
            # ボタンのテキストを更新（数を追加）
            button.setText(f"{count} | 位置 {loc_value}")
            
            # カウントに応じてスタイルを調整
            color = get_location_color(loc_value)
            if count > 0:
                # アノテーションがある場合は少し濃い色にする
                button.setStyleSheet(f"""
                    QPushButton {{
                        padding: 8px;
                        border: 1px solid {color.name()};
                        border-radius: 4px;
                        background-color: {color.lighter(140).name()};
                        color: black;
                    }}
                    QPushButton:checked {{
                        background-color: {color.name()};
                        color: white;
                        font-weight: bold;
                    }}
                """)
            else:
                # アノテーションがない場合はグレーっぽくする
                button.setStyleSheet(f"""
                    QPushButton {{
                        padding: 8px;
                        border: 1px solid #cccccc;
                        border-radius: 4px;
                        background-color: #f0f0f0;
                        color: #888888;
                    }}
                    QPushButton:checked {{
                        background-color: {color.name()};
                        color: white;
                        font-weight: bold;
                    }}
                """)

    def add_location_button(self):
        """位置情報選択ボタンを追加する"""
        location_value = self.new_location_input.value()
        
        # 同じ値のボタンが既にある場合は追加しない
        for button in self.location_buttons:
            if button.property("location_value") == location_value:
                QMessageBox.warning(self, "警告", f"位置情報 {location_value} は既に存在します。")
                return
        
        # 新しいボタンを作成
        button = QPushButton(f"位置 {location_value}")
        button.setProperty("location_value", location_value)
        button.setCheckable(True)  # チェック可能に設定
        button.clicked.connect(lambda checked, value=location_value: self.set_location(value))
        
        # スタイルシートを設定
        button.setStyleSheet("""
            QPushButton {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)
        
        # レイアウトに追加
        self.location_buttons_layout.addWidget(button)
        self.location_buttons.append(button)
        
        # 次の値にインクリメント
        self.new_location_input.setValue(location_value + 1)
        
        # 初期ボタンを生成するだけの場合はメッセージを表示しない
        if len(self.location_buttons) > 1:
            QMessageBox.information(self, "追加完了", f"位置情報 {location_value} を追加しました。")

    def set_location(self, location_value):
        print("set location")
        """位置情報を設定する - ユーザーが明示的に位置ボタンをクリックした時のみ呼ばれる"""
        if not self.images:
            return
        
        # 現在の画像パスを取得
        current_img_path = self.images[self.current_index]
        
        # 削除済みの場合は処理しない
        if hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes:
            QMessageBox.warning(
                self, 
                "警告", 
                "削除済みの画像には位置情報を設定できません。\n"
                "先に「削除状態を復元」を実行してください。"
            )
            
            # ボタンの選択状態をリセット（すべて非選択）
            for button in self.location_buttons:
                button.setChecked(False)
                
            return
        
        # デバッグ情報
        print(f"位置情報を設定: {location_value} for image {os.path.basename(current_img_path)}")
        
        # 選択済みのボタンを選択
        if self.current_location == location_value:
            print("ボタン解除")
            self.location_buttons[location_value].setChecked(False)
            self.current_location = None

            # 位置情報ラベルを更新
            self.current_location_label.setText("現在の位置情報: なし")
            self.current_location_label.setStyleSheet("")
            
            # アノテーションから位置情報を削除（すでにアノテーションがある場合）
            if self.current_index in self.annotations:
                if "loc" in self.annotations[self.current_index]:
                    del self.annotations[self.current_index]["loc"]
                
                # 位置情報アノテーションからも削除
                if self.current_index in self.location_annotations:
                    del self.location_annotations[self.current_index]

        else:
            # 現在の位置情報を更新
            self.current_location = location_value
            
            # すべてのボタンの選択状態を更新
            for button in self.location_buttons:
                button.setChecked(button.property("location_value") == location_value)
            
            # 位置情報アノテーションを更新
            self.location_annotations[self.current_index] = location_value
            
            # 現在の位置情報ラベルを更新
            loc_color = get_location_color(location_value)
            self.current_location_label.setText(f"現在の位置情報: {location_value}")
            self.current_location_label.setStyleSheet(f"color: {loc_color.name()}; font-weight: bold;")
            
            # アノテーションがある場合はアノテーションにも位置情報を追加
            if self.current_index in self.annotations:
                self.annotations[self.current_index]["loc"] = location_value
        
        # 保存用のデータ形式を更新するため、アノテーションタイムスタンプも更新
        self.annotation_timestamps[current_img_path] = int(time.time() * 1000)
        
        # 位置ボタンのカウント表示を更新
        self.update_location_button_counts()
        
        # UI更新
        self.display_current_image()
        self.update_gallery()

    def on_method_changed(self, index):
        """学習方法が変更されたときの処理"""
        # モデル選択部分の表示/非表示は現在は常に表示するようにする
        # 保存済みのモデルリストを更新
        self.refresh_model_list()

    def refresh_model_list(self):
        """保存されているモデルのリストを更新 - モデルアーキによるフィルタリング機能付き"""
        self.model_combo.clear()
        
        # 更新開始のダイアログを表示
        self.statusBar().showMessage("モデルリストを更新中...")
        QApplication.processEvents()
                
        # 現在選択しているモデルアーキ
        current_arch = self.auto_method_combo.currentText()
        
        # モデルファイルを検索
        all_model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
        # モデルアーキでフィルタリング
        model_files = []
        for model_file in all_model_files:
            # モデルファイル名にアーキ名が含まれていれば対象とする
            # 通常、モデルファイル名は「アーキ名_日時.pth」等の形式
            if current_arch.lower() in model_file.lower():
                model_files.append(model_file)
        
        if not model_files:
            # フィルタリングした結果がなければ、その旨を表示
            self.model_combo.addItem(f"{current_arch}のモデルが見つかりません")
            self.statusBar().showMessage(f"{current_arch}のモデルが見つかりません。他のアーキを選択するか、モデルを学習してください", 3000)
            return
        
        # モデルファイルを日付順にソート（新しいものが上）
        model_files.sort(reverse=True)
        
        # コンボボックスに追加
        for model_file in model_files:
            self.model_combo.addItem(model_file)
        
        # 更新完了メッセージ
        self.statusBar().showMessage(f"{len(model_files)}個の{current_arch}モデルを読み込みました", 3000)

    def play_forward(self):
        """自動再生（順方向）"""
        # 再生中かどうかをチェック
        is_playing = hasattr(self, 'auto_play_timer') and self.auto_play_timer.isActive()
        
        # 再生または停止
        self.auto_play(forward=True)
        
        # 再生状態に応じてボタンテキストを更新
        sender = self.sender()
        if sender:
            if is_playing:
                # 停止した場合
                sender.setText("⏵")
                self.statusBar().clearMessage()
            else:
                # 再生開始した場合
                sender.setText("⏹")

    def auto_play(self, forward=True):
        """画像を自動再生する（スキップ枚数対応、推論表示時は速度調整）"""
        if not self.images:
            return
        
        # 現在の自動再生状態を確認
        if hasattr(self, 'auto_play_timer') and self.auto_play_timer.isActive():
            # タイマーが動いている場合は停止
            self.auto_play_timer.stop()
            return
        
        # スキップ枚数を取得
        skip_count = self.skip_count_spin.value()
        
        # 再生方向に基づいて、次の画像へ進むためのステップを決定
        step = skip_count if forward else -skip_count
        
        # 再生タイマーをセットアップ
        self.auto_play_timer = QTimer()
        self.auto_play_timer.timeout.connect(lambda: self.skip_images(step))
        
        # タイマー開始（再生速度を設定）
        # 推論表示がONの場合は速度を落とす
        if self.inference_checkbox.isChecked():
            interval = 100  # 推論表示ONのときは遅めに設定（150ミリ秒）
        else:
            interval = 20   # 通常は高速（20ミリ秒）
        
        self.auto_play_timer.start(interval)
        
        # 再生中であることをステータスバーに表示
        direction = "順方向" if forward else "逆方向"
        playback_speed = "低速" if self.inference_checkbox.isChecked() else "高速"
        self.statusBar().showMessage(f"自動再生中 ({direction}, {skip_count}枚スキップ, {playback_speed}) - 停止するには再度ボタンをクリック")

    def play_reverse(self):
        """自動再生（逆方向）"""
        # 再生中かどうかをチェック
        is_playing = hasattr(self, 'auto_play_timer') and self.auto_play_timer.isActive()
        
        # 再生または停止
        self.auto_play(forward=False)
        
        # 再生状態に応じてボタンテキストを更新
        sender = self.sender()
        if sender:
            if is_playing:
                # 停止した場合
                sender.setText("⏪")
                self.statusBar().clearMessage()
            else:
                # 再生開始した場合
                sender.setText("⏹")

    def on_variant_changed(self, variant):
        """
        ラジオボタンで選択された画像ソースキーが変更された時に呼ばれるメソッド
        
        Args:
            variant (str): 選択された新しいキー名
        """
        # キー情報が初期化されていない場合は早期リターン
        if not hasattr(self, 'available_variants') or not self.available_variants:
            print("キー情報がまだ初期化されていません。load_images後に設定されます。")
            self.current_variant = variant  # キー名だけは保存しておく
            return
        
        # 以前と同じキーが選択された場合は何もしない
        if hasattr(self, 'current_variant') and self.current_variant == variant:
            return
        
        # 現在のキーを更新
        self.current_variant = variant
        print(f"キーを '{variant}' に変更しました")
        
        # キーに対応する画像リストを取得・更新
        if hasattr(self, 'variant_images') and variant in self.variant_images:
            # キー別の画像リストを設定
            self.images = self.variant_images[variant]
            print(f"キー '{variant}' の画像数: {len(self.images)}")
            
            # 同じインデックスを維持（可能であれば）
            if self.current_index >= len(self.images):
                self.current_index = 0
            
            # スライダーの設定を更新
            if self.images:
                self.image_slider.setMaximum(len(self.images) - 1)
                self.image_slider.setValue(self.current_index)
                self.slider_value_label.setText(f"{self.current_index + 1}/{len(self.images)}")
            else:
                self.image_slider.setMaximum(0)
                self.image_slider.setValue(0)
                self.slider_value_label.setText("0/0")
            
            # UI更新
            self.update_stats()
            self.display_current_image()
            self.update_gallery()
        
        # キーボタン群を更新
        self.update_variant_buttons()

    def update_variant_buttons(self):
        """
        available_variantsの内容に基づいてキーボタン群を更新する
        """
        # GroupBoxを探す
        variant_box = None
        left_panel = self.centralWidget().layout().itemAt(0).widget()
        for i in range(left_panel.layout().count()):
            item = left_panel.layout().itemAt(i).widget()
            if isinstance(item, QGroupBox) and item.title() == "画像ソース":
                variant_box = item
                break
        
        if not variant_box:
            print("画像ソースのGroupBoxが見つかりませんでした")
            return
        
        # 現在のレイアウトと全てのボタンをクリア
        variant_layout = variant_box.layout()
        while variant_layout.count():
            item = variant_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # ボタングループをリセット
        if hasattr(self, 'variant_button_group'):
            for button in self.variant_button_group.buttons():
                self.variant_button_group.removeButton(button)
        else:
            self.variant_button_group = QButtonGroup(self)
            self.variant_button_group.setExclusive(True)
        
        # 新しいキーボタンを追加
        for var in self.available_variants:
            rb = QRadioButton(var)
            variant_layout.addWidget(rb)
            self.variant_button_group.addButton(rb)
            if var == self.current_variant:
                rb.setChecked(True)
            # 切替時、チェックされた変化のみ受け取る
            rb.toggled.connect(lambda checked, v=var: self.on_variant_changed(v) if checked else None)

    def update_images_for_variant(self):
        """
        選択されたキーに基づいて画像表示を更新する
        """
        # 現在のキーに応じた画像の読み込みや表示更新
        # 実際のアプリケーションに合わせて実装
        print(f"キー '{self.current_variant}' の画像を更新します")
        
        # 例：特定の処理を実行
        if self.current_variant == "cam":
            # カメラビュー用の処理
            pass
        elif self.current_variant == "cam1":
            pass
        elif self.current_variant == "lidar":
            pass
        
        # 画像表示の更新
        if hasattr(self, 'current_index') and self.images:
            self.display_current_image()
            self.update_gallery()

    def toggle_auto_apply_location(self, state):
        """位置情報の自動適用を有効/無効にする"""
        self.auto_apply_location = (state == Qt.Checked)
        
        # チェックがオンになり、現在位置情報がない場合は即座に適用
        if self.auto_apply_location and self.current_location is not None and self.images:
            # 削除済みの場合は適用しない
            if hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes:
                return
                
            # 現在の画像に既に位置情報がある場合は上書きしない
            if self.current_index in self.location_annotations:
                return
                
            # 現在選択されている位置情報を適用
            self.set_location(self.current_location)
            
            # ステータスバーに表示
            self.statusBar().showMessage(f"位置情報 {self.current_location} を自動適用しました", 3000)

    def toggle_auto_apply_bbox(self, state):
        """前回のバウンディングボックスを自動適用するかどうかを設定"""
        self.auto_apply_last_bbox = (state == Qt.Checked)
        
        # 現在の画像に対して、前回のバウンディングボックスを適用
        if self.auto_apply_last_bbox and self.last_bbox is not None and self.images:
            current_img_path = self.images[self.current_index]
            
            # 削除済みの場合は適用しない
            if hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes:
                return
            
            # すでにアノテーションがある場合は確認
            if current_img_path in self.bbox_annotations and self.bbox_annotations[current_img_path]:
                # すでにバウンディングボックスがある場合は適用しない
                return
            
            # 前回のバウンディングボックスを適用
            self.add_bbox_annotation(self.last_bbox.copy())
            
            # ステータスバーに表示
            self.statusBar().showMessage(f"前回の '{self.last_bbox['class']}' バウンディングボックスを適用しました", 3000)

    def toggle_detection_inference_display(self, state):
        """物体検知推論表示の切り替え"""
        show_inference = (state == Qt.Checked)
        self.show_detection_inference = show_inference
        
        # 画面更新
        self.main_image_view.update()
        
        # 表示状態をステータスバーに反映
        if show_inference:
            self.statusBar().showMessage("物体検知推論結果表示をオンにしました", 3000)
        else:
            self.statusBar().showMessage("物体検知推論結果表示をオフにしました", 3000)
        
        # 画像表示を更新
        self.display_current_image()

    def update_driving_info_panel(self):
        """自動運転推論結果の情報パネルを更新する"""
        if not self.images:
            return False
            
        current_img_path = self.images[self.current_index]
        is_deleted = hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes
        
        if hasattr(self, 'inference_checkbox') and self.inference_checkbox.isChecked() and not is_deleted:
            if current_img_path in self.inference_results:
                # 推論結果を取得
                inference = self.inference_results[current_img_path]
                
                # 新しいキー形式があればそれを使い、なければ古い形式を使う
                if "pilot/angle" in inference and "pilot/throttle" in inference:
                    angle = inference["pilot/angle"]
                    throttle = inference["pilot/throttle"]
                else:
                    angle = inference["angle"]
                    throttle = inference["throttle"]

                # 推論情報のリッチテキスト
                inference_text = f"<b>自動運転推論結果:</b><br>"
                inference_text += f"angle = <span style='color: #6666FF;'>{angle:.4f}</span><br>"
                inference_text += f"throttle = <span style='color: #6666FF;'>{throttle:.4f}</span>"

                # 位置情報を取得
                location = None
                if "pilot/loc" in inference:
                    location = inference["pilot/loc"]
                elif "loc" in inference:
                    location = inference["loc"]

                # 位置情報があれば色付きバッジとして表示
                if location is not None:
                    loc_color = get_location_color(location)
                    
                    inference_text += f"<br><div style='margin-top: 10px;'>"
                    inference_text += f"<div style='display: inline-block; background-color: {loc_color.name()}; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'>"
                    inference_text += f"推論位置 {location}</div></div>"

                # リッチテキストとして設定
                if hasattr(self, 'inference_info_label'):
                    self.inference_info_label.setText(inference_text)
                    self.inference_info_label.setTextFormat(Qt.RichText)
                    self.inference_info_label.repaint()
                    QApplication.processEvents()  # UIを即時更新

                # ImageLabelに推論ポイントを設定
                self.main_image_view.inference_point = QPoint(inference['x'], inference['y'])
                
                return True
                
            elif hasattr(self, 'run_inference_check'):
                # 推論結果がない場合は実行
                self.run_inference_check(False)
                
                # 推論実行後に再度チェック
                if current_img_path in self.inference_results:
                    # 再帰的に呼び出して情報パネルを更新
                    return self.update_driving_info_panel()
                
                return False
        else:
            # 表示がオフの場合は情報パネルをクリア
            if hasattr(self, 'inference_info_label'):
                self.inference_info_label.setText("")
            
            self.main_image_view.inference_point = None
            
            return False

    def update_location_info_panel(self):
        """位置推論結果の情報パネルを更新する"""
        if not self.images:
            return False
                
        current_img_path = self.images[self.current_index]
        is_deleted = hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes
        
        if hasattr(self, 'location_inference_checkbox') and self.location_inference_checkbox.isChecked() and not is_deleted:
            if current_img_path in self.location_inference_results:
                # 推論結果を取得
                inference = self.location_inference_results[current_img_path]
                
                # 位置推論情報のリッチテキスト
                inference_text = "<b>位置推論結果:</b><br>"
                
                # 位置情報を取得
                location = inference.get("loc", None)
                
                # 位置情報があれば色付きバッジとして表示
                if location is not None:
                    loc_color = get_location_color(location)
                    
                    inference_text += f"<br><div style='margin-top: 10px;'>"
                    inference_text += f"<div style='display: inline-block; background-color: {loc_color.name()}; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'>"
                    inference_text += f"推論位置 {location}</div></div>"

                # リッチテキストとして設定
                if hasattr(self, 'location_inference_info_label'):
                    self.location_inference_info_label.setText(inference_text)
                    self.location_inference_info_label.setTextFormat(Qt.RichText)
                    self.location_inference_info_label.repaint()
                    QApplication.processEvents()  # UIを即時更新

                # ImageLabelに位置推論ポイントを設定（後で実装するdraw_location_inference関数用）
                self.main_image_view.location_inference_result = inference
                
                return True
                    
            elif hasattr(self, 'run_location_inference_check'):
                # 推論結果がない場合は実行
                self.run_location_inference_check(False)
                
                # 推論実行後に再度チェック
                if current_img_path in self.location_inference_results:
                    # 再帰的に呼び出して情報パネルを更新
                    return self.update_location_info_panel()
                
                return False
        else:
            # 表示がオフの場合は情報パネルをクリア
            if hasattr(self, 'location_inference_info_label'):
                self.location_inference_info_label.setText("")
            
            # 位置推論ポイントをクリア
            if hasattr(self, 'main_image_view'):
                self.main_image_view.location_inference_result = None
            
            return False

    def toggle_training_mode(self):
        """学習モードを切り替える"""
        # 送信元ボタンを確認（クリックされたボタン）
        sender = self.sender()
        
        if sender == self.auto_train_mode_button:
            # 自動運転モデル学習モードが選択された
            self.auto_train_mode_button.setChecked(True)
            self.obj_train_mode_button.setChecked(False)
            self.current_training_mode = 0  # 0 = 自動運転モデル学習モード
            self.statusBar().showMessage("自動運転モデル学習モードに切り替えました。", 3000)
            
            # コンテナの表示/非表示を切り替え
            self.auto_method_container.setVisible(True)
            self.object_detection_container.setVisible(False)
        elif sender == self.obj_train_mode_button:
            # 物体検知モデル学習モードが選択された
            self.auto_train_mode_button.setChecked(False)
            self.obj_train_mode_button.setChecked(True)
            self.current_training_mode = 1  # 1 = 物体検知モデル学習モード
            self.statusBar().showMessage("物体検知モデル学習モードに切り替えました。", 3000)
            
            # コンテナの表示/非表示を切り替え
            self.auto_method_container.setVisible(False)
            self.object_detection_container.setVisible(True)

    def toggle_annotation_mode(self, checked=None):
        """アノテーションモードを切り替える"""
        # 送信元ボタンを確認（クリックされたボタン）
        sender = self.sender()
        
        if sender == self.auto_mode_button:
            # 自動運転モードが選択された
            self.auto_mode_button.setChecked(True)
            self.detection_mode_button.setChecked(False)
            self.current_mode = 0  # 0 = 自動運転モード
            self.statusBar().showMessage("自動運転アノテーションモードに切り替えました。", 3000)
        elif sender == self.detection_mode_button:
            # 物体検知モードが選択された
            self.auto_mode_button.setChecked(False)
            self.detection_mode_button.setChecked(True)
            self.current_mode = 1  # 1 = 物体検知モード
            self.statusBar().showMessage("物体検知アノテーションモードに切り替えました。", 3000)
        else:
            # Bキーで呼び出された場合は現在のモードを反転
            if hasattr(self, 'current_mode') and self.current_mode == 1:
                # 物体検知から自動運転へ
                self.current_mode = 0
                self.auto_mode_button.setChecked(True)
                self.detection_mode_button.setChecked(False)
                self.statusBar().showMessage("自動運転アノテーションモードに切り替えました。", 3000)
            else:
                # 自動運転から物体検知へ
                self.current_mode = 1
                self.auto_mode_button.setChecked(False)
                self.detection_mode_button.setChecked(True)
                self.statusBar().showMessage("物体検知アノテーションモードに切り替えました。", 3000)
        
        # UI更新
        self.main_image_view.update()

    # yolo 関数
    def on_yolo_model_type_changed(self, index):
        """YOLOモデルタイプが変更されたときの処理"""
        # 現在選択されているモデルタイプを取得
        selected_model_type = self.yolo_model_combo.currentText()
        self.statusBar().showMessage(f"YOLOモデルタイプを「{selected_model_type}」に変更しました。モデルリストを更新します...")
        
        # モデルリストを更新
        self.refresh_yolo_model_list()

    def export_to_yolo(self):
        """バウンディングボックスアノテーションをYOLO形式でエクスポートする"""
        if not hasattr(self, 'bbox_annotations') or not self.bbox_annotations:
            QMessageBox.information(self, "情報", "エクスポートするバウンディングボックスアノテーションがありません。")
            return
        
        try:
            # プログレスダイアログを表示
            progress = QProgressDialog("YOLOフォーマットでエクスポート中...", "キャンセル", 0, len(self.bbox_annotations), self)
            progress.setWindowTitle("エクスポート")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # エクスポート実行
            try:
                # exports_fileモジュールから関数をインポート
                
                # 関数を呼び出し
                yaml_path = export_to_yolo(annotation_folder, self.bbox_annotations)
                
                progress.setValue(len(self.bbox_annotations))
                
                # エクスポート成功メッセージ
                yolo_folder = os.path.dirname(yaml_path)
                
                # クラス情報の取得
                all_classes = set()
                for bboxes in self.bbox_annotations.values():
                    for bbox in bboxes:
                        all_classes.add(bbox.get('class', 'unknown'))
                class_list = sorted(list(all_classes))
                
                # バウンディングボックス数のカウント
                total_bboxes = sum(len(bboxes) for bboxes in self.bbox_annotations.values())
                
                QMessageBox.information(
                    self, 
                    "エクスポート完了", 
                    f"バウンディングボックスアノテーションをYOLO形式でエクスポートしました。\n"
                    f"保存先: {yolo_folder}\n"
                    f"処理画像数: {len(self.bbox_annotations)}\n"
                    f"バウンディングボックス数: {total_bboxes}\n"
                    f"クラス: {', '.join(class_list)}"
                )
            
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "エラー", 
                    f"YOLOフォーマットでのエクスポート中にエラーが発生しました: {str(e)}"
                )
            
            finally:
                progress.close()
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "エラー", 
                f"エクスポート準備中にエラーが発生しました: {str(e)}"
            )

    def delete_selected_bbox(self):
        """選択されたバウンディングボックスを削除する"""
        if not self.images or not hasattr(self, 'main_image_view'):
            return
        
        selected_index = self.main_image_view.selected_bbox_index
        if selected_index is None:
            # 選択されていない場合は何もしない
            return
        
        current_img_path = self.images[self.current_index]
        if current_img_path in self.bbox_annotations and selected_index is not None:
            bboxes = self.bbox_annotations[current_img_path]
            if 0 <= selected_index < len(bboxes):
                # ボックス情報を取得
                bbox = bboxes[selected_index]
                class_name = bbox.get('class', 'unknown')
                
                # 確認ダイアログ
                reply = QMessageBox.question(
                    self, 
                    "バウンディングボックス削除", 
                    f"選択された '{class_name}' のバウンディングボックスを削除しますか？",
                    QMessageBox.Yes | QMessageBox.No, 
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # 削除実行
                    del bboxes[selected_index]
                    # 選択をクリア
                    self.main_image_view.selected_bbox_index = None
                    # 画面更新
                    self.main_image_view.update()
                    # 統計情報更新
                    self.update_bbox_stats()
                    
                    # 確認メッセージ
                    self.statusBar().showMessage(f"'{class_name}' のバウンディングボックスを削除しました", 3000)

    def refresh_yolo_model_list(self):
        """保存されているYOLOモデルのリストを更新 - サブフォルダとweightsフォルダ内も検索し、直下のモデルも含める - 選択したタイプでフィルタリング"""
        if not hasattr(self, 'yolo_saved_model_combo'):
            return
                    
        self.yolo_saved_model_combo.clear()
        
        # 更新開始のメッセージを表示
        self.statusBar().showMessage("YOLOモデルリストを更新中...")
        
        # 現在選択されているモデルタイプを取得
        selected_model_type = self.yolo_model_combo.currentText()
                            
        # YOLOモデルファイルを検索 - サブフォルダも含めて
        yolo_model_files = []
        
        # 1. まず直下の.ptファイルを検索 - 選択したタイプのモデルのみ
        for file in os.listdir(models_dir):
            if file.endswith('.pt') and ('yolo' in file.lower()) and (selected_model_type.lower() in file.lower()):
                # ファイルのフルパスを取得
                full_path = os.path.join(models_dir, file)
                
                # 各ファイルに関する情報をまとめる
                model_info = {
                    'path': file,  # 直下のファイルは相対パスとしてファイル名のみ
                    'parent': 'root',  # 直下のファイルは親フォルダを'root'として識別
                    'type': 'model',  # 通常のモデルファイル
                    'date': ''  # 日付情報なし
                }
                yolo_model_files.append(model_info)
        
        # 2. サブフォルダを含めて再帰的に検索（既存の処理）- 選択したタイプのモデルのみ
        for root, dirs, files in os.walk(models_dir):
            if root == models_dir:
                continue  # 直下のファイルは上で既に処理したのでスキップ
                    
            for file in files:
                if file.endswith('.pt') and ('best' in file.lower() or 'last' in file.lower()):
                    # ファイルのフルパスを取得
                    full_path = os.path.join(root, file)
                    # models_dir からの相対パスに変換
                    rel_path = os.path.relpath(full_path, models_dir)
                    
                    # サブフォルダ名またはその親フォルダ名を確認
                    folder_name = os.path.basename(root)
                    parent_folder_name = os.path.basename(os.path.dirname(root))
                    
                    # 選択したモデルタイプがフォルダ名に含まれているかチェック
                    model_match = (
                        selected_model_type.lower() in folder_name.lower() or 
                        selected_model_type.lower() in parent_folder_name.lower()
                    )
                    
                    if not model_match:
                        continue  # モデルタイプが一致しない場合はスキップ
                    
                    # 日時情報を取得するための処理
                    # フォルダ名からモデル名と日時情報を抽出（例: yolov8n_20250502_204810）
                    date_info = ""
                    parts = folder_name.split('_')
                    if len(parts) >= 3:  # モデル名_日付_時間 の形式を想定
                        date_parts = parts[1:]  # モデル名を除いた部分を日時情報として使用
                        date_info = '_'.join(date_parts)
                    else:
                        # 親フォルダから日時情報を取得
                        parent_parts = parent_folder_name.split('_')
                        if len(parent_parts) >= 3:
                            date_info = '_'.join(parent_parts[1:])
                    
                    # パス情報と一緒にモデル種類と日時情報を保持
                    model_info = {
                        'path': rel_path,
                        'parent': folder_name,
                        'type': 'best' if 'best' in file.lower() else 'last',
                        'date': date_info
                    }
                    yolo_model_files.append(model_info)
        
        if not yolo_model_files:
            self.yolo_saved_model_combo.addItem(f"{selected_model_type}のYOLOモデルが見つかりません")
            self.statusBar().showMessage(f"{selected_model_type}のYOLOモデルが見つかりません。学習を実行するか他のタイプを選択してください", 3000)
            return
        
        # モデルファイルをソート - 直下のモデルを最初に、次に日付が新しいもの順にソート
        def sort_key(model_info):
            if model_info['parent'] == 'root':
                return ('0', '')  # 直下のファイルを最初に
            else:
                return ('1', model_info['date'])  # 次に日付の新しい順
        
        yolo_model_files.sort(key=sort_key, reverse=False)  # 直下のファイルを先頭に
        
        # コンボボックスに追加
        for model_info in yolo_model_files:
            if model_info['parent'] == 'root':
                # 直下のファイルの表示名: "yolov8n.pt"
                model_name = f"{model_info['path']} "
            else:
                # サブフォルダ内のファイルの表示名: "yolov8n [20250411_183737] (best)"
                model_name = f"{model_info['parent'].split('_')[0]} [{model_info['date']}] ({model_info['type']})"
            
            # コンボボックスにアイテムを追加し、ユーザーデータとして相対パスを設定
            self.yolo_saved_model_combo.addItem(model_name, model_info['path'])
        
        # 更新完了メッセージ
        self.statusBar().showMessage(f"{len(yolo_model_files)}個の{selected_model_type}モデルを読み込みました", 3000)

    def download_pretrained_yolo_model(self, model_type):
        """事前学習済みのYOLOモデルをダウンロードしてmodelsフォルダに保存する"""
        if not model_type or not model_type.startswith("yolov8"):
            return None
        
        # モデルファイル名（例: yolov8n.pt）
        model_filename = f"{model_type}.pt"
        
        # 保存先パス
        save_path = os.path.join(models_dir, model_filename)
        
        # 既にファイルが存在する場合はそのパスを返す
        if os.path.exists(save_path):
            return save_path
        
        # 進捗ダイアログを表示
        progress = QProgressDialog(
            f"事前学習済み {model_type} モデルをダウンロード中...", 
            "キャンセル", 0, 100, self
        )
        progress.setWindowTitle("モデルダウンロード")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        progress.setValue(10)
        QApplication.processEvents()
        
        try:
            # UltralyticsのYOLOモデルをダウンロード
            progress.setLabelText(f"{model_type} モデルをダウンロード中...")
            progress.setValue(30)
            QApplication.processEvents()
            
            # YOLO.hubから直接モデルをダウンロード
            model = YOLO(model_type)
            
            progress.setLabelText(f"{model_type} モデルをmodelsフォルダに保存中...")
            progress.setValue(70)
            QApplication.processEvents()
            
            # モデルファイルをmodelsフォルダにコピー
            model_path = model.ckpt_path
            if os.path.exists(model_path):
                shutil.copy2(model_path, save_path)
            
            progress.setValue(100)
            progress.close()
            
            # モデルリストを更新
            self.refresh_yolo_model_list()
            
            self.statusBar().showMessage(f"事前学習済み {model_type} モデルをmodelsフォルダに保存しました: {save_path}", 3000)
            return save_path
        
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"事前学習済みモデルのダウンロード中にエラーが発生しました: {str(e)}"
            )
            return None

    def load_yolo_model(self):
        """選択されたYOLOモデルを読み込む - サブフォルダ対応版"""
        if not self.images:
            QMessageBox.warning(self, "警告", "画像が読み込まれていません。")
            return
        
        # モデル情報を取得 - 表示名と実際のパス
        current_index = self.yolo_saved_model_combo.currentIndex()
        selected_model_display = self.yolo_saved_model_combo.currentText()
        
        # ユーザーデータからパスを取得（相対パス）
        relative_path = self.yolo_saved_model_combo.itemData(current_index)
        
        if not relative_path or selected_model_display == "YOLOモデルが見つかりません" :
            QMessageBox.warning(self, "警告", "有効なYOLOモデルが選択されていません。")
            return
        
        # モデルのパスを取得 - 相対パスからフルパスに変換
        models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
        model_path = os.path.join(models_dir, relative_path)
        
        # モデルが存在するか確認
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "警告", f"選択されたモデルが見つかりません: {model_path}")
            return
            
        # 進捗ダイアログを表示
        progress = QProgressDialog(
            f"YOLOモデル '{selected_model_display}' を読み込み中...",
            "キャンセル", 0, 100, self
        )
        progress.setWindowTitle("モデル読み込み")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        try:            
            # 信頼度閾値の設定
            confidence, ok = QInputDialog.getDouble(
                self, 
                "検出閾値", 
                "検出信頼度閾値 (0.0-1.0):",
                0.6, 0.01, 1.0, 2
            )
            
            if not ok:
                progress.close()
                return
            
            # 進捗更新
            progress.setValue(30)
            QApplication.processEvents()
            
            # モデルをロード
            progress.setLabelText(f"モデル '{selected_model_display}' をメモリに読み込み中...")
            progress.setValue(50)
            QApplication.processEvents()
            
            # モデルを読み込み
            self.yolo_model = YOLO(model_path)
            self.yolo_confidence_threshold = confidence
            
            # モデル情報を保存
            self.yolo_model_file = model_path
            
            progress.setValue(70)
            QApplication.processEvents()
            
            # 現在の画像に対して推論を実行
            progress.setLabelText("現在の画像に対して推論実行中...")
            progress.setValue(80)
            QApplication.processEvents()
            
            self.run_single_yolo_inference()
            
            progress.setValue(90)
            QApplication.processEvents()
            
            # 推論結果表示チェックボックスを自動的にオンにする
            if hasattr(self, 'detection_inference_checkbox'):
                self.detection_inference_checkbox.setChecked(True)
            
            progress.setValue(100)
            QApplication.processEvents()
            
            # 成功メッセージ
            model_name = os.path.basename(model_path)
            QMessageBox.information(
                self,
                "モデル読み込み完了",
                f"YOLOモデル「{model_name}」を読み込みました。\n"
                f"検出閾値: {confidence}\n\n"
                f"画像送りごとに自動的に推論が実行されます。"
            )
            
            progress.close()
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "エラー",
                f"YOLOモデルの読み込み中にエラーが発生しました: {str(e)}"
            )

    def load_yolo_annotations(self):
        """YOLO形式のアノテーションを読み込む"""
        if not self.images:
            QMessageBox.warning(self, "警告", "先に画像を読み込んでください。")
            return
        
        # YOLOアノテーションフォルダを選択
        yolo_dir = QFileDialog.getExistingDirectory(
            self, "YOLOアノテーションフォルダを選択", 
            self.folder_path,
            QFileDialog.ShowDirsOnly
        )
        
        if not yolo_dir:
            return
        
        # ラベルフォルダを確認
        labels_dir = os.path.join(yolo_dir, "labels")
        if not os.path.exists(labels_dir):
            # 直接選択されたフォルダがlabelsフォルダかもしれない
            if os.path.basename(yolo_dir) == "labels":
                labels_dir = yolo_dir
            else:
                # サブフォルダの中にlabelsディレクトリがあるか確認
                possible_labels_dir = [
                    os.path.join(yolo_dir, d, "labels") 
                    for d in os.listdir(yolo_dir) 
                    if os.path.isdir(os.path.join(yolo_dir, d))
                ]
                possible_labels_dir = [d for d in possible_labels_dir if os.path.exists(d)]
                
                if possible_labels_dir:
                    labels_dir = possible_labels_dir[0]
                else:
                    QMessageBox.warning(
                        self, "警告", 
                        "選択されたフォルダ内にlabelsディレクトリが見つかりません。"
                    )
                    return
        
        # クラス情報を読み込む
        classes_path = os.path.join(os.path.dirname(labels_dir), "classes.txt")
        classes = []
        
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
        else:
            # クラス情報がない場合は選択してもらう
            text, ok = QInputDialog.getText(
                self, 
                "クラス情報", 
                "クラス名をカンマで区切って入力してください（例: car,person,sign,cone）:",
                text=self.classes_input.text() if hasattr(self, 'classes_input') else "car,person,sign,cone"
            )
            
            if ok and text:
                classes = [cls.strip() for cls in text.split(',') if cls.strip()]
            else:
                return
        
        # プログレスダイアログ
        progress = QProgressDialog("YOLOアノテーションを読み込み中...", "キャンセル", 0, len(self.images), self)
        progress.setWindowTitle("読み込み中")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # 既存のアノテーションがある場合は確認
        if hasattr(self, 'bbox_annotations') and self.bbox_annotations:
            reply = QMessageBox.question(
                self,
                "既存のアノテーション",
                "既存のバウンディングボックスアノテーションを上書きしますか？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.bbox_annotations = {}
            else:
                # 既存のアノテーションに追加
                pass
        else:
            self.bbox_annotations = {}
        
        # 各画像のアノテーションを読み込む
        loaded_count = 0
        
        try:
            for i, img_path in enumerate(self.images):
                if progress.wasCanceled():
                    break
                
                progress.setValue(i)
                
                # 画像ファイル名からラベルファイル名を生成
                img_basename = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(labels_dir, f"{img_basename}.txt")
                
                # ラベルファイルが存在する場合のみ処理
                if os.path.exists(label_path):
                    # 画像サイズを取得（正規化された座標を元に戻すため）
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    
                    # ラベルファイルを読み込む
                    bboxes = []
                    
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:  # クラスID, x_center, y_center, width, height
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # YOLO形式（中心x,y,幅,高さ）から左上と右下の座標に変換
                                x1 = x_center - (width / 2)
                                y1 = y_center - (height / 2)
                                x2 = x_center + (width / 2)
                                y2 = y_center + (height / 2)
                                
                                # クラス名を取得
                                class_name = "unknown"
                                if 0 <= class_id < len(classes):
                                    class_name = classes[class_id]
                                
                                # バウンディングボックスを追加
                                bbox = {
                                    'x1': x1,
                                    'y1': y1,
                                    'x2': x2,
                                    'y2': y2,
                                    'class': class_name
                                }
                                
                                bboxes.append(bbox)
                    
                    # アノテーションを保存
                    if bboxes:
                        self.bbox_annotations[img_path] = bboxes
                        loaded_count += 1
            
            progress.close()
            
            # 統計情報を更新
            self.update_bbox_stats()
            
            # 表示を更新
            self.display_current_image()
            self.update_gallery()
            
            # 完了メッセージ
            QMessageBox.information(
                self,
                "読み込み完了",
                f"YOLOアノテーションを読み込みました。\n処理画像数: {loaded_count}/{len(self.images)}\nクラス: {', '.join(classes)}"
            )
        
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "エラー",
                f"YOLOアノテーションの読み込み中にエラーが発生しました: {str(e)}"
            )

    def run_single_yolo_inference(self):
        """現在表示中の画像に対してYOLO推論を実行"""
        if not self.images or not hasattr(self, 'yolo_model'):
            return
        
        current_img_path = self.images[self.current_index]
        
        try:
            # 推論実行
            results = self.yolo_model(current_img_path, conf=self.yolo_confidence_threshold)
            
            # 推論結果をクリア（現在の画像のみ）
            if current_img_path in self.detection_inference_results:
                del self.detection_inference_results[current_img_path]
            
            # 検出結果を保存
            bboxes = []
            
            # 画像サイズを取得
            img = Image.open(current_img_path)
            img_width, img_height = img.size
            
            for result in results:
                for det in result.boxes.data.cpu().numpy():
                    if len(det) >= 6:  # x1, y1, x2, y2, confidence, class_id
                        x1, y1, x2, y2, conf, class_id = det[:6]
                        
                        # 画像サイズで正規化（0-1の範囲に）
                        x1_norm = x1 / img_width
                        y1_norm = y1 / img_height
                        x2_norm = x2 / img_width
                        y2_norm = y2 / img_height
                        
                        # クラス名を取得
                        class_id = int(class_id)
                        class_name = result.names[class_id] if class_id in result.names else f"class_{class_id}"
                        
                        # バウンディングボックスを追加
                        bbox = {
                            'x1': x1_norm,
                            'y1': y1_norm,
                            'x2': x2_norm,
                            'y2': y2_norm,
                            'class': class_name,
                            'confidence': float(conf)
                        }
                        
                        bboxes.append(bbox)
            
            # 推論結果を保存
            if bboxes:
                self.detection_inference_results[current_img_path] = bboxes
            
            # 表示を更新
            self.main_image_view.update()
            
            # 情報パネル更新
            if hasattr(self, 'update_detection_inference_display'):
                self.update_detection_inference_display()
            
            return True
        
        except Exception as e:
            print(f"単一画像YOLO推論エラー: {e}")
            return False

    # 5. 情報パネルに物体検知推論結果を表示する処理の追加
    def update_detection_inference_display(self):
        """物体検知推論結果の表示を更新"""
        if not self.images:
            return
        
        current_img_path = self.images[self.current_index]
        is_deleted = hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes
        
        # 削除済みか推論表示OFFの場合は何も表示しない
        if is_deleted or not self.show_detection_inference:
            return
        
        # 物体検知推論結果がある場合は表示を更新
        if current_img_path in self.detection_inference_results:
            inference_bboxes = self.detection_inference_results[current_img_path]
            
            # クラスごとのカウント辞書
            class_counts = {}
            for bbox in inference_bboxes:
                class_name = bbox.get('class', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # 既存の推論情報ラベルに追加（または新規作成）
            inference_text = "<b>物体検知推論結果:</b><br>"
            inference_text += "検出オブジェクト:<br>"
            
            for class_name, count in class_counts.items():
                # クラスに応じた色を設定
                class_colors = {
                    'car': "#FF0000",     # 赤
                    'person': "#00FF00",  # 緑
                    'sign': "#0000FF",    # 青
                    'cone': "#FFFF00",    # 黄
                    'unknown': "#808080"  # グレー
                }
                color = class_colors.get(class_name, "#FF0000")
                
                inference_text += f"<span style='color: {color}; font-weight: bold;'>● {class_name}</span>: {count}個<br>"
            
            inference_text += f"合計: {len(inference_bboxes)}個のオブジェクト<br>"
            
            # 既存の推論情報ラベルがあればそれを更新
            if hasattr(self, 'detection_inference_info_label'):
                self.detection_inference_info_label.setText(inference_text)
                self.detection_inference_info_label.setTextFormat(Qt.RichText)
            else:
                # 既存の推論情報ラベルが見つからない場合は新規作成
                # アノテーション情報ラベルと同じ場所に表示するか、
                # 別の場所に配置して表示することができる
                self.detection_inference_info_label = QLabel(inference_text)
                self.detection_inference_info_label.setTextFormat(Qt.RichText)
                
                # レイアウトに追加（例: 推論情報ラベルの下に配置）
                if hasattr(self, 'inference_info_label') and self.inference_info_label.parent():
                    parent_layout = self.inference_info_label.parent().layout()
                    if parent_layout:
                        parent_layout.addWidget(self.detection_inference_info_label)
        else:
            # 推論結果がない場合は表示をクリア
            if hasattr(self, 'detection_inference_info_label'):
                self.detection_inference_info_label.setText("")

    def train_and_save_yolo_model(self):
        """Ultralytics YOLOモデルを学習し保存する"""        
        if not self.bbox_annotations:
            QMessageBox.warning(self, "警告", "物体検知アノテーションがありません。")
            return
        
        # 選択したモデルとクラス設定を取得
        model_type = self.yolo_model_combo.currentText()
        classes = [cls.strip() for cls in self.classes_input.text().split(',') if cls.strip()]
        
        if not classes:
            QMessageBox.warning(self, "警告", "検知クラスを最低1つ設定してください。")
            return
        
        # 学習設定ダイアログを表示
        training_settings = QDialog(self)
        training_settings.setWindowTitle("YOLOモデル学習設定")
        training_settings.setMinimumWidth(500)
        training_settings.setMinimumHeight(600)
        
        settings_layout = QVBoxLayout(training_settings)
        
        # タブウィジェットを作成
        tabs = QTabWidget()
        
        # 基本設定タブ
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # モデル初期化設定
        init_group = QGroupBox("モデル初期化設定")
        init_layout = QVBoxLayout(init_group)
        
        # 初期重みの選択
        weights_radio_pretrained = QRadioButton("事前学習済みの重みを使用 (推奨)")
        weights_radio_pretrained.setChecked(True)  # デフォルト選択
        init_layout.addWidget(weights_radio_pretrained)
        
        # 現在のモデルを選択
        weights_radio_current = QRadioButton("現在読み込まれているモデルの重みを使用")
        init_layout.addWidget(weights_radio_current)
        
        # 現在読み込まれているモデルの情報を表示
        current_model_info = QLabel("現在のモデル: なし")
        if hasattr(self, 'yolo_model') and hasattr(self, 'yolo_model_file'):
            model_name = os.path.basename(self.yolo_model_file) if hasattr(self, 'yolo_model_file') else "Unknown"
            current_model_info.setText(f"現在のモデル: {model_name}")
            weights_radio_current.setEnabled(True)
        else:
            weights_radio_current.setEnabled(False)
            current_model_info.setText("現在のモデル: なし（先にモデルを読み込んでください）")
        
        init_layout.addWidget(current_model_info)
        basic_layout.addWidget(init_group)
        
        # エポック数設定
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("学習エポック数:"))
        epoch_spin = QSpinBox()
        epoch_spin.setRange(1, 1000)
        epoch_spin.setValue(30)  # デフォルト: 30エポック
        epoch_layout.addWidget(epoch_spin)
        basic_layout.addLayout(epoch_layout)
        
        # バッチサイズ設定
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("バッチサイズ:"))
        batch_spin = QSpinBox()
        batch_spin.setRange(1, 128)
        batch_spin.setValue(16)  # デフォルト: 16
        batch_layout.addWidget(batch_spin)
        basic_layout.addLayout(batch_layout)
        
        # 入力サイズ設定
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("入力画像サイズ:"))
        size_combo = QComboBox()
        size_options = [str(self.original_image_size),"320", "416", "512", "640", "768", "896", "1024"]
        default_index = 4  # デフォルトは640

        # 説明ラベルを追加
        size_layout.addWidget(QLabel(f"元画像: {self.original_image_width}×{self.original_image_height}"))

        size_combo.addItems(size_options)
        size_combo.setCurrentIndex(default_index)
        size_layout.addWidget(size_combo)
        basic_layout.addLayout(size_layout)

        # 注意書き
        size_note = QLabel("注: 640以外のサイズを選択すると精度や速度に影響します")
        size_note.setStyleSheet("color: #888; font-style: italic;")
        basic_layout.addWidget(size_note)

        # Early Stopping設定
        early_stopping_check = QCheckBox("Early Stopping を有効にする")
        early_stopping_check.setChecked(True)
        basic_layout.addWidget(early_stopping_check)
        
        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("忍耐エポック数:"))
        patience_spin = QSpinBox()
        patience_spin.setRange(1, 20)
        patience_spin.setValue(10)
        patience_spin.setEnabled(True)
        patience_layout.addWidget(patience_spin)
        basic_layout.addLayout(patience_layout)
        
        # 学習率設定
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学習率:"))
        
        lr_combo = QComboBox()
        learning_rates = ["0.01", "0.005", "0.001", "0.0005", "0.0001"]
        lr_combo.addItems(learning_rates)
        lr_combo.setCurrentIndex(2)  # デフォルト: 0.001
        lr_layout.addWidget(lr_combo)
        basic_layout.addLayout(lr_layout)
        
        # タブに追加
        tabs.addTab(basic_tab, "基本設定")
        
        # データオーグメンテーションタブ
        aug_tab = QWidget()
        aug_layout = QVBoxLayout(aug_tab)
        
        # データオーグメンテーション有効化チェックボックス
        aug_enable_check = QCheckBox("データオーグメンテーションを有効にする")
        aug_enable_check.setChecked(True)
        aug_layout.addWidget(aug_enable_check)
        
        # オーグメンテーション設定のスクロールエリア
        aug_scroll = QScrollArea()
        aug_scroll.setWidgetResizable(True)
        aug_scroll.setFrameShape(QFrame.NoFrame)
        
        aug_scroll_content = QWidget()
        aug_options_layout = QVBoxLayout(aug_scroll_content)
        
        # モザイク
        mosaic_layout = QHBoxLayout()
        aug_mosaic_checkbox = QCheckBox("モザイク")
        aug_mosaic_checkbox.setChecked(True)
        aug_mosaic_proba_label = QLabel("確率:")
        aug_mosaic_proba = QDoubleSpinBox()
        aug_mosaic_proba.setRange(0.0, 1.0)
        aug_mosaic_proba.setSingleStep(0.1)
        aug_mosaic_proba.setValue(1.0)
        mosaic_layout.addWidget(aug_mosaic_checkbox)
        mosaic_layout.addWidget(aug_mosaic_proba_label)
        mosaic_layout.addWidget(aug_mosaic_proba)
        mosaic_layout.addStretch()
        aug_options_layout.addLayout(mosaic_layout)
        
        # 水平反転
        flip_layout = QHBoxLayout()
        aug_flip_checkbox = QCheckBox("水平反転")
        aug_flip_checkbox.setChecked(True)
        aug_flip_proba_label = QLabel("確率:")
        aug_flip_proba = QDoubleSpinBox()
        aug_flip_proba.setRange(0.0, 1.0)
        aug_flip_proba.setSingleStep(0.1)
        aug_flip_proba.setValue(0.5)
        flip_layout.addWidget(aug_flip_checkbox)
        flip_layout.addWidget(aug_flip_proba_label)
        flip_layout.addWidget(aug_flip_proba)
        flip_layout.addStretch()
        aug_options_layout.addLayout(flip_layout)
        
        # HSV調整
        hsv_layout = QHBoxLayout()
        aug_hsv_checkbox = QCheckBox("HSV調整")
        aug_hsv_checkbox.setChecked(True)
        hsv_layout.addWidget(aug_hsv_checkbox)
        hsv_layout.addStretch()
        aug_options_layout.addLayout(hsv_layout)
        
        # HSVの詳細設定
        hsv_details_layout = QGridLayout()
        hsv_details_layout.setContentsMargins(20, 0, 0, 0)
        
        hsv_details_layout.addWidget(QLabel("色相 (H):"), 0, 0)
        aug_hsv_h = QDoubleSpinBox()
        aug_hsv_h.setRange(0.0, 0.1)
        aug_hsv_h.setSingleStep(0.005)
        aug_hsv_h.setValue(0.015)
        hsv_details_layout.addWidget(aug_hsv_h, 0, 1)
        
        hsv_details_layout.addWidget(QLabel("彩度 (S):"), 1, 0)
        aug_hsv_s = QDoubleSpinBox()
        aug_hsv_s.setRange(0.0, 1.0)
        aug_hsv_s.setSingleStep(0.1)
        aug_hsv_s.setValue(0.7)
        hsv_details_layout.addWidget(aug_hsv_s, 1, 1)
        
        hsv_details_layout.addWidget(QLabel("明度 (V):"), 2, 0)
        aug_hsv_v = QDoubleSpinBox()
        aug_hsv_v.setRange(0.0, 1.0)
        aug_hsv_v.setSingleStep(0.1)
        aug_hsv_v.setValue(0.4)
        hsv_details_layout.addWidget(aug_hsv_v, 2, 1)
        
        aug_options_layout.addLayout(hsv_details_layout)
        
        # 幾何変換
        geometry_layout = QHBoxLayout()
        aug_geometry_checkbox = QCheckBox("幾何変換")
        aug_geometry_checkbox.setChecked(True)
        geometry_layout.addWidget(aug_geometry_checkbox)
        geometry_layout.addStretch()
        aug_options_layout.addLayout(geometry_layout)
        
        # 幾何変換の詳細設定
        geometry_details_layout = QGridLayout()
        geometry_details_layout.setContentsMargins(20, 0, 0, 0)
        
        geometry_details_layout.addWidget(QLabel("平行移動:"), 0, 0)
        aug_translate = QDoubleSpinBox()
        aug_translate.setRange(0.0, 0.5)
        aug_translate.setSingleStep(0.05)
        aug_translate.setValue(0.1)
        geometry_details_layout.addWidget(aug_translate, 0, 1)
        
        geometry_details_layout.addWidget(QLabel("スケール:"), 1, 0)
        aug_scale = QDoubleSpinBox()
        aug_scale.setRange(0.0, 1.0)
        aug_scale.setSingleStep(0.05)
        aug_scale.setValue(0.5)
        geometry_details_layout.addWidget(aug_scale, 1, 1)
        
        aug_options_layout.addLayout(geometry_details_layout)
        
        # RandomErase
        erase_layout = QHBoxLayout()
        aug_erase_checkbox = QCheckBox("ランダムイレース")
        aug_erase_checkbox.setChecked(True)
        aug_erase_proba_label = QLabel("確率:")
        aug_erase_proba = QDoubleSpinBox()
        aug_erase_proba.setRange(0.0, 1.0)
        aug_erase_proba.setSingleStep(0.1)
        aug_erase_proba.setValue(0.4)
        erase_layout.addWidget(aug_erase_checkbox)
        erase_layout.addWidget(aug_erase_proba_label)
        erase_layout.addWidget(aug_erase_proba)
        erase_layout.addStretch()
        aug_options_layout.addLayout(erase_layout)
        
        # オプションの有効/無効を連動させる
        def toggle_aug_options(checked):
            for w in aug_scroll_content.findChildren(QWidget):
                if w != aug_enable_check:
                    w.setEnabled(checked)
        
        aug_enable_check.toggled.connect(toggle_aug_options)
        
        # スクロールエリアに設定
        aug_scroll.setWidget(aug_scroll_content)
        aug_layout.addWidget(aug_scroll)
        
        # タブに追加
        tabs.addTab(aug_tab, "データオーグメンテーション")
        
        # タブをレイアウトに追加
        settings_layout.addWidget(tabs)
        
        # ボタンの配置
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(training_settings.accept)
        button_box.rejected.connect(training_settings.reject)
        settings_layout.addWidget(button_box)
        
        # ダイアログを表示
        if not training_settings.exec_():
            return
        
        # 設定値の取得
        use_pretrained = weights_radio_pretrained.isChecked()
        num_epochs = epoch_spin.value()
        batch_size = batch_spin.value()
        img_size = int(size_combo.currentText())
        use_early_stopping = early_stopping_check.isChecked()
        patience = patience_spin.value() if use_early_stopping else 0
        learning_rate = float(lr_combo.currentText())
        
        # オーグメンテーション設定の取得
        augmentation_enabled = aug_enable_check.isChecked()
        mosaic = aug_mosaic_proba.value() if aug_mosaic_checkbox.isChecked() and augmentation_enabled else 0.0
        fliplr = aug_flip_proba.value() if aug_flip_checkbox.isChecked() and augmentation_enabled else 0.0
        hsv_h = aug_hsv_h.value() if aug_hsv_checkbox.isChecked() and augmentation_enabled else 0.0
        hsv_s = aug_hsv_s.value() if aug_hsv_checkbox.isChecked() and augmentation_enabled else 0.0
        hsv_v = aug_hsv_v.value() if aug_hsv_checkbox.isChecked() and augmentation_enabled else 0.0
        translate = aug_translate.value() if aug_geometry_checkbox.isChecked() and augmentation_enabled else 0.0
        scale = aug_scale.value() if aug_geometry_checkbox.isChecked() and augmentation_enabled else 0.0
        erasing = aug_erase_proba.value() if aug_erase_checkbox.isChecked() and augmentation_enabled else 0.0
        
        # YOLOフォーマット用のデータを生成（YOLO用ディレクトリ構造を作成）
        try:            
            # データディレクトリ構造の作成
            train_dir = os.path.join(yolo_dataset_dir, "train")
            val_dir = os.path.join(yolo_dataset_dir, "val")
            os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
            os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
            
            # クラス名ファイルの保存
            with open(os.path.join(yolo_dataset_dir, "classes.txt"), 'w') as f:
                for cls in classes:
                    f.write(f"{cls}\n")
            
            # データセット設定YAMLファイルの作成
            yaml_content = f"""
    path: {yolo_dataset_dir}
    train: train/images
    val: val/images
    test: test/images

    nc: {len(classes)}
    names: {classes}
            """
            
            yaml_file = os.path.join(yolo_dataset_dir, "dataset.yaml")
            with open(yaml_file, 'w') as f:
                f.write(yaml_content)
            
            # アノテーションデータのエクスポート
            self.export_annotations_to_yolo(train_dir, val_dir, classes)
            
            # Ultralytics YOLOモデルとMLflowのインポート
            try:
                settings.update({"mlflow": True})
            except ImportError as e:
                missing_package = "ultralytics" if "ultralytics" in str(e) else "mlflow" if "mlflow" in str(e) else "依存パッケージ"
                QMessageBox.critical(self, "エラー", f"{missing_package}パッケージがインストールされていません。\npip install {missing_package} でインストールしてください。")
                return
            
            # デバイスの選択
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device for YOLO training: {device}")
            
            # 学習用の進捗ダイアログ
            progress = QProgressDialog("YOLOモデルの学習中...", "キャンセル", 0, 100, self)
            progress.setWindowTitle("YOLOモデル学習")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
                                    
            # Windows環境での正しいURI形式を構築
            if sys.platform.startswith('win'):
                tracking_uri = f"file:///{normalized_path}"
            else:
                tracking_uri = f"file://{normalized_path}"
            
            print(f"YOLOトレーニング用MLflowトラッキングURI: {tracking_uri}")
                                    
            # 重要: 実験名を固定の文字列に設定
            experiment_name = "yolo_training"
            
            # 実験が存在するか確認し、なければ作成
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            
            # YOLOの設定は環境変数から読み込みになる
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
                        
            # トレーニング設定のカスタマイズ
            run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            progress.setLabelText("YOLOモデルを初期化中...")
            progress.setValue(10)
            QApplication.processEvents()
                        
            # 分離プロセスで学習を実行
            try:
                # モデルの読み込み - 選択に基づいて初期重みを設定
                if use_pretrained:
                    # 事前学習済みモデルを使用
                    model = YOLO(f"{model_type}.pt")
                    pretrained_info = "事前学習済みの重み"
                else:
                    # 現在読み込まれているモデルを使用
                    if hasattr(self, 'yolo_model_file') and os.path.exists(self.yolo_model_file):
                        model = YOLO(self.yolo_model_file)
                        pretrained_info = f"現在のモデル重み: {os.path.basename(self.yolo_model_file)}"
                    else:
                        # モデルが見つからない場合は事前学習済みモデルにフォールバック
                        model = YOLO(f"{model_type}.pt")
                        pretrained_info = "事前学習済みの重み (現在のモデルが見つからないため)"
                
                progress.setLabelText("学習開始...")
                progress.setValue(20)
                QApplication.processEvents()
                
                # 学習設定
                results = model.train(
                    data=yaml_file,
                    epochs=num_epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    project=models_dir,
                    name=run_name,
                    device=device.type,
                    workers=0,
                    close_mosaic=10 if mosaic > 0 else 0,
                    patience=patience,
                    exist_ok=True,
                    lr0=learning_rate,
                    lrf=learning_rate / 10,
                    # オーグメンテーション設定
                    mosaic=mosaic,
                    fliplr=fliplr,
                    hsv_h=hsv_h,
                    hsv_s=hsv_s,
                    hsv_v=hsv_v,
                    translate=translate,
                    scale=scale,
                    erasing=erasing
                )
                
                progress.setValue(95)
                QApplication.processEvents()
                
                # モデルリストを更新
                if hasattr(self, 'refresh_yolo_model_list'):
                    self.refresh_yolo_model_list()
                
                progress.setValue(100)
                progress.close()
                
                # 学習結果を表示
                QMessageBox.information(
                    self,
                    "学習完了",
                    f"YOLOモデルの学習が完了しました。\n"
                    f"最終mAP: {results.maps}\n"
                    f"使用デバイス: {device}\n"
                    f"初期化: {pretrained_info}\n\n"
                    f"モデル保存先: {os.path.join(models_dir, run_name, 'weights')}\n"
                    f"MLflow実験名: {experiment_name}"
                )
            
            except Exception as inner_e:
                print(f"YOLO学習中の内部エラー: {str(inner_e)}")
                progress.close()
                QMessageBox.critical(
                    self,
                    "トレーニングエラー",
                    f"YOLO学習プロセス中にエラーが発生しました: {str(inner_e)}"
                )
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "エラー",
                f"YOLOモデル学習中にエラーが発生しました: {str(e)}"
            )
    
    def train_and_save_yolo_model(self):
        """Ultralytics YOLOモデルを学習し保存する - 事前学習済みモデルを自動ダウンロード"""        
        if not self.bbox_annotations:
            QMessageBox.warning(self, "警告", "物体検知アノテーションがありません。")
            return
        
        # 選択したモデルとクラス設定を取得
        model_type = self.yolo_model_combo.currentText()
        classes = [cls.strip() for cls in self.classes_input.text().split(',') if cls.strip()]
        
        if not classes:
            QMessageBox.warning(self, "警告", "検知クラスを最低1つ設定してください。")
            return
        
        # 学習設定ダイアログを表示
        training_settings = QDialog(self)
        training_settings.setWindowTitle("YOLOモデル学習設定")
        training_settings.setMinimumWidth(500)
        training_settings.setMinimumHeight(600)
        
        settings_layout = QVBoxLayout(training_settings)
        
        # タブウィジェットを作成
        tabs = QTabWidget()
        
        # 基本設定タブ
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # モデル初期化設定
        init_group = QGroupBox("モデル初期化設定")
        init_layout = QVBoxLayout(init_group)
        
        # 初期重みの選択
        weights_radio_pretrained = QRadioButton("事前学習済みの重みを使用 (推奨)")
        weights_radio_pretrained.setChecked(True)  # デフォルト選択
        init_layout.addWidget(weights_radio_pretrained)
        
        # 現在のモデルを選択
        weights_radio_current = QRadioButton("現在読み込まれているモデルの重みを使用")
        init_layout.addWidget(weights_radio_current)
        
        # 現在読み込まれているモデルの情報を表示
        current_model_info = QLabel("現在のモデル: なし")
        if hasattr(self, 'yolo_model') and hasattr(self, 'yolo_model_file'):
            model_name = os.path.basename(self.yolo_model_file) if hasattr(self, 'yolo_model_file') else "Unknown"
            current_model_info.setText(f"現在のモデル: {model_name}")
            weights_radio_current.setEnabled(True)
        else:
            weights_radio_current.setEnabled(False)
            current_model_info.setText("現在のモデル: なし（先にモデルを読み込んでください）")
        
        init_layout.addWidget(current_model_info)
        basic_layout.addWidget(init_group)
        
        # エポック数設定
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("学習エポック数:"))
        epoch_spin = QSpinBox()
        epoch_spin.setRange(1, 1000)
        epoch_spin.setValue(30)  # デフォルト: 30エポック
        epoch_layout.addWidget(epoch_spin)
        basic_layout.addLayout(epoch_layout)
        
        # バッチサイズ設定
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("バッチサイズ:"))
        batch_spin = QSpinBox()
        batch_spin.setRange(1, 128)
        batch_spin.setValue(16)  # デフォルト: 16
        batch_layout.addWidget(batch_spin)
        basic_layout.addLayout(batch_layout)
        
        # 入力サイズ設定
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("入力画像サイズ:"))
        size_combo = QComboBox()
        size_options = [str(self.original_image_size),"320", "416", "512", "640", "768", "896", "1024"]
        default_index = 4  # デフォルトは640

        # 説明ラベルを追加
        size_layout.addWidget(QLabel(f"元画像: {self.original_image_width}×{self.original_image_height}"))

        size_combo.addItems(size_options)
        size_combo.setCurrentIndex(default_index)
        size_layout.addWidget(size_combo)
        basic_layout.addLayout(size_layout)

        # 注意書き
        size_note = QLabel("注: 640以外のサイズを選択すると精度や速度に影響します")
        size_note.setStyleSheet("color: #888; font-style: italic;")
        basic_layout.addWidget(size_note)

        # Early Stopping設定
        early_stopping_check = QCheckBox("Early Stopping を有効にする")
        early_stopping_check.setChecked(True)
        basic_layout.addWidget(early_stopping_check)
        
        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("忍耐エポック数:"))
        patience_spin = QSpinBox()
        patience_spin.setRange(1, 20)
        patience_spin.setValue(10)
        patience_spin.setEnabled(True)
        patience_layout.addWidget(patience_spin)
        basic_layout.addLayout(patience_layout)
        
        # 学習率設定
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学習率:"))
        
        lr_combo = QComboBox()
        learning_rates = ["0.01", "0.005", "0.001", "0.0005", "0.0001"]
        lr_combo.addItems(learning_rates)
        lr_combo.setCurrentIndex(2)  # デフォルト: 0.001
        lr_layout.addWidget(lr_combo)
        basic_layout.addLayout(lr_layout)
        
        # タブに追加
        tabs.addTab(basic_tab, "基本設定")
        
        # データオーグメンテーションタブ
        aug_tab = QWidget()
        aug_layout = QVBoxLayout(aug_tab)
        
        # データオーグメンテーション有効化チェックボックス
        aug_enable_check = QCheckBox("データオーグメンテーションを有効にする")
        aug_enable_check.setChecked(True)
        aug_layout.addWidget(aug_enable_check)
        
        # オーグメンテーション設定のスクロールエリア
        aug_scroll = QScrollArea()
        aug_scroll.setWidgetResizable(True)
        aug_scroll.setFrameShape(QFrame.NoFrame)
        
        aug_scroll_content = QWidget()
        aug_options_layout = QVBoxLayout(aug_scroll_content)
        
        # モザイク
        mosaic_layout = QHBoxLayout()
        aug_mosaic_checkbox = QCheckBox("モザイク")
        aug_mosaic_checkbox.setChecked(True)
        aug_mosaic_proba_label = QLabel("確率:")
        aug_mosaic_proba = QDoubleSpinBox()
        aug_mosaic_proba.setRange(0.0, 1.0)
        aug_mosaic_proba.setSingleStep(0.1)
        aug_mosaic_proba.setValue(1.0)
        mosaic_layout.addWidget(aug_mosaic_checkbox)
        mosaic_layout.addWidget(aug_mosaic_proba_label)
        mosaic_layout.addWidget(aug_mosaic_proba)
        mosaic_layout.addStretch()
        aug_options_layout.addLayout(mosaic_layout)
        
        # 水平反転
        flip_layout = QHBoxLayout()
        aug_flip_checkbox = QCheckBox("水平反転")
        aug_flip_checkbox.setChecked(True)
        aug_flip_proba_label = QLabel("確率:")
        aug_flip_proba = QDoubleSpinBox()
        aug_flip_proba.setRange(0.0, 1.0)
        aug_flip_proba.setSingleStep(0.1)
        aug_flip_proba.setValue(0.5)
        flip_layout.addWidget(aug_flip_checkbox)
        flip_layout.addWidget(aug_flip_proba_label)
        flip_layout.addWidget(aug_flip_proba)
        flip_layout.addStretch()
        aug_options_layout.addLayout(flip_layout)
        
        # HSV調整
        hsv_layout = QHBoxLayout()
        aug_hsv_checkbox = QCheckBox("HSV調整")
        aug_hsv_checkbox.setChecked(True)
        hsv_layout.addWidget(aug_hsv_checkbox)
        hsv_layout.addStretch()
        aug_options_layout.addLayout(hsv_layout)
        
        # HSVの詳細設定
        hsv_details_layout = QGridLayout()
        hsv_details_layout.setContentsMargins(20, 0, 0, 0)
        
        hsv_details_layout.addWidget(QLabel("色相 (H):"), 0, 0)
        aug_hsv_h = QDoubleSpinBox()
        aug_hsv_h.setRange(0.0, 0.1)
        aug_hsv_h.setSingleStep(0.005)
        aug_hsv_h.setValue(0.015)
        hsv_details_layout.addWidget(aug_hsv_h, 0, 1)
        
        hsv_details_layout.addWidget(QLabel("彩度 (S):"), 1, 0)
        aug_hsv_s = QDoubleSpinBox()
        aug_hsv_s.setRange(0.0, 1.0)
        aug_hsv_s.setSingleStep(0.1)
        aug_hsv_s.setValue(0.7)
        hsv_details_layout.addWidget(aug_hsv_s, 1, 1)
        
        hsv_details_layout.addWidget(QLabel("明度 (V):"), 2, 0)
        aug_hsv_v = QDoubleSpinBox()
        aug_hsv_v.setRange(0.0, 1.0)
        aug_hsv_v.setSingleStep(0.1)
        aug_hsv_v.setValue(0.4)
        hsv_details_layout.addWidget(aug_hsv_v, 2, 1)
        
        aug_options_layout.addLayout(hsv_details_layout)
        
        # 幾何変換
        geometry_layout = QHBoxLayout()
        aug_geometry_checkbox = QCheckBox("幾何変換")
        aug_geometry_checkbox.setChecked(True)
        geometry_layout.addWidget(aug_geometry_checkbox)
        geometry_layout.addStretch()
        aug_options_layout.addLayout(geometry_layout)
        
        # 幾何変換の詳細設定
        geometry_details_layout = QGridLayout()
        geometry_details_layout.setContentsMargins(20, 0, 0, 0)
        
        geometry_details_layout.addWidget(QLabel("平行移動:"), 0, 0)
        aug_translate = QDoubleSpinBox()
        aug_translate.setRange(0.0, 0.5)
        aug_translate.setSingleStep(0.05)
        aug_translate.setValue(0.1)
        geometry_details_layout.addWidget(aug_translate, 0, 1)
        
        geometry_details_layout.addWidget(QLabel("スケール:"), 1, 0)
        aug_scale = QDoubleSpinBox()
        aug_scale.setRange(0.0, 1.0)
        aug_scale.setSingleStep(0.05)
        aug_scale.setValue(0.5)
        geometry_details_layout.addWidget(aug_scale, 1, 1)
        
        aug_options_layout.addLayout(geometry_details_layout)
        
        # RandomErase
        erase_layout = QHBoxLayout()
        aug_erase_checkbox = QCheckBox("ランダムイレース")
        aug_erase_checkbox.setChecked(True)
        aug_erase_proba_label = QLabel("確率:")
        aug_erase_proba = QDoubleSpinBox()
        aug_erase_proba.setRange(0.0, 1.0)
        aug_erase_proba.setSingleStep(0.1)
        aug_erase_proba.setValue(0.4)
        erase_layout.addWidget(aug_erase_checkbox)
        erase_layout.addWidget(aug_erase_proba_label)
        erase_layout.addWidget(aug_erase_proba)
        erase_layout.addStretch()
        aug_options_layout.addLayout(erase_layout)
        
        # オプションの有効/無効を連動させる
        def toggle_aug_options(checked):
            for w in aug_scroll_content.findChildren(QWidget):
                if w != aug_enable_check:
                    w.setEnabled(checked)
        
        aug_enable_check.toggled.connect(toggle_aug_options)
        
        # スクロールエリアに設定
        aug_scroll.setWidget(aug_scroll_content)
        aug_layout.addWidget(aug_scroll)
        
        # タブに追加
        tabs.addTab(aug_tab, "データオーグメンテーション")
        
        # タブをレイアウトに追加
        settings_layout.addWidget(tabs)
        
        # ボタンの配置
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(training_settings.accept)
        button_box.rejected.connect(training_settings.reject)
        settings_layout.addWidget(button_box)
        
        # ダイアログを表示
        if not training_settings.exec_():
            return
        
        # 設定値の取得
        use_pretrained = weights_radio_pretrained.isChecked()
        num_epochs = epoch_spin.value()
        batch_size = batch_spin.value()
        img_size = int(size_combo.currentText())
        use_early_stopping = early_stopping_check.isChecked()
        patience = patience_spin.value() if use_early_stopping else 0
        learning_rate = float(lr_combo.currentText())
        
        # オーグメンテーション設定の取得
        augmentation_enabled = aug_enable_check.isChecked()
        mosaic = aug_mosaic_proba.value() if aug_mosaic_checkbox.isChecked() and augmentation_enabled else 0.0
        fliplr = aug_flip_proba.value() if aug_flip_checkbox.isChecked() and augmentation_enabled else 0.0
        hsv_h = aug_hsv_h.value() if aug_hsv_checkbox.isChecked() and augmentation_enabled else 0.0
        hsv_s = aug_hsv_s.value() if aug_hsv_checkbox.isChecked() and augmentation_enabled else 0.0
        hsv_v = aug_hsv_v.value() if aug_hsv_checkbox.isChecked() and augmentation_enabled else 0.0
        translate = aug_translate.value() if aug_geometry_checkbox.isChecked() and augmentation_enabled else 0.0
        scale = aug_scale.value() if aug_geometry_checkbox.isChecked() and augmentation_enabled else 0.0
        erasing = aug_erase_proba.value() if aug_erase_checkbox.isChecked() and augmentation_enabled else 0.0
        
        # YOLOフォーマット用のデータを生成（YOLO用ディレクトリ構造を作成）
        try:            
            # データディレクトリ構造の作成
            train_dir = os.path.join(yolo_dataset_dir, "train")
            val_dir = os.path.join(yolo_dataset_dir, "val")
            os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
            os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
            
            # クラス名ファイルの保存
            with open(os.path.join(yolo_dataset_dir, "classes.txt"), 'w') as f:
                for cls in classes:
                    f.write(f"{cls}\n")
            
            # データセット設定YAMLファイルの作成
            yaml_content = f"""
    path: {yolo_dataset_dir}
    train: train/images
    val: val/images
    test: test/images

    nc: {len(classes)}
    names: {classes}
            """
            
            yaml_file = os.path.join(yolo_dataset_dir, "dataset.yaml")
            with open(yaml_file, 'w') as f:
                f.write(yaml_content)
            
            # アノテーションデータのエクスポート
            self.export_annotations_to_yolo(train_dir, val_dir, classes)
            
            # Ultralytics YOLOモデルとMLflowのインポート
            try:
                settings.update({"mlflow": True})
            except ImportError as e:
                missing_package = "ultralytics" if "ultralytics" in str(e) else "mlflow" if "mlflow" in str(e) else "依存パッケージ"
                QMessageBox.critical(self, "エラー", f"{missing_package}パッケージがインストールされていません。\npip install {missing_package} でインストールしてください。")
                return
            
            # デバイスの選択
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device for YOLO training: {device}")
            
            # 学習用の進捗ダイアログ
            progress = QProgressDialog(
                f"YOLOモデル '{model_type}' の学習準備中...", 
                "キャンセル", 0, 100, self
            )
            progress.setWindowTitle("YOLOモデル学習")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # ここが新しい部分: 事前学習済みモデルを自動ダウンロード
            pretrained_model_path = None
            model_path = None
            
            if use_pretrained:
                # 事前学習済みモデルをダウンロード
                progress.setLabelText(f"事前学習済み {model_type} モデルをダウンロードしています...")
                progress.setValue(5)
                QApplication.processEvents()
                
                pretrained_model_path = self.download_pretrained_yolo_model(model_type)
                if not pretrained_model_path:
                    progress.close()
                    QMessageBox.critical(self, "エラー", f"事前学習済み {model_type} モデルの準備に失敗しました。")
                    return
            else:
                # 現在ロードされているモデルを使用
                if hasattr(self, 'yolo_model_file') and os.path.exists(self.yolo_model_file):
                    model_path = self.yolo_model_file
                else:
                    progress.close()
                    QMessageBox.critical(self, "エラー", "現在のモデルが読み込まれていません。事前学習済みモデルを使用するか、モデルを読み込んでから再試行してください。")
                    return
                            
            
            # Windows環境での正しいURI形式を構築
            if sys.platform.startswith('win'):
                tracking_uri = f"file:///{normalized_path}"
            else:
                tracking_uri = f"file://{normalized_path}"
            
            print(f"YOLOトレーニング用MLflowトラッキングURI: {tracking_uri}")
                                    
            # 重要: 実験名を固定の文字列に設定
            experiment_name = "yolo_training"
            
            # 実験が存在するか確認し、なければ作成
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            
            # YOLOの設定は環境変数から読み込みになる
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
                        
            # トレーニング設定のカスタマイズ
            run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            progress.setLabelText("YOLOモデルを初期化中...")
            progress.setValue(10)
            QApplication.processEvents()
                        
            # 分離プロセスで学習を実行
            try:
                # モデルの読み込み - 選択に基づいて初期重みを設定
                if use_pretrained:
                    # ダウンロードした事前学習済みモデルを使用
                    model = YOLO(pretrained_model_path)
                    pretrained_info = f"事前学習済みの重み (ダウンロード済み: {os.path.basename(pretrained_model_path)})"
                else:
                    # 現在読み込まれているモデルを使用
                    model = YOLO(model_path)
                    pretrained_info = f"現在のモデル重み: {os.path.basename(model_path)}"
                
                progress.setLabelText("学習開始...")
                progress.setValue(20)
                QApplication.processEvents()
                
                # 学習設定
                results = model.train(
                    data=yaml_file,
                    epochs=num_epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    project=models_dir,
                    name=run_name,
                    device=device.type,
                    workers=0,
                    close_mosaic=10 if mosaic > 0 else 0,
                    patience=patience,
                    exist_ok=True,
                    lr0=learning_rate,
                    lrf=learning_rate / 10,
                    # オーグメンテーション設定
                    mosaic=mosaic,
                    fliplr=fliplr,
                    hsv_h=hsv_h,
                    hsv_s=hsv_s,
                    hsv_v=hsv_v,
                    translate=translate,
                    scale=scale,
                    erasing=erasing
                )
                
                progress.setValue(95)
                QApplication.processEvents()
                
                # モデルリストを更新 - 選択したタイプでフィルタリングするように変更
                self.refresh_yolo_model_list()
                
                progress.setValue(100)
                progress.close()
                
                # 学習結果を表示
                QMessageBox.information(
                    self,
                    "学習完了",
                    f"YOLOモデルの学習が完了しました。\n"
                    f"最終mAP: {results.maps}\n"
                    f"使用デバイス: {device}\n"
                    f"初期化: {pretrained_info}\n\n"
                    f"モデル保存先: {os.path.join(models_dir, run_name, 'weights')}\n"
                    f"MLflow実験名: {experiment_name}"
                )
            
            except Exception as inner_e:
                print(f"YOLO学習中の内部エラー: {str(inner_e)}")
                progress.close()
                QMessageBox.critical(
                    self,
                    "トレーニングエラー",
                    f"YOLO学習プロセス中にエラーが発生しました: {str(inner_e)}"
                )
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "エラー",
                f"YOLOモデル学習中にエラーが発生しました: {str(e)}"
            )


    def export_annotations_to_yolo(self, train_dir, val_dir, classes):
        """アノテーションデータをYOLO形式にエクスポート"""
        # アノテーションをトレーニング用とバリデーション用に分割
        all_annotations = list(self.bbox_annotations.items())
        random.shuffle(all_annotations)
        split_idx = int(len(all_annotations) * 0.8)  # 80%をトレーニング用
        
        train_annotations = all_annotations[:split_idx]
        val_annotations = all_annotations[split_idx:]
        
        # トレーニングデータの処理
        self._process_yolo_annotations(train_annotations, train_dir, classes)
        
        # バリデーションデータの処理
        self._process_yolo_annotations(val_annotations, val_dir, classes)
        
    def _process_yolo_annotations(self, annotations, output_dir, classes):
        """YOLO形式でアノテーションを処理"""
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        
        for img_path, bboxes in annotations:
            # 画像をコピー
            img_filename = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(images_dir, img_filename))
            
            # ラベルファイルを作成
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            with open(os.path.join(labels_dir, label_filename), 'w') as f:
                for bbox in bboxes:
                    # クラスインデックスを取得
                    class_idx = classes.index(bbox['class']) if bbox['class'] in classes else 0
                    
                    # YOLO形式に変換
                    x_center = (bbox['x1'] + bbox['x2']) / 2
                    y_center = (bbox['y1'] + bbox['y2']) / 2
                    width = bbox['x2'] - bbox['x1']
                    height = bbox['y2'] - bbox['y1']
                    
                    # クラスインデックス x_center y_center width height
                    f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")

    def on_training_mode_changed(self, index):
        """互換性のために残しておく（現在は toggle_training_mode を使用）"""
        # 現在のインデックスに基づいてボタンの状態を設定
        if hasattr(self, 'auto_train_mode_button') and hasattr(self, 'obj_train_mode_button'):
            if index == 0:
                self.auto_train_mode_button.setChecked(True)
                self.obj_train_mode_button.setChecked(False)
                self.auto_method_container.setVisible(True)
                self.object_detection_container.setVisible(False)
            else:
                self.auto_train_mode_button.setChecked(False)
                self.obj_train_mode_button.setChecked(True)
                self.auto_method_container.setVisible(False)
                self.object_detection_container.setVisible(True)
        else:
            # 従来の方法でコンテナの表示/非表示を切り替え
            if index == 0:
                # 自動運転モデル学習モード
                self.auto_method_container.setVisible(True)
                self.object_detection_container.setVisible(False)
            else:
                # 物体検知モデル学習モード
                self.auto_method_container.setVisible(False)
                self.object_detection_container.setVisible(True)

    def select_object_class(self):
        """物体クラスを選択するダイアログを表示 - 前回選択したクラスを初期選択にする"""
        classes = [cls.strip() for cls in self.classes_input.text().split(',') if cls.strip()]
        if not classes:
            classes = ["car", "person", "sign", "cone"]  # デフォルト
        
        # 前回選択したクラスのインデックスを取得
        default_index = 0
        if self.last_selected_bbox_class and self.last_selected_bbox_class in classes:
            default_index = classes.index(self.last_selected_bbox_class)
        
        class_name, ok = QInputDialog.getItem(
            self, 
            "クラス選択", 
            "オブジェクトのクラスを選択してください:",
            classes, 
            default_index,  # 前回選択したクラスのインデックスを初期選択にする
            False
        )
        
        if ok and class_name:
            # 選択したクラスを記録
            self.last_selected_bbox_class = class_name
            return class_name
        return None

    def add_bbox_annotation(self, bbox):
        """バウンディングボックスアノテーションを追加"""
        if not self.images:
            return
        
        current_img_path = self.images[self.current_index]
        
        # 既存のアノテーションを取得または新規作成
        if current_img_path not in self.bbox_annotations:
            self.bbox_annotations[current_img_path] = []
        
        # バウンディングボックスを追加
        self.bbox_annotations[current_img_path].append(bbox)
        
        # 前回のバウンディングボックスとして保存
        self.last_bbox = bbox.copy()
            
        # 現在のすべてのバウンディングボックスを保存
        self.last_bboxes = [box.copy() for box in self.bbox_annotations[current_img_path]]
    
        # 統計情報更新
        self.update_bbox_stats()
        
        # 画面更新
        self.main_image_view.update()
        
        # 左パネルのアノテーション情報を更新
        if hasattr(self, 'update_annotation_info_label'):
            # 物体検知アノテーション情報を取得
            bbox_info = self.update_annotation_info_label()
            
            # 既存のアノテーション情報と結合
            if self.current_index in self.annotations :
                # 自動運転アノテーションがある場合
                anno = self.annotations[self.current_index]
                
                # 基本的なアノテーション情報
                annotation_text = f"<b>運転アノテーション情報:</b><br>"
                annotation_text += f"angle = <span style='color: #FF6666;'>{anno['angle']:.4f}</span><br>"
                annotation_text += f"throttle = <span style='color: #FF6666;'>{anno['throttle']:.4f}</span>"
                
                # 位置情報があれば追加
                if 'loc' in anno:
                    loc_value = anno['loc']
                    loc_color = get_location_color(loc_value)
                    
                    annotation_text += f"<br><div style='margin-top: 10px;'>"
                    annotation_text += f"<div style='display: inline-block; background-color: {loc_color.name()}; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'>"
                    annotation_text += f"位置 {loc_value}</div></div>"
                
                # 物体検知情報を追加
                if bbox_info:
                    annotation_text += f"<br><br>{bbox_info}"
                
                self.annotation_info_label.setText(annotation_text)
                self.annotation_info_label.setTextFormat(Qt.RichText)
            else:
                # 自動運転アノテーションがない場合は物体検知情報のみ表示
                if bbox_info:
                    self.annotation_info_label.setText(bbox_info)
                    self.annotation_info_label.setTextFormat(Qt.RichText)
        
        # ギャラリーを更新
        self.update_gallery()
        
        # メッセージ表示
        class_name = bbox.get('class', 'unknown')
        self.statusBar().showMessage(f"'{class_name}' のバウンディングボックスを追加しました", 3000)

    def add_session_check_to_init_ui(self):
        """init_uiメソッドの最後に追加する初期セッション確認コード"""
        # 保存されたセッション情報を読み込む
        session_info = self.load_session_info()
        
        # 複数フォルダを優先的に使用
        has_folders = False
        
        # 前回の複数フォルダパスがあるか確認
        if session_info and "last_folder_paths" in session_info and session_info["last_folder_paths"]:
            folder_paths = session_info["last_folder_paths"]
            
            # フォルダが存在するか確認
            valid_paths = [path for path in folder_paths if os.path.exists(path)]
            
            if valid_paths:
                # 確認ダイアログを表示
                reply = QMessageBox.question(
                    self, 
                    "前回のセッションを復元", 
                    f"前回の作業フォルダ（{len(valid_paths)}個）を読み込みますか？\n\n"
                    f"最初のフォルダ: {valid_paths[0]}\n" +
                    (f"他 {len(valid_paths)-1} フォルダ" if len(valid_paths) > 1 else ""),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    # フォルダパスを設定し、画像を読み込む
                    self.folder_input.setText(";".join(valid_paths))
                    has_folders = True
                    
                    # UIが完全に初期化された後で画像読み込みを実行するために遅延実行
                    QTimer.singleShot(100, self.load_images)
        
        # 複数フォルダが見つからなかった場合は単一フォルダを使用
        if not has_folders and session_info and "last_folder_path" in session_info and session_info["last_folder_path"]:
            last_folder = session_info["last_folder_path"]
            
            # フォルダが存在するか確認
            if os.path.exists(last_folder):
                # 確認ダイアログを表示
                reply = QMessageBox.question(
                    self, 
                    "前回のセッションを復元", 
                    f"前回の作業フォルダを読み込みますか？\n\nフォルダ: {last_folder}",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    # フォルダパスを設定し、画像を読み込む
                    self.folder_input.setText(last_folder)
                    
                    # UIが完全に初期化された後で画像読み込みを実行するために遅延実行
                    QTimer.singleShot(100, self.load_images)

    def set_annotation_buttons_enabled(self, enabled):
        """アノテーション関連ボタンの有効/無効を一括制御する"""
        # 無効化対象となるアノテーション関連ボタンのリスト
        annotation_buttons = [
            self.load_annotation_button,       # アノテーションデータを読込ボタン
            self.model_load_button,            # モデル読込ボタン
            self.model_refresh_button,         # モデル一覧更新ボタン
            self.inference_checkbox,           # 推論結果表示チェックボックス
        ]
        
        # 検索してボタン追加（UIから見つける方法）
        additional_buttons = []
        for button in self.findChildren(QPushButton):
            # ボタンのテキストで判断
            button_text = button.text()
            if any(keyword in button_text for keyword in [
                "Donkeycar形式", "Jetracer形式", "アノテーション動画作成", 
                "オートアノテーション実行", "一括推論実行",
                "モデルを学習・保存"
            ]):
                additional_buttons.append(button)
        
        # すべてのボタンリストを統合
        all_buttons = annotation_buttons + additional_buttons
        
        # ボタンの有効/無効を設定
        for button in all_buttons:
            if button:  # Noneでない場合のみ設定
                button.setEnabled(enabled)
        
        # ボタンの色も状態に応じて変更
        button_style = "" if enabled else "QPushButton:disabled { color: #aaaaaa; }"
        for button in all_buttons:
            if button and not isinstance(button, QCheckBox):  # チェックボックス以外のボタンにスタイル適用
                current_style = button.styleSheet()
                if "background-color" not in current_style:  # 特殊スタイルがないボタンのみ
                    button.setStyleSheet(button_style)

    def set_clip_start_to_current(self):
        """現在のインデックスをクリップ開始位置に設定する"""
        if not self.images:
            return
        
        self.clip_start_spin.setValue(self.current_index)

    def set_clip_end_to_current(self):
        """現在のインデックスをクリップ終了位置に設定する"""
        if not self.images:
            return
        
        self.clip_end_spin.setValue(self.current_index)

    def delete_current_annotation(self):
        """現在表示中のアノテーションを削除する"""
        if not self.images:
            return
                
        current_img_path = self.images[self.current_index]
        
        # 確認ダイアログ
        reply = QMessageBox.question(
            self, 
            "削除確認", 
            f"現在のアノテーション（インデックス: {self.current_index}）を削除しますか？\n"
            f"ファイル: {os.path.basename(current_img_path)}",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # アノテーションが存在する場合は削除
        if self.current_index in self.annotations:
            # 削除したインデックスを記録
            # 元のインデックスがある場合はそれを使用
            original_index = self.annotations[self.current_index].get("original_index")
            if original_index is not None:
                self.deleted_indexes.append(original_index)
            else:
                self.deleted_indexes.append(self.current_index)
                
            # 削除したインデックスをソートして保持
            self.deleted_indexes = sorted(list(set(self.deleted_indexes)))
            
            # アノテーション削除
            del self.annotations[self.current_index]
            
            # アノテーションタイムスタンプも削除
            if current_img_path in self.annotation_timestamps:
                del self.annotation_timestamps[current_img_path]
            
            # 位置情報も削除
            if current_img_path in self.location_annotations:
                del self.location_annotations[current_img_path]
            
            # 推論結果も削除
            if current_img_path in self.inference_results:
                del self.inference_results[current_img_path]
            
            # アノテーション数を更新
            self.annotated_count = len(self.annotations)
            
            # UI更新
            self.update_stats()
            self.display_current_image()
            self.update_gallery()
            self.update_location_button_counts()
            
            QMessageBox.information(
                self, 
                "削除完了", 
                f"インデックス {self.current_index} のアノテーションを削除しました。"
                f"\n\n削除済みインデックス数: {len(self.deleted_indexes)}"
            )
        else:
            QMessageBox.information(
                self, 
                "情報", 
                "このインデックスにはアノテーションがありません。"
            )

    def delete_clip_range(self):
        """指定範囲のアノテーションを削除する"""
        if not self.images:
            return
        
        # スピンボックスから範囲を取得
        start_idx = self.clip_start_spin.value()
        end_idx = self.clip_end_spin.value()
        
        # 範囲の正当性をチェック
        if start_idx > end_idx:
            QMessageBox.warning(
                self, 
                "警告", 
                "開始インデックスは終了インデックス以下にしてください。"
            )
            return
        
        if start_idx < 0 or end_idx >= len(self.images):
            QMessageBox.warning(
                self, 
                "警告", 
                f"インデックスの範囲は0から{len(self.images)-1}の間で指定してください。"
            )
            return
        
        # 削除対象の範囲内にあるアノテーション数をカウント
        target_paths = self.images[start_idx:end_idx+1]
        annotations_in_range = sum(1 for idx in range(start_idx, end_idx + 1) if idx in self.annotations)

        # 確認ダイアログ
        reply = QMessageBox.question(
            self, 
            "範囲削除確認", 
            f"インデックス {start_idx} から {end_idx} までの"
            f"\n{len(target_paths)}個の画像のうち、{annotations_in_range}個のアノテーションを削除します。"
            f"\n\nこの操作は元に戻せません。続行しますか？",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # 削除を実行
        deleted_count = 0
        for idx in range(start_idx, end_idx + 1):
            if idx in self.annotations:
                # 削除したインデックスを記録
                # 元のインデックスがある場合はそれを使用
                original_index = self.annotations[idx].get("original_index")
                if original_index is not None:
                    self.deleted_indexes.append(original_index)
                else:
                    self.deleted_indexes.append(idx)
                    
                # アノテーション削除
                del self.annotations[idx]
                deleted_count += 1
                
                # 関連データも削除
                if idx in self.annotation_timestamps:
                    del self.annotation_timestamps[idx]
                    
                if idx in self.location_annotations:
                    del self.location_annotations[idx]
                    
                if idx in self.inference_results:
                    del self.inference_results[idx]


        # 削除したインデックスをソートして重複を排除
        self.deleted_indexes = sorted(list(set(self.deleted_indexes)))
        
        # アノテーション数を更新
        self.annotated_count = len(self.annotations)
        
        # UI更新
        self.update_stats()
        self.display_current_image()
        self.update_gallery()
        self.update_location_button_counts()
        
        QMessageBox.information(
            self, 
            "範囲削除完了", 
            f"インデックス {start_idx} から {end_idx} までの範囲から"
            f"\n{deleted_count}個のアノテーションを削除しました。"
            f"\n\n削除済みインデックス数: {len(self.deleted_indexes)}"
        )

    def on_folder_path_changed(self, text):
        """フォルダパスが変更されたときの処理"""
        # パスが入力されているかどうかでボタンの有効/無効を切り替え
        has_path = bool(text.strip())
        self.load_button.setEnabled(has_path)
        self.load_annotation_button.setEnabled(has_path)
        
        # アノテーション関連ボタンは画像が読み込まれるまで無効化
        # 画像読み込みボタンと直接関連するアノテーション読み込みボタンは例外
        if not self.images:
            self.set_annotation_buttons_enabled(False)

    def load_sibling_annotations(self):
        """選択したフォルダと同じ階層にあるアノテーションデータを読み込む - imagesフォルダと同階層のみに限定"""
        if not self.folder_path or not self.images:
            QMessageBox.warning(self, "警告", "先に画像フォルダを選択して画像を読み込んでください。")
            return
        
        # 現在のフォルダの親ディレクトリを取得
        parent_dir = os.path.dirname(self.folder_path)
        
        # アノテーションデータの検索と読み込みを実行
        annotations_loaded = False
        
        try:
            # 読み込み前に既存のデータをクリア（安全のため）
            if self.annotations:
                self.clear_annotations()
                
            # フォルダ直下（imagesフォルダと同じ階層）のマニフェストファイルを確認
            # 以前のバージョンでは、親ディレクトリ自体もチェックしていたが、
            # 今回は選択したフォルダ直下のみに絞る
            manifest_path = os.path.join(self.folder_path, "manifest.json")
            if os.path.exists(manifest_path):
                # マニフェストベースの読み込み（複数カタログ対応）
                if self.load_catalog_annotations(self.folder_path):
                    annotations_loaded = True
                    QMessageBox.information(
                        self, 
                        "読み込み成功", 
                        f"同階層から{len(self.annotations)}個のアノテーションを読み込みました。"
                    )
            else:
                # 単一カタログファイルの確認 - フォルダ直下のみ
                catalog_files = [f for f in os.listdir(self.folder_path) if f.endswith('.catalog')]
                if catalog_files:
                    catalog_path = os.path.join(self.folder_path, catalog_files[0])
                    if self.load_catalog_annotations(os.path.dirname(catalog_path)):
                        annotations_loaded = True
                        QMessageBox.information(
                            self, 
                            "読み込み成功", 
                            f"同階層から{len(self.annotations)}個のアノテーションを読み込みました。"
                        )
            
            if not annotations_loaded:
                QMessageBox.warning(
                    self, 
                    "警告", 
                    "選択したフォルダと同じ階層から読み込めるアノテーションデータがありませんでした。"
                )
                return
            
            # Update UI
            self.update_stats()
            self.display_current_image()
            self.update_gallery()
            
            # 位置ボタンのカウント表示を更新
            self.update_location_button_counts()
            
            print(f"同階層アノテーション読み込み完了: {len(self.annotations)}個のアノテーション")
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "エラー", 
                f"同階層アノテーションの読み込み中にエラーが発生しました: {str(e)}"
            )

    def update_skip_button_labels(self, value):
        """スキップボタンのラベルを更新する"""
        self.prev_multi_button.setText(f"◀◀ -{value}")
        self.next_multi_button.setText(f"+{value} ▶▶")
        
        # 再生停止中の場合のみ再生ボタンのラベルを更新
        if not hasattr(self, 'auto_play_timer') or not self.auto_play_timer.isActive():
            # 逆再生と順再生のボタンを見つける
            for button in self.findChildren(QPushButton):
                if button.text() == "⏪":
                    button.setText("⏪")
                elif button.text() == "⏵":
                    button.setText("⏵")

    def slider_changed(self, value):
        """スライダーの値が変更されたときの処理"""
        if self.images and value != self.current_index:
            self.current_index = value
            self.display_current_image()
            
            # 推論表示チェックボックスがONの場合、自動的に現在の画像の推論を実行
            if self.inference_checkbox.isChecked():
                current_img_path = self.images[self.current_index]
                # 推論結果がまだない場合のみ推論を実行
                if current_img_path not in self.inference_results:
                    self.run_inference_check(False)
            
            # 物体検知推論表示の更新
            if self.detection_inference_checkbox.isChecked():
                current_img_path = self.images[self.current_index]
                # 推論結果がまだない場合のみ推論を実行
                if current_img_path not in self.detection_inference_results:
                    self.update_detection_info_panel()
            
            self.update_gallery()

    def toggle_inference_display(self, state):
        show_inference = (state == Qt.Checked)
        self.main_image_view.show_inference = show_inference
        
        # 表示を更新
        self.update_inference_display()
        self.main_image_view.update()
        
        # 再生中なら一度停止して再開（速度調整のため）
        if hasattr(self, 'auto_play_timer') and self.auto_play_timer.isActive():
            is_forward = True  # デフォルト方向
            
            # 再生方向を特定（現在実装では確実に特定できる方法がないため概算）
            if hasattr(self, 'prev_index') and self.prev_index > self.current_index:
                is_forward = False
                
            # 一度停止
            self.auto_play_timer.stop()
            
            # 少し待ってから再開（UIが更新される時間を確保）
            QTimer.singleShot(100, lambda: self.auto_play(is_forward))
        
        # モデル選択部分を更新
        if hasattr(self, 'model_combo'):
            self.refresh_model_list()    

    def run_inference_check(self, all_images=False):
        """推論を実行するメソッド - モデル情報表示を強化、推論実行後に推論表示をオン"""
        if not self.images:
            return
        
        # 現在のモデル情報を取得
        model_type = self.auto_method_combo.currentText()
        selected_model = self.model_combo.currentText()
        
        # 推論対象の画像を決定
        if all_images:
            # 既存の推論結果がある場合は確認ダイアログを表示
            if self.inference_results and len(self.inference_results) > 0:
                reply = QMessageBox.question(
                    self, 
                    "推論結果の再計算確認", 
                    f"現在、{len(self.inference_results)}個の推論結果が保存されています。\n"
                    f"一括推論を実行すると、すべての推論結果が現在のモデル '{model_type} ({selected_model})' を使って再計算されます。\n\n"
                    "続行しますか？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.No:
                    return  # 操作をキャンセル
            
            target_images = self.images
            progress_title = "全画像の推論を実行中..."
        else:
            target_images = [self.images[self.current_index]]
            progress_title = "推論実行中..."
        
        # モデルのパスを取得 (コンボボックスから選択されたモデル)
        model_path = None
        if hasattr(self, 'model_combo') and self.model_combo.currentText() not in ["モデルが見つかりません", "フォルダを選択してください"] and "が見つかりません" not in self.model_combo.currentText():
            # アノテーションフォルダ内のモデルのフルパスを作成
            selected_model = self.model_combo.currentText()
            models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
            model_path = os.path.join(models_dir, selected_model)
            
            # モデルが存在するか確認
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "警告", f"選択されたモデルが見つかりません: {selected_model}")
                return
        
        # モデル変更を検出するための状態を保持
        current_model_info = (model_type, model_path)
        force_reload = False
        
        # モデルが変更された場合のみ強制再読み込み
        if not hasattr(self, '_last_model_info') or self._last_model_info != current_model_info:
            force_reload = True
            self._last_model_info = current_model_info
        
        try:
            # ステータスバーにメッセージ表示
            model_desc = os.path.basename(model_path) if model_path else '事前学習済み'
            self.statusBar().showMessage(f"推論処理中... モデル: {model_type} ({model_desc})")
            QApplication.processEvents()

            # 推論を実行
            if model_type in list_available_models():
                # モデルを使用した推論 - force_reloadはモデル変更時のみTrue
                inference_results = batch_inference(
                    target_images, 
                    method="model", 
                    model_type=model_type,
                    model_path=model_path,
                    force_reload=force_reload
                )
            else:
                QMessageBox.warning(self, "警告", "サポートされていない推論方法です。")
                return
            
            # 推論結果を保存
            old_count = len(self.inference_results)
            self.inference_results.update(inference_results)
            new_count = len(self.inference_results)
            
            # 推論表示チェックボックスを自動的にオンにする
            was_checked = self.inference_checkbox.isChecked()
            self.inference_checkbox.setChecked(True)
            
            # 表示を更新
            self.update_inference_display()
            self.main_image_view.update()
            self.update_gallery()

            # ステータスバーのメッセージをクリア
            self.statusBar().clearMessage()

            # 全画像の推論の場合はメッセージ表示
            if all_images:
                added_results = new_count - old_count
                updated_results = len(target_images) - added_results
                
                check_message = ""
                if not was_checked:
                    check_message = "\n\n推論結果表示が自動的にオンになりました。"
                    
                QMessageBox.information(
                    self, 
                    "推論完了", 
                    f"{len(target_images)}枚の画像に対する推論を完了しました。\n"
                    f"{added_results}個の新しい結果が追加され、{updated_results}個の結果が更新されました。\n\n"
                    f"使用モデル: {model_type} ({model_desc}){check_message}"
                )
            
        except Exception as e:
            self.statusBar().clearMessage()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"推論中にエラーが発生しました: {str(e)}"
            )
    
    def run_batch_inference(self):
        """全ての画像に対して推論を実行する"""
        if not self.images:
            QMessageBox.warning(self, "警告", "画像が読み込まれていません。")
            return
        
        # 現在のモデル情報を取得
        model_type = self.auto_method_combo.currentText()
        selected_model = self.model_combo.currentText()
        
        # モデルのパスを取得
        model_path = None
        if selected_model not in ["モデルが見つかりません", "フォルダを選択してください"] and "が見つかりません" not in selected_model:
            # アノテーションフォルダ内のモデルのフルパスを作成
            models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
            model_path = os.path.join(models_dir, selected_model)
            
            # モデルが存在するか確認
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "警告", f"選択されたモデルが見つかりません: {selected_model}")
                return
        
        # 確認ダイアログ
        reply = QMessageBox.question(
            self, 
            "一括推論実行確認", 
            f"全{len(self.images)}枚の画像に対して推論を実行します。\n"
            f"現在のモデル: {model_type}" + (f" ({os.path.basename(model_path)})" if model_path else " (事前学習済み)") + "\n\n"
            "進行中は操作ができなくなります。続行しますか？",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # 既存の推論結果がある場合の確認
        if hasattr(self, 'inference_results') and self.inference_results:
            clear_reply = QMessageBox.question(
                self, 
                "既存の推論結果", 
                f"現在、{len(self.inference_results)}個の推論結果が保存されています。これらを上書きしますか？\n\n"
                "「はい」: 全ての推論結果を新しいモデルで上書きします。\n"
                "「いいえ」: 推論結果がない画像のみ処理します。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            clear_existing = (clear_reply == QMessageBox.Yes)
        else:
            clear_existing = True
            
        # 進捗ダイアログ
        progress = QProgressDialog(
            "推論処理の準備中...", 
            "キャンセル", 0, len(self.images), self
        )
        progress.setWindowTitle("一括推論実行中")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        try:
            # バッチサイズを設定
            batch_size = 50  # 大量の画像を一度に処理するとメモリ不足になる可能性があるため
            total_batches = (len(self.images) + batch_size - 1) // batch_size
            
            # 初期化
            if clear_existing or not hasattr(self, 'inference_results'):
                self.inference_results = {}
            
            processed_count = 0
            success_count = 0
            skipped_count = 0
            
            # バッチ処理
            for batch_idx in range(total_batches):
                if progress.wasCanceled():
                    break
                    
                # 現在のバッチの画像取得
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(self.images))
                current_batch = self.images[start_idx:end_idx]
                
                # スキップすべき画像をフィルタリング
                if not clear_existing:
                    batch_to_process = [img for img in current_batch if img not in self.inference_results]
                    skipped_count += len(current_batch) - len(batch_to_process)
                else:
                    batch_to_process = current_batch
                
                if not batch_to_process:  # 処理すべき画像がない場合はスキップ
                    processed_count += len(current_batch)
                    progress.setValue(processed_count)
                    continue
                
                progress.setLabelText(
                    f"バッチ {batch_idx+1}/{total_batches} 処理中...\n"
                    f"画像 {start_idx+1}-{end_idx}/{len(self.images)}"
                )
                progress.setValue(processed_count)
                QApplication.processEvents()
                
                # 推論を実行
                try:
                    batch_results = batch_inference(
                        batch_to_process, 
                        method="model", 
                        model_type=model_type,
                        model_path=model_path,
                        force_reload=(batch_idx == 0)  # 最初のバッチのみ強制再読込
                    )
                    
                    # 結果を保存
                    self.inference_results.update(batch_results)
                    success_count += len(batch_results)
                    
                except Exception as e:
                    print(f"バッチ {batch_idx+1} 処理中にエラー: {e}")
                
                processed_count += len(current_batch)
                progress.setValue(processed_count)
                QApplication.processEvents()
            
            # 推論表示を自動的にONにする
            self.inference_checkbox.setChecked(True)
            
            # 現在の画像の表示を更新
            self.update_inference_display()
            self.main_image_view.update()
            self.update_gallery()  # ギャラリー表示も更新
            
            # 処理完了メッセージ
            if progress.wasCanceled():
                QMessageBox.information(
                    self, 
                    "キャンセル", 
                    f"一括推論がキャンセルされました。\n"
                    f"処理済み: {processed_count}/{len(self.images)}枚\n"
                    f"成功: {success_count}枚, スキップ: {skipped_count}枚"
                )
            else:
                QMessageBox.information(
                    self, 
                    "完了", 
                    f"全画像の推論が完了しました。\n"
                    f"処理済み: {len(self.images)}枚\n"
                    f"成功: {success_count}枚, スキップ: {skipped_count}枚\n\n"
                    f"推論結果表示がONになりました。"
                )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"一括推論実行中にエラーが発生しました: {str(e)}"
            )
        finally:
            progress.close()

    def run_location_inference_check(self, all_images=False):
        """位置推論を実行するメソッド"""
        if not self.images:
            return
        
        # 現在のモデル情報を取得
        model_type = self.auto_method_combo.currentText()
        selected_model = self.model_combo.currentText()
        
        # 推論対象の画像を決定
        if all_images:
            target_images = self.images
            progress_title = "全画像の位置推論を実行中..."
        else:
            target_images = [self.images[self.current_index]]
            progress_title = "位置推論実行中..."
        
        # モデルのパスを取得
        model_path = None
        if hasattr(self, 'model_combo') and self.model_combo.currentText() not in ["モデルが見つかりません", "フォルダを選択してください"] and "が見つかりません" not in self.model_combo.currentText():
            models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
            model_path = os.path.join(models_dir, selected_model)
            
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "警告", f"選択されたモデルが見つかりません: {selected_model}")
                return
        
        # モデル変更を検出するための状態を保持
        current_model_info = (model_type, model_path)
        force_reload = False
        
        # モデルが変更された場合のみ強制再読み込み
        if not hasattr(self, '_last_location_model_info') or self._last_location_model_info != current_model_info:
            force_reload = True
            self._last_location_model_info = current_model_info
        
        try:
            # ステータスバーにメッセージ表示
            model_desc = os.path.basename(model_path) if model_path else '事前学習済み'
            self.statusBar().showMessage(f"位置推論処理中... モデル: {model_type} ({model_desc})")
            QApplication.processEvents()

            # 推論を実行
            if model_type in list_available_models():
                # モデルを使用した推論
                inference_results = batch_inference(
                    target_images, 
                    method="model", 
                    model_type=model_type,
                    model_path=model_path,
                    force_reload=force_reload
                )
            else:
                QMessageBox.warning(self, "警告", "サポートされていない推論方法です。")
                return
            
            # 推論結果を保存
            old_count = len(self.location_inference_results)
            for img_path, result in inference_results.items():
                # 結果から位置情報を抽出
                if "pilot/loc" in result:
                    location = result["pilot/loc"]
                elif "loc" in result:
                    location = result["loc"]
                else:
                    # 位置情報がない場合、何らかのロジックで判断（例：角度から推定）
                    location = self.estimate_location_from_angle(result.get("angle", 0))
                
                # 位置情報付きの結果を保存
                self.location_inference_results[img_path] = {
                    "loc": location,
                    "x": result.get("x", 0),
                    "y": result.get("y", 0)
                }
            
            new_count = len(self.location_inference_results)
            
            # 位置推論表示チェックボックスを自動的にオンにする
            was_checked = self.location_inference_checkbox.isChecked()
            self.location_inference_checkbox.setChecked(True)
            
            # 表示を更新
            self.update_location_info_panel()
            self.main_image_view.update()

            # ステータスバーのメッセージをクリア
            self.statusBar().clearMessage()

            # 全画像の推論の場合はメッセージ表示
            if all_images:
                added_results = new_count - old_count
                updated_results = len(target_images) - added_results
                
                check_message = ""
                if not was_checked:
                    check_message = "\n\n位置推論結果表示が自動的にオンになりました。"
                    
                QMessageBox.information(
                    self, 
                    "位置推論完了", 
                    f"{len(target_images)}枚の画像に対する位置推論を完了しました。\n"
                    f"{added_results}個の新しい結果が追加され、{updated_results}個の結果が更新されました。\n\n"
                    f"使用モデル: {model_type} ({model_desc}){check_message}"
                )
            
        except Exception as e:
            self.statusBar().clearMessage()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"位置推論中にエラーが発生しました: {str(e)}"
            )
    
    def update_inference_display(self):
        """推論結果の表示を更新する"""
        if not self.images:
            return
                
        current_img_path = self.images[self.current_index]
        
        # 推論結果がある場合、表示を更新
        if current_img_path in self.inference_results:
            inference = self.inference_results[current_img_path]
            
            # 新しいキー形式があればそれを使い、なければ古い形式を使う
            if "pilot/angle" in inference and "pilot/throttle" in inference:
                angle = inference["pilot/angle"]
                throttle = inference["pilot/throttle"]
            else:
                angle = inference["angle"]
                throttle = inference["throttle"]

            # 推論情報のリッチテキスト
            inference_text = f"<b>推論結果:</b><br>"
            inference_text += f"angle = <span style='color: #6666FF;'>{angle:.4f}</span><br>"
            inference_text += f"throttle = <span style='color: #6666FF;'>{throttle:.4f}</span>"

            # 位置情報を取得
            location = None
            if "pilot/loc" in inference:
                location = inference["pilot/loc"]
            elif "loc" in inference:
                location = inference["loc"]

            # 位置情報があれば色付きバッジとして表示（アノテーションと同じスタイル）
            if location is not None:
                loc_color = get_location_color(location)
                
                inference_text += f"<br><div style='margin-top: 10px;'>"
                inference_text += f"<div style='display: inline-block; background-color: {loc_color.name()}; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'>"
                inference_text += f"推論位置 {location}</div></div>"

            # リッチテキストとして設定
            self.inference_info_label.setText(inference_text)
            self.inference_info_label.setTextFormat(Qt.RichText)

            # ImageLabelに推論ポイントを設定
            self.main_image_view.inference_point = QPoint(inference['x'], inference['y'])
        else:
            # 推論結果がない場合はクリア
            self.inference_info_label.setText("")
            self.main_image_view.inference_point = None
        
        # 推論表示のチェック状態を反映
        self.main_image_view.show_inference = self.inference_checkbox.isChecked()
                
    def keyPressEvent(self, event):
        # Bキーでアノテーションモードを切り替え
        if event.key() == Qt.Key_B:
            self.toggle_annotation_mode()
        # 左右キーでの10枚移動（既存の機能）
        elif event.key() == Qt.Key_Left:
            self.skip_images(-10)
        elif event.key() == Qt.Key_Right:
            self.skip_images(10)
        else:
            super().keyPressEvent(event)
            
    def handle_delete_key(self):
        """Delete/Backspace キーの処理を行う"""
        if self.selected_bbox_index is not None:
            current_img_path = self.main_window.images[self.main_window.current_index]
            if current_img_path in self.main_window.bbox_annotations:
                bboxes = self.main_window.bbox_annotations[current_img_path]
                if 0 <= self.selected_bbox_index < len(bboxes):
                    # 選択されたバウンディングボックスを削除
                    del bboxes[self.selected_bbox_index]
                    # インデックスをリセット
                    self.selected_bbox_index = None
                    # 再描画
                    self.update()
                    
                    # バウンディングボックスの統計情報を更新
                    self.main_window.update_bbox_stats()
                    
                    print("バウンディングボックスを削除しました")
                    return True  # キーが処理されたことを示す
        
        return False  # キーが処理されなかったことを示す

    def browse_folder(self):
        """
        画像フォルダを選択するダイアログを表示
        選択されたフォルダの下のimagesフォルダを画像フォルダとして取り扱う
        """        
        # 複数フォルダ選択が可能なダイアログを表示
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        
        # QFileDialogのリストビューを取得して複数選択を可能にする
        listView = dialog.findChild(QListView, "listView")
        if listView:
            listView.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        treeView = dialog.findChild(QTreeView)
        if treeView:
            treeView.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        # 選択されたフォルダを取得
        if dialog.exec_():
            selected_folders = dialog.selectedFiles()
            
            # 複数のフォルダを選択した場合は、セミコロン区切りでテキストフィールドに表示
            if selected_folders:
                # 選択されたフォルダ内にimagesフォルダが存在するか確認
                valid_folders = []
                missing_images_folders = []
                
                for folder in selected_folders:
                    images_path = os.path.join(folder, "images")
                    if os.path.exists(images_path) and os.path.isdir(images_path):
                        valid_folders.append(folder)
                    else:
                        missing_images_folders.append(folder)
                
                # imagesフォルダが見つからなかった場合は警告メッセージを表示
                if missing_images_folders:
                    missing_folders_str = "\n".join(missing_images_folders)
                    QMessageBox.warning(
                        self,
                        "imagesフォルダ未検出",
                        f"次のフォルダ内にimagesフォルダが見つかりませんでした：\n{missing_folders_str}\n\n有効なフォルダのみ処理を続行します。"
                    )
                
                # 有効なフォルダをテキストフィールドに設定
                if valid_folders:
                    self.folder_input.setText(";".join(valid_folders))
                else:
                    self.folder_input.setText("")
                    QMessageBox.critical(
                        self,
                        "エラー",
                        "選択されたフォルダのいずれにもimagesフォルダが含まれていません。\n処理を中止します。"
                    )
                    return
        
        # 少し遅延させてから画像読み込みを実行（UIが更新される時間を確保）
        QTimer.singleShot(100, self.load_images)
        
    def load_images(self):
        """
        選択した各フォルダの下のimagesフォルダから画像を読み込む
        アノテーションは自動では読み込まない
        """
        folder_paths_text = self.folder_input.text()
        
        # セミコロン区切りでフォルダパスを取得
        folder_paths = folder_paths_text.split(";")
        
        # 有効なフォルダパスをチェック
        valid_paths = []
        image_folders = []  # 実際の画像フォルダ（各フォルダ下のimagesフォルダ）
        
        for folder_path in folder_paths:
            folder_path = folder_path.strip()
            if not os.path.exists(folder_path):
                QMessageBox.warning(self, "エラー", f"フォルダが存在しません: {folder_path}")
                continue
                
            # imagesフォルダのパスを取得
            images_folder = os.path.join(folder_path, "images")
            if os.path.exists(images_folder) and os.path.isdir(images_folder):
                # imagesフォルダが存在する場合
                valid_paths.append(folder_path)  # 親フォルダを有効パスとして記録
                image_folders.append(images_folder)  # 実際の画像フォルダを記録
            else:
                QMessageBox.warning(self, "エラー", f"フォルダの下にimagesフォルダが見つかりません: {folder_path}")
        
        if not valid_paths or not image_folders:
            return
        
        # 全画像フォルダの画像を集める
        all_images = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        
        print(f"{len(image_folders)}個のimagesフォルダを検索中...")
        
        for img_folder in image_folders:
            print(f"画像フォルダを検索中: {img_folder}")
            
            # imagesフォルダ内の画像を検索
            try:
                for file in os.listdir(img_folder):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        all_images.append(os.path.join(img_folder, file))
            except Exception as e:
                print(f"画像フォルダ {img_folder} の読み込みエラー: {e}")
        
        if not all_images:
            QMessageBox.warning(self, "エラー", "選択されたフォルダ内のimagesフォルダに画像ファイルがありません。")
            return
        
        print(f"{len(all_images)}枚の画像が見つかりました")
        
        # ファイル名からインデックスを抽出してソート
        image_with_indices = []
        for img_path in all_images:
            basename = os.path.basename(img_path)
            # ファイル名からインデックスを抽出（例: 10900_cam_image_array_.jpg -> 10900）
            try:
                match = re.match(r'^(\d+)_', basename)
                if match:
                    index = int(match.group(1))
                    image_with_indices.append((img_path, index))
                else:
                    # インデックスが抽出できない場合は、高い値（後ろに配置）
                    image_with_indices.append((img_path, float('inf')))
            except Exception as e:
                print(f"ファイル名からインデックス抽出エラー: {basename} - {e}")
                # エラーの場合も高い値で後ろに配置
                image_with_indices.append((img_path, float('inf')))
        
        # インデックスでソート
        image_with_indices.sort(key=lambda x: x[1])
        
        # ソート後の画像パスリストを作成
        images = [img_path for img_path, _ in image_with_indices]

         # --- 追加: 画像グルーピング（インデックス＆キー単位） ---
        self.image_groups = {}  # { index: { variant: path, ... } }
        self.variant_images = {}  # 各キーの画像リスト

        for img_path in images:
            basename = os.path.basename(img_path)
            m = re.match(r'^(\d+)_([A-Za-z0-9]+)_image_array', basename)
            if not m:
                continue
            idx = int(m.group(1))
            variant = m.group(2)
            self.image_groups.setdefault(idx, {})[variant] = img_path
        
            # キー別に画像リストを作成
            if variant not in self.variant_images:
                self.variant_images[variant] = []
            self.variant_images[variant].append(img_path)
        
        self.sorted_indices = sorted(self.image_groups.keys())

        # キー一覧を更新
        self.available_variants = sorted(self.variant_images.keys())
        # デフォルトキーを設定
        if hasattr(self, 'current_variant') and self.current_variant in self.available_variants:
            # 既に選択されているキーがある場合は保持
            pass
        elif 'cam' in self.available_variants:
            # デフォルトはcam
            self.current_variant = 'cam'
        else:
            # camがなければ最初のキー
            self.current_variant = self.available_variants[0]

        # 現在のキーの画像を選択
        images = self.variant_images[self.current_variant]
        print(f"画像データのキー: {self.available_variants}")
        print(f"現在のキー '{self.current_variant}' の画像数: {len(images)}")

        # キーボタン群を更新
        self.update_variant_buttons()

        # 画像ファイルのリストを取得後、最初の画像サイズを取得
        if images:
            try:
                first_image = Image.open(images[0])
                self.original_image_width, self.original_image_height = first_image.size
                self.original_image_size = max(self.original_image_width, self.original_image_height)
                print(f"元の画像サイズ: {self.original_image_width}x{self.original_image_height}")
            except Exception as e:
                print(f"画像サイズの取得エラー: {e}")        
                
        # Reset state
        self.folder_path = valid_paths[0]  # 最初の親フォルダをメインフォルダとして設定
        self.folder_paths = valid_paths    # すべての有効な親フォルダパスを保存
        self.image_folders = image_folders # すべての画像フォルダパス（imagesフォルダ）を保存
        self.images = images
        self.current_index = 0
        self.annotations = {}
        self.annotation_history = []
        self.annotated_count = 0
        self.annotation_timestamps = {}
        self.inference_results = {}
        self.location_annotations = {}
        
        if hasattr(self, 'deleted_indexes'):
            self.deleted_indexes = []

        # スライダーの設定を更新
        if images:
            self.image_slider.setMaximum(len(images) - 1)
            self.image_slider.setValue(0)
            self.slider_value_label.setText(f"1/{len(images)}")
        else:
            self.image_slider.setMaximum(0)
            self.image_slider.setValue(0)
            self.slider_value_label.setText("0/0")
        
        # Update UI
        self.update_stats()
        self.display_current_image()
        self.update_gallery()
        
        # モデルリストを更新
        self.refresh_model_list()
        if use_yolo:
            self.refresh_yolo_model_list()

        # 位置ボタンのカウント表示を更新
        self.update_location_button_counts()
        
        # アノテーション関連ボタンをアクティブ化
        self.set_annotation_buttons_enabled(True)
        
        QMessageBox.information(
            self, 
            "読み込み完了", 
            f"{len(valid_paths)}個のフォルダから合計{len(all_images)}枚の画像を読み込みました。\n"
            f"画像データのキー: {self.available_variants}\n"
            f"現在のキー '{self.current_variant}' の画像数: {len(self.images)}\n"
            f"元の画像サイズ: {self.original_image_width}x{self.original_image_height}\n"
            "\nアノテーションデータは読み込まれていません。"
        )
        
        # 自動的にアノテーションデータ読み込みを促す確認ダイアログ
        reply = QMessageBox.question(
            self, 
            "アノテーションデータ読み込み", 
            "画像読み込みが完了しました。\n"
            "続けてアノテーションデータを読み込みますか？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # アノテーションデータ読み込みメソッドを呼び出す
            self.load_annotations()

    def load_annotations(self):
        """
        アノテーションデータを読み込む
        画像フォルダ（imagesフォルダ）と同じ階層にあるアノテーションデータだけを読み込む
        """
        if not hasattr(self, 'folder_paths') or not self.folder_paths or not self.images:
            QMessageBox.warning(self, "警告", "先に画像フォルダを選択して画像を読み込んでください。")
            return
        
        # 既存のアノテーションがある場合は確認
        if self.annotations:
            reply = QMessageBox.question(
                self, 
                "既存のアノテーションをクリア", 
                f"現在、{len(self.annotations)}個のアノテーションが読み込まれています。\n"
                "新しいアノテーションデータを読み込む前に、既存のデータをクリアしますか？",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Yes:
                self.clear_annotations()
        
        # 進捗ダイアログを表示
        progress = QProgressDialog(
            f"{len(self.folder_paths)}個のフォルダからアノテーションを検索中...", 
            "キャンセル", 0, len(self.folder_paths), self
        )
        progress.setWindowTitle("アノテーション読み込み")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        # アノテーションデータの検索と読み込みを実行
        annotations_loaded = False
        loaded_count = 0
        annotations_by_dir = {} 
        
        try:
            for idx, parent_dir in enumerate(self.folder_paths):
                progress.setValue(idx)
                progress.setLabelText(f"フォルダ {idx+1}/{len(self.folder_paths)} を処理中...\n{parent_dir}")
                QApplication.processEvents()
                
                if progress.wasCanceled():
                    break
                
                # parent_dir直下のみ検索する
                annotations_before = len(self.annotations)
                
                # manifest.jsonを確認（parent_dir直下のみ）
                manifest_path = os.path.join(parent_dir, "manifest.json")
                if os.path.exists(manifest_path):
                    # マニフェストベースの読み込み
                    if self.load_catalog_annotations(parent_dir):
                        annotations_loaded = True
                        loaded_in_dir = len(self.annotations) - annotations_before
                        annotations_by_dir[parent_dir] = loaded_in_dir
                        loaded_count += loaded_in_dir
                        print(f"親ディレクトリ {parent_dir} から {loaded_in_dir} 個のアノテーションを読み込みました")
                else:
                    # カタログファイルの確認（parent_dir直下のみ）
                    try:
                        catalog_files = [f for f in os.listdir(parent_dir) if f.endswith('.catalog')]
                        if catalog_files:
                            catalog_path = os.path.join(parent_dir, catalog_files[0])
                            if self.load_catalog_annotations(os.path.dirname(catalog_path)):
                                annotations_loaded = True
                                loaded_in_dir = len(self.annotations) - annotations_before
                                annotations_by_dir[parent_dir] = loaded_in_dir
                                loaded_count += loaded_in_dir
                                print(f"親ディレクトリ {parent_dir} から {loaded_in_dir} 個のアノテーションを読み込みました")
                    except Exception as e:
                        print(f"カタログファイル検索エラー {parent_dir}: {e}")
                            
            progress.setValue(len(self.folder_paths))
            progress.close()
            
            if annotations_loaded:
                # Update UI
                self.update_stats()
                self.display_current_image()
                self.update_gallery()
                self.update_distribution_graph()  # 追加：分布グラフを更新
                
                # 位置ボタンのカウント表示を更新
                self.update_location_button_counts()
                
                # 詳細情報を生成
                details = ""
                if len(annotations_by_dir) > 0:
                    details = "\n\n詳細:\n"
                    for dir_path, count in annotations_by_dir.items():
                        if count > 0:
                            dir_name = os.path.basename(dir_path)
                            details += f"• {dir_name}: {count}個\n"
                
                QMessageBox.information(
                    self, 
                    "読み込み成功", 
                    f"{len(self.folder_paths)}個のフォルダから合計{loaded_count}個のアノテーションを読み込みました。{details}"
                )
            else:
                QMessageBox.warning(
                    self, 
                    "警告", 
                    "選択したフォルダからアノテーションデータが見つかりませんでした。"
                )
                return
                
            print(f"複数フォルダからのアノテーション読み込み完了: {loaded_count}個のアノテーション")
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            traceback.print_exc()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"アノテーションの読み込み中にエラーが発生しました: {str(e)}"
            )

    def clear_annotations(self):
        """既存のアノテーションデータをクリアする"""
        self.annotations = {}
        self.annotation_history = []
        self.annotated_count = 0
        self.annotation_timestamps = {}
        self.inference_results = {}
        self.location_annotations = {}
        
        if hasattr(self, 'deleted_indexes'):
            self.deleted_indexes = []
        
        # UI更新
        self.update_stats()
        self.display_current_image()
        self.update_gallery()
        
        # 位置ボタンのカウント表示を更新
        self.update_location_button_counts()
        
        # 分布グラフを更新
        if hasattr(self, 'distribution_label'):
            self.distribution_label.clear()
            self.distribution_label.setText("アノテーションがありません")

        print("アノテーションデータをクリアしました")

    def load_subfolder_annotations(self):
        """現在のフォルダの下の階層からアノテーションデータを読み込む"""
        if not self.folder_path or not self.images:
            QMessageBox.warning(self, "警告", "先に画像フォルダを選択して画像を読み込んでください。")
            return
        
        # 現在のフォルダ内のサブフォルダを探す
        sub_dirs = []
        for item in os.listdir(self.folder_path):
            full_path = os.path.join(self.folder_path, item)
            if os.path.isdir(full_path):
                sub_dirs.append(full_path)
        
        if not sub_dirs:
            QMessageBox.warning(self, "警告", "現在のフォルダ内にサブフォルダが見つかりません。")
            return
        
        # ユーザーに選択させるダイアログを表示
        selected_dir, ok = QInputDialog.getItem(
            self, 
            "サブフォルダの選択", 
            "アノテーションを読み込むサブフォルダを選択してください:",
            [os.path.basename(dir_path) for dir_path in sub_dirs], 
            0, 
            False
        )
        
        if not ok or not selected_dir:
            return
        
        # 選択されたフォルダのフルパスを取得
        selected_path = os.path.join(self.folder_path, selected_dir)
        
        # アノテーションデータの検索と読み込みを実行
        annotations_loaded = False
        
        try:
            # 読み込み前に既存のデータをクリア（安全のため）
            if self.annotations:
                self.clear_annotations()
                
            # 最初に選択されたフォルダ自体がDonkeycar形式かどうか確認する
            # マニフェストファイルを確認
            manifest_path = os.path.join(selected_path, "manifest.json")
            if os.path.exists(manifest_path):
                # マニフェストベースの読み込み（複数カタログ対応）
                if self.load_catalog_annotations(selected_path):
                    annotations_loaded = True
                    QMessageBox.information(
                        self, 
                        "読み込み成功", 
                        f"サブフォルダ「{selected_dir}」から{len(self.annotations)}個のアノテーションを読み込みました。"
                    )
            else:
                # 単一カタログファイルの確認
                catalog_files = [f for f in os.listdir(selected_path) if f.endswith('.catalog')]
                if catalog_files:
                    catalog_path = os.path.join(selected_path, catalog_files[0])
                    if self.load_catalog_annotations(os.path.dirname(catalog_path)):
                        annotations_loaded = True
                        QMessageBox.information(
                            self, 
                            "読み込み成功", 
                            f"サブフォルダ「{selected_dir}」から{len(self.annotations)}個のアノテーションを読み込みました。"
                        )
            
            # 選択されたフォルダ内のdata_donkeyフォルダも確認する
            if not annotations_loaded:
                donkey_folder = os.path.join(selected_path, DATA_DONKEY_DIR_NAME)
                if os.path.exists(donkey_folder):
                    # マニフェストファイルを確認
                    manifest_path = os.path.join(donkey_folder, "manifest.json")
                    if os.path.exists(manifest_path):
                        # マニフェストベースの読み込み（複数カタログ対応）
                        if self.load_catalog_annotations(donkey_folder):
                            annotations_loaded = True
                            QMessageBox.information(
                                self, 
                                "読み込み成功", 
                                f"サブフォルダ「{selected_dir}/data_donkey」から{len(self.annotations)}個のアノテーションを読み込みました。"
                            )
                    else:
                        # 従来の単一カタログファイルの確認
                        catalog_files = [f for f in os.listdir(donkey_folder) if f.endswith('.catalog')]
                        if catalog_files:
                            catalog_path = os.path.join(donkey_folder, catalog_files[0])
                            if self.load_catalog_annotations(os.path.dirname(catalog_path)):
                                annotations_loaded = True
                                QMessageBox.information(
                                    self, 
                                    "読み込み成功", 
                                    f"サブフォルダ「{selected_dir}/data_donkey」から{len(self.annotations)}個のアノテーションを読み込みました。"
                                )
            
            # 選択されたフォルダ内のannotationフォルダも確認する
            if not annotations_loaded:
                annotation_folder = os.path.join(selected_path, "annotation")
                if os.path.exists(annotation_folder):
                    # Donkeycar形式のデータを確認
                    donkey_folder = os.path.join(annotation_folder, DATA_DONKEY_DIR_NAME)
                    if os.path.exists(donkey_folder):
                        # マニフェストファイルを確認
                        manifest_path = os.path.join(donkey_folder, "manifest.json")
                        if os.path.exists(manifest_path):
                            # マニフェストベースの読み込み（複数カタログ対応）
                            if self.load_catalog_annotations(donkey_folder):
                                annotations_loaded = True
                                QMessageBox.information(
                                    self, 
                                    "読み込み成功", 
                                    f"サブフォルダ「{selected_dir}/annotation/data_donkey」から{len(self.annotations)}個のアノテーションを読み込みました。"
                                )
                        else:
                            # 従来の単一カタログファイルの確認
                            catalog_files = [f for f in os.listdir(donkey_folder) if f.endswith('.catalog')]
                            if catalog_files:
                                catalog_path = os.path.join(donkey_folder, catalog_files[0])
                                if self.load_catalog_annotations(os.path.dirname(catalog_path)):
                                    annotations_loaded = True
                                    QMessageBox.information(
                                        self, 
                                        "読み込み成功", 
                                        f"サブフォルダ「{selected_dir}/annotation/data_donkey」から{len(self.annotations)}個のアノテーションを読み込みました。"
                                    )
            
            if not annotations_loaded:
                QMessageBox.warning(
                    self, 
                    "警告", 
                    f"選択されたサブフォルダ「{selected_dir}」から読み込めるアノテーションデータがありませんでした。"
                )
                return
            
            # Update UI
            self.update_stats()
            self.display_current_image()
            self.update_gallery()
            
            # 位置ボタンのカウント表示を更新
            self.update_location_button_counts()
            
            print(f"サブフォルダアノテーション読み込み完了: {len(self.annotations)}個のアノテーション")
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "エラー", 
                f"サブフォルダアノテーションの読み込み中にエラーが発生しました: {str(e)}"
            )

    def load_selected_model(self):
        """選択されたモデルを明示的に読み込む - 詳細な進捗メッセージ付き"""
        if not self.images:
            QMessageBox.warning(self, "警告", "画像が読み込まれていません。")
            return
        
        # モデル情報を取得
        model_type = self.auto_method_combo.currentText()
        selected_model = self.model_combo.currentText()
        
        if selected_model == "モデルが見つかりません" or selected_model == "フォルダを選択してください" or "が見つかりません" in selected_model:
            QMessageBox.warning(self, "警告", "有効なモデルが選択されていません。")
            return
        
        # モデルのパスを取得
        models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
        model_path = os.path.join(models_dir, selected_model)
        
        # モデルが存在するか確認
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "警告", f"選択されたモデルが見つかりません: {selected_model}")
            return
        
        # 進捗ダイアログを表示
        progress = QProgressDialog(
            f"モデル '{model_type} ({selected_model})' を読み込み中...", 
            "キャンセル", 0, 100, self
        )
        progress.setWindowTitle("モデル読み込み")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)  # すぐに表示
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        # 既存の推論結果がある場合は確認ダイアログを表示
        clear_inference = False
        if self.inference_results:
            progress.setLabelText(f"既存の推論結果: {len(self.inference_results)}個\n確認ダイアログを表示します...")
            progress.setValue(5)
            QApplication.processEvents()
            
            # 進捗ダイアログを一時的に非表示
            progress.hide()
            
            reply = QMessageBox.question(
                self, 
                "推論結果のクリア確認", 
                f"現在、{len(self.inference_results)}個の推論結果が保存されています。\n"
                f"モデルを変更すると古い推論結果が新しいモデルと不整合を起こす可能性があります。\n\n"
                f"既存の推論結果をクリアしますか？",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Cancel:
                progress.cancel()
                return  # 操作をキャンセル
            
            clear_inference = (reply == QMessageBox.Yes)
            
            # 進捗ダイアログを再表示
            progress.show()
        
        try:
            # 推論結果をクリアする場合
            if clear_inference:
                progress.setLabelText("既存の推論結果をクリア中...")
                progress.setValue(10)
                QApplication.processEvents()
                
                old_count = len(self.inference_results)
                self.inference_results = {}
                self.statusBar().showMessage(f"{old_count}個の古い推論結果をクリアしました", 2000)
            
            # モデルの初期化
            progress.setLabelText("モデルアーキテクチャの初期化中...")
            progress.setValue(20)
            QApplication.processEvents()
            
            # PyTorchモデルの読み込み
            progress.setLabelText(f"モデルファイルを読み込み中: {os.path.basename(model_path)}")
            progress.setValue(40)
            QApplication.processEvents()
            
            # モデルをメモリに読み込む
            progress.setLabelText("モデルを初期化中...")
            progress.setValue(50)
            QApplication.processEvents()
            
            # GPU/CPUへの転送
            device_type = "GPU" if torch.cuda.is_available() else "CPU"
            progress.setLabelText(f"モデルを{device_type}に転送中...")
            progress.setValue(60)
            QApplication.processEvents()
            
            # 現在の画像に対する推論を実行
            current_img_path = self.images[self.current_index]
            progress.setLabelText(f"推論実行中: {os.path.basename(current_img_path)}")
            progress.setValue(70)
            QApplication.processEvents()
            
            # モデルを強制的に再読み込み（現在表示中の画像だけ推論）
            inference_results = batch_inference(
                [current_img_path],
                method="model", 
                model_type=model_type,
                model_path=model_path,
                force_reload=True  # 強制再読み込み
            )
            
            progress.setLabelText("推論結果を保存中...")
            progress.setValue(80)
            QApplication.processEvents()
            
            # 推論結果を保存
            self.inference_results.update(inference_results)
            
            # モデル変更を検出するための状態を保持
            self._last_model_info = (model_type, model_path)
            
            # 推論表示チェックボックスを自動的にオンにする
            progress.setLabelText("推論表示を更新中...")
            progress.setValue(90)
            QApplication.processEvents()
            
            self.inference_checkbox.setChecked(True)
            
            # 推論表示を更新
            self.update_inference_display()
            self.main_image_view.update()
            self.update_gallery()  # ギャラリー表示も更新
            
            progress.setValue(100)
            QApplication.processEvents()
            
            # 成功メッセージ
            message_suffix = ""
            if clear_inference:
                message_suffix = " (古い推論結果はクリアされました)"
            self.statusBar().showMessage(f"モデル '{model_type} ({selected_model})' を読み込みました{message_suffix}", 3000)
            
            # 確認ダイアログ
            confirm_message = f"モデル '{model_type} ({selected_model})' を読み込みました。"
            if clear_inference:
                confirm_message += f"\n\n{len(self.inference_results)}個の新しい推論結果が利用可能です。"
            else:
                confirm_message += f"\n\n既存の推論結果は保持されています。必要に応じて「一括推論実行」ボタンで更新してください。"
            
            confirm_message += "\n\n推論結果表示が自動的にオンになりました。"
            
            # 進捗ダイアログを閉じる
            progress.close()
            
            QMessageBox.information(
                self, 
                "モデル読み込み完了", 
                confirm_message
            )
            
        except Exception as e:
            # エラー発生時も進捗ダイアログを閉じる
            progress.close()
            
            self.statusBar().clearMessage()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"モデル読み込み中にエラーが発生しました: {str(e)}"
            )
  
    def load_catalog_annotations(self, catalog_folder):
        """カタログファイルからアノテーションを読み込む - 進捗表示付き"""
        if not os.path.exists(catalog_folder):
            return False
        
        # 進捗ダイアログを表示
        progress = QProgressDialog(
            f"アノテーションデータ読み込み準備中...", 
            "キャンセル", 0, 100, self
        )
        progress.setWindowTitle("アノテーション読み込み")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)  # すぐに表示
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        # 問題を診断するためのデバッグ情報
        print(f"カタログフォルダを読み込み中: {catalog_folder}")
        
        try:
            # manifest.jsonの確認
            progress.setLabelText("マニフェストファイルを確認中...")
            progress.setValue(5)
            QApplication.processEvents()
            
            manifest_path = os.path.join(catalog_folder, "manifest.json")
            if not os.path.exists(manifest_path):
                print(f"manifest.jsonが見つかりません: {manifest_path}")
                progress.close()
                return False
            
            # manifest.jsonからカタログファイルのリストを取得
            progress.setLabelText("マニフェストファイルを解析中...")
            progress.setValue(10)
            QApplication.processEvents()
            
            catalog_files = []
            deleted_indexes = []  # 削除されたインデックスを保存するリスト
            
            with open(manifest_path, 'r') as mf:
                manifest_lines = mf.readlines()
                if len(manifest_lines) >= 5:  # マニフェストには少なくとも5行必要
                    # 5行目にカタログファイル情報がある
                    catalog_info = json.loads(manifest_lines[4])
                    if "paths" in catalog_info:
                        catalog_files = catalog_info["paths"]
                    
                    # deleted_indexesも取得
                    if "deleted_indexes" in catalog_info:
                        deleted_indexes = catalog_info["deleted_indexes"]
                        print(f"manifest.jsonから{len(deleted_indexes)}個の削除済みインデックスを読み込みました")
                        
                        # クラスの削除済みインデックスリストを更新
                        if hasattr(self, 'deleted_indexes'):
                            self.deleted_indexes = deleted_indexes.copy()
            
            if not catalog_files:
                print("manifest.jsonからカタログファイルを取得できませんでした")
                progress.close()
                return False
            
            progress.setLabelText(f"{len(catalog_files)}個のカタログファイルを検出しました")
            progress.setValue(15)
            QApplication.processEvents()
            
            # 画像フォルダの特定（通常はcatalogと同じフォルダか、その下のimagesフォルダ）
            progress.setLabelText("画像フォルダを検索中...")
            progress.setValue(20)
            QApplication.processEvents()
            
            images_folder = os.path.join(catalog_folder, "images")
            if not os.path.exists(images_folder):
                images_folder = catalog_folder  # imagesフォルダがなければカタログと同じフォルダを使用
            
            print(f"画像フォルダ: {images_folder}")
            
            # 画像のインデックスとファイル名のマッピングを作成
            progress.setLabelText("画像ファイルのインデックスを解析中...")
            progress.setValue(25)
            QApplication.processEvents()
            
            image_index_map = {}
            for img_path in self.images:
                basename = os.path.basename(img_path)
                # ファイル名からインデックスを抽出 (10900_cam_image_array_.jpg から 10900 を取得)
                try:
                    # 数字部分を抽出するための正規表現
                    match = re.match(r'^(\d+)_', basename)
                    if match:
                        index = int(match.group(1))
                        image_index_map[basename] = index
                except Exception as e:
                    print(f"ファイル名からインデックスを抽出できません: {basename} - {e}")
            
            # 全カタログファイルを処理
            loaded_count = 0
            total_entries = 0
            progress_step = 50 / len(catalog_files)  # カタログファイル処理に50%の進捗割り当て
            
            for i, catalog_file in enumerate(catalog_files):
                if progress.wasCanceled():
                    progress.close()
                    return False
                    
                progress.setLabelText(f"カタログファイル処理中: {catalog_file} ({i+1}/{len(catalog_files)})")
                progress.setValue(30 + int(i * progress_step))
                QApplication.processEvents()
                
                catalog_path = os.path.join(catalog_folder, catalog_file)
                if not os.path.exists(catalog_path):
                    print(f"カタログファイルが見つかりません: {catalog_path}")
                    continue
                
                print(f"カタログファイルを処理中: {catalog_file}")
                
                # カタログファイルの行数をカウント
                with open(catalog_path, 'r') as f:
                    lines = f.readlines()
                    total_entries += len(lines)
                
                # 処理開始
                entry_count = 0
                with open(catalog_path, 'r') as f:
                    for line in f:
                        if progress.wasCanceled():
                            progress.close()
                            return False
                        
                        entry_count += 1
                        if entry_count % 100 == 0 or entry_count == total_entries:
                            progress.setLabelText(f"カタログエントリ処理中: {entry_count}/{total_entries} エントリ")
                            sub_progress = 30 + int(i * progress_step) + int((entry_count / total_entries) * progress_step)
                            progress.setValue(min(80, sub_progress))
                            QApplication.processEvents()
                        
                        entry = json.loads(line)
                        
                        # エントリのインデックスを取得
                        entry_index = entry.get('_index', None)
                        
                        # 削除されたインデックスの場合はスキップ
                        if entry_index in deleted_indexes:
                            continue
                        
                        # 画像ファイル名を取得
                        img_name = entry.get('cam/image_array', '')
                        if not img_name:
                            continue
                        
                        # 画像パスの処理 - 複数のパターンを試す
                        img_path = None
                        
                        # 様々なパターンで画像を検索
                        path_patterns = [
                            os.path.join(images_folder, img_name),
                            os.path.join(catalog_folder, img_name),
                            os.path.join(os.path.dirname(catalog_path), img_name),
                            os.path.join(self.folder_path, img_name)
                        ]
                        
                        for path in path_patterns:
                            if os.path.exists(path) and path in self.images:
                                img_path = path
                                break
                        
                        # 画像が見つからない場合、ファイル名のみで探す
                        if img_path is None:
                            basename = os.path.basename(img_name)
                            for path in self.images:
                                if os.path.basename(path) == basename:
                                    img_path = path
                                    break
                        
                        # 画像が見つからない場合はスキップ
                        if img_path is None:
                            continue
                        
                        try:
                            # 画像寸法の取得とアノテーション座標の計算
                            img = Image.open(img_path)
                            img_width, img_height = img.size
                            
                            # ユーザーのアノテーション（または自動アノテーション）を取得
                            angle = entry.get('user/angle', entry.get('pilot/angle', 0))
                            throttle = entry.get('user/throttle', entry.get('pilot/throttle', 0))

                            # 位置情報を取得
                            location = entry.get('user/loc', entry.get('pilot/loc', None))
                                
                            # 座標に変換
                            x = int((angle + 1) / 2 * img_width)
                            y = int((1 - throttle) / 2 * img_height)
                            
                            # 範囲内に収める
                            x = max(0, min(x, img_width - 1))
                            y = max(0, min(y, img_height - 1))
                            
                            # アノテーションを保存
                            self.annotations[entry_index] = {
                                "angle": angle,
                                "throttle": throttle,
                                "x": x,
                                "y": y,
                                "original_index": entry_index  # 元のインデックスを保存
                            }

                            # 位置情報があれば追加
                            if location is not None:
                                self.annotations[entry_index]["loc"] = location
                                self.location_annotations[entry_index] = location
                                
                                # 位置情報ボタンがまだなければ追加
                                self.ensure_location_button_exists(location)

                            # タイムスタンプを保存
                            self.annotation_timestamps[entry_index] = entry.get('_timestamp_ms', int(time.time() * 1000))
                            
                            loaded_count += 1
                            
                            # 推論結果があれば保存（ユーザーアノテーションと異なる場合）
                            if "pilot/angle" in entry and "pilot/throttle" in entry and \
                            (entry.get("user/angle") != entry.get("pilot/angle") or 
                                entry.get("user/throttle") != entry.get("pilot/throttle")):
                                
                                pilot_angle = entry.get("pilot/angle", 0)
                                pilot_throttle = entry.get("pilot/throttle", 0)
                                
                                # 推論座標を計算
                                pilot_x = int((pilot_angle + 1) / 2 * img_width)
                                pilot_y = int((1 - pilot_throttle) / 2 * img_height)
                                
                                # 範囲内に収める
                                pilot_x = max(0, min(pilot_x, img_width - 1))
                                pilot_y = max(0, min(pilot_y, img_height - 1))
                                
                                # 推論結果を保存
                                self.inference_results[entry_index] = {
                                    "angle": pilot_angle,
                                    "throttle": pilot_throttle,
                                    "pilot/angle": pilot_angle,
                                    "pilot/throttle": pilot_throttle,
                                    "x": pilot_x,
                                    "y": pilot_y
                                }
                                
                                # 推論結果に位置情報があれば追加
                                if "pilot/loc" in entry:
                                    self.inference_results[entry_index]["pilot/loc"] = entry["pilot/loc"]
                                    self.inference_results[entry_index]["loc"] = entry["pilot/loc"]
                                    
                        except Exception as e:
                            print(f"画像 {img_path} の処理中にエラー: {e}")
                            continue
            
            # 位置情報の更新処理
            progress.setLabelText("位置情報ボタンを更新中...")
            progress.setValue(85)
            QApplication.processEvents()
            
            # 読み込んだmanifest.jsonのパスを保存
            self.last_manifest_path = manifest_path
            
            # ギャラリー更新
            progress.setLabelText("ギャラリー表示を更新中...")
            progress.setValue(90)
            QApplication.processEvents()
            
            # アノテーション数を更新
            self.annotated_count = len(self.annotations)
            progress.setLabelText(f"{loaded_count}個のアノテーションを読み込みました")
            progress.setValue(95)
            QApplication.processEvents()
            
            # 進捗ダイアログを閉じる
            progress.setValue(100)
            QApplication.processEvents()
            progress.close()

            # 分布グラフを更新
            if self.annotated_count > 0:
                try:
                    self.update_distribution_graph()
                except Exception as graph_error:
                    print(f"分布グラフ更新時にエラー: {graph_error}")
            
            print(f"読み込み完了: {loaded_count}個のアノテーションを読み込みました")
            print(f"削除済みインデックス数: {len(deleted_indexes)}")
            return self.annotated_count > 0
                
        except Exception as e:
            # エラー発生時も進捗ダイアログを閉じる
            if 'progress' in locals():
                progress.close()
                
            print(f"カタログフォルダ {catalog_folder} の読み込み中にエラー: {str(e)}")
            traceback.print_exc()
            return False
    
    def ensure_location_button_exists(self, location_value):
        """指定した位置情報のボタンが存在することを確認し、なければ作成する"""
        # 既存のボタンをチェック
        for button in self.location_buttons:
            if button.property("location_value") == location_value:
                return True
        
        # ボタンが存在しない場合は新規作成
        self.new_location_input.setValue(location_value)
        self.add_location_button()
        return True

    def update_stats(self):
        self.stats_label.setText(f"アノテーション済み: {self.annotated_count} / {len(self.images)}")
    
    #mTODO:削除
    # 置き換え＞enhanced_annotations import apply_enhanced_annotations_display
    def display_current_image(self):
        pass

    def update_gallery(self):
        """ギャラリー表示を更新する - 位置情報の問題を根本的に修正"""
        # Clear current gallery
        for i in reversed(range(self.gallery_layout.count())): 
            self.gallery_layout.itemAt(i).widget().deleteLater()
        
        if not self.images:
            return
        
        # スキップ枚数を取得
        skip_count = self.skip_count_spin.value()
        
        # 現在の位置とインデックス情報
        current_idx = self.current_index
        total_images = len(self.images)
        
        # 前に表示する2枚の画像インデックスを計算
        prev_indices = []
        for i in range(1, 3):
            idx = current_idx - i * skip_count
            if idx >= 0:
                prev_indices.append(idx)
        prev_indices.reverse()  # 近い順に並べる
        
        # 次に表示する2枚の画像インデックスを計算
        next_indices = []
        for i in range(1, 3):
            idx = current_idx + i * skip_count
            if idx < total_images:
                next_indices.append(idx)
        
        # 表示する画像インデックスを組み合わせる
        display_indices = prev_indices + [current_idx] + next_indices
        
        # ギャラリーのグリッドレイアウトを調整
        col_count = 5  # 一行あたりの列数
        
        # ギャラリーにサムネイルを追加
        for i, idx in enumerate(display_indices):
            if 0 <= idx < total_images:
                img_path = self.images[idx]
                
                # 削除されたインデックスの場合、削除済みフラグをセット
                is_deleted = hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes
                
                # アノテーション情報を取得
                annotation = None
                location_value = None  # Initialize here to prevent the error
                
                if not is_deleted and img_path in self.annotations:
                    annotation = self.annotations[img_path]
                
                    # 位置情報を事前に特定
                    if annotation and 'loc' in annotation:
                        location_value = annotation['loc']
                    # あるいは位置情報専用の辞書を確認 (uncomment this)
                    elif img_path in self.location_annotations:
                        location_value = self.location_annotations[img_path]
                
                # サムネイルウィジェットを作成
                thumb = ThumbnailWidget(
                    img_path=img_path,
                    index=idx,
                    is_selected=(idx == current_idx),
                    annotation=annotation,
                    on_click=self.select_image,
                    location_value=location_value,
                    is_deleted=is_deleted
                )
                
                # col_count列のグリッドで配置
                row = i // col_count
                col = i % col_count
                
                self.gallery_layout.addWidget(thumb, row, col)

    def select_image(self, index):
        if 0 <= index < len(self.images):
            # インデックスが変わらない場合は何もしない
            if index == self.current_index:
                return
            
            # 現在の画像に変更
            self.current_index = index
            
            # スライダーの値を更新
            self.image_slider.setValue(index)
            self.slider_value_label.setText(f"{index + 1}/{len(self.images)}")
            
            # 画像表示を更新
            self.display_current_image()
            
            # 推論表示チェックボックスがONの場合、推論結果を表示
            if self.inference_checkbox.isChecked():
                current_img_path = self.images[self.current_index]
                # 推論結果がまだない場合のみ推論を実行
                if current_img_path not in self.inference_results:
                    self.run_inference_check(False)
            
            # ギャラリー更新
            print("3010")
            self.update_gallery()
            
            # スキップ枚数分だけ自動的に次に進める（選択したのが現在の画像より前の場合は戻る）
            if self.skip_images_on_click.isChecked():  # チェックボックスで制御
                skip_count = self.skip_count_spin.value()
                QTimer.singleShot(300, lambda: self.skip_images(skip_count))

    def skip_images(self, count):
        """指定した数だけ画像をスキップする - 位置推論の自動実行も追加"""
        new_index = self.current_index + count
        
        # Ensure the new index is within bounds
        if new_index < 0:
            new_index = 0
        elif new_index >= len(self.images):
            new_index = len(self.images) - 1
        
        # インデックスが変わらない場合は何もしない
        if new_index == self.current_index:
            return
        
        # スキップ前に現在の画像のバウンディングボックス情報を確認し、すべてのボックスを記録
        if hasattr(self, 'bbox_annotations') and len(self.images) > 0:
            current_img_path = self.images[self.current_index]
            if current_img_path in self.bbox_annotations and self.bbox_annotations[current_img_path]:
                # すべてのバウンディングボックスをリストとして保存
                self.last_bboxes = [bbox.copy() for bbox in self.bbox_annotations[current_img_path]]
                print(f"スキップ時にバウンディングボックス情報を更新: {len(self.last_bboxes)}個のボックス")
                
                # 互換性のため、最後のボックスも個別に保存
                if self.last_bboxes:
                    self.last_bbox = self.last_bboxes[-1].copy()

        # 自動位置設定をする前の現在の位置情報を保存
        old_current_location = self.current_location
        
        # 現在のインデックスを更新
        self.current_index = new_index

        # スライダーの値を更新（valueChangedシグナルが発生し、slider_changedが呼ばれる）
        self.image_slider.setValue(new_index)
        self.slider_value_label.setText(f"{new_index + 1}/{len(self.images)}")
        
        # 新しい画像が削除済みかどうかをチェック
        is_deleted = hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes
        
        # 削除されていない場合のみ位置情報の処理
        if not is_deleted:
            # 自動チェックオン
            if self.auto_apply_location:
                # 前の値がある場合は上書き
                if old_current_location is not None:
                    self.current_location = old_current_location
                    self.annotations[self.current_index]['loc'] = self.current_location
                    self.location_annotations[self.current_index] = self.current_location
            else:
                if self.current_index in self.location_annotations:
                    self.current_location = self.location_annotations[self.current_index]
                else:
                    self.current_location = None

        # 画像表示を更新
        self.display_current_image()
        
        # ここから追加: 位置推論表示チェックボックスがONの場合、自動的に推論実行
        if hasattr(self, 'location_inference_checkbox') and self.location_inference_checkbox.isChecked():
            current_img_path = self.images[self.current_index]
            if not is_deleted:
                # 推論結果がまだない場合のみ推論を実行
                if current_img_path not in self.location_inference_results:
                    self.run_location_inference()
                # 表示を更新
                self.update_location_inference_display()
        
        # 推論表示チェックボックスがONの場合、推論結果がなければ実行
        if self.inference_checkbox.isChecked():
            current_img_path = self.images[self.current_index]
            if current_img_path not in self.inference_results:
                self.run_inference_check(False)

        # 物体検知推論表示の更新
        self.update_detection_info_panel()

        # 画面を更新
        self.update_gallery()
        self.update_inference_display()
            
        # 前回のバウンディングボックスの適用
        if hasattr(self, 'auto_apply_last_bbox') and not is_deleted and self.auto_apply_last_bbox:
            # 現在の画像にボックスがない場合に適用
            if current_img_path not in self.bbox_annotations or not self.bbox_annotations[current_img_path]:
                # last_bboxesが存在すればそれを使用、なければlast_bboxを使用
                if hasattr(self, 'last_bboxes') and self.last_bboxes:
                    # すべてのボックスを適用
                    for bbox in self.last_bboxes:
                        self.add_bbox_annotation(bbox.copy())
                    
                    # ステータスバーに表示
                    self.statusBar().showMessage(f"前回の {len(self.last_bboxes)}個のバウンディングボックスを適用しました", 3000)
                
                elif hasattr(self, 'last_bbox') and self.last_bbox is not None:
                    # 後方互換性のため、単一ボックスの場合も処理
                    self.add_bbox_annotation(self.last_bbox.copy())
                    self.statusBar().showMessage(f"前回の '{self.last_bbox['class']}' バウンディングボックスを適用しました", 3000)

    def get_normalized_coordinates(self, click_x, click_y, img_width, img_height):
        """Convert pixel coordinates to normalized coordinates"""
        # Convert x from pixels to -1 (left) to 1 (right)
        angle = (click_x / img_width) * 2 - 1
        
        # Convert y from pixels to 1 (top) to -1 (bottom)
        throttle = -((click_y / img_height) * 2 - 1)
        
        return angle, throttle
    
    def handle_annotation(self, x, y):
        """画像のアノテーションを処理する - 削除済み画像への再アノテーションをサポート"""
        if not self.images:
            return
        
        current_img_path = self.images[self.current_index]
        
        # Get image dimensions
        img = Image.open(current_img_path)
        width, height = img.size
        
        # Get normalized coordinates
        angle, throttle = self.get_normalized_coordinates(x, y, width, height)
        
        # Store current state in history before changing
        if current_img_path in self.annotations:
            previous = self.annotations.copy()
            self.annotation_history.append(previous)
        
        # 削除済みインデックスの場合、削除リストから削除
        if hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes:
            # 現在のインデックスを削除済みリストから除外
            self.deleted_indexes.remove(self.current_index)
            # 確認メッセージ
            QMessageBox.information(
                self, 
                "再アノテーション", 
                f"インデックス {self.current_index} は削除済みでしたが、再アノテーションにより復元されました。"
            )
        
        # Update annotation for this image
        if self.current_index not in self.annotations:
            self.annotated_count += 1
        
        # アノテーション時のタイムスタンプを保存（ミリ秒）
        current_timestamp = int(time.time() * 1000)
        self.annotation_timestamps[self.current_index] = current_timestamp
        
        self.annotations[self.current_index] = {
            "angle": angle,
            "throttle": throttle,
            "x": x,
            "y": y
        }

        # 位置情報があれば追加
        if self.current_location is not None:
            self.annotations[self.current_index]["loc"] = self.current_location
            # 位置情報アノテーションも更新
            self.location_annotations[current_img_path] = self.current_location 

        # 位置ボタンのカウント表示を更新
        self.update_location_button_counts()

        # Update UI
        self.update_stats()
        self.display_current_image()
        print("アノテーション実行")
        self.update_gallery()

        # 分布グラフを更新
        self.update_distribution_graph()

    def restore_deleted_annotation(self):
        """現在表示中の削除済みの画像を復元する（削除状態を解除する）"""
        if not self.images or not hasattr(self, 'deleted_indexes'):
            return
        
        # 現在のインデックスが削除済みかチェック
        if self.current_index not in self.deleted_indexes:
            QMessageBox.information(
                self, 
                "情報", 
                "現在の画像は削除済みではありません。"
            )
            return
        
        # 削除済みリストから削除
        self.deleted_indexes.remove(self.current_index)
        
        # UI更新
        self.display_current_image()
        self.update_gallery()
        
        QMessageBox.information(
            self, 
            "復元完了", 
            f"インデックス {self.current_index} の削除状態を解除しました。\n"
            "この画像にアノテーションを追加できるようになりました。"
        )

    def restore_all_deleted_annotations(self):
        """全ての削除済みアノテーションの状態を復元する"""
        if not self.images or not hasattr(self, 'deleted_indexes') or not self.deleted_indexes:
            QMessageBox.information(
                self, 
                "情報", 
                "復元する削除済みのアノテーションがありません。"
            )
            return
        
        # 削除済みインデックスの数を取得
        count = len(self.deleted_indexes)
        
        # 確認ダイアログ
        reply = QMessageBox.question(
            self, 
            "全ての削除状態を復元", 
            f"削除済みの{count}個のインデックスをすべて復元しますか？",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # 削除済みリストをクリア
        self.deleted_indexes = []
        
        # UI更新
        self.display_current_image()
        self.update_gallery()
        
        QMessageBox.information(
            self, 
            "復元完了", 
            f"{count}個の削除済みインデックスをすべて復元しました。\n"
            "これらのインデックスにアノテーションを追加できるようになりました。"
        )
   #TODO:リファクタリング
    def export_to_donkey(self):
        """Donkeycar形式でエクスポートする - 複数画像ソース選択に対応"""
        if not self.annotations:
            QMessageBox.information(self, "情報", "エクスポートするアノテーションがありません。")
            return
        
        if not self.folder_path:
            QMessageBox.warning(self, "警告", "画像フォルダが設定されていません。")
            return
        
        if not hasattr(self, 'available_variants') or not self.available_variants:
            QMessageBox.warning(self, "警告", "利用可能な画像ソースがありません。")
            return
        
        try:
            # 画像ソース選択ダイアログを表示
            source_dialog = QDialog(self)
            source_dialog.setWindowTitle("エクスポート設定")
            source_dialog.setMinimumWidth(400)
            
            dialog_layout = QVBoxLayout(source_dialog)
            
            # 説明ラベル
            info_label = QLabel("エクスポートする画像ソースを選択してください（複数選択可）：")
            dialog_layout.addWidget(info_label)
            
            # 画像ソース選択グループ
            source_group = QGroupBox("画像ソース")
            source_layout = QVBoxLayout(source_group)
            
            # 利用可能な画像ソースに基づいてチェックボックスを作成（複数選択可）
            source_checks = {}
            for variant in self.available_variants:
                check = QCheckBox(f"{variant} ({len(self.variant_images.get(variant, []))}枚)")
                check.setProperty("variant", variant)
                # 現在のバリアントは自動的にチェック
                if variant == self.current_variant:
                    check.setChecked(True)
                source_layout.addWidget(check)
                source_checks[variant] = check
            
            dialog_layout.addWidget(source_group)
            
            # カタログキー設定
            keys_group = QGroupBox("カタログキー設定")
            keys_layout = QVBoxLayout(keys_group)
            
            # 各ソースタイプのキー名設定
            key_inputs = {}
            for variant in self.available_variants:
                key_layout = QHBoxLayout()
                key_layout.addWidget(QLabel(f"{variant} キー名:"))
                default_key = f"{variant}/image_array"
                key_input = QLineEdit(default_key)
                key_layout.addWidget(key_input)
                keys_layout.addLayout(key_layout)
                key_inputs[variant] = key_input
            
            # 説明ラベル
            key_note = QLabel("※ Donkeycarのデフォルトキーは 'cam/image_array' です。他のキーを使用する場合は対応が必要です。")
            key_note.setStyleSheet("color: #666; font-style: italic;")
            keys_layout.addWidget(key_note)
            
            dialog_layout.addWidget(keys_group)
            
            # 削除したインデックスの情報表示
            if hasattr(self, 'deleted_indexes') and self.deleted_indexes:
                deletion_info = QLabel(f"削除済みインデックス数: {len(self.deleted_indexes)}")
                dialog_layout.addWidget(deletion_info)
            
            # ボタン
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(source_dialog.accept)
            button_box.rejected.connect(source_dialog.reject)
            dialog_layout.addWidget(button_box)
            
            # ダイアログ表示
            if not source_dialog.exec_():
                return
            
            # 選択された画像ソースを取得
            selected_variants = []
            variant_keys = {}  # 各バリアントのキー名
            
            for variant, check in source_checks.items():
                if check.isChecked():
                    selected_variants.append(variant)
                    # 対応するキー名を取得
                    variant_keys[variant] = key_inputs[variant].text().strip()
                    if not variant_keys[variant]:
                        variant_keys[variant] = f"{variant}/image_array"  # デフォルト値
            
            if not selected_variants:
                QMessageBox.warning(self, "警告", "画像ソースが選択されていません。")
                return
            
            # 確認ダイアログ用のメッセージ作成
            confirm_message = "以下の設定でDonkeycar形式でエクスポートします：\n\n"
            
            # 選択されたソースとキー名を表示
            for variant in selected_variants:
                confirm_message += f"・画像ソース: {variant} ({len(self.variant_images.get(variant, []))}枚)\n"
                confirm_message += f"  キー名: {variant_keys[variant]}\n"
            
            confirm_message += f"\nアノテーション数: {len(self.annotations)}個"
            
            if hasattr(self, 'deleted_indexes') and self.deleted_indexes:
                confirm_message += f"\n削除済みインデックス数: {len(self.deleted_indexes)}"
            
            reply = QMessageBox.question(
                self, "エクスポート確認", confirm_message + "\n\n続行しますか？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            
            if reply == QMessageBox.No:
                return
            
            # 選択されたソースの画像マップを作成
            # {index: {variant: image_path, ...}, ...}
            image_map = {}
            
            for variant in selected_variants:
                variant_images = self.variant_images.get(variant, [])
                if not variant_images:
                    continue
                    
                for img_path in variant_images:
                    # 画像パスからインデックスを抽出
                    try:
                        import re
                        basename = os.path.basename(img_path)
                        match = re.match(r'^(\d+)_', basename)
                        if match:
                            idx = int(match.group(1))
                            if idx not in image_map:
                                image_map[idx] = {}
                            image_map[idx][variant] = img_path
                    except Exception as e:
                        print(f"インデックス抽出エラー ({img_path}): {e}")
            
            if not image_map:
                QMessageBox.warning(self, "警告", "画像インデックスの抽出に失敗しました。")
                return
            
            # Donkeycar形式でエクスポート
            output_folder = donkey_dataset_dir
            
            # エクスポート関数を実行
            try:
                catalog_path = export_to_donkey(
                    output_folder, 
                    self.annotations, 
                    inference_results=self.inference_results,
                    deleted_indexes=self.deleted_indexes if hasattr(self, 'deleted_indexes') else [],
                    image_map=image_map,
                    variant_keys=variant_keys
                )
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "エクスポートエラー", 
                    f"Donkeycarエクスポート中にエラーが発生しました: {str(e)}\n\n"
                    f"詳細: {traceback.format_exc()}"
                )
                return
            
            if not catalog_path:
                QMessageBox.warning(
                    self,
                    "エクスポート警告",
                    "エクスポート可能なエントリがありませんでした。"
                )
                return
                
            QMessageBox.information(
                self, 
                "完了", 
                f"アノテーションをDonkeycar形式でエクスポートしました。\n"
                f"選択画像ソース: {', '.join(selected_variants)}\n"
                f"保存先: {output_folder}\n"
                f"エクスポート数: {len(self.annotations)}個"
            )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "エラー", 
                f"エクスポート設定中にエラーが発生しました: {str(e)}\n\n"
                f"詳細: {traceback.format_exc()}"
            )

    def export_to_jetracer(self):
        """Jetracer形式でエクスポートする"""
        if not self.annotations:
            QMessageBox.information(self, "情報", "エクスポートするアノテーションがありません。")
            return
        
        if not self.folder_path:
            QMessageBox.warning(self, "警告", "画像フォルダが設定されていません。")
            return
        
        try:
            # Jetracer形式でエクスポート
            output_folder = jetracer_dataset_dir
            catalog_path = export_to_jetracer(
                output_folder, 
                self.annotations, 
                inference_results=self.inference_results
            )
            
            QMessageBox.information(
                self, 
                "完了", 
                f"アノテーションをJetracer形式でエクスポートしました。\n"
                f"保存先: {output_folder}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "エラー", 
                f"エクスポート中にエラーが発生しました: {str(e)}"
            )

    # def create_annotation_video(self):
    #     """アノテーション動画を作成する - フレームレートと画像ソースを選択可能"""
    #     if not self.annotations:
    #         QMessageBox.information(self, "情報", "アノテーションがありません。")
    #         return
                            
    #     # タイムスタンプを使用してファイル名を生成
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     default_filename = f"annotation_video_{timestamp}.mp4"
    #     output_file = os.path.join(video_folder, default_filename)
        
    #     # 動画作成設定ダイアログを作成
    #     settings_dialog = QDialog(self)
    #     settings_dialog.setWindowTitle("動画作成設定")
    #     settings_dialog.setMinimumWidth(400)
        
    #     dialog_layout = QVBoxLayout(settings_dialog)
        
    #     # フレームレート設定
    #     fps_layout = QHBoxLayout()
    #     fps_layout.addWidget(QLabel("フレームレート (fps):"))
    #     fps_spin = QSpinBox()
    #     fps_spin.setRange(1, 60)
    #     fps_spin.setValue(30)  # デフォルト: 30fps
    #     fps_layout.addWidget(fps_spin)
    #     dialog_layout.addLayout(fps_layout)
        
    #     # スキップ設定
    #     skip_layout = QHBoxLayout()
    #     skip_layout.addWidget(QLabel("画像スキップ数:"))
    #     skip_spin = QSpinBox()
    #     skip_spin.setRange(1, 100)
    #     skip_spin.setValue(self.skip_count_spin.value())  # UIのスキップ値を初期値に
    #     skip_layout.addWidget(skip_spin)
    #     dialog_layout.addLayout(skip_layout)
        
    #     # 推論結果表示設定
    #     inference_check = QCheckBox("推論結果を表示する（青丸）")
    #     inference_check.setChecked(self.inference_checkbox.isChecked())  # UIの設定を初期値に
    #     dialog_layout.addWidget(inference_check)
        
    #     # 画像ソース選択
    #     if hasattr(self, 'available_variants') and self.available_variants:
    #         source_group = QGroupBox("画像ソース")
    #         source_layout = QVBoxLayout(source_group)
            
    #         # 現在のバリアント
    #         current_variant = self.current_variant if hasattr(self, 'current_variant') else None
            
    #         # 選択ボタングループ
    #         source_buttons = QButtonGroup(settings_dialog)
    #         source_buttons.setExclusive(True)  # 排他的選択
            
    #         # 各ソースのラジオボタンを作成
    #         selected_source = None
    #         for variant in self.available_variants:
    #             count = len(self.variant_images.get(variant, [])) if hasattr(self, 'variant_images') else 0
    #             rb = QRadioButton(f"{variant} ({count}枚)")
    #             rb.setProperty("variant", variant)
    #             source_layout.addWidget(rb)
    #             source_buttons.addButton(rb)
                
    #             # 現在選択中のバリアントを初期選択に
    #             if variant == current_variant:
    #                 rb.setChecked(True)
    #                 selected_source = variant
            
    #         # 何も選択されていない場合は最初の項目を選択
    #         if selected_source is None and source_buttons.buttons():
    #             source_buttons.buttons()[0].setChecked(True)
    #             selected_source = source_buttons.buttons()[0].property("variant")
            
    #         dialog_layout.addWidget(source_group)
    #     else:
    #         # バリアント情報がない場合（バックアップ用）
    #         selected_source = None
        
    #     # 合計フレーム数の表示
    #     total_frames_label = QLabel("合計フレーム数: 計算中...")
    #     dialog_layout.addWidget(total_frames_label)
        
    #     # フレーム数を計算して表示を更新する関数
    #     def update_total_frames():
    #         skip = skip_spin.value()
    #         source = None
            
    #         # 選択された画像ソースを取得
    #         if hasattr(self, 'available_variants') and self.available_variants:
    #             for button in source_buttons.buttons():
    #                 if button.isChecked():
    #                     source = button.property("variant")
    #                     break
            
    #         # 選択されたソースの画像数を取得
    #         if source and hasattr(self, 'variant_images') and source in self.variant_images:
    #             images = self.variant_images[source]
    #             count = len(images)
    #         else:
    #             images = self.images if hasattr(self, 'images') else []
    #             count = len(images)
            
    #         # スキップを適用した合計フレーム数
    #         total_frames = (count + skip - 1) // skip  # 切り上げ除算
            
    #         # 予測時間の計算（30fpsの場合）
    #         fps = fps_spin.value()
    #         if fps > 0:
    #             seconds = total_frames / fps
    #             minutes = int(seconds // 60)
    #             seconds = int(seconds % 60)
    #             time_str = f"{minutes}分{seconds}秒" if minutes > 0 else f"{seconds}秒"
    #             total_frames_label.setText(f"合計フレーム数: {total_frames}フレーム (約{time_str})")
    #         else:
    #             total_frames_label.setText(f"合計フレーム数: {total_frames}フレーム")
        
    #     # 設定変更時にフレーム数更新
    #     skip_spin.valueChanged.connect(update_total_frames)
    #     fps_spin.valueChanged.connect(update_total_frames)
    #     if hasattr(self, 'available_variants') and self.available_variants:
    #         source_buttons.buttonClicked.connect(update_total_frames)
        
    #     # 初期フレーム数計算
    #     update_total_frames()
        
    #     # ボタンボックス
    #     button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    #     button_box.accepted.connect(settings_dialog.accept)
    #     button_box.rejected.connect(settings_dialog.reject)
    #     dialog_layout.addWidget(button_box)
        
    #     # ダイアログを表示
    #     if not settings_dialog.exec_():
    #         return  # キャンセルされた場合
        
    #     # 設定値の取得
    #     fps = fps_spin.value()
    #     skip_count = skip_spin.value()
    #     show_inference = inference_check.isChecked()
        
    #     # 選択された画像ソースを取得
    #     selected_variant = None
    #     selected_images = None
        
    #     if hasattr(self, 'available_variants') and self.available_variants:
    #         for button in source_buttons.buttons():
    #             if button.isChecked():
    #                 selected_variant = button.property("variant")
    #                 break
            
    #         if selected_variant and hasattr(self, 'variant_images') and selected_variant in self.variant_images:
    #             selected_images = self.variant_images[selected_variant]
        
    #     if not selected_images:
    #         selected_images = self.images
        
    #     # 動画保存先を選択（デフォルトパスを設定）
    #     selected_file, _ = QFileDialog.getSaveFileName(
    #         self, "動画の保存先を選択", 
    #         output_file,
    #         "MP4 Files (*.mp4)"
    #     )
        
    #     if not selected_file:
    #         return
        
    #     try:
    #         # 進捗ダイアログを表示
    #         progress = QProgressDialog("動画作成中...", "キャンセル", 0, 100, self)
    #         progress.setWindowTitle("処理中")
    #         progress.setWindowModality(Qt.WindowModal)
    #         progress.show()
            
    #         # プログレスコールバック関数
    #         def update_progress(current, total, message=None):
    #             if message:
    #                 progress.setLabelText(message)
    #             progress.setValue(current)
    #             QApplication.processEvents()
    #             return not progress.wasCanceled()
            
    #         # 動画エクスポート実行
    #         frames_count = export_to_video(
    #             self.annotations, 
    #             self.inference_results, 
    #             selected_file, 
    #             show_inference=show_inference, 
    #             skip_count=skip_count, 
    #             fps=fps,  # フレームレートを指定
    #             progress_callback=update_progress,
    #             images_list=selected_images  # 選択された画像ソース
    #         )
            
    #         progress.close()
            
    #         if frames_count > 0:
    #             # 成功メッセージ - ソース情報も追加
    #             source_info = f" (ソース: {selected_variant})" if selected_variant else ""
    #             QMessageBox.information(
    #                 self, 
    #                 "成功", 
    #                 f"アノテーション動画を作成しました:\n"
    #                 f"ファイル: {os.path.basename(selected_file)}\n"
    #                 f"フレーム数: {frames_count}フレーム{source_info}\n"
    #                 f"設定: {fps}fps, {skip_count}枚ごと"
    #             )
    #         else:
    #             QMessageBox.warning(
    #                 self,
    #                 "警告",
    #                 "動画の作成に失敗しました。処理可能なアノテーションデータがありませんでした。"
    #             )
            
    #     except Exception as e:
    #         progress.close()
    #         import traceback
    #         traceback.print_exc()  # デバッグ用にスタックトレースを出力
    #         QMessageBox.critical(
    #             self, 
    #             "エラー", 
    #             f"動画作成中にエラーが発生しました: {str(e)}"
    #         )
    #m
    def create_annotation_video(self):
        """アノテーション動画を作成する - フレームレートと画像ソースを選択可能、複数ソースを横に並べる機能付き"""
        if not self.annotations:
            QMessageBox.information(self, "情報", "アノテーションがありません。")
            return
                            
        # タイムスタンプを使用してファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"annotation_video_{timestamp}.mp4"
        output_file = os.path.join(video_folder, default_filename)
        
        # 動画作成設定ダイアログを作成
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("動画作成設定")
        settings_dialog.setMinimumWidth(500)
        settings_dialog.setMinimumHeight(500)  # サイズを大きくして表示を見やすく
        
        dialog_layout = QVBoxLayout(settings_dialog)
        
        # フレームレート設定
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("フレームレート (fps):"))
        fps_spin = QSpinBox()
        fps_spin.setRange(1, 60)
        fps_spin.setValue(30)  # デフォルト: 30fps
        fps_layout.addWidget(fps_spin)
        dialog_layout.addLayout(fps_layout)
        
        # スキップ設定
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("画像スキップ数:"))
        skip_spin = QSpinBox()
        skip_spin.setRange(1, 100)
        skip_spin.setValue(self.skip_count_spin.value())  # UIのスキップ値を初期値に
        skip_layout.addWidget(skip_spin)
        dialog_layout.addLayout(skip_layout)
        
        # 推論結果表示設定
        inference_check = QCheckBox("推論結果を表示する（青丸）")
        inference_check.setChecked(self.inference_checkbox.isChecked())  # UIの設定を初期値に
        dialog_layout.addWidget(inference_check)
        
        # 出力モード設定のグループボックス
        output_mode_group = QGroupBox("出力モード")
        output_mode_layout = QVBoxLayout(output_mode_group)
        
        # 出力モードのラジオボタン
        single_source_radio = QRadioButton("単一ソース出力（通常モード）")
        single_source_radio.setChecked(True)  # デフォルトで選択
        output_mode_layout.addWidget(single_source_radio)
        
        multi_source_radio = QRadioButton("複数ソース出力（横に並べる）")
        output_mode_layout.addWidget(multi_source_radio)
        
        # 複数ソース選択時の説明
        multi_source_info = QLabel("※複数ソース選択時は下記から複数の画像ソースを選択してください。\n"
                                "選択した順に左から配置されます。")
        multi_source_info.setStyleSheet("color: #666; font-style: italic;")
        output_mode_layout.addWidget(multi_source_info)
        
        dialog_layout.addWidget(output_mode_group)
        
        # 画像ソース選択
        sources_group = QGroupBox("画像ソース")
        sources_layout = QVBoxLayout(sources_group)
        
        # 使用可能なバリアント一覧
        available_variants = []
        variant_images = {}
        current_variant = None
        
        if hasattr(self, 'available_variants'):
            available_variants = self.available_variants
            current_variant = self.current_variant if hasattr(self, 'current_variant') else None
            
            if hasattr(self, 'variant_images'):
                variant_images = self.variant_images
        
        # バリアントがない場合のバックアップ処理
        if not available_variants:
            available_variants = ["cam"]
            current_variant = "cam"
            variant_images = {"cam": self.images if hasattr(self, 'images') else []}
        
        # 単一ソース用ラジオボタングループ
        single_source_buttons = QButtonGroup(settings_dialog)
        single_source_buttons.setExclusive(True)
        
        # 複数ソース用チェックボックスのリスト
        multi_source_checks = []
        
        # 各ソースのラジオボタン・チェックボックスを作成
        source_widgets_layout = QGridLayout()
        
        for i, variant in enumerate(available_variants):
            count = len(variant_images.get(variant, []))
            
            # 単一ソース用ラジオボタン（左列）
            rb = QRadioButton(variant)
            rb.setProperty("variant", variant)
            single_source_buttons.addButton(rb)
            
            # 現在選択中のバリアントを初期選択に
            if variant == current_variant:
                rb.setChecked(True)
            
            # 複数ソース用チェックボックス（右列）
            cb = QCheckBox(variant)
            cb.setProperty("variant", variant)
            cb.setEnabled(False)  # 初期状態では無効（モード選択に連動）
            multi_source_checks.append(cb)
            
            # 利用可能な画像数表示（中央列）
            count_label = QLabel(f"({count}枚)")
            
            # グリッドに追加（行、列）
            source_widgets_layout.addWidget(rb, i, 0)
            source_widgets_layout.addWidget(count_label, i, 1)
            source_widgets_layout.addWidget(cb, i, 2)
        
        sources_layout.addLayout(source_widgets_layout)
        dialog_layout.addWidget(sources_group)
        
        # モード選択によるUIの有効/無効を切り替える関数
        def toggle_source_selection_mode():
            is_multi_mode = multi_source_radio.isChecked()
            # 単一ソース選択ラジオボタン
            for button in single_source_buttons.buttons():
                button.setEnabled(not is_multi_mode)
            
            # 複数ソース選択チェックボックス
            for check in multi_source_checks:
                check.setEnabled(is_multi_mode)
                
            # 出力モード切替時に選択状態をリセット
            if is_multi_mode:
                # 複数モードに切り替えた場合、現在のラジオボタン選択に基づいてチェックボックスを初期選択
                selected_variant = None
                for button in single_source_buttons.buttons():
                    if button.isChecked():
                        selected_variant = button.property("variant")
                        break
                
                # そのバリアントのチェックボックスを選択
                if selected_variant:
                    for check in multi_source_checks:
                        if check.property("variant") == selected_variant:
                            check.setChecked(True)
                            break
        
        # 出力モードラジオボタンの切り替えイベントを接続
        single_source_radio.toggled.connect(toggle_source_selection_mode)
        multi_source_radio.toggled.connect(toggle_source_selection_mode)
        
        # 初期状態の設定
        toggle_source_selection_mode()
        
        # 合計フレーム数の表示
        total_frames_label = QLabel("合計フレーム数: 計算中...")
        dialog_layout.addWidget(total_frames_label)
        
        # フレーム数を計算して表示を更新する関数
        def update_total_frames():
            skip = skip_spin.value()
            is_multi_mode = multi_source_radio.isChecked()
            
            if is_multi_mode:
                # 複数ソースモードの場合、選択されたすべてのソースで最小の画像数を取得
                selected_sources = []
                for check in multi_source_checks:
                    if check.isChecked():
                        selected_sources.append(check.property("variant"))
                
                if not selected_sources:
                    total_frames_label.setText("合計フレーム数: 画像ソースが選択されていません")
                    return
                    
                # 各ソースの画像数を取得
                source_counts = []
                for source in selected_sources:
                    if source in variant_images:
                        source_counts.append(len(variant_images[source]))
                
                if not source_counts:
                    total_frames_label.setText("合計フレーム数: 有効な画像ソースがありません")
                    return
                    
                # 最小の画像数を使用（すべてのソースで揃えるため）
                min_count = min(source_counts)
                count = min_count
            else:
                # 単一ソースモードの場合、選択されたソースの画像数を取得
                selected_source = None
                for button in single_source_buttons.buttons():
                    if button.isChecked():
                        selected_source = button.property("variant")
                        break
                
                if selected_source and selected_source in variant_images:
                    count = len(variant_images[selected_source])
                else:
                    count = len(self.images) if hasattr(self, 'images') else 0
            
            # スキップを適用した合計フレーム数（端数は切り上げ）
            total_frames = (count + skip - 1) // skip
            
            # 予測時間の計算
            fps = fps_spin.value()
            if fps > 0:
                seconds = total_frames / fps
                minutes = int(seconds // 60)
                seconds = int(seconds % 60)
                time_str = f"{minutes}分{seconds}秒" if minutes > 0 else f"{seconds}秒"
                
                # ソース情報を追加
                if is_multi_mode:
                    selected_sources_str = ", ".join(selected_sources)
                    source_info = f"\n選択ソース: {selected_sources_str} (各{min_count}枚)"
                else:
                    source_info = ""
                    
                total_frames_label.setText(
                    f"合計フレーム数: {total_frames}フレーム (約{time_str}){source_info}"
                )
            else:
                total_frames_label.setText(f"合計フレーム数: {total_frames}フレーム")
        
        # 設定変更時にフレーム数更新
        skip_spin.valueChanged.connect(update_total_frames)
        fps_spin.valueChanged.connect(update_total_frames)
        single_source_radio.toggled.connect(update_total_frames)
        multi_source_radio.toggled.connect(update_total_frames)
        
        for button in single_source_buttons.buttons():
            button.toggled.connect(update_total_frames)
        
        for check in multi_source_checks:
            check.stateChanged.connect(update_total_frames)
        
        # 初期フレーム数計算
        update_total_frames()
        
        # ボタンボックス
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(settings_dialog.accept)
        button_box.rejected.connect(settings_dialog.reject)
        dialog_layout.addWidget(button_box)
        
        # ダイアログを表示
        if not settings_dialog.exec_():
            return  # キャンセルされた場合
        
        # 設定値の取得
        fps = fps_spin.value()
        skip_count = skip_spin.value()
        show_inference = inference_check.isChecked()
        is_multi_mode = multi_source_radio.isChecked()
        
        # 選択された画像ソースを取得
        if is_multi_mode:
            # 複数ソースモード
            selected_variants = []
            for check in multi_source_checks:
                if check.isChecked():
                    selected_variants.append(check.property("variant"))
            
            if not selected_variants:
                QMessageBox.warning(settings_dialog, "警告", "画像ソースが選択されていません。")
                return
                
            # 各ソースの画像リスト
            multi_source_images = []
            for variant in selected_variants:
                if variant in variant_images:
                    multi_source_images.append(variant_images[variant])
                else:
                    QMessageBox.warning(settings_dialog, "警告", f"ソース '{variant}' の画像が見つかりません。")
                    return
            
            # 各ソースで利用可能な画像数の最小値を取得
            min_images_count = min(len(images) for images in multi_source_images)
            
            if min_images_count == 0:
                QMessageBox.warning(settings_dialog, "警告", "選択したソースのいずれかに画像がありません。")
                return
        else:
            # 単一ソースモード
            selected_variant = None
            for button in single_source_buttons.buttons():
                if button.isChecked():
                    selected_variant = button.property("variant")
                    break
            
            if not selected_variant:
                QMessageBox.warning(settings_dialog, "警告", "画像ソースが選択されていません。")
                return
                
            # 選択されたソースの画像を取得
            if selected_variant in variant_images:
                selected_images = variant_images[selected_variant]
            else:
                selected_images = self.images if hasattr(self, 'images') else []
                
            if not selected_images:
                QMessageBox.warning(settings_dialog, "警告", f"選択したソース '{selected_variant}' に画像がありません。")
                return
        
        # 動画保存先を選択（デフォルトパスを設定）
        selected_file, _ = QFileDialog.getSaveFileName(
            self, "動画の保存先を選択", 
            output_file,
            "MP4 Files (*.mp4)"
        )
        
        if not selected_file:
            return
        
        try:
            # 進捗ダイアログを表示
            progress = QProgressDialog("動画作成中...", "キャンセル", 0, 100, self)
            progress.setWindowTitle("処理中")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # プログレスコールバック関数
            def update_progress(current, total, message=None):
                if message:
                    progress.setLabelText(message)
                progress.setValue(current)
                QApplication.processEvents()
                return not progress.wasCanceled()
            
            # 動画エクスポート実行
            if is_multi_mode:
                # 複数ソースモードの場合は特別な処理
                frames_count = export_to_video_multi_source(
                    self.annotations,
                    self.inference_results,
                    selected_file,
                    source_images_lists=multi_source_images,
                    source_names=selected_variants,
                    show_inference=show_inference,
                    skip_count=skip_count,
                    fps=fps,
                    progress_callback=update_progress
                )
            else:
                # 単一ソースモードは従来通り
                frames_count = export_to_video(
                    self.annotations, 
                    self.inference_results, 
                    selected_file, 
                    show_inference=show_inference, 
                    skip_count=skip_count, 
                    fps=fps,
                    progress_callback=update_progress,
                    images_list=selected_images
                )
            
            progress.close()
            
            if frames_count > 0:
                # 成功メッセージ
                if is_multi_mode:
                    source_info = f"複数ソース: {', '.join(selected_variants)}"
                else:
                    source_info = f"ソース: {selected_variant}"
                    
                QMessageBox.information(
                    self, 
                    "成功", 
                    f"アノテーション動画を作成しました:\n"
                    f"ファイル: {os.path.basename(selected_file)}\n"
                    f"フレーム数: {frames_count}フレーム\n"
                    f"{source_info}\n"
                    f"設定: {fps}fps, {skip_count}枚ごと"
                )
            else:
                QMessageBox.warning(
                    self,
                    "警告",
                    "動画の作成に失敗しました。処理可能なアノテーションデータがありませんでした。"
                )
            
        except Exception as e:
            progress.close()
            import traceback
            traceback.print_exc()  # デバッグ用にスタックトレースを出力
            QMessageBox.critical(
                self, 
                "エラー", 
                f"動画作成中にエラーが発生しました: {str(e)}"
            )
    #m
    def create_video_progress_callback(self, progress_dialog):
        """動画作成用の進捗コールバック関数を返す
        
        Args:
            progress_dialog: 進捗表示用のQProgressDialogインスタンス
            
        Returns:
            進捗更新用のコールバック関数
        """
        def update_progress(current, total, message=None):
            if message:
                progress_dialog.setLabelText(message)
            progress_dialog.setValue(current)
            QApplication.processEvents()
            return not progress_dialog.wasCanceled()
        
        return update_progress


    def train_and_save_model(self):
        if not self.annotations:
            QMessageBox.warning(self, "警告", "モデルを学習するにはアノテーションが必要です。")
            return
        
        # 現在選択されているモデルを取得
        model_type = self.auto_method_combo.currentText()
        
        # 学習設定ダイアログを表示
        training_settings = QDialog(self)
        training_settings.setWindowTitle("学習設定")
        training_settings.setMinimumWidth(550)
        training_settings.setMinimumHeight(600)  # ダイアログサイズを大きくする
        
        settings_layout = QVBoxLayout(training_settings)
        
        # タブウィジェットを作成
        tabs = QTabWidget()
        
        # 基本設定タブ
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # エポック数設定
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("学習エポック数:"))
        epoch_spin = QSpinBox()
        epoch_spin.setRange(1, 1000)
        epoch_spin.setValue(30)  # デフォルト: 30エポック
        epoch_layout.addWidget(epoch_spin)
        basic_layout.addLayout(epoch_layout)
        
        # スキップ設定
        skip_layout = QVBoxLayout()
        skip_group = QGroupBox("データサンプリング設定")
        skip_inner_layout = QVBoxLayout()
        
        # スキップオプション選択
        skip_radio_all = QRadioButton("すべてのアノテーションデータを使用")
        skip_radio_all.setChecked(True)  # デフォルトですべて使用
        skip_inner_layout.addWidget(skip_radio_all)
        
        skip_radio_ui = QRadioButton(f"UIと同じスキップ設定を使用 (現在: {self.skip_count_spin.value()}枚ごと)")
        skip_inner_layout.addWidget(skip_radio_ui)
        
        skip_radio_custom = QRadioButton("カスタムスキップ設定を使用")
        skip_inner_layout.addWidget(skip_radio_custom)
        
        # カスタムスキップ設定
        custom_skip_layout = QHBoxLayout()
        custom_skip_layout.addWidget(QLabel("       "))  # インデント用スペース
        custom_skip_layout.addWidget(QLabel("スキップ枚数:"))
        custom_skip_spin = QSpinBox()
        custom_skip_spin.setRange(2, 100)
        custom_skip_spin.setValue(5)  # デフォルト: 5枚
        custom_skip_spin.setEnabled(False)  # 初期状態では無効
        custom_skip_layout.addWidget(custom_skip_spin)
        skip_inner_layout.addLayout(custom_skip_layout)
        
        # カスタムスキップのラジオボタン連動
        def on_skip_radio_toggled():
            custom_skip_spin.setEnabled(skip_radio_custom.isChecked())
        
        skip_radio_custom.toggled.connect(on_skip_radio_toggled)
        
        skip_group.setLayout(skip_inner_layout)
        skip_layout.addWidget(skip_group)
        basic_layout.addLayout(skip_layout)
        
        # Early Stopping設定
        early_stopping_check = QCheckBox("Early Stopping を有効にする")
        early_stopping_check.setChecked(True)
        basic_layout.addWidget(early_stopping_check)
        
        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("忍耐エポック数:"))
        patience_spin = QSpinBox()
        patience_spin.setRange(1, 20)
        patience_spin.setValue(5)
        patience_spin.setEnabled(True)
        patience_layout.addWidget(patience_spin)
        basic_layout.addLayout(patience_layout)
        
        # 学習率設定
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学習率:"))
        
        lr_combo = QComboBox()
        learning_rates = ["0.001", "0.0005", "0.0001", "0.00005", "0.00001"]
        lr_combo.addItems(learning_rates)
        lr_combo.setCurrentIndex(0)  # デフォルト: 0.001
        lr_layout.addWidget(lr_combo)
        basic_layout.addLayout(lr_layout)
        
        # タブに追加
        tabs.addTab(basic_tab, "基本設定")
        
        # データオーグメンテーションタブ
        aug_tab = QWidget()
        aug_layout = QVBoxLayout(aug_tab)
        
        # データオーグメンテーション有効化チェックボックス
        aug_enable_check = QCheckBox("データオーグメンテーションを有効にする")
        aug_enable_check.setChecked(True)
        aug_layout.addWidget(aug_enable_check)
        
        # オーグメンテーション設定のスクロールエリア
        aug_scroll = QScrollArea()
        aug_scroll.setWidgetResizable(True)
        aug_scroll.setFrameShape(QFrame.NoFrame)
        
        aug_scroll_content = QWidget()
        aug_options_layout = QVBoxLayout(aug_scroll_content)
        
        # 水平反転
        flip_layout = QHBoxLayout()
        aug_flip_checkbox = QCheckBox("水平反転")
        aug_flip_checkbox.setChecked(False)
        aug_flip_proba_label = QLabel("確率:")
        aug_flip_proba = QDoubleSpinBox()
        aug_flip_proba.setRange(0.0, 1.0)
        aug_flip_proba.setSingleStep(0.1)
        aug_flip_proba.setValue(0.5)
        flip_layout.addWidget(aug_flip_checkbox)
        flip_layout.addWidget(aug_flip_proba_label)
        flip_layout.addWidget(aug_flip_proba)
        flip_layout.addStretch()
        aug_options_layout.addLayout(flip_layout)
        
        # 色調整
        color_layout = QHBoxLayout()
        aug_color_checkbox = QCheckBox("色調整")
        aug_color_checkbox.setChecked(True)
        color_layout.addWidget(aug_color_checkbox)
        color_layout.addStretch()
        aug_options_layout.addLayout(color_layout)
        
        # 色調整の詳細設定
        color_details_layout = QGridLayout()
        color_details_layout.setContentsMargins(20, 0, 0, 0)
        
        # 明るさ
        color_details_layout.addWidget(QLabel("明るさ:"), 0, 0)
        aug_brightness = QDoubleSpinBox()
        aug_brightness.setRange(0.0, 1.0)
        aug_brightness.setSingleStep(0.05)
        aug_brightness.setValue(0.5)
        color_details_layout.addWidget(aug_brightness, 0, 1)
        
        # コントラスト
        color_details_layout.addWidget(QLabel("コントラスト:"), 1, 0)
        aug_contrast = QDoubleSpinBox()
        aug_contrast.setRange(0.0, 1.0)
        aug_contrast.setSingleStep(0.05)
        aug_contrast.setValue(0.5)
        color_details_layout.addWidget(aug_contrast, 1, 1)
        
        # 彩度
        color_details_layout.addWidget(QLabel("彩度:"), 2, 0)
        aug_saturation = QDoubleSpinBox()
        aug_saturation.setRange(0.0, 1.0)
        aug_saturation.setSingleStep(0.05)
        aug_saturation.setValue(0.5)
        color_details_layout.addWidget(aug_saturation, 2, 1)
        
        aug_options_layout.addLayout(color_details_layout)
        
        # 幾何変換
        geometry_layout = QHBoxLayout()
        aug_geometry_checkbox = QCheckBox("幾何変換")
        aug_geometry_checkbox.setChecked(False)
        geometry_layout.addWidget(aug_geometry_checkbox)
        geometry_layout.addStretch()
        aug_options_layout.addLayout(geometry_layout)
        
        # 幾何変換の詳細設定
        geometry_details_layout = QGridLayout()
        geometry_details_layout.setContentsMargins(20, 0, 0, 0)
        
        # 回転角度
        geometry_details_layout.addWidget(QLabel("回転角度 (±度):"), 0, 0)
        aug_rotation = QSpinBox()
        aug_rotation.setRange(0, 90)
        aug_rotation.setValue(5)
        geometry_details_layout.addWidget(aug_rotation, 0, 1)
        
        # 平行移動
        geometry_details_layout.addWidget(QLabel("平行移動 (±比率):"), 1, 0)
        aug_translate = QDoubleSpinBox()
        aug_translate.setRange(0.0, 0.5)
        aug_translate.setSingleStep(0.01)
        aug_translate.setValue(0.1)
        geometry_details_layout.addWidget(aug_translate, 1, 1)
        
        aug_options_layout.addLayout(geometry_details_layout)
        
        # ランダムイレース
        erase_layout = QHBoxLayout()
        aug_erase_checkbox = QCheckBox("ランダムイレース")
        aug_erase_checkbox.setChecked(True)
        aug_erase_proba_label = QLabel("確率:")
        aug_erase_proba = QDoubleSpinBox()
        aug_erase_proba.setRange(0.0, 1.0)
        aug_erase_proba.setSingleStep(0.1)
        aug_erase_proba.setValue(0.2)
        erase_layout.addWidget(aug_erase_checkbox)
        erase_layout.addWidget(aug_erase_proba_label)
        erase_layout.addWidget(aug_erase_proba)
        erase_layout.addStretch()
        aug_options_layout.addLayout(erase_layout)
        
        # イレースの詳細設定
        erase_details_layout = QHBoxLayout()
        erase_details_layout.setContentsMargins(20, 0, 0, 0)
        
        # 最小比率
        erase_details_layout.addWidget(QLabel("最小比率:"))
        aug_erase_min_ratio = QDoubleSpinBox()
        aug_erase_min_ratio.setRange(0.02, 0.4)
        aug_erase_min_ratio.setSingleStep(0.01)
        aug_erase_min_ratio.setValue(0.02)
        erase_details_layout.addWidget(aug_erase_min_ratio)
        
        # スペーサーを追加して間隔を確保
        erase_details_layout.addSpacing(10)
        
        # 最大比率
        erase_details_layout.addWidget(QLabel("最大比率:"))
        aug_erase_max_ratio = QDoubleSpinBox()
        aug_erase_max_ratio.setRange(0.05, 0.5)
        aug_erase_max_ratio.setSingleStep(0.01)
        aug_erase_max_ratio.setValue(0.2)
        erase_details_layout.addWidget(aug_erase_max_ratio)
        
        # レイアウトの右側に伸縮スペースを追加
        erase_details_layout.addStretch()
        
        aug_options_layout.addLayout(erase_details_layout)
        
        # プレビューボタン
        preview_layout = QHBoxLayout()
        preview_button = QPushButton("オーグメンテーションプレビュー")
        preview_button.clicked.connect(lambda: self.show_augmentation_preview_dialog({
            'enabled': aug_enable_check.isChecked(),
            'use_flip': aug_flip_checkbox.isChecked(),
            'flip_prob': aug_flip_proba.value(),
            'use_color': aug_color_checkbox.isChecked(),
            'brightness': aug_brightness.value(),
            'contrast': aug_contrast.value(),
            'saturation': aug_saturation.value(),
            'use_geometry': aug_geometry_checkbox.isChecked(),
            'rotation_degrees': aug_rotation.value(),
            'translate_ratio': aug_translate.value(),
            'use_erase': aug_erase_checkbox.isChecked(),
            'erase_prob': aug_erase_proba.value(),
            'erase_min_ratio': aug_erase_min_ratio.value(),
            'erase_max_ratio': aug_erase_max_ratio.value()
        }))
        preview_layout.addStretch()
        preview_layout.addWidget(preview_button)
        aug_options_layout.addLayout(preview_layout)
        
        # オプションの有効/無効を連動させる
        def toggle_aug_options(checked):
            for w in aug_scroll_content.findChildren(QWidget):
                if w != aug_enable_check:
                    w.setEnabled(checked)
        
        aug_enable_check.toggled.connect(toggle_aug_options)
        
        # スクロールエリアに設定
        aug_scroll.setWidget(aug_scroll_content)
        aug_layout.addWidget(aug_scroll)
        
        # タブに追加
        tabs.addTab(aug_tab, "データオーグメンテーション")
        
        # タブをレイアウトに追加
        settings_layout.addWidget(tabs)
        
        # ボタンの配置
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(training_settings.accept)
        button_box.rejected.connect(training_settings.reject)
        settings_layout.addWidget(button_box)
        
        # ダイアログを表示
        if not training_settings.exec_():
            return
        
        # 設定値の取得
        num_epochs = epoch_spin.value()
        use_early_stopping = early_stopping_check.isChecked()
        patience = patience_spin.value() if use_early_stopping else 0
        learning_rate = float(lr_combo.currentText())
        
        # スキップ設定の取得
        if skip_radio_all.isChecked():
            # すべてのデータを使用（スキップなし）
            use_skip = False
            skip_count = 1
        elif skip_radio_ui.isChecked():
            # UIのスキップ設定を使用
            use_skip = True
            skip_count = self.skip_count_spin.value()
        else:  # skip_radio_custom.isChecked()
            # カスタムスキップ設定を使用
            use_skip = True
            skip_count = custom_skip_spin.value()
        
        # オーグメンテーション設定の取得
        augmentation_params = {
            'enabled': aug_enable_check.isChecked(),
            'use_flip': aug_flip_checkbox.isChecked(),
            'flip_prob': aug_flip_proba.value(),
            'use_color': aug_color_checkbox.isChecked(),
            'brightness': aug_brightness.value(),
            'contrast': aug_contrast.value(),
            'saturation': aug_saturation.value(),
            'use_geometry': aug_geometry_checkbox.isChecked(),
            'rotation_degrees': aug_rotation.value(),
            'translate_ratio': aug_translate.value(),
            'use_erase': aug_erase_checkbox.isChecked(),
            'erase_prob': aug_erase_proba.value(),
            'erase_min_ratio': aug_erase_min_ratio.value(),
            'erase_max_ratio': aug_erase_max_ratio.value()
        }
        
        # モデルトレーニングの続きの確認
        sampling_info = ""
        if use_skip:
            total_annotations = len(self.annotations)
            sampled_count = total_annotations // skip_count + (1 if total_annotations % skip_count > 0 else 0)
            sampling_info = f"データサンプリング: {skip_count}枚ごとに1枚 ({sampled_count}/{total_annotations}枚使用)"
        else:
            sampling_info = f"データサンプリング: すべて使用 ({len(self.annotations)}枚)"

        # オーグメンテーション情報を生成
        aug_info = "データオーグメンテーション: "
        if augmentation_params['enabled']:
            aug_info += "有効\n"
            if augmentation_params['use_flip']:
                aug_info += f"・水平反転 (確率: {augmentation_params['flip_prob']})\n"
            if augmentation_params['use_color']:
                aug_info += f"・色調整 (明るさ: {augmentation_params['brightness']}, "
                aug_info += f"コントラスト: {augmentation_params['contrast']}, "
                aug_info += f"彩度: {augmentation_params['saturation']})\n"
            if augmentation_params['use_geometry']:
                aug_info += f"・幾何変換 (回転: ±{augmentation_params['rotation_degrees']}度, "
                aug_info += f"平行移動: ±{augmentation_params['translate_ratio']})\n"
            if augmentation_params['use_erase']:
                aug_info += f"・ランダムイレース (確率: {augmentation_params['erase_prob']}, "
                aug_info += f"範囲: {augmentation_params['erase_min_ratio']}～{augmentation_params['erase_max_ratio']})\n"
        else:
            aug_info += "無効\n"

        reply = QMessageBox.question(
            self, 
            "モデル学習確認", 
            f"選択されたモデル '{model_type}' を以下の設定で学習しますか？\n\n"
            f"エポック数: {num_epochs}\n"
            f"学習率: {learning_rate}\n"
            f"{sampling_info}\n"
            f"Early Stopping: {'有効（忍耐値: {0}）'.format(patience) if use_early_stopping else '無効'}\n\n"
            f"{aug_info}",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.Yes
        )

        if reply == QMessageBox.No:
            return
        
        # 既存のモデル重みをロードするかの確認
        load_weights = False
        model_path = None
        selected_model = self.model_combo.currentText()
        
        if selected_model and selected_model not in ["モデルが見つかりません", "フォルダを選択してください"] and "が見つかりません" not in selected_model:
            # アノテーションフォルダ内のモデルのフルパスを作成
            annotation_folder = os.path.join(self.folder_path, ANNOTATION_DIR_NAME)
            models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
            model_path = os.path.join(models_dir, selected_model)
            
            # モデルが存在するか確認
            if os.path.exists(model_path):
                weights_reply = QMessageBox.question(
                    self, 
                    "モデル重みの読み込み", 
                    f"現在選択されているモデル '{selected_model}' の重みを使って学習を始めますか？\n\n"
                    f"「はい」: 選択したモデルの重みからファインチューニングします。\n"
                    f"「いいえ」: 新しくモデルをトレーニングします。",
                    QMessageBox.Yes | QMessageBox.No, 
                    QMessageBox.Yes
                )
                
                load_weights = (weights_reply == QMessageBox.Yes)
        
        try:
            # 学習データの準備（スキップ設定を適用）
            image_paths = list(self.annotations.keys())
            if use_skip and skip_count > 1:
                image_paths = image_paths[::skip_count]
            
            if not image_paths:
                QMessageBox.warning(self, "警告", "学習データがありません。")
                return
            
            annotation_values = [self.annotations[img_path] for img_path in image_paths]
            
            # データ数の確認と最小バッチサイズの調整
            batch_size = min(32, len(image_paths))  # バッチサイズを調整
            if batch_size < 2:
                QMessageBox.warning(self, "警告", "データ数が不足しています。最低2枚の画像が必要です。")
                return
                        
            # 進捗ダイアログ
            progress = QProgressDialog(
                f"モデル '{model_type}' の学習中...", 
                "キャンセル", 0, 100, self
            )
            progress.setWindowTitle("モデル学習")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 進捗コールバック
            def update_progress(current, total, message=None):
                value = int(current * 100 / total)
                progress.setValue(value)
                if message:
                    progress.setLabelText(message)
                QApplication.processEvents()
                return not progress.wasCanceled()
                        
            # データセットの作成（バッチサイズと詳細オーグメンテーション設定を明示的に指定）
            train_loader, val_loader, dataset_info = create_datasets(
                image_paths=image_paths,
                annotations=annotation_values,
                model_name=model_type,
                use_augmentation=augmentation_params if augmentation_params['enabled'] else False,
                batch_size=batch_size  # バッチサイズを追加
            )

            # 最初の画像から実際のサイズを取得
            sample_img_path = image_paths[0]
            sample_img = Image.open(sample_img_path)
            input_size = (sample_img.height, sample_img.width)  # 高さ、幅の順

            progress.setLabelText(f"入力サイズ: {input_size} で学習準備中...")
            progress.setValue(10)
            QApplication.processEvents()

            # モデルの学習 - 事前学習済み重みをロードするかどうかのフラグを追加
            training_results = train_model(
                model_name=model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                save_dir=models_dir,
                progress_callback=update_progress,
                pretrained=not load_weights,  # 既存モデルを使う場合はTrueにしない
                model_path=model_path if load_weights else None,  # ロードする場合はパスを指定
                num_epochs=num_epochs,  # 指定されたエポック数
                learning_rate=learning_rate,  # 指定された学習率
                use_early_stopping=use_early_stopping,  # Early Stoppingの有効/無効
                patience=patience  # 忍耐値
            )
            
            progress.close()

            # MLflowに結果を記録
            try:
                # MLflowが初期化されていない場合は初期化
                if not hasattr(self, 'mlflow_tracking_uri'):
                    self.initialize_mlflow()
                
                # データセット情報を定義（先に定義する）
                dataset_info_mlflow = {
                    "total_annotations": len(self.annotations),
                    "used_samples": len(image_paths),
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "input_shape": dataset_info.get('actual_image_size', input_size)  # 実際の画像サイズを使用
                    #"input_shape": dataset_info.get('actual_image_size', (0, 0))
                }
                
                # メトリクス情報を定義（先に定義する）
                metrics = {
                    "best_val_loss": training_results.get('best_val_loss', 0.0),
                    "final_train_loss": training_results['train_losses'][-1] if 'train_losses' in training_results else 0.0,
                    "final_val_loss": training_results['val_losses'][-1] if 'val_losses' in training_results else 0.0,
                    "train_losses": training_results.get('train_losses', []),
                    "val_losses": training_results.get('val_losses', [])
                }
                
                # 学習パラメータを整形
                training_params = {
                    "model_type": model_type,
                    "num_epochs": num_epochs,
                    "completed_epochs": training_results.get('completed_epochs', num_epochs),
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "use_early_stopping": use_early_stopping,
                    "patience": patience if use_early_stopping else 0,
                    "early_stopped": training_results.get('early_stopped', False),
                    "initial_weights": "fine-tuned" if load_weights else "pretrained",
                    "augmentation_enabled": augmentation_params['enabled'],
                    "sampling_strategy": "all" if not use_skip else f"skip_{skip_count}"
                }

                # MLflowにログを記録
                # ベストモデルのパスを正規化
                best_model_path = training_results['best_model_path'].replace("\\", "/")
                
                # MLflowに記録
                with mlflow.start_run(run_name=f"{model_type}_{len(image_paths)}samples"):
                    # パラメータの記録
                    for key, value in training_params.items():
                        mlflow.log_param(key, value)
                    
                    # データセット情報の記録
                    for key, value in dataset_info_mlflow.items():
                        if key != "input_shape":  # タプルは直接記録できないため
                            mlflow.log_param(f"dataset_{key}", value)
                    
                    # 入力形状は文字列に変換
                    if "input_shape" in dataset_info_mlflow:
                        input_shape = dataset_info_mlflow["input_shape"]
                        mlflow.log_param("dataset_image_dims", f"{input_shape[0]}x{input_shape[1]}")
                    
                    # メトリクスの記録
                    mlflow.log_metric("best_val_loss", metrics["best_val_loss"])
                    mlflow.log_metric("final_train_loss", metrics["final_train_loss"])
                    mlflow.log_metric("final_val_loss", metrics["final_val_loss"])
                    
                    # 学習曲線をログ
                    for epoch, (train_loss, val_loss) in enumerate(zip(
                            metrics["train_losses"], metrics["val_losses"])):
                        mlflow.log_metric("train_loss", train_loss, step=epoch)
                        mlflow.log_metric("val_loss", val_loss, step=epoch)
                    
                    # モデルファイルをアーティファクトとして保存
                    mlflow.log_artifact(best_model_path)
                
                mlflow_info = "MLflowに学習履歴を記録しました。\n「MLflow比較」ボタンで結果を確認できます。"
                print("MLflowに記録成功")
                
            except ImportError:
                mlflow_info = "MLflowがインストールされていないため、学習履歴は記録されませんでした。\npip install mlflow でインストールできます。"
                print("MLflowインポートエラー")
                
            except Exception as e:
                print(f"MLflow記録エラー: {e}")
                traceback.print_exc()  # スタックトレースを出力して詳細を確認
                mlflow_info = f"MLflowへの記録中にエラーが発生しました: {str(e)}"
            
            # オーグメンテーション情報を取得
            aug_details = ""
            if augmentation_params['enabled']:
                aug_details = "データオーグメンテーション: 有効\n"
                if augmentation_params['use_flip']:
                    aug_details += f"  - 水平反転 (確率: {augmentation_params['flip_prob']})\n"
                if augmentation_params['use_color']:
                    aug_details += f"  - 色調整 (明るさ: ±{augmentation_params['brightness']}, "
                    aug_details += f"コントラスト: ±{augmentation_params['contrast']}, "
                    aug_details += f"彩度: ±{augmentation_params['saturation']})\n"
                if augmentation_params['use_geometry']:
                    aug_details += f"  - 幾何変換 (回転: ±{augmentation_params['rotation_degrees']}度, "
                    aug_details += f"平行移動: ±{augmentation_params['translate_ratio']})\n"
                if augmentation_params['use_erase']:
                    aug_details += f"  - ランダムイレース (確率: {augmentation_params['erase_prob']}, "
                    aug_details += f"範囲: {augmentation_params['erase_min_ratio']}～{augmentation_params['erase_max_ratio']})\n"
            else:
                aug_details = "データオーグメンテーション: 無効\n"

            # 初期重みの情報を追加
            weights_info = ""
            if load_weights:
                weights_info = f"初期重み: {selected_model} から読み込み\n"
            else:
                weights_info = "初期重み: 事前学習済みモデル\n"
            
            # Early Stopping情報
            early_stopping_info = ""
            if use_early_stopping:
                if training_results.get('early_stopped', False):
                    early_stopping_info = f"Early Stopping: {training_results.get('stopped_epoch', 0)}エポックで停止\n"
                else:
                    early_stopping_info = f"Early Stopping: 発動せず (忍耐値: {patience})\n"
            
            # 入力サイズ情報を追加
            input_size_info = f"入力サイズ: {input_size[0]}x{input_size[1]} (H x W)\n"

            # 成功メッセージを表示
            QMessageBox.information(
                self, 
                "成功", 
                f"{model_type} モデルを学習し保存しました: {os.path.basename(training_results['model_path'])}\n" +
                f"最良検証損失: {training_results['best_val_loss']:.6f}\n" +
                f"実施エポック数: {training_results.get('completed_epochs', num_epochs)}/{num_epochs}\n" +
                early_stopping_info +
                f"学習データ数: {len(image_paths)}枚 {sampling_info}\n" +
                input_size_info + 
                weights_info +
                f"学習率: {learning_rate}\n" +
                f"バッチサイズ: {batch_size}\n" +
                aug_details
            )
            
            # モデルリストを更新
            self.refresh_model_list()
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"モデル学習中にエラーが発生しました: {str(e)}"
            )    

    #TODO:annotationsのpathからindexへ変更
    def auto_annotate(self):
        """オートアノテーションを実行する - 詳細な進捗表示付き"""
        if not self.annotations:
            QMessageBox.warning(self, "警告", "オートアノテーションを実行するには、まず数枚の画像に手動でアノテーションを行ってください。")
            return
        
        # アノテーションされていない画像を取得
        unannotated_images = [img for img in self.images if img not in self.annotations]
        
        if not unannotated_images:
            QMessageBox.information(self, "情報", "すべての画像がすでにアノテーションされています。")
            return
        
        # 選択された学習方法（モデル）を取得
        model_type = self.auto_method_combo.currentText()
        selected_model = self.model_combo.currentText()
        
        # モデルのパスを取得
        model_path = None
        if hasattr(self, 'model_combo') and selected_model not in ["モデルが見つかりません", "フォルダを選択してください"] and "が見つかりません" not in selected_model:
            models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
            model_path = os.path.join(models_dir, selected_model)
            
            # モデルが存在するか確認
            if not os.path.exists(model_path):
                model_path = None
        
        # 進捗ダイアログを表示
        progress = QProgressDialog(
            f"オートアノテーション準備中... ({len(unannotated_images)}枚の画像)", 
            "キャンセル", 0, 100, self
        )
        progress.setWindowTitle("オートアノテーション実行中")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)  # すぐに表示
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        # 処理前の確認
        progress.setLabelText(f"モデル '{model_type}' を使用した処理を準備中...")
        progress.setValue(5)
        QApplication.processEvents()
        
        # バッチサイズ - 大量の画像を一度に処理するとメモリ不足になる可能性があるため
        batch_size = 50
        total_batches = (len(unannotated_images) + batch_size - 1) // batch_size
        
        try:
            # モデル初期化
            progress.setLabelText(f"モデル '{model_type}' を初期化中...")
            progress.setValue(10)
            QApplication.processEvents()
            
            # 既存モデルの読み込み
            device_type = "GPU" if torch.cuda.is_available() else "CPU"
            
            if model_path:
                progress.setLabelText(f"モデル '{os.path.basename(model_path)}' を読み込み中...")
            else:
                progress.setLabelText(f"事前学習済みモデル '{model_type}' を準備中...")
            
            progress.setValue(15)
            QApplication.processEvents()
            
            # バッチ処理での進捗管理
            processed_count = 0
            success_count = 0
            
            # 実行中のモデル情報を保存
            self._last_model_info = (model_type, model_path)
            
            # バッチ処理
            for batch_idx in range(total_batches):
                if progress.wasCanceled():
                    break
                    
                # 現在のバッチの画像取得
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(unannotated_images))
                current_batch = unannotated_images[start_idx:end_idx]
                
                progress.setLabelText(
                    f"バッチ {batch_idx+1}/{total_batches} 処理中...\n"
                    f"画像 {start_idx+1}-{end_idx}/{len(unannotated_images)}"
                )
                
                # 進捗値計算 - バッチ処理に80%の進捗を割り当て (15-95%)
                batch_progress = 15 + int((batch_idx / total_batches) * 80)
                progress.setValue(batch_progress)
                QApplication.processEvents()
                
                # 推論を実行
                try:
                    inference_results = batch_inference(
                        current_batch, 
                        method="model", 
                        model_type=model_type,
                        model_path=model_path,
                        force_reload=(batch_idx == 0)  # 最初のバッチのみ強制再読込
                    )
                    
                    # サブ進捗表示
                    batch_size = len(current_batch)
                    for i, (img_path, result) in enumerate(inference_results.items()):
                        if progress.wasCanceled():
                            break
                            
                        # 10画像ごとに進捗更新
                        if i % 10 == 0 or i == batch_size - 1:
                            sub_progress = batch_progress + int((i / batch_size) * (80 / total_batches))
                            progress.setValue(min(95, sub_progress))
                            progress.setLabelText(
                                f"バッチ {batch_idx+1}/{total_batches} 処理中...\n"
                                f"画像 {start_idx+i+1}/{len(unannotated_images)} を処理中"
                            )
                            QApplication.processEvents()
                        
                        # アノテーションを保存
                        self.annotations[img_path] = {
                            "angle": result.get("angle", 0),
                            "throttle": result.get("throttle", 0),
                            "x": result.get("x", 0),
                            "y": result.get("y", 0)
                        }

                        # 位置情報があれば追加
                        if "loc" in result or "pilot/loc" in result:
                            loc_value = result.get("pilot/loc", result.get("loc", 0))
                            self.annotations[self.current_index]["loc"] = loc_value
                            self.location_annotations[img_path] = loc_value
                            
                            # 位置情報ボタンがまだなimg_pathければ追加
                            self.ensure_location_button_exists(loc_value)

                        # タイムスタンプを記録
                        self.annotation_timestamps[img_path] = int(time.time() * 1000)
                        
                        # 推論結果も保存
                        self.inference_results[img_path] = result
                        
                        # カウント更新
                        processed_count += 1
                        success_count += 1
                    
                except Exception as e:
                    print(f"バッチ {batch_idx+1} 処理中にエラー: {e}")
                    # エラーがあっても次のバッチを処理する
                    processed_count += len(current_batch)
            
            # 最終処理
            if not progress.wasCanceled():
                # アノテーションカウントを更新
                self.annotated_count = len(self.annotations)
                
                # 位置ボタンのカウント表示を更新
                progress.setLabelText("位置情報ボタンを更新中...")
                progress.setValue(96)
                QApplication.processEvents()
                self.update_location_button_counts()
                
                # UI更新
                progress.setLabelText("UI表示を更新中...")
                progress.setValue(98)
                QApplication.processEvents()
                self.update_stats()
                self.display_current_image()
                self.update_gallery()

                # 分布グラフを更新
                progress.setLabelText("分布グラフを更新中...")
                progress.setValue(99)
                QApplication.processEvents()
                self.update_distribution_graph()

                # 完了表示
                progress.setLabelText(f"完了: {success_count}枚の画像にオートアノテーションを適用しました")
                progress.setValue(100)
                QApplication.processEvents()
            
            # 処理完了
            progress.close()
            
            if not progress.wasCanceled():
                QMessageBox.information(
                    self, 
                    "完了", 
                    f"{success_count}枚の画像にオートアノテーションを適用しました。\n"
                    f"使用モデル: {model_type}" + 
                    (f" ({os.path.basename(model_path)})" if model_path else " (事前学習済み)")
                )
            else:
                QMessageBox.information(
                    self, 
                    "キャンセル", 
                    f"オートアノテーションがキャンセルされました。\n"
                    f"{success_count}枚の画像が処理されました。"
                )
                
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"オートアノテーション中にエラーが発生しました: {str(e)}"
            )

    def show_augmentation_preview_dialog(self, aug_params):        
        if not self.images:
            QMessageBox.warning(self, "警告", "プレビュー対象の画像がありません。")
            return
        
        # オーグメンテーションが無効の場合
        if not aug_params['enabled']:
            QMessageBox.information(self, "情報", "データオーグメンテーションが無効になっています。")
            return
        
        # 現在表示中の画像を取得
        current_img_path = self.images[self.current_index]
        
        try:
            # モジュールのインポートをここで行う
            print("model_training モジュールをインポート中...")
            
            print(f"現在の画像パス: {current_img_path}")
            
            # オーグメンテーションサンプルを生成
            print("サンプル生成開始...")
            samples = generate_augmentation_samples(
                current_img_path,
                num_samples=5,  # オリジナル含め5枚表示
                use_flip=aug_params['use_flip'],
                flip_prob=aug_params['flip_prob'],
                use_color=aug_params['use_color'],
                brightness=aug_params['brightness'],
                contrast=aug_params['contrast'],
                saturation=aug_params['saturation'],
                use_geometry=aug_params['use_geometry'],
                rotation_degrees=aug_params['rotation_degrees'],
                translate_ratio=aug_params['translate_ratio'],
                use_erase=aug_params['use_erase'],
                erase_prob=aug_params['erase_prob'],
                erase_min_ratio=aug_params['erase_min_ratio'],
                erase_max_ratio=aug_params['erase_max_ratio']
            )
            print(f"サンプル生成完了: {len(samples)}枚")
            
            # プレビューダイアログを作成
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("オーグメンテーションプレビュー")
            preview_dialog.setMinimumWidth(800)
            preview_dialog.setMinimumHeight(500)
            
            preview_layout = QVBoxLayout(preview_dialog)
            
            # タイトルラベル
            title_label = QLabel("オーグメンテーションプレビュー")
            title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
            title_label.setAlignment(Qt.AlignCenter)
            preview_layout.addWidget(title_label)
            
            # 適用中の設定を表示
            settings_text = "適用中の設定:\n"
            if aug_params['use_flip']:
                settings_text += f"・水平反転 (確率: {aug_params['flip_prob']})\n"
            if aug_params['use_color']:
                settings_text += f"・色調整 (明るさ: ±{aug_params['brightness']}, "
                settings_text += f"コントラスト: ±{aug_params['contrast']}, "
                settings_text += f"彩度: ±{aug_params['saturation']})\n"
            if aug_params['use_geometry']:
                settings_text += f"・幾何変換 (回転: ±{aug_params['rotation_degrees']}度, "
                settings_text += f"平行移動: ±{aug_params['translate_ratio']})\n"
            if aug_params['use_erase']:
                settings_text += f"・ランダムイレース (確率: {aug_params['erase_prob']}, "
                settings_text += f"範囲: {aug_params['erase_min_ratio']}～{aug_params['erase_max_ratio']})\n"
                
            settings_label = QLabel(settings_text)
            settings_label.setStyleSheet("font-size: 12px;")
            preview_layout.addWidget(settings_label)
            
            # 画像表示用のグリッドレイアウト
            images_widget = QWidget()
            images_layout = QGridLayout(images_widget)
            images_layout.setContentsMargins(10, 10, 10, 10)
            images_layout.setSpacing(10)
            
            # 画像を配置（最初はオリジナル）
            original_img = samples[0]
            original_label = QLabel()
            original_pixmap = QPixmap.fromImage(self.pil_to_qimage(original_img))
            original_label.setPixmap(original_pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            original_label.setAlignment(Qt.AlignCenter)
            
            label_text = QLabel("オリジナル画像")
            label_text.setAlignment(Qt.AlignCenter)
            
            # オリジナルを配置
            images_layout.addWidget(original_label, 0, 0)
            images_layout.addWidget(label_text, 1, 0)
            
            # オーグメンテーションサンプルを配置
            for i, sample in enumerate(samples[1:], 1):
                img, description = sample
                sample_label = QLabel()
                sample_pixmap = QPixmap.fromImage(self.pil_to_qimage(img))
                sample_label.setPixmap(sample_pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                sample_label.setAlignment(Qt.AlignCenter)
                
                # 説明ラベル（適用された変換）
                desc_label = QLabel(description)
                desc_label.setAlignment(Qt.AlignCenter)
                desc_label.setWordWrap(True)
                
                col = i % 2
                row = (i // 2) * 2
                
                images_layout.addWidget(sample_label, row, col)
                images_layout.addWidget(desc_label, row + 1, col)
            
            # スクロールエリアに配置
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(images_widget)
            preview_layout.addWidget(scroll_area)
            
            # 閉じるボタン
            close_button = QPushButton("閉じる")
            close_button.clicked.connect(preview_dialog.accept)
            preview_layout.addWidget(close_button)
            
            # ダイアログを表示
            preview_dialog.exec_()
        
        except Exception as e:
            print(f"プレビュー生成中にエラー: {str(e)}")
            traceback.print_exc()  # スタックトレースを出力
            QMessageBox.critical(self, "エラー", f"プレビュー生成中にエラーが発生しました: {str(e)}")
        
    def get_augmentation_params(self):
        """現在のオーグメンテーション設定をパラメータ辞書として取得する"""
        return {
            'enabled': self.augmentation_checkbox.isChecked(),
            'use_flip': self.aug_flip_checkbox.isChecked(),
            'flip_prob': self.aug_flip_proba.value(),
            'use_color': self.aug_color_checkbox.isChecked(),
            'brightness': self.aug_brightness.value(),
            'contrast': self.aug_contrast.value(),
            'saturation': self.aug_saturation.value(),
            'use_geometry': self.aug_geometry_checkbox.isChecked(),
            'rotation_degrees': self.aug_rotation.value(),
            'translate_ratio': self.aug_translate.value(),
            'use_erase': self.aug_erase_checkbox.isChecked(),
            'erase_prob': self.aug_erase_proba.value(),
            'erase_min_ratio': self.aug_erase_min_ratio.value(),
            'erase_max_ratio': self.aug_erase_max_ratio.value()
        }

    def pil_to_qimage(self, pil_image):
        """PIL Imageをqtで使用可能なQImageに変換する"""
        
        # RGBに変換して確実にフォーマットを統一
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # NumPy配列に変換
        img_array = np.array(pil_image)
        
        # QImageに変換（RGBフォーマット）
        height, width, channels = img_array.shape
        bytes_per_line = channels * width
        
        return QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def update_detection_info_panel(self):
        """物体検知推論結果の情報パネルを更新する"""
        if not self.images:
            return
            
        current_img_path = self.images[self.current_index]
        is_deleted = hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes
        
        if hasattr(self, 'detection_inference_checkbox') and self.detection_inference_checkbox.isChecked() and not is_deleted:
            if current_img_path in self.detection_inference_results:
                # クラスごとのカウント辞書を作成
                class_counts = {}
                inference_bboxes = self.detection_inference_results[current_img_path]
                
                for bbox in inference_bboxes:
                    class_name = bbox.get('class', 'unknown')
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # 情報テキストを構築
                inference_text = "<b>物体検知推論結果:</b><br>"
                inference_text += "検出オブジェクト:<br>"
                
                for class_name, count in class_counts.items():
                    # クラスに応じた色を設定
                    class_colors = {
                        'car': "#FF0000",     # 赤
                        'person': "#00FF00",  # 緑
                        'sign': "#0000FF",    # 青
                        'cone': "#FFFF00",    # 黄
                        'unknown': "#808080"  # グレー
                    }
                    color = class_colors.get(class_name, "#FF0000")
                    
                    inference_text += f"<span style='color: {color}; font-weight: bold;'>● {class_name}</span>: {count}個<br>"
                
                inference_text += f"合計: {len(inference_bboxes)}個のオブジェクト<br>"
                
                # テキストをラベルに直接設定
                if hasattr(self, 'detection_inference_info_label'):
                    self.detection_inference_info_label.setText(inference_text)
                    self.detection_inference_info_label.setTextFormat(Qt.RichText)
                    # 更新を強制
                    self.detection_inference_info_label.repaint()
                
                return True
                
            elif hasattr(self, 'run_single_yolo_inference') and hasattr(self, 'yolo_model'):
                # 推論結果がない場合は実行
                return self.run_single_yolo_inference()
        else:
            # 表示がオフの場合はラベルをクリア
            if hasattr(self, 'detection_inference_info_label'):
                self.detection_inference_info_label.setText("")
        
        return False

### location 系
    def estimate_location_from_angle(self, angle):
        """角度から位置情報を推定する（簡易的な実装）"""
        # 角度の範囲に基づいて位置を推定
        # 例: -1.0～1.0の範囲を8つの位置に分割
        angle_range = 2.0  # -1.0～1.0
        section_size = angle_range / 8
        
        # 角度を位置番号に変換（0～7）
        normalized_angle = angle + 1.0  # 0～2.0に正規化
        location = int(normalized_angle / section_size)
        
        # 範囲外の値を調整
        location = max(0, min(location, 7))
        
        return location

    def init_location_buttons(self):
        """初期位置情報ボタンを設定する"""
        # 位置ボタン数
        num_buttons = 8
        
        # 既存のボタンをクリア
        for button in self.location_buttons:
            if button.parent():
                button.setParent(None)
        self.location_buttons.clear()
        
        # 8つの位置情報ボタンを作成
        for i in range(num_buttons):
            button = QPushButton(f"0 | 位置 {i}")  # カウント0で初期化
            button.setProperty("location_value", i)
            button.setCheckable(True)  # チェック可能に設定
            button.clicked.connect(lambda checked, value=i: self.set_location(value))
            
            # 対応する色を取得
            color = get_location_color(i)
            
            # ボタンのスタイルを設定
            button.setStyleSheet(f"""
                QPushButton {{
                    padding: 8px;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    background-color: #f0f0f0;
                    color: #888888;
                }}
                QPushButton:checked {{
                    background-color: {color.name()};
                    color: white;
                    font-weight: bold;
                }}
            """)
            
            # レイアウトに追加
            self.location_buttons_layout.addWidget(button)
            self.location_buttons.append(button)

    def train_location_model(self):
        """位置情報推論モデル（分類）の学習を行い、成果物を保存する"""
        if not self.images or not self.location_annotations:
            QMessageBox.warning(self, "警告", "位置情報アノテーションが存在しません。")
            return


        image_paths = list(self.location_annotations.keys())
        labels = [self.location_annotations[p] for p in image_paths]

        save_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)

        save_path, best_acc = train_location_model(
            model_name="location_simplecnn",
            image_paths=image_paths,
            labels=labels,
            save_dir=save_dir
        )

        QMessageBox.information(self, "完了", f"位置情報モデルの学習が完了しました。\n最良精度: {best_acc:.3f}\nモデルは {save_path} に保存されました。")

    def add_location_model_section(self):
        """位置推論モデルのセクションを追加する"""
        # 位置モデルマネージャーの初期化
        self.location_model_manager = LocationModelManager(APP_DIR_PATH, MODELS_DIR_NAME)

        # YOLOモデル領域の上に位置推論モデルセクションを配置するため、
        # オリジナルのレイアウトを取得
        left_layout = self.findChild(QVBoxLayout, "left_layout")
        if not left_layout:
            # レイアウトが見つからない場合、centralWidgetを取得して探す
            left_layout = self.centralWidget().layout().itemAt(0).widget().layout()
        
        # 既存のYOLO物体検知コンテナのインデックスを見つける
        object_detection_index = -1
        for i in range(left_layout.count()):
            item = left_layout.itemAt(i)
            if item.widget() == self.object_detection_container:
                object_detection_index = i
                break
        
        if object_detection_index == -1:
            print("物体検知コンテナが見つかりません。位置モデルセクションは末尾に追加します。")
            object_detection_index = left_layout.count()
        
        # 位置推論モデルコンテナを作成
        self.location_model_container = QWidget()
        location_model_layout = QVBoxLayout(self.location_model_container)
        
        # ヘッダータイトル
        location_model_label = QLabel("位置推論モデル:")
        location_model_label.setStyleSheet("font-weight: bold")
        location_model_layout.addWidget(location_model_label)
        
        # モデル選択
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("モデルタイプ:"))
        self.location_model_combo = QComboBox()
        self.location_model_combo.addItems(["donkey_location", "resnet18_location"])
        # イベントハンドラを接続（追加）
        self.location_model_combo.currentIndexChanged.connect(self.on_location_model_type_changed)
        model_type_layout.addWidget(self.location_model_combo)
        location_model_layout.addLayout(model_type_layout)
        
        # 事前学習済みモデル選択
        self.location_saved_model_combo = QComboBox()
        self.location_saved_model_combo.setMinimumWidth(180)
        self.location_saved_model_combo.setStyleSheet("combobox-popup: 0;")
        location_model_layout.addWidget(self.location_saved_model_combo)
        
        # モデル操作ボタン
        location_model_buttons_layout = QHBoxLayout()
        
        # モデルリスト更新ボタン
        self.location_refresh_button = QPushButton("モデル一覧更新")
        self.location_refresh_button.clicked.connect(self.refresh_location_model_list)
        location_model_buttons_layout.addWidget(self.location_refresh_button)
        
        # モデル読み込みボタン
        self.location_load_button = QPushButton("モデル読込")
        self.location_load_button.setToolTip("modelsフォルダのモデルを読込む")
        self.location_load_button.clicked.connect(self.load_location_model)
        apply_style(self.location_load_button, 'model')
        location_model_buttons_layout.addWidget(self.location_load_button)
        
        location_model_layout.addLayout(location_model_buttons_layout)
        
        # 位置モデル学習ボタン
        train_location_button = QPushButton("位置モデルを学習・保存")
        train_location_button.clicked.connect(self.train_and_save_location_model)
        apply_style(train_location_button, 'training')
        location_model_layout.addWidget(train_location_button)
        
        # 位置推論表示チェックボックス
        location_inference_layout = QHBoxLayout()
        self.location_inference_checkbox = QCheckBox("位置推論結果表示")
        self.location_inference_checkbox.setChecked(False)
        self.location_inference_checkbox.stateChanged.connect(self.toggle_location_inference_display)
        location_inference_layout.addWidget(self.location_inference_checkbox)
        location_model_layout.addLayout(location_inference_layout)
        
        # YOLOコンテナの前に位置モデルコンテナを挿入
        left_layout.insertWidget(object_detection_index, self.location_model_container)
        
        # 推論結果格納用の辞書を初期化
        self.location_inference_results = {}
        
        # モデルリストの取得は新しいマネージャーを使用
        self.refresh_location_model_list()
        
        # 推論結果格納用の辞書を初期化
        self.location_inference_results = {}

    def run_location_inference(self):
        """現在の画像に対して位置推論を実行"""
        if not self.images or not hasattr(self, 'location_model_manager'):
            return
        
        current_img_path = self.images[self.current_index]
        
        # マネージャーを使用して推論を実行
        result = self.location_model_manager.run_inference(current_img_path)
        
        if result:
            # 推論結果を保存
            self.location_inference_results[current_img_path] = result
            
            # 表示を更新
            self.update_location_inference_display()
            
            return True
        
        return False

    def refresh_location_model_list(self):
        """保存されている位置モデルのリストを更新 - 選択したタイプでフィルタリング"""
        self.location_saved_model_combo.clear()
        
        # 更新開始のメッセージを表示
        self.statusBar().showMessage("位置モデルリストを更新中...")
        
        # 現在選択されているモデルタイプを取得
        selected_model_type = self.location_model_combo.currentText()
        
        # モデルマネージャーからモデルリストを取得 - タイプ指定
        model_files = self.location_model_manager.get_model_list(model_type=selected_model_type)
        
        if not model_files:
            # フィルタリングした結果がなければ、その旨を表示
            self.location_saved_model_combo.addItem(f"{selected_model_type}の位置モデルが見つかりません")
            self.statusBar().showMessage(f"{selected_model_type}の位置モデルが見つかりません。モデルを学習してください", 3000)
            return
        
        # コンボボックスに追加
        for model_file in model_files:
            self.location_saved_model_combo.addItem(model_file)
        
        # 更新完了メッセージ
        self.statusBar().showMessage(f"{len(model_files)}個の{selected_model_type}位置モデルを読み込みました", 3000)

    def load_location_model(self):
        """選択された位置モデルを読み込む"""
        if not self.images:
            QMessageBox.warning(self, "警告", "画像が読み込まれていません。")
            return
        
        # モデル情報を取得
        model_type = self.location_model_combo.currentText()
        selected_model = self.location_saved_model_combo.currentText()
        
        if selected_model == "位置モデルが見つかりません" or selected_model == "フォルダを選択してください":
            QMessageBox.warning(self, "警告", "有効な位置モデルが選択されていません。")
            return
        
        # モデルのパスを取得
        model_path = os.path.join(models_dir, selected_model)
        
        # モデルが存在するか確認
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "警告", f"選択されたモデルが見つかりません: {selected_model}")
            return
        
        # 進捗ダイアログを表示
        progress = QProgressDialog(
            f"位置モデル '{model_type} ({selected_model})' を読み込み中...", 
            "キャンセル", 0, 100, self
        )
        progress.setWindowTitle("モデル読み込み")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        # 進捗コールバック関数
        def update_progress(value, message=None):
            if message:
                progress.setLabelText(message)
            progress.setValue(value)
            QApplication.processEvents()
            return not progress.wasCanceled()
        
        try:
            # マネージャーを使用してモデルをロード
            success, result = self.location_model_manager.load_model(
                model_type, model_path, update_progress
            )
            
            if not success:
                progress.close()
                QMessageBox.critical(
                    self, 
                    "エラー", 
                    f"位置モデルの読み込み中にエラーが発生しました: {result}"
                )
                return
            
            num_classes = result
            
            update_progress(80, "初期推論を実行中...")
            
            # 現在の画像に対して推論を実行
            self.run_location_inference()
            
            update_progress(90, "推論表示を更新中...")
            
            # 推論表示チェックボックスを自動的にオンにする
            self.location_inference_checkbox.setChecked(True)
            
            update_progress(100)
            progress.close()
            
            # 成功メッセージ
            self.statusBar().showMessage(f"位置モデル '{model_type} ({selected_model})' を読み込みました (クラス数: {num_classes})", 5000)
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"位置モデルの読み込み中にエラーが発生しました: {str(e)}"
            )
    
    def on_location_model_type_changed(self, index):
        """位置モデルタイプが変更されたときの処理"""
        # 現在選択されているモデルタイプを取得
        selected_model_type = self.location_model_combo.currentText()
        self.statusBar().showMessage(f"位置モデルタイプを「{selected_model_type}」に変更しました。モデルリストを更新します...")
        
        # モデルリストを更新
        self.refresh_location_model_list()

    def run_location_inference(self):
        """現在の画像に対して位置推論を実行"""
        if not self.images or not hasattr(self, 'location_model_manager'):
            return
        
        current_img_path = self.images[self.current_index]
        
        # マネージャーを使用して推論を実行
        result = self.location_model_manager.run_inference(current_img_path)
        
        if result:
            # 推論結果を保存
            self.location_inference_results[current_img_path] = result
            
            # 表示を更新
            self.update_location_inference_display()
            
            return True
        
        return False

    # 位置推論表示更新
    def update_location_inference_display(self):
        """位置推論表示を更新する - 上位3クラスのみ表示"""
        if not self.images:
            return
        
        current_img_path = self.images[self.current_index]
        is_deleted = hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes
        
        if hasattr(self, 'location_inference_checkbox') and self.location_inference_checkbox.isChecked() and not is_deleted:
            if current_img_path in self.location_inference_results:
                # 推論結果を取得
                result = self.location_inference_results[current_img_path]
                pred_class = result['pred_class']
                confidence = result['confidence']
                all_probs = result.get('all_probs', [])
                
                # 情報テキストを構築
                inference_text = "<b>位置推論結果:</b><br>"
                
                # 一番高いクラスは背景色付きで表示
                loc_color = get_location_color(pred_class)
                
                # 予測クラスを背景色付きで表示
                inference_text += f"<div style='background-color: {loc_color.name()}; color: white; font-weight: bold; padding: 5px; border-radius: 5px; margin: 5px 0;'>"
                inference_text += f"予測位置: {pred_class} (確信度: {confidence:.4f})</div>"
                
                # 上位3クラスの予測結果を表示（すでに確率でソートされている前提）
                if all_probs:
                    # 確率が高い順にインデックスをソート
                    sorted_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)
                    
                    # 上位3つ（または全部、少ない方）を取得
                    top_k = min(3, len(sorted_indices))
                    top_indices = sorted_indices[:top_k]
                    
                    inference_text += "<br>上位クラス:<br>"
                    
                    for i, idx in enumerate(top_indices):
                        # すでに表示したクラスは飛ばす
                        if idx == pred_class and i > 0:
                            continue
                        
                        # 位置によって色を変える
                        color = get_location_color(idx).name()
                        
                        # 各クラスの予測確率
                        inference_text += f"{i+1}. 位置 {idx}: <span style='color: {color}; font-weight: bold;'>{all_probs[idx]:.4f}</span><br>"
                
                # テキストをラベルに設定
                if hasattr(self, 'location_inference_info_label'):
                    self.location_inference_info_label.setText(inference_text)
                    self.location_inference_info_label.setTextFormat(Qt.RichText)
                else:
                    # ラベルがない場合は新規作成
                    self.location_inference_info_label = QLabel(inference_text)
                    self.location_inference_info_label.setWordWrap(True)
                    self.location_inference_info_label.setStyleSheet("color: purple;")
                    
                    # レイアウトに追加（推論結果表示の下）
                    if hasattr(self, 'detection_inference_info_label') and self.detection_inference_info_label.parent():
                        parent_layout = self.detection_inference_info_label.parent().layout()
                        if parent_layout:
                            parent_layout.addWidget(self.location_inference_info_label)
                    elif hasattr(self, 'inference_info_label') and self.inference_info_label.parent():
                        parent_layout = self.inference_info_label.parent().layout()
                        if parent_layout:
                            parent_layout.addWidget(self.location_inference_info_label)
                
                return True
                    
            # モデルがロードされていて推論結果がない場合は実行
            elif hasattr(self, 'location_model_manager') and self.location_model_manager.is_model_loaded():
                self.run_location_inference()
                # 再帰的に呼び出して表示を更新
                return self.update_location_inference_display()
        else:
            # 表示がオフの場合はラベルをクリア
            if hasattr(self, 'location_inference_info_label'):
                self.location_inference_info_label.setText("")
        
        return False
    
    # 位置モデル学習
    def train_and_save_location_model(self):
        """位置モデルを学習して保存する"""
        if not self.images or not self.location_annotations:
            QMessageBox.warning(self, "警告", "位置モデルを学習するには位置アノテーションが必要です。")
            return
        
        # 使用可能な位置情報のユニークなリストを取得
        unique_locations = sorted(list(set(self.location_annotations.values())))
        actual_classes = len(unique_locations)
        
        if actual_classes < 2:
            QMessageBox.warning(self, "警告", f"位置モデルを学習するには少なくとも2つの異なる位置ラベルが必要です。現在: {actual_classes}種類")
            return
        
        # 常に8クラスを使用（実際のクラス数に関わらず）
        num_classes = 8
        
        # 選択されたモデル
        model_type = self.location_model_combo.currentText()
        
        # 学習設定ダイアログを表示
        training_settings = QDialog(self)
        training_settings.setWindowTitle("位置モデル学習設定")
        training_settings.setMinimumWidth(500)
        
        settings_layout = QVBoxLayout(training_settings)
        
        # 現在の位置情報の概要を表示
        info_label = QLabel(f"検出された位置ラベル: {actual_classes}種類 ({', '.join(map(str, unique_locations))})")
        settings_layout.addWidget(info_label)
        
        # 固定クラス数の情報表示
        fixed_class_label = QLabel(f"※ 位置モデルは常に{num_classes}クラス出力で作成されます。")
        fixed_class_label.setStyleSheet("color: #666666; font-style: italic;")
        settings_layout.addWidget(fixed_class_label)
        
        # エポック数設定
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("学習エポック数:"))
        epoch_spin = QSpinBox()
        epoch_spin.setRange(1, 1000)
        epoch_spin.setValue(30)  # デフォルト: 30エポック
        epoch_layout.addWidget(epoch_spin)
        settings_layout.addLayout(epoch_layout)
        
        # バッチサイズ設定
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("バッチサイズ:"))
        batch_spin = QSpinBox()
        batch_spin.setRange(1, 128)
        batch_spin.setValue(16)  # デフォルト: 16
        batch_layout.addWidget(batch_spin)
        settings_layout.addLayout(batch_layout)
        
        # データオーグメンテーション設定
        aug_check = QCheckBox("データオーグメンテーションを有効にする")
        aug_check.setChecked(True)
        settings_layout.addWidget(aug_check)
        
        # Early Stopping設定
        early_stopping_check = QCheckBox("Early Stopping を有効にする")
        early_stopping_check.setChecked(True)
        settings_layout.addWidget(early_stopping_check)
        
        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("忍耐エポック数:"))
        patience_spin = QSpinBox()
        patience_spin.setRange(1, 20)
        patience_spin.setValue(5)
        patience_spin.setEnabled(True)
        patience_layout.addWidget(patience_spin)
        settings_layout.addLayout(patience_layout)
        
        # 学習率設定
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学習率:"))
        
        lr_combo = QComboBox()
        learning_rates = ["0.001", "0.0005", "0.0001", "0.00005", "0.00001"]
        lr_combo.addItems(learning_rates)
        lr_combo.setCurrentIndex(0)  # デフォルト: 0.001
        lr_layout.addWidget(lr_combo)
        settings_layout.addLayout(lr_layout)
        
        # ボタンの配置
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(training_settings.accept)
        button_box.rejected.connect(training_settings.reject)
        settings_layout.addWidget(button_box)
        
        # ダイアログを表示
        if not training_settings.exec_():
            return
        
        # 設定値の取得
        num_epochs = epoch_spin.value()
        batch_size = batch_spin.value()
        use_augmentation = aug_check.isChecked()
        use_early_stopping = early_stopping_check.isChecked()
        patience = patience_spin.value() if use_early_stopping else 0
        learning_rate = float(lr_combo.currentText())
        
        # データの準備
        try:
            # 位置ラベルのマッピングを作成（実際の位置値をインデックスに変換）
            location_to_index = {loc: i for i, loc in enumerate(unique_locations)}
                        
            # データを準備
            image_paths, location_labels = prepare_location_data_from_annotations(
                self.annotations, 
                self.location_annotations,
                self.images  # self.imagesリストを追加で渡す
            )

            # ラベルをインデックスに変換
            location_indices = [location_to_index[label] for label in location_labels]
            
            # 進捗ダイアログを表示
            progress = QProgressDialog(
                f"位置モデル '{model_type}' の学習データを準備中...", 
                "キャンセル", 0, 100, self
            )
            progress.setWindowTitle("位置モデル学習")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()
            
            # データセット作成
            train_loader, val_loader, dataset_info = create_location_datasets(
                image_paths=image_paths,
                location_labels=location_indices,
                val_split=0.2,
                model_name=model_type,
                batch_size=batch_size,
                use_augmentation=use_augmentation
            )
            
            progress.setValue(20)
            progress.setLabelText(f"モデル '{model_type}' を初期化中... (固定{num_classes}クラス)")
            QApplication.processEvents()
                        
            # 進捗コールバック関数
            def update_progress(current, total, message=None):
                if message:
                    progress.setLabelText(message)
                progress.setValue(20 + int(current * 70 / total))
                QApplication.processEvents()
                return not progress.wasCanceled()
            
            # モデル学習 - 常に8クラスで初期化
            training_results = train_location_model(
                model_name=model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,  # 常に8クラスを使用
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                save_dir=models_dir,
                progress_callback=update_progress,
                use_early_stopping=use_early_stopping,
                patience=patience
            )
            
            # モデルのメタデータに実際のクラス数を保存
            # チェックポイントを読み込み
            checkpoint_path = training_results['best_model_path']
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 'num_classes'キーに実際のクラス数とマッピング情報を追加
            checkpoint['num_classes'] = num_classes
            checkpoint['actual_classes'] = actual_classes
            checkpoint['location_mapping'] = location_to_index
            
            # 更新したチェックポイントを保存
            torch.save(checkpoint, checkpoint_path)
            
            progress.setValue(95)
            progress.setLabelText("モデルリストを更新中...")
            QApplication.processEvents()
            
            # MLflowに結果を記録
            if hasattr(self, 'initialize_mlflow') and hasattr(self, 'log_model_to_mlflow'):
                try:
                    # MLflow初期化
                    self.initialize_mlflow()
                    
                    # MLflowに記録
                    self.log_model_to_mlflow(
                        model_path=training_results['best_model_path'],
                        model_type=model_type,
                        training_params={
                            "num_epochs": num_epochs,
                            "completed_epochs": training_results['completed_epochs'],
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "use_early_stopping": use_early_stopping,
                            "patience": patience,
                            "early_stopped": training_results['early_stopped'],
                            "augmentation_enabled": use_augmentation,
                            "fixed_classes": num_classes,
                            "actual_classes": actual_classes
                        },
                        metrics={
                            "best_val_loss": training_results['best_val_loss'],
                            "best_val_acc": training_results['best_val_acc'],
                            "train_losses": training_results['train_losses'],
                            "val_losses": training_results['val_losses'],
                            "train_accuracies": training_results['train_accuracies'],
                            "val_accuracies": training_results['val_accuracies']
                        },
                        dataset_info={
                            "total_annotations": len(self.location_annotations),
                            "used_samples": len(image_paths),
                            "train_samples": dataset_info['train_samples'],
                            "val_samples": dataset_info['val_samples'],
                            "input_shape": dataset_info['actual_image_size'],
                            "num_classes": num_classes,
                            "actual_classes": actual_classes
                        }
                    )
                    mlflow_msg = "MLflowに学習履歴を記録しました。"
                except Exception as e:
                    print(f"MLflow記録エラー: {e}")
                    mlflow_msg = f"MLflowへの記録に失敗しました: {str(e)}"
            else:
                mlflow_msg = "MLflow機能が利用できないため、学習履歴は記録されませんでした。"
            
            # モデルリストを更新
            self.refresh_location_model_list()
            
            progress.setValue(100)
            progress.close()
            
            # 学習完了メッセージ
            QMessageBox.information(
                self, 
                "学習完了", 
                f"{model_type} 位置モデルを学習し保存しました: {os.path.basename(training_results['best_model_path'])}\n" +
                f"最良検証損失: {training_results['best_val_loss']:.6f}\n" +
                f"最良検証精度: {training_results['best_val_acc']:.2f}%\n" +
                f"実施エポック数: {training_results['completed_epochs']}/{num_epochs}\n" +
                f"出力クラス数: {num_classes} (実際の位置クラス数: {actual_classes})\n" +
                f"学習データ数: {len(image_paths)}枚\n" +
                f"{mlflow_msg}"
            )
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            traceback.print_exc()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"位置モデル学習中にエラーが発生しました: {str(e)}"
            )

    # 一括位置推論実行
    def run_batch_location_inference(self):
        """すべての画像に対して位置推論を実行"""
        if not self.images or not hasattr(self, 'location_model'):
            QMessageBox.warning(self, "警告", "位置モデルが読み込まれていないか、画像がありません。")
            return
        
        # 確認ダイアログ
        reply = QMessageBox.question(
            self,
            "一括推論確認",
            f"全{len(self.images)}枚の画像に対して位置推論を実行しますか？\n"
            f"処理には時間がかかる場合があります。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # 進捗ダイアログ
        progress = QProgressDialog("位置推論実行中...", "キャンセル", 0, len(self.images), self)
        progress.setWindowTitle("一括推論中")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            # バッチサイズ
            batch_size = 64
            
            # 全画像処理
            processed_count = 0
            
            for i in range(0, len(self.images), batch_size):
                if progress.wasCanceled():
                    break
                
                # 現在のバッチ
                batch_paths = self.images[i:i+batch_size]
                progress.setValue(i)
                
                # 各画像に対して推論実行
                for img_path in batch_paths:
                    try:
                        # 画像を読み込む
                        img = Image.open(img_path).convert('RGB')
                        
                        # モデルの前処理を取得
                        if not hasattr(self.location_model, '_preprocess') or self.location_model._preprocess is None:
                            self.location_model._preprocess = self.location_model.get_preprocess()
                        
                        # 前処理を適用
                        tensor_image = self.location_model._preprocess(img)
                        tensor_image = tensor_image.unsqueeze(0)
                        
                        # デバイスを取得
                        device = next(self.location_model.parameters()).device
                        tensor_image = tensor_image.to(device)
                        
                        # 推論実行
                        with torch.no_grad():
                            logits = self.location_model(tensor_image)
                            probs = torch.softmax(logits, dim=1)
                            
                            # クラスインデックスと確率を取得
                            max_prob, pred_class = torch.max(probs, dim=1)
                            
                            # 全クラスの確率をリストとして取得
                            all_probs = probs[0].cpu().numpy().tolist()
                        
                        # 推論結果を保存
                        self.location_inference_results[img_path] = {
                            'pred_class': pred_class.item(),
                            'confidence': max_prob.item(),
                            'all_probs': all_probs
                        }
                        
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"画像 {img_path} の推論中にエラー: {e}")
                        continue
                
                # 進捗表示を更新
                progress.setValue(processed_count)
            
            progress.close()
            
            # 推論表示チェックボックスを自動的にオンにする
            self.location_inference_checkbox.setChecked(True)
            
            # 表示を更新
            self.update_location_inference_display()
            
            # 完了メッセージ
            if not progress.wasCanceled():
                QMessageBox.information(
                    self,
                    "一括推論完了",
                    f"{processed_count}枚の画像に対する位置推論を完了しました。"
                )
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "エラー",
                f"一括位置推論実行中にエラーが発生しました: {str(e)}"
            )

    # UI関連
    def toggle_theme(self, theme_name):
        """テーマを切り替える"""
        # styles.pyのset_theme関数を呼び出す
        current_theme = set_theme(theme_name)
        
        # テーマ切り替えに応じてボタンの状態を更新
        self.light_theme_button.setChecked(current_theme == "light")
        self.dark_theme_button.setChecked(current_theme == "dark")
        
        # テーマ変更を適用するには、すべてのウィジェットに新しいスタイルを適用し直す必要がある
        self.apply_theme_to_widgets()
        
        # テーマ切り替えボタンのスタイルを特別に更新
        self.update_theme_button_styles()
        
        # テーマ変更の通知
        theme_display_name = "ライト" if theme_name == "light" else "ダーク"
        self.statusBar().showMessage(f"テーマを {theme_display_name} に変更しました", 3000)

    def update_theme_button_styles(self):
        """テーマ切り替えボタンのスタイルを現在のテーマに合わせて更新"""
        current_theme = get_current_theme()  # styles.pyから関数をインポートする必要があります
        
        if current_theme == "light":
            # ライトテーマの場合
            self.light_theme_button.setStyleSheet("""
                QPushButton {
                    background-color: #F9FAFB; /* ライトテーマの背景色 */
                    color: #111827; /* 暗いテキスト */
                    font-weight: bold;
                    border: 1px solid #E5E7EB;
                    border-radius: 4px;
                    padding: 6px 12px;
                }
                QPushButton:checked {
                    background-color: #E5E7EB; /* 選択時は少し暗い */
                    border: 2px solid #2563EB; /* 選択時は青いボーダー */
                }
            """)
            
            self.dark_theme_button.setStyleSheet("""
                QPushButton {
                    background-color: #1F2937; /* ダークテーマの背景色 */
                    color: white; /* 白いテキスト */
                    font-weight: bold;
                    border: 1px solid #4B5563;
                    border-radius: 4px;
                    padding: 6px 12px;
                }
                QPushButton:checked {
                    background-color: #374151; /* 選択時は少し明るい */
                    border: 2px solid #3B82F6; /* 選択時は青いボーダー */
                }
            """)
        else:
            # ダークテーマの場合
            self.light_theme_button.setStyleSheet("""
                QPushButton {
                    background-color: #F9FAFB; /* ライトテーマの背景色 */
                    color: #111827; /* 暗いテキスト */
                    font-weight: bold;
                    border: 1px solid #E5E7EB;
                    border-radius: 4px;
                    padding: 6px 12px;
                }
                QPushButton:checked {
                    background-color: #E5E7EB; /* 選択時は少し暗い */
                    border: 2px solid #3B82F6; /* 選択時は青いボーダー */
                }
            """)
            
            self.dark_theme_button.setStyleSheet("""
                QPushButton {
                    background-color: #1F2937; /* ダークテーマの背景色 */
                    color: white; /* 白いテキスト */
                    font-weight: bold;
                    border: 1px solid #4B5563;
                    border-radius: 4px;
                    padding: 6px 12px;
                }
                QPushButton:checked {
                    background-color: #374151; /* 選択時は少し明るい */
                    border: 2px solid #3B82F6; /* 選択時は青いボーダー */
                }
            """)

    def apply_theme_to_widgets(self):
        """テーマ変更後に主要なウィジェットにスタイルを再適用する"""
        # 各種ウィジェットに適用されているスタイルを再適用
        # ボタン
        if hasattr(self, 'load_button'):
            apply_style(self.load_button, 'primary')
        if hasattr(self, 'load_annotation_button'):
            apply_style(self.load_annotation_button, 'primary')
        
        # モデル関連
        if hasattr(self, 'model_refresh_button'):
            apply_style(self.model_refresh_button, 'model')
        if hasattr(self, 'model_load_button'):
            apply_style(self.model_load_button, 'model')
            
        # YOLO関連
        if hasattr(self, 'yolo_refresh_button'):
            apply_style(self.yolo_refresh_button, 'model')
        if hasattr(self, 'yolo_load_button'):
            apply_style(self.yolo_load_button, 'model')
        
        #TODO:追加修正
        # その他主要なウィジェットのスタイルを再適用
        # （ここに他のウィジェットのスタイル適用コードを追加）
        
        # テーマに応じた色調整
        #self.adjust_colors_for_theme()

    def adjust_colors_for_theme(self):
        """テーマに応じて文字色と背景色を調整する"""
        current_theme = get_current_theme()
        
        if current_theme == "dark":
            # ダークモードの場合
            text_color = "white"
            secondary_text_color = "#D1D5DB"  # 薄い白/グレー
            background_color = "#1F2937"  # ダークモードの背景色
        else:
            # ライトモードの場合
            text_color = "#111827"  # 暗いテキスト
            secondary_text_color = "#4B5563"  # やや薄い黒/グレー
            background_color = "#F9FAFB"  # ライトモードの背景色
        
        # メインウィンドウの背景色を設定
        self.setStyleSheet(f"background-color: {background_color};")
        
        # 各種ラベルの文字色を調整
        label_style = f"color: {text_color};"
        secondary_label_style = f"color: {secondary_text_color};"
        
        # 情報パネルのラベル
        if hasattr(self, 'current_image_info'):
            self.current_image_info.setStyleSheet(f"color: {text_color}; font-weight: bold;")
        
        if hasattr(self, 'annotation_info_label'):
            self.annotation_info_label.setStyleSheet(f"color: {text_color};")
        
        # 各種タイトルラベル
        for label in self.findChildren(QLabel):
            # 特定のラベルは除外（すでに特別なスタイルが設定されている可能性がある）
            if hasattr(label, 'objectName') and label.objectName() in ["detection_inference_info_label", "inference_info_label"]:
                continue
                
            if hasattr(label, 'font') and label.font().bold():
                # ボールドフォントのラベル
                current_style = label.styleSheet()
                if "color:" not in current_style:  # すでに色指定があるものは除外
                    label.setStyleSheet(label_style)
            else:
                # 通常のラベル
                current_style = label.styleSheet()
                if "color:" not in current_style:  # すでに色指定があるものは除外
                    label.setStyleSheet(secondary_label_style)


# メインプログラムのセクション
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageAnnotationTool()
    window.show()
    
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"アプリケーション実行中にエラーが発生: {e}")
        # 例外発生時のみセッション情報を保存（closeEventが呼ばれない場合の保険）
        if 'window' in locals() and hasattr(window, 'save_session_info'):
            window.save_session_info()