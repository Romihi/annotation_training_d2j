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
from datetime import datetime
import math
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')  # GUIバックエンドを使用しない設定
import matplotlib.pyplot as plt
import numpy as np
import io

import torch
import torch.nn as nn
import torch.optim as optim
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
                            QDoubleSpinBox, QDialog, QDialogButtonBox, QFormLayout,
                            QGroupBox, QRadioButton, QTabWidget, QSizePolicy,QButtonGroup,
                            QListView, QTreeView, QAbstractItemView,QStyleOptionSlider,QStyle, QTextEdit, QPlainTextEdit,
                            QGraphicsOpacityEffect, QListWidget, QListWidgetItem)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QBrush, QFont, QPolygon, QCursor
from PyQt5.QtCore import Qt, QRect, QPoint, QTimer, QEvent, QThread, pyqtSignal

from PIL import Image, ImageDraw

# ユーティリティのインポート
#from utils.color_utils import get_location_color, get_class_color, get_segmentation_color
from utils.file_utils import (
    get_image_files, extract_index_from_filename, 
    extract_variant_info, ensure_directory_exists
)
from utils.geometry_utils import (
    estimate_location_from_angle,
    normalize_coordinates, denormalize_coordinates,
    is_point_in_polygon, calculate_bbox_center, calculate_bbox_dimensions
)
from utils.image_utils import (
    pil_to_qimage, pil_to_qpixmap, 
    load_image_safely, get_image_size
)

# カスタムモジュールのインポート
from model_catalog import get_model, list_available_models
from utils.inference_utils import batch_inference
from utils.export_utils import export_to_donkey, export_to_jetracer, export_to_video, export_to_video_multi_source
from model_training import train_model, create_datasets
from model_training import train_location_model, create_location_datasets, LocationModelManager, create_waypoint_datasets
from model_training import generate_augmentation_samples
# TODO:ボタンのスタイルを実装、他のUIの移植については後ほど検討
from styles import get_location_color, apply_style, set_theme, get_current_theme, PRIMARY_STYLE, MODEL_STYLE, TRAINING_STYLE, EXPORT_STYLE, SPECIAL_STYLE, DESTRUCTIVE_STYLE, NAV_STYLE


from managers import AnnotationDataManager, MLflowManager, ModelType
from utils.yolo_utils import train_yolo_with_ui, TrainingOutputDialog
from data_analysis import DataAnalysisDialog
from utils.databricks_transfer import DatabricksTransferManager

import traceback
def exception_hook(exc_type, exc_value, exc_traceback):
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_hook

# グローバル設定変数（移植中）
from config import *

# マウス操作画面表示系
class ImageLabel(QLabel):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(MAIN_IMAGE_MIN_WIDTH, MAIN_IMAGE_MIN_HEIGHT)
        self.annotation_point = None
        self.show_grid = True  
        self.grid_size = DEFAULT_GRID_SIZE    
        self.inference_point = None 
        self.show_inference = False 
        self.zoom_factor = DEFAULT_ZOOM_FACTOR  
        self.is_deleted = False
        self.is_downsampled = False  # ダウンサンプリング対象フラグ

        # 画像描画とクリック制御関連
        self.is_image_loading = False  # 画像読み込み中フラグ
        self.click_disabled = False  # クリック無効化フラグ
        self.last_click_time = 0  # 最後のクリック時間（デバウンス用）
        self.debounce_delay = 10 #100  # デバウンス時間（ミリ秒）
        self.original_cursor = None  # 元のカーソルを保存  

        # バウンディングボックス関連
        self.bbox_start = None  
        self.bbox_end = None    
        self.bboxes = []        
        self.current_class = 0  
        self.is_drawing_bbox = False 
        self.selected_bbox_index = None 
        self.is_moving_bbox = False     
        self.move_start_pos = None      
        self.setMouseTracking(True)  
        self.hovering_bbox_index = None  
        self.is_resizing_bbox = False    
        self.resize_handle = None 
        self.resize_start_pos = None     

        # セグメンテーション関連の新規追加
        self.segmentation_polygons = []  
        self.current_segmentation_polygon = []  
        self.is_drawing_segmentation = False
        self.segmentation_inference_masks = []          
        self.selected_segmentation_index = None  
        self.is_moving_segmentation = False      
        self.seg_move_start_pos = None           
        self.hovering_segmentation_index = None 
        self.close_threshold = SEGMENTATION_CLOSE_THRESHOLD
        self.selected_polygon_index = None
        self.selected_vertex_index = None

        # ウェイポイントドラッグ関連
        self.selected_waypoint_index = None
        self.is_moving_waypoint = False
        self.waypoint_move_start_pos = None
        self.hovering_waypoint_index = None

        # 一筆書きウェイポイント関連
        self.is_drawing_waypoints = False
        self.drawing_waypoint_path = []  # ドラッグ中の軌跡を記録
        self.drawing_start_pos = None
        
        # 頂点編集関連
        self.is_moving_vertex = False
        self.hovering_polygon_index = None
        self.hovering_vertex_index = None
        self.vertex_radius = SEGMENTATION_VERTEX_RADIUS

        # マウス座標表示関連
        self.current_mouse_pos = None  # 現在のマウス位置
        self.normalized_coords = None  # 正規化座標 (x, y) -1～1の範囲

        # Speedバー関連
        self.is_adjusting_speed = False  # speedバー調整中フラグ
        self.hovering_speed_bar = False  # speedバーにホバー中フラグ
        self.speed_bar_hover_y = None  # speedバー上のホバーY座標

        # 将来アノテーション表示
        self.show_future_annotations = True  # デフォルトON

        # CAM表示関連
        self.show_gradcam = False  # CAM表示フラグ
        self.gradcam_overlay = None  # CAMオーバーレイ画像 (QPixmap)
        self.gradcam_alpha = 0.5  # CAMの透明度
        self.gradcam_target = 'angle'  # CAM対象出力 ('angle', 'throttle', 'speed')
        self.gradcam_method = 'gradcam'  # CAM手法 ('gradcam', 'gradcam++', 'eigencam', 'layercam', 'scorecam')
        self.gradcam_direction = 'both'  # 勾配方向 ('both', 'positive', 'negative')

    def add_point_to_polygon(self, polygon_index, x, y):
        """指定されたポリゴンに新しい点を追加する"""
        if not hasattr(self.main_window, 'segmentation_annotations'):
            return
            
        current_index = self.main_window.current_index
        if current_index not in self.main_window.segmentation_annotations:
            return
            
        segmentations = self.main_window.segmentation_annotations[current_index]
        if polygon_index >= len(segmentations):
            return
            
        seg_data = segmentations[polygon_index]
        points = seg_data['points']
        
        if len(points) < 3:
            return
            
        # 新しい点を挿入する最適な位置を見つける
        insert_index = self.find_best_insertion_point(points, x, y)
        
        # 新しい点を挿入
        points.insert(insert_index, (x, y))
        
        # 画面を更新
        self.update()
        
        # ステータスメッセージを表示
        if hasattr(self.main_window, 'statusBar'):
            self.main_window.statusBar().showMessage(f"ポリゴンに新しい点を追加しました (位置: {insert_index})", 3000)
    
    def find_best_insertion_point(self, points, x, y):
        """新しい点を挿入する最適な位置を見つける"""
        if len(points) < 2:
            return len(points)
            
        min_distance = float('inf')
        best_index = 1  # デフォルトで最初の辺の後に挿入
        
        # 各辺について、クリック位置との距離を計算
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]  # 次の点（最後の場合は最初の点）
            
            # 線分p1-p2とクリック位置(x,y)との距離を計算
            distance = self.point_to_line_distance(x, y, p1[0], p1[1], p2[0], p2[1])
            
            if distance < min_distance:
                min_distance = distance
                best_index = i + 1  # p1の次に挿入
                
        return best_index
    
    def point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """点(px, py)から線分(x1,y1)-(x2,y2)までの距離を計算"""
        # 線分の長さの二乗
        line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        if line_length_sq == 0:
            # 点と点の距離
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        
        # 点から線分への垂線の足のパラメータt
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
        
        # 垂線の足の座標
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        
        # 点と垂線の足の距離
        return ((px - projection_x) ** 2 + (py - projection_y) ** 2) ** 0.5

    def get_waypoint_at_position(self, pos):
        """指定位置にあるウェイポイントのインデックスを取得"""
        if not hasattr(self.main_window, 'waypoint_annotations'):
            return None

        current_index = self.main_window.current_index
        if current_index not in self.main_window.waypoint_annotations:
            return None

        waypoints = self.main_window.waypoint_annotations[current_index]
        if not waypoints:
            return None

        # 画像内の相対位置を計算
        if not self.target_rect.contains(pos):
            return None

        click_threshold = 15  # ウェイポイントをクリックできる範囲（ピクセル）

        for i, (orig_x, orig_y) in enumerate(waypoints):
            # 元の画像座標をスクリーン座標に変換
            screen_x = self.target_rect.x() + (orig_x / self.pix_width) * self.target_rect.width()
            screen_y = self.target_rect.y() + (orig_y / self.pix_height) * self.target_rect.height()

            # クリック位置との距離を計算
            distance = ((pos.x() - screen_x) ** 2 + (pos.y() - screen_y) ** 2) ** 0.5
            if distance <= click_threshold:
                return i

        return None

    def handle_waypoint_drag(self, event):
        """ウェイポイントドラッグ処理"""
        if not hasattr(self.main_window, 'waypoint_annotations'):
            return

        current_index = self.main_window.current_index
        if current_index not in self.main_window.waypoint_annotations:
            return

        waypoints = self.main_window.waypoint_annotations[current_index]
        if (self.selected_waypoint_index is None or
            self.selected_waypoint_index >= len(waypoints)):
            return

        # 新しい位置を取得
        pos = event.pos()

        # 画像内かチェック
        if not self.target_rect.contains(pos):
            return

        # X座標のみ更新（Y座標は固定）
        rel_x = (pos.x() - self.target_rect.x()) / self.target_rect.width()
        new_x = int(rel_x * self.pix_width)

        # 画像境界内にクランプ
        new_x = max(0, min(new_x, self.pix_width - 1))

        # ウェイポイントのX座標を更新（Y座標は維持）
        old_x, old_y = waypoints[self.selected_waypoint_index]
        waypoints[self.selected_waypoint_index] = (new_x, old_y)

        # 画面を更新
        self.update()

    def handle_waypoint_hover(self, event):
        """ウェイポイントホバー検出処理"""
        pos = event.pos()

        # ホバー中のウェイポイントを検出
        hovered_waypoint = self.get_waypoint_at_position(pos)

        # ホバー状態が変わった場合のみ更新
        if hovered_waypoint != self.hovering_waypoint_index:
            self.hovering_waypoint_index = hovered_waypoint

            # カーソルを変更
            if self.hovering_waypoint_index is not None:
                self.setCursor(Qt.OpenHandCursor)  # ドラッグ可能を示すカーソル
            else:
                self.setCursor(Qt.ArrowCursor)  # 通常のカーソル

            # 画面を更新
            self.update()

    def handle_waypoint_drawing(self, event):
        """一筆書きウェイポイント描画処理"""
        pos = event.pos()

        # 画像内かチェック
        if not self.target_rect.contains(pos):
            return

        # 軌跡に追加
        self.drawing_waypoint_path.append(pos)

        # 設定されたY座標ラインとの交差を検出
        self.detect_waypoint_intersections()

        # 画面を更新
        self.update()

    def detect_waypoint_intersections(self):
        """描画軌跡とガイドラインの交差を検出してウェイポイントを配置"""
        if not hasattr(self.main_window, 'waypoint_count_spin'):
            return

        # 設定を取得
        count = self.main_window.waypoint_count_spin.value()
        start_y = self.main_window.waypoint_start_y_spin.value()
        end_y = self.main_window.waypoint_end_y_spin.value()

        # 現在のウェイポイント情報
        current_index = self.main_window.current_index
        if current_index not in self.main_window.waypoint_annotations:
            self.main_window.waypoint_annotations[current_index] = []

        current_waypoints = self.main_window.waypoint_annotations[current_index]

        # 軌跡の最後の線分を取得
        if len(self.drawing_waypoint_path) < 2:
            return

        last_point = self.drawing_waypoint_path[-2]
        current_point = self.drawing_waypoint_path[-1]

        # 各ガイドラインとの交差をチェック
        for i in range(count):
            if i >= len(current_waypoints):  # まだ配置されていないウェイポイント
                # Y座標を計算
                if count == 1:
                    y = (start_y + end_y) / 2
                else:
                    y = start_y + (end_y - start_y) * i / (count - 1)

                # スクリーン座標に変換
                screen_y = self.target_rect.y() + (y / self.pix_height) * self.target_rect.height()

                # 線分とガイドラインの交差をチェック
                if self.check_line_intersection(last_point, current_point, screen_y):
                    # 交差点のX座標を計算
                    intersection_x = self.calculate_intersection_x(last_point, current_point, screen_y)

                    # 画像座標に変換
                    rel_x = (intersection_x - self.target_rect.x()) / self.target_rect.width()
                    orig_x = int(rel_x * self.pix_width)
                    orig_x = max(0, min(orig_x, self.pix_width - 1))

                    # ウェイポイントを追加
                    current_waypoints.append((orig_x, int(y)))

    def check_line_intersection(self, p1, p2, screen_y):
        """線分がY座標ラインと交差するかチェック"""
        y1, y2 = p1.y(), p2.y()
        return (y1 <= screen_y <= y2) or (y2 <= screen_y <= y1)

    def calculate_intersection_x(self, p1, p2, screen_y):
        """線分とY座標ラインの交差点のX座標を計算"""
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()

        if y2 == y1:  # 水平線の場合
            return x1

        # 線形補間で交差点のX座標を計算
        t = (screen_y - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)

    def check_waypoint_completion_and_advance(self, current_index, target_count):
        """ウェイポイント配置完了をチェックして自動遷移を実行"""
        if not hasattr(self.main_window, 'auto_advance_waypoint'):
            return

        # 自動遷移が有効でない場合は何もしない
        if not self.main_window.auto_advance_waypoint:
            return

        # 現在のウェイポイント数をチェック
        if (current_index in self.main_window.waypoint_annotations and
            len(self.main_window.waypoint_annotations[current_index]) >= target_count):

            # ステータスメッセージ
            if hasattr(self.main_window, 'statusBar'):
                self.main_window.statusBar().showMessage(f"waypoint配置完了 ({target_count}個) - 次の画像に自動遷移", 2000)

            # 少し遅延させて次の画像に遷移（スキップ設定を考慮）
            def advance_with_skip():
                if hasattr(self.main_window, 'skip_images_on_click') and self.main_window.skip_images_on_click.isChecked():
                    skip_count = self.main_window.skip_count_spin.value()
                    self.main_window.skip_images(skip_count)
                else:
                    self.main_window.skip_images(1)

            QTimer.singleShot(500, advance_with_skip)

    #　paintEventはリファクタリング済 ~
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

        self.draw_initial_frame(painter)

        # CAMオーバーレイを描画（ベース画像の上、他のアノテーションの下）
        self.draw_gradcam_overlay(painter, self.target_rect)

        # 各機能毎に描画（描画順序を調整）
        self.draw_grid(painter, self.target_rect)
        self.draw_background_frame(painter, self.target_rect)
        # バウンディングボックス（YOLO推論結果含む）を先に描画
        self.draw_bbox(self.pix_width, self.pix_height, painter, self.target_rect)
        self.draw_segmentation(self.pix_width, self.pix_height, painter, self.target_rect)
        self.draw_waypoints(self.pix_width, self.pix_height, painter, self.target_rect)
        # アノテーション点、推論点、差分ベクトルを最後に描画（上に表示）
        self.draw_control_points(painter, self.target_rect)

        # セグメンテーション走行方向矢印を描画
        self.draw_seg_driving_direction(self.pix_width, self.pix_height, painter, self.target_rect)

        # Speedバーを描画（画像の右側）
        self.draw_speed_bar(self.pix_width, self.pix_height, painter, self.target_rect)

        # マウス座標表示（自動運転モードのみ、最後に描画して常に最前面に）
        if self.main_window and self.main_window.current_mode == 0:
            self.draw_mouse_coordinates(painter)

        painter.end()

    def draw_initial_frame(self, painter):
        # 元の画像のサイズ
        self.pix_width = self.pixmap().width()
        self.pix_height = self.pixmap().height()
        
        # ズーム係数を使用して拡大後のサイズを計算
        self.scaled_width = int(self.pix_width * self.zoom_factor)
        self.scaled_height = int(self.pix_height * self.zoom_factor)
        
        # 中央に配置するための座標計算
        self.x = (self.width() - self.scaled_width) // 2
        self.y = (self.height() - self.scaled_height) // 2

        # 画像を拡大して描画
        self.target_rect = QRect(self.x, self.y, self.scaled_width, self.scaled_height)
        painter.drawPixmap(self.target_rect, self.pixmap())

    def draw_gradcam_overlay(self, painter, target_rect):
        """CAMヒートマップオーバーレイを描画"""
        if not self.show_gradcam or self.gradcam_overlay is None:
            return

        # CAMオーバーレイをtarget_rectにスケーリングして描画
        # アルファチャンネルは既に画像に含まれているのでそのまま描画
        painter.drawPixmap(target_rect, self.gradcam_overlay)

        # CAM手法と対象の表示ラベル
        method_name = getattr(self, 'gradcam_method', 'gradcam').upper()
        label_text = f"{method_name}: {self.gradcam_target}"
        painter.setPen(QPen(Qt.white, 2))
        painter.setFont(QFont("Arial", 10, QFont.Bold))

        # ラベル背景
        text_rect = painter.fontMetrics().boundingRect(label_text)
        bg_rect = QRect(
            target_rect.x() + 5,
            target_rect.y() + 5,
            text_rect.width() + 10,
            text_rect.height() + 6
        )
        painter.fillRect(bg_rect, QColor(0, 0, 0, 180))
        painter.drawText(bg_rect, Qt.AlignCenter, label_text)

    def draw_background_frame(self, painter, target_rect):
        """削除状態や位置番号などの背景フレームを描画する"""
        scaled_width = target_rect.width()
        scaled_height = target_rect.height()
        x = target_rect.x()
        y = target_rect.y()

        if self.is_deleted:
            painter.setPen(QPen(QColor(255, 85, 85), 6))  # 赤い枠線
            border_rect = QRect(x-6, y-6, scaled_width+12, scaled_height+12)
            painter.drawRect(border_rect)

            badge_rect = QRect(x - 100, y, 80, 40)
            painter.fillRect(badge_rect, QColor(255, 85, 85))
            painter.setPen(QPen(Qt.white, 2))
            painter.setFont(QFont("Arial", 12, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, "削除済み")

            # ダウンサンプリング対象の場合は削除済みバッジの下にDS対象バッジを表示
            if self.is_downsampled:
                ds_badge_rect = QRect(x - 100, y + 45, 80, 40)
                painter.fillRect(ds_badge_rect, QColor(50, 100, 255))  # 青色
                painter.setPen(QPen(Qt.white, 2))
                painter.setFont(QFont("Arial", 12, QFont.Bold))
                painter.drawText(ds_badge_rect, Qt.AlignCenter, "DS対象")

            # 削除済みの場合は半透明の赤オーバーレイを表示
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

        elif self.is_downsampled:
            # ダウンサンプリング対象のみの場合（削除済みでない）
            painter.setPen(QPen(QColor(50, 100, 255), 4))  # 青い枠線
            border_rect = QRect(x-4, y-4, scaled_width+8, scaled_height+8)
            painter.drawRect(border_rect)

            ds_badge_rect = QRect(x - 100, y, 80, 40)
            painter.fillRect(ds_badge_rect, QColor(50, 100, 255))  # 青色
            painter.setPen(QPen(Qt.white, 2))
            painter.setFont(QFont("Arial", 12, QFont.Bold))
            painter.drawText(ds_badge_rect, Qt.AlignCenter, "DS対象")

        elif self.main_window and hasattr(self.main_window, 'current_location') and self.main_window.current_location is not None:
            loc_value = self.main_window.current_location
            loc_color = get_location_color(loc_value)

            painter.setPen(QPen(loc_color, 6))
            border_rect = QRect(x-3, y-3, scaled_width+6, scaled_height+6)
            painter.drawRect(border_rect)

            badge_size = 40
            badge_rect = QRect(x - badge_size - 10, y, badge_size, badge_size)
            painter.fillRect(badge_rect, loc_color)
            painter.setPen(QPen(Qt.white, 2))
            painter.setFont(QFont("Arial", 16, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, str(loc_value))

    def draw_bbox(self, pix_width, pix_height, painter: QPainter, target_rect: QRect):
        """バウンディングボックスの描画と編集"""
        if self.main_window and hasattr(self.main_window, 'bbox_annotations'):
            current_index = self.main_window.current_index  # インデックスベースに変更
            
            # 現在のインデックスが有効かチェック
            if (current_index is not None and 
                isinstance(current_index, int) and 
                current_index in self.main_window.bbox_annotations):
                
                bboxes = self.main_window.bbox_annotations[current_index]
                
                for i, bbox in enumerate(bboxes):
                    # クラスに応じた色を設定
                    class_name = bbox.get('class', 'unknown')
                    class_colors = CLASS_COLORS
                    color = QColor(*class_colors.get(class_name, (255, 0, 0, 180)))
                    
                    # 選択またはホバーされているバウンディングボックスかどうかで線の太さを変更
                    is_selected = (hasattr(self, 'selected_bbox_index') and 
                                self.selected_bbox_index is not None and 
                                i == self.selected_bbox_index)
                    is_hovered = (hasattr(self, 'hovering_bbox_index') and 
                                self.hovering_bbox_index is not None and 
                                i == self.hovering_bbox_index)
                    
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
                    
                    # 選択されているバウンディングボックスにはリサイズ用のハンドルを表示
                    if is_selected:
                        # ハンドルのサイズを設定（より大きく、視認性向上）
                        handle_size = 8
                        painter.setBrush(QBrush(color))
                        
                        # 四隅にハンドルを描画
                        # 左上
                        painter.drawRect(QRect(x1-handle_size//2, y1-handle_size//2, handle_size, handle_size))
                        # 右上
                        painter.drawRect(QRect(x2-handle_size//2, y1-handle_size//2, handle_size, handle_size))
                        # 左下
                        painter.drawRect(QRect(x1-handle_size//2, y2-handle_size//2, handle_size, handle_size))
                        # 右下
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
            if (hasattr(self, 'is_drawing_bbox') and self.is_drawing_bbox and 
                hasattr(self, 'bbox_start') and self.bbox_start and 
                hasattr(self, 'bbox_end') and self.bbox_end):
                
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

        # 推論結果のバウンディングボックスを表示
        if (self.main_window and 
            hasattr(self.main_window, 'show_detection_inference') and 
            self.main_window.show_detection_inference and
            hasattr(self.main_window, 'detection_inference_results')):
            
            # 推論結果はパスベースで管理されている可能性があるため、パスを取得
            if (hasattr(self.main_window, 'images') and 
                hasattr(self.main_window, 'current_index') and
                self.main_window.current_index is not None and
                self.main_window.current_index < len(self.main_window.images)):
                
                current_index = self.main_window.current_index
                if current_index in self.main_window.detection_inference_results:
                    inference_bboxes = self.main_window.detection_inference_results[current_index]
                    
                    for i, bbox in enumerate(inference_bboxes):
                        # クラスに応じた色を設定 (推論結果は別の透明度で表示)
                        class_name = bbox.get('class', 'unknown')
                        class_colors = SEGMENTATION_CLASS_COLORS
                        color = QColor(*class_colors.get(class_name, (255, 0, 0, 120)))
                        
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

    def get_bbox_at_position(self, pos, target_rect, pix_width, pix_height):
        """指定した位置にあるバウンディングボックスのインデックスを取得"""
        if not (self.main_window and hasattr(self.main_window, 'bbox_annotations')):
            return None
        
        current_index = self.main_window.current_index
        if (current_index is None or 
            not isinstance(current_index, int) or 
            current_index not in self.main_window.bbox_annotations):
            return None
        
        bboxes = self.main_window.bbox_annotations[current_index]
        
        # クリック位置を正規化座標に変換
        rel_x = (pos.x() - target_rect.x()) / target_rect.width()
        rel_y = (pos.y() - target_rect.y()) / target_rect.height()
        
        # 後ろから検索（上に描画されたものを優先）
        for i in reversed(range(len(bboxes))):
            bbox = bboxes[i]
            if (bbox['x1'] <= rel_x <= bbox['x2'] and 
                bbox['y1'] <= rel_y <= bbox['y2']):
                return i
        
        return None

    def get_resize_handle_at_position(self, pos, target_rect, bbox_index):
        """指定した位置にあるリサイズハンドルの種類を取得"""
        if not (self.main_window and hasattr(self.main_window, 'bbox_annotations')):
            return None
        
        current_index = self.main_window.current_index
        if (current_index is None or 
            not isinstance(current_index, int) or 
            current_index not in self.main_window.bbox_annotations or
            bbox_index >= len(self.main_window.bbox_annotations[current_index])):
            return None
        
        bbox = self.main_window.bbox_annotations[current_index][bbox_index]
        
        # バウンディングボックスの座標を画面座標に変換
        x1 = int(target_rect.x() + bbox['x1'] * target_rect.width())
        y1 = int(target_rect.y() + bbox['y1'] * target_rect.height())
        x2 = int(target_rect.x() + bbox['x2'] * target_rect.width())
        y2 = int(target_rect.y() + bbox['y2'] * target_rect.height())
        
        handle_size = 8
        tolerance = handle_size // 2
        
        # 各ハンドルの位置をチェック
        handles = {
            'top_left': (x1, y1),
            'top_right': (x2, y1),
            'bottom_left': (x1, y2),
            'bottom_right': (x2, y2)
        }
        
        for handle_type, (hx, hy) in handles.items():
            if (abs(pos.x() - hx) <= tolerance and 
                abs(pos.y() - hy) <= tolerance):
                return handle_type
        
        return None

    def update_bbox_coordinates(self, bbox_index, new_coords):
        """バウンディングボックスの座標を更新"""
        if not (self.main_window and hasattr(self.main_window, 'bbox_annotations')):
            return False
        
        current_index = self.main_window.current_index
        if (current_index is None or 
            not isinstance(current_index, int) or 
            current_index not in self.main_window.bbox_annotations or
            bbox_index >= len(self.main_window.bbox_annotations[current_index])):
            return False
        
        # 座標を正規化してソート
        x1, y1, x2, y2 = new_coords
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        # 座標を0-1の範囲にクランプ
        min_x = max(0, min(1, min_x))
        max_x = max(0, min(1, max_x))
        min_y = max(0, min(1, min_y))
        max_y = max(0, min(1, max_y))
        
        # 最小サイズをチェック
        min_size = 0.01  # 1%の最小サイズ
        if (max_x - min_x) < min_size or (max_y - min_y) < min_size:
            return False
        
        # 座標を更新
        self.main_window.bbox_annotations[current_index][bbox_index].update({
            'x1': min_x,
            'y1': min_y,
            'x2': max_x,
            'y2': max_y
        })
        
        return True

    def delete_bbox(self, bbox_index):
        """バウンディングボックスを削除"""
        if not (self.main_window and hasattr(self.main_window, 'bbox_annotations')):
            return False
        
        current_index = self.main_window.current_index
        if (current_index is None or 
            not isinstance(current_index, int) or 
            current_index not in self.main_window.bbox_annotations or
            bbox_index >= len(self.main_window.bbox_annotations[current_index])):
            return False
        
        # バウンディングボックスを削除
        del self.main_window.bbox_annotations[current_index][bbox_index]
        
        # 選択状態をリセット
        if hasattr(self, 'selected_bbox_index'):
            if self.selected_bbox_index == bbox_index:
                self.selected_bbox_index = None
            elif self.selected_bbox_index > bbox_index:
                self.selected_bbox_index -= 1
        
        # ホバー状態をリセット
        if hasattr(self, 'hovering_bbox_index'):
            if self.hovering_bbox_index == bbox_index:
                self.hovering_bbox_index = None
            elif self.hovering_bbox_index > bbox_index:
                self.hovering_bbox_index -= 1
        
        return True


    def draw_grid(self, painter: QPainter, target_rect: QRect):
        """ズーム済み領域にグリッドと目盛りを描画"""
        if not self.show_grid:
            return

        grid_size = self.grid_size
        step_x = target_rect.width() / grid_size
        step_y = target_rect.height() / grid_size

        # グレーの薄い線で描画
        painter.setPen(QPen(QColor(100, 100, 100, 100), 1))
        for i in range(1, grid_size):
            x_pos = target_rect.x() + i * step_x
            y_pos = target_rect.y() + i * step_y
            painter.drawLine(int(x_pos), target_rect.y(), int(x_pos), target_rect.y() + target_rect.height())
            painter.drawLine(target_rect.x(), int(y_pos), target_rect.x() + target_rect.width(), int(y_pos))

        # 中央線
        painter.setPen(QPen(QColor(200, 200, 200, 150), 2))
        mid_x = target_rect.x() + target_rect.width() // 2
        mid_y = target_rect.y() + target_rect.height() // 2
        painter.drawLine(mid_x, target_rect.y(), mid_x, target_rect.y() + target_rect.height())
        painter.drawLine(target_rect.x(), mid_y, target_rect.x() + target_rect.width(), mid_y)

        # 目盛り表示
        painter.setFont(QFont("Arial", 10))
        # ダークモードの場合は明るい色、ライトモードの場合は暗い色を使用
        is_dark = self.main_window.is_dark_mode if self.main_window else False
        text_color = QColor(200, 200, 200, 200) if is_dark else QColor(80, 80, 80, 200)
        painter.setPen(QPen(text_color, 1))
        painter.drawText(target_rect.x() - 25, target_rect.y() - 5, "-1")
        painter.drawText(target_rect.x() + target_rect.width() + 5, target_rect.y() - 5, "1")

        for i in range(1, grid_size):
            value = -1 + (2.0 * i / grid_size)
            x_pos = target_rect.x() + i * step_x
            if abs(value) < 0.1:
                painter.drawText(int(x_pos) - 5, target_rect.y() - 5, "0")
            elif i % 2 == 0:
                painter.drawText(int(x_pos) - 15, target_rect.y() - 5, f"{value:.1f}")

        painter.drawText(target_rect.x() - 35, target_rect.y() + 15, "1")
        painter.drawText(target_rect.x() - 35, target_rect.y() + target_rect.height(), "-1")

        for i in range(1, grid_size):
            value = 1 - (2.0 * i / grid_size)
            y_pos = target_rect.y() + i * step_y
            if abs(value) < 0.1:
                painter.drawText(target_rect.x() - 35, int(y_pos) + 5, "0")
            elif i % 2 == 0:
                painter.drawText(target_rect.x() - 35, int(y_pos) + 5, f"{value:.1f}")

    def draw_control_points(self, painter: QPainter, target_rect: QRect):
        """アノテーション点、推論点、ベクトル矢印を描画"""
        if not self.pixmap():
            return

        pix_width = self.pixmap().width()
        pix_height = self.pixmap().height()

        # 赤：アノテーション点（教師データ）- 現在のフレーム（最初に描画）
        if self.annotation_point:
            rel_x = self.annotation_point.x() / pix_width
            rel_y = self.annotation_point.y() / pix_height
            scaled_x = int(target_rect.x() + rel_x * target_rect.width())
            scaled_y = int(target_rect.y() + rel_y * target_rect.height())

            painter.setPen(QPen(QColor(255, 0, 0), 4))
            painter.setBrush(QBrush())  # 塗りつぶしなし（透明）
            painter.drawEllipse(scaled_x - 15, scaled_y - 15, 30, 30)

        # 将来のアノテーション点を描画（5, 10フレーム先）- 現在の点の上に描画
        if self.show_future_annotations and self.main_window and hasattr(self.main_window, 'annotations'):
            current_index = self.main_window.current_index
            future_offsets = [10, 5]  # 先に遠い方を描画（後に描画されるものが上に来る）
            future_sizes = {5: 22, 10: 14}  # インデックスごとのサイズ（現在は30）
            future_colors = {5: QColor(255, 165, 0), 10: QColor(200, 180, 0)}  # 5:オレンジ, 10:濃い黄色

            for offset in future_offsets:
                future_index = current_index + offset
                if future_index in self.main_window.annotations:
                    future_ann = self.main_window.annotations[future_index]
                    if 'x' in future_ann and 'y' in future_ann:
                        # 将来のアノテーション座標を取得（ピクセル座標）
                        future_x = future_ann['x']
                        future_y = future_ann['y']
                        # ピクセル座標から相対座標に変換
                        future_rel_x = future_x / pix_width
                        future_rel_y = future_y / pix_height
                        future_scaled_x = int(target_rect.x() + future_rel_x * target_rect.width())
                        future_scaled_y = int(target_rect.y() + future_rel_y * target_rect.height())

                        # サイズと色を取得
                        size = future_sizes.get(offset, 20)
                        color = future_colors.get(offset, QColor(255, 165, 0))

                        # オレンジ/黄色で描画
                        painter.setPen(QPen(color, 3))
                        painter.setBrush(QBrush())  # 塗りつぶしなし
                        painter.drawEllipse(future_scaled_x - size // 2, future_scaled_y - size // 2, size, size)

        # 将来の推論点を描画（t+5, t+10）- 現在の推論点の下に描画
        if self.show_inference and self.show_future_annotations and self.main_window:
            current_index = self.main_window.current_index
            if hasattr(self.main_window, 'inference_results') and current_index in self.main_window.inference_results:
                inference_data = self.main_window.inference_results[current_index]
                future_offsets = [10, 5]  # 先に遠い方を描画（後に描画されるものが上に来る）
                future_sizes = {5: 22, 10: 14}  # インデックスごとのサイズ（現在は30）
                # シアン系の色（アノテーションのオレンジ/黄色に対応）
                future_colors = {5: QColor(0, 200, 200), 10: QColor(0, 150, 150)}  # 5:明るいシアン, 10:暗いシアン

                for offset in future_offsets:
                    future_key = f"future_{offset}"
                    if future_key in inference_data:
                        future_data = inference_data[future_key]
                        future_x = future_data['x']
                        future_y = future_data['y']
                        # ピクセル座標から相対座標に変換
                        future_rel_x = future_x / pix_width
                        future_rel_y = future_y / pix_height
                        future_scaled_x = int(target_rect.x() + future_rel_x * target_rect.width())
                        future_scaled_y = int(target_rect.y() + future_rel_y * target_rect.height())

                        # サイズと色を取得
                        size = future_sizes.get(offset, 20)
                        color = future_colors.get(offset, QColor(0, 200, 200))

                        # シアン系の色で描画
                        painter.setPen(QPen(color, 3))
                        painter.setBrush(QBrush())  # 塗りつぶしなし
                        painter.drawEllipse(future_scaled_x - size // 2, future_scaled_y - size // 2, size, size)

        # 明るい水色：推論点（推論結果）
        if self.show_inference and self.inference_point:
            rel_x = self.inference_point.x() / pix_width
            rel_y = self.inference_point.y() / pix_height
            scaled_x = int(target_rect.x() + rel_x * target_rect.width())
            scaled_y = int(target_rect.y() + rel_y * target_rect.height())

            painter.setPen(QPen(QColor(0, 255, 255), 4))  # 明るい水色(cyan)
            painter.setBrush(QBrush())  # 塗りつぶしなし（透明）
            painter.drawEllipse(scaled_x - 15, scaled_y - 15, 30, 30)

            # 緑のベクトル：教師 → 推論
            if (
                self.annotation_point and
                hasattr(self.main_window, 'inference_diff_vectors') and
                hasattr(self.main_window, 'show_diff_vectors') and
                self.main_window.show_diff_vectors
            ):
                current_index = self.main_window.current_index
                if current_index in self.main_window.inference_diff_vectors:
                    anno_rel_x = self.annotation_point.x() / pix_width
                    anno_rel_y = self.annotation_point.y() / pix_height
                    anno_scaled_x = int(target_rect.x() + anno_rel_x * target_rect.width())
                    anno_scaled_y = int(target_rect.y() + anno_rel_y * target_rect.height())

                    self.draw_vector_arrow(painter, anno_scaled_x, anno_scaled_y, scaled_x, scaled_y)


    def draw_vector_arrow(self, painter, start_x, start_y, end_x, end_y):
        """教師データから推論結果への矢印を描画する"""
        # 矢印の色とスタイル設定
        painter.setPen(QPen(QColor(0, 255, 0), 2))  # 緑色
        painter.setBrush(QBrush(QColor(0, 255, 0)))

        # 矢印の線を描画
        painter.drawLine(start_x, start_y, end_x, end_y)

        # 矢印の線を描画
        painter.drawLine(start_x, start_y, end_x, end_y)
                
        # ベクトルの角度を計算
        dx = end_x - start_x
        dy = end_y - start_y
        
        # 矢印が短すぎる場合は描画しない
        vector_length = math.sqrt(dx*dx + dy*dy)
        if vector_length < 5:
            return
            
        angle = math.atan2(dy, dx)
        
        # 矢印の先端のサイズ
        arrow_length = 10
        arrow_angle = math.pi / 6  # 30度
        
        # 矢印の先端の座標を計算
        arrow_x1 = end_x - arrow_length * math.cos(angle - arrow_angle)
        arrow_y1 = end_y - arrow_length * math.sin(angle - arrow_angle)
        arrow_x2 = end_x - arrow_length * math.cos(angle + arrow_angle)
        arrow_y2 = end_y - arrow_length * math.sin(angle + arrow_angle)
        
        # 矢印の先端を描画
        arrow_points = [
            QPoint(int(end_x), int(end_y)),
            QPoint(int(arrow_x1), int(arrow_y1)),
            QPoint(int(arrow_x2), int(arrow_y2))
        ]
        
        arrow_polygon = QPolygon(arrow_points)
        painter.drawPolygon(arrow_polygon)

    def draw_mouse_coordinates(self, painter: QPainter):
        """マウスカーソルの右上に正規化座標を表示"""
        if self.current_mouse_pos is None or self.normalized_coords is None:
            return

        x_norm, y_norm = self.normalized_coords
        pos = self.current_mouse_pos

        # painterの状態をリセット（セグメンテーション描画の影響を受けないように）
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setBrush(Qt.NoBrush)
        painter.setOpacity(1.0)

        # 表示テキストを作成
        text_lines = [
            f"x: {x_norm:+.3f}",
            f"y: {y_norm:+.3f}"
        ]

        # フォントとサイズ設定
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        # テキストの幅と高さを計算
        max_width = max(metrics.horizontalAdvance(line) for line in text_lines)
        line_height = metrics.height()
        total_height = line_height * len(text_lines)

        # 表示位置（カーソルの右上）
        offset_x = 15
        offset_y = -10
        text_x = pos.x() + offset_x
        text_y = pos.y() + offset_y

        # ウィンドウの端を超えないように調整
        if text_x + max_width + 10 > self.width():
            text_x = pos.x() - offset_x - max_width - 10  # 左側に表示
        if text_y - total_height - 10 < 0:
            text_y = pos.y() + offset_y + total_height + 20  # 下側に表示

        # 背景矩形を描画（半透明の黒背景）
        padding = 5
        bg_rect = QRect(
            text_x - padding,
            text_y - total_height - padding,
            max_width + padding * 2,
            total_height + padding * 2
        )
        bg_color = QColor(0, 0, 0, 180)
        painter.fillRect(bg_rect, bg_color)

        # 枠線を描画
        painter.setPen(QPen(QColor(100, 200, 255), 2))
        painter.drawRect(bg_rect)

        # テキストを描画
        painter.setPen(QPen(Qt.white))
        for i, line in enumerate(text_lines):
            y_position = text_y - total_height + line_height * (i + 1) - 3
            painter.drawText(text_x, y_position, line)

    def draw_seg_driving_direction(self, pix_width, pix_height, painter: QPainter, target_rect: QRect):
        """セグメンテーション推論結果から計算した走行軌跡（円弧）またはウェイポイントを描画"""
        if not hasattr(self.main_window, 'show_seg_driving_direction'):
            return

        if not self.main_window.show_seg_driving_direction:
            return

        current_index = self.main_window.current_index
        if current_index is None:
            return

        # Y座標の設定値を取得
        target_y_image = self.main_window.seg_driving_direction_y

        # Y座標のガイドラインを常に描画（太い黄色の点線）
        screen_target_y = target_rect.y() + (target_y_image / pix_height) * target_rect.height()
        painter.setPen(QPen(QColor(255, 255, 0, 150), 3, Qt.DashLine))
        painter.drawLine(target_rect.x(), int(screen_target_y),
                        target_rect.x() + target_rect.width(), int(screen_target_y))

        # 走行方向の座標を計算
        direction_coord = self.main_window.calculate_seg_driving_direction(current_index)
        if direction_coord is None:
            # 推論結果がない場合はY軸の点線だけ表示して終了
            return

        target_x, target_y = direction_coord

        # 画像座標からスクリーン座標に変換
        screen_target_x = target_rect.x() + (target_x / pix_width) * target_rect.width()
        screen_target_y = target_rect.y() + (target_y / pix_height) * target_rect.height()

        # 開始位置（画像下部中央）
        start_x = target_rect.x() + target_rect.width() // 2
        start_y = target_rect.y() + target_rect.height()

        # painterの状態をリセット
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setOpacity(1.0)

        # 表示モードに応じて描画
        display_mode = getattr(self.main_window, 'seg_display_mode', 'trajectory')

        if display_mode == 'waypoint':
            # ウェイポイントモード：画像下部中央から目標Y座標まで4点を等間隔配置
            # 画像下部のY座標
            start_y_image = pix_height
            # 目標Y座標
            end_y_image = target_y

            # 4点を等間隔配置（開始点と終了点を含む）
            waypoint_count = 4
            waypoints = []
            for i in range(waypoint_count):
                # Y座標を等間隔で計算
                ratio = i / (waypoint_count - 1)
                wp_y_image = start_y_image - (start_y_image - end_y_image) * ratio

                # X座標を計算
                if i == 0:
                    # 最初の点：画像下端中央
                    wp_x_image = pix_width / 2
                else:
                    # 中間点と最終点：各Y座標におけるセグメンテーションエリアの中央値
                    wp_x = self.main_window.calculate_seg_x_at_y(current_index, wp_y_image)
                    if wp_x is None:
                        # セグメンテーションエリアがない場合は画像中央を使用
                        wp_x_image = pix_width / 2
                    else:
                        wp_x_image = wp_x

                # スクリーン座標に変換
                wp_screen_x = target_rect.x() + (wp_x_image / pix_width) * target_rect.width()
                wp_screen_y = target_rect.y() + (wp_y_image / pix_height) * target_rect.height()

                waypoints.append((wp_screen_x, wp_screen_y, wp_x_image, wp_y_image))

            # ウェイポイント間を線で結ぶ
            painter.setPen(QPen(QColor(0, 255, 0), 3))
            for i in range(len(waypoints) - 1):
                x1, y1, _, _ = waypoints[i]
                x2, y2, _, _ = waypoints[i + 1]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            # ウェイポイントを描画
            for i, (wp_x, wp_y, img_x, img_y) in enumerate(waypoints):
                # ウェイポイントマーカー（緑色の丸）
                painter.setPen(QPen(QColor(0, 200, 0), 2))
                painter.setBrush(QBrush(QColor(0, 255, 0, 180)))
                painter.drawEllipse(int(wp_x - 6), int(wp_y - 6), 12, 12)

                # すべてのウェイポイントに座標情報を表示（正規化: -1～1）
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.setFont(QFont("Arial", 9, QFont.Bold))
                # 座標を正規化: X軸は中央を0、左端を-1、右端を1、Y軸は下端を-1、上端を1
                norm_x = (img_x - pix_width / 2) / (pix_width / 2)
                norm_y = (pix_height - img_y) / (pix_height / 2) - 1
                info_text = f"({norm_x:.2f}, {norm_y:.2f})"
                painter.drawText(int(wp_x + 10), int(wp_y - 5), info_text)

        else:
            # 軌跡モード：従来の円弧描画
            # 円弧パラメータを計算（スクリーン座標系で）
            arc_params = self.main_window.calculate_steering_arc_params(
                start_x, start_y,
                screen_target_x, screen_target_y,
                self.main_window.seg_max_steering_angle
            )

            if arc_params is None:
                # 直線の場合（舵角が小さい）
                painter.setPen(QPen(QColor(0, 255, 0), 4))
                painter.drawLine(int(start_x), int(start_y), int(screen_target_x), int(screen_target_y))
            else:
                # 円弧を描画
                center_x = arc_params['center_x']
                center_y = arc_params['center_y']
                radius = arc_params['radius']
                start_angle = arc_params['start_angle']
                end_angle = arc_params['end_angle']
                direction = arc_params['direction']

                # QPainterのdrawArcは角度を1/16度単位で指定
                # また、0度は3時方向、反時計回りが正
                # atan2の角度をQt用に変換
                qt_start_angle = -math.degrees(start_angle) * 16

                # 角度差を計算
                angle_diff = end_angle - start_angle

                # 角度差を-π～πの範囲に正規化
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi

                # directionに応じて描画方向を決定
                # direction > 0 (右旋回): 時計回り（負のspan）
                # direction < 0 (左旋回): 反時計回り（正のspan）
                qt_span_angle = -math.degrees(angle_diff) * 16

                print(f"[円弧描画] start={math.degrees(start_angle):.1f}°, end={math.degrees(end_angle):.1f}°")
                print(f"[円弧描画] angle_diff={math.degrees(angle_diff):.1f}°, direction={direction}")
                print(f"[円弧描画] qt_start={qt_start_angle/16:.1f}°, qt_span={qt_span_angle/16:.1f}°")

                # 円弧の外接矩形を計算
                arc_rect = QRect(
                    int(center_x - radius),
                    int(center_y - radius),
                    int(radius * 2),
                    int(radius * 2)
                )

                # 円弧を描画（太い緑色の線）
                painter.setPen(QPen(QColor(0, 255, 0), 4))
                painter.setBrush(Qt.NoBrush)
                painter.drawArc(arc_rect, int(qt_start_angle), int(qt_span_angle))

            # 目標点にマーカーを描画（濃い黄色）
            painter.setPen(QPen(QColor(200, 200, 0), 2))
            painter.setBrush(QBrush(QColor(200, 200, 0, 180)))
            painter.drawEllipse(int(screen_target_x - 8), int(screen_target_y - 8), 16, 16)

            # 座標と舵角情報を表示（正規化: -1～1）
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            # 座標を正規化: X軸は中央を0、左端を-1、右端を1、Y軸は下端を-1、上端を1
            norm_x = (target_x - pix_width / 2) / (pix_width / 2)
            norm_y = (pix_height - target_y) / (pix_height / 2) - 1
            if arc_params:
                info_text = f"({norm_x:.2f}, {norm_y:.2f}) {arc_params['actual_steering_deg']:.1f}°"
            else:
                info_text = f"({norm_x:.2f}, {norm_y:.2f}) 直進"
            painter.drawText(int(screen_target_x + 12), int(screen_target_y - 5), info_text)

    def draw_segmentation_inference_results(self, pix_width, pix_height, painter: QPainter, target_rect: QRect):
        """セグメンテーション推論結果の描画"""
        if not (hasattr(self.main_window, 'segmentation_inference_results') and 
                hasattr(self.main_window, 'show_segmentation_inference')):
            return
        
        # セグメンテーション推論表示がOFFの場合は描画しない
        show_seg_inference = getattr(self.main_window, 'show_segmentation_inference', False)
        if not show_seg_inference:
            return
        
        # 現在の画像の推論結果を取得
        if self.main_window.images and self.main_window.current_index < len(self.main_window.images):
            current_img_path = self.main_window.images[self.main_window.current_index]
            
            if (current_img_path in self.main_window.segmentation_inference_results and
                self.main_window.segmentation_inference_results[current_img_path]):
                
                result = self.main_window.segmentation_inference_results[current_img_path]
                segments = result.get('segments', [])
                
                # 各セグメンテーションを描画
                for segment in segments:
                    class_name = segment['class']
                    points = segment['points']
                    confidence = segment.get('confidence', 0.0)
                    
                    if len(points) >= 3:
                        # 手動アノテーションと同じ色定義を使用
                        class_colors = SEGMENTATION_CLASS_COLORS
                        color_tuple = class_colors.get(class_name, (128, 128, 128, 120))
                        
                        # 推論結果は少し透明度を低くして手動アノテーションと区別（透明度80）
                        base_color = QColor(color_tuple[0], color_tuple[1], color_tuple[2], 80)
                        
                        # 推論結果は点線で描画して区別
                        pen = QPen(base_color.darker(), 2, Qt.DashLine)
                        painter.setPen(pen)
                        painter.setBrush(QBrush(base_color))
                        
                        # ポリゴンの描画
                        # 推論結果のpointsは正規化座標（0-1）なので直接画面座標に変換
                        polygon_points = []
                        for px, py in points:
                            # 正規化座標を画面座標に変換（バウンディングボックスと同じ方式）
                            screen_x = int(target_rect.x() + px * target_rect.width())
                            screen_y = int(target_rect.y() + py * target_rect.height())
                            polygon_points.append(QPoint(screen_x, screen_y))
                        
                        # 塗りつぶし
                        painter.drawPolygon(polygon_points)
                        
                        # クラス名と信頼度を表示（推論結果のラベル）
                        if polygon_points:
                            # ポリゴンの重心を計算
                            center_x = sum(p.x() for p in polygon_points) // len(polygon_points)
                            center_y = sum(p.y() for p in polygon_points) // len(polygon_points)
                            
                            # ラベルテキスト
                            label = f"{class_name} ({confidence:.2f})"
                            painter.setFont(QFont("Arial", 8))
                            
                            # ラベル背景
                            text_width = painter.fontMetrics().horizontalAdvance(label)
                            text_height = painter.fontMetrics().height()
                            
                            # 背景矩形を描画
                            bg_rect = QRect(center_x - text_width//2 - 2, center_y - text_height//2 - 2,
                                          text_width + 4, text_height + 4)
                            # base_colorを少し濃くして背景に使用
                            bg_color = base_color.darker()
                            bg_color.setAlpha(180)
                            painter.fillRect(bg_rect, bg_color)
                            
                            # テキストを描画
                            painter.setPen(QPen(Qt.white))
                            painter.drawText(center_x - text_width//2, center_y + text_height//4, label)

    ###
    def draw_segmentation(self, pix_width, pix_height, painter: QPainter, target_rect: QRect):
        """セグメンテーションポリゴンの描画と編集（手動アノテーション + 推論結果）"""
        
        # 1. 推論結果のセグメンテーションを描画
        self.draw_segmentation_inference_results(pix_width, pix_height, painter, target_rect)
        
        # 2. 手動アノテーションのセグメンテーションを描画
        if hasattr(self.main_window, 'segmentation_annotations'):
            current_index = self.main_window.current_index  # インデックスベースに変更
            if current_index in self.main_window.segmentation_annotations:
                polygons = self.main_window.segmentation_annotations[current_index]
                
                for i, polygon_data in enumerate(polygons):
                    if polygon_data is None:  # Noneの場合はスキップ
                        continue
                    class_name = polygon_data.get('class', 'unknown')
                    points = polygon_data.get('points', [])
                    
                    if len(points) >= 3:
                        # クラスに応じた色を設定
                        class_colors = SEGMENTATION_CLASS_COLORS
                        base_color = QColor(*class_colors.get(class_name, (255, 0, 0, 120)))


                        # 選択またはホバーされているセグメンテーションの強調表示
                        is_selected = i == self.selected_segmentation_index
                        is_hovered = i == self.hovering_segmentation_index
                        
                        if is_selected:
                            # 選択時は濃い色で縁取り
                            painter.setPen(QPen(base_color.darker(), 4))
                            painter.setBrush(QBrush(QColor(base_color.red(), base_color.green(), base_color.blue(), 150)))
                        elif is_hovered:
                            # ホバー時は少し濃い色
                            painter.setPen(QPen(base_color.darker(), 3))
                            painter.setBrush(QBrush(QColor(base_color.red(), base_color.green(), base_color.blue(), 100)))
                        else:
                            # 通常時
                            painter.setPen(QPen(base_color.darker(), 2))
                            painter.setBrush(QBrush(base_color))
                        
                        # ポリゴンの描画
                        polygon_points = []
                        for px, py in points:
                            screen_x = int(target_rect.x() + (px / pix_width) * target_rect.width())
                            screen_y = int(target_rect.y() + (py / pix_height) * target_rect.height())
                            polygon_points.append(QPoint(screen_x, screen_y))
                        
                        # 塗りつぶし
                        painter.drawPolygon(polygon_points)
                        
                        # 選択されているセグメンテーションには頂点を表示 - ここに移動
                        if is_selected:
                            for vertex_index, point in enumerate(polygon_points):
                                # 頂点の状態に応じて色とサイズを変更
                                if (self.selected_polygon_index == i and 
                                    self.selected_vertex_index == vertex_index):
                                    # 選択された頂点（編集中）
                                    painter.setBrush(QBrush(QColor(255, 255, 0)))  # 黄色
                                    painter.setPen(QPen(Qt.black, 3))
                                    painter.drawEllipse(point.x() - 6, point.y() - 6, 12, 12)
                                elif (self.hovering_polygon_index == i and 
                                    self.hovering_vertex_index == vertex_index):
                                    # ホバー中の頂点
                                    painter.setBrush(QBrush(QColor(255, 165, 0)))  # オレンジ色
                                    painter.setPen(QPen(base_color.darker(), 2))
                                    painter.drawEllipse(point.x() - 5, point.y() - 5, 10, 10)  # 少し大きく
                                else:
                                    # 通常の頂点
                                    painter.setBrush(QBrush(Qt.white))
                                    painter.setPen(QPen(base_color.darker(), 2))
                                    painter.drawEllipse(point.x() - 4, point.y() - 4, 8, 8)
                        
                        # ラベル表示
                        if polygon_points:
                            center_x = sum(p.x() for p in polygon_points) // len(polygon_points)
                            center_y = sum(p.y() for p in polygon_points) // len(polygon_points)
                            painter.setPen(QPen(Qt.white, 1))
                            painter.setFont(QFont("Arial", 10, QFont.Bold))
                            
                            # ラベル背景
                            text_width = painter.fontMetrics().horizontalAdvance(class_name)
                            painter.fillRect(center_x - text_width//2 - 2, center_y - 10, text_width + 4, 16, base_color.darker())
                            
                            painter.drawText(center_x - text_width//2, center_y + 2, class_name)

        # 現在描画中のポリゴンの表示（修正）
        if self.is_drawing_segmentation and len(self.current_segmentation_polygon) > 0:
            painter.setPen(QPen(QColor(255, 255, 0), 3))
            
            # 点を線で結ぶ
            screen_points = []
            for point in self.current_segmentation_polygon:
                screen_x = int(target_rect.x() + (point.x() / pix_width) * target_rect.width())
                screen_y = int(target_rect.y() + (point.y() / pix_height) * target_rect.height())
                screen_points.append(QPoint(screen_x, screen_y))
            
            # 線を描画
            for i in range(len(screen_points)):
                # 点を描画
                painter.setBrush(QBrush(QColor(255, 255, 0)))
                painter.drawEllipse(screen_points[i].x() - 4, screen_points[i].y() - 4, 8, 8)
                
                if i < len(screen_points) - 1:
                    # 線を描画
                    painter.drawLine(screen_points[i], screen_points[i + 1])
            
            # 最初の点と最後の点を点線で結ぶ（閉じる候補を表示）
            if len(screen_points) >= 3:
                painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
                painter.drawLine(screen_points[-1], screen_points[0])
                
                # 最初の点を強調表示
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(QColor(255, 255, 0), 3))
                painter.drawEllipse(screen_points[0].x() - 6, screen_points[0].y() - 6, 12, 12)

    def draw_waypoints(self, pix_width, pix_height, painter: QPainter, target_rect: QRect):
        """waypointの描画（緑色の丸）とY軸ガイドライン"""

        # waypointモードでY軸ガイドラインを描画
        if (hasattr(self.main_window, 'current_mode') and
            self.main_window.current_mode == 3 and
            hasattr(self.main_window, 'waypoint_control_widget') and
            self.main_window.waypoint_control_widget.isVisible()):

            self.draw_waypoint_guidelines(pix_width, pix_height, painter, target_rect)

        # 既存のwaypoint描画
        current_index = self.main_window.current_index

        # アノテーションwaypointがある場合のみ描画
        if (hasattr(self.main_window, 'waypoint_annotations') and
            current_index in self.main_window.waypoint_annotations):

            waypoints = self.main_window.waypoint_annotations[current_index]
            if not waypoints:
                waypoints = None
        else:
            waypoints = None

        # waypointsがある場合のみアノテーション描画
        if waypoints:
            # 緑色で描画設定
            painter.setBrush(QBrush(QColor(0, 255, 0, 180)))  # 半透明の緑
            painter.setPen(QPen(QColor(0, 128, 0), 2))  # 濃い緑の境界線

            # waypoint間を点線で繋ぐ
            if len(waypoints) > 1:
                painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
                for i in range(len(waypoints) - 1):
                    current_x, current_y = waypoints[i]
                    next_x, next_y = waypoints[i + 1]

                    # 座標をスクリーン座標に変換
                    screen_x1 = target_rect.x() + (current_x / pix_width) * target_rect.width()
                    screen_y1 = target_rect.y() + (current_y / pix_height) * target_rect.height()
                    screen_x2 = target_rect.x() + (next_x / pix_width) * target_rect.width()
                    screen_y2 = target_rect.y() + (next_y / pix_height) * target_rect.height()

                    # 点線を描画
                    painter.drawLine(int(screen_x1), int(screen_y1), int(screen_x2), int(screen_y2))

            # 各waypointを描画
            for i, (orig_x, orig_y) in enumerate(waypoints):
                # 元の画像座標をスクリーン座標に変換
                screen_x = target_rect.x() + (orig_x / pix_width) * target_rect.width()
                screen_y = target_rect.y() + (orig_y / pix_height) * target_rect.height()

                # ホバー中のウェイポイントに外側の縁取りを追加
                if (hasattr(self, 'hovering_waypoint_index') and
                    self.hovering_waypoint_index == i):
                    # 外側にオレンジの縁取りを描画
                    painter.setBrush(QBrush(Qt.transparent))
                    painter.setPen(QPen(QColor(255, 140, 0), 3))  # オレンジの太い線
                    painter.drawEllipse(int(screen_x - 11), int(screen_y - 11), 22, 22)

                # 緑色の丸を描画（半径8ピクセル）
                painter.setBrush(QBrush(QColor(0, 255, 0, 180)))  # 半透明の緑
                painter.setPen(QPen(QColor(0, 128, 0), 2))  # 濃い緑の境界線
                painter.drawEllipse(int(screen_x - 8), int(screen_y - 8), 16, 16)

                # waypoint番号を表示
                painter.setPen(QPen(QColor(255, 255, 255), 1))  # 白文字
                painter.setFont(QFont("Arial", 10, QFont.Bold))
                painter.drawText(int(screen_x - 6), int(screen_y + 4), str(i + 1))

                # 座標(x,y)を表示
                painter.setPen(QPen(QColor(0, 200, 0), 1))  # 緑文字
                painter.setFont(QFont("Arial", 9))
                coord_text = f"({int(orig_x)},{int(orig_y)})"
                painter.drawText(int(screen_x + 12), int(screen_y - 5), coord_text)

            # 一筆書き中の軌跡を描画
            if (self.is_drawing_waypoints and
                len(self.drawing_waypoint_path) > 1):
                painter.setPen(QPen(QColor(255, 255, 0), 2))  # 黄色の線
                for i in range(len(self.drawing_waypoint_path) - 1):
                    start_point = self.drawing_waypoint_path[i]
                    end_point = self.drawing_waypoint_path[i + 1]
                    painter.drawLine(start_point, end_point)

        # 推論結果の描画
        if (self.main_window and
            hasattr(self.main_window, 'waypoint_inference_checkbox') and
            self.main_window.waypoint_inference_checkbox.isChecked() and
            hasattr(self.main_window, 'waypoint_inference_results')):

            current_index = self.main_window.current_index

            if current_index in self.main_window.waypoint_inference_results:
                inference_waypoints = self.main_window.waypoint_inference_results[current_index]

                if inference_waypoints:
                    # 推論waypoint間を点線で繋ぐ
                    if len(inference_waypoints) > 1:
                        painter.setPen(QPen(QColor(0, 255, 255), 2, Qt.DashLine))
                        for i in range(len(inference_waypoints) - 1):
                            current_x, current_y = inference_waypoints[i]
                            next_x, next_y = inference_waypoints[i + 1]

                            # 正規化座標を画面座標に変換
                            screen_x1 = target_rect.x() + current_x * target_rect.width()
                            screen_y1 = target_rect.y() + current_y * target_rect.height()
                            screen_x2 = target_rect.x() + next_x * target_rect.width()
                            screen_y2 = target_rect.y() + next_y * target_rect.height()

                            # 点線を描画
                            painter.drawLine(int(screen_x1), int(screen_y1), int(screen_x2), int(screen_y2))

                    # 各推論waypointを描画 (明るい水色)
                    for i, (wx, wy) in enumerate(inference_waypoints):
                        # 正規化座標を画面座標に変換
                        scaled_x = int(target_rect.x() + wx * target_rect.width())
                        scaled_y = int(target_rect.y() + wy * target_rect.height())

                        # 正規化座標を元画像のピクセル座標に変換
                        pixel_x = int(wx * pix_width)
                        pixel_y = int(wy * pix_height)

                        # 推論waypointを描画 (明るい水色、位置推論と同じ色)
                        painter.setBrush(QBrush(QColor(0, 255, 255, 150)))  # 半透明の水色
                        painter.setPen(QPen(QColor(0, 180, 180), 2))  # 濃い水色の境界線
                        painter.drawEllipse(scaled_x - 8, scaled_y - 8, 16, 16)

                        # waypoint番号を表示
                        painter.setPen(QPen(QColor(255, 255, 255), 1))  # 白文字
                        painter.setFont(QFont("Arial", 10, QFont.Bold))
                        painter.drawText(scaled_x - 6, scaled_y + 4, str(i + 1))

                        # ピクセル座標を表示
                        painter.setPen(QPen(QColor(0, 200, 255), 1))  # 明るい水色文字
                        painter.setFont(QFont("Arial", 9))
                        coord_text = f"({pixel_x},{pixel_y})"
                        painter.drawText(scaled_x + 12, scaled_y - 5, coord_text)

    def draw_waypoint_guidelines(self, pix_width, pix_height, painter: QPainter, target_rect: QRect):
        """waypointのY軸ガイドライン描画"""
        if not hasattr(self.main_window, 'waypoint_count_spin'):
            return

        # 設定を取得
        count = self.main_window.waypoint_count_spin.value()
        start_y = self.main_window.waypoint_start_y_spin.value()
        end_y = self.main_window.waypoint_end_y_spin.value()

        # Y座標の範囲チェック
        if start_y >= pix_height or end_y >= pix_height:
            # 範囲外の場合は描画しない
            return

        # ガイドライン用の描画設定（点線）
        pen = QPen(QColor(255, 255, 0, 150), 2, Qt.DashLine)  # 半透明の黄色点線
        painter.setPen(pen)

        # Y座標を計算して点線を描画
        for i in range(count):
            if count == 1:
                y = (start_y + end_y) / 2  # Y座標の中央
            else:
                # Y座標を等間隔で配置
                y = start_y + (end_y - start_y) * i / (count - 1)

            # 画像座標をスクリーン座標に変換
            screen_y = target_rect.y() + (y / pix_height) * target_rect.height()

            # 画像の左端から右端まで点線を描画
            painter.drawLine(target_rect.left(), int(screen_y), target_rect.right(), int(screen_y))

            # ラベルを表示（左端に番号）
            painter.setPen(QPen(QColor(255, 255, 0), 1))  # 黄色文字
            painter.setFont(QFont("Arial", 12, QFont.Bold))
            painter.drawText(target_rect.left() + 5, int(screen_y - 5), f"{i + 1}")

            # 点線に戻す
            painter.setPen(pen)

    def get_speed_bar_rect(self, target_rect: QRect):
        """speedバーの描画領域を取得するヘルパーメソッド"""
        bar_width = 30
        bar_height = target_rect.height() - 40
        bar_x = target_rect.right() + 20  # 少し右に移動
        bar_y = target_rect.y() + 20
        return QRect(bar_x, bar_y, bar_width, bar_height)

    def is_point_in_speed_bar(self, pos, target_rect: QRect):
        """指定された座標がspeedバー領域内にあるかチェック"""
        if not hasattr(self, 'target_rect'):
            return False

        bar_rect = self.get_speed_bar_rect(target_rect)
        # クリック可能領域を少し広げる（±5ピクセル）
        expanded_rect = bar_rect.adjusted(-5, -5, 5, 5)
        return expanded_rect.contains(pos)

    def handle_speed_bar_click(self, pos):
        """speedバーのクリック処理"""
        if not self.main_window:
            return

        current_index = self.main_window.current_index
        if current_index not in self.main_window.annotations:
            return

        annotation = self.main_window.annotations[current_index]
        if 'speed' not in annotation:
            return

        # speedバーの描画領域を取得
        bar_rect = self.get_speed_bar_rect(self.target_rect)

        # クリック位置のY座標からspeed値を計算
        # バーの下端が0、上端が10に対応
        rel_y = (pos.y() - bar_rect.y()) / bar_rect.height()
        rel_y = max(0, min(1, rel_y))  # 0-1にクランプ

        # Y座標を反転してspeed値に変換（上が高速=10、下が低速=0）
        new_speed = 10 - (rel_y * 10)  # 上端(0) -> 10、下端(1) -> 0

        # speed値を更新
        annotation['speed'] = new_speed

        # 調整中フラグを立てる
        self.is_adjusting_speed = True

        # 画面を更新
        self.update()

        # ステータスバーに表示
        if hasattr(self.main_window, 'statusBar'):
            self.main_window.statusBar().showMessage(f"Speed値を更新: {new_speed:.2f}", 2000)

    def draw_speed_bar(self, pix_width, pix_height, painter: QPainter, target_rect: QRect):
        """画像の右側にspeed値を縦型バーで表示"""
        if not self.main_window:
            return

        # 現在の画像のアノテーションを取得
        current_index = self.main_window.current_index
        if current_index not in self.main_window.annotations:
            return

        annotation = self.main_window.annotations[current_index]

        # speedデータがあるかチェック
        if 'speed' not in annotation:
            return

        speed = annotation['speed']

        # speedバーの描画領域設定
        bar_rect = self.get_speed_bar_rect(target_rect)
        bar_x = bar_rect.x()
        bar_y = bar_rect.y()
        bar_width = bar_rect.width()
        bar_height = bar_rect.height()

        # 背景を描画（グレーの枠）
        # ホバー中または調整中の場合は強調表示
        if self.hovering_speed_bar or self.is_adjusting_speed:
            painter.setPen(QPen(QColor(255, 200, 0), 3))  # オレンジの太い枠
            painter.setBrush(QBrush(QColor(60, 60, 60, 220)))  # やや明るい背景
        else:
            painter.setPen(QPen(QColor(150, 150, 150), 2))
            painter.setBrush(QBrush(QColor(40, 40, 40, 200)))  # 半透明の暗い背景
        painter.drawRect(bar_x, bar_y, bar_width, bar_height)

        # speed値を0-1の範囲に正規化（0～10の範囲を想定）
        normalized_speed = speed / 10.0  # 0～10 -> 0～1
        normalized_speed = max(0, min(1, normalized_speed))  # 0-1にクランプ

        # speedバーの色を自動運転アノテーション（赤丸）と同じ色に
        color = QColor(255, 0, 0, 200)  # 赤（アノテーション点と同色）

        # ホバー中の場合、ホバー位置までのプレビューバーをハッチパターンで表示
        if self.hovering_speed_bar and self.speed_bar_hover_y is not None:
            # ホバー位置から正規化されたspeed値を計算
            hover_normalized = 1.0 - (self.speed_bar_hover_y - bar_y) / bar_height
            hover_normalized = max(0, min(1, hover_normalized))
            preview_height = int(bar_height * hover_normalized)

            # ハッチパターンでプレビューバーを描画
            hatch_color = QColor(255, 100, 100, 150)  # 薄い赤
            hatch_brush = QBrush(hatch_color, Qt.BDiagPattern)  # 斜線ハッチ
            painter.setBrush(hatch_brush)
            painter.setPen(QPen(hatch_color.darker(120), 4))  # 外枠を太く
            painter.drawRect(bar_x, bar_y + bar_height - preview_height, bar_width, preview_height)

        # 現在のspeedバーを描画（下から上に）- 最初に描画
        fill_height = int(bar_height * normalized_speed)
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(120), 1))
        painter.drawRect(bar_x, bar_y + bar_height - fill_height, bar_width, fill_height)

        # 将来のspeedバーを描画（5, 10フレーム先）- 現在のバーの上に描画
        if self.show_future_annotations:
            future_offsets = [10, 5]  # 先に遠い方を描画（後に描画されるものが上に来る）
            future_bar_widths = {5: 20, 10: 12}  # インデックスごとのバー幅（現在は30）
            future_colors = {5: QColor(255, 165, 0), 10: QColor(200, 180, 0)}  # 5:オレンジ, 10:濃い黄色

            for offset in future_offsets:
                future_index = current_index + offset
                if future_index in self.main_window.annotations:
                    future_ann = self.main_window.annotations[future_index]
                    if 'speed' in future_ann:
                        future_speed = future_ann['speed']
                        future_normalized = future_speed / 10.0
                        future_normalized = max(0, min(1, future_normalized))

                        # サイズと色を取得
                        future_width = future_bar_widths.get(offset, 20)
                        future_color = future_colors.get(offset, QColor(255, 165, 0))

                        # バーを中央揃えで描画
                        future_bar_x = bar_x + (bar_width - future_width) // 2
                        future_fill_height = int(bar_height * future_normalized)

                        # オレンジ/黄色で描画
                        painter.setBrush(QBrush(future_color))
                        painter.setPen(QPen(future_color.darker(120), 1))
                        painter.drawRect(future_bar_x, bar_y + bar_height - future_fill_height, future_width, future_fill_height)

        # ダークモード判定
        is_dark = self.main_window.is_dark_mode if self.main_window else False
        text_color = QColor(200, 200, 200) if is_dark else QColor(50, 50, 50)

        # 目盛りを描画（0～10の範囲、11段階）- 右側に配置
        painter.setFont(QFont("Arial", 8))
        for i in range(11):  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            tick_y = bar_y + bar_height - int(bar_height * i / 10)
            # 目盛り線（バーの右側に描画）
            painter.setPen(QPen(text_color, 1))
            painter.drawLine(bar_x + bar_width, tick_y, bar_x + bar_width + 3, tick_y)
            # 目盛り値（0～10の範囲で表示、バーの右側、2刻みで表示）
            tick_value = i
            if i % 2 == 0:  # 0, 2, 4, 6, 8, 10のみ表示
                painter.drawText(bar_x + bar_width + 6, tick_y + 4, f"{tick_value}")

        # 現在のspeed値をテキストで表示（バーの上部）
        painter.setPen(QPen(text_color, 2))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        speed_text = f"{speed:.2f}"
        text_rect = painter.fontMetrics().boundingRect(speed_text)
        painter.drawText(
            bar_x + bar_width // 2 - text_rect.width() // 2,
            bar_y - 5,
            speed_text
        )

        # ラベル "SPEED" を下部に表示
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(bar_x, bar_y + bar_height + 20, "SPEED")  # 少し下に移動

        # 推論結果のspeedがある場合は太い横線で表示
        if (hasattr(self.main_window, 'inference_checkbox') and
            self.main_window.inference_checkbox.isChecked() and
            hasattr(self.main_window, 'inference_results') and
            current_index in self.main_window.inference_results):

            inference = self.main_window.inference_results[current_index]

            # 将来の推論speedバーを描画（t+5, t+10）- 現在の横線の下に描画
            if self.show_future_annotations:
                future_offsets = [10, 5]  # 先に遠い方を描画
                future_line_widths = {5: 3, 10: 2}  # 線の太さ
                future_colors = {5: QColor(0, 200, 200), 10: QColor(0, 150, 150)}  # シアン系

                for offset in future_offsets:
                    future_key = f"future_{offset}"
                    if future_key in inference:
                        future_data = inference[future_key]
                        if 'speed' in future_data:
                            future_infer_speed = future_data['speed']
                            future_infer_speed_display = future_infer_speed * 10.0
                            future_normalized = future_infer_speed_display / 10.0
                            future_normalized = max(0, min(1, future_normalized))

                            future_y = bar_y + bar_height - int(bar_height * future_normalized)
                            future_color = future_colors.get(offset, QColor(0, 200, 200))
                            future_line_width = future_line_widths.get(offset, 2)

                            painter.setPen(QPen(future_color, future_line_width))
                            painter.drawLine(bar_x, future_y, bar_x + bar_width, future_y)

            if 'speed' in inference:
                infer_speed = inference['speed']
                # 推論speed値を正規化（0～10 -> 0～1、学習時に10で正規化されているため10倍して戻す）
                infer_speed_display = infer_speed * 10.0  # 正規化を元に戻す
                normalized_infer_speed = infer_speed_display / 10.0
                normalized_infer_speed = max(0, min(1, normalized_infer_speed))

                # 推論結果の位置にシアン色の太い横線を描画（推論点と同色）
                infer_y = bar_y + bar_height - int(bar_height * normalized_infer_speed)
                painter.setPen(QPen(QColor(0, 255, 255), 4))  # 明るい水色（推論点と同色）
                painter.drawLine(bar_x - 2, infer_y, bar_x + bar_width + 2, infer_y)

                # 推論speed値を横線の右側に表示（少し暗めの色）
                painter.setPen(QPen(QColor(0, 200, 200), 1))  # 少し暗めのシアン
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                painter.drawText(bar_x + bar_width + 25, infer_y + 4, f"({infer_speed_display:.2f})")

    def mousePressEvent(self, event):
        if self.pixmap() and self.main_window:
            # デバウンス処理（連続クリック防止）
            current_time = int(time.time() * 1000)  # ミリ秒に変換
            if current_time - self.last_click_time < self.debounce_delay:
                if hasattr(self.main_window, 'debug_mode') and self.main_window.debug_mode:
                    print(f"デバウンス: クリック無視 (間隔: {current_time - self.last_click_time}ms)")
                return  # デバウンス時間内のクリックは無視
            
            # クリック無効化チェック
            if self.click_disabled or self.is_image_loading:
                if hasattr(self.main_window, 'debug_mode') and self.main_window.debug_mode:
                    print(f"クリック無効化: disabled={self.click_disabled}, loading={self.is_image_loading}")
                return  # クリック処理を無視
            
            # クリック位置を取得
            pos = event.pos()
            self.last_click_time = current_time  # クリック時間を更新

            # speedバーのクリック判定（画像外でもspeedバー領域内なら処理）
            if hasattr(self, 'target_rect') and self.is_point_in_speed_bar(pos, self.target_rect):
                if event.button() == Qt.LeftButton:
                    self.handle_speed_bar_click(pos)
                    return

            # クリック位置が画像内かチェック
            if not self.target_rect.contains(pos):
                return
            
            # 画像内の相対位置を計算
            rel_x = (pos.x() - self.target_rect.x()) / self.target_rect.width()
            rel_y = (pos.y() - self.target_rect.y()) / self.target_rect.height()
            
            # 元の画像の座標に変換
            orig_x = int(rel_x * self.pix_width)
            orig_y = int(rel_y * self.pix_height)
            
            # 現在のモードに基づいて処理
            ## 物体検知モード
            if hasattr(self.main_window, 'current_mode') and self.main_window.current_mode == 1:
                # 物体検知アノテーションモード
                current_index = self.main_window.current_index  # インデックスベースに変更
                
                # 選択されたバウンディングボックスがある場合、ハンドルのチェック
                if (self.selected_bbox_index is not None and 
                    current_index in self.main_window.bbox_annotations):
                    bboxes = self.main_window.bbox_annotations[current_index]
                    if 0 <= self.selected_bbox_index < len(bboxes):
                        bbox = bboxes[self.selected_bbox_index]
                        
                        # バウンディングボックスの座標を取得
                        x1 = int(bbox['x1'] * self.pix_width)
                        y1 = int(bbox['y1'] * self.pix_height)
                        x2 = int(bbox['x2'] * self.pix_width)
                        y2 = int(bbox['y2'] * self.pix_height)
                        
                        # ハンドルのサイズ
                        handle_size = 10  # ハンドルの検出範囲を大きくする
                        
                        # 左上ハンドル
                        if abs(orig_x - x1) <= handle_size and abs(orig_y - y1) <= handle_size:
                            self.is_resizing_bbox = True
                            self.resize_handle = "tl"  # top-left
                            self.resize_start_pos = QPoint(orig_x, orig_y)
                            self.setCursor(Qt.SizeFDiagCursor)  # 斜め矢印カーソル
                            return
                        
                        # 右上ハンドル
                        if abs(orig_x - x2) <= handle_size and abs(orig_y - y1) <= handle_size:
                            self.is_resizing_bbox = True
                            self.resize_handle = "tr"  # top-right
                            self.resize_start_pos = QPoint(orig_x, orig_y)
                            self.setCursor(Qt.SizeBDiagCursor)  # 斜め矢印カーソル
                            return
                        
                        # 左下ハンドル
                        if abs(orig_x - x1) <= handle_size and abs(orig_y - y2) <= handle_size:
                            self.is_resizing_bbox = True
                            self.resize_handle = "bl"  # bottom-left
                            self.resize_start_pos = QPoint(orig_x, orig_y)
                            self.setCursor(Qt.SizeBDiagCursor)  # 斜め矢印カーソル
                            return
                        
                        # 右下ハンドル
                        if abs(orig_x - x2) <= handle_size and abs(orig_y - y2) <= handle_size:
                            self.is_resizing_bbox = True
                            self.resize_handle = "br"  # bottom-right
                            self.resize_start_pos = QPoint(orig_x, orig_y)
                            self.setCursor(Qt.SizeFDiagCursor)  # 斜め矢印カーソル
                            return

                # 既存のバウンディングボックスを選択するかチェック
                if (hasattr(self.main_window, 'bbox_annotations') and 
                    current_index in self.main_window.bbox_annotations):
                    bboxes = self.main_window.bbox_annotations[current_index]
                    
                    # 各バウンディングボックスについて、クリック位置が内部にあるかチェック
                    for i, bbox in enumerate(bboxes):
                        # バウンディングボックスの座標を計算
                        x1 = int(bbox['x1'] * self.pix_width)
                        y1 = int(bbox['y1'] * self.pix_height)
                        x2 = int(bbox['x2'] * self.pix_width)
                        y2 = int(bbox['y2'] * self.pix_height)
                        
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
                                                        
                            self.update() 
                            return
                    
                    # どのバウンディングボックスにも含まれない場合、新規描画開始
                    self.selected_bbox_index = None
                    self.bbox_start = QPoint(orig_x, orig_y)
                    self.is_drawing_bbox = True
                    self.bbox_end = self.bbox_start
                    # 作成中はホバー状態をクリア
                    self.hovering_bbox_index = None
                    self.hovering_segmentation_index = None
                    self.update()  
                else:
                    # バウンディングボックスがない場合、新規描画開始
                    self.selected_bbox_index = None
                    self.bbox_start = QPoint(orig_x, orig_y)
                    self.is_drawing_bbox = True
                    self.bbox_end = self.bbox_start
                    # 作成中はホバー状態をクリア
                    self.hovering_bbox_index = None
                    self.hovering_segmentation_index = None
                    self.update()

            ## セグモード
            elif hasattr(self.main_window, 'current_mode') and self.main_window.current_mode == 2:
                # セグメンテーションモード
                current_index = self.main_window.current_index  # インデックスベースに変更
                
                # ホバー中の頂点がある場合はそれを優先
                if (self.hovering_polygon_index is not None and 
                    self.hovering_vertex_index is not None):
                    # 頂点がクリックされた場合
                    self.selected_polygon_index = self.hovering_polygon_index
                    self.selected_vertex_index = self.hovering_vertex_index
                    self.is_moving_vertex = True
                    
                    # 選択されたセグメンテーションも更新
                    self.selected_segmentation_index = self.hovering_polygon_index
                    
                    # ステータスバーに表示
                    if hasattr(self.main_window, 'statusBar'):
                        self.main_window.statusBar().showMessage(f"頂点を編集中... (頂点 {self.hovering_vertex_index+1})", 3000)
                    
                    self.update()
                    return

                # 既存のセグメンテーションを選択するかチェック
                if (hasattr(self.main_window, 'segmentation_annotations') and 
                    current_index in self.main_window.segmentation_annotations):
                    segmentations = self.main_window.segmentation_annotations[current_index]
                    
                    # 各セグメンテーションについて、クリック位置が内部にあるかチェック
                    for i, seg_data in enumerate(segmentations):
                        if is_point_in_polygon(orig_x, orig_y, seg_data['points']):
                            if event.button() == Qt.LeftButton:
                                # 選択済みのセグメンテーションをクリックした場合
                                if self.selected_segmentation_index == i:
                                    # 選択解除が必要かどうかを判断
                                    if event.modifiers() & Qt.ShiftModifier:
                                        self.selected_segmentation_index = None
                                        self.update()
                                        if hasattr(self.main_window, 'statusBar'):
                                            self.main_window.statusBar().showMessage("セグメンテーションの選択を解除しました", 3000)
                                        return
                                
                                # 新規選択の場合
                                self.selected_segmentation_index = i
                                self.is_moving_segmentation = True
                                self.seg_move_start_pos = QPoint(orig_x, orig_y)
                                                                
                                self.update()
                                return
                            elif event.button() == Qt.RightButton and self.selected_segmentation_index == i:
                                # 右クリックでポリゴンに新しい点を追加
                                self.add_point_to_polygon(i, orig_x, orig_y)
                                return
                
                # 新しいポリゴンの描画処理
                if event.button() == Qt.LeftButton:
                    if not self.is_drawing_segmentation:
                        # 新しいポリゴンを開始
                        self.current_segmentation_polygon = [QPoint(orig_x, orig_y)]
                        self.is_drawing_segmentation = True
                        self.selected_segmentation_index = None
                        # 作成中はホバー状態をクリア
                        self.hovering_bbox_index = None
                        self.hovering_segmentation_index = None
                        self.hovering_polygon_index = None
                        self.hovering_vertex_index = None
                    else:
                        # ポリゴンに点を追加
                        new_point = QPoint(orig_x, orig_y)
                        
                        # 最初の点に近い場合はポリゴンを閉じる
                        if len(self.current_segmentation_polygon) >= 3:
                            first_point = self.current_segmentation_polygon[0]
                            distance = ((new_point.x() - first_point.x())**2 + (new_point.y() - first_point.y())**2)**0.5
                            
                            if distance <= self.close_threshold:
                                # ポリゴンを閉じる
                                polygon_data = self.complete_segmentation_polygon()
                                self.main_window.add_segmentation_annotation(polygon_data)

                                # 描画状態をリセット
                                self.current_segmentation_polygon = []
                                self.is_drawing_segmentation = False
                                self.update()
                                return
                        
                        self.current_segmentation_polygon.append(new_point)
                    self.update()
                elif event.button() == Qt.RightButton and self.is_drawing_segmentation:
                    # 右クリックでポリゴンを完了（3点以上必要）
                    if len(self.current_segmentation_polygon) >= 3:
                        polygon_data = self.complete_segmentation_polygon()
                        if polygon_data:  # キャンセルされていない場合のみ追加
                            self.main_window.add_segmentation_annotation(polygon_data)

                        # 描画状態をリセット（キャンセルした場合も含む）
                        self.current_segmentation_polygon = []
                        self.is_drawing_segmentation = False
                        self.update()

            elif hasattr(self.main_window, 'current_mode') and self.main_window.current_mode == 3:
                # waypointアノテーションモード
                current_index = self.main_window.current_index

                if event.button() == Qt.LeftButton:
                    # 既存のウェイポイントをクリックしているかチェック
                    clicked_waypoint_index = self.get_waypoint_at_position(pos)

                    if clicked_waypoint_index is not None:
                        # 既存のウェイポイントをクリック：ドラッグ開始
                        self.selected_waypoint_index = clicked_waypoint_index
                        self.is_moving_waypoint = True
                        self.waypoint_move_start_pos = pos
                        return

                    # 左クリック: X座標を取得し、対応するY座標をガイドラインから計算
                    if current_index not in self.main_window.waypoint_annotations:
                        self.main_window.waypoint_annotations[current_index] = []

                    # 現在のwaypoint数を取得
                    current_waypoints = self.main_window.waypoint_annotations[current_index]
                    next_waypoint_index = len(current_waypoints)

                    # 設定を取得
                    count = self.main_window.waypoint_count_spin.value()
                    start_y = self.main_window.waypoint_start_y_spin.value()
                    end_y = self.main_window.waypoint_end_y_spin.value()

                    # 打点数の上限チェック
                    if next_waypoint_index >= count:
                        if hasattr(self.main_window, 'statusBar'):
                            self.main_window.statusBar().showMessage(f"設定された打点数({count})に達しています", 2000)
                        return

                    # 画像サイズ取得
                    if not hasattr(self.main_window.main_image_view, 'pix_height'):
                        if hasattr(self.main_window, 'statusBar'):
                            self.main_window.statusBar().showMessage("画像が読み込まれていません", 2000)
                        return

                    img_height = self.main_window.main_image_view.pix_height

                    # Y座標の範囲チェック
                    if start_y >= img_height or end_y >= img_height:
                        if hasattr(self.main_window, 'statusBar'):
                            self.main_window.statusBar().showMessage(f"Y座標が画像サイズ({img_height})を超えています", 2000)
                        return

                    # 対応するY座標を計算
                    if count == 1:
                        waypoint_y = (start_y + end_y) / 2
                    else:
                        # 等間隔でY座標を配置
                        waypoint_y = start_y + (end_y - start_y) * next_waypoint_index / (count - 1)

                    waypoint_y = int(waypoint_y)

                    # Y座標を画像範囲内に制限
                    waypoint_y = max(0, min(waypoint_y, img_height - 1))

                    # waypoint座標をリストに追加（X座標はクリック位置、Y座標は計算値）
                    self.main_window.waypoint_annotations[current_index].append((orig_x, waypoint_y))

                    if hasattr(self.main_window, 'statusBar'):
                        waypoint_count = len(self.main_window.waypoint_annotations[current_index])
                        self.main_window.statusBar().showMessage(f"waypoint{next_waypoint_index + 1}追加: ({orig_x}, {waypoint_y}) - 総数: {waypoint_count}/{count}", 2000)

                    # waypoint配置完了チェックと自動遷移
                    self.check_waypoint_completion_and_advance(current_index, count)

                elif event.button() == Qt.RightButton:
                    # 右クリック: 一筆書きウェイポイント描画開始
                    self.is_drawing_waypoints = True
                    self.drawing_waypoint_path = [pos]
                    self.drawing_start_pos = pos

                    # 既存のウェイポイントをクリア
                    if current_index not in self.main_window.waypoint_annotations:
                        self.main_window.waypoint_annotations[current_index] = []
                    else:
                        self.main_window.waypoint_annotations[current_index].clear()

                    if hasattr(self.main_window, 'statusBar'):
                        self.main_window.statusBar().showMessage("一筆書きモード開始 - ドラッグしてウェイポイントを配置", 2000)
                    return

                self.update()  # 画面を更新してwaypointを表示

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

    def mouseReleaseEvent(self, event):
        # speedバー調整完了処理
        if self.is_adjusting_speed:
            self.is_adjusting_speed = False
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        # 一筆書きウェイポイント描画完了処理
        if self.is_drawing_waypoints:
            self.is_drawing_waypoints = False

            # 一筆書きモードの後処理
            if hasattr(self.main_window, 'waypoint_annotations'):
                current_index = self.main_window.current_index
                if current_index in self.main_window.waypoint_annotations:
                    waypoint_count = len(self.main_window.waypoint_annotations[current_index])
                    if hasattr(self.main_window, 'statusBar'):
                        self.main_window.statusBar().showMessage(f"一筆書きで{waypoint_count}個のウェイポイントを配置しました", 3000)

                    # waypoint配置完了チェックと自動遷移
                    count = self.main_window.waypoint_count_spin.value()
                    self.check_waypoint_completion_and_advance(current_index, count)

            # 描画データをクリア
            self.drawing_waypoint_path.clear()
            self.drawing_start_pos = None

            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        # ウェイポイントドラッグ完了処理
        if self.is_moving_waypoint:
            self.is_moving_waypoint = False
            self.selected_waypoint_index = None
            self.waypoint_move_start_pos = None

            # ステータスバーに完了メッセージ
            if hasattr(self.main_window, 'statusBar'):
                self.main_window.statusBar().showMessage("ウェイポイントの位置を調整しました", 2000)

            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        # 頂点移動完了処理を追加（既存のコードの前に追加）
        if self.is_moving_vertex:
            # 頂点移動が完了
            self.is_moving_vertex = False
            
            # 選択状態は維持（頂点の選択は解除しない）
            
            # ステータスバーに完了メッセージ
            if hasattr(self.main_window, 'statusBar') and self.selected_polygon_index is not None:
                # インデックスベースに変更
                current_index = self.main_window.current_index
                if (current_index is not None and 
                    isinstance(current_index, int) and 
                    current_index in self.main_window.segmentation_annotations):
                    segmentations = self.main_window.segmentation_annotations[current_index]
                    if 0 <= self.selected_polygon_index < len(segmentations):
                        class_name = segmentations[self.selected_polygon_index].get('class', 'unknown')
                        self.main_window.statusBar().showMessage(
                            f"'{class_name}' の頂点編集が完了しました", 3000
                        )
            
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        # 既存のコードに追加
        if self.is_moving_segmentation:
            # セグメンテーション移動が完了
            self.is_moving_segmentation = False
            self.seg_move_start_pos = None
                        
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        if self.is_moving_bbox:
            # 移動が完了したのでフラグをリセット
            self.is_moving_bbox = False
            self.move_start_pos = None
                        
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
        
        elif self.is_resizing_bbox:
            # サイズ変更が完了したのでフラグをリセット
            self.is_resizing_bbox = False
            self.resize_handle = None
            self.resize_start_pos = None
                        
            # 通常のカーソルに戻す
            self.setCursor(Qt.ArrowCursor)
            self.update()    

    def mouseMoveEvent(self, event):
        """マウス移動時の処理 - ハンドルによるサイズ変更機能を追加"""
        pos = event.pos()

        # speedバーの調整中の処理
        if self.is_adjusting_speed:
            self.handle_speed_bar_click(pos)  # 同じロジックを使用
            return

        # speedバーのホバー検出
        if hasattr(self, 'target_rect'):
            is_hovering = self.is_point_in_speed_bar(pos, self.target_rect)
            if is_hovering:
                # ホバー位置のY座標を保存
                self.speed_bar_hover_y = pos.y()
            else:
                self.speed_bar_hover_y = None

            if is_hovering != self.hovering_speed_bar:
                self.hovering_speed_bar = is_hovering
                if is_hovering:
                    self.setCursor(Qt.PointingHandCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)

            # ホバー中は常に再描画（プレビューバー更新のため）
            if is_hovering:
                self.update()

        # ウェイポイントドラッグ中のカーソル設定
        if self.is_moving_waypoint:
            self.setCursor(Qt.ClosedHandCursor)  # つかんでいる状態
        # 物体検知モードでドラッグ中はカーソルを変更
        elif self.is_moving_bbox:
            self.setCursor(Qt.ClosedHandCursor)  # つかんでいる状態
        elif self.is_drawing_bbox:
            self.setCursor(Qt.CrossCursor)  # 描画中は十字
        elif self.is_resizing_bbox:
            # リサイズ中のカーソル設定（ハンドルに応じて）
            if self.resize_handle in ["tl", "br"]:
                self.setCursor(Qt.SizeFDiagCursor)  # 左上-右下
            else:  # "tr", "bl"
                self.setCursor(Qt.SizeBDiagCursor)  # 右上-左下

        # ウェイポイントドラッグ処理
        if self.is_moving_waypoint and self.selected_waypoint_index is not None:
            self.handle_waypoint_drag(event)
            return

        # 一筆書きウェイポイント描画処理
        if self.is_drawing_waypoints:
            self.handle_waypoint_drawing(event)
            return

        # ウェイポイントホバー検出（ドラッグ中でない場合のみ）
        if (hasattr(self.main_window, 'current_mode') and
            self.main_window.current_mode == 3 and
            not self.is_moving_waypoint):
            self.handle_waypoint_hover(event)
        else:
            # ウェイポイントモード以外の場合はホバー状態をクリア
            if hasattr(self, 'hovering_waypoint_index') and self.hovering_waypoint_index is not None:
                self.hovering_waypoint_index = None
                self.update()

        # 既存の移動/描画/リサイズ処理
        if self.pixmap() and (self.is_drawing_bbox or self.is_moving_bbox or self.is_resizing_bbox):
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
                current_index = self.main_window.current_index  # インデックスベースに変更
                if current_index in self.main_window.bbox_annotations:
                    bboxes = self.main_window.bbox_annotations[current_index]
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
            
            elif self.is_resizing_bbox and self.selected_bbox_index is not None and self.resize_handle:
                # サイズ変更処理
                current_index = self.main_window.current_index  # インデックスベースに変更
                if current_index in self.main_window.bbox_annotations:
                    bboxes = self.main_window.bbox_annotations[current_index]
                    if 0 <= self.selected_bbox_index < len(bboxes):
                        bbox = bboxes[self.selected_bbox_index]
                        
                        # 現在のバウンディングボックスの座標を取得（ピクセル単位）
                        x1 = bbox['x1'] * pix_width
                        y1 = bbox['y1'] * pix_height
                        x2 = bbox['x2'] * pix_width
                        y2 = bbox['y2'] * pix_height
                        
                        # リサイズハンドルに応じて座標を更新
                        if self.resize_handle == "tl":  # 左上
                            x1 = orig_x
                            y1 = orig_y
                        elif self.resize_handle == "tr":  # 右上
                            x2 = orig_x
                            y1 = orig_y
                        elif self.resize_handle == "bl":  # 左下
                            x1 = orig_x
                            y2 = orig_y
                        elif self.resize_handle == "br":  # 右下
                            x2 = orig_x
                            y2 = orig_y
                        
                        # 最小サイズの確保（10x10ピクセル）
                        if x2 - x1 < 10:
                            if self.resize_handle in ["tl", "bl"]:
                                x1 = x2 - 10
                            else:
                                x2 = x1 + 10
                        
                        if y2 - y1 < 10:
                            if self.resize_handle in ["tl", "tr"]:
                                y1 = y2 - 10
                            else:
                                y2 = y1 + 10
                        
                        # 画像の端を超えないように制限
                        x1 = max(0, min(x1, pix_width - 10))
                        y1 = max(0, min(y1, pix_height - 10))
                        x2 = max(10, min(x2, pix_width))
                        y2 = max(10, min(y2, pix_height))
                        
                        # バウンディングボックスを更新（正規化座標に戻す）
                        bboxes[self.selected_bbox_index]['x1'] = x1 / pix_width
                        bboxes[self.selected_bbox_index]['y1'] = y1 / pix_height
                        bboxes[self.selected_bbox_index]['x2'] = x2 / pix_width
                        bboxes[self.selected_bbox_index]['y2'] = y2 / pix_height
                        
                        # ステータスバーに情報表示
                        if hasattr(self.main_window, 'statusBar'):
                            class_name = bbox.get('class', 'unknown')
                            width = x2 - x1
                            height = y2 - y1
                            self.main_window.statusBar().showMessage(
                                f"'{class_name}' バウンディングボックスのサイズ変更中... "
                                f"[位置: ({x1:.0f}, {y1:.0f}), サイズ: {width:.0f}x{height:.0f}]", 
                                500
                            )
            
            elif self.is_drawing_bbox:
                # 新規バウンディングボックスの描画処理
                self.bbox_end = QPoint(orig_x, orig_y)
                
                # サイズ情報をステータスバーに表示
                if hasattr(self.main_window, 'statusBar'):
                    width = abs(self.bbox_end.x() - self.bbox_start.x())
                    height = abs(self.bbox_end.y() - self.bbox_start.y())
                    self.main_window.statusBar().showMessage(f"新規バウンディングボックス作成中... 幅: {width}px, 高さ: {height}px", 500)
            
            self.update()  # 画面を更新
        
        # 既存のホバー効果（別の条件）- セグメンテーション作成中も除外
        elif not self.is_moving_bbox and not self.is_drawing_bbox and not self.is_resizing_bbox and not self.is_drawing_segmentation:
            self.check_bbox_hover_and_resize_handles(event.pos())

        # 頂点移動処理を追加（セグメンテーション移動処理の前に追加）
        if self.is_moving_vertex and self.selected_polygon_index is not None and self.selected_vertex_index is not None:
            if not self.pixmap():
                return
                
            # 座標変換
            pos = event.pos()
            pix_width = self.pixmap().width()
            pix_height = self.pixmap().height()
            scaled_width = int(pix_width * self.zoom_factor)
            scaled_height = int(pix_height * self.zoom_factor)
            
            x = (self.width() - scaled_width) // 2
            y = (self.height() - scaled_height) // 2
            target_rect = QRect(x, y, scaled_width, scaled_height)
            
            if not target_rect.contains(pos):
                constrained_x = max(target_rect.left(), min(pos.x(), target_rect.right()))
                constrained_y = max(target_rect.top(), min(pos.y(), target_rect.bottom()))
                pos = QPoint(constrained_x, constrained_y)
            
            rel_x = (pos.x() - target_rect.x()) / target_rect.width()
            rel_y = (pos.y() - target_rect.y()) / target_rect.height()
            orig_x = int(rel_x * pix_width)
            orig_y = int(rel_y * pix_height)
            
            # 頂点の位置を更新
            current_index = self.main_window.current_index  # インデックスベースに変更
            if current_index in self.main_window.segmentation_annotations:
                segmentations = self.main_window.segmentation_annotations[current_index]
                if 0 <= self.selected_polygon_index < len(segmentations):
                    points = segmentations[self.selected_polygon_index]['points']
                    if 0 <= self.selected_vertex_index < len(points):
                        # 画像境界内に制限
                        new_x = max(0, min(orig_x, pix_width))
                        new_y = max(0, min(orig_y, pix_height))
                        points[self.selected_vertex_index] = (new_x, new_y)
                        
                        # ステータスバーに情報表示
                        if hasattr(self.main_window, 'statusBar'):
                            class_name = segmentations[self.selected_polygon_index].get('class', 'unknown')
                            self.main_window.statusBar().showMessage(
                                f"'{class_name}' 頂点 {self.selected_vertex_index+1} を移動中... ({new_x}, {new_y})", 500
                            )
            
            self.update()
            return

        ###
        # セグメンテーションホバー検出（移動中でない場合のみ）- 既存のコードを修正
        elif (not self.is_moving_segmentation and not self.is_moving_vertex and 
            hasattr(self.main_window, 'current_mode') and self.main_window.current_mode == 2):
            
            # 座標変換
            if not self.pixmap():
                return
                
            pos = event.pos()
            pix_width = self.pixmap().width()
            pix_height = self.pixmap().height()
            scaled_width = int(pix_width * self.zoom_factor)
            scaled_height = int(pix_height * self.zoom_factor)
            
            x = (self.width() - scaled_width) // 2
            y = (self.height() - scaled_height) // 2
            target_rect = QRect(x, y, scaled_width, scaled_height)
            
            if not target_rect.contains(pos):
                # マウスが画像外の場合はホバー状態をクリア
                if self.hovering_vertex_index is not None or self.hovering_polygon_index is not None:
                    self.hovering_vertex_index = None
                    self.hovering_polygon_index = None
                    self.setCursor(Qt.ArrowCursor)
                    self.update()
                return
            
            rel_x = (pos.x() - target_rect.x()) / target_rect.width()
            rel_y = (pos.y() - target_rect.y()) / target_rect.height()
            orig_x = int(rel_x * pix_width)
            orig_y = int(rel_y * pix_height)
            
            # 頂点のホバー検出
            hovered_vertex = self.check_vertex_hover(orig_x, orig_y)
            
            # ホバー状態が変化した場合の処理
            if hovered_vertex != (self.hovering_polygon_index, self.hovering_vertex_index):
                if hovered_vertex is not None:
                    self.hovering_polygon_index, self.hovering_vertex_index = hovered_vertex
                    self.setCursor(Qt.PointingHandCursor)  # 手のカーソルに変更
                else:
                    # 頂点ホバーがない場合、セグメンテーション全体のホバーをチェック
                    hover_index = self.check_segmentation_hover(event.pos())
                    self.hovering_polygon_index = hover_index
                    self.hovering_vertex_index = None
                    
                    if hover_index is not None:
                        self.setCursor(Qt.OpenHandCursor)
                    else:
                        self.setCursor(Qt.ArrowCursor)
                
                self.update()

        ###

        # セグメンテーション移動処理
        if self.is_moving_segmentation and self.selected_segmentation_index is not None:
            if not self.pixmap():
                return
                
            # 座標変換（既存のコードと同様）
            pos = event.pos()
            pix_width = self.pixmap().width()
            pix_height = self.pixmap().height()
            scaled_width = int(pix_width * self.zoom_factor)
            scaled_height = int(pix_height * self.zoom_factor)
            
            x = (self.width() - scaled_width) // 2
            y = (self.height() - scaled_height) // 2
            target_rect = QRect(x, y, scaled_width, scaled_height)
            
            if not target_rect.contains(pos):
                constrained_x = max(target_rect.left(), min(pos.x(), target_rect.right()))
                constrained_y = max(target_rect.top(), min(pos.y(), target_rect.bottom()))
                pos = QPoint(constrained_x, constrained_y)
            
            rel_x = (pos.x() - target_rect.x()) / target_rect.width()
            rel_y = (pos.y() - target_rect.y()) / target_rect.height()
            orig_x = int(rel_x * pix_width)
            orig_y = int(rel_y * pix_height)
            
            # セグメンテーションの移動処理
            current_index = self.main_window.current_index  # インデックスベースに変更
            if current_index in self.main_window.segmentation_annotations:
                segmentations = self.main_window.segmentation_annotations[current_index]
                if 0 <= self.selected_segmentation_index < len(segmentations):
                    # 移動距離を計算
                    delta_x = orig_x - self.seg_move_start_pos.x()
                    delta_y = orig_y - self.seg_move_start_pos.y()
                    
                    # ポリゴンの全ての点を移動
                    seg_data = segmentations[self.selected_segmentation_index]
                    new_points = []
                    for px, py in seg_data['points']:
                        new_x = max(0, min(px + delta_x, pix_width))
                        new_y = max(0, min(py + delta_y, pix_height))
                        new_points.append((new_x, new_y))
                    
                    segmentations[self.selected_segmentation_index]['points'] = new_points
                    self.seg_move_start_pos = QPoint(orig_x, orig_y)
                    
                    # ステータスバーに情報表示
                    if hasattr(self.main_window, 'statusBar'):
                        class_name = seg_data.get('class', 'unknown')
                        self.main_window.statusBar().showMessage(f"'{class_name}' セグメンテーションを移動中...", 500)
            
            self.update()

        ###
        # セグメンテーションホバー検出（移動中および作成中でない場合のみ）
        elif not self.is_moving_segmentation and not self.is_drawing_segmentation and hasattr(self.main_window, 'current_mode') and self.main_window.current_mode == 2:
            hover_index = self.check_segmentation_hover(event.pos())
            
            if hover_index != self.hovering_segmentation_index:
                self.hovering_segmentation_index = hover_index
                
                if hover_index is not None:
                    self.setCursor(Qt.OpenHandCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)

                self.update()

        # マウス座標表示の更新（mouseMoveEventの最後で全モード共通）
        self._update_mouse_coordinates(event.pos())

    def leaveEvent(self, event):
        """マウスがウィジェットから離れた時の処理"""
        self.setCursor(Qt.ArrowCursor)
        self.hovering_bbox_index = None

        # セグメンテーション関連のホバー状態もクリア
        self.hovering_segmentation_index = None
        self.hovering_polygon_index = None
        self.hovering_vertex_index = None

        # マウス座標表示もクリア
        self.current_mouse_pos = None
        self.normalized_coords = None

        self.update()  # 画面を更新してホバー効果を消す
        super().leaveEvent(event)
    
    def check_bbox_hover(self, pos):
        """マウス位置がバウンディングボックス上にあるかチェック"""
        if not self.pixmap() or not hasattr(self.main_window, 'current_mode'):
            return None
        
        # 物体検知モードでない場合は処理しない
        if self.main_window.current_mode != 1:
            return None
        
        # マウス位置が画像内かチェック
        if not self.target_rect.contains(pos):
            return None
        
        # 画像内の相対位置を計算
        rel_x = (pos.x() - self.target_rect.x()) / self.target_rect.width()
        rel_y = (pos.y() - self.target_rect.y()) / self.target_rect.height()
        
        # 元の画像の座標に変換
        orig_x = int(rel_x * self.pix_width)
        orig_y = int(rel_y * self.pix_height)
        
        # 現在の画像のバウンディングボックスをチェック
        current_index = self.main_window.current_index  # インデックスベースに変更
        if current_index in self.main_window.bbox_annotations:
            bboxes = self.main_window.bbox_annotations[current_index]
            
            # 各バウンディングボックスについて、マウス位置が内部にあるかチェック
            for i, bbox in enumerate(bboxes):
                # バウンディングボックスの座標を計算
                x1 = int(bbox['x1'] * self.pix_width)
                y1 = int(bbox['y1'] * self.pix_height)
                x2 = int(bbox['x2'] * self.pix_width)
                y2 = int(bbox['y2'] * self.pix_height)
                
                # マウス位置がバウンディングボックス内にあるか
                if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                    return i
        
        return None

    def check_vertex_hover(self, x, y):
        """指定された座標が既存のセグメンテーションの頂点上にホバーしているかチェック"""
        if not hasattr(self.main_window, 'segmentation_annotations'):
            return None
        
        current_index = self.main_window.current_index  # インデックスベースに変更
        if current_index not in self.main_window.segmentation_annotations:
            return None
        
        segmentations = self.main_window.segmentation_annotations[current_index]
        
        # ホバー検出の範囲を少し大きくする
        hover_radius = self.vertex_radius + 2
        
        # 各セグメンテーションの各頂点をチェック
        for polygon_index, seg_data in enumerate(segmentations):
            points = seg_data.get('points', [])
            
            for vertex_index, (px, py) in enumerate(points):
                # 頂点との距離を計算
                distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                
                # ホバー半径内にマウスがある場合
                if distance <= hover_radius:
                    return (polygon_index, vertex_index)
        
        return None

    def check_bbox_hover_and_resize_handles(self, pos):
        """バウンディングボックスのホバー状態とリサイズハンドルのチェック"""
        if not self.pixmap() or not hasattr(self.main_window, 'current_mode'):
            return
        
        # 物体検知モードでない場合は処理しない
        if self.main_window.current_mode != 1:
            return
                
        # マウス位置が画像内かチェック
        if not self.target_rect.contains(pos):
            self.setCursor(Qt.ArrowCursor)
            self.hovering_bbox_index = None
            return
        
        # 画像内の相対位置を計算
        rel_x = (pos.x() - self.target_rect.x()) / self.target_rect.width()
        rel_y = (pos.y() - self.target_rect.y()) / self.target_rect.height()
        
        # 元の画像の座標に変換
        orig_x = int(rel_x * self.pix_width)
        orig_y = int(rel_y * self.pix_height)
        
        # 選択されたバウンディングボックスがある場合、ハンドルのチェック
        if self.selected_bbox_index is not None:
            current_index = self.main_window.current_index  # インデックスベースに変更
            if current_index in self.main_window.bbox_annotations:
                bboxes = self.main_window.bbox_annotations[current_index]
                if 0 <= self.selected_bbox_index < len(bboxes):
                    bbox = bboxes[self.selected_bbox_index]
                    
                    # バウンディングボックスの座標を取得
                    x1 = int(bbox['x1'] * self.pix_width)
                    y1 = int(bbox['y1'] * self.pix_height)
                    x2 = int(bbox['x2'] * self.pix_width)
                    y2 = int(bbox['y2'] * self.pix_height)
                    
                    # ハンドルのサイズ
                    handle_size = 10  # ハンドルの検出範囲
                    
                    # 左上ハンドル
                    if abs(orig_x - x1) <= handle_size and abs(orig_y - y1) <= handle_size:
                        self.setCursor(Qt.SizeFDiagCursor)  # 斜め矢印カーソル
                        return
                    
                    # 右上ハンドル
                    if abs(orig_x - x2) <= handle_size and abs(orig_y - y1) <= handle_size:
                        self.setCursor(Qt.SizeBDiagCursor)  # 斜め矢印カーソル
                        return
                    
                    # 左下ハンドル
                    if abs(orig_x - x1) <= handle_size and abs(orig_y - y2) <= handle_size:
                        self.setCursor(Qt.SizeBDiagCursor)  # 斜め矢印カーソル
                        return
                    
                    # 右下ハンドル
                    if abs(orig_x - x2) <= handle_size and abs(orig_y - y2) <= handle_size:
                        self.setCursor(Qt.SizeFDiagCursor)  # 斜め矢印カーソル
                        return
        
        # 現在の画像のバウンディングボックスをチェック
        hover_index = None
        current_index = self.main_window.current_index  # インデックスベースに変更
        if current_index in self.main_window.bbox_annotations:
            bboxes = self.main_window.bbox_annotations[current_index]
            
            # 各バウンディングボックスについて、マウス位置が内部にあるかチェック
            for i, bbox in enumerate(bboxes):
                # バウンディングボックスの座標を計算
                x1 = int(bbox['x1'] * self.pix_width)
                y1 = int(bbox['y1'] * self.pix_height)
                x2 = int(bbox['x2'] * self.pix_width)
                y2 = int(bbox['y2'] * self.pix_height)
                
                # マウス位置がバウンディングボックス内にあるか
                if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                    hover_index = i
                    break
        
        # ホバー状態が変わった場合は再描画
        if hover_index != self.hovering_bbox_index:
            self.hovering_bbox_index = hover_index

            # カーソルを更新
            if hover_index is not None:
                self.setCursor(Qt.OpenHandCursor)  # バウンディングボックス上では手の形
            else:
                self.setCursor(Qt.ArrowCursor)  # 通常は矢印

            self.update()  # 再描画

    def _update_mouse_coordinates(self, pos):
        """マウス座標を更新して正規化座標を計算"""
        if not self.pixmap():
            self.current_mouse_pos = None
            self.normalized_coords = None
            self.update()
            return

        # 画像の表示領域を取得
        if not hasattr(self, 'target_rect') or not self.target_rect.contains(pos):
            # 画像外の場合はクリア
            self.current_mouse_pos = None
            self.normalized_coords = None
            self.update()
            return

        # 画像内の相対位置を計算
        rel_x = (pos.x() - self.target_rect.x()) / self.target_rect.width()
        rel_y = (pos.y() - self.target_rect.y()) / self.target_rect.height()

        # 元の画像の座標に変換
        orig_x = int(rel_x * self.pix_width)
        orig_y = int(rel_y * self.pix_height)

        # 正規化座標を計算 (x: -1～1, y: -1～1)
        # X座標を-1（左）から1（右）に変換
        x_norm = (rel_x * 2) - 1

        # Y座標を1（上）から-1（下）に変換
        y_norm = -((rel_y * 2) - 1)

        # 座標を保存
        self.current_mouse_pos = pos
        self.normalized_coords = (x_norm, y_norm)

        # 再描画をトリガー
        self.update()

    def complete_segmentation_polygon(self):
        """ポリゴンを完了してクラス選択を行う"""
        if len(self.current_segmentation_polygon) >= 3:
            # クラス選択ダイアログ
            class_name = self.main_window.select_object_class()
            if class_name:
                # ポリゴンを保存
                polygon_data = {
                    'class': class_name,
                    'points': [(p.x(), p.y()) for p in self.current_segmentation_polygon]
                }
                return polygon_data
        return None  # キャンセルまたは無効な場合はNoneを返す

    def check_segmentation_hover(self, pos):
        """マウス位置がセグメンテーション上にあるかチェック"""
        if not self.pixmap() or not hasattr(self.main_window, 'current_mode'):
            return None
        
        if self.main_window.current_mode != 2:
            return None
                
        if not self.target_rect.contains(pos):
            return None
        
        rel_x = (pos.x() - self.target_rect.x()) / self.target_rect.width()
        rel_y = (pos.y() - self.target_rect.y()) / self.target_rect.height()
        orig_x = int(rel_x * self.pix_width)
        orig_y = int(rel_y * self.pix_height)
        
        # 現在の画像のセグメンテーションをチェック
        current_index = self.main_window.current_index  # インデックスベースに変更
        if current_index in self.main_window.segmentation_annotations:
            segmentations = self.main_window.segmentation_annotations[current_index]
            
            for i, seg_data in enumerate(segmentations):
                if is_point_in_polygon(orig_x, orig_y, seg_data['points']):
                    return i
        
        return None

# 下部ギャラリー系（物体検知・セグメンテーション対応強化版）
class ThumbnailWidget(QWidget):
    def __init__(self, parent=None, img_path="", index=0, is_selected=False,
                 annotation=None, on_click=None, location_value=None, is_deleted=False,
                 bbox_annotations=None, segmentation_annotations=None, waypoint_annotations=None):
        super().__init__(parent)
        self.img_path = img_path
        self.index = index
        self.on_click = on_click
        self.is_selected = is_selected
        self.annotation = annotation
        self.location_value = location_value
        self.is_deleted = is_deleted
        self.bbox_annotations = bbox_annotations
        self.segmentation_annotations = segmentation_annotations
        self.waypoint_annotations = waypoint_annotations

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

        # インデックス番号
        self.idx_label = QLabel(f"{index + 1}")
        self.idx_label.setAlignment(Qt.AlignCenter)
        self.idx_label.setStyleSheet("font-weight: bold;")
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
        # アノテーション情報（angleとthrottleが実際に存在し、有効な値の場合のみ表示）
        if annotation:
            # angleが存在し、かつ有効な値（None, 空文字列ではない）の場合のみ表示
            if ('angle' in annotation and
                annotation['angle'] is not None and
                annotation['angle'] != ''):
                angle_label = QLabel(f"A: {annotation['angle']:.2f}")
                angle_label.setStyleSheet("color: #FF6666; font-size: 12px;font-weight: bold;")
                info_layout.addWidget(angle_label)

            # throttleが存在し、かつ有効な値（None, 空文字列ではない）の場合のみ表示
            if ('throttle' in annotation and
                annotation['throttle'] is not None and
                annotation['throttle'] != ''):
                throttle_label = QLabel(f"T: {annotation['throttle']:.2f}")
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

        # 物体検知アノテーション情報を追加 (新規追加)
        if bbox_annotations : #and not is_deleted:
            # オブジェクト数を表示するバッジ
            obj_count = len(bbox_annotations)
            bbox_badge = QLabel(f"物体: {obj_count}")
            bbox_badge.setAlignment(Qt.AlignCenter)
            bbox_badge.setStyleSheet("""
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 10px;
                min-width: 20px;
                min-height: 20px;
                padding: 1px;
                font-size: 10px;
            """)
            info_layout.addWidget(bbox_badge)

            # クラスごとのカウントを集計
            class_counts = {}
            for bbox in bbox_annotations:
                class_name = bbox.get('class', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # 主要なクラスを最大2つまで表示
            for i, (class_name, count) in enumerate(class_counts.items()):
                if i >= 2:  # 最大2クラスまで表示
                    break

                class_label = QLabel(f"{class_name}: {count}")
                class_label.setStyleSheet("font-size: 10px; color: #333;")
                info_layout.addWidget(class_label)

        # セグメンテーションアノテーション情報を追加（物体検知アノテーション情報の後に）
        if segmentation_annotations and not is_deleted:
            # セグメンテーション数を表示するバッジ
            seg_count = len(segmentation_annotations)
            seg_badge = QLabel(f"セグ: {seg_count}")
            seg_badge.setAlignment(Qt.AlignCenter)
            seg_badge.setStyleSheet("""
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                border-radius: 10px;
                min-width: 20px;
                min-height: 20px;
                padding: 1px;
                font-size: 10px;
            """)
            info_layout.addWidget(seg_badge)

            # クラスごとのカウントを集計
            seg_class_counts = {}
            for seg in segmentation_annotations:
                class_name = seg.get('class', 'unknown')
                seg_class_counts[class_name] = seg_class_counts.get(class_name, 0) + 1

            # 主要なクラスを最大2つまで表示
            for i, (class_name, count) in enumerate(seg_class_counts.items()):
                if i >= 2:  # 最大2クラスまで表示
                    break

                seg_class_label = QLabel(f"{class_name}: {count}")
                seg_class_label.setStyleSheet("font-size: 10px; color: #9C27B0;")
                info_layout.addWidget(seg_class_label)

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
        elif bbox_annotations:  # 物体検知アノテーションがある場合は青い枠線
            border_style = "border: 2px solid #2196F3;"
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
            # PILで画像を開く
            pil_img = Image.open(img_path)

            # 画像のコピーを作成して描画する
            draw_img = pil_img.copy()
            draw = ImageDraw.Draw(draw_img)

            # 基本的なアノテーションを描画（座標が存在する場合のみ）
            if self.annotation and 'x' in self.annotation and 'y' in self.annotation:
                # アノテーションの座標を取得
                x, y = self.annotation["x"], self.annotation["y"]

                # 丸を描画
                circle_size = 15  # サムネイル用の丸のサイズ
                draw.ellipse((x-circle_size, y-circle_size, x+circle_size, y+circle_size),
                            outline='red', width=4)

           # セグメンテーションアノテーションの描画（新規追加）
            if self.segmentation_annotations and not self.is_deleted:
                for seg_data in self.segmentation_annotations:
                    class_name = seg_data.get('class', 'unknown')
                    points = seg_data.get('points', [])

                    if len(points) >= 3:
                        # クラスに応じた色を定義
                        class_colors = {
                            'car': (255, 0, 0, 120),      # 赤
                            'person': (0, 255, 0, 120),   # 緑
                            'sign': (0, 0, 255, 120),     # 青
                            'cone': (255, 255, 0, 120),   # 黄
                            'unknown': (128, 128, 128, 120)  # グレー
                        }

                        color = class_colors.get(class_name, (255, 0, 0, 120))

                        # ポリゴンを描画（アウトライン）
                        outline_color = (color[0], color[1], color[2])  # アルファ値なし
                        draw.polygon(points, outline=outline_color, width=2)

                        # ラベルを表示（中心点に）
                        if points:
                            center_x = sum(p[0] for p in points) // len(points)
                            center_y = sum(p[1] for p in points) // len(points)

                            # ラベル背景を描画
                            label_text = class_name[0].upper()  # 頭文字のみ
                            text_size = 12

                            # 背景矩形
                            label_bg = (center_x-text_size//2, center_y-text_size//2,
                                    center_x+text_size//2, center_y+text_size//2)
                            draw.rectangle(label_bg, fill=outline_color)

                            # テキスト描画
                            draw.text((center_x-4, center_y-6), label_text, fill=(255, 255, 255))

            # 物体検知アノテーションがある場合は矩形を描画
            if self.bbox_annotations: # and not self.is_deleted:
                img_width, img_height = pil_img.size

                for bbox in self.bbox_annotations:
                    # クラスに応じた色を定義
                    class_colors = {
                        'car': (255, 0, 0),      # 赤
                        'person': (0, 255, 0),   # 緑
                        'sign': (0, 0, 255),     # 青
                        'cone': (255, 255, 0),   # 黄
                        'unknown': (128, 128, 128)  # グレー
                    }

                    class_name = bbox.get('class', 'unknown')
                    color = class_colors.get(class_name, (255, 0, 0))

                    # 正規化された座標を実際の座標に変換
                    x1 = int(bbox['x1'] * img_width)
                    y1 = int(bbox['y1'] * img_height)
                    x2 = int(bbox['x2'] * img_width)
                    y2 = int(bbox['y2'] * img_height)

                    # 矩形を描画
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                    # ラベルを表示 (サムネイルでは小さいのでクラス名の1文字目だけ表示)
                    label_text = class_name[0].upper()  # 頭文字のみ

                    # ラベル背景
                    text_size = 10  # 大まかなテキストサイズ
                    label_bg = (x1, y1-text_size, x1+text_size, y1)
                    draw.rectangle(label_bg, fill=color)

                    # テキスト描画
                    draw.text((x1+2, y1-text_size), label_text, fill=(255, 255, 255))

            # waypointアノテーションがある場合は緑色の丸を描画
            if self.waypoint_annotations and not self.is_deleted:
                for i, (x, y) in enumerate(self.waypoint_annotations):
                    # 緑色の丸を描画（サムネイル用に小さめ）
                    circle_size = 5  # サムネイル用の丸のサイズ
                    draw.ellipse((x-circle_size, y-circle_size, x+circle_size, y+circle_size),
                                fill=(0, 255, 0, 180), outline=(0, 128, 0), width=1)

                    # waypoint番号を表示
                    label_text = str(i + 1)
                    text_size = 8

                    # テキスト描画（白文字）
                    draw.text((x-2, y-4), label_text, fill=(255, 255, 255))

            # 画像をQImageに変換
            draw_img = draw_img.convert("RGBA")
            data = draw_img.tobytes("raw", "RGBA")
            qimg = QImage(data, draw_img.width, draw_img.height, QImage.Format_RGBA8888)

            # QImageをQPixmapに変換してサムネイルに設定
            pixmap = QPixmap.fromImage(qimg)

            if not pixmap.isNull():
                # 画像ラベルのサイズを取得
                label_width = self.img_label.width()
                label_height = self.img_label.height()

                # サイズが0の場合は固定サイズを使用
                if label_width == 0 or label_height == 0:
                    label_width = 150
                    label_height = 140

                # 重要な変更: ラベルサイズと同じサイズでスケーリング
                scaled_pixmap = pixmap.scaled(
                    label_width,
                    label_height,
                    Qt.KeepAspectRatio,  # アスペクト比を維持
                    Qt.SmoothTransformation  # 滑らかな変換
                )

                self.img_label.setPixmap(scaled_pixmap)
                self.img_label.setAlignment(Qt.AlignCenter)  # 中央揃え

                # 削除済みの場合は半透明にする追加の処理
                if self.is_deleted:
                    opacity_effect = QGraphicsOpacityEffect()
                    opacity_effect.setOpacity(0.5)
                    self.img_label.setGraphicsEffect(opacity_effect)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
# データ操作全体系
class ImageAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.data_manager = AnnotationDataManager()
        self.mlflow_manager = MLflowManager(mlflow_dir)

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

        # セグメンテーション関連の初期化
        self.segmentation_annotations = {}  # セグメンテーションアノテーション用
        self.segmentation_inference_results = {}  # セグメンテーション推論結果
        self.yolo_seg_model = None  # YOLOセグメンテーションモデル
        self.last_segmentations = []
        self.show_segmentation_inference = False  # セグメンテーション推論表示フラグ
        self.seg_driving_direction_class_id = 0  # 走行方向計算に使用するセグメンテーションクラスID
        self.seg_driving_direction_y = 100  # 走行方向計算のY座標（固定値、ユーザーが変更可能）
        self.show_seg_driving_direction = False  # 走行方向矢印の表示フラグ
        self.seg_max_steering_angle = 30.0  # 最大舵角（度）
        self.seg_display_mode = 'trajectory'  # 表示モード: 'trajectory' or 'waypoint'

        # waypoint関連の初期化
        self.waypoint_annotations = {}  # waypointアノテーション用 {image_index: [(x, y), ...]}
        self.last_waypoints = []  # 前回の画像のwaypointを保存
        self.auto_apply_last_waypoint = False  # 前回のwaypointを自動適用するかどうか
        self.auto_advance_waypoint = True  # waypoint配置完了時に次の画像へ自動遷移するかどうか（デフォルトでON）

        # 削除インデックス
        self.deleted_indexes = []

        # ダウンサンプリング対象インデックス（直進時など）
        self.downsampled_indexes = []

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

        # スライダー推論デバウンス用タイマー
        self.inference_debounce_timer = QTimer()
        self.inference_debounce_timer.setSingleShot(True)
        self.inference_debounce_timer.timeout.connect(self.execute_slider_inference)
        self.inference_debounce_delay = 300  # 300ms後に推論実行

        # 分布グラフ更新デバウンス用タイマー
        self.graph_update_timer = QTimer()
        self.graph_update_timer.setSingleShot(True)
        self.graph_update_timer.timeout.connect(self._execute_distribution_graph_update)
        self.graph_update_delay = 500  # 500ms後にグラフ更新

        # ギャラリー更新デバウンス用タイマー
        self.gallery_update_timer = QTimer()
        self.gallery_update_timer.setSingleShot(True)
        self.gallery_update_timer.timeout.connect(self._execute_gallery_update)
        self.gallery_update_delay = 150  # 150ms後にギャラリー更新

        # 推論実行デバウンス用タイマー
        self.inference_timer = QTimer()
        self.inference_timer.setSingleShot(True)
        self.inference_timer.timeout.connect(self._execute_deferred_inference)
        self.inference_delay = 50  # 50ms後に推論実行

        # 画像サイズキャッシュ（パフォーマンス改善）
        self.image_size_cache = {}  # {image_path: (width, height)}
        
        # 推論結果のキャッシュ
        self.inference_results = {}
        self.inference_diff_vectors = {}
        self.show_diff_vectors = False  

        # YOLO関連の初期化を追加
        self.yolo_model = None  # YOLOモデルのインスタンス
        self.yolo_confidence_threshold = 0.6  
        self.bbox_annotations = {} 
        
        # ダークモード状態
        self.is_dark_mode = False

        # Setup UI
        self.init_ui()
        
        # Load saved display settings
        self.load_display_settings()

        # Update UI
        self.display_current_image()
        self.update_gallery()
        self.update_slider_deleted_indexes()

        if hasattr(self, 'prev_multi_button') and hasattr(self, 'next_multi_button'):
            self.update_skip_button_labels(10)  # デフォルト値は10

        # セッション復元チェックを遅延実行（UIが完全に表示された後）
        QTimer.singleShot(500, self.add_session_check_to_init_ui)

        # Databricks接続確認を遅延実行
        QTimer.singleShot(1000, self._check_databricks_connection_on_startup)

        QApplication.instance().installEventFilter(self)

    def init_ui(self):
        self.setWindowTitle("画像アノテーションツール")
        self.setGeometry(MAIN_WINDOW_X, MAIN_WINDOW_Y, MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls with scroll area
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll_area.setMaximumWidth(LEFT_PANEL_MAX_WIDTH + 20)  # スクロールバー分の余裕
        left_scroll_area.setMinimumWidth(LEFT_PANEL_MIN_WIDTH)  # 最小幅を設定
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(LEFT_PANEL_MIN_WIDTH - 20)  # スクロールバー分を考慮した最小幅を確保
        
        left_scroll_area.setWidget(left_panel)
        main_layout.addWidget(left_scroll_area)

        # Folder selection
        folder_label = QLabel("データ読込（imagesフォルダの親フォルダ:")
        folder_label.setStyleSheet("font-weight: bold;")  
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

        self.load_button = QPushButton("画像読込")
        self.load_button.clicked.connect(self.load_images)
        self.load_button.setEnabled(False)  # 初期状態は無効
        apply_style(self.load_button, 'primary')
        load_button_layout.addWidget(self.load_button)

        # アノテーションデータ読み込みボタンを追加
        self.load_annotation_button = QPushButton("アノテーション読込")
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
        save_label = QLabel("アノテーションデータ保存:")
        save_label.setStyleSheet("font-weight: bold;")  # 太文字にするスタイルを追加
        left_layout.addWidget(save_label)

        export_layout = QHBoxLayout()
        
        # donkey保存
        donkey_btn = QPushButton("Donkey")
        donkey_btn.clicked.connect(self.export_to_donkey)
        apply_style(donkey_btn, 'export')
        export_layout.addWidget(donkey_btn)

        # jetracer保存保存
        jetracer_btn = QPushButton("Jetracr")
        jetracer_btn.clicked.connect(self.export_to_jetracer)
        apply_style(jetracer_btn, 'export')
        export_layout.addWidget(jetracer_btn)

        # 統合YOLOエクスポートボタン（修正）
        yolo_btn = QPushButton("YOLO")
        yolo_btn.clicked.connect(self.export_to_yolo_unified)
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
        pilot_label = QLabel("自動運転モデル:")
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

        # モデル学習ボタン（自動運転用）
        train_model_button = QPushButton("モデル学習・保存")
        train_model_button.clicked.connect(self.train_and_save_model)
        apply_style(train_model_button, 'training')
        model_buttons_layout.addWidget(train_model_button)  

        # モデル明示的読み込みボタン
        self.model_load_button = QPushButton("モデル読込")
        self.model_load_button.setToolTip("modelsフォルダのモデルを読込む")
        self.model_load_button.clicked.connect(self.load_selected_model)
        apply_style(self.model_load_button, 'model')
        model_buttons_layout.addWidget(self.model_load_button)

        pilot_layout.addLayout(model_buttons_layout)

        # オートアノテーションボタン
        self.auto_annotate_button = QPushButton("オートアノテーション実行")
        self.auto_annotate_button.clicked.connect(self.auto_annotate)
        self.auto_annotate_button.setEnabled(False)  # 初期状態で非アクティブ
        apply_style(self.auto_annotate_button, 'special')
        pilot_layout.addWidget(self.auto_annotate_button)

        # 将来アノテーション表示オプション
        future_layout = QHBoxLayout()
        self.future_annotation_checkbox = QCheckBox("5,10個先のアノテーション表示（燈色）")
        self.future_annotation_checkbox.setChecked(True)  # デフォルトON
        self.future_annotation_checkbox.setToolTip("5フレーム先と10フレーム先のアノテーションを表示")
        self.future_annotation_checkbox.stateChanged.connect(self.toggle_future_annotation_display)
        future_layout.addWidget(self.future_annotation_checkbox)
        future_layout.addStretch()
        pilot_layout.addLayout(future_layout)

        # 推論結果表示オプション
        inference_layout = QHBoxLayout()
        self.inference_checkbox = QCheckBox("推論結果表示（青丸）")
        self.inference_checkbox.setChecked(False)
        self.inference_checkbox.setEnabled(False)  # 初期状態は無効
        self.inference_checkbox.setToolTip("自動運転モデルが読み込まれていません")
        self.inference_checkbox.stateChanged.connect(self.toggle_inference_display)
        inference_layout.addWidget(self.inference_checkbox)


        # 一括推論実行ボタンを追加
        self.batch_inference_button = QPushButton("全画像を推論")
        self.batch_inference_button.setToolTip("自動運転モデルが読み込まれていません")
        self.batch_inference_button.setEnabled(False)  # 初期状態は無効
        self.batch_inference_button.clicked.connect(self.run_batch_inference)
        inference_layout.addWidget(self.batch_inference_button)

        pilot_layout.addLayout(inference_layout)

        # 差分ベクトル表示オプション
        diff_vector_layout = QHBoxLayout()
        self.diff_vector_checkbox = QCheckBox("差分ベクトル表示（緑矢印）")
        self.diff_vector_checkbox.setChecked(False)
        self.diff_vector_checkbox.setEnabled(False)  # 初期状態は無効
        self.diff_vector_checkbox.setToolTip("自動運転モデルが読み込まれていません")
        self.diff_vector_checkbox.stateChanged.connect(self.toggle_diff_vector_display)
        diff_vector_layout.addWidget(self.diff_vector_checkbox)

        pilot_layout.addLayout(diff_vector_layout)

        # CAM表示オプション - 2行レイアウト
        gradcam_container = QVBoxLayout()

        # 1行目: CAMチェックボックス + 手法選択
        gradcam_row1 = QHBoxLayout()
        self.gradcam_checkbox = QCheckBox("CAM")
        self.gradcam_checkbox.setChecked(False)
        self.gradcam_checkbox.setEnabled(False)  # 初期状態は無効
        self.gradcam_checkbox.setToolTip("自動運転モデルが読み込まれていません")
        self.gradcam_checkbox.stateChanged.connect(self.toggle_gradcam_display)
        gradcam_row1.addWidget(self.gradcam_checkbox)

        gradcam_method_label = QLabel("手法:")
        gradcam_row1.addWidget(gradcam_method_label)

        self.gradcam_method_combo = QComboBox()
        self.gradcam_method_combo.addItems(["gradcam", "gradcam++", "eigencam", "layercam", "scorecam"])
        self.gradcam_method_combo.setCurrentText("gradcam")
        self.gradcam_method_combo.setEnabled(False)
        self.gradcam_method_combo.setToolTip("CAM可視化手法を選択\nScoreCAM: 勾配を使わない高精度手法（計算時間長め）")
        self.gradcam_method_combo.currentTextChanged.connect(self.change_gradcam_method)
        gradcam_row1.addWidget(self.gradcam_method_combo)

        gradcam_row1.addStretch()
        gradcam_container.addLayout(gradcam_row1)

        # 2行目: 対象選択 + 方向選択
        gradcam_row2 = QHBoxLayout()

        gradcam_target_label = QLabel("対象:")
        gradcam_row2.addWidget(gradcam_target_label)

        self.gradcam_target_combo = QComboBox()
        self.gradcam_target_combo.addItems(["angle", "throttle", "speed"])
        self.gradcam_target_combo.setCurrentText("angle")
        self.gradcam_target_combo.setEnabled(False)
        self.gradcam_target_combo.setToolTip("CAMで可視化する出力を選択")
        self.gradcam_target_combo.currentTextChanged.connect(self.change_gradcam_target)
        gradcam_row2.addWidget(self.gradcam_target_combo)

        gradcam_direction_label = QLabel("方向:")
        gradcam_row2.addWidget(gradcam_direction_label)

        self.gradcam_direction_combo = QComboBox()
        self.gradcam_direction_combo.addItems(["both", "positive", "negative"])
        self.gradcam_direction_combo.setCurrentText("both")
        self.gradcam_direction_combo.setEnabled(False)
        self.gradcam_direction_combo.setToolTip(
            "可視化する勾配の方向\n"
            "both: 正負両方を同時表示（赤=正/青=負）\n"
            "positive: 出力を増加させる根拠（右に切る/加速）\n"
            "negative: 出力を減少させる根拠（左に切る/減速）"
        )
        self.gradcam_direction_combo.currentTextChanged.connect(self.change_gradcam_direction)
        gradcam_row2.addWidget(self.gradcam_direction_combo)

        gradcam_row2.addStretch()
        gradcam_container.addLayout(gradcam_row2)

        pilot_layout.addLayout(gradcam_container)

        left_layout.addWidget(self.pilot_container)
                    
        # 物体検知推論結果表示フラグの初期化
        self.show_detection_inference = False

        # 物体検知推論結果格納用の辞書を初期化
        self.detection_inference_results = {}

        # 物体検知推論結果表示用ラベルを作成
        self.detection_inference_info_label = QLabel("")
        self.detection_inference_info_label.setStyleSheet("color: #009999;")  # ダークシアン（ライトモード対応）
        self.detection_inference_info_label.setWordWrap(True)

        # 推論実行ボタン
        inference_button_layout = QHBoxLayout()

        left_layout.addLayout(inference_button_layout)


        # 物体検知設定コンテナ
        self.object_detection_container = QWidget()
        obj_detection_layout = QVBoxLayout(self.object_detection_container)

        # ラベル
        obj_detection_label = QLabel("物体検知・セグメンテーションモデル:")
        obj_detection_label.setStyleSheet("font-weight: bold")
        obj_detection_layout.addWidget(obj_detection_label)

        # YOLOアノテーション読み込みボタン
        load_yolo_btn = QPushButton("YOLOアノテーション読込")
        load_yolo_btn.clicked.connect(self.load_yolo_annotations)
        apply_style(load_yolo_btn, 'primary')
        obj_detection_layout.addWidget(load_yolo_btn)

        # クラス入力フィールド
        classes_layout = QHBoxLayout()
        class_label = QLabel("検知クラス:")
        class_label.setMinimumWidth(70)  # ラベルの最小幅を設定
        classes_layout.addWidget(class_label)
        
        self.classes_input = QLineEdit("car,red_sign,green_sign,dog")
        self.classes_input.setPlaceholderText("カンマ区切りでクラス名を入力")
        self.classes_input.setToolTip("例: car,red_sign,green_sign,dog")
        self.classes_input.setMinimumWidth(200)  # 入力フィールドの最小幅を設定（拡張）
        self.classes_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 横は拡張、縦は固定
        classes_layout.addWidget(self.classes_input)
        obj_detection_layout.addLayout(classes_layout)
        
        # クラス操作ボタン
        class_buttons_layout = QHBoxLayout()
        preset_button = QPushButton("プリセット")
        preset_button.setMinimumWidth(80)  # ボタンの最小幅を設定
        preset_button.clicked.connect(self.show_class_preset_dialog)
        class_buttons_layout.addWidget(preset_button)
        
        apply_button = QPushButton("反映")
        apply_button.setMinimumWidth(80)  # ボタンの最小幅を設定
        apply_button.clicked.connect(self.apply_classes)
        class_buttons_layout.addWidget(apply_button)
        
        # ボタンレイアウトにストレッチを追加して均等配置
        class_buttons_layout.addStretch()
        obj_detection_layout.addLayout(class_buttons_layout)
        
        # クラス入力フィールドの変更を監視
        self.classes_input.textChanged.connect(self.on_classes_changed)

        # YOLOモデルタイプ選択
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("YOLOモデル:"))
        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.addItems(["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        self.yolo_model_combo.currentIndexChanged.connect(self.on_yolo_model_type_changed)
        model_type_layout.addWidget(self.yolo_model_combo)
        obj_detection_layout.addLayout(model_type_layout)

        self.yolo_unified_model_combo = QComboBox()
        self.yolo_unified_model_combo.setMinimumWidth(180)
        self.yolo_unified_model_combo.setStyleSheet("combobox-popup: 0; font-size: 12px;")
        obj_detection_layout.addWidget(self.yolo_unified_model_combo)

        # YOLOモデル操作ボタン
        yolo_model_buttons_layout = QHBoxLayout()

        # 統合された学習ボタン（物体検知・セグメンテーション両対応）
        train_yolo_button = QPushButton("YOLO学習・保存")
        train_yolo_button.clicked.connect(self.train_yolo_unified)  # 新しいメソッドに変更
        train_yolo_button.setToolTip("物体検知またはセグメンテーションを学習")
        apply_style(train_yolo_button, 'training')
        yolo_model_buttons_layout.addWidget(train_yolo_button)

        # モデル読み込みボタン
        self.yolo_load_button = QPushButton("モデル読込")
        self.yolo_load_button.setToolTip("modelsフォルダのモデルを読込む")
        self.yolo_load_button.clicked.connect(self.load_yolo_model_unified)  # 新しいメソッドに変更
        #self.yolo_load_button.clicked.connect(self.load_yolo_model_unified)  # 新しいメソッドに変更
        apply_style(self.yolo_load_button, 'model')
        yolo_model_buttons_layout.addWidget(self.yolo_load_button)

        obj_detection_layout.addLayout(yolo_model_buttons_layout)

        # YOLOオートアノテーション実行ボタン
        self.yolo_auto_annotate_btn = QPushButton("YOLO オートアノテーション実行")
        self.yolo_auto_annotate_btn.clicked.connect(self.yolo_auto_annotate)
        self.yolo_auto_annotate_btn.setEnabled(False)  # 初期状態で非アクティブ
        apply_style(self.yolo_auto_annotate_btn, 'special')
        obj_detection_layout.addWidget(self.yolo_auto_annotate_btn)

        # 推論結果表示オプション
        inference_layout = QHBoxLayout()
        
        # 物体検知推論結果表示チェックボックス
        self.detection_inference_checkbox = QCheckBox("物体検知推論結果表示")
        self.detection_inference_checkbox.setChecked(False)
        self.detection_inference_checkbox.setEnabled(False)  # 初期状態は無効
        self.detection_inference_checkbox.setToolTip("物体検知モデルが読み込まれていません")
        self.detection_inference_checkbox.stateChanged.connect(self.toggle_detection_inference_display)
        inference_layout.addWidget(self.detection_inference_checkbox)
        
        # セグメンテーション推論結果表示チェックボックス
        self.segmentation_inference_checkbox = QCheckBox("セグメンテーション推論結果表示")
        self.segmentation_inference_checkbox.setChecked(False)
        self.segmentation_inference_checkbox.setEnabled(False)  # 初期状態は無効
        self.segmentation_inference_checkbox.setToolTip("セグメンテーションモデルが読み込まれていません")
        self.segmentation_inference_checkbox.stateChanged.connect(self.toggle_segmentation_inference_display)
        inference_layout.addWidget(self.segmentation_inference_checkbox)
        
        obj_detection_layout.addLayout(inference_layout)
        
        # 位置推論モデル追加（left_layoutが確実に存在する段階で呼び出し）
        self.add_location_model_section()

        # ウェイポイントモデル追加
        self.add_waypoint_model_section()

        # 物体検知コンテナを追加
        left_layout.addWidget(self.object_detection_container)

        # --- モデル管理セクション ---
        model_mgmt_layout = QVBoxLayout()

        model_mgmt_label = QLabel("モデル管理やクラウド学習:")
        model_mgmt_label.setStyleSheet("font-weight: bold;")
        model_mgmt_layout.addWidget(model_mgmt_label)

        # --- 1. MLflow（ローカル）セクション ---
        mlflow_section_layout = QHBoxLayout()

        mlflow_local_label = QLabel("MLflow（ローカル）:")
        mlflow_section_layout.addWidget(mlflow_local_label)

        # MLflowを開くボタン
        mlflow_open_button = QPushButton("MLflowを開く")
        apply_style(mlflow_open_button, 'special')
        mlflow_open_button.clicked.connect(self._open_local_mlflow_ui)
        mlflow_open_button.setToolTip("ローカルMLflow UIを起動")
        mlflow_section_layout.addWidget(mlflow_open_button)

        mlflow_section_layout.addStretch()
        model_mgmt_layout.addLayout(mlflow_section_layout)

        # --- 2. Databricksセクション ---
        # Databricks連携チェックボックスとステータス
        databricks_header_layout = QHBoxLayout()

        self.databricks_checkbox = QCheckBox("Databricks連携")
        self.databricks_checkbox.setChecked(self.mlflow_manager.use_databricks)
        self.databricks_checkbox.stateChanged.connect(self._on_databricks_toggle)
        databricks_header_layout.addWidget(self.databricks_checkbox)

        self.databricks_status_label = QLabel()
        self._update_databricks_status_label()
        databricks_header_layout.addWidget(self.databricks_status_label)

        databricks_header_layout.addStretch()
        model_mgmt_layout.addLayout(databricks_header_layout)

        # Databricksボタン（開く、同期、設定を横並び）
        databricks_buttons_layout = QHBoxLayout()

        # Databricksを開くボタン
        databricks_open_button = QPushButton("Databricksを開く")
        apply_style(databricks_open_button, 'special')
        databricks_open_button.clicked.connect(self._open_databricks_ui)
        databricks_open_button.setToolTip("Databricks MLflow UIを開く")
        databricks_buttons_layout.addWidget(databricks_open_button)

        # 同期ボタン
        self.databricks_sync_button = QPushButton("同期")
        apply_style(self.databricks_sync_button, 'special')
        self.databricks_sync_button.clicked.connect(self._sync_to_databricks)
        self.databricks_sync_button.setToolTip("ローカルの学習記録をDatabricksにアップロード")
        databricks_buttons_layout.addWidget(self.databricks_sync_button)

        # データ転送ボタン
        self.databricks_transfer_button = QPushButton("転送")
        apply_style(self.databricks_transfer_button, 'special')
        self.databricks_transfer_button.clicked.connect(self._transfer_to_databricks)
        self.databricks_transfer_button.setToolTip("現在のアノテーションをDatabricksに転送")
        databricks_buttons_layout.addWidget(self.databricks_transfer_button)

        # 設定ボタン
        databricks_settings_button = QPushButton("設定")
        databricks_settings_button.setMaximumWidth(60)
        apply_style(databricks_settings_button, 'special')
        databricks_settings_button.clicked.connect(self._show_databricks_settings)
        databricks_buttons_layout.addWidget(databricks_settings_button)

        model_mgmt_layout.addLayout(databricks_buttons_layout)

        # --- 3. Google Colabセクション ---
        # Colab連携チェックボックスとステータス
        colab_header_layout = QHBoxLayout()

        self.colab_checkbox = QCheckBox("Google Colab連携")
        self.colab_checkbox.setChecked(self._is_colab_enabled())
        self.colab_checkbox.stateChanged.connect(self._on_colab_toggle)
        colab_header_layout.addWidget(self.colab_checkbox)

        self.colab_status_label = QLabel()
        self._update_colab_status_label()
        colab_header_layout.addWidget(self.colab_status_label)

        colab_header_layout.addStretch()
        model_mgmt_layout.addLayout(colab_header_layout)

        # Colabボタン（開く、転送、設定を横並び）
        colab_buttons_layout = QHBoxLayout()

        # Colabを開くボタン
        colab_open_button = QPushButton("Colabを開く")
        apply_style(colab_open_button, 'special')
        colab_open_button.clicked.connect(self._open_colab_ui)
        colab_open_button.setToolTip("Google Colabをブラウザで開く")
        colab_buttons_layout.addWidget(colab_open_button)

        # データ転送ボタン
        self.colab_transfer_button = QPushButton("転送")
        apply_style(self.colab_transfer_button, 'special')
        self.colab_transfer_button.clicked.connect(self._transfer_to_colab)
        self.colab_transfer_button.setToolTip("現在のアノテーションをGoogle Driveに転送してColabで学習")
        colab_buttons_layout.addWidget(self.colab_transfer_button)

        # モデル取得ボタン
        self.colab_download_button = QPushButton("取得")
        apply_style(self.colab_download_button, 'special')
        self.colab_download_button.clicked.connect(self._download_model_from_colab)
        self.colab_download_button.setToolTip("Colabで学習したモデルをGoogle Driveからダウンロード")
        colab_buttons_layout.addWidget(self.colab_download_button)

        # 設定ボタン
        colab_settings_button = QPushButton("設定")
        colab_settings_button.setMaximumWidth(60)
        apply_style(colab_settings_button, 'special')
        colab_settings_button.clicked.connect(self._show_colab_settings)
        colab_buttons_layout.addWidget(colab_settings_button)

        model_mgmt_layout.addLayout(colab_buttons_layout)

        left_layout.addLayout(model_mgmt_layout)

        # --- 表示設定ボタンを追加 ---
        settings_layout = QVBoxLayout()
        
        settings_label = QLabel("表示設定:")
        settings_label.setStyleSheet("font-weight: bold;")
        settings_layout.addWidget(settings_label)
        
        # ボタンを横並びにするレイアウト
        settings_buttons_layout = QHBoxLayout()
        
        settings_button = QPushButton("ウィンドウ・フォントサイズ設定")
        settings_button.clicked.connect(self.show_display_settings)
        apply_style(settings_button, 'special')
        settings_buttons_layout.addWidget(settings_button)
        
        # ダークモード切替ボタンを追加
        self.dark_mode_button = QPushButton("ダークモード")
        self.dark_mode_button.setCheckable(True)
        self.dark_mode_button.clicked.connect(self.toggle_dark_mode)
        apply_style(self.dark_mode_button, 'special')
        settings_buttons_layout.addWidget(self.dark_mode_button)
        
        settings_layout.addLayout(settings_buttons_layout)
        
        left_layout.addLayout(settings_layout)

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
        self.annotation_info_label.setMinimumHeight(45)  # アノテーション情報の最小高さを固定
        self.annotation_info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        info_layout.addWidget(self.annotation_info_label)
        
        self.inference_info_label = QLabel("")
        self.inference_info_label.setWordWrap(True)
        self.inference_info_label.setStyleSheet("color: #009999;")  # ダークシアン（ライトモード対応）
        self.inference_info_label.setMinimumHeight(45)  # 推論結果に十分な高さを設定
        self.inference_info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        info_layout.addWidget(self.inference_info_label)
        
        # 位置推論結果表示ラベル（推論結果の直下）
        self.location_inference_info_label = QLabel("")
        self.location_inference_info_label.setWordWrap(True)
        self.location_inference_info_label.setStyleSheet("color: purple;")  # 紫色で表示して区別
        self.location_inference_info_label.setMinimumHeight(25)  # 固定の最小高さを設定
        self.location_inference_info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        info_layout.addWidget(self.location_inference_info_label)

        # 物体検知推論結果表示ラベル（位置推論結果の下）
        self.detection_inference_info_label = QLabel("")
        self.detection_inference_info_label.setWordWrap(True)
        self.detection_inference_info_label.setStyleSheet("color: #009999;")  # ダークシアン（ライトモード対応）
        self.detection_inference_info_label.setMinimumHeight(40)  # YOLO推論結果は複数行になる可能性があるため高めに設定
        self.detection_inference_info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        info_layout.addWidget(self.detection_inference_info_label)
        
        # 上部のウィジェットと分布グラフの間にスペーサーを入れる
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        info_layout.addWidget(spacer)

        # 分布タイトルとデータ分析ボタン
        graph_title_layout = QHBoxLayout()
        self.graph_title = QLabel("データ分布")
        self.graph_title.setStyleSheet("font-weight: bold; color: #333333;")
        graph_title_layout.addWidget(self.graph_title)

        self.data_analysis_button = QPushButton("分析")
        self.data_analysis_button.setToolTip("アノテーションデータの統計分析と可視化")
        self.data_analysis_button.clicked.connect(self.open_data_analysis)
        self.data_analysis_button.setFixedWidth(50)
        apply_style(self.data_analysis_button, 'primary')
        graph_title_layout.addWidget(self.data_analysis_button)

        info_layout.addLayout(graph_title_layout)

        # 分布グラフ用ラベル - 固定サイズで配置
        self.distribution_label = QLabel()
        self.distribution_label.setAlignment(Qt.AlignCenter)
        self.distribution_label.setFixedHeight(360)  # 高さを20%拡大（300→360）
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
        
        #self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider = DeletedIndexesSlider()
        self.image_slider.setMinimum(0)
        self.image_slider.setMaximum(0) 
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
        
        prev_button = QPushButton("-1")
        prev_button.clicked.connect(lambda: self.skip_images(-1))
        nav_layout.addWidget(prev_button)
        
        next_button = QPushButton("+1")
        next_button.clicked.connect(lambda: self.skip_images(1))
        nav_layout.addWidget(next_button)
        
        self.next_multi_button = QPushButton("▶▶")  # 早送りマーク
        self.next_multi_button.clicked.connect(lambda: self.skip_images(self.skip_count_spin.value()))
        nav_layout.addWidget(self.next_multi_button)
        
        nav_container_layout.addLayout(nav_layout)
        
        # 再生ボタンの配置
        play_layout = QHBoxLayout()
        
        play_layout.addWidget(QLabel("再生:"))
        self.reverse_play_button = QPushButton("◀逆再生")
        self.reverse_play_button.clicked.connect(self.play_reverse)
        play_layout.addWidget(self.reverse_play_button)
        
        self.play_button = QPushButton("▶再生")
        self.play_button.clicked.connect(self.play_forward)
        play_layout.addWidget(self.play_button)
        
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
        clip_layout.addWidget(QLabel("削除範囲指定:"))

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

        # ダウンサンプリング機能（直進時データの間引き）
        downsample_layout = QHBoxLayout()
        downsample_layout.addWidget(QLabel("ダウンサンプリング:"))

        # angle範囲設定
        downsample_layout.addWidget(QLabel("angle範囲:"))
        self.downsample_angle_min = QDoubleSpinBox()
        self.downsample_angle_min.setRange(-1.0, 1.0)
        self.downsample_angle_min.setValue(-0.05)
        self.downsample_angle_min.setSingleStep(0.05)
        self.downsample_angle_min.setDecimals(2)
        self.downsample_angle_min.setFixedWidth(60)
        downsample_layout.addWidget(self.downsample_angle_min)

        downsample_layout.addWidget(QLabel("〜"))

        self.downsample_angle_max = QDoubleSpinBox()
        self.downsample_angle_max.setRange(-1.0, 1.0)
        self.downsample_angle_max.setValue(0.05)
        self.downsample_angle_max.setSingleStep(0.05)
        self.downsample_angle_max.setDecimals(2)
        self.downsample_angle_max.setFixedWidth(60)
        downsample_layout.addWidget(self.downsample_angle_max)

        # 連続フレーム数
        downsample_layout.addWidget(QLabel("連続:"))
        self.downsample_consecutive = QSpinBox()
        self.downsample_consecutive.setRange(2, 100)
        self.downsample_consecutive.setValue(10)
        self.downsample_consecutive.setFixedWidth(50)
        self.downsample_consecutive.setToolTip("この数以上連続した場合にダウンサンプリング対象とする")
        downsample_layout.addWidget(self.downsample_consecutive)

        # 残す間隔
        downsample_layout.addWidget(QLabel("間隔:"))
        self.downsample_keep_every = QSpinBox()
        self.downsample_keep_every.setRange(0, 20)
        self.downsample_keep_every.setValue(3)
        self.downsample_keep_every.setFixedWidth(50)
        self.downsample_keep_every.setToolTip("何枚ごとに1枚残すか（例：3なら3枚中1枚を残す、0なら全て対象）")
        downsample_layout.addWidget(self.downsample_keep_every)

        # 検出ボタン
        self.detect_downsample_button = QPushButton("検出")
        self.detect_downsample_button.clicked.connect(self.detect_downsampling_targets)
        self.detect_downsample_button.setToolTip("条件に該当するインデックスを検出してダウンサンプリング対象に設定")
        self.detect_downsample_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90d9;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3a7fc8;
            }
        """)
        downsample_layout.addWidget(self.detect_downsample_button)

        # ダウンサンプリング数表示ラベル
        self.downsample_count_label = QLabel("0件")
        self.downsample_count_label.setStyleSheet("color: #3366ff;")
        downsample_layout.addWidget(self.downsample_count_label)

        # 左揃えにするためストレッチを追加
        downsample_layout.addStretch()

        nav_container_layout.addLayout(downsample_layout)

        # Throttleダウンサンプリング機能（低速走行時データの間引き）
        throttle_downsample_layout = QHBoxLayout()
        throttle_downsample_layout.addWidget(QLabel("                  "))

        # throttle範囲設定
        throttle_downsample_layout.addWidget(QLabel("throttle範囲:"))
        self.downsample_throttle_min = QDoubleSpinBox()
        self.downsample_throttle_min.setRange(-1.0, 1.0)
        self.downsample_throttle_min.setValue(-0.05)
        self.downsample_throttle_min.setSingleStep(0.05)
        self.downsample_throttle_min.setDecimals(2)
        self.downsample_throttle_min.setFixedWidth(60)
        throttle_downsample_layout.addWidget(self.downsample_throttle_min)

        throttle_downsample_layout.addWidget(QLabel("〜"))

        self.downsample_throttle_max = QDoubleSpinBox()
        self.downsample_throttle_max.setRange(-1.0, 1.0)
        self.downsample_throttle_max.setValue(0.05)
        self.downsample_throttle_max.setSingleStep(0.05)
        self.downsample_throttle_max.setDecimals(2)
        self.downsample_throttle_max.setFixedWidth(60)
        throttle_downsample_layout.addWidget(self.downsample_throttle_max)

        # 連続フレーム数
        throttle_downsample_layout.addWidget(QLabel("連続:"))
        self.downsample_throttle_consecutive = QSpinBox()
        self.downsample_throttle_consecutive.setRange(2, 100)
        self.downsample_throttle_consecutive.setValue(3)
        self.downsample_throttle_consecutive.setFixedWidth(50)
        self.downsample_throttle_consecutive.setToolTip("この数以上連続した場合にダウンサンプリング対象とする")
        throttle_downsample_layout.addWidget(self.downsample_throttle_consecutive)

        # 残す間隔
        throttle_downsample_layout.addWidget(QLabel("間隔:"))
        self.downsample_throttle_keep_every = QSpinBox()
        self.downsample_throttle_keep_every.setRange(0, 20)
        self.downsample_throttle_keep_every.setValue(0)
        self.downsample_throttle_keep_every.setFixedWidth(50)
        self.downsample_throttle_keep_every.setToolTip("何枚ごとに1枚残すか（例：3なら3枚中1枚を残す、0なら全て対象）")
        throttle_downsample_layout.addWidget(self.downsample_throttle_keep_every)

        # 検出ボタン
        self.detect_throttle_downsample_button = QPushButton("検出")
        self.detect_throttle_downsample_button.clicked.connect(self.detect_throttle_downsampling_targets)
        self.detect_throttle_downsample_button.setToolTip("条件に該当するインデックスを検出してダウンサンプリング対象に設定")
        self.detect_throttle_downsample_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90d9;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3a7fc8;
            }
        """)
        throttle_downsample_layout.addWidget(self.detect_throttle_downsample_button)

        # ダウンサンプリング数表示ラベル（throttle用）
        self.throttle_downsample_count_label = QLabel("(0件)")
        self.throttle_downsample_count_label.setStyleSheet("color: #3366ff;")
        throttle_downsample_layout.addWidget(self.throttle_downsample_count_label)

        # 解除ボタン（全てのダウンサンプリング対象を解除）
        clear_throttle_downsample_button = QPushButton("解除")
        clear_throttle_downsample_button.clicked.connect(self.clear_downsampling_targets)
        clear_throttle_downsample_button.setToolTip("ダウンサンプリング対象をすべて解除")
        throttle_downsample_layout.addWidget(clear_throttle_downsample_button)

        # 左揃えにするためストレッチを追加
        throttle_downsample_layout.addStretch()

        nav_container_layout.addLayout(throttle_downsample_layout)

        # ナビゲーションコンテナをメイン画像コンテナに追加
        main_image_container.addWidget(nav_container)
        
        # 中央パネルをメインパネルに追加 - 比率4に設定
        main_panel_layout.addLayout(main_image_container, 4)
        
        # 3. 右側の位置情報パネル - 既存のright_layoutをそのまま利用、比率1に設定
        location_panel = QWidget()
        location_layout = QVBoxLayout(location_panel)
        location_layout.setSpacing(5)
        
        mode_layout_label = QLabel("アノテーションモード:")
        mode_layout_label.setStyleSheet("font-weight: bold;")
        location_layout.addWidget(mode_layout_label)

        # アノテーションモード切替ボタン
        mode_layout = QHBoxLayout()
        
        self.auto_mode_button = QPushButton("自動運転")
        self.auto_mode_button.setCheckable(True)
        self.auto_mode_button.setChecked(True)
        self.auto_mode_button.clicked.connect(self.toggle_annotation_mode)
        self.auto_mode_button.setToolTip(
            "自動運転アノテーションモード\n"
            "・画像上をクリックして角度(angle)とスロットル(throttle)を設定\n"
            "・左クリック: ポイント追加/移動\n"
            "・右クリック: ポイント削除\n"
            "・数字キー(0-7): 運転位置を設定（同じキー再押下で解除）\n"
            "・Deleteキー: 現在の画像のアノテーション（angle/throttle/位置）を削除"
        )

        self.detection_mode_button = QPushButton("物体検知")
        self.detection_mode_button.setCheckable(True)
        self.detection_mode_button.clicked.connect(self.toggle_annotation_mode)
        self.detection_mode_button.setToolTip(
            "物体検知アノテーションモード\n"
            "・ドラッグしてバウンディングボックスを作成\n"
            "・作成したボックスをクリックして選択/移動\n"
            "・ボックスの角をドラッグしてサイズ調整\n"
            "・右クリック: 選択したボックスを削除\n"
            "・Deleteキー: 選択したボックスを削除"
        )

        # 新規追加: セグメンテーションモードボタン
        self.segmentation_mode_button = QPushButton("セグメンテーション")
        self.segmentation_mode_button.setCheckable(True)
        self.segmentation_mode_button.clicked.connect(self.toggle_annotation_mode)
        self.segmentation_mode_button.setToolTip(
            "セグメンテーションアノテーションモード\n"
            "・左クリック: ポリゴン頂点を追加\n"
            "・右クリック: ポリゴンを閉じる/完成させる\n"
            "・ポリゴン上で右クリック: 新しい頂点を追加\n"
            "・頂点をドラッグ: 頂点位置を調整\n"
            "・Deleteキー: 選択したポリゴンを削除\n"
            "・Escキー: 作成中のポリゴンをキャンセル"
        )

        # 新規追加: waypointモードボタン
        self.waypoint_mode_button = QPushButton("ウェイポイント")
        self.waypoint_mode_button.setCheckable(True)
        self.waypoint_mode_button.clicked.connect(self.toggle_annotation_mode)
        self.waypoint_mode_button.setToolTip(
            "waypointアノテーションモード\n"
            "・左クリック: waypoint座標を追加\n"
            "・右クリック: 最後のwaypointを削除\n"
            "・緑色の丸でwaypointが表示されます\n"
            "・Deleteキー: 現在の画像のwaypointを全削除"
        )

        mode_layout.addWidget(self.auto_mode_button)
        mode_layout.addWidget(self.detection_mode_button)
        mode_layout.addWidget(self.segmentation_mode_button)  # 追加
        mode_layout.addWidget(self.waypoint_mode_button)  # 追加

        location_layout.addLayout(mode_layout)

        # waypoint制御パネル
        self.waypoint_control_widget = QWidget()
        waypoint_control_layout = QVBoxLayout(self.waypoint_control_widget)
        waypoint_control_layout.setContentsMargins(5, 5, 5, 5)

        # waypoint制御ラベル
        waypoint_label = QLabel("打点制御:")
        waypoint_label.setStyleSheet("font-weight: bold; color: #333;")
        waypoint_control_layout.addWidget(waypoint_label)

        # 打点数制御
        # 打点数と打点位置（横並び）
        points_and_pos_layout = QHBoxLayout()

        points_and_pos_layout.addWidget(QLabel("打点数:"))
        self.waypoint_count_spin = QSpinBox()
        self.waypoint_count_spin.setRange(1, 20)
        self.waypoint_count_spin.setValue(4)  # デフォルト4
        self.waypoint_count_spin.setToolTip("配置するwaypoint数")
        self.waypoint_count_spin.valueChanged.connect(self.update_waypoint_guidelines)
        points_and_pos_layout.addWidget(self.waypoint_count_spin)

        points_and_pos_layout.addSpacing(20)
        points_and_pos_layout.addWidget(QLabel("打点位置:"))

        # 開始Y位置
        self.waypoint_start_y_spin = QSpinBox()
        self.waypoint_start_y_spin.setRange(0, 1000)
        self.waypoint_start_y_spin.setValue(200)  # デフォルト値を200に変更
        self.waypoint_start_y_spin.setToolTip("waypoint開始位置のY座標")
        self.waypoint_start_y_spin.valueChanged.connect(self.update_waypoint_guidelines)
        points_and_pos_layout.addWidget(self.waypoint_start_y_spin)

        points_and_pos_layout.addWidget(QLabel("~"))

        # 終了Y位置
        self.waypoint_end_y_spin = QSpinBox()
        self.waypoint_end_y_spin.setRange(0, 1000)
        self.waypoint_end_y_spin.setValue(120)  # デフォルト値を120に変更
        self.waypoint_end_y_spin.setToolTip("waypoint終了位置のY座標")
        self.waypoint_end_y_spin.valueChanged.connect(self.update_waypoint_guidelines)
        points_and_pos_layout.addWidget(self.waypoint_end_y_spin)

        waypoint_control_layout.addLayout(points_and_pos_layout)

        # waypointモード選択ラジオボタン（横並び）
        waypoint_mode_layout = QHBoxLayout()

        # ラジオボタングループを作成
        self.waypoint_mode_button_group = QButtonGroup(self)

        # waypoint配置完了時の自動遷移モード
        self.auto_advance_waypoint_radio = QRadioButton("自動遷移")
        self.auto_advance_waypoint_radio.setChecked(True)  # デフォルトを自動遷移に変更
        self.auto_advance_waypoint_radio.setToolTip("最後のwaypointが配置されたら自動で次の画像に遷移")
        self.waypoint_mode_button_group.addButton(self.auto_advance_waypoint_radio, 1)
        waypoint_mode_layout.addWidget(self.auto_advance_waypoint_radio)

        # 前回waypoint自動適用モード
        self.apply_last_waypoint_radio = QRadioButton("前回のウエイポイントを適用")
        self.apply_last_waypoint_radio.setToolTip("前回の画像のwaypointを次の画像に自動適用")
        self.waypoint_mode_button_group.addButton(self.apply_last_waypoint_radio, 0)
        waypoint_mode_layout.addWidget(self.apply_last_waypoint_radio)

        # ラジオボタンの変更を監視
        self.waypoint_mode_button_group.buttonClicked.connect(self.on_waypoint_mode_changed)

        waypoint_control_layout.addLayout(waypoint_mode_layout)


        # 初期状態では非表示
        self.waypoint_control_widget.setVisible(False)
        location_layout.addWidget(self.waypoint_control_widget)

        # セグメンテーション制御パネル
        self.segmentation_control_widget = QWidget()
        segmentation_control_layout = QVBoxLayout(self.segmentation_control_widget)
        segmentation_control_layout.setContentsMargins(5, 5, 5, 5)

        # セグメンテーション制御ラベル
        segmentation_label = QLabel("走行方向計算:")
        segmentation_label.setStyleSheet("font-weight: bold; color: #333;")
        segmentation_control_layout.addWidget(segmentation_label)

        # 走行方向矢印表示チェックボックス
        self.show_seg_driving_direction_checkbox = QCheckBox("走行方向を表示")
        self.show_seg_driving_direction_checkbox.setChecked(False)
        self.show_seg_driving_direction_checkbox.setToolTip("セグメンテーション推論結果から走行方向を計算して矢印で表示")
        self.show_seg_driving_direction_checkbox.stateChanged.connect(self.toggle_seg_driving_direction)
        segmentation_control_layout.addWidget(self.show_seg_driving_direction_checkbox)

        # クラスIDとY座標の設定（横並び）
        seg_params_layout = QHBoxLayout()

        seg_params_layout.addWidget(QLabel("クラスID:"))
        self.seg_class_id_spin = QSpinBox()
        self.seg_class_id_spin.setRange(0, 99)
        self.seg_class_id_spin.setValue(0)  # デフォルト0
        self.seg_class_id_spin.setToolTip("走行可能エリアのセグメンテーションクラスID")
        self.seg_class_id_spin.valueChanged.connect(self.update_seg_driving_direction_class)
        seg_params_layout.addWidget(self.seg_class_id_spin)

        seg_params_layout.addSpacing(20)
        seg_params_layout.addWidget(QLabel("Y座標:"))
        self.seg_direction_y_spin = QSpinBox()
        self.seg_direction_y_spin.setRange(0, 1000)
        self.seg_direction_y_spin.setValue(100)  # デフォルト100
        self.seg_direction_y_spin.setToolTip("走行方向計算に使用するY座標（画像上からのピクセル）")
        self.seg_direction_y_spin.valueChanged.connect(self.update_seg_driving_direction_y)
        seg_params_layout.addWidget(self.seg_direction_y_spin)

        segmentation_control_layout.addLayout(seg_params_layout)

        # 最大舵角の設定
        steering_layout = QHBoxLayout()
        steering_layout.addWidget(QLabel("最大舵角:"))
        self.seg_max_steering_spin = QDoubleSpinBox()
        self.seg_max_steering_spin.setRange(0.1, 90.0)
        self.seg_max_steering_spin.setValue(30.0)  # デフォルト30度
        self.seg_max_steering_spin.setSuffix("°")
        self.seg_max_steering_spin.setDecimals(1)
        self.seg_max_steering_spin.setToolTip("走行軌跡計算に使用する最大舵角（度）")
        self.seg_max_steering_spin.valueChanged.connect(self.update_seg_max_steering_angle)
        steering_layout.addWidget(self.seg_max_steering_spin)
        steering_layout.addStretch()
        segmentation_control_layout.addLayout(steering_layout)

        # 表示モード選択
        display_mode_layout = QHBoxLayout()
        display_mode_layout.addWidget(QLabel("表示モード:"))

        # ラジオボタングループを作成
        self.seg_display_mode_button_group = QButtonGroup(self)

        # 軌跡表示モード
        self.seg_trajectory_mode_radio = QRadioButton("軌跡")
        self.seg_trajectory_mode_radio.setChecked(True)
        self.seg_trajectory_mode_radio.setToolTip("走行軌跡を円弧で表示")
        self.seg_display_mode_button_group.addButton(self.seg_trajectory_mode_radio, 0)
        display_mode_layout.addWidget(self.seg_trajectory_mode_radio)

        # ウェイポイント表示モード
        self.seg_waypoint_mode_radio = QRadioButton("ウェイポイント")
        self.seg_waypoint_mode_radio.setToolTip("目標Y座標までのウェイポイント（4点等間隔）を表示")
        self.seg_display_mode_button_group.addButton(self.seg_waypoint_mode_radio, 1)
        display_mode_layout.addWidget(self.seg_waypoint_mode_radio)

        display_mode_layout.addStretch()
        segmentation_control_layout.addLayout(display_mode_layout)

        # ラジオボタンの変更を監視
        self.seg_display_mode_button_group.buttonClicked.connect(self.on_seg_display_mode_changed)

        # 初期状態では非表示
        self.segmentation_control_widget.setVisible(False)
        location_layout.addWidget(self.segmentation_control_widget)

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

        # セグメンテーション用の前回適用チェックボックス
        self.apply_last_segmentation_checkbox = QCheckBox("前回のセグメンテーションを適用")
        self.apply_last_segmentation_checkbox.setChecked(False)
        self.apply_last_segmentation_checkbox.setToolTip("前回作成したセグメンテーションを現在の画像にも適用します")
        self.apply_last_segmentation_checkbox.stateChanged.connect(self.toggle_auto_apply_segmentation)
        location_layout.addWidget(self.apply_last_segmentation_checkbox)

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
        self.gallery_layout.setContentsMargins(0, 0, 0, 0)  # マージンをゼロに
        self.gallery_layout.setSpacing(2)
        
        gallery_scroll = QScrollArea()
        gallery_scroll.setWidgetResizable(True)
        gallery_scroll.setWidget(self.gallery_widget)
        gallery_scroll.setMinimumHeight(GALLERY_MIN_HEIGHT)
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


    def update_ui(self):
            """アノテーション変更後のUI更新を一括処理"""
            # メイン画像表示を更新
            if hasattr(self, 'main_image_view'):
                self.main_image_view.update()
            
            if hasattr(self, 'update_slider_deleted_indexes'):
                self.update_slider_deleted_indexes()

            # ギャラリー表示を更新
            if hasattr(self, 'update_gallery'):
                self.update_gallery()
            
            # 情報パネルを更新
            if hasattr(self, 'display_current_image'):
                self.display_current_image()
            
            # 統計情報を更新
            if hasattr(self, 'update_driving_annotation_stats'):
                self.update_driving_annotation_stats()
            
            # 推論情報パネルを更新（各関数内部でチェックボックス状態を確認）
            if hasattr(self, 'update_inference_display'):
                self.update_inference_display()
            
            # 物体検知推論情報パネルを更新
            if hasattr(self, 'update_detection_info_panel'):
                self.update_detection_info_panel()
            
            # 自動運転推論情報パネルを更新
            if hasattr(self, 'update_driving_info_panel'):
                self.update_driving_info_panel()

            # 位置情報パネルを更新
            if hasattr(self, 'update_location_info_panel'):
                self.update_location_info_panel()

            # 統計情報を更新
            if hasattr(self, 'update_bbox_stats'):
                self.update_bbox_stats()
            
            if hasattr(self, 'update_segmentation_stats'):
                self.update_segmentation_stats()
            
            # 分布グラフを更新（自動運転アノテーションがある場合）
            if hasattr(self, 'update_distribution_graph') and hasattr(self, 'annotations') and self.annotations:
                self.update_distribution_graph()
            
            # 位置情報ボタンのカウント表示を更新
            if hasattr(self, 'update_location_button_counts'):
                self.update_location_button_counts()
                
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
    
    def closeEvent(self, event):
        """アプリケーション終了時の処理"""
        # セッション情報を保存
        self.save_session_info()
        
        # 親クラスのcloseEventを呼び出す
        super().closeEvent(event)
        event.accept()

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
        """アノテーションの角度とスロットル値の分布を縦並びのヒストグラムで表示
        元データ（薄い色）とダウンサンプリング後（濃い色）を重ねて表示"""

        if not self.annotations:
            # アノテーションがない場合は空のグラフを表示
            self.distribution_label.clear()
            self.distribution_label.setText("アノテーションがありません")
            return

        # 既存のアノテーションからangleとthrottleの値を抽出
        # 元データ（削除済みのみ除外）とダウンサンプリング後（削除済み+ダウンサンプリング対象を除外）
        angles_orig = []
        throttles_orig = []
        angles_ds = []
        throttles_ds = []

        for idx, anno in self.annotations.items():
            # 削除済みインデックスをスキップ
            if hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes:
                continue

            if 'angle' in anno and 'throttle' in anno:
                # 元データに追加
                angles_orig.append(anno['angle'])
                throttles_orig.append(anno['throttle'])

                # ダウンサンプリング対象でなければダウンサンプリング後データにも追加
                if not (hasattr(self, 'downsampled_indexes') and idx in self.downsampled_indexes):
                    angles_ds.append(anno['angle'])
                    throttles_ds.append(anno['throttle'])

        # データがない場合は終了
        if not angles_orig or not throttles_orig:
            self.distribution_label.clear()
            self.distribution_label.setText("有効なアノテーションがありません")
            return

        # グラフ作成
        try:
            # 情報パネルの幅に合わせてグラフサイズを調整
            panel_width = self.info_panel_width - self.info_panel_margin
            # 縦に2つのグラフを配置（縦長）
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(panel_width/100, 3.6))

            bins = 20

            # ダウンサンプリングが有効かどうか
            has_downsampling = hasattr(self, 'downsampled_indexes') and len(self.downsampled_indexes) > 0

            # === Angle分布 ===
            if has_downsampling:
                # 元データを薄い色で表示（背景）
                ax1.hist(angles_orig, bins=bins, color='steelblue', edgecolor='none',
                        alpha=0.25, label=f'元: {len(angles_orig)}')
                # ダウンサンプリング後のデータを濃い色で表示（前景）
                if angles_ds:
                    ax1.hist(angles_ds, bins=bins, color='steelblue', edgecolor='white',
                            alpha=0.8, label=f'DS後: {len(angles_ds)}')
                ax1.legend(fontsize=6, loc='upper right')
            else:
                # ダウンサンプリングなしの場合は通常表示
                ax1.hist(angles_orig, bins=bins, color='steelblue', edgecolor='white',
                        alpha=0.7)

            # スタイル設定
            ax1.set_title(f'Angle (n={len(angles_ds) if has_downsampling else len(angles_orig)})', fontsize=10)
            ax1.tick_params(axis='both', which='major', labelsize=7)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-1.05, 1.05)

            # === Throttle分布 ===
            if has_downsampling:
                # 元データを薄い色で表示（背景）
                ax2.hist(throttles_orig, bins=bins, color='forestgreen', edgecolor='none',
                        alpha=0.25, label=f'元: {len(throttles_orig)}')
                # ダウンサンプリング後のデータを濃い色で表示（前景）
                if throttles_ds:
                    ax2.hist(throttles_ds, bins=bins, color='forestgreen', edgecolor='white',
                            alpha=0.8, label=f'DS後: {len(throttles_ds)}')
                ax2.legend(fontsize=6, loc='upper right')
            else:
                # ダウンサンプリングなしの場合は通常表示
                ax2.hist(throttles_orig, bins=bins, color='forestgreen', edgecolor='white',
                        alpha=0.7)

            # スタイル設定
            ax2.set_title(f'Throttle (n={len(throttles_ds) if has_downsampling else len(throttles_orig)})', fontsize=10)
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

            # グラフタイトルラベルを更新
            if hasattr(self, 'distribution_label'):
                self.distribution_label.setPixmap(pixmap)
                self.distribution_label.setScaledContents(True)

            # 後始末
            plt.close(fig)

        except Exception as e:
            print(f"分布グラフ作成中にエラー: {str(e)}")
            traceback.print_exc()
            self.distribution_label.setText(f"グラフ作成エラー: {str(e)}")

    def _schedule_distribution_graph_update(self):
        """分布グラフ更新をスケジュール（デバウンス処理）"""
        # 既存のタイマーをリセットして新たにスケジュール
        if hasattr(self, 'graph_update_timer'):
            self.graph_update_timer.stop()
            self.graph_update_timer.start(self.graph_update_delay)

    def _execute_distribution_graph_update(self):
        """分布グラフ更新を実際に実行（タイマーから呼ばれる）"""
        self.update_distribution_graph()

    def _schedule_gallery_update(self):
        """ギャラリー更新をスケジュール（デバウンス処理）"""
        # 既存のタイマーをリセットして新たにスケジュール
        if hasattr(self, 'gallery_update_timer'):
            self.gallery_update_timer.stop()
            self.gallery_update_timer.start(self.gallery_update_delay)

    def _execute_gallery_update(self):
        """ギャラリー更新を実際に実行（タイマーから呼ばれる）"""
        self.update_gallery()

    def _schedule_inference(self):
        """推論実行をスケジュール（デバウンス処理）"""
        # 既存のタイマーをリセットして新たにスケジュール
        if hasattr(self, 'inference_timer'):
            self.inference_timer.stop()
            self.inference_timer.start(self.inference_delay)

    def _execute_deferred_inference(self):
        """推論を実際に実行（タイマーから呼ばれる）"""
        if not self.images or not hasattr(self, 'inference_checkbox'):
            return

        if self.inference_checkbox.isChecked():
            current_img_path = self.images[self.current_index]
            # 推論結果がまだない場合のみ推論を実行
            if self.current_index not in self.inference_results:
                self.run_inference_check(False)

    def update_segmentation_stats(self):
        """セグメンテーションアノテーションの統計情報を更新"""
        seg_count = len(self.segmentation_annotations) if hasattr(self, 'segmentation_annotations') else 0
        
        # Update the stats through the comprehensive update method
        if hasattr(self, 'update_driving_annotation_stats'):
            self.update_driving_annotation_stats()
    
    def update_driving_annotation_stats(self):
        """運転アノテーションの統計情報を更新"""
        if not hasattr(self, 'stats_label'):
            return
            
        # アノテーション数の更新
        self.annotated_count = len(self.annotations)
        total_images = len(self.images) if self.images else 0
        
        # 統計ラベルのテキストを構築
        stats_text = f"アノテーション済み: {self.annotated_count} / {total_images}"
        
        # バウンディングボックスの統計情報を追加
        if hasattr(self, 'bbox_annotations'):
            bbox_count = len(self.bbox_annotations) if self.bbox_annotations else 0
            if bbox_count > 0:
                stats_text += f" | bbox_images: {bbox_count}"
        
        # セグメンテーションの統計情報を追加
        if hasattr(self, 'segmentation_annotations'):
            seg_count = len(self.segmentation_annotations) if self.segmentation_annotations else 0
            if seg_count > 0:
                stats_text += f" | seg_images: {seg_count}"
        
        # ラベルを更新
        self.stats_label.setText(stats_text)

    def eventFilter(self, obj, event):
        # キーイベントを処理
        if event.type() == QEvent.KeyPress:
            key = event.key()

            # テキスト入力フィールドにフォーカスがある場合は、そのフィールドに処理を委ねる
            focused_widget = QApplication.focusWidget()
    
            # テキスト入力系のウィジェットの場合は、アプリケーション固有のキー処理をスキップ
            if isinstance(focused_widget, (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox)):
                # ただし、一部のキーは例外的に処理する
                if key in [Qt.Key_F1, Qt.Key_F2, Qt.Key_F3, Qt.Key_F4, Qt.Key_F5]:
                    # ファンクションキーは処理を続行
                    pass
                else:
                    # その他のキーはウィジェットに委ねる（親クラスのイベントフィルターを呼び出す）
                    return super().eventFilter(obj, event)

            # Bキーでアノテーションモード切り替え
            if key == Qt.Key_B:
                self.toggle_annotation_mode()
                return True

            # 左右矢印キーでの画像移動
            elif key == Qt.Key_Left:
                # 自動スキップ設定に応じてスキップ枚数を決定
                skip_count = self.skip_count_spin.value() if self.skip_images_on_click.isChecked() else 1
                self.skip_images(-skip_count)
                return True
            elif key == Qt.Key_Right:
                # 自動スキップ設定に応じてスキップ枚数を決定
                skip_count = self.skip_count_spin.value() if self.skip_images_on_click.isChecked() else 1
                self.skip_images(skip_count)
                return True

            # 削除キーが押された場合の処理
            elif key in [Qt.Key_Delete, Qt.Key_Backspace]:
                if self.current_mode == 0:  # 自動運転モードの場合
                    # 現在の画像の自動運転アノテーションを削除
                    self.delete_current_driving_annotation()
                elif self.current_mode == 1:  # 物体検知モードの場合
                    if self.main_image_view.selected_bbox_index is not None:
                        self.delete_selected_bbox()
                elif self.current_mode == 2:  # セグメンテーションモードの場合
                    # 頂点が選択されている場合は頂点のみを削除
                    if (self.main_image_view.selected_polygon_index is not None and
                        self.main_image_view.selected_vertex_index is not None):
                        self.delete_selected_vertex()
                    # セグメンテーション全体が選択されている場合はセグメンテーションを削除
                    elif self.main_image_view.selected_segmentation_index is not None:
                        self.delete_selected_segmentation()
                elif self.current_mode == 3:  # waypointモードの場合
                    # 現在の画像のwaypointアノテーションを全削除
                    self.delete_current_waypoint_annotations()
                return True
            
            # 数字キー（0-7）による位置アノテーション（自動運転モードのみ）
            elif key in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6, Qt.Key_7]:
                if self.current_mode == 0:  # 自動運転モードの場合のみ
                    # キーコードから数字を取得
                    location_value = key - Qt.Key_0  # Qt.Key_0 = 48, Qt.Key_1 = 49, ...
                    
                    # 同じ位置が既に選択されている場合は解除、そうでなければ設定
                    # set_locationメソッド内で自動的に判定されるため、そのまま呼び出す
                    self.set_location(location_value)
                    return True  

        # 親クラスのイベントフィルタを呼び出す
        return super().eventFilter(obj, event)

    def delete_selected_bbox(self):
        """選択されたバウンディングボックスを削除する"""
        if not self.images or not hasattr(self, 'main_image_view'):
            return
        
        selected_index = self.main_image_view.selected_bbox_index
        if selected_index is None:
            # 選択されていない場合は何もしない
            return
        
        # インデックスベースに変更
        current_index = self.current_index
        if (current_index is not None and 
            isinstance(current_index, int) and 
            current_index in self.bbox_annotations and 
            selected_index is not None):
            
            bboxes = self.bbox_annotations[current_index]
            if 0 <= selected_index < len(bboxes):
                # ボックス情報を取得
                bbox = bboxes[selected_index]
                class_name = bbox.get('class', 'unknown')
                                
                # 削除実行
                del bboxes[selected_index]
                
                # もしこの画像のバウンディングボックスが全てなくなった場合、辞書のキーを削除
                if not bboxes:  # リストが空になった場合
                    del self.bbox_annotations[current_index]
                
                # 選択をクリア
                self.main_image_view.selected_bbox_index = None

                # 重要: last_bboxesを更新する（削除後の最新状態を保存）
                if hasattr(self, 'last_bboxes'):
                    # 現在の画像のバウンディングボックスリストを全て取得して保存
                    self.last_bboxes = [bbox.copy() for bbox in bboxes]
                    
                    # last_bboxも更新（互換性のため）
                    if self.last_bboxes:
                        self.last_bbox = self.last_bboxes[-1].copy()
                    else:
                        self.last_bbox = None

                # 画面更新
                self.update_ui()
                
                # 確認メッセージ
                self.statusBar().showMessage(f"'{class_name}' のバウンディングボックスを削除しました", 3000)

    def delete_selected_vertex(self):
        """選択されたセグメンテーションの頂点を削除する"""
        if not self.images or not hasattr(self, 'segmentation_annotations'):
            return

        current_index = self.current_index
        if current_index is None or not isinstance(current_index, int):
            return

        polygon_index = self.main_image_view.selected_polygon_index
        vertex_index = self.main_image_view.selected_vertex_index

        if polygon_index is None or vertex_index is None:
            return

        if current_index not in self.segmentation_annotations:
            return

        segmentations = self.segmentation_annotations[current_index]
        if not (0 <= polygon_index < len(segmentations)):
            return

        seg_data = segmentations[polygon_index]
        points = seg_data.get('points', [])

        # 頂点が3つ以下の場合は削除できない（ポリゴンとして成立しない）
        if len(points) <= 3:
            QMessageBox.warning(
                self,
                "警告",
                "ポリゴンは最低3つの頂点が必要です。\n頂点を削除できません。"
            )
            return

        if not (0 <= vertex_index < len(points)):
            return

        # 頂点を削除
        class_name = seg_data.get('class', 'unknown')
        del points[vertex_index]

        # 選択状態をクリア
        self.main_image_view.selected_vertex_index = None
        self.main_image_view.selected_polygon_index = None

        # 画面を更新
        self.display_current_image()

        # ステータスバーに情報表示
        self.statusBar().showMessage(
            f"'{class_name}' の頂点を削除しました（残り{len(points)}個の頂点）",
            3000
        )

    def delete_selected_segmentation(self, index=None):
        """選択されたセグメンテーションを削除する"""
        if not self.images or not hasattr(self, 'segmentation_annotations'):
            return

        # インデックスベースに変更
        current_index = self.current_index

        # current_indexの有効性をチェック
        if current_index is None or not isinstance(current_index, int):
            return

        if index is None:
            index = self.main_image_view.selected_segmentation_index

        if (current_index in self.segmentation_annotations and
            index is not None):
            segmentations = self.segmentation_annotations[current_index]
            if 0 <= index < len(segmentations):
                # 現在情報を取得
                seg = segmentations[index]
                class_name = seg.get('class', 'unknown')

                # 削除実行
                del segmentations[index]

                # もしこの画像のセグメンテーションが全てなくなった場合、辞書のキーを削除
                if not segmentations:  # リストが空になった場合
                    del self.segmentation_annotations[current_index]

                # 選択をクリア
                self.main_image_view.selected_segmentation_index = None

                # 削除後の最新状態を保存
                if hasattr(self, 'last_segmentations'):
                    # 現在のリストを全て取得して保存
                    self.last_segmentations = [seg.copy() for seg in segmentations]

                    # last_segmentationも更新（互換性のため）
                    if self.last_segmentations:
                        self.last_segmentation = self.last_segmentations[-1].copy()
                    else:
                        self.last_segmentation = None

                # 画面更新
                self.update_ui()

                # 確認メッセージ
                self.statusBar().showMessage(f"'{class_name}' のセグメンテーションを削除しました", 3000)

    def delete_current_driving_annotation(self):
        """現在の画像の自動運転アノテーション（angle/throttle/位置）を削除する"""
        if not self.images:
            return
        
        current_index = self.current_index
        if current_index is None:
            return
        
        deleted_items = []
        
        # angle/throttleアノテーションを削除（インデックスベース）
        if current_index in self.annotations:
            annotation = self.annotations[current_index]
            
            # angleを削除（キーが存在する場合は無条件で削除）
            if 'angle' in annotation:
                del annotation['angle']
                deleted_items.append('angle')
            
            # throttleを削除（キーが存在する場合は無条件で削除）
            if 'throttle' in annotation:
                del annotation['throttle']
                deleted_items.append('throttle')
            
            # 位置情報を削除
            if 'loc' in annotation:
                del annotation['loc']
                deleted_items.append('位置情報')
            
            # 座標情報も削除（赤丸表示用）
            if 'x' in annotation:
                del annotation['x']
            if 'y' in annotation:
                del annotation['y']
            
            # アノテーションが空になった場合は削除
            if not annotation:
                del self.annotations[current_index]
        
        # パスベースのアノテーションも削除（重複格納対策）
        if hasattr(self, 'images') and 0 <= current_index < len(self.images):
            current_img_path = self.images[current_index]
            if current_img_path in self.annotations:
                path_annotation = self.annotations[current_img_path]
                
                # angleを削除
                if 'angle' in path_annotation:
                    del path_annotation['angle']
                    if 'angle' not in deleted_items:
                        deleted_items.append('angle (path)')
                
                # throttleを削除
                if 'throttle' in path_annotation:
                    del path_annotation['throttle']
                    if 'throttle' not in deleted_items:
                        deleted_items.append('throttle (path)')
                
                # 位置情報を削除
                if 'loc' in path_annotation:
                    del path_annotation['loc']
                    if '位置情報' not in deleted_items:
                        deleted_items.append('位置情報 (path)')
                
                # 座標情報も削除
                if 'x' in path_annotation:
                    del path_annotation['x']
                if 'y' in path_annotation:
                    del path_annotation['y']
                
                # アノテーションが空になった場合は削除
                if not path_annotation:
                    del self.annotations[current_img_path]
        
        # 位置アノテーションを削除
        if current_index in self.location_annotations:
            del self.location_annotations[current_index]
            if '位置情報' not in deleted_items:
                deleted_items.append('位置情報')
        
        # 現在の位置選択をクリア
        self.current_location = None
        
        # 位置ボタンの選択状態をクリア
        for button in self.location_buttons:
            button.setChecked(False)
        
        # 位置情報ラベルを更新
        self.current_location_label.setText("現在の位置情報: なし")
        self.current_location_label.setStyleSheet("")
        
        # アノテーションポイントもクリア（赤丸表示を削除）
        if hasattr(self.main_image_view, 'annotation_point'):
            self.main_image_view.annotation_point = None
        
        # 画面更新
        self.display_current_image()
        
        # 明示的にメイン画像ビューを更新
        if hasattr(self.main_image_view, 'update'):
            self.main_image_view.update()
        
        self.update_gallery()  # ギャラリー更新を追加
        self.update_driving_annotation_stats()
        
        # 分布グラフも更新
        if hasattr(self, 'update_distribution_graph'):
            self.update_distribution_graph()
        
        # デバッグ：削除後のアノテーション内容を確認
        
        
        # 確認メッセージ
        if deleted_items:
            items_str = '、'.join(deleted_items)
            self.statusBar().showMessage(f"自動運転アノテーション ({items_str}) を削除しました", 3000)
        else:
            self.statusBar().showMessage("削除するアノテーションがありませんでした", 3000)

    def delete_current_waypoint_annotations(self):
        """現在の画像のwaypointアノテーションを全削除する"""
        if not self.images:
            return

        current_index = self.current_index
        if current_index is None:
            return

        # waypointアノテーションを削除
        deleted_count = 0
        if current_index in self.waypoint_annotations:
            deleted_count = len(self.waypoint_annotations[current_index])
            del self.waypoint_annotations[current_index]

        # 画面更新
        self.display_current_image()

        # 明示的にメイン画像ビューを更新
        if hasattr(self.main_image_view, 'update'):
            self.main_image_view.update()

        self.update_gallery()

        # 確認メッセージ
        if deleted_count > 0:
            self.statusBar().showMessage(f"waypoint {deleted_count}個を削除しました", 3000)
        else:
            self.statusBar().showMessage("削除するwaypointがありませんでした", 3000)

    def _check_waypoint_count_before_transition(self):
        """画像遷移前にwaypoint数をチェックする

        Returns:
            bool: 遷移可能な場合True、遷移を中止する場合False
        """
        # waypointモードでない場合は常にOK
        if not hasattr(self, 'current_mode') or self.current_mode != 3:
            return True

        # waypointコントロールが表示されていない場合はOK
        if not hasattr(self, 'waypoint_control_widget') or not self.waypoint_control_widget.isVisible():
            return True

        # 再生中の場合はチェックをスキップ
        if hasattr(self, 'auto_play_timer') and self.auto_play_timer.isActive():
            return True

        # 現在の画像にwaypointアノテーションがない場合はOK（未アノテーション画像）
        current_index = self.current_index
        if current_index not in self.waypoint_annotations:
            return True

        # 必要なwaypoint数を取得
        target_count = self.waypoint_count_spin.value() if hasattr(self, 'waypoint_count_spin') else 4

        # 現在のwaypoint数を取得
        current_waypoints = self.waypoint_annotations.get(current_index, [])
        current_count = len(current_waypoints)

        # waypoint数が不足している場合
        if current_count > 0 and current_count < target_count:
            QMessageBox.warning(
                self,
                "Waypoint不足",
                f"現在の画像には{current_count}個のwaypointが配置されていますが、\n"
                f"{target_count}個必要です。\n\n"
                f"残り{target_count - current_count}個のwaypointを配置してから次の画像に進んでください。\n\n"
                f"※配置を中止する場合は、Deleteキーで全てのwaypointを削除してください。"
            )
            return False

        return True

    def set_current_y_position(self, position_type):
        """現在のマウス位置のY座標を開始/終了位置に設定"""
        # マウス位置を取得（相対位置をグローバル座標に変換）
        cursor_pos = QCursor.pos()
        # メイン画像ビューの位置を取得
        if hasattr(self, 'main_image_view'):
            local_pos = self.main_image_view.mapFromGlobal(cursor_pos)

            # 画像内かチェック
            if hasattr(self.main_image_view, 'target_rect') and self.main_image_view.target_rect.contains(local_pos):
                # スクリーン座標を画像座標に変換
                rel_x = (local_pos.x() - self.main_image_view.target_rect.x()) / self.main_image_view.target_rect.width()
                rel_y = (local_pos.y() - self.main_image_view.target_rect.y()) / self.main_image_view.target_rect.height()

                # 元の画像の座標に変換
                if hasattr(self.main_image_view, 'pix_height'):
                    orig_y = int(rel_y * self.main_image_view.pix_height)

                    if position_type == 'start':
                        self.waypoint_start_y_spin.setValue(orig_y)
                        self.statusBar().showMessage(f"開始Y位置を{orig_y}に設定しました", 2000)
                    else:  # end
                        self.waypoint_end_y_spin.setValue(orig_y)
                        self.statusBar().showMessage(f"終了Y位置を{orig_y}に設定しました", 2000)
                else:
                    self.statusBar().showMessage("画像が読み込まれていません", 2000)
            else:
                self.statusBar().showMessage("マウスが画像内にありません", 2000)
        else:
            self.statusBar().showMessage("画像ビューが初期化されていません", 2000)

    def update_waypoint_guidelines(self):
        """waypoint設定変更時にガイドラインを更新"""
        if hasattr(self, 'main_image_view') and hasattr(self.main_image_view, 'update'):
            self.main_image_view.update()

    def on_waypoint_mode_changed(self, button):
        """waypointモードラジオボタンの変更時の処理"""
        button_id = self.waypoint_mode_button_group.id(button)

        # 全ての機能を一旦無効化
        self.auto_apply_last_waypoint = False
        self.auto_advance_waypoint = False

        if button_id == 0:  # 前回waypoint自動適用モード
            self.auto_apply_last_waypoint = True

            # 現在の画像に対して、前回のwaypointを適用
            if self.last_waypoints:
                current_index = self.current_index
                if current_index is not None:
                    # 既存のwaypointがない場合のみ適用
                    if current_index not in self.waypoint_annotations or not self.waypoint_annotations[current_index]:
                        self.waypoint_annotations[current_index] = self.last_waypoints.copy()

                        # 画面を更新
                        self.display_current_image()
                        self.update_gallery()

                        # ステータスメッセージ
                        if hasattr(self, 'statusBar'):
                            self.statusBar().showMessage(f"前回waypoint自動適用モードに切り替え - {len(self.last_waypoints)}個を適用しました", 2000)
                    else:
                        if hasattr(self, 'statusBar'):
                            self.statusBar().showMessage("前回waypoint自動適用モードに切り替えました", 2000)
            else:
                if hasattr(self, 'statusBar'):
                    self.statusBar().showMessage("前回waypoint自動適用モードに切り替えました（適用するwaypointがありません）", 2000)

        elif button_id == 1:  # 配置完了時自動遷移モード
            self.auto_advance_waypoint = True
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage("配置完了時自動遷移モードに切り替えました", 2000)

    def toggle_seg_driving_direction(self, state):
        """走行方向矢印の表示/非表示を切り替え"""
        self.show_seg_driving_direction = (state == Qt.Checked)
        if hasattr(self, 'main_image_view'):
            self.main_image_view.update()

    def update_seg_driving_direction_class(self, value):
        """走行方向計算に使用するクラスIDを更新"""
        self.seg_driving_direction_class_id = value
        if hasattr(self, 'main_image_view') and self.show_seg_driving_direction:
            self.main_image_view.update()

    def update_seg_driving_direction_y(self, value):
        """走行方向計算に使用するY座標を更新"""
        self.seg_driving_direction_y = value
        if hasattr(self, 'main_image_view') and self.show_seg_driving_direction:
            self.main_image_view.update()

    def update_seg_max_steering_angle(self, value):
        """最大舵角を更新"""
        self.seg_max_steering_angle = value
        if hasattr(self, 'main_image_view') and self.show_seg_driving_direction:
            self.main_image_view.update()

    def on_seg_display_mode_changed(self):
        """セグメンテーション走行方向の表示モード変更"""
        if self.seg_trajectory_mode_radio.isChecked():
            self.seg_display_mode = 'trajectory'
        else:
            self.seg_display_mode = 'waypoint'

        if hasattr(self, 'main_image_view') and self.show_seg_driving_direction:
            self.main_image_view.update()

    def calculate_steering_arc_params(self, start_x, start_y, target_x, target_y, max_steering_angle_deg):
        """舵角制約下での走行軌跡の円弧パラメータを計算

        Args:
            start_x, start_y: 開始位置（画像座標）
            target_x, target_y: 目標位置（画像座標）
            max_steering_angle_deg: 最大舵角（度）

        Returns:
            dict: 円弧パラメータ {'center_x', 'center_y', 'radius', 'start_angle', 'end_angle', 'direction'}
                  計算できない場合はNone
        """
        # 目標点への直線距離と角度を計算
        dx = target_x - start_x
        dy = target_y - start_y  # 画像座標系ではY軸が下向きに増加

        if abs(dx) < 1 and abs(dy) < 1:
            return None  # 距離が近すぎる

        # 最大舵角から最小回転半径を計算
        max_steering_rad = math.radians(max_steering_angle_deg)

        # 目標点への横方向のオフセット
        lateral_offset = dx
        longitudinal_distance = abs(dy)

        if longitudinal_distance < 1:
            return None

        # 舵角がほぼ0の場合は直線
        if abs(lateral_offset) < 1:
            return None

        # 左右どちらに曲がるかを決定
        turn_direction = 1 if lateral_offset > 0 else -1

        # 2点を通り、開始点での接線が垂直（Y軸方向）な円を求める
        # 開始点: (start_x, start_y), 接線方向: (0, -1)（上向き）
        # 円の中心: (start_x + R * turn_direction, start_y)

        # 目標点が円周上にある条件:
        # (target_x - center_x)^2 + (target_y - center_y)^2 = R^2
        # center_x = start_x + R * turn_direction
        # center_y = start_y

        # (target_x - start_x - R*turn_direction)^2 + (target_y - start_y)^2 = R^2
        # dx^2 - 2*dx*R*turn_direction + R^2 + dy^2 = R^2
        # dx^2 + dy^2 = 2*dx*R*turn_direction
        # R = (dx^2 + dy^2) / (2*dx*turn_direction)

        turn_radius = (dx*dx + dy*dy) / (2.0 * dx * turn_direction)

        # 半径が負になる場合は逆方向
        if turn_radius < 0:
            turn_radius = -turn_radius
            turn_direction = -turn_direction

        # 最大舵角による最小回転半径の制約をチェック
        # 最小回転半径 ≈ 進行距離 / tan(max_steering)
        min_radius = longitudinal_distance / math.tan(max_steering_rad)

        if turn_radius < min_radius:
            # 舵角が大きすぎる場合は制限
            turn_radius = min_radius
            actual_steering_rad = math.atan(longitudinal_distance / turn_radius)
        else:
            # 実際の舵角を計算
            actual_steering_rad = math.atan(longitudinal_distance / turn_radius)

        # 円の中心を計算
        center_x = start_x + turn_direction * turn_radius
        center_y = start_y

        # 開始角度と終了角度を計算（標準的なatan2、X軸から反時計回り）
        start_angle_rad = math.atan2(start_y - center_y, start_x - center_x)
        end_angle_rad = math.atan2(target_y - center_y, target_x - center_x)

        print(f"[円弧計算] dx={dx:.1f}, dy={dy:.1f}, 方向={turn_direction}")
        print(f"[円弧計算] 回転半径={turn_radius:.1f}, 中心=({center_x:.1f}, {center_y:.1f})")
        print(f"[円弧計算] 開始角度={math.degrees(start_angle_rad):.1f}°, 終了角度={math.degrees(end_angle_rad):.1f}°")

        return {
            'center_x': center_x,
            'center_y': center_y,
            'radius': abs(turn_radius),
            'start_angle': start_angle_rad,
            'end_angle': end_angle_rad,
            'direction': turn_direction,
            'actual_steering_deg': math.degrees(actual_steering_rad) * turn_direction
        }

    def calculate_seg_x_at_y(self, current_index, y_coord):
        """指定されたY座標におけるセグメンテーションエリアのX中央値を計算

        Args:
            current_index: 現在の画像インデックス
            y_coord: Y座標

        Returns:
            int: X座標の中央値、計算できない場合はNone
        """
        if not hasattr(self, 'segmentation_inference_results'):
            return None

        if current_index not in self.segmentation_inference_results:
            return None

        seg_results = self.segmentation_inference_results[current_index]
        if not seg_results:
            return None

        if 'masks' not in seg_results or 'classes' not in seg_results:
            return None

        # 指定されたクラスIDのマスクを探す
        target_mask = None
        for i, class_id in enumerate(seg_results['classes']):
            if class_id == self.seg_driving_direction_class_id:
                target_mask = seg_results['masks'][i]
                break

        if target_mask is None:
            return None

        # マスクから指定Y座標における走行可能エリアのX座標範囲を取得
        if y_coord < 0 or y_coord >= target_mask.shape[0]:
            return None

        # Y座標ラインのマスク値を取得
        line_mask = target_mask[int(y_coord), :]

        # True（走行可能エリア）のX座標を取得
        x_indices = np.where(line_mask > 0.5)[0]

        if len(x_indices) == 0:
            return None

        # X座標の中央値を計算
        x_center = int(np.median(x_indices))
        return x_center

    def calculate_polygon_iou(self, poly1_points, poly2_points, img_width, img_height):
        """2つのポリゴンのIoU (Intersection over Union) を計算

        Args:
            poly1_points: ポリゴン1の頂点リスト [(x1, y1), (x2, y2), ...]
            poly2_points: ポリゴン2の頂点リスト
            img_width: 画像の幅
            img_height: 画像の高さ

        Returns:
            float: IoU値 (0.0~1.0)
        """
        from shapely.geometry import Polygon
        from shapely.validation import make_valid

        try:
            # Shapely Polygonオブジェクトを作成
            poly1 = Polygon(poly1_points)
            poly2 = Polygon(poly2_points)

            # ポリゴンが無効な場合は修正を試みる
            if not poly1.is_valid:
                poly1 = make_valid(poly1)
            if not poly2.is_valid:
                poly2 = make_valid(poly2)

            # 交差エリアを計算
            if not poly1.intersects(poly2):
                return 0.0

            intersection_area = poly1.intersection(poly2).area

            # 和集合エリアを計算
            union_area = poly1.area + poly2.area - intersection_area

            # IoUを計算
            if union_area == 0:
                return 0.0

            iou = intersection_area / union_area
            return iou

        except Exception as e:
            # エラーが発生した場合は重なりなしとみなす
            print(f"IoU計算エラー: {e}")
            return 0.0

    def calculate_bbox_iou(self, bbox1, bbox2):
        """2つのバウンディングボックスのIoU (Intersection over Union) を計算

        Args:
            bbox1: バウンディングボックス1 {'x1': float, 'y1': float, 'x2': float, 'y2': float}
            bbox2: バウンディングボックス2

        Returns:
            float: IoU値 (0.0~1.0)
        """
        # 交差領域の座標を計算
        x1_inter = max(bbox1['x1'], bbox2['x1'])
        y1_inter = max(bbox1['y1'], bbox2['y1'])
        x2_inter = min(bbox1['x2'], bbox2['x2'])
        y2_inter = min(bbox1['y2'], bbox2['y2'])

        # 交差領域がない場合
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0

        # 交差領域の面積
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # 各バウンディングボックスの面積
        bbox1_area = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
        bbox2_area = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])

        # 和集合の面積
        union_area = bbox1_area + bbox2_area - inter_area

        # IoUを計算
        if union_area == 0:
            return 0.0

        iou = inter_area / union_area
        return iou

    def check_bbox_overlap(self, new_bbox, existing_bboxes, iou_threshold=0.5):
        """新しいバウンディングボックスが既存のバウンディングボックスと重複しているか確認

        Args:
            new_bbox: 新しいバウンディングボックス {'x1': float, 'y1': float, 'x2': float, 'y2': float, 'class': str, ...}
            existing_bboxes: 既存のバウンディングボックスリスト
            iou_threshold: 重複と判定するIoUの閾値 (デフォルト: 0.5)

        Returns:
            bool: True = 重複あり（追加すべきでない）, False = 重複なし（追加可能）
        """
        new_class = new_bbox['class']

        # 同じクラスの既存バウンディングボックスとの重なりをチェック
        for existing_bbox in existing_bboxes:
            if existing_bbox['class'] != new_class:
                # 異なるクラスの場合はスキップ
                continue

            # IoUを計算
            iou = self.calculate_bbox_iou(new_bbox, existing_bbox)

            # 閾値以上の重なりがある場合は重複と判定
            if iou >= iou_threshold:
                return True

        return False

    def check_segmentation_overlap(self, new_seg, existing_segs, img_width, img_height, iou_threshold=0.5):
        """新しいセグメンテーションが既存のセグメンテーションと重複しているか確認

        Args:
            new_seg: 新しいセグメンテーション {'class': str, 'points': [(x, y), ...], ...}
            existing_segs: 既存のセグメンテーションリスト
            img_width: 画像の幅
            img_height: 画像の高さ
            iou_threshold: 重複と判定するIoUの閾値 (デフォルト: 0.5)

        Returns:
            bool: True = 重複あり（追加すべきでない）, False = 重複なし（追加可能）
        """
        new_class = new_seg['class']
        new_points = new_seg['points']

        # 同じクラスの既存セグメンテーションとの重なりをチェック
        for existing_seg in existing_segs:
            if existing_seg['class'] != new_class:
                # 異なるクラスの場合はスキップ
                continue

            existing_points = existing_seg['points']

            # IoUを計算
            iou = self.calculate_polygon_iou(new_points, existing_points, img_width, img_height)

            # 閾値以上の重なりがある場合は重複と判定
            if iou >= iou_threshold:
                return True

        return False

    def calculate_seg_driving_direction(self, current_index):
        """セグメンテーション推論結果から走行方向を計算

        Args:
            current_index: 現在の画像インデックス

        Returns:
            tuple: (target_x, target_y) 走行方向の座標、計算できない場合はNone
        """
        if not hasattr(self, 'segmentation_inference_results'):
            print("[走行方向計算] segmentation_inference_resultsが存在しません")
            return None

        if current_index not in self.segmentation_inference_results:
            print(f"[走行方向計算] インデックス {current_index} の推論結果が存在しません")
            print(f"[走行方向計算] 利用可能なキー: {list(self.segmentation_inference_results.keys())}")
            return None

        seg_results = self.segmentation_inference_results[current_index]
        if not seg_results:
            print("[走行方向計算] 推論結果が空です")
            return None

        if 'masks' not in seg_results or 'classes' not in seg_results:
            print(f"[走行方向計算] masksまたはclassesキーが存在しません。利用可能なキー: {seg_results.keys()}")
            return None

        print(f"[走行方向計算] 検出されたクラス: {seg_results['classes']}, マスク数: {len(seg_results['masks'])}")
        print(f"[走行方向計算] 検索対象クラスID: {self.seg_driving_direction_class_id}")

        # 指定されたクラスIDのマスクを探す
        target_mask = None
        for i, class_id in enumerate(seg_results['classes']):
            if class_id == self.seg_driving_direction_class_id:
                target_mask = seg_results['masks'][i]
                print(f"[走行方向計算] クラスID {class_id} のマスクを発見 (インデックス {i})")
                break

        if target_mask is None:
            print(f"[走行方向計算] クラスID {self.seg_driving_direction_class_id} のマスクが見つかりません")
            return None

        # マスクから指定Y座標における走行可能エリアのX座標範囲を取得
        y = self.seg_driving_direction_y
        print(f"[走行方向計算] マスクサイズ: {target_mask.shape}, Y座標: {y}")

        if y < 0 or y >= target_mask.shape[0]:
            print(f"[走行方向計算] Y座標 {y} がマスクの範囲外です (0-{target_mask.shape[0]-1})")
            return None

        # Y座標ラインのマスク値を取得
        line_mask = target_mask[y, :]

        # True（走行可能エリア）のX座標を取得
        x_indices = np.where(line_mask > 0.5)[0]  # 閾値0.5を追加

        if len(x_indices) == 0:
            print(f"[走行方向計算] Y座標 {y} に走行可能エリアが存在しません")
            return None

        # X座標の中央値を計算
        x_center = int(np.median(x_indices))
        print(f"[走行方向計算] 計算成功: ({x_center}, {y}), X範囲: [{x_indices[0]}, {x_indices[-1]}]")

        return (x_center, y)

    def toggle_auto_apply_segmentation(self, state):
        """前回のセグメンテーションを自動適用するかどうかを設定"""
        self.auto_apply_last_segmentation = (state == Qt.Checked)

        # 現在の画像に対して、前回のセグメンテーションを適用
        if (self.auto_apply_last_segmentation and
            hasattr(self, 'last_segmentation') and
            self.last_segmentation and
            self.images):

            # インデックスベースに変更
            current_index = self.current_index
            
            # current_indexの有効性をチェック
            if current_index is None or not isinstance(current_index, int):
                return
            
            # 削除済みの場合は適用しない
            if hasattr(self, 'deleted_indexes') and current_index in self.deleted_indexes:
                return
            
            # すでにアノテーションがある場合は確認
            if (current_index in self.segmentation_annotations and 
                self.segmentation_annotations[current_index]):
                return
            
            # 前回のセグメンテーションを適用（ディープコピーで完全に独立させる）
            self.add_segmentation_annotation(deepcopy(self.last_segmentation))
            
    def calculate_and_store_diff_vector(self, index_or_path):
        """教師データと推論結果の差分ベクトルを計算して保存する"""
        # インデックスベースで処理
        if isinstance(index_or_path, int):
            index = index_or_path
        else:
            # パスからインデックスを取得
            try:
                index = self.images.index(index_or_path)
            except ValueError:
                print(f"警告: パス {index_or_path} からインデックスを取得できませんでした")
                return
        
        # 教師データの取得（インデックスベース）
        annotation = None
        if index in self.annotations:
            annotation = self.annotations[index]
        
        # 推論結果の取得（インデックスベース）
        inference = None
        if index in self.inference_results:
            inference = self.inference_results[index]
                
        if annotation and inference:
            # アノテーションと推論結果の両方にangle/throttleキーが存在するかチェック
            if ("angle" in annotation and "throttle" in annotation and 
                annotation["angle"] is not None and annotation["throttle"] is not None):
                # 角度と速度の差分を計算
                if "pilot/angle" in inference and "pilot/throttle" in inference:
                    angle_diff = inference["pilot/angle"] - annotation["angle"]
                    throttle_diff = inference["pilot/throttle"] - annotation["throttle"]
                elif "angle" in inference and "throttle" in inference:
                    angle_diff = inference["angle"] - annotation["angle"]
                    throttle_diff = inference["throttle"] - annotation["throttle"]
                else:
                    # 推論結果にangle/throttleがない場合はスキップ
                    return
            else:
                # アノテーションにangle/throttleがない場合（削除済み等）は差分ベクトルも削除
                if index in self.inference_diff_vectors:
                    del self.inference_diff_vectors[index]
                return
            
            # ベクトルの大きさと角度を計算
            import math
            vector_magnitude = math.sqrt(angle_diff**2 + throttle_diff**2)
            vector_angle = math.atan2(throttle_diff, angle_diff)
            
            # インデックスをキーとして差分ベクトルを保存
            self.inference_diff_vectors[index] = {
                'angle_diff': angle_diff,
                'throttle_diff': throttle_diff,
                'vector_magnitude': vector_magnitude,
                'vector_angle': vector_angle
            }
            
        else:
            # アノテーションまたは推論結果がない場合は差分ベクトルを削除
            if index in self.inference_diff_vectors:
                del self.inference_diff_vectors[index]
            print(f"差分ベクトル計算スキップ: アノテーションまたは推論結果が不足")

    def update_bbox_stats(self):
        """
        Update statistics for bounding box annotations
        """
        # Count the number of bounding box annotations
        bbox_count = len(self.bbox_annotations) if hasattr(self, 'bbox_annotations') else 0
        
        # Update the stats through the comprehensive update method
        if hasattr(self, 'update_driving_annotation_stats'):
            self.update_driving_annotation_stats()
            
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

        # ONにした時、現在の画像に推論結果がなければ推論を実行
        if show_inference and self.images and hasattr(self, 'yolo_model') and self.yolo_model is not None:
            current_img_path = self.images[self.current_index]
            if current_img_path not in self.detection_inference_results:
                # 現在の画像に対して推論を実行
                self.run_single_yolo_inference()

        # 画面更新
        self.update_detection_info_panel()        
        self.update_ui()
        
    def toggle_diff_vector_display(self, state):
            """差分ベクトル矢印表示の切り替え"""
            show_diff_vectors = (state == Qt.Checked)
            self.show_diff_vectors = show_diff_vectors
            
            # 画面更新
            self.update_ui()
            
    def toggle_location_inference_display(self, state):
        """位置推論表示の切り替え"""
        show_inference = (state == Qt.Checked)
        print(f"位置推論表示切替: {show_inference} (state={state}, Qt.Checked={Qt.Checked})")

        self.show_location_inference = show_inference
        
        # 表示情報の更新
        if show_inference:
            # チェックボックスをONにした時は現在の画像に対して推論を実行
            if hasattr(self, 'location_model_manager') and self.location_model_manager.model is not None:
                self.run_location_inference()
                self.update_location_inference_display()
            self.update_location_info_panel()
            self.statusBar().showMessage("位置推論結果表示をオンにしました", 3000)
        else:
            # 表示をクリア
            if hasattr(self, 'location_inference_info_label'):
                self.location_inference_info_label.setText(" ")  # スペースで高さを維持
            self.statusBar().showMessage("位置推論結果表示をオフにしました", 3000)
        
        # 画面更新
        if hasattr(self, 'main_image_view'):
            self.main_image_view.update()
        

    # location関連

    def update_location_button_counts(self):
        """各位置ボタンのアノテーション数を更新する"""
        if not hasattr(self, 'location_buttons') or not self.location_buttons:
            return

        # 位置情報ごとのアノテーション数をカウント
        location_counts = {}
        #for img_path, anno in self.annotations.items():
        for idx, anno in self.annotations.items():
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
            
            # 運転アノテーション（角度・スロットル）がある場合はそこに位置情報を追加
            if self.current_index in self.annotations:
                self.annotations[self.current_index]["loc"] = location_value
            else:
                # 運転アノテーションがない場合でも位置アノテーション専用エントリを作成
                self.annotations[self.current_index] = {"loc": location_value}
        
        # 保存用のデータ形式を更新するため、アノテーションタイムスタンプも更新
        self.annotation_timestamps[self.current_index] = int(time.time() * 1000)
        
        # 位置ボタンのカウント表示を更新
        self.update_location_button_counts()
        
        # UI更新
        self.display_current_image()
        self.update_gallery()
        
        # 情報パネルの統計情報を更新
        self.update_driving_annotation_stats()
        
        # 分布グラフも更新（位置情報変更時）
        if hasattr(self, 'update_distribution_graph'):
            self.update_distribution_graph()

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
        
        # モデルアーキでフィルタリング（自動運転モデルのみ）
        model_files = []
        for model_file in all_model_files:
            # 位置推論モデルとYOLOモデルを除外
            if "_location_" in model_file or "yolo" in model_file.lower():
                continue
            
            # モデルファイル名にアーキ名が含まれていれば対象とする
            # 通常、モデルファイル名は「アーキ名_日時.pth」等の形式
            if current_arch.lower() in model_file.lower():
                model_files.append(model_file)
        
        if not model_files:
            # フィルタリングした結果がなければ、その旨を表示
            self.model_combo.addItem(f"{current_arch}のモデルが見つかりません")
            self.statusBar().showMessage(f"{current_arch}のモデルが見つかりません。他のアーキを選択するか、モデルを学習してください", 3000)
            return

        # モデルファイルを作成日時順にソート（新しいものが上）
        # カスタムサフィックスが追加された場合でも正しくソートされるよう、mtimeを使用
        model_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)
        
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
        if is_playing:
            # 停止した場合
            self.play_button.setText("▶再生")
            self.statusBar().clearMessage()
        else:
            # 再生開始した場合
            self.play_button.setText("■停止")

    def auto_play(self, forward=True):
        """画像を自動再生する（スキップ枚数対応、推論表示時は速度調整）"""
        if not self.images:
            return
        
        # 現在の自動再生状態を確認
        if hasattr(self, 'auto_play_timer') and self.auto_play_timer.isActive():
            # タイマーが動いている場合は停止
            self.auto_play_timer.stop()
            return
        
        # スキップ枚数を取得（チェックボックスがONの時のみ）
        if self.skip_images_on_click.isChecked():
            skip_count = self.skip_count_spin.value()
        else:
            skip_count = 1

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
        skip_info = f"{skip_count}枚スキップ" if skip_count > 1 else "スキップなし"
        self.statusBar().showMessage(f"自動再生中 ({direction}, {skip_info}, {playback_speed}) - 停止するには再度ボタンをクリック")

    def play_reverse(self):
        """自動再生（逆方向）"""
        # 再生中かどうかをチェック
        is_playing = hasattr(self, 'auto_play_timer') and self.auto_play_timer.isActive()

        # 再生または停止
        self.auto_play(forward=False)

        # 再生状態に応じてボタンテキストを更新
        if is_playing:
            # 停止した場合
            self.reverse_play_button.setText("◀逆再生")
            self.statusBar().clearMessage()
        else:
            # 再生開始した場合
            self.reverse_play_button.setText("■停止")

    def stop_auto_play(self):
        """自動再生を停止する"""
        if hasattr(self, 'auto_play_timer') and self.auto_play_timer.isActive():
            self.auto_play_timer.stop()

            # ボタンのテキストを元に戻す
            self.play_button.setText("▶再生")
            self.reverse_play_button.setText("◀逆再生")

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
            self.display_current_image()
            self.update_gallery()   
            self.update_slider_deleted_indexes()
            
            # 画像ソース（キー）切り替え時の推論実行
            self.run_inference_after_image_source_change()
        
        # キーボタン群を更新
        self.update_variant_buttons()

    def get_left_layout(self):
        """左パネルのレイアウトを安全に取得するヘルパーメソッド"""
        try:
            central_widget = self.centralWidget()
            if central_widget is None:
                return None
            main_layout = central_widget.layout()
            if main_layout is None:
                return None
            left_scroll_area = main_layout.itemAt(0).widget()  # QScrollArea
            if left_scroll_area is None:
                return None
            left_panel = left_scroll_area.widget()  # QWidget
            if left_panel is None:
                return None
            return left_panel.layout()  # QVBoxLayout
        except Exception as e:
            print(f"エラー: left_layoutの取得に失敗しました: {e}")
            return None

    def update_variant_buttons(self):
        """
        available_variantsの内容に基づいてキーボタン群を更新する
        """
        # GroupBoxを探す
        variant_box = None
        left_layout = self.get_left_layout()
        
        if left_layout is None:
            print("警告: left_layoutが見つかりません")
            return
            
        try:
            for i in range(left_layout.count()):
                item = left_layout.itemAt(i).widget()
                if isinstance(item, QGroupBox) and item.title() == "画像ソース":
                    variant_box = item
                    break
        except Exception as e:
            print(f"エラー: GroupBox検索に失敗しました: {e}")
            return
        
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
            # インデックスベースに変更
            current_index = self.current_index
            
            # current_indexの有効性をチェック
            if current_index is None or not isinstance(current_index, int):
                return
            
            # 削除済みの場合は適用しない
            if hasattr(self, 'deleted_indexes') and current_index in self.deleted_indexes:
                return
            
            # すでにアノテーションがある場合は確認
            if (current_index in self.bbox_annotations and 
                self.bbox_annotations[current_index]):
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
        
        if hasattr(self, 'inference_checkbox') and self.inference_checkbox.isChecked() :
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
                inference_text += f"angle = <span style='color: #009999;'>{angle:.4f}</span><br>"
                inference_text += f"throttle = <span style='color: #009999;'>{throttle:.4f}</span>"

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
                self.inference_info_label.setText(" ")  # スペースで高さを維持
            
            self.main_image_view.inference_point = None
            
            return False

    def update_location_info_panel(self):
        """位置推論結果の情報パネルを更新する"""
        if not self.images:
            return False
                
        current_img_path = self.images[self.current_index]
        
        # 位置推論表示がOFFの場合は表示をクリア
        show_location_inference = getattr(self, 'show_location_inference', False)
        checkbox_checked = hasattr(self, 'location_inference_checkbox') and self.location_inference_checkbox.isChecked()
        
        
        if not show_location_inference:
            if hasattr(self, 'location_inference_info_label'):
                self.location_inference_info_label.setText(" ")  # スペースで高さを維持
            
            # 位置推論ポイントをクリア
            if hasattr(self, 'main_image_view'):
                self.main_image_view.location_inference_result = None
            
            return False
        
        if hasattr(self, 'location_inference_checkbox') and self.location_inference_checkbox.isChecked():
            current_index = self.current_index
            if current_index in self.location_inference_results:
                # 推論結果を取得
                inference = self.location_inference_results[current_index]
                

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
            # 表示がオフの場合は情報パネルをクリア（スペースで高さを維持）
            if hasattr(self, 'location_inference_info_label'):
                # モデル読み込み直後はクリアしないようにする
                if not hasattr(self, 'show_location_inference') or not self.show_location_inference:
                    self.location_inference_info_label.setText(" ")
            
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
        # 既存のコードを修正して3つのモードに対応
        sender = self.sender()
        
        # モード切り替え前に選択状態をクリア
        self.clear_all_selections()
        
        if sender == self.auto_mode_button:
            self.current_mode = 0
            self.auto_mode_button.setChecked(True)
            self.detection_mode_button.setChecked(False)
            self.segmentation_mode_button.setChecked(False)
            self.waypoint_mode_button.setChecked(False)
            self.waypoint_control_widget.setVisible(False)  # waypoint制御パネルを非表示
            self.segmentation_control_widget.setVisible(False)  # セグメンテーション制御パネルを非表示
            self.statusBar().showMessage("自動運転アノテーションモードに切り替えました。", 3000)
        elif sender == self.detection_mode_button:
            self.current_mode = 1
            self.auto_mode_button.setChecked(False)
            self.detection_mode_button.setChecked(True)
            self.segmentation_mode_button.setChecked(False)
            self.waypoint_mode_button.setChecked(False)
            self.waypoint_control_widget.setVisible(False)  # waypoint制御パネルを非表示
            self.segmentation_control_widget.setVisible(False)  # セグメンテーション制御パネルを非表示
            self.statusBar().showMessage("物体検知アノテーションモードに切り替えました。", 3000)
        elif sender == self.segmentation_mode_button:
            self.current_mode = 2  # 新規追加
            self.auto_mode_button.setChecked(False)
            self.detection_mode_button.setChecked(False)
            self.segmentation_mode_button.setChecked(True)
            self.waypoint_mode_button.setChecked(False)
            self.waypoint_control_widget.setVisible(False)  # waypoint制御パネルを非表示
            self.segmentation_control_widget.setVisible(True)  # セグメンテーション制御パネルを表示
            self.statusBar().showMessage("セグメンテーションアノテーションモードに切り替えました。", 3000)
        elif sender == self.waypoint_mode_button:
            self.current_mode = 3  # waypoint mode
            self.auto_mode_button.setChecked(False)
            self.detection_mode_button.setChecked(False)
            self.segmentation_mode_button.setChecked(False)
            self.waypoint_mode_button.setChecked(True)
            self.waypoint_control_widget.setVisible(True)  # waypoint制御パネルを表示
            self.segmentation_control_widget.setVisible(False)  # セグメンテーション制御パネルを非表示
            self.statusBar().showMessage("waypointアノテーションモードに切り替えました。", 3000)
        else:
            # Bキーでの切り替え（4つのモードをサイクル）
            self.current_mode = (self.current_mode + 1) % 4
            if self.current_mode == 0:
                self.auto_mode_button.setChecked(True)
                self.detection_mode_button.setChecked(False)
                self.segmentation_mode_button.setChecked(False)
                self.waypoint_mode_button.setChecked(False)
                self.waypoint_control_widget.setVisible(False)
                self.segmentation_control_widget.setVisible(False)
                self.statusBar().showMessage("自動運転アノテーションモードに切り替えました。", 3000)
            elif self.current_mode == 1:
                self.auto_mode_button.setChecked(False)
                self.detection_mode_button.setChecked(True)
                self.segmentation_mode_button.setChecked(False)
                self.waypoint_mode_button.setChecked(False)
                self.waypoint_control_widget.setVisible(False)
                self.segmentation_control_widget.setVisible(False)
                self.statusBar().showMessage("物体検知アノテーションモードに切り替えました。", 3000)
            elif self.current_mode == 2:
                self.auto_mode_button.setChecked(False)
                self.detection_mode_button.setChecked(False)
                self.segmentation_mode_button.setChecked(True)
                self.waypoint_mode_button.setChecked(False)
                self.waypoint_control_widget.setVisible(False)
                self.segmentation_control_widget.setVisible(True)
                self.statusBar().showMessage("セグメンテーションアノテーションモードに切り替えました。", 3000)
            else:  # current_mode == 3
                self.auto_mode_button.setChecked(False)
                self.detection_mode_button.setChecked(False)
                self.segmentation_mode_button.setChecked(False)
                self.waypoint_mode_button.setChecked(True)
                self.waypoint_control_widget.setVisible(True)
                self.segmentation_control_widget.setVisible(False)
                self.statusBar().showMessage("waypointアノテーションモードに切り替えました。", 3000)
        
        self.main_image_view.update()
    
    def clear_all_selections(self):
        """全ての選択状態をクリア"""
        # バウンディングボックスの選択をクリア
        if hasattr(self.main_image_view, 'selected_bbox_index'):
            self.main_image_view.selected_bbox_index = None
        if hasattr(self.main_image_view, 'hovering_bbox_index'):
            self.main_image_view.hovering_bbox_index = None
            
        # セグメンテーションの選択をクリア
        if hasattr(self.main_image_view, 'selected_segmentation_index'):
            self.main_image_view.selected_segmentation_index = None
        if hasattr(self.main_image_view, 'selected_polygon_index'):
            self.main_image_view.selected_polygon_index = None
        if hasattr(self.main_image_view, 'hovering_polygon_index'):
            self.main_image_view.hovering_polygon_index = None
            
        # セグメンテーション描画状態をクリア
        if hasattr(self.main_image_view, 'current_segmentation_polygon'):
            self.main_image_view.current_segmentation_polygon = []
        if hasattr(self.main_image_view, 'is_drawing_segmentation'):
            self.main_image_view.is_drawing_segmentation = False
            
        # 移動・編集状態をクリア
        if hasattr(self.main_image_view, 'is_moving_segmentation'):
            self.main_image_view.is_moving_segmentation = False
        if hasattr(self.main_image_view, 'is_moving_vertex'):
            self.main_image_view.is_moving_vertex = False
        if hasattr(self.main_image_view, 'selected_vertex_index'):
            self.main_image_view.selected_vertex_index = None
    
    def update_inference_checkboxes_status(self):
        """各モデルの読み込み状態に応じてチェックボックスの有効/無効を更新"""
        # 自動運転モデル
        if hasattr(self, 'model') and self.model is not None:
            self.inference_checkbox.setEnabled(True)
            self.inference_checkbox.setToolTip("自動運転モデルが読み込まれています")
        else:
            self.inference_checkbox.setEnabled(False)
            self.inference_checkbox.setChecked(False)
            self.inference_checkbox.setToolTip("自動運転モデルが読み込まれていません")
        
        # YOLOモデル（相互排他制御）
        has_detection_model = hasattr(self, 'yolo_model') and self.yolo_model is not None
        has_segmentation_model = hasattr(self, 'yolo_seg_model') and self.yolo_seg_model is not None
        
        # 物体検知モデル
        if has_detection_model:
            self.detection_inference_checkbox.setEnabled(True)
            self.detection_inference_checkbox.setToolTip("物体検知モデルが読み込まれています")
            # セグメンテーションチェックボックスを無効化
            if has_segmentation_model:
                self.segmentation_inference_checkbox.setEnabled(False)
                self.segmentation_inference_checkbox.setChecked(False)
                self.segmentation_inference_checkbox.setToolTip("物体検知モデルが読み込まれているため無効")
        else:
            self.detection_inference_checkbox.setEnabled(False)
            self.detection_inference_checkbox.setChecked(False)
            self.detection_inference_checkbox.setToolTip("物体検知モデルが読み込まれていません")
        
        # セグメンテーションモデル
        if has_segmentation_model and not has_detection_model:
            self.segmentation_inference_checkbox.setEnabled(True)
            self.segmentation_inference_checkbox.setToolTip("セグメンテーションモデルが読み込まれています")
        elif not has_segmentation_model:
            self.segmentation_inference_checkbox.setEnabled(False)
            self.segmentation_inference_checkbox.setChecked(False)
            self.segmentation_inference_checkbox.setToolTip("セグメンテーションモデルが読み込まれていません")
        
        # 位置モデル
        if hasattr(self, 'location_model_manager') and self.location_model_manager.is_model_loaded():
            self.location_inference_checkbox.setEnabled(True)
            self.location_inference_checkbox.setToolTip("位置モデルが読み込まれています")
        else:
            self.location_inference_checkbox.setEnabled(False)
            self.location_inference_checkbox.setChecked(False)
            self.location_inference_checkbox.setToolTip("位置モデルが読み込まれていません")
            
        # 差分ベクトル表示（自動運転モデルに依存）
        if hasattr(self, 'model') and self.model is not None:
            self.diff_vector_checkbox.setEnabled(True)
            self.diff_vector_checkbox.setToolTip("自動運転モデルが読み込まれています")
        else:
            self.diff_vector_checkbox.setEnabled(False)
            self.diff_vector_checkbox.setChecked(False)
            self.diff_vector_checkbox.setToolTip("自動運転モデルが読み込まれていません")
            
        # 全画像推論ボタン（自動運転モデルに依存）
        if hasattr(self, 'batch_inference_button'):
            if hasattr(self, 'model') and self.model is not None:
                self.batch_inference_button.setEnabled(True)
                self.batch_inference_button.setToolTip("全ての画像に対して推論を実行します")
            else:
                self.batch_inference_button.setEnabled(False)
                self.batch_inference_button.setToolTip("自動運転モデルが読み込まれていません")

        # CAM表示（自動運転モデルに依存）
        if hasattr(self, 'gradcam_checkbox'):
            if hasattr(self, 'model') and self.model is not None:
                self.gradcam_checkbox.setEnabled(True)
                self.gradcam_checkbox.setToolTip("モデルの注目領域をヒートマップで表示")
                self.gradcam_target_combo.setEnabled(True)
                self.gradcam_method_combo.setEnabled(True)
                self.gradcam_direction_combo.setEnabled(True)
            else:
                self.gradcam_checkbox.setEnabled(False)
                self.gradcam_checkbox.setChecked(False)
                self.gradcam_checkbox.setToolTip("自動運転モデルが読み込まれていません")
                self.gradcam_target_combo.setEnabled(False)
                self.gradcam_method_combo.setEnabled(False)
                self.gradcam_direction_combo.setEnabled(False)

    def add_segmentation_annotation(self, polygon_data):
        """セグメンテーションアノテーションを追加"""
        if not self.images or not polygon_data:
            return

        # インデックスベースに変更
        current_index = self.current_index

        if current_index not in self.segmentation_annotations:
            self.segmentation_annotations[current_index] = []

        # polygon_dataをディープコピーしてから追加（参照の共有を防ぐ）
        self.segmentation_annotations[current_index].append(deepcopy(polygon_data))

        # 前回のセグメンテーションとして保存（ディープコピーで完全に独立させる）
        self.last_segmentation = deepcopy(polygon_data)

        # データクリーンアップ: Noneエントリを削除
        self.segmentation_annotations[current_index] = [
            seg for seg in self.segmentation_annotations[current_index] if seg is not None
        ]

        # 現在のすべてのセグメンテーションを保存（ディープコピーで完全に独立させる）
        self.last_segmentations = [deepcopy(seg) for seg in self.segmentation_annotations[current_index]]

        # UI更新
        self.main_image_view.update()
        self.update_gallery()
            
    def train_yolo_unified(self):
        """統合されたYOLO学習 - 学習時にタスクを選択（MLflow統合版）"""
        
        # アノテーションの確認
        has_bbox = bool(getattr(self, 'bbox_annotations', {}))
        has_seg = bool(getattr(self, 'segmentation_annotations', {}))
        
        if not has_bbox and not has_seg:
            QMessageBox.warning(self, "警告", "学習用のアノテーションがありません。")
            return

        # 学習タスク選択ダイアログ
        task_dialog = self._create_yolo_task_selection_dialog(has_bbox, has_seg)
        
        if not task_dialog.exec_():
            return
        
        # 選択されたタスクに基づいて学習を実行
        if task_dialog.detect_radio.isChecked():
            self.train_and_save_yolo_model_internal("detect")
        elif task_dialog.segment_radio.isChecked():
            self.train_and_save_yolo_model_internal("segment")
        else:
            QMessageBox.warning(self, "警告", "タスクが選択されていません。")

    def _create_yolo_task_selection_dialog(self, has_bbox, has_seg):
        """YOLOタスク選択ダイアログを作成"""
        
        task_dialog = QDialog(self)
        task_dialog.setWindowTitle("YOLO学習タスク選択")
        task_dialog.setMinimumWidth(500)
        
        task_layout = QVBoxLayout(task_dialog)
        
        # タイトル
        title_label = QLabel("学習するタスクを選択してください:")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        task_layout.addWidget(title_label)
        
        # アノテーション状況の表示
        status_group = QGroupBox("アノテーション状況")
        status_layout = QVBoxLayout(status_group)
        
        if has_bbox:
            bbox_count = sum(len(bboxes) for bboxes in self.bbox_annotations.values())
            bbox_images = len(self.bbox_annotations)
            bbox_status = QLabel(f"✓ バウンディングボックス: {bbox_count}個 ({bbox_images}枚の画像)")
            bbox_status.setStyleSheet("color: #2E7D32; font-weight: bold;")
        else:
            bbox_status = QLabel("✗ バウンディングボックス: なし")
            bbox_status.setStyleSheet("color: #D32F2F;")
        status_layout.addWidget(bbox_status)
        
        if has_seg:
            seg_count = sum(len(segs) for segs in self.segmentation_annotations.values())
            seg_images = len(self.segmentation_annotations)
            seg_status = QLabel(f"✓ セグメンテーション: {seg_count}個 ({seg_images}枚の画像)")
            seg_status.setStyleSheet("color: #2E7D32; font-weight: bold;")
        else:
            seg_status = QLabel("✗ セグメンテーション: なし")
            seg_status.setStyleSheet("color: #D32F2F;")
        status_layout.addWidget(seg_status)
        
        task_layout.addWidget(status_group)
        
        # タスク選択
        task_group = QGroupBox("学習タスク")
        task_group_layout = QVBoxLayout(task_group)
        
        # ラジオボタン
        task_dialog.detect_radio = QRadioButton("物体検知 (Detection)")
        task_dialog.detect_radio.setEnabled(has_bbox)
        task_dialog.detect_radio.setToolTip("バウンディングボックスを使用した物体検知モデルを学習")
        
        task_dialog.segment_radio = QRadioButton("セグメンテーション (Segmentation)")
        task_dialog.segment_radio.setEnabled(has_seg)
        task_dialog.segment_radio.setToolTip("ポリゴンを使用したセグメンテーションモデルを学習")
        
        # デフォルト選択
        if has_bbox and has_seg:
            task_dialog.detect_radio.setChecked(True)  # 両方ある場合は検知を優先
        elif has_bbox:
            task_dialog.detect_radio.setChecked(True)
        elif has_seg:
            task_dialog.segment_radio.setChecked(True)
        
        task_group_layout.addWidget(task_dialog.detect_radio)
        task_group_layout.addWidget(task_dialog.segment_radio)
        task_layout.addWidget(task_group)
        
        # ボタン
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(task_dialog.accept)
        button_box.rejected.connect(task_dialog.reject)
        task_layout.addWidget(button_box)
        
        return task_dialog

    def train_and_save_yolo_model_internal(self, task_type):
        """YOLO学習の内部処理 - MLflow統合版"""
        
        # アノテーションの確認とバリデーション
        annotations, annotation_info = self._validate_yolo_annotations(task_type)
        if not annotations:
            return
        
        # クラス設定
        classes = self.get_current_classes()
        if not classes:
            QMessageBox.warning(self, "警告", "検知クラスが設定されていません。\n先にクラス設定を行ってください。")
            return
        
        # モデルタイプの取得と調整
        model_type = self.yolo_model_combo.currentText()
        if task_type == "segment" and not model_type.endswith("-seg"):
            model_type = model_type + "-seg"
        
        task_name = "物体検知" if task_type == "detect" else "セグメンテーション"
        
        # 学習設定ダイアログを表示
        training_config = self._get_yolo_training_config(task_name, model_type, annotation_info)
        if not training_config:
            return
        
        # 進捗ダイアログを事前に初期化
        progress = None

        # 学習出力ダイアログを表示
        training_dialog = TrainingOutputDialog(self, f"YOLO {task_name} ({model_type})")
        training_dialog.show()
        QApplication.processEvents()

        # 学習前の準備情報をダイアログに追加
        training_dialog.add_preparation_message(f"=== {task_name}データセット作成 ===")
        training_dialog.add_preparation_message(f"{task_name}アノテーションエクスポート開始")

        # クラス情報を表示
        class_mapping = {classes[i]: i for i in range(len(classes))}
        training_dialog.add_preparation_message(f"クラス-インデックスマッピング: {class_mapping}")

        QApplication.processEvents()

        try:
            # データセット準備（ダイアログ参照を渡して出力を転送）
            dataset_info = self._prepare_yolo_dataset(task_type, classes, annotations, training_dialog)
            
            # MLflowManagerの初期化
            if not hasattr(self, 'mlflow_manager'):
                self.mlflow_manager = MLflowManager(self.folder_path)

            # Ultralytics YOLOモデルとMLflowのインポート
            try:
                from ultralytics import YOLO, settings
                settings.update({"mlflow": False})

            except ImportError as e:
                missing_package = "ultralytics" if "ultralytics" in str(e) else "mlflow" if "mlflow" in str(e) else "依存パッケージ"
                QMessageBox.critical(self, "エラー", f"{missing_package}パッケージがインストールされていません。\npip install {missing_package} でインストールしてください。")
                return

            # デバイスの選択
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._log_to_dialog(f"Using device for YOLO training: {device}", training_dialog)

            # MLflow情報を表示
            mlflow_uri = f"file:///{self.folder_path}/mlruns"
            self._log_to_dialog(f"MLflowトラッキングURI: {mlflow_uri}", training_dialog)
            self._log_to_dialog("MLflow初期化成功: " + mlflow_uri, training_dialog)
            self._log_to_dialog("実験を設定: yolo_detection_models", training_dialog)
            
            # 学習用の進捗ダイアログ
            progress = QProgressDialog(
                f"YOLO{task_name}モデル '{model_type}' の学習準備中...", 
                "キャンセル", 0, 100, self
            )
            progress.setWindowTitle(f"YOLO{task_name}モデル学習")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # モデルの準備
            model, pretrained_info = self._prepare_yolo_model(
                model_type, training_config, progress
            )
            if not model:
                return
            
            # MLflow環境設定
            self._setup_yolo_mlflow_environment(task_type)
            
            # トレーニング設定のカスタマイズ（カスタムモデル名があればそれを使用）
            if training_config.get('model_name'):
                run_name = training_config['model_name']
            else:
                run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            progress.setLabelText(f"{task_name}学習開始...")
            progress.setValue(20)
            QApplication.processEvents()

            # 学習パラメータを準備
            training_params = {
                'data': dataset_info['yaml_file'],
                'epochs': training_config['num_epochs'],
                'batch': training_config['batch_size'],
                'imgsz': training_config['img_size'],
                'project': models_dir,
                'name': run_name,
                'device': device.type,
                'workers': 0,
                'close_mosaic': 10 if training_config['mosaic'] > 0 else 0,
                'patience': training_config['patience'],
                'exist_ok': True,
                'lr0': training_config['learning_rate'],
                'lrf': training_config['learning_rate'] / 10,
                # オーグメンテーション設定
                'mosaic': training_config['mosaic'],
                'fliplr': training_config['fliplr'],
                'hsv_h': training_config['hsv_h'],
                'hsv_s': training_config['hsv_s'],
                'hsv_v': training_config['hsv_v'],
                'translate': training_config['translate'],
                'scale': training_config['scale'],
                'erasing': training_config['erasing']
            }

            # 進捗ダイアログを閉じる
            if progress:
                progress.close()

            # 学習を開始
            training_dialog.start_training(model, training_params)
            training_dialog.exec_()

            # 学習が完了したかチェック
            if not training_dialog.training_completed:
                print("学習がキャンセルまたは失敗しました")
                return

            # 学習結果を取得（ワーカーから保存された結果を取得）
            results = training_dialog.worker.results

            # 新しい進捗ダイアログを作成（MLflow記録用）
            progress = QProgressDialog(
                "MLflowに学習結果を記録中...",
                None, 0, 100, self
            )
            progress.setWindowTitle("学習結果記録")
            progress.setWindowModality(Qt.WindowModal)
            progress.setCancelButton(None)  # キャンセル不可
            progress.show()
            progress.setValue(10)
            QApplication.processEvents()
            
            # MLflowに学習結果を記録
            mlflow_info = self._log_yolo_training(
                task_type=task_type,
                model_type=model_type,
                results=results,
                training_config=training_config,
                dataset_info={
                    "num_classes": len(classes),
                    "classes": classes,
                    "train_samples": len([f for f in os.listdir(os.path.join(dataset_info['train_dir'], "images")) if f.endswith(('.jpg', '.jpeg', '.png'))]),
                    "val_samples": len([f for f in os.listdir(os.path.join(dataset_info['val_dir'], "images")) if f.endswith(('.jpg', '.jpeg', '.png'))]),
                    "total_annotations": annotation_info['total_count'],
                    "annotation_images": annotation_info['image_count'],
                    "task_type": task_type
                }
            )
            
            # モデルリストを更新
            self.refresh_yolo_unified_model_list()
            
            progress.setValue(100)
            progress.close()
            
            # 学習結果を表示
            self._show_yolo_training_success(
                task_name=task_name,
                model_type=model_type,
                results=results,
                device=device,
                pretrained_info=pretrained_info,
                run_name=run_name,
                mlflow_info=mlflow_info
            )
            
        except Exception as e:
            if progress is not None:
                progress.close()
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "エラー",
                f"YOLO{task_name}モデル学習中にエラーが発生しました: {str(e)}"
            )

    def _validate_yolo_annotations(self, task_type):
        """YOLOアノテーションの検証"""
        
        if task_type == "detect":
            if not self.bbox_annotations:
                QMessageBox.warning(self, "警告", "物体検知アノテーションがありません。")
                return None, None
            
            annotations = self.bbox_annotations
            total_boxes = sum(len(boxes) for boxes in annotations.values())
            
            print(f"\n=== バウンディングボックスアノテーション確認 ===")
            print(f"アノテーション数: {len(annotations)}")
            print(f"総バウンディングボックス数: {total_boxes}")
            print("=" * 50)
            
            return annotations, {"total_count": total_boxes, "image_count": len(annotations)}
            
        elif task_type == "segment":
            if not self.segmentation_annotations:
                QMessageBox.warning(self, "警告", "セグメンテーションアノテーションがありません。")
                return None, None
            
            annotations = self.segmentation_annotations
            
            # セグメンテーションデータの詳細検証
            total_segments = 0
            valid_segments = 0
            
            for index, segments in annotations.items():
                if segments:
                    total_segments += len(segments)
                    for seg in segments:
                        points = None
                        if isinstance(seg, dict):
                            points = seg.get('points', [])
                        else:
                            points = getattr(seg, 'points', [])
                        
                        if points and len(points) >= 3:
                            valid_segments += 1
            
            print(f"\n=== セグメンテーションアノテーション確認 ===")
            print(f"アノテーション数: {len(annotations)}")
            print(f"総セグメンテーション数: {total_segments}")
            print(f"有効なセグメンテーション数: {valid_segments}")
            print("=" * 50)
            
            if valid_segments == 0:
                result = QMessageBox.question(
                    self,
                    "セグメンテーションデータなし",
                    "有効なセグメンテーションアノテーションが見つかりません。\n\n"
                    "バウンディングボックスから矩形セグメンテーションを自動生成しますか？\n"
                    "（より高精度な結果を得るには、手動でポリゴンアノテーションを作成することを推奨します）",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if result != QMessageBox.Yes:
                    return None, None
                
                # バウンディングボックスからセグメンテーションを自動生成
                self.generate_segmentation_from_bbox()
            
            return annotations, {"total_count": valid_segments, "image_count": len(annotations)}
        
        return None, None

    # def _prepare_yolo_dataset(self, task_type, classes, annotations):
    #     """YOLOデータセットの準備"""
        
    #     # データディレクトリ構造の作成
    #     train_dir = os.path.join(yolo_dataset_dir, "train")
    #     val_dir = os.path.join(yolo_dataset_dir, "val")
    #     os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    #     os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    #     os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
    #     os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
        
    #     # クラス名ファイルの保存
    #     with open(os.path.join(yolo_dataset_dir, "classes.txt"), 'w') as f:
    #         for cls in classes:
    #             f.write(f"{cls}\n")
        
    #     # データセット設定YAMLファイルの作成
    #     task_name = "物体検知" if task_type == "detect" else "セグメンテーション"
    #     yaml_content = f"""# {task_name}用データセット設定
    # path: {yolo_dataset_dir}
    # train: train/images
    # val: val/images

    # # クラス数
    # nc: {len(classes)}

    # # クラス名
    # names: {classes}
    # """
        
    #     yaml_file = os.path.join(yolo_dataset_dir, "dataset.yaml")
    #     with open(yaml_file, 'w') as f:
    #         f.write(yaml_content)
        
    #     # タスクに応じたアノテーションデータのエクスポート
    #     if task_type == "detect":
    #         self.export_annotations_to_yolo(train_dir, val_dir, classes)
    #     elif task_type == "segment":
    #         if hasattr(self, 'export_segmentation_annotations_to_yolo'):
    #             self.export_segmentation_annotations_to_yolo(train_dir, val_dir, classes)
    #         else:
    #             QMessageBox.critical(
    #                 self, 
    #                 "エラー", 
    #                 "export_segmentation_annotations_to_yoloメソッドが実装されていません。"
    #             )
    #             raise Exception("Segmentation export method not implemented")
        
    #     return {
    #         'train_dir': train_dir,
    #         'val_dir': val_dir,
    #         'yaml_file': yaml_file
    #     }

    def _log_to_dialog(self, message, training_dialog=None):
        """メッセージをダイアログとターミナルの両方に出力"""
        print(message)  # ターミナルにも出力
        if training_dialog:
            training_dialog.add_preparation_message(message)

    def _prepare_yolo_dataset(self, task_type, classes, annotations, training_dialog=None):
        """YOLOデータセットの準備 - タスク別データ分離版"""

        # データディレクトリ構造の作成
        train_dir = os.path.join(yolo_dataset_dir, "train")
        val_dir = os.path.join(yolo_dataset_dir, "val")
        
        # 既存のデータをクリア（混在を防ぐため）
        import shutil
        if os.path.exists(yolo_dataset_dir):
            shutil.rmtree(yolo_dataset_dir)
        
        os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
        
        # クラス名ファイルの保存
        with open(os.path.join(yolo_dataset_dir, "classes.txt"), 'w') as f:
            for cls in classes:
                f.write(f"{cls}\n")
        
        # データセット設定YAMLファイルの作成
        task_name = "物体検知" if task_type == "detect" else "セグメンテーション"
        yaml_content = f"""# {task_name}用データセット設定
    path: {yolo_dataset_dir}
    train: train/images
    val: val/images

    # クラス数
    nc: {len(classes)}

    # クラス名
    names: {classes}
    """
        
        yaml_file = os.path.join(yolo_dataset_dir, "dataset.yaml")
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        # **重要**: タスクに応じて適切なアノテーションのみをエクスポート
        if task_type == "detect":
            print("=== バウンディングボックス専用データセット作成 ===")
            # バウンディングボックスのみをエクスポート
            self.export_bbox_only_to_yolo(train_dir, val_dir, classes)
        elif task_type == "segment":
            print("=== セグメンテーション専用データセット作成 ===")
            # セグメンテーションデータのみをエクスポート
            self.export_segmentation_only_to_yolo(train_dir, val_dir, classes)
        
        return {
            'train_dir': train_dir,
            'val_dir': val_dir,
            'yaml_file': yaml_file
        }

    def export_segmentation_only_to_yolo(self, train_dir, val_dir, classes):
        """セグメンテーションデータのみをYOLO形式でエクスポート"""
        
        print("セグメンテーション専用アノテーションエクスポート開始")
        
        # クラス名のインデックスマッピング
        class_to_index = {class_name: i for i, class_name in enumerate(classes)}
        print(f"クラス-インデックスマッピング: {class_to_index}")
        
        # セグメンテーションアノテーションがあるインデックスのみを取得
        valid_indices = []
        for idx, segments in self.segmentation_annotations.items():
            if segments and len(segments) > 0:
                # 各セグメンテーションに有効な座標があるかチェック
                has_valid_segments = False
                for seg in segments:
                    points = None
                    if isinstance(seg, dict):
                        points = seg.get('points', [])
                    else:
                        points = getattr(seg, 'points', [])
                    
                    if points and len(points) >= 3:  # 最低3点必要
                        has_valid_segments = True
                        break
                
                if has_valid_segments:
                    valid_indices.append(idx)
        
        print(f"有効なセグメンテーションデータがあるインデックス: {len(valid_indices)}個")
        
        if len(valid_indices) == 0:
            raise Exception("有効なセグメンテーションアノテーションが見つかりません。")
        
        # データを分割
        import random
        random.shuffle(valid_indices)
        
        split_point = int(len(valid_indices) * 0.8)
        train_indices = valid_indices[:split_point]
        val_indices = valid_indices[split_point:]
        
        print(f"学習用: {len(train_indices)}枚, 検証用: {len(val_indices)}枚")
        
        # 学習用データのエクスポート
        train_success = self._export_segmentation_subset(train_indices, train_dir, class_to_index)
        
        # 検証用データのエクスポート
        val_success = self._export_segmentation_subset(val_indices, val_dir, class_to_index)
        
        print(f"セグメンテーション専用アノテーションエクスポート完了")
        print(f"学習用成功: {train_success}/{len(train_indices)}, 検証用成功: {val_success}/{len(val_indices)}")
        
        if train_success == 0 or val_success == 0:
            raise Exception("セグメンテーションデータのエクスポートに失敗しました。")

    def export_bbox_only_to_yolo(self, train_dir, val_dir, classes):
        """バウンディングボックスのみをYOLO形式でエクスポート（既存メソッドの改良版）"""
        
        print("バウンディングボックス専用アノテーションエクスポート開始")
        
        # クラス名のインデックスマッピング
        class_to_index = {class_name: i for i, class_name in enumerate(classes)}
        print(f"クラス-インデックスマッピング: {class_to_index}")
        
        # バウンディングボックスアノテーションがあるインデックスのみを使用
        valid_indices = list(self.bbox_annotations.keys())
        
        import random
        random.shuffle(valid_indices)
        
        split_point = int(len(valid_indices) * 0.8)
        train_indices = valid_indices[:split_point]
        val_indices = valid_indices[split_point:]
        
        print(f"学習用: {len(train_indices)}枚, 検証用: {len(val_indices)}枚")
        
        # 学習用データのエクスポート
        train_success = self._export_bbox_subset(train_indices, train_dir, class_to_index)
        
        # 検証用データのエクスポート
        val_success = self._export_bbox_subset(val_indices, val_dir, class_to_index)
        
        print(f"バウンディングボックス専用アノテーションエクスポート完了")
        print(f"学習用成功: {train_success}/{len(train_indices)}, 検証用成功: {val_success}/{len(val_indices)}")

    # def _export_bbox_subset(self, indices, output_dir, class_to_index):
    #     """バウンディングボックスサブセットのエクスポート"""
        
    #     success_count = 0
        
    #     for idx in indices:
    #         if idx in self.bbox_annotations:
    #             try:
    #                 # 画像をコピー
    #                 source_image_path = self.images[idx]
    #                 image_filename = os.path.basename(source_image_path)
    #                 dest_image_path = os.path.join(output_dir, "images", image_filename)
                    
    #                 import shutil
    #                 shutil.copy2(source_image_path, dest_image_path)
                    
    #                 # バウンディングボックスアノテーションを処理
    #                 label_filename = os.path.splitext(image_filename)[0] + ".txt"
    #                 label_path = os.path.join(output_dir, "labels", label_filename)
                    
    #                 # 画像サイズを取得
    #                 from PIL import Image
    #                 with Image.open(source_image_path) as img:
    #                     img_width, img_height = img.size
                    
    #                 # ラベルファイルを作成（バウンディングボックス形式のみ）
    #                 with open(label_path, 'w') as f:
    #                     for bbox in self.bbox_annotations[idx]:
    #                         if bbox.class_name in class_to_index:
    #                             class_id = class_to_index[bbox.class_name]
                                
    #                             # 正規化座標に変換
    #                             x_center = (bbox.x + bbox.width / 2) / img_width
    #                             y_center = (bbox.y + bbox.height / 2) / img_height
    #                             width = bbox.width / img_width
    #                             height = bbox.height / img_height
                                
    #                             # YOLO 検出形式で書き込み
    #                             # フォーマット: class_id x_center y_center width height
    #                             f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
    #                 success_count += 1
                    
    #             except Exception as e:
    #                 print(f"バウンディングボックス インデックス {idx} の処理中にエラー: {e}")
        
    #     return success_count
###
    def _export_bbox_subset(self, indices, output_dir, class_to_index):
        """バウンディングボックスサブセットのエクスポート（修正版）"""
        
        success_count = 0
        
        for idx in indices:
            if idx in self.bbox_annotations:
                try:
                    # 画像をコピー
                    source_image_path = self.images[idx]
                    image_filename = os.path.basename(source_image_path)
                    dest_image_path = os.path.join(output_dir, "images", image_filename)
                    
                    import shutil
                    shutil.copy2(source_image_path, dest_image_path)
                    
                    # バウンディングボックスアノテーションを処理
                    label_filename = os.path.splitext(image_filename)[0] + ".txt"
                    label_path = os.path.join(output_dir, "labels", label_filename)
                    
                    # ラベルファイルを作成（バウンディングボックス形式のみ）
                    with open(label_path, 'w') as f:
                        for bbox in self.bbox_annotations[idx]:
                            # クラス名の取得（辞書形式とオブジェクト形式の両方に対応）
                            class_name = None
                            if isinstance(bbox, dict):
                                class_name = bbox.get('class') or bbox.get('class_name')
                            else:
                                class_name = getattr(bbox, 'class', None) or getattr(bbox, 'class_name', None)
                            
                            if class_name and class_name in class_to_index:
                                class_id = class_to_index[class_name]
                                
                                # バウンディングボックス座標を取得（既に正規化済み）
                                if isinstance(bbox, dict):
                                    # 辞書形式（現在のアプリケーション形式）
                                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                                else:
                                    # オブジェクト形式（古い形式への対応）
                                    # 画像サイズを取得して正規化
                                    from PIL import Image
                                    with Image.open(source_image_path) as img:
                                        img_width, img_height = img.size
                                    
                                    x1 = bbox.x / img_width
                                    y1 = bbox.y / img_height
                                    x2 = (bbox.x + bbox.width) / img_width
                                    y2 = (bbox.y + bbox.height) / img_height
                                
                                # YOLO形式に変換（左上・右下座標 → 中心・幅・高さ）
                                x_center = (x1 + x2) / 2
                                y_center = (y1 + y2) / 2
                                width = x2 - x1
                                height = y2 - y1
                                
                                # YOLO検出形式で書き込み
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"バウンディングボックス インデックス {idx} の処理中にエラー: {e}")
                    import traceback
                    traceback.print_exc()
        
        return success_count

    def _export_segmentation_subset(self, indices, output_dir, class_to_index):
        """セグメンテーションサブセットのエクスポート（修正版）"""
        
        success_count = 0
        
        for idx in indices:
            if idx in self.segmentation_annotations:
                try:
                    # 画像をコピー
                    source_image_path = self.images[idx]
                    image_filename = os.path.basename(source_image_path)
                    dest_image_path = os.path.join(output_dir, "images", image_filename)
                    
                    import shutil
                    shutil.copy2(source_image_path, dest_image_path)
                    
                    # セグメンテーションアノテーションを処理
                    label_filename = os.path.splitext(image_filename)[0] + ".txt"
                    label_path = os.path.join(output_dir, "labels", label_filename)
                    
                    # 画像サイズを取得
                    from PIL import Image
                    with Image.open(source_image_path) as img:
                        img_width, img_height = img.size
                    
                    # ラベルファイルを作成（セグメンテーション形式）
                    with open(label_path, 'w') as f:
                        for seg in self.segmentation_annotations[idx]:
                            # クラス名を取得
                            class_name = None
                            if isinstance(seg, dict):
                                class_name = seg.get('class') or seg.get('class_name')
                                points = seg.get('points', [])
                            else:
                                class_name = getattr(seg, 'class', None) or getattr(seg, 'class_name', None)
                                points = getattr(seg, 'points', [])
                            
                            if class_name and class_name in class_to_index and points and len(points) >= 3:
                                class_id = class_to_index[class_name]
                                
                                # ポイントを正規化座標に変換
                                normalized_points = []
                                for point in points:
                                    # アプリではピクセル座標で保存されているため正規化が必要
                                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                                        # タプル/リスト形式: (x, y)
                                        x = point[0] / img_width
                                        y = point[1] / img_height
                                    elif isinstance(point, dict):
                                        # 辞書形式: {'x': x, 'y': y}
                                        x = point.get('x', 0) / img_width
                                        y = point.get('y', 0) / img_height
                                    else:
                                        # オブジェクト形式: point.x, point.y
                                        x = getattr(point, 'x', 0) / img_width
                                        y = getattr(point, 'y', 0) / img_height
                                    
                                    # 座標を0-1の範囲にクランプ
                                    x = max(0.0, min(1.0, x))
                                    y = max(0.0, min(1.0, y))
                                    
                                    normalized_points.extend([x, y])
                                
                                # YOLO セグメンテーション形式で書き込み
                                if len(normalized_points) >= 6:  # 最低3点 (6座標)
                                    points_str = ' '.join([f"{coord:.6f}" for coord in normalized_points])
                                    f.write(f"{class_id} {points_str}\n")
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"セグメンテーション インデックス {idx} の処理中にエラー: {e}")
                    import traceback
                    traceback.print_exc()
        
        return success_count


    # セグメンテーション学習前のバリデーション強化
    def _validate_yolo_annotations(self, task_type):
        """YOLOアノテーションの検証 - セグメンテーション強化版"""
        
        if task_type == "detect":
            if not self.bbox_annotations:
                QMessageBox.warning(self, "警告", "物体検知アノテーションがありません。")
                return None, None
            
            annotations = self.bbox_annotations
            total_boxes = sum(len(boxes) for boxes in annotations.values())
            
            print(f"\n=== バウンディングボックスアノテーション確認 ===")
            print(f"アノテーション数: {len(annotations)}")
            print(f"総バウンディングボックス数: {total_boxes}")
            print("=" * 50)
            
            return annotations, {"total_count": total_boxes, "image_count": len(annotations)}
            
        elif task_type == "segment":
            if not self.segmentation_annotations:
                QMessageBox.warning(self, "警告", "セグメンテーションアノテーションがありません。")
                return None, None
            
            annotations = self.segmentation_annotations
            
            # セグメンテーションデータの詳細検証
            total_segments = 0
            valid_segments = 0
            valid_images = 0
            
            for index, segments in annotations.items():
                image_has_valid_segments = False
                if segments and len(segments) > 0:
                    total_segments += len(segments)
                    for seg in segments:
                        points = None
                        if isinstance(seg, dict):
                            points = seg.get('points', [])
                        else:
                            points = getattr(seg, 'points', [])
                        
                        if points and len(points) >= 3:  # 最低3点必要
                            valid_segments += 1
                            image_has_valid_segments = True
                
                if image_has_valid_segments:
                    valid_images += 1
            
            print(f"\n=== セグメンテーションアノテーション確認 ===")
            print(f"アノテーション辞書数: {len(annotations)}")
            print(f"総セグメンテーション数: {total_segments}")
            print(f"有効なセグメンテーション数: {valid_segments}")
            print(f"有効なセグメンテーションがある画像数: {valid_images}")
            print("=" * 50)
            
            if valid_segments == 0:
                QMessageBox.critical(
                    self,
                    "セグメンテーションデータなし",
                    "有効なセグメンテーションアノテーションが見つかりません。\n\n"
                    "セグメンテーション学習には最低3点以上のポリゴンアノテーションが必要です。\n"
                    "手動でポリゴンアノテーションを作成してから再試行してください。"
                )
                return None, None
            
            if valid_images < 4:  # 最低限の学習データ
                QMessageBox.warning(
                    self,
                    "データ不足",
                    f"有効なセグメンテーションデータが {valid_images} 枚しかありません。\n"
                    f"セグメンテーション学習には最低4枚以上の画像が推奨されます。"
                )
            
            return annotations, {"total_count": valid_segments, "image_count": valid_images}
        
        return None, None

    def _prepare_yolo_model(self, model_type, training_config, progress):
        """YOLOモデルの準備"""

        pretrained_model_path = None
        model_path = None

        if training_config['use_pretrained']:
            # 事前学習済みモデルをダウンロード
            progress.setLabelText(f"事前学習済み {model_type} モデルをダウンロードしています...")
            progress.setValue(5)
            QApplication.processEvents()

            pretrained_model_path = self.download_pretrained_yolo_model(model_type)
            if not pretrained_model_path:
                progress.close()
                QMessageBox.critical(self, "エラー", f"事前学習済み {model_type} モデルの準備に失敗しました。")
                return None, None

            model = YOLO(pretrained_model_path)
            pretrained_info = f"事前学習済みの重み (ダウンロード済み: {os.path.basename(pretrained_model_path)})"
        else:
            # 現在ロードされているモデルを使用
            # セグメンテーションモデルか物体検知モデルかを判定
            is_segmentation = 'seg' in model_type.lower()

            if is_segmentation:
                # セグメンテーションモデル
                if hasattr(self, 'yolo_seg_model_file') and os.path.exists(self.yolo_seg_model_file):
                    model_path = self.yolo_seg_model_file
                    model = YOLO(model_path)
                    pretrained_info = f"現在のモデル重み: {os.path.basename(model_path)}"
                else:
                    progress.close()
                    QMessageBox.critical(self, "エラー", "現在のセグメンテーションモデルが読み込まれていません。事前学習済みモデルを使用するか、モデルを読み込んでから再試行してください。")
                    return None, None
            else:
                # 物体検知モデル
                if hasattr(self, 'yolo_model_file') and os.path.exists(self.yolo_model_file):
                    model_path = self.yolo_model_file
                    model = YOLO(model_path)
                    pretrained_info = f"現在のモデル重み: {os.path.basename(model_path)}"
                else:
                    progress.close()
                    QMessageBox.critical(self, "エラー", "現在の物体検知モデルが読み込まれていません。事前学習済みモデルを使用するか、モデルを読み込んでから再試行してください。")
                    return None, None

        return model, pretrained_info

    def _setup_yolo_mlflow_environment(self, task_type):
        """YOLO MLflow環境の設定"""
        
        # MLflowManagerを使用してYOLO実験を設定
        if task_type == "detect":
            self.mlflow_manager.set_experiment(ModelType.YOLO_DETECTION)
        elif task_type == "segment":
            self.mlflow_manager.set_experiment(ModelType.YOLO_SEGMENTATION)

    def _log_yolo_training(self, task_type, model_type, results, training_config, dataset_info):
        """YOLO学習結果をMLflowに記録"""
        
        try:
            # 学習パラメータの準備
            training_params = {
                "model_type": model_type,
                "epochs": training_config['num_epochs'],
                "batch_size": training_config['batch_size'],
                "img_size": training_config['img_size'],
                "learning_rate": training_config['learning_rate'],
                "patience": training_config['patience'],
                "initial_weights": "pretrained" if training_config['use_pretrained'] else "current_model",
                "augmentation_enabled": training_config['augmentation_enabled'],
                "mosaic": training_config['mosaic'],
                "fliplr": training_config['fliplr'],
                "hsv_h": training_config['hsv_h'],
                "hsv_s": training_config['hsv_s'],
                "hsv_v": training_config['hsv_v'],
                "translate": training_config['translate'],
                "scale": training_config['scale'],
                "erasing": training_config['erasing'],
                "data_folder": self.folder_path if hasattr(self, 'folder_path') and self.folder_path else "unknown",
                "model_name": training_config.get('model_name', ''),
                "comment": training_config.get('comment', '')
            }
            
            # タスクタイプに応じてMLflowに記録
            if task_type == "detect":
                success = self.mlflow_manager.log_yolo_model(
                    model_type=model_type,
                    results=results,
                    training_params=training_params,
                    dataset_info=dataset_info
                )
            elif task_type == "segment":
                success = self.mlflow_manager.log_yolo_segmentation_model(
                    model_path=os.path.join(results.save_dir, "weights", "best.pt"),
                    training_params={
                        "model_type": model_type,
                        "architecture": "yolo_segmentation",
                        **training_params
                    },
                    metrics={
                        "box_mAP": float(results.box.map) if hasattr(results, 'box') and hasattr(results.box, 'map') else 0.0,
                        "mask_mAP": float(results.masks.map) if hasattr(results, 'masks') and hasattr(results.masks, 'map') else 0.0,
                        "final_loss": 0.0  # YOLOの場合は最終損失を取得
                    },
                    dataset_info=dataset_info
                )
            
            if success:
                return "MLflowに学習履歴を記録しました。\n「MLflow比較」ボタンで結果を確認できます。"
            else:
                return "MLflowへの記録中にエラーが発生しました。"
                
        except Exception as e:
            print(f"YOLO MLflow記録エラー: {e}")
            return f"MLflowへの記録中にエラーが発生しました: {str(e)}"

    def _show_yolo_training_success(self, task_name, model_type, results, device, pretrained_info, run_name, mlflow_info):
        """YOLO学習成功メッセージを表示"""

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("学習完了")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(
            f"YOLO{task_name}モデルの学習が完了しました。\n"
            f"最終mAP: {results.maps}\n"
            f"使用デバイス: {device}\n"
            f"初期化: {pretrained_info}\n\n"
            f"モデル保存先: {os.path.join(models_dir, run_name, 'weights')}\n"
            f"{mlflow_info}"
        )

        # OKボタン
        ok_button = msg_box.addButton(QMessageBox.Ok)

        # MLflow を開くボタンを追加
        mlflow_button = msg_box.addButton("MLflowを開く", QMessageBox.ActionRole)

        msg_box.exec_()

        # MLflowボタンがクリックされた場合
        if msg_box.clickedButton() == mlflow_button:
            self.mlflow_manager.open_ui()

    def _create_yolo_training_dialog(self, task_name, model_type, annotation_info):
        """YOLO学習設定ダイアログを作成"""
        
        training_settings = QDialog(self)
        training_settings.setWindowTitle(f"YOLO{task_name}モデル学習設定")
        training_settings.setMinimumWidth(500)
        training_settings.setMinimumHeight(750)
        
        settings_layout = QVBoxLayout(training_settings)
        
        # 統計情報を表示（削除済みマークを考慮）
        excluded_count = annotation_info.get("excluded_count", 0)
        total_images = len(self.images) if self.images else 0
        total_count = annotation_info.get("total_count", 0)
        image_count = annotation_info.get("image_count", 0)
        
        # アノテーション総数から削除済みを取得（物体検知/セグメンテーション別）
        if task_name == "物体検知":
            total_annotated_images = len(getattr(self, 'bbox_annotations', {}))
        else:  # セグメンテーション
            total_annotated_images = len(getattr(self, 'segmentation_annotations', {}))
        
        stats_label = QLabel(f"<b>学習データ統計:</b><br>"
                           f"総読み込み画像数: {total_images}枚<br>"
                           f"{task_name}アノテーション済み画像数: {total_annotated_images}枚<br>"
                           f"<b style='color: #2E7D32; font-size: 14px;'>実際の学習使用枚数: {image_count}枚</b><br>"
                           f"({total_annotated_images}枚 - 削除済み{excluded_count}枚)<br>"
                           f"総{task_name}アノテーション数: {total_count}個<br>"
                           f"<span style='color: #FF6600;'>※ 削除マークされた画像は学習対象から除外されます</span>")
        stats_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px;")
        settings_layout.addWidget(stats_label)
        
        settings_layout.addWidget(QLabel(""))  # スペース追加
        
        # タブウィジェットを作成
        tabs = QTabWidget()
        
        # 基本設定タブ
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # モデル初期化設定
        init_group = QGroupBox("モデル初期化設定")
        init_layout = QVBoxLayout(init_group)
        
        # 初期重みの選択
        training_settings.weights_radio_pretrained = QRadioButton("事前学習済みの重みを使用 (推奨)")
        training_settings.weights_radio_pretrained.setChecked(True)  # デフォルト選択
        init_layout.addWidget(training_settings.weights_radio_pretrained)
        
        # 現在のモデルを選択
        training_settings.weights_radio_current = QRadioButton("現在読み込まれているモデルの重みを使用")
        init_layout.addWidget(training_settings.weights_radio_current)
        
        # 現在読み込まれているモデルの情報を表示
        current_model_info = QLabel("現在のモデル: なし")
        model_loaded = False
        model_name = "Unknown"

        # 物体検知モデルまたはセグメンテーションモデルがロードされているかチェック
        if task_name == "物体検知":
            if hasattr(self, 'yolo_model') and hasattr(self, 'yolo_model_file'):
                model_name = os.path.basename(self.yolo_model_file)
                model_loaded = True
        else:  # セグメンテーション
            if hasattr(self, 'yolo_seg_model') and hasattr(self, 'yolo_seg_model_file'):
                model_name = os.path.basename(self.yolo_seg_model_file)
                model_loaded = True

        if model_loaded:
            current_model_info.setText(f"現在のモデル: {model_name}")
            training_settings.weights_radio_current.setEnabled(True)
        else:
            training_settings.weights_radio_current.setEnabled(False)
            current_model_info.setText("現在のモデル: なし（先にモデルを読み込んでください）")
        
        init_layout.addWidget(current_model_info)
        basic_layout.addWidget(init_group)
        
        # エポック数設定
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("学習エポック数:"))
        training_settings.epoch_spin = QSpinBox()
        training_settings.epoch_spin.setRange(1, 1000)
        training_settings.epoch_spin.setValue(30)  # デフォルト: 30エポック
        epoch_layout.addWidget(training_settings.epoch_spin)
        basic_layout.addLayout(epoch_layout)
        
        # バッチサイズ設定
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("バッチサイズ:"))
        training_settings.batch_spin = QSpinBox()
        training_settings.batch_spin.setRange(1, 128)
        training_settings.batch_spin.setValue(16)  # デフォルト: 16
        batch_layout.addWidget(training_settings.batch_spin)
        basic_layout.addLayout(batch_layout)
        
        # 入力サイズ設定
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("入力画像サイズ:"))
        training_settings.size_combo = QComboBox()
        size_options = [str(self.original_image_size), "320", "416", "512", "640", "768", "896", "1024"]
        default_index = 4  # デフォルトは640

        # 説明ラベルを追加
        size_layout.addWidget(QLabel(f"元画像: {self.original_image_width}×{self.original_image_height}"))

        training_settings.size_combo.addItems(size_options)
        training_settings.size_combo.setCurrentIndex(default_index)
        size_layout.addWidget(training_settings.size_combo)
        basic_layout.addLayout(size_layout)

        # 注意書き
        size_note = QLabel("注: 640以外のサイズを選択すると精度や速度に影響します")
        size_note.setStyleSheet("color: #888; font-style: italic;")
        basic_layout.addWidget(size_note)

        # Early Stopping設定
        training_settings.early_stopping_check = QCheckBox("Early Stopping を有効にする")
        training_settings.early_stopping_check.setChecked(True)
        basic_layout.addWidget(training_settings.early_stopping_check)
        
        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("忍耐エポック数:"))
        training_settings.patience_spin = QSpinBox()
        training_settings.patience_spin.setRange(1, 20)
        training_settings.patience_spin.setValue(10)
        training_settings.patience_spin.setEnabled(True)
        patience_layout.addWidget(training_settings.patience_spin)
        basic_layout.addLayout(patience_layout)
        
        # 学習率設定
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学習率:"))
        
        training_settings.lr_combo = QComboBox()
        learning_rates = ["0.01", "0.005", "0.001", "0.0005", "0.0001"]
        training_settings.lr_combo.addItems(learning_rates)
        training_settings.lr_combo.setCurrentIndex(2)  # デフォルト: 0.001
        lr_layout.addWidget(training_settings.lr_combo)
        basic_layout.addLayout(lr_layout)
        
        # タブに追加
        tabs.addTab(basic_tab, "基本設定")
        
        # データオーグメンテーションタブ
        aug_tab = QWidget()
        aug_layout = QVBoxLayout(aug_tab)
        
        # データオーグメンテーション有効化チェックボックス
        training_settings.aug_enable_check = QCheckBox("データオーグメンテーションを有効にする")
        training_settings.aug_enable_check.setChecked(True)
        aug_layout.addWidget(training_settings.aug_enable_check)
        
        # オーグメンテーション設定のスクロールエリア
        aug_scroll = QScrollArea()
        aug_scroll.setWidgetResizable(True)
        aug_scroll.setFrameShape(QFrame.NoFrame)
        
        aug_scroll_content = QWidget()
        aug_options_layout = QVBoxLayout(aug_scroll_content)
        
        # モザイク
        mosaic_layout = QHBoxLayout()
        training_settings.aug_mosaic_checkbox = QCheckBox("モザイク")
        training_settings.aug_mosaic_checkbox.setChecked(True)
        aug_mosaic_proba_label = QLabel("確率:")
        training_settings.aug_mosaic_proba = QDoubleSpinBox()
        training_settings.aug_mosaic_proba.setRange(0.0, 1.0)
        training_settings.aug_mosaic_proba.setSingleStep(0.1)
        training_settings.aug_mosaic_proba.setValue(1.0)
        mosaic_layout.addWidget(training_settings.aug_mosaic_checkbox)
        mosaic_layout.addWidget(aug_mosaic_proba_label)
        mosaic_layout.addWidget(training_settings.aug_mosaic_proba)
        mosaic_layout.addStretch()
        aug_options_layout.addLayout(mosaic_layout)
        
        # 水平反転
        flip_layout = QHBoxLayout()
        training_settings.aug_flip_checkbox = QCheckBox("水平反転")
        training_settings.aug_flip_checkbox.setChecked(True)
        aug_flip_proba_label = QLabel("確率:")
        training_settings.aug_flip_proba = QDoubleSpinBox()
        training_settings.aug_flip_proba.setRange(0.0, 1.0)
        training_settings.aug_flip_proba.setSingleStep(0.1)
        training_settings.aug_flip_proba.setValue(0.5)
        flip_layout.addWidget(training_settings.aug_flip_checkbox)
        flip_layout.addWidget(aug_flip_proba_label)
        flip_layout.addWidget(training_settings.aug_flip_proba)
        flip_layout.addStretch()
        aug_options_layout.addLayout(flip_layout)
        
        # HSV調整
        hsv_layout = QHBoxLayout()
        training_settings.aug_hsv_checkbox = QCheckBox("HSV調整")
        training_settings.aug_hsv_checkbox.setChecked(True)
        hsv_layout.addWidget(training_settings.aug_hsv_checkbox)
        hsv_layout.addStretch()
        aug_options_layout.addLayout(hsv_layout)
        
        # HSVの詳細設定
        hsv_details_layout = QGridLayout()
        hsv_details_layout.setContentsMargins(20, 0, 0, 0)
        
        hsv_details_layout.addWidget(QLabel("色相 (H):"), 0, 0)
        training_settings.aug_hsv_h = QDoubleSpinBox()
        training_settings.aug_hsv_h.setRange(0.0, 0.1)
        training_settings.aug_hsv_h.setSingleStep(0.005)
        training_settings.aug_hsv_h.setValue(0.015)
        hsv_details_layout.addWidget(training_settings.aug_hsv_h, 0, 1)
        
        hsv_details_layout.addWidget(QLabel("彩度 (S):"), 1, 0)
        training_settings.aug_hsv_s = QDoubleSpinBox()
        training_settings.aug_hsv_s.setRange(0.0, 1.0)
        training_settings.aug_hsv_s.setSingleStep(0.1)
        training_settings.aug_hsv_s.setValue(0.7)
        hsv_details_layout.addWidget(training_settings.aug_hsv_s, 1, 1)
        
        hsv_details_layout.addWidget(QLabel("明度 (V):"), 2, 0)
        training_settings.aug_hsv_v = QDoubleSpinBox()
        training_settings.aug_hsv_v.setRange(0.0, 1.0)
        training_settings.aug_hsv_v.setSingleStep(0.1)
        training_settings.aug_hsv_v.setValue(0.4)
        hsv_details_layout.addWidget(training_settings.aug_hsv_v, 2, 1)
        
        aug_options_layout.addLayout(hsv_details_layout)
        
        # 幾何変換
        geometry_layout = QHBoxLayout()
        training_settings.aug_geometry_checkbox = QCheckBox("幾何変換")
        training_settings.aug_geometry_checkbox.setChecked(True)
        geometry_layout.addWidget(training_settings.aug_geometry_checkbox)
        geometry_layout.addStretch()
        aug_options_layout.addLayout(geometry_layout)
        
        # 幾何変換の詳細設定
        geometry_details_layout = QGridLayout()
        geometry_details_layout.setContentsMargins(20, 0, 0, 0)
        
        geometry_details_layout.addWidget(QLabel("平行移動:"), 0, 0)
        training_settings.aug_translate = QDoubleSpinBox()
        training_settings.aug_translate.setRange(0.0, 0.5)
        training_settings.aug_translate.setSingleStep(0.05)
        training_settings.aug_translate.setValue(0.1)
        geometry_details_layout.addWidget(training_settings.aug_translate, 0, 1)
        
        geometry_details_layout.addWidget(QLabel("スケール:"), 1, 0)
        training_settings.aug_scale = QDoubleSpinBox()
        training_settings.aug_scale.setRange(0.0, 1.0)
        training_settings.aug_scale.setSingleStep(0.05)
        training_settings.aug_scale.setValue(0.5)
        geometry_details_layout.addWidget(training_settings.aug_scale, 1, 1)
        
        aug_options_layout.addLayout(geometry_details_layout)
        
        # RandomErase
        erase_layout = QHBoxLayout()
        training_settings.aug_erase_checkbox = QCheckBox("ランダムイレース")
        training_settings.aug_erase_checkbox.setChecked(True)
        aug_erase_proba_label = QLabel("確率:")
        training_settings.aug_erase_proba = QDoubleSpinBox()
        training_settings.aug_erase_proba.setRange(0.0, 1.0)
        training_settings.aug_erase_proba.setSingleStep(0.1)
        training_settings.aug_erase_proba.setValue(0.4)
        erase_layout.addWidget(training_settings.aug_erase_checkbox)
        erase_layout.addWidget(aug_erase_proba_label)
        erase_layout.addWidget(training_settings.aug_erase_proba)
        erase_layout.addStretch()
        aug_options_layout.addLayout(erase_layout)
        
        # オプションの有効/無効を連動させる
        def toggle_aug_options(checked):
            for w in aug_scroll_content.findChildren(QWidget):
                if w != training_settings.aug_enable_check:
                    w.setEnabled(checked)
        
        training_settings.aug_enable_check.toggled.connect(toggle_aug_options)
        
        # スクロールエリアに設定
        aug_scroll.setWidget(aug_scroll_content)
        aug_layout.addWidget(aug_scroll)
        
        # タブに追加
        tabs.addTab(aug_tab, "データオーグメンテーション")

        # タブをレイアウトに追加
        settings_layout.addWidget(tabs)

        # モデル名とコメント欄を追加
        settings_layout.addWidget(QLabel(""))  # スペース追加

        # モデル名編集欄
        model_name_group = QGroupBox("モデル名設定")
        model_name_layout = QVBoxLayout(model_name_group)

        # プレフィックス（固定）とサフィックス（編集可能）を分離
        # YOLOの場合はモデルタイプ（yolov8n, yolov8n-seg等）をプレフィックスとする
        yolo_prefix = f"{model_type}_"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # プレフィックスとサフィックスを横並びで表示
        name_input_layout = QHBoxLayout()
        name_input_layout.addWidget(QLabel("モデル名:"))

        # プレフィックス（固定、編集不可）
        prefix_label = QLabel(yolo_prefix)
        prefix_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; font-family: monospace;")
        name_input_layout.addWidget(prefix_label)

        # サフィックス（編集可能）
        training_settings.model_name_suffix_input = QLineEdit()
        training_settings.model_name_suffix_input.setText(timestamp)
        training_settings.model_name_suffix_input.setPlaceholderText("カスタム名を入力")
        name_input_layout.addWidget(training_settings.model_name_suffix_input)

        model_name_layout.addLayout(name_input_layout)

        # プレフィックスを保存（後で使用）
        training_settings.model_name_prefix = yolo_prefix

        model_name_note = QLabel(f"※ モデルタイプ ({model_type}) のプレフィックスは変更できません。.ptは自動的に付与されます")
        model_name_note.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        model_name_layout.addWidget(model_name_note)

        settings_layout.addWidget(model_name_group)

        # コメント欄
        comment_group = QGroupBox("学習コメント (MLflowに記録)")
        comment_layout = QVBoxLayout(comment_group)

        comment_layout.addWidget(QLabel("コメント:"))
        training_settings.comment_input = QPlainTextEdit()
        training_settings.comment_input.setPlaceholderText("この学習についてのメモやコメントを入力してください (任意)")
        training_settings.comment_input.setMaximumHeight(80)
        comment_layout.addWidget(training_settings.comment_input)

        settings_layout.addWidget(comment_group)

        # ボタンの配置
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(training_settings.accept)
        button_box.rejected.connect(training_settings.reject)
        settings_layout.addWidget(button_box)

        return training_settings

    def _get_yolo_training_config(self, task_name, model_type, annotation_info):
        """YOLO学習設定ダイアログから設定を取得（修正版）"""
        
        # 設定ダイアログを作成
        training_settings = self._create_yolo_training_dialog(task_name, model_type, annotation_info)
        
        if not training_settings.exec_():
            return None
        
        # 設定値の取得
        config = {
            'use_pretrained': training_settings.weights_radio_pretrained.isChecked(),
            'num_epochs': training_settings.epoch_spin.value(),
            'batch_size': training_settings.batch_spin.value(),
            'img_size': int(training_settings.size_combo.currentText()),
            'use_early_stopping': training_settings.early_stopping_check.isChecked(),
            'patience': training_settings.patience_spin.value() if training_settings.early_stopping_check.isChecked() else 0,
            'learning_rate': float(training_settings.lr_combo.currentText()),
            'augmentation_enabled': training_settings.aug_enable_check.isChecked(),
            'mosaic': training_settings.aug_mosaic_proba.value() if training_settings.aug_mosaic_checkbox.isChecked() and training_settings.aug_enable_check.isChecked() else 0.0,
            'fliplr': training_settings.aug_flip_proba.value() if training_settings.aug_flip_checkbox.isChecked() and training_settings.aug_enable_check.isChecked() else 0.0,
            'hsv_h': training_settings.aug_hsv_h.value() if training_settings.aug_hsv_checkbox.isChecked() and training_settings.aug_enable_check.isChecked() else 0.0,
            'hsv_s': training_settings.aug_hsv_s.value() if training_settings.aug_hsv_checkbox.isChecked() and training_settings.aug_enable_check.isChecked() else 0.0,
            'hsv_v': training_settings.aug_hsv_v.value() if training_settings.aug_hsv_checkbox.isChecked() and training_settings.aug_enable_check.isChecked() else 0.0,
            'translate': training_settings.aug_translate.value() if training_settings.aug_geometry_checkbox.isChecked() and training_settings.aug_enable_check.isChecked() else 0.0,
            'scale': training_settings.aug_scale.value() if training_settings.aug_geometry_checkbox.isChecked() and training_settings.aug_enable_check.isChecked() else 0.0,
            'erasing': training_settings.aug_erase_proba.value() if training_settings.aug_erase_checkbox.isChecked() and training_settings.aug_enable_check.isChecked() else 0.0,
            'model_name': training_settings.model_name_prefix + training_settings.model_name_suffix_input.text().strip(),
            'comment': training_settings.comment_input.toPlainText().strip()
        }

        return config
    #

    def generate_segmentation_from_bbox(self):
        """バウンディングボックスから矩形セグメンテーションを自動生成"""
        if not self.bbox_annotations:
            return
        
        generated_count = 0
        
        # インデックスベースでバウンディングボックスを処理
        for index, bboxes in self.bbox_annotations.items():
            if not bboxes:
                continue
                
            # 対応する画像のサイズを取得
            if index < len(self.images):
                img_path = self.images[index]
                
                # 画像サイズを取得
                try:
                    from PIL import Image
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except:
                    print(f"画像 {img_path} のサイズ取得に失敗")
                    continue
                
                # このインデックスのセグメンテーションリストを初期化
                if index not in self.segmentation_annotations:
                    self.segmentation_annotations[index] = []
                
                # 各バウンディングボックスを矩形セグメンテーションに変換
                for bbox in bboxes:
                    # 正規化座標から実際の座標に変換
                    x1 = int(bbox['x1'] * img_width)
                    y1 = int(bbox['y1'] * img_height)
                    x2 = int(bbox['x2'] * img_width)
                    y2 = int(bbox['y2'] * img_height)
                    
                    # 矩形の4つの角をポイントとして作成
                    points = [
                        (x1, y1),  # 左上
                        (x2, y1),  # 右上
                        (x2, y2),  # 右下
                        (x1, y2)   # 左下
                    ]
                    
                    # セグメンテーションデータを作成
                    seg_data = {
                        'class': bbox.get('class', 'unknown'),
                        'points': points
                    }
                    
                    self.segmentation_annotations[index].append(seg_data)
                    generated_count += 1
        
        print(f"バウンディングボックスから {generated_count}個のセグメンテーションを自動生成しました")
        
        # UI更新
        self.update_ui()

    def toggle_segmentation_inference_display(self, state):
        """セグメンテーション推論表示の切り替え"""
        show_inference = (state == Qt.Checked)
        self.show_segmentation_inference = show_inference
        
        # 画面更新
        self.main_image_view.update()
        
        # 表示情報の更新
        if show_inference:
            self.update_segmentation_inference_display()
            self.statusBar().showMessage("セグメンテーション推論結果表示をオンにしました", 3000)
        else:
            # 表示をクリア（物体検知推論結果ラベルをクリア）
            if hasattr(self, 'detection_inference_info_label'):
                self.detection_inference_info_label.setText(" ")  # スペースで高さを維持
            self.statusBar().showMessage("セグメンテーション推論結果表示をオフにしました", 3000)

    def refresh_yolo_unified_model_list(self):
        """統合されたYOLOモデルリストを更新 - 物体検知とセグメンテーションを統合"""
        if not hasattr(self, 'yolo_unified_model_combo'):
            return
                    
        self.yolo_unified_model_combo.clear()
        
        # 更新開始のメッセージを表示
        self.statusBar().showMessage("統合YOLOモデルリストを更新中...")
        
        # 現在選択されているモデルタイプを取得
        selected_model_type = self.yolo_model_combo.currentText()
        
        # YOLOモデルファイルを検索 - 物体検知とセグメンテーション両方
        unified_model_files = []
        
        # 1. 直下の.ptファイルを検索
        for file in os.listdir(models_dir):
            if file.endswith('.pt') and ('yolo' in file.lower()) and (selected_model_type.lower() in file.lower()):
                # タスクタイプを判定
                task_type = "セグメンテーション" if 'seg' in file.lower() else "物体検知"
                
                model_info = {
                    'path': file,
                    'parent': 'root',
                    'type': 'model',
                    'task': task_type,
                    'date': ''
                }
                unified_model_files.append(model_info)
        
        # 2. サブフォルダを検索
        for root, dirs, files in os.walk(models_dir):
            if root == models_dir:
                continue
                    
            for file in files:
                if file.endswith('.pt') and ('best' in file.lower() or 'last' in file.lower()):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, models_dir)
                    
                    folder_name = os.path.basename(root)
                    parent_folder_name = os.path.basename(os.path.dirname(root))
                    
                    # モデルタイプがフォルダ名に含まれているかチェック
                    model_match = (
                        selected_model_type.lower() in folder_name.lower() or 
                        selected_model_type.lower() in parent_folder_name.lower()
                    )
                    
                    if not model_match:
                        continue
                    
                    # タスクタイプを判定（フォルダ名、親フォルダ名、ファイル名から）
                    full_folder_path = root  # フルパスを取得
                    if ('seg' in folder_name.lower() or 'segment' in folder_name.lower() or 
                        'seg' in file.lower() or 'segment' in file.lower() or
                        'seg' in full_folder_path.lower() or 'segment' in full_folder_path.lower()):
                        task_type = "セグメンテーション"
                    else:
                        task_type = "物体検知"
                    
                    # 日時情報を取得
                    date_info = ""
                    parts = folder_name.split('_')
                    if len(parts) >= 3:
                        date_info = '_'.join(parts[1:])
                    else:
                        parent_parts = parent_folder_name.split('_')
                        if len(parent_parts) >= 3:
                            date_info = '_'.join(parent_parts[1:])
                    
                    model_info = {
                        'path': rel_path,
                        'parent': folder_name,
                        'type': 'best' if 'best' in file.lower() else 'last',
                        'task': task_type,
                        'date': date_info
                    }
                    unified_model_files.append(model_info)
        
        if not unified_model_files:
            self.yolo_unified_model_combo.addItem(f"{selected_model_type}のYOLOモデルが見つかりません")
            self.statusBar().showMessage(f"{selected_model_type}のYOLOモデルが見つかりません", 3000)
            return
        
        # モデルファイルをソート（ファイル作成日時順、新しいものが上）
        # カスタムサフィックスが追加された場合でも正しくソートされるよう、mtimeを使用
        def sort_key(model_info):
            # ファイルの完全なパスを取得
            if model_info['parent'] == 'root':
                file_path = os.path.join(models_dir, model_info['path'])
            else:
                file_path = os.path.join(models_dir, model_info['path'])

            # ファイルの作成日時を取得（存在しない場合は0）
            try:
                mtime = os.path.getmtime(file_path)
            except:
                mtime = 0

            # タスクタイプの優先順位を設定（物体検知を優先）
            task_priority = 0 if model_info['task'] == "物体検知" else 1

            # タスクタイプ優先、その後新しいものが上に来るように負の値を返す
            return (task_priority, -mtime)

        unified_model_files.sort(key=sort_key)
        
        # コンボボックスに追加
        for model_info in unified_model_files:
            # タスクタイプを短縮表示
            task_label = "物検" if model_info['task'] == "物体検知" else "セグ"

            if model_info['parent'] == 'root':
                display_name = f"[{task_label}] {model_info['path']}"
            else:
                display_name = f"[{task_label}] {model_info['parent'].split('_')[0]} [{model_info['date']}] ({model_info['type']})"

            self.yolo_unified_model_combo.addItem(display_name, model_info['path'])


        # 更新完了メッセージ
        detection_count = sum(1 for m in unified_model_files if m['task'] == "物体検知")
        segmentation_count = sum(1 for m in unified_model_files if m['task'] == "セグメンテーション")
        
        self.statusBar().showMessage(
            f"{len(unified_model_files)}個の{selected_model_type}モデルを読み込みました "
            f"(物体検知: {detection_count}, セグメンテーション: {segmentation_count})", 3000
        )

    def load_yolo_model_unified(self):
        """統合されたYOLOモデル読み込み - タスクタイプを自動判別"""
        if not self.images:
            QMessageBox.warning(self, "警告", "画像が読み込まれていません。")
            return
        
        # 選択されたモデル情報を取得
        current_index = self.yolo_unified_model_combo.currentIndex()
        selected_model_display = self.yolo_unified_model_combo.currentText()
        relative_path = self.yolo_unified_model_combo.itemData(current_index)
        
        if not relative_path or "が見つかりません" in selected_model_display:
            QMessageBox.warning(self, "警告", "有効なYOLOモデルが選択されていません。")
            return
        
        # タスクタイプを判定
        is_segmentation = "[セグ]" in selected_model_display
        task_type = "セグメンテーション" if is_segmentation else "物体検知"
        
        # モデルパスを構築
        model_path = os.path.join(models_dir, relative_path)
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "警告", f"選択されたモデルが見つかりません: {model_path}")
            return
        
        # 信頼度閾値の設定
        confidence, ok = QInputDialog.getDouble(
            self, 
            f"{task_type}モデル信頼度設定", 
            f"{task_type}の信頼度閾値 (0.0-1.0):",
            0.6, 0.01, 1.0, 2
        )
        
        if not ok:
            return
        
        # 進捗ダイアログ
        progress = QProgressDialog(
            f"{task_type}モデル '{os.path.basename(model_path)}' を読み込み中...",
            "キャンセル", 0, 100, self
        )
        progress.setWindowTitle("統合モデル読み込み")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        try:
            from ultralytics import YOLO
            
            progress.setValue(30)
            progress.setLabelText(f"{task_type}モデルをメモリに読み込み中...")
            QApplication.processEvents()
            
            # モデルを読み込み
            yolo_model = YOLO(model_path)

            progress.setValue(50)
            progress.setLabelText("クラス情報を取得中...")
            QApplication.processEvents()

            # モデルのクラス名情報を取得して反映
            if hasattr(yolo_model, 'names') and yolo_model.names:
                # クラス名を取得（辞書形式 {0: 'person', 1: 'car', ...}）
                class_names = list(yolo_model.names.values())
                # クラス名をカンマ区切りで結合
                class_names_str = ','.join(class_names)
                # UIのクラス入力欄に反映
                if hasattr(self, 'classes_input'):
                    self.classes_input.setText(class_names_str)
                # クラス色を初期化して実際に反映
                self._apply_class_changes(class_names)
                print(f"[モデル読み込み] クラス情報を取得: {len(class_names)}個のクラス")
                print(f"[モデル読み込み] クラス名: {class_names_str}")
                print(f"[モデル読み込み] クラス色を初期化しました")

            progress.setValue(60)
            QApplication.processEvents()

            # 新しいモデルを読み込む前に、既存のYOLOモデルをクリア
            if hasattr(self, 'yolo_model'):
                self.yolo_model = None
            if hasattr(self, 'yolo_seg_model'):
                self.yolo_seg_model = None

            # タスクタイプに応じてモデルを適切な変数に設定
            if is_segmentation:
                self.yolo_seg_model = yolo_model
                self.yolo_seg_confidence_threshold = confidence
                self.yolo_seg_model_file = model_path
                
                # セグメンテーション推論チェックボックスを有効にしてオン
                if hasattr(self, 'segmentation_inference_checkbox'):
                    self.segmentation_inference_checkbox.setEnabled(True)
                    self.segmentation_inference_checkbox.setToolTip("セグメンテーションモデルが読み込まれています")
                    self.segmentation_inference_checkbox.setChecked(True)
                
                # 物体検知推論チェックボックスを無効化
                if hasattr(self, 'detection_inference_checkbox'):
                    self.detection_inference_checkbox.setEnabled(False)
                    self.detection_inference_checkbox.setChecked(False)
                    self.detection_inference_checkbox.setToolTip("セグメンテーションモデルが読み込まれているため無効")
            else:
                self.yolo_model = yolo_model
                self.yolo_confidence_threshold = confidence
                self.yolo_model_file = model_path
                
                # 物体検知推論チェックボックスを有効にしてオン
                if hasattr(self, 'detection_inference_checkbox'):
                    self.detection_inference_checkbox.setEnabled(True)
                    self.detection_inference_checkbox.setToolTip("物体検知モデルが読み込まれています")
                    self.detection_inference_checkbox.setChecked(True)
                
                # セグメンテーション推論チェックボックスを無効化
                if hasattr(self, 'segmentation_inference_checkbox'):
                    self.segmentation_inference_checkbox.setEnabled(False)
                    self.segmentation_inference_checkbox.setChecked(False)
                    self.segmentation_inference_checkbox.setToolTip("物体検知モデルが読み込まれているため無効")
            
            # 各モデルの状態を更新
            self.update_inference_checkboxes_status()
            
            progress.setValue(70)
            progress.setLabelText("推論テストを実行中...")
            QApplication.processEvents()
            
            # 現在の画像で推論テスト
            if is_segmentation:
                self.run_single_yolo_segmentation_inference()
            else:
                self.run_single_yolo_inference()
            
            progress.setValue(100)
            progress.close()

            # 成功メッセージ
            model_name = os.path.basename(model_path)
            QMessageBox.information(
                self,
                "モデル読み込み完了",
                f"{task_type}モデル「{model_name}」を読み込みました。\n"
                f"信頼度閾値: {confidence}\n\n"
                f"画像送りごとに自動的に{task_type}推論が実行されます。"
            )

            # 自動運転モデル読み込み完了後、オートアノテーションボタンを有効化
            if hasattr(self, 'auto_annotate_button'):
                self.auto_annotate_button.setEnabled(True)

            # YOLOオートアノテーションボタンを有効化
            if hasattr(self, 'yolo_auto_annotate_btn'):
                self.yolo_auto_annotate_btn.setEnabled(True)

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "エラー",
                f"{task_type}モデルの読み込み中にエラーが発生しました: {str(e)}"
            )

    def on_yolo_model_type_changed(self, index):
        """YOLOモデルタイプが変更されたときの処理"""
        # 現在選択されているモデルタイプを取得
        selected_model_type = self.yolo_model_combo.currentText()
        self.statusBar().showMessage(f"YOLOモデルタイプを「{selected_model_type}」に変更しました。モデルリストを更新します...")
        
        # モデルリストを更新
        #self.refresh_yolo_model_list()
        self.refresh_yolo_unified_model_list()                

    # def export_bbox_yolo_format(self, output_folder, classes, progress=None):
    #     """バウンディングボックスをYOLO形式でエクスポート（セグメンテーションと同じ構造）"""
    #     import shutil
        
    #     # セグメンテーションと同じフォルダ構造を作成
    #     images_dir = os.path.join(output_folder, 'images')
    #     labels_dir = os.path.join(output_folder, 'labels')
    #     os.makedirs(images_dir, exist_ok=True)
    #     os.makedirs(labels_dir, exist_ok=True)
        
    #     # 削除されたインデックスを除外
    #     deleted_indexes = getattr(self, 'deleted_indexes', set())
        
    #     # アノテーションがある画像のリストを作成
    #     annotated_images = list(self.bbox_annotations.keys()) if hasattr(self, 'bbox_annotations') else []
        
    #     # 削除されたインデックスに対応する画像を除外
    #     if hasattr(self, 'images') and deleted_indexes:
    #         excluded_images = {self.images[i] for i in deleted_indexes if i < len(self.images)}
    #         annotated_images = [img for img in annotated_images if img not in excluded_images]
        
    #     total_images = len(annotated_images)
        
    #     for i, img_path in enumerate(annotated_images):
    #         if progress and progress.wasCanceled():
    #             break
            
    #         if progress:
    #             progress_value = int((i / total_images) * 100) if total_images > 0 else 100
    #             progress.setValue(progress_value)
    #             progress.setLabelText(f"バウンディングボックスエクスポート中: {os.path.basename(img_path)}")
    #             QApplication.processEvents()
            
    #         # 画像をコピー
    #         img_filename = os.path.basename(img_path)
    #         try:
    #             shutil.copy2(img_path, os.path.join(images_dir, img_filename))
    #         except Exception as e:
    #             print(f"画像コピーエラー: {img_path} - {e}")
    #             continue
            
    #         # ラベルファイルを作成
    #         label_filename = os.path.splitext(img_filename)[0] + '.txt'
    #         label_path = os.path.join(labels_dir, label_filename)
            
    #         # 画像サイズを取得
    #         try:
    #             from PIL import Image
    #             img = Image.open(img_path)
    #             img_width, img_height = img.size
    #         except Exception as e:
    #             print(f"画像読み込みエラー: {img_path} - {e}")
    #             continue
            
    #         # バウンディングボックスのラベルを作成
    #         with open(label_path, 'w') as f:
    #             if img_path in self.bbox_annotations:
    #                 for bbox in self.bbox_annotations[img_path]:
    #                     class_name = bbox.get('class', 'unknown')
    #                     if class_name in classes:
    #                         class_id = classes.index(class_name)
                            
    #                         # YOLO形式に変換（正規化された座標）
    #                         center_x = ((bbox['x1'] + bbox['x2']) / 2) / img_width
    #                         center_y = ((bbox['y1'] + bbox['y2']) / 2) / img_height
    #                         width = (bbox['x2'] - bbox['x1']) / img_width
    #                         height = (bbox['y2'] - bbox['y1']) / img_height
                            
    #                         f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
    #     # classes.txtを作成
    #     classes_path = os.path.join(output_folder, 'classes.txt')
    #     with open(classes_path, 'w', encoding='utf-8') as f:
    #         for class_name in classes:
    #             f.write(f"{class_name}\n")
        
    #     # dataset.yamlを作成
    #     yaml_content = f"""path: {output_folder}
    # train: images
    # val: images
    # test: images

    # nc: {len(classes)}
    # names: {classes}

    # # バウンディングボックス検知用データセット
    # # フォーマット: class_id center_x center_y width height (全て正規化済み)
    # """
        
    #     yaml_path = os.path.join(output_folder, 'dataset.yaml')
    #     with open(yaml_path, 'w', encoding='utf-8') as f:
    #         f.write(yaml_content)
        
    #     return yaml_path
    
    def show_unified_export_dialog(self, export_type):
        """統一エクスポート設定ダイアログを表示
        
        Args:
            export_type: "donkey", "jetracer"
            
        Returns:
            設定辞書またはNone（キャンセル時）
        """
        # フォルダ名とタイトル
        type_config = {
            "donkey": {
                "folder_name": "data_donkey",
                "title": "Donkeycarエクスポート設定",
                "format_name": "Donkeycar"
            },
            "jetracer": {
                "folder_name": "data_jetracer", 
                "title": "Jetracerエクスポート設定",
                "format_name": "Jetracer"
            }
        }
        
        config = type_config.get(export_type, type_config["donkey"])
        default_folder_name = config["folder_name"]
        dialog_title = config["title"]
        format_name = config["format_name"]
        
        # 読み込み元フォルダがある場合はその名前を追加（重複を避ける）
        if hasattr(self, 'folder_path') and self.folder_path:
            parent_folder_name = os.path.basename(self.folder_path)
            if parent_folder_name and parent_folder_name not in default_folder_name:
                default_folder_name = f"{default_folder_name}_{parent_folder_name}"
        
        # ダイアログを作成
        dialog = QDialog(self)
        dialog.setWindowTitle(dialog_title)
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        
        # 保存先フォルダ選択
        folder_group = QGroupBox("保存先設定")
        folder_layout = QVBoxLayout(folder_group)
        
        # 保存先フォルダ選択
        folder_selection_layout = QHBoxLayout()
        folder_selection_layout.addWidget(QLabel("保存先フォルダ:"))
        
        folder_input = QLineEdit()
        # annotationフォルダ内に保存するように修正
        if hasattr(self, 'folder_path') and self.folder_path:
            # 親フォルダのannotationフォルダ内に保存
            annotation_base = os.path.join(self.folder_path, "annotation") 
        else:
            # デフォルトの場合
            annotation_base = annotation_folder
        
        default_output_path = os.path.join(annotation_base, default_folder_name)
        folder_input.setText(default_output_path)
        folder_selection_layout.addWidget(folder_input)
        
        browse_folder_button = QPushButton("参照...")
        browse_folder_button.clicked.connect(lambda: self.browse_output_folder(folder_input))
        folder_selection_layout.addWidget(browse_folder_button)
        
        folder_layout.addLayout(folder_selection_layout)
        layout.addWidget(folder_group)
                
        # 画像ソース選択（Donkeycarの場合のみ）
        selected_variants = []
        variant_keys = {}
        image_map = {}
        
        if export_type == "donkey" and hasattr(self, 'available_variants') and self.available_variants:
            # 画像ソース選択グループ
            source_group = QGroupBox("画像ソース選択")
            source_layout = QVBoxLayout(source_group)
            
            # 説明ラベル
            info_label = QLabel("エクスポートする画像ソースを選択してください（複数選択可）：")
            source_layout.addWidget(info_label)
            
            # 利用可能な画像ソースに基づいてチェックボックスを作成
            source_checks = {}
            for variant in self.available_variants:
                check = QCheckBox(f"{variant} ({len(self.variant_images.get(variant, []))}枚)")
                check.setProperty("variant", variant)
                # 現在のバリアントは自動的にチェック
                if variant == getattr(self, 'current_variant', None):
                    check.setChecked(True)
                source_layout.addWidget(check)
                source_checks[variant] = check
            
            layout.addWidget(source_group)
            
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
            key_note = QLabel("※ Donkeycarのデフォルトキーは 'cam/image_array' です。")
            key_note.setStyleSheet("color: #666; font-style: italic;")
            keys_layout.addWidget(key_note)
            
            layout.addWidget(keys_group)
        
        # 削除したインデックスの情報表示
        if hasattr(self, 'deleted_indexes') and self.deleted_indexes:
            deletion_info = QLabel(f"削除済みインデックス数: {len(self.deleted_indexes)}個（削除情報も併せてエクスポートされます）")
            deletion_info.setStyleSheet("color: #666; font-style: italic;")
            layout.addWidget(deletion_info)
        
        # ボタン
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # ダイアログ表示
        if not dialog.exec_():
            return None
        
        # 設定値を取得
        output_folder = folder_input.text().strip()
        if not output_folder:
            QMessageBox.warning(self, "警告", "保存先フォルダが指定されていません。")
            return None
                
        # Donkeycarの場合は画像ソース設定を取得
        if export_type == "donkey" and hasattr(self, 'available_variants') and self.available_variants:
            # 選択された画像ソースを取得
            for variant, check in source_checks.items():
                if check.isChecked():
                    selected_variants.append(variant)
                    # 対応するキー名を取得
                    variant_keys[variant] = key_inputs[variant].text().strip()
                    if not variant_keys[variant]:
                        variant_keys[variant] = f"{variant}/image_array"
            
            if not selected_variants:
                QMessageBox.warning(self, "警告", "画像ソースが選択されていません。")
                return None
            
            # 画像マップを作成（actual_indexをキーとして使用）
            for variant in selected_variants:
                variant_images = self.variant_images.get(variant, [])
                if not variant_images:
                    continue

                for img_path in variant_images:
                    try:
                        # self.imagesリストからactual_indexを取得
                        if img_path in self.images:
                            actual_idx = self.images.index(img_path)

                            if actual_idx not in image_map:
                                image_map[actual_idx] = {}
                            image_map[actual_idx][variant] = img_path
                    except Exception as e:
                        print(f"画像マップ作成エラー ({img_path}): {e}")
        else:
            # 他のエクスポート形式の場合はデフォルト値を設定
            selected_variants = ["cam"]  # デフォルト
        
        # 確認メッセージ
        confirm_message = f"以下の設定で{format_name}形式でエクスポートします：\n\n"
        confirm_message += f"保存先: {output_folder}\n"
        
        if export_type == "donkey" and selected_variants:
            for variant in selected_variants:
                image_count = len(self.variant_images.get(variant, []))
                confirm_message += f"・画像ソース: {variant} ({image_count}枚)\n"
                if variant in variant_keys:
                    confirm_message += f"  キー名: {variant_keys[variant]}\n"
                
        if export_type == "donkey":
            confirm_message += f"\nアノテーション数: {len(self.annotations)}個"
        
        if hasattr(self, 'deleted_indexes') and self.deleted_indexes:
            confirm_message += f"\n削除済みインデックス数: {len(self.deleted_indexes)}個"
        
        reply = QMessageBox.question(
            self, f"{format_name}エクスポート確認", confirm_message + "\n\n続行しますか？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        
        if reply == QMessageBox.No:
            return None
        
        return {
            'output_folder': output_folder,
            'selected_variants': selected_variants,
            'variant_keys': variant_keys,
            'image_map': image_map,
            'export_type': export_type,
        }

    def export_to_yolo_unified(self):
        """統合YOLO形式でエクスポートする - バウンディングボックスとセグメンテーションを統合"""
        # アノテーション状況を確認（修正版）
        has_bbox = bool(getattr(self, 'bbox_annotations', {}))
        has_seg = bool(getattr(self, 'segmentation_annotations', {}))
                
        if not has_bbox and not has_seg:
            QMessageBox.information(self, "情報", "エクスポートするアノテーションがありません。")
            return

        # 統合YOLOエクスポートダイアログを表示
        export_config = self.show_yolo_unified_export_dialog(has_bbox, has_seg)
        if not export_config:
            return  # キャンセルされた場合
        
        try:
            # エクスポート実行
            self.execute_yolo_unified_export(export_config)
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "エラー", 
                f"YOLO統合エクスポート中にエラーが発生しました: {str(e)}"
            )

    def show_yolo_unified_export_dialog(self, has_bbox, has_seg):
        """統合YOLOエクスポートダイアログを表示
        
        Args:
            has_bbox: バウンディングボックスアノテーションの有無
            has_seg: セグメンテーションアノテーションの有無
            
        Returns:
            設定辞書またはNone（キャンセル時）
        """
        # ダイアログを作成
        dialog = QDialog(self)
        dialog.setWindowTitle("YOLO統合エクスポート設定")
        dialog.setMinimumWidth(550)
        dialog.setMinimumHeight(400)
        
        layout = QVBoxLayout(dialog)
        
        # タイトル情報
        title_label = QLabel("YOLOアノテーションエクスポート")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # アノテーション状況の表示
        status_group = QGroupBox("アノテーション状況")
        status_layout = QVBoxLayout(status_group)
        
        # バウンディングボックス状況
        if has_bbox:
            bbox_count = sum(len(bboxes) for bboxes in self.bbox_annotations.values())
            bbox_images = len(self.bbox_annotations)
            bbox_status = QLabel(f"✓ バウンディングボックス: {bbox_count}個 ({bbox_images}枚の画像)")
            bbox_status.setStyleSheet("color: #2E7D32; font-weight: bold;")
        else:
            bbox_status = QLabel("✗ バウンディングボックス: なし")
            bbox_status.setStyleSheet("color: #D32F2F;")
        status_layout.addWidget(bbox_status)
        
        # セグメンテーション状況
        if has_seg:
            seg_count = sum(len(segs) for segs in self.segmentation_annotations.values())
            seg_images = len(self.segmentation_annotations)
            seg_status = QLabel(f"✓ セグメンテーション: {seg_count}個 ({seg_images}枚の画像)")
            seg_status.setStyleSheet("color: #2E7D32; font-weight: bold;")
        else:
            seg_status = QLabel("✗ セグメンテーション: なし")
            seg_status.setStyleSheet("color: #D32F2F;")
        status_layout.addWidget(seg_status)
        
        layout.addWidget(status_group)
        
        # エクスポート形式選択
        export_group = QGroupBox("エクスポート形式選択")
        export_layout = QVBoxLayout(export_group)
        
        # バウンディングボックスエクスポートチェックボックス
        bbox_check = QCheckBox("バウンディングボックス (物体検知用)")
        bbox_check.setChecked(has_bbox)  # アノテーションがある場合は自動でチェック
        bbox_check.setEnabled(has_bbox)  # アノテーションがない場合は無効
        if has_bbox:
            bbox_count = sum(len(bboxes) for bboxes in self.bbox_annotations.values())
            bbox_check.setToolTip(f"{bbox_count}個のバウンディングボックスをエクスポートします")
        else:
            bbox_check.setToolTip("バウンディングボックスアノテーションがありません")
        export_layout.addWidget(bbox_check)
        
        # セグメンテーションエクスポートチェックボックス
        seg_check = QCheckBox("セグメンテーション (インスタンスセグメンテーション用)")
        seg_check.setChecked(has_seg)  # アノテーションがある場合は自動でチェック
        seg_check.setEnabled(has_seg)  # アノテーションがない場合は無効
        if has_seg:
            seg_count = sum(len(segs) for segs in self.segmentation_annotations.values())
            seg_check.setToolTip(f"{seg_count}個のセグメンテーションをエクスポートします")
        else:
            seg_check.setToolTip("セグメンテーションアノテーションがありません")
        export_layout.addWidget(seg_check)
        
        # 統合エクスポートオプション
        if has_bbox and has_seg:
            unified_check = QCheckBox("統合形式 (1つのデータセットに両方を含める)")
            unified_check.setChecked(False)  # デフォルトは個別エクスポート
            unified_check.setToolTip("バウンディングボックスとセグメンテーションを1つのYOLOデータセットに統合します")
            export_layout.addWidget(unified_check)
        else:
            unified_check = QCheckBox("統合形式 (利用不可)")
            unified_check.setChecked(False)
            unified_check.setEnabled(False)
            unified_check.setToolTip("両方のアノテーション形式が必要です")
            export_layout.addWidget(unified_check)
        
        layout.addWidget(export_group)
        
        # クラス設定
        class_group = QGroupBox("クラス設定")
        class_layout = QVBoxLayout(class_group)
        
        classes_layout = QHBoxLayout()
        classes_layout.addWidget(QLabel("検知クラス:"))
        classes_input = QLineEdit("car,red_sign,green_sign,dog")
        classes_input.setPlaceholderText("カンマ区切りでクラス名を入力")
        classes_layout.addWidget(classes_input)
        class_layout.addLayout(classes_layout)
        
        layout.addWidget(class_group)
        
        # 保存先設定
        folder_group = QGroupBox("保存先設定")
        folder_layout = QVBoxLayout(folder_group)
        
        # 自動保存先の生成
        if hasattr(self, 'folder_path') and self.folder_path:
            annotation_base = os.path.join(self.folder_path, "annotation")
        else:
            annotation_base = annotation_folder
        
        default_output_path = os.path.join(annotation_base, "data_yolo")
        
        folder_selection_layout = QHBoxLayout()
        folder_selection_layout.addWidget(QLabel("保存先フォルダ:"))
        
        folder_input = QLineEdit()
        folder_input.setText(default_output_path)
        folder_selection_layout.addWidget(folder_input)
        
        browse_folder_button = QPushButton("参照...")
        browse_folder_button.clicked.connect(lambda: self.browse_output_folder(folder_input))
        folder_selection_layout.addWidget(browse_folder_button)
        
        folder_layout.addLayout(folder_selection_layout)
        layout.addWidget(folder_group)
        
        # 削除したインデックスの情報表示
        if hasattr(self, 'deleted_indexes') and self.deleted_indexes:
            deletion_info = QLabel(f"削除済みインデックス数: {len(self.deleted_indexes)}個（エクスポートから除外されます）")
            deletion_info.setStyleSheet("color: #666; font-style: italic; margin-top: 10px;")
            layout.addWidget(deletion_info)
        
        # ボタン
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # ダイアログ表示
        if not dialog.exec_():
            return None
        
        # 設定値を取得
        output_folder = folder_input.text().strip()
        if not output_folder:
            QMessageBox.warning(self, "警告", "保存先フォルダが指定されていません。")
            return None
        
        # エクスポート形式の確認
        export_bbox = bbox_check.isChecked()
        export_seg = seg_check.isChecked()
        export_unified = unified_check.isChecked() if has_bbox and has_seg else False
        
        if not export_bbox and not export_seg:
            QMessageBox.warning(self, "警告", "エクスポートする形式が選択されていません。")
            return None
        
        # クラス設定の取得
        classes = [cls.strip() for cls in classes_input.text().split(',') if cls.strip()]
        if not classes:
            QMessageBox.warning(self, "警告", "クラスが設定されていません。")
            return None
        
        # 確認メッセージの生成
        confirm_message = "以下の設定でYOLO形式でエクスポートします：\n\n"
        confirm_message += f"保存先: {output_folder}\n"
        confirm_message += f"クラス: {', '.join(classes)}\n\n"
        
        export_items = []
        if export_bbox:
            bbox_count = sum(len(bboxes) for bboxes in self.bbox_annotations.values())
            export_items.append(f"バウンディングボックス: {bbox_count}個")
        if export_seg:
            seg_count = sum(len(segs) for segs in self.segmentation_annotations.values())
            export_items.append(f"セグメンテーション: {seg_count}個")
        
        confirm_message += "エクスポート内容:\n"
        for item in export_items:
            confirm_message += f"・{item}\n"
        
        if export_unified:
            confirm_message += "\n※ 統合形式で1つのデータセットに保存されます"
        else:
            confirm_message += "\n※ 各形式別々のデータセットとして保存されます"
        
        if hasattr(self, 'deleted_indexes') and self.deleted_indexes:
            confirm_message += f"\n\n削除済みインデックス: {len(self.deleted_indexes)}個（除外）"
        
        reply = QMessageBox.question(
            self, "YOLO統合エクスポート確認", confirm_message + "\n\n続行しますか？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        
        if reply == QMessageBox.No:
            return None
        
        return {
            'output_folder': output_folder,
            'classes': classes,
            'export_bbox': export_bbox,
            'export_seg': export_seg,
            'export_unified': export_unified,
            'has_bbox': has_bbox,
            'has_seg': has_seg
        }

    def execute_yolo_unified_export(self, config):
        """統合YOLOエクスポートを実行"""
        output_folder = config['output_folder']
        classes = config['classes']
        export_bbox = config['export_bbox']
        export_seg = config['export_seg']
        export_unified = config['export_unified']
        
        # プログレスダイアログを表示
        total_steps = (1 if export_bbox else 0) + (1 if export_seg else 0) + (1 if export_unified else 0)
        progress = QProgressDialog("YOLOエクスポート準備中...", "キャンセル", 0, total_steps * 100, self)
        progress.setWindowTitle("エクスポート実行中")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            current_step = 0
            results = []
            
            if export_unified:
                # 統合エクスポート
                progress.setLabelText("統合YOLO形式でエクスポート中...")
                progress.setValue(current_step * 100)
                
                unified_folder = os.path.join(output_folder, "unified")
                yaml_path = self.export_unified_yolo_format(unified_folder, classes, progress)
                
                if yaml_path:
                    results.append(f"統合形式: {unified_folder}")
                
                current_step += 1
            
            else:
                # 個別エクスポート（統一されたフォルダ構造）
                if export_bbox:
                    progress.setLabelText("バウンディングボックス形式でエクスポート中...")
                    progress.setValue(current_step * 100)
                    
                    bbox_folder = os.path.join(output_folder, "detection")
                    os.makedirs(bbox_folder, exist_ok=True)
                    
                    # インデックスベースのバウンディングボックスエクスポート
                    yaml_path = self.export_bbox_yolo_format_index_based(bbox_folder, classes, progress)
                    
                    if yaml_path:
                        results.append(f"物体検知: {bbox_folder}")
                    
                    current_step += 1
                    progress.setValue(current_step * 100)
                
                if export_seg:
                    progress.setLabelText("セグメンテーション形式でエクスポート中...")
                    progress.setValue(current_step * 100)
                    
                    seg_folder = os.path.join(output_folder, "segmentation")
                    os.makedirs(seg_folder, exist_ok=True)
                    
                    # インデックスベースのセグメンテーションエクスポート
                    yaml_path = self.export_segmentation_yolo_format_index_based(seg_folder, classes, progress)
                    
                    if yaml_path:
                        results.append(f"セグメンテーション: {seg_folder}")
                    
                    current_step += 1
                    progress.setValue(current_step * 100)
            
            progress.close()
            
            # 結果表示
            if results:
                # 統計情報の計算
                bbox_count = sum(len(bboxes) for bboxes in self.bbox_annotations.values()) if export_bbox and hasattr(self, 'bbox_annotations') else 0
                seg_count = sum(len(segs) for segs in self.segmentation_annotations.values()) if export_seg and hasattr(self, 'segmentation_annotations') else 0
                
                result_message = "YOLOエクスポートが完了しました。\n\n"
                result_message += "保存先:\n"
                for result in results:
                    result_message += f"・{result}\n"
                
                result_message += f"\nエクスポート統計:\n"
                if export_bbox:
                    result_message += f"・バウンディングボックス: {bbox_count}個\n"
                if export_seg:
                    result_message += f"・セグメンテーション: {seg_count}個\n"
                result_message += f"・クラス: {', '.join(classes)}"
                
                QMessageBox.information(self, "エクスポート完了", result_message)
            else:
                QMessageBox.warning(self, "警告", "エクスポートに失敗しました。")
        
        except Exception as e:
            progress.close()
            raise e

    def export_bbox_yolo_format_index_based(self, output_folder, classes, progress=None):
        """インデックスベースでバウンディングボックスをYOLO形式でエクスポート"""
        import shutil
        
        # フォルダ構造を作成
        images_dir = os.path.join(output_folder, 'images')
        labels_dir = os.path.join(output_folder, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # 削除されたインデックスを除外
        deleted_indexes = getattr(self, 'deleted_indexes', set())
        
        # アノテーションがあるインデックスのリストを作成
        annotated_indexes = list(self.bbox_annotations.keys()) if hasattr(self, 'bbox_annotations') else []
        
        # 削除されたインデックスを除外
        valid_indexes = [idx for idx in annotated_indexes if idx not in deleted_indexes]
        
        total_images = len(valid_indexes)
        
        for i, idx in enumerate(valid_indexes):
            if progress and progress.wasCanceled():
                break
            
            if progress:
                progress_value = int((i / total_images) * 100) if total_images > 0 else 100
                progress.setValue(progress_value)
                
            # インデックスから画像パスを取得
            if not hasattr(self, 'images') or idx >= len(self.images):
                continue
                
            img_path = self.images[idx]
            img_filename = os.path.basename(img_path)
            
            if progress:
                progress.setLabelText(f"バウンディングボックスエクスポート中: {img_filename}")
                QApplication.processEvents()
            
            # 画像をコピー
            try:
                shutil.copy2(img_path, os.path.join(images_dir, img_filename))
            except Exception as e:
                print(f"画像コピーエラー: {img_path} - {e}")
                continue
            
            # ラベルファイルを作成
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)
            
            # 画像サイズを取得
            try:
                from PIL import Image
                img = Image.open(img_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"画像読み込みエラー: {img_path} - {e}")
                continue
            
            # バウンディングボックスのラベルを作成
            with open(label_path, 'w') as f:
                if idx in self.bbox_annotations:
                    for bbox in self.bbox_annotations[idx]:
                        class_name = bbox.get('class', 'unknown')
                        if class_name in classes:
                            class_id = classes.index(class_name)
                            
                            # YOLO形式に変換
                            center_x = ((bbox['x1'] + bbox['x2']) / 2) 
                            center_y = ((bbox['y1'] + bbox['y2']) / 2) 
                            width = (bbox['x2'] - bbox['x1']) 
                            height = (bbox['y2'] - bbox['y1']) 
                            
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # classes.txtを作成
        classes_path = os.path.join(output_folder, 'classes.txt')
        with open(classes_path, 'w', encoding='utf-8') as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
        
        # dataset.yamlを作成
        yaml_content = f"""path: {output_folder}
    train: images
    val: images
    test: images

    nc: {len(classes)}
    names: {classes}

    # バウンディングボックス検知用データセット
    # フォーマット: class_id center_x center_y width height (全て正規化済み)
    """
        
        yaml_path = os.path.join(output_folder, 'dataset.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        return yaml_path

    def export_segmentation_yolo_format_index_based(self, output_folder, classes, progress=None):
        """インデックスベースでセグメンテーションをYOLO形式でエクスポート"""
        import shutil
        
        # フォルダ構造を作成
        images_dir = os.path.join(output_folder, 'images')
        labels_dir = os.path.join(output_folder, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # 削除されたインデックスを除外
        deleted_indexes = getattr(self, 'deleted_indexes', set())
        
        # アノテーションがあるインデックスのリストを作成
        annotated_indexes = list(self.segmentation_annotations.keys()) if hasattr(self, 'segmentation_annotations') else []
        
        # 削除されたインデックスを除外
        valid_indexes = [idx for idx in annotated_indexes if idx not in deleted_indexes]
        
        total_images = len(valid_indexes)
        
        for i, idx in enumerate(valid_indexes):
            if progress and progress.wasCanceled():
                break
            
            if progress:
                progress_value = int((i / total_images) * 100) if total_images > 0 else 100
                progress.setValue(progress_value)
                
            # インデックスから画像パスを取得
            if not hasattr(self, 'images') or idx >= len(self.images):
                continue
                
            img_path = self.images[idx]
            img_filename = os.path.basename(img_path)
            
            if progress:
                progress.setLabelText(f"セグメンテーションエクスポート中: {img_filename}")
                QApplication.processEvents()
            
            # 画像をコピー
            try:
                shutil.copy2(img_path, os.path.join(images_dir, img_filename))
            except Exception as e:
                print(f"画像コピーエラー: {img_path} - {e}")
                continue
            
            # ラベルファイルを作成
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)
            
            # 画像サイズを取得
            try:
                from PIL import Image
                img = Image.open(img_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"画像読み込みエラー: {img_path} - {e}")
                continue
            
            # セグメンテーションのラベルを作成
            with open(label_path, 'w') as f:
                if idx in self.segmentation_annotations:
                    for seg in self.segmentation_annotations[idx]:
                        class_name = seg.get('class', 'unknown')
                        if class_name in classes:
                            class_id = classes.index(class_name)
                            
                            # ポリゴンポイントを正規化
                            normalized_points = []
                            points = seg.get('points', [])
                            
                            for point in points:
                                if isinstance(point, (list, tuple)) and len(point) >= 2:
                                    norm_x = point[0] / img_width
                                    norm_y = point[1] / img_height
                                    normalized_points.extend([norm_x, norm_y])
                            
                            if normalized_points:
                                # YOLO形式でポリゴンを出力
                                points_str = ' '.join(f"{p:.6f}" for p in normalized_points)
                                f.write(f"{class_id} {points_str}\n")
        
        # classes.txtを作成
        classes_path = os.path.join(output_folder, 'classes.txt')
        with open(classes_path, 'w', encoding='utf-8') as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
        
        # dataset.yamlを作成
        yaml_content = f"""path: {output_folder}
    train: images
    val: images
    test: images

    nc: {len(classes)}
    names: {classes}

    # セグメンテーション用データセット
    # フォーマット: class_id x1 y1 x2 y2 ... xn yn (全て正規化済み)
    """
        
        yaml_path = os.path.join(output_folder, 'dataset.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        return yaml_path


    ### TODO:統合
    # def export_all_to_yolo(self):
    #     """物体検知とセグメンテーションを統合してYOLO形式でエクスポートする"""
    #     has_bbox = hasattr(self, 'bbox_annotations') and self.bbox_annotations
    #     has_seg = hasattr(self, 'segmentation_annotations') and self.segmentation_annotations
        
    #     if not has_bbox and not has_seg:
    #         QMessageBox.information(self, "情報", "エクスポートするアノテーションがありません。")
    #         return
        
    #     # アノテーションフォルダを作成
    #     annotation_folder = os.path.join(self.folder_path, ANNOTATION_DIR_NAME)
    #     yolo_export_folder = os.path.join(annotation_folder, "yolo_unified")
    #     os.makedirs(yolo_export_folder, exist_ok=True)
        
    #     try:
    #         # プログレスダイアログを表示
    #         total_images = len(set(list(self.bbox_annotations.keys() if has_bbox else []) + 
    #                             list(self.segmentation_annotations.keys() if has_seg else [])))
    #         progress = QProgressDialog("YOLO統合データをエクスポート中...", "キャンセル", 0, total_images, self)
    #         progress.setWindowTitle("エクスポート")
    #         progress.setWindowModality(Qt.WindowModal)
    #         progress.show()
            
    #         # クラス名を取得
    #         classes = [cls.strip() for cls in self.classes_input.text().split(',') if cls.strip()]
            
    #         # エクスポート実行
    #         yaml_path = self.export_unified_yolo_format(yolo_export_folder, classes, progress)
            
    #         progress.setValue(total_images)
    #         progress.close()
            
    #         # 統計情報を計算
    #         bbox_count = sum(len(bboxes) for bboxes in self.bbox_annotations.values()) if has_bbox else 0
    #         seg_count = sum(len(segs) for segs in self.segmentation_annotations.values()) if has_seg else 0
            
    #         QMessageBox.information(
    #             self, 
    #             "エクスポート完了", 
    #             f"アノテーションをYOLO統合形式でエクスポートしました。\n"
    #             f"保存先: {yolo_export_folder}\n"
    #             f"処理画像数: {total_images}\n"
    #             f"バウンディングボックス数: {bbox_count}\n"
    #             f"セグメンテーション数: {seg_count}\n"
    #             f"クラス: {', '.join(classes)}"
    #         )
            
    #     except Exception as e:
    #         QMessageBox.critical(
    #             self, 
    #             "エラー", 
    #             f"YOLO統合エクスポート中にエラーが発生しました: {str(e)}"
    #         )

    def export_unified_yolo_format(self, output_folder, classes, progress=None):
        """物体検知とセグメンテーションを統合してYOLO形式でエクスポート"""
        import shutil
        
        # フォルダ構造を作成
        images_dir = os.path.join(output_folder, 'images')
        labels_dir = os.path.join(output_folder, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # 全ての画像を取得
        all_images = set()
        if hasattr(self, 'bbox_annotations'):
            all_images.update(self.bbox_annotations.keys())
        if hasattr(self, 'segmentation_annotations'):
            all_images.update(self.segmentation_annotations.keys())
        
        all_images = list(all_images)
        
        for i, img_path in enumerate(all_images):
            if progress and progress.wasCanceled():
                break
            
            if progress:
                progress.setValue(i)
                progress.setLabelText(f"処理中: {os.path.basename(img_path)}")
                QApplication.processEvents()
            
            # 画像をコピー
            img_filename = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(images_dir, img_filename))
            
            # ラベルファイルを作成
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)
            
            # 画像サイズを取得
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            with open(label_path, 'w') as f:
                # バウンディングボックスの処理
                if hasattr(self, 'bbox_annotations') and img_path in self.bbox_annotations:
                    for bbox in self.bbox_annotations[img_path]:
                        class_name = bbox.get('class', 'unknown')
                        if class_name in classes:
                            class_id = classes.index(class_name)
                            
                            # YOLO形式に変換 (中心x, 中心y, 幅, 高さ)
                            center_x = (bbox['x1'] + bbox['x2']) / 2
                            center_y = (bbox['y1'] + bbox['y2']) / 2
                            width = bbox['x2'] - bbox['x1']
                            height = bbox['y2'] - bbox['y1']
                            
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                # セグメンテーションの処理
                if hasattr(self, 'segmentation_annotations') and img_path in self.segmentation_annotations:
                    for seg in self.segmentation_annotations[img_path]:
                        class_name = seg.get('class', 'unknown')
                        points = seg.get('points', [])
                        
                        if class_name in classes and len(points) >= 3:
                            class_id = classes.index(class_name)
                            
                            # ポリゴンの座標を正規化
                            normalized_points = []
                            for x, y in points:
                                norm_x = x / img_width
                                norm_y = y / img_height
                                normalized_points.extend([norm_x, norm_y])
                            
                            # YOLO セグメンテーション形式
                            line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_points)
                            f.write(line + '\n')
        
        # classes.txtを作成
        classes_path = os.path.join(output_folder, 'classes.txt')
        with open(classes_path, 'w') as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
        
        # dataset.yamlを作成
        yaml_content = f"""path: {output_folder}
    train: images
    val: images
    test: images

    nc: {len(classes)}
    names: {classes}

    # 物体検知とセグメンテーションの統合データセット
    # バウンディングボックス: class_id center_x center_y width height
    # セグメンテーション: class_id x1 y1 x2 y2 x3 y3 ...
    """
        
        yaml_path = os.path.join(output_folder, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        return yaml_path

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
        
        # モデルファイルをソート - ファイル作成日時順、新しいものが上
        # カスタムサフィックスが追加された場合でも正しくソートされるよう、mtimeを使用
        def sort_key(model_info):
            # ファイルの完全なパスを取得
            if model_info['parent'] == 'root':
                file_path = os.path.join(models_dir, model_info['path'])
            else:
                file_path = os.path.join(models_dir, model_info['path'])

            # ファイルの作成日時を取得（存在しない場合は0）
            try:
                mtime = os.path.getmtime(file_path)
            except:
                mtime = 0

            # 直下のファイルを優先し、その後新しいものが上に来るように負の値を返す
            priority = 0 if model_info['parent'] == 'root' else 1
            return (priority, -mtime)

        yolo_model_files.sort(key=sort_key)
        
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
        if not model_type or not (model_type.startswith("yolov8") or model_type.startswith("yolo11")):
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
            #self.refresh_yolo_model_list()
            self.refresh_yolo_unified_model_list()                
            
            self.statusBar().showMessage(f"事前学習済み {model_type} モデルをmodelsフォルダに保存しました: {save_path}", 3000)
            return save_path
        
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"事前学習済みモデル {model_type} のダウンロード中にエラーが発生しました: {str(e)}\n\n"
                f"モデル名 '{model_type}' がUltralyticsでサポートされていない可能性があります。"
            )
            print(f"YOLO model download error for {model_type}: {e}")
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
        # models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
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
            
            # YOLOモデル読み込み完了後、YOLOオートアノテーションボタンを有効化
            if hasattr(self, 'yolo_auto_annotate_btn'):
                self.yolo_auto_annotate_btn.setEnabled(True)
            
            progress.close()
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "エラー",
                f"YOLOモデルの読み込み中にエラーが発生しました: {str(e)}"
            )

    # def load_yolo_annotations(self):
    #     """YOLO形式のアノテーションを読み込む"""
    #     if not self.images:
    #         QMessageBox.warning(self, "警告", "先に画像を読み込んでください。")
    #         return
        
    #     # YOLOアノテーションフォルダを選択
    #     yolo_dir = QFileDialog.getExistingDirectory(
    #         self, "YOLOアノテーションフォルダを選択", 
    #         self.folder_path,
    #         QFileDialog.ShowDirsOnly
    #     )
        
    #     if not yolo_dir:
    #         return
        
    #     # ラベルフォルダを確認
    #     labels_dir = os.path.join(yolo_dir, "labels")
    #     if not os.path.exists(labels_dir):
    #         # 直接選択されたフォルダがlabelsフォルダかもしれない
    #         if os.path.basename(yolo_dir) == "labels":
    #             labels_dir = yolo_dir
    #         else:
    #             # サブフォルダの中にlabelsディレクトリがあるか確認
    #             possible_labels_dir = [
    #                 os.path.join(yolo_dir, d, "labels") 
    #                 for d in os.listdir(yolo_dir) 
    #                 if os.path.isdir(os.path.join(yolo_dir, d))
    #             ]
    #             possible_labels_dir = [d for d in possible_labels_dir if os.path.exists(d)]
                
    #             if possible_labels_dir:
    #                 labels_dir = possible_labels_dir[0]
    #             else:
    #                 QMessageBox.warning(
    #                     self, "警告", 
    #                     "選択されたフォルダ内にlabelsディレクトリが見つかりません。"
    #                 )
    #                 return
        
    #     # クラス情報を読み込む
    #     classes_path = os.path.join(os.path.dirname(labels_dir), "classes.txt")
    #     classes = []
        
    #     if os.path.exists(classes_path):
    #         with open(classes_path, 'r') as f:
    #             classes = [line.strip() for line in f.readlines()]
    #     else:
    #         # クラス情報がない場合は選択してもらう
    #         text, ok = QInputDialog.getText(
    #             self, 
    #             "クラス情報", 
    #             "クラス名をカンマで区切って入力してください（例: car,red_sign,green_sign,dog）:",
    #             text=self.classes_input.text() if hasattr(self, 'classes_input') else "car,red_sign,green_sign,dog"
    #         )
            
    #         if ok and text:
    #             classes = [cls.strip() for cls in text.split(',') if cls.strip()]
    #         else:
    #             return
        
    #     # プログレスダイアログ
    #     progress = QProgressDialog("YOLOアノテーションを読み込み中...", "キャンセル", 0, len(self.images), self)
    #     progress.setWindowTitle("読み込み中")
    #     progress.setWindowModality(Qt.WindowModal)
    #     progress.show()
        
    #     # 既存のアノテーションがある場合は確認
    #     if hasattr(self, 'bbox_annotations') and self.bbox_annotations:
    #         reply = QMessageBox.question(
    #             self,
    #             "既存のアノテーション",
    #             "既存のバウンディングボックスアノテーションを上書きしますか？",
    #             QMessageBox.Yes | QMessageBox.No,
    #             QMessageBox.No
    #         )
            
    #         if reply == QMessageBox.Yes:
    #             self.bbox_annotations = {}
    #         else:
    #             # 既存のアノテーションに追加
    #             pass
    #     else:
    #         self.bbox_annotations = {}
        
    #     # 各画像のアノテーションを読み込む
    #     loaded_count = 0
        
    #     try:
    #         for i, img_path in enumerate(self.images):
    #             if progress.wasCanceled():
    #                 break
                
    #             progress.setValue(i)
                
    #             # 画像ファイル名からラベルファイル名を生成
    #             img_basename = os.path.splitext(os.path.basename(img_path))[0]
    #             label_path = os.path.join(labels_dir, f"{img_basename}.txt")
                
    #             # ラベルファイルが存在する場合のみ処理
    #             if os.path.exists(label_path):
    #                 # 画像サイズを取得（正規化された座標を元に戻すため）
    #                 img = Image.open(img_path)
    #                 img_width, img_height = img.size
                    
    #                 # ラベルファイルを読み込む
    #                 bboxes = []
                    
    #                 with open(label_path, 'r') as f:
    #                     for line in f:
    #                         parts = line.strip().split()
    #                         if len(parts) == 5:  # クラスID, x_center, y_center, width, height
    #                             class_id = int(parts[0])
    #                             x_center = float(parts[1])
    #                             y_center = float(parts[2])
    #                             width = float(parts[3])
    #                             height = float(parts[4])
                                
    #                             # YOLO形式（中心x,y,幅,高さ）から左上と右下の座標に変換
    #                             x1 = x_center - (width / 2)
    #                             y1 = y_center - (height / 2)
    #                             x2 = x_center + (width / 2)
    #                             y2 = y_center + (height / 2)
                                
    #                             # クラス名を取得
    #                             class_name = "unknown"
    #                             if 0 <= class_id < len(classes):
    #                                 class_name = classes[class_id]
                                
    #                             # バウンディングボックスを追加
    #                             bbox = {
    #                                 'x1': x1,
    #                                 'y1': y1,
    #                                 'x2': x2,
    #                                 'y2': y2,
    #                                 'class': class_name
    #                             }
                                
    #                             bboxes.append(bbox)
                    
    #                 # アノテーションを保存
    #                 if bboxes:
    #                     self.bbox_annotations[img_path] = bboxes
    #                     loaded_count += 1
            
    #         progress.close()
            
    #         # 統計情報を更新
    #         self.update_bbox_stats()
            
    #         # 表示を更新
    #         self.display_current_image()
    #         self.update_gallery()
            
    #         # 完了メッセージ
    #         QMessageBox.information(
    #             self,
    #             "読み込み完了",
    #             f"YOLOアノテーションを読み込みました。\n処理画像数: {loaded_count}/{len(self.images)}\nクラス: {', '.join(classes)}"
    #         )
        
    #     except Exception as e:
    #         progress.close()
    #         QMessageBox.critical(
    #             self,
    #             "エラー",
    #             f"YOLOアノテーションの読み込み中にエラーが発生しました: {str(e)}"
    #         )
    def load_yolo_annotations(self):
        """YOLO形式のアノテーションを読み込む - 表示スケール対応版"""
        if not self.images:
            QMessageBox.warning(self, "警告", "先に画像を読み込んでください。")
            return
        
        # zoom_factorを確認・設定
        if not hasattr(self, 'zoom_factor'):
            self.zoom_factor = 2.5  # デフォルト値
        
        # YOLOアノテーションフォルダを選択
        yolo_dir = QFileDialog.getExistingDirectory(
            self, "YOLOアノテーションフォルダを選択", 
            self.folder_path,
            QFileDialog.ShowDirsOnly
        )
        
        if not yolo_dir:
            return
        
        # ラベルフォルダを確認・検索
        labels_dir = self._find_labels_directory(yolo_dir)
        if not labels_dir:
            return
        
        # クラス情報を読み込む
        classes = self._load_yolo_classes(labels_dir)
        if not classes:
            return
        
        # 既存のアノテーションがある場合は確認
        if not self._confirm_annotation_overwrite():
            return
        
        # アノテーション読み込み実行
        self._execute_yolo_annotation_loading(labels_dir, classes)

    def _find_labels_directory(self, yolo_dir):
        """YOLOアノテーションのlabelsディレクトリを検索"""
        # 直接指定されたフォルダがlabelsフォルダかチェック
        if os.path.basename(yolo_dir) == "labels":
            return yolo_dir
        
        # 指定フォルダ内にlabelsディレクトリがあるかチェック
        labels_dir = os.path.join(yolo_dir, "labels")
        if os.path.exists(labels_dir):
            return labels_dir
        
        # サブフォルダの中にlabelsディレクトリがあるか確認
        try:
            for subdir in os.listdir(yolo_dir):
                subdir_path = os.path.join(yolo_dir, subdir)
                if os.path.isdir(subdir_path):
                    potential_labels_dir = os.path.join(subdir_path, "labels")
                    if os.path.exists(potential_labels_dir):
                        return potential_labels_dir
        except (OSError, PermissionError):
            pass
        
        QMessageBox.warning(
            self, "警告", 
            "選択されたフォルダ内にlabelsディレクトリが見つかりません。\n"
            "YOLOデータセットの構造:\n"
            "- dataset/\n"
            "  - images/\n"
            "  - labels/\n"
            "  - classes.txt"
        )
        return None

    def _load_yolo_classes(self, labels_dir):
        """YOLOクラス情報を読み込む"""
        classes = []
        
        # classes.txtファイルを探す
        possible_class_files = [
            os.path.join(os.path.dirname(labels_dir), "classes.txt"),
            os.path.join(labels_dir, "classes.txt"),
            os.path.join(os.path.dirname(os.path.dirname(labels_dir)), "classes.txt")
        ]
        
        for classes_path in possible_class_files:
            if os.path.exists(classes_path):
                try:
                    with open(classes_path, 'r', encoding='utf-8') as f:
                        classes = [line.strip() for line in f.readlines() if line.strip()]
                    break
                except (IOError, UnicodeDecodeError) as e:
                    print(f"クラスファイル読み込みエラー: {e}")
                    continue
        
        # クラス情報がない場合は手動入力
        if not classes:
            classes = self._get_classes_from_user()
        
        return classes

    def _get_classes_from_user(self):
        """ユーザーからクラス情報を取得"""
        default_classes = "car,red_sign,green_sign,dog"
        if hasattr(self, 'classes_input') and self.classes_input.text():
            default_classes = self.classes_input.text()
        
        text, ok = QInputDialog.getText(
            self,
            "クラス情報",
            "クラス名をカンマで区切って入力してください（例: car,red_sign,green_sign,dog）:",
            text=default_classes
        )
        
        if ok and text:
            return [cls.strip() for cls in text.split(',') if cls.strip()]
        return []

    def _confirm_annotation_overwrite(self):
        """既存のアノテーションがある場合の上書き確認"""
        has_bbox = hasattr(self, 'bbox_annotations') and self.bbox_annotations
        has_seg = hasattr(self, 'segmentation_annotations') and self.segmentation_annotations
        
        if has_bbox or has_seg:
            reply = QMessageBox.question(
                self,
                "既存のアノテーション",
                "既存のアノテーションを上書きしますか？\n"
                "「いいえ」を選択すると既存のアノテーションに追加されます。",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Cancel:
                return False
            elif reply == QMessageBox.Yes:
                # 既存のアノテーションをクリア
                self.bbox_annotations = {}
                self.segmentation_annotations = {}
                # 関連する変数もクリア
                self.last_bbox = None
                self.last_bboxes = []
                self.last_segmentation = None
                self.last_segmentations = []
            # reply == QMessageBox.No の場合は既存のアノテーションを保持
        else:
            # 初回読み込み時はアノテーション辞書を初期化
            if not hasattr(self, 'bbox_annotations'):
                self.bbox_annotations = {}
            if not hasattr(self, 'segmentation_annotations'):
                self.segmentation_annotations = {}
        
        return True

    def _execute_yolo_annotation_loading(self, labels_dir, classes):
        """YOLOアノテーション読み込みのメイン処理"""
        # プログレスダイアログ
        progress = QProgressDialog(
            "YOLOアノテーションを読み込み中...", 
            "キャンセル", 0, len(self.images), self
        )
        progress.setWindowTitle("読み込み中")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # 統計情報を記録
        loading_stats = {
            'total_images': len(self.images),
            'processed_images': 0,
            'images_with_annotations': 0,
            'total_bbox_annotations': 0,
            'total_seg_annotations': 0,
            'class_distribution': {cls: 0 for cls in classes},
            'errors': []
        }
        
        try:
            for i, img_path in enumerate(self.images):
                if progress.wasCanceled():
                    break
                
                progress.setValue(i)
                progress.setLabelText(f"処理中: {os.path.basename(img_path)}")
                QApplication.processEvents()
                
                # 画像ファイル名を基準にアノテーション読み込み
                bbox_annotations, seg_annotations = self._load_single_image_annotations(
                    img_path, i, labels_dir, classes, loading_stats
                )
                
                # バウンディングボックスアノテーションを保存（インデックスベース）
                if bbox_annotations:
                    if i not in self.bbox_annotations:
                        self.bbox_annotations[i] = []
                    self.bbox_annotations[i].extend(bbox_annotations)
                    loading_stats['images_with_annotations'] += 1
                
                # セグメンテーションアノテーションを保存（インデックスベース）
                if seg_annotations:
                    if i not in self.segmentation_annotations:
                        self.segmentation_annotations[i] = []
                    self.segmentation_annotations[i].extend(seg_annotations)
                
                loading_stats['processed_images'] += 1
            
            progress.close()
            
            # 結果の表示と更新
            self._finalize_yolo_annotation_loading(loading_stats, classes)
            
        except Exception as e:
            progress.close()
            loading_stats['errors'].append(f"予期しないエラー: {str(e)}")
            self._show_loading_error(loading_stats)

    def _load_single_image_annotations(self, img_path, img_index, labels_dir, classes, stats):
        """単一画像のYOLOアノテーションを読み込む（画像ファイル名ベースでマッチング）"""
        try:
            # 画像ファイル名からラベルファイル名を生成
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, f"{img_basename}.txt")
            
            # 対応するラベルファイルが存在しない場合は空のリストを返す
            if not os.path.exists(label_path):
                return [], []
            
            # 画像サイズを取得（正規化された座標を元に戻すため）
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                stats['errors'].append(f"画像サイズ取得エラー {img_basename}: {str(e)}")
                return [], []
            
            # ラベルファイルを読み込む
            bbox_annotations = []
            seg_annotations = []
            
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # アノテーション解析（バウンディングボックス or セグメンテーション）
                        bbox_annotation, seg_annotation = self._parse_yolo_annotation_line(
                            line, classes, img_width, img_height, 
                            img_basename, line_num, stats
                        )
                        
                        if bbox_annotation:
                            bbox_annotations.append(bbox_annotation)
                            stats['total_bbox_annotations'] += 1
                        
                        if seg_annotation:
                            seg_annotations.append(seg_annotation)
                            stats['total_seg_annotations'] += 1
                            
                    except Exception as e:
                        stats['errors'].append(
                            f"アノテーション解析エラー {img_basename}:{line_num}: {str(e)}"
                        )
            
            # デバッグ用ログ出力
            if bbox_annotations or seg_annotations:
                print(f"読み込み完了: {img_basename} -> インデックス{img_index} "
                    f"(bbox: {len(bbox_annotations)}, seg: {len(seg_annotations)})")
            
            return bbox_annotations, seg_annotations
            
        except Exception as e:
            stats['errors'].append(f"ファイル読み込みエラー {img_basename}: {str(e)}")
            return [], []

    def _parse_yolo_annotation_line(self, line, classes, img_width, img_height, 
                                img_basename, line_num, stats):
        """YOLOアノテーション行を解析（アプリの座標形式に合わせた変換）"""
        parts = line.split()
        
        if len(parts) < 5:
            stats['errors'].append(
                f"不正なアノテーション形式 {img_basename}:{line_num}: {line}"
            )
            return None, None
        
        try:
            class_id = int(parts[0])
            
            # クラス名を取得
            if 0 <= class_id < len(classes):
                class_name = classes[class_id]
                stats['class_distribution'][class_name] += 1
            else:
                class_name = f"unknown_class_{class_id}"
                stats['errors'].append(
                    f"不明なクラスID {img_basename}:{line_num}: {class_id}"
                )
            
            # バウンディングボックス形式の場合（5つの値）
            if len(parts) == 5:
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 正規化座標の範囲チェック
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                        0 <= width <= 1 and 0 <= height <= 1):
                    stats['errors'].append(
                        f"座標範囲エラー {img_basename}:{line_num}: 正規化座標が範囲外"
                    )
                    return None, None
                
                # YOLO形式（中心x,y,幅,高さ）から左上と右下の座標に変換
                x1_norm = max(0, min(1, x_center - (width / 2)))
                y1_norm = max(0, min(1, y_center - (height / 2)))
                x2_norm = max(0, min(1, x_center + (width / 2)))
                y2_norm = max(0, min(1, y_center + (height / 2)))
                
                # アプリ内のフォーマットに合わせた正規化座標のバウンディングボックス情報
                # マウスイベントと同じ形式（正規化座標）
                bbox = {
                    'x1': x1_norm,  # 0-1の範囲
                    'y1': y1_norm,  # 0-1の範囲
                    'x2': x2_norm,  # 0-1の範囲
                    'y2': y2_norm,  # 0-1の範囲
                    'class': class_name,
                    'confidence': 1.0,
                    'source': 'yolo_annotation'
                }
                
                print(bbox)
                return bbox, None
            
            # セグメンテーション形式の場合（6つ以上の値、座標ペア）
            elif len(parts) >= 6 and len(parts) % 2 == 1:  # class_id + 座標ペア
                coordinates = []
                
                for i in range(1, len(parts), 2):
                    x_norm = float(parts[i])
                    y_norm = float(parts[i + 1])
                    
                    # 正規化座標の範囲チェック
                    if not (0 <= x_norm <= 1 and 0 <= y_norm <= 1):
                        stats['errors'].append(
                            f"セグメンテーション座標範囲エラー {img_basename}:{line_num}"
                        )
                        return None, None
                    
                    # 実際の画像サイズにスケール（セグメンテーションはピクセル座標で保存）
                    x = int(x_norm * img_width)
                    y = int(y_norm * img_height)
                    coordinates.append((x, y))  # タプル形式で保存
                
                # アプリ内のフォーマットに合わせたセグメンテーション情報（ピクセル座標）
                segmentation = {
                    'points': coordinates,
                    'class': class_name,
                    'confidence': 1.0,
                    'source': 'yolo_annotation'
                }
                
                return None, segmentation
            
            else:
                stats['errors'].append(
                    f"不明なアノテーション形式 {img_basename}:{line_num}: {len(parts)}個の値"
                )
                return None, None
                
        except (ValueError, IndexError) as e:
            stats['errors'].append(
                f"データ変換エラー {img_basename}:{line_num}: {str(e)}"
            )
            return None, None

    def get_current_zoom_factor(self):
        """現在のズーム係数を取得"""
        # main_image_viewからzoom_factorを取得
        if hasattr(self, 'main_image_view') and hasattr(self.main_image_view, 'zoom_factor'):
            return self.main_image_view.zoom_factor
        
        # アプリ全体のzoom_factorを取得
        if hasattr(self, 'zoom_factor'):
            return self.zoom_factor
        
        # デフォルト値を返す
        return 2.5

    def update_yolo_annotations_with_zoom(self):
        """既存のYOLOアノテーションをズーム変更に合わせて更新"""
        if not hasattr(self, 'bbox_annotations') or not self.bbox_annotations:
            return
        
        current_zoom = self.get_current_zoom_factor()
        
        # 既存のアノテーションがYOLOソースの場合、ズーム変更に対応
        for index, annotations in self.bbox_annotations.items():
            for bbox in annotations:
                if bbox.get('source') == 'yolo_annotation':
                    # 元の正規化座標から再計算する場合は、
                    # 元の座標情報を保持する必要がある
                    # 今回は簡単のため、比率で調整
                    pass  # 実装は必要に応じて

    def _finalize_yolo_annotation_loading(self, stats, classes):
        """YOLOアノテーション読み込み完了後の処理"""
        # 統計情報を更新 - 全体の統計情報を更新
        if hasattr(self, 'update_driving_annotation_stats'):
            self.update_driving_annotation_stats()
        
        # 表示を更新
        if hasattr(self, 'display_current_image'):
            self.display_current_image()
        
        if hasattr(self, 'update_gallery'):
            self.update_gallery()
        
        # 画像ビューを更新
        if hasattr(self, 'main_image_view'):
            self.main_image_view.update()
        
        # クラス情報をGUIに反映
        if hasattr(self, 'classes_input') and classes:
            classes_str = ','.join(classes)
            self.classes_input.setText(classes_str)
            # クラス色を初期化して反映
            self._apply_class_changes(classes)
            print(f"[YOLOアノテーション読み込み] クラス情報を反映: {classes_str}")

        # 読み込み完了をログに記録
        print(f"YOLOアノテーション読み込み完了: ズーム係数={self.get_current_zoom_factor()}")
        
        # 成功メッセージを表示
        self._show_loading_success(stats, classes)

    def _show_loading_success(self, stats, classes):
        """読み込み成功メッセージを表示"""
        # クラス分布情報を作成
        class_info = []
        for cls, count in stats['class_distribution'].items():
            if count > 0:
                class_info.append(f"{cls}: {count}")
        
        class_summary = "\n".join(class_info) if class_info else "アノテーションが見つかりませんでした"
        
        # エラー情報
        error_summary = ""
        if stats['errors']:
            error_count = len(stats['errors'])
            error_summary = f"\n\n警告: {error_count}件のエラーが発生しました"
            if error_count <= 5:
                error_summary += ":\n" + "\n".join(stats['errors'][:5])
            else:
                error_summary += f":\n" + "\n".join(stats['errors'][:3]) + f"\n...他{error_count-3}件"
        
        # 完了メッセージ
        message = (
            f"YOLOアノテーションを読み込みました。\n\n"
            f"処理画像数: {stats['processed_images']}/{stats['total_images']}\n"
            f"アノテーション付き画像: {stats['images_with_annotations']}\n"
            f"バウンディングボックス: {stats['total_bbox_annotations']}\n"
            f"セグメンテーション: {stats['total_seg_annotations']}\n"
            f"クラス: {', '.join(classes)}\n\n"
            f"クラス別アノテーション数:\n{class_summary}"
            f"{error_summary}"
        )
        
        if stats['errors']:
            QMessageBox.warning(self, "読み込み完了（警告あり）", message)
        else:
            QMessageBox.information(self, "読み込み完了", message)

    def _show_loading_error(self, stats):
        """読み込みエラーメッセージを表示"""
        error_summary = "\n".join(stats['errors'][:10])  # 最初の10件のエラーを表示
        
        message = (
            f"YOLOアノテーションの読み込み中にエラーが発生しました。\n\n"
            f"処理済み画像: {stats['processed_images']}/{stats['total_images']}\n"
            f"読み込み済みバウンディングボックス: {stats['total_bbox_annotations']}\n"
            f"読み込み済みセグメンテーション: {stats['total_seg_annotations']}\n\n"
            f"エラー詳細:\n{error_summary}"
        )
        
        if len(stats['errors']) > 10:
            message += f"\n...他{len(stats['errors'])-10}件のエラー"
        
        QMessageBox.critical(self, "読み込みエラー", message)

    def get_current_classes(self):
        """現在設定されているクラス情報を取得"""
        if hasattr(self, 'classes_input') and self.classes_input.text():
            return [cls.strip() for cls in self.classes_input.text().split(',') if cls.strip()]
        
        # bbox_annotationsとsegmentation_annotationsからクラス情報を抽出
        classes = set()
        
        if hasattr(self, 'bbox_annotations') and self.bbox_annotations:
            for annotations in self.bbox_annotations.values():
                for annotation in annotations:
                    if 'class' in annotation:
                        classes.add(annotation['class'])
        
        if hasattr(self, 'segmentation_annotations') and self.segmentation_annotations:
            for annotations in self.segmentation_annotations.values():
                for annotation in annotations:
                    if 'class' in annotation:
                        classes.add(annotation['class'])
        
        return sorted(list(classes))

    def run_single_yolo_inference(self):
        """現在表示中の画像に対してYOLO推論を実行"""
        if not self.images or not hasattr(self, 'yolo_model'):
            return
        
        current_img_path = self.images[self.current_index]
        current_index = self.current_index
        
        try:
            # 推論実行
            results = self.yolo_model(current_img_path, conf=self.yolo_confidence_threshold)
            
            # 推論結果をクリア（現在の画像のみ）
            if current_index in self.detection_inference_results:
                del self.detection_inference_results[current_index]
            
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
            
            # 推論結果を保存（インデックスベース）
            if bboxes:
                self.detection_inference_results[current_index] = bboxes
            
            # 表示を更新
            self.main_image_view.update()
            
            # 情報パネル更新
            if hasattr(self, 'update_detection_inference_display'):
                self.update_detection_inference_display()
            
            return True
        
        except Exception as e:
            print(f"単一画像YOLO推論エラー: {e}")
            return False

    def run_single_yolo_segmentation_inference(self):
        """現在表示中の画像に対してYOLOセグメンテーション推論を実行"""
        if not self.images or not hasattr(self, 'yolo_seg_model'):
            return False
        
        current_img_path = self.images[self.current_index]
        
        try:
            # 推論実行
            results = self.yolo_seg_model(current_img_path, conf=self.yolo_seg_confidence_threshold)
            
            # 推論結果をクリア（現在の画像のみ）
            if current_img_path in self.segmentation_inference_results:
                del self.segmentation_inference_results[current_img_path]
            
            # セグメンテーション結果を保存
            segments = []
            bboxes = []
            masks = []  # マスク配列を保存
            class_ids = []  # クラスIDを保存

            # 画像サイズを取得
            img = Image.open(current_img_path)
            img_width, img_height = img.size

            for result in results:
                # バウンディングボックス処理
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for det in result.boxes.data.cpu().numpy():
                        if len(det) >= 6:
                            x1, y1, x2, y2, conf, class_id = det[:6]
                            
                            # クラス名を取得
                            class_name = result.names[int(class_id)]
                            
                            bboxes.append({
                                'class': class_name,
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(conf)
                            })
                
                # セグメンテーションマスク処理
                if hasattr(result, 'masks') and result.masks is not None:
                    import cv2
                    import numpy as np

                    for i, mask in enumerate(result.masks.data):
                        # マスクを numpy 配列に変換
                        mask_array = mask.cpu().numpy()

                        # マスクサイズを取得
                        mask_height, mask_width = mask_array.shape

                        # マスクを元画像サイズにリサイズ
                        mask_resized = cv2.resize(mask_array, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

                        # マスクから輪郭ポイントを抽出
                        mask_uint8 = (mask_array * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if contours:
                            # 最大の輪郭を取得
                            largest_contour = max(contours, key=cv2.contourArea)

                            # 輪郭を簡略化
                            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                            # ポイントリストに変換（正規化座標に変換）
                            points = []
                            for point in approx:
                                x, y = point[0]

                                # マスク座標を元画像座標にスケール
                                scaled_x = float(x) * img_width / mask_width
                                scaled_y = float(y) * img_height / mask_height

                                # スケール後の座標を0-1の正規化座標に変換
                                normalized_x = scaled_x / img_width
                                normalized_y = scaled_y / img_height

                                # 0-1の範囲内にクリップ
                                normalized_x = max(0.0, min(1.0, normalized_x))
                                normalized_y = max(0.0, min(1.0, normalized_y))

                                points.append([normalized_x, normalized_y])

                            if len(points) >= 3:
                                # 対応するクラス情報を取得
                                class_name = "unknown"
                                confidence = 0.0
                                class_id = 0
                                if i < len(bboxes):
                                    class_name = bboxes[i]['class']
                                    confidence = bboxes[i]['confidence']
                                    # クラス名からクラスIDを取得
                                    if hasattr(result, 'names'):
                                        for cid, cname in result.names.items():
                                            if cname == class_name:
                                                class_id = int(cid)
                                                break

                                segments.append({
                                    'class': class_name,
                                    'points': points,
                                    'confidence': confidence
                                })

                                # マスク配列とクラスIDを保存
                                masks.append(mask_resized)
                                class_ids.append(class_id)
            
            # 結果を保存（パスとインデックスの両方で保存）
            result_data = {
                'segments': segments,
                'bboxes': bboxes,
                'masks': masks,  # マスク配列を追加
                'classes': class_ids  # クラスIDを追加
            }
            self.segmentation_inference_results[current_img_path] = result_data
            # インデックスでもアクセスできるように保存
            self.segmentation_inference_results[self.current_index] = result_data
            
            print(f"セグメンテーション推論完了: {len(segments)}個のセグメント, {len(bboxes)}個のボックス")
            print(f"使用した信頼度閾値: {self.yolo_seg_confidence_threshold}")
            
            # 低信頼度でもテスト推論実行
            if len(segments) == 0:
                print("低信頼度（0.1）でも推論テスト...")
                low_conf_results = self.yolo_seg_model(current_img_path, conf=0.1)
                low_segments = 0
                low_boxes = 0
                for result in low_conf_results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        low_boxes = len(result.boxes.data)
                    if hasattr(result, 'masks') and result.masks is not None:
                        low_segments = len(result.masks.data)
                print(f"低信頼度結果: {low_segments}個のセグメント, {low_boxes}個のボックス")
            
            # セグメンテーション推論チェックボックスを自動的にON
            if hasattr(self, 'segmentation_inference_checkbox'):
                self.segmentation_inference_checkbox.setChecked(True)
            
            # 推論結果表示を更新
            self.update_segmentation_inference_display()
            # 画像表示も更新
            self.main_image_view.update()
            
            return True
            
        except Exception as e:
            print(f"YOLOセグメンテーション推論エラー: {e}")
            import traceback
            traceback.print_exc()
            self.segmentation_inference_results[current_img_path] = {'segments': [], 'bboxes': []}
            return False

    def test_model_comparison(self):
        """セグメンテーションモデルと物体検知モデルの比較テスト"""
        if not self.images:
            print("画像が読み込まれていません")
            return
            
        current_img_path = self.images[self.current_index]
        print(f"\n=== モデル比較テスト: {os.path.basename(current_img_path)} ===")
        
        # セグメンテーションモデルのテスト
        if hasattr(self, 'yolo_seg_model'):
            print("\n--- セグメンテーションモデル ---")
            for conf in [0.1, 0.3, 0.5, 0.6, 0.8]:
                try:
                    results = self.yolo_seg_model(current_img_path, conf=conf)
                    seg_count = 0
                    bbox_count = 0
                    for result in results:
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            bbox_count = len(result.boxes.data)
                        if hasattr(result, 'masks') and result.masks is not None:
                            seg_count = len(result.masks.data)
                    print(f"信頼度 {conf}: セグメント={seg_count}個, ボックス={bbox_count}個")
                except Exception as e:
                    print(f"信頼度 {conf}: エラー - {e}")
        else:
            print("セグメンテーションモデルが読み込まれていません")
        
        # 物体検知モデルのテスト
        if hasattr(self, 'yolo_model'):
            print("\n--- 物体検知モデル ---")
            for conf in [0.1, 0.3, 0.5, 0.6, 0.8]:
                try:
                    results = self.yolo_model(current_img_path, conf=conf)
                    bbox_count = 0
                    for result in results:
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            bbox_count = len(result.boxes.data)
                    print(f"信頼度 {conf}: ボックス={bbox_count}個")
                except Exception as e:
                    print(f"信頼度 {conf}: エラー - {e}")
        else:
            print("物体検知モデルが読み込まれていません")

    def convert_segmentation_to_bbox_dataset(self, seg_dataset_path, bbox_dataset_path):
        """セグメンテーション用YOLOデータセットからバウンディングボックス用データセットを生成"""
        import shutil
        
        try:
            # 出力ディレクトリの作成
            os.makedirs(bbox_dataset_path, exist_ok=True)
            
            # 画像ディレクトリのコピー
            seg_images_dir = os.path.join(seg_dataset_path, 'train', 'images')
            seg_val_images_dir = os.path.join(seg_dataset_path, 'val', 'images')
            
            bbox_images_dir = os.path.join(bbox_dataset_path, 'train', 'images')
            bbox_val_images_dir = os.path.join(bbox_dataset_path, 'val', 'images')
            bbox_labels_dir = os.path.join(bbox_dataset_path, 'train', 'labels')
            bbox_val_labels_dir = os.path.join(bbox_dataset_path, 'val', 'labels')
            
            # ディレクトリ作成
            for dir_path in [bbox_images_dir, bbox_val_images_dir, bbox_labels_dir, bbox_val_labels_dir]:
                os.makedirs(dir_path, exist_ok=True)
            
            # 画像をコピー
            if os.path.exists(seg_images_dir):
                shutil.copytree(seg_images_dir, bbox_images_dir, dirs_exist_ok=True)
            if os.path.exists(seg_val_images_dir):
                shutil.copytree(seg_val_images_dir, bbox_val_images_dir, dirs_exist_ok=True)
            
            # ラベル変換
            self._convert_seg_labels_to_bbox('train', seg_dataset_path, bbox_dataset_path)
            self._convert_seg_labels_to_bbox('val', seg_dataset_path, bbox_dataset_path)
            
            # dataset.yamlをコピー・修正
            seg_yaml = os.path.join(seg_dataset_path, 'dataset.yaml')
            bbox_yaml = os.path.join(bbox_dataset_path, 'dataset.yaml')
            
            if os.path.exists(seg_yaml):
                with open(seg_yaml, 'r', encoding='utf-8') as f:
                    yaml_content = f.read()
                # パスを更新
                yaml_content = yaml_content.replace(seg_dataset_path, bbox_dataset_path)
                # コメントを更新
                yaml_content = yaml_content.replace('セグメンテーション用', '物体検知用（セグメンテーションから変換）')
                
                with open(bbox_yaml, 'w', encoding='utf-8') as f:
                    f.write(yaml_content)
            
            print(f"セグメンテーションデータセットを物体検知用に変換完了: {bbox_dataset_path}")
            return bbox_dataset_path
            
        except Exception as e:
            print(f"データセット変換エラー: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _convert_seg_labels_to_bbox(self, split, seg_dataset_path, bbox_dataset_path):
        """セグメンテーションラベルをバウンディングボックスラベルに変換"""
        seg_labels_dir = os.path.join(seg_dataset_path, split, 'labels')
        bbox_labels_dir = os.path.join(bbox_dataset_path, split, 'labels')
        
        if not os.path.exists(seg_labels_dir):
            return
        
        converted_count = 0
        for label_file in os.listdir(seg_labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            seg_label_path = os.path.join(seg_labels_dir, label_file)
            bbox_label_path = os.path.join(bbox_labels_dir, label_file)
            
            try:
                with open(seg_label_path, 'r') as f:
                    lines = f.readlines()
                
                bbox_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 7:  # class + at least 3 points (x1 y1 x2 y2 x3 y3)
                        continue
                    
                    class_id = parts[0]
                    # 座標ポイントを取得
                    coords = [float(x) for x in parts[1:]]
                    
                    # ポイントをx,yのペアに変換
                    if len(coords) % 2 != 0:
                        continue  # 奇数個の座標は無効
                    
                    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    
                    if len(points) < 3:
                        continue  # 最低3点必要
                    
                    # バウンディングボックスを計算
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # YOLO形式のバウンディングボックス: center_x, center_y, width, height
                    center_x = (x_min + x_max) / 2.0
                    center_y = (y_min + y_max) / 2.0
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    bbox_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                    bbox_lines.append(bbox_line)
                
                # バウンディングボックスラベルファイルを保存
                with open(bbox_label_path, 'w') as f:
                    f.writelines(bbox_lines)
                
                converted_count += 1
                
            except Exception as e:
                print(f"ラベル変換エラー {label_file}: {e}")
        
        print(f"{split} セット: {converted_count}個のラベルファイルを変換しました")

    # 5. 情報パネルに物体検知推論結果を表示する処理の追加
    def update_detection_inference_display(self):
        """物体検知推論結果の表示を更新"""
        if not self.images:
            return
        
        current_index = self.current_index
        
        # 削除済みか推論表示OFFの場合は何も表示しない
        if not self.show_detection_inference:
            return
        
        # 物体検知推論結果がある場合は表示を更新（インデックスベース）
        if current_index in self.detection_inference_results:
            inference_bboxes = self.detection_inference_results[current_index]
            
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
                class_colors = DETECTION_INFERENCE_TEXT_COLORS
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
                self.detection_inference_info_label.setText(" ")  # スペースで高さを維持

    def update_segmentation_inference_display(self):
        """セグメンテーション推論結果の表示を更新（物体検知推論結果ラベルと同じ場所に表示）"""
        if not self.images or not hasattr(self, 'show_segmentation_inference'):
            return
        
        current_img_path = self.images[self.current_index]
        
        # セグメンテーション推論表示がOFFの場合は表示をクリア
        if not self.show_segmentation_inference:
            if hasattr(self, 'detection_inference_info_label'):
                self.detection_inference_info_label.setText(" ")  # スペースで高さを維持
            return
        
        # セグメンテーション推論結果がある場合は表示を更新
        if (current_img_path in self.segmentation_inference_results and 
            self.segmentation_inference_results[current_img_path]):
            
            result = self.segmentation_inference_results[current_img_path]
            segments = result.get('segments', [])
            
            # クラス別にカウント（セグメントのみカウント、バウンディングボックスは重複なので除外）
            all_objects = {}
            
            # セグメントをカウント（セグメンテーションではこれがメインのオブジェクト）
            for segment in segments:
                class_name = segment['class']
                all_objects[class_name] = all_objects.get(class_name, 0) + 1
            
            # HTML形式で表示テキストを作成（物体検知と同じ形式）
            inference_text = "<b>セグメンテーション推論結果:</b><br>"
            
            # 物体検知推論結果と同じ色定数を使用
            from config import DETECTION_INFERENCE_TEXT_COLORS
            
            # オブジェクトごとに表示
            total_count = 0
            for class_name, count in all_objects.items():
                total_count += count
                # クラス名に対応する色を取得（なければデフォルト色）
                color = DETECTION_INFERENCE_TEXT_COLORS.get(class_name, "#808080")
                inference_text += f"<span style='color: {color}; font-weight: bold;'>□ {class_name}</span>: {count}個<br>"
            
            # 合計を表示
            inference_text += f"合計: {total_count}個のオブジェクト<br>"
            
            # 物体検知推論結果と同じラベルに表示
            if hasattr(self, 'detection_inference_info_label'):
                self.detection_inference_info_label.setText(inference_text)
                self.detection_inference_info_label.setTextFormat(Qt.RichText)
        else:
            # 推論結果がない場合は表示をクリア
            if hasattr(self, 'detection_inference_info_label'):
                self.detection_inference_info_label.setText(" ")  # スペースで高さを維持

    ###TODO:統合
    def on_classes_changed(self, text):
        """クラス入力フィールドが変更された時の処理"""
        # テキスト変更時の処理は最小限に抑える
        pass

    def show_class_preset_dialog(self):
        """クラスプリセット選択ダイアログを表示"""
        dialog = QDialog(self)
        dialog.setWindowTitle("クラスプリセット選択")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        
        # タイトル
        title_label = QLabel("よく使われるクラスセットを選択してください:")
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)
        
        # プリセット定義
        presets = {
            "基本セット": "car,person,bicycle,motorcycle",
            "交通セット": "car,truck,bus,motorcycle,bicycle,person,traffic_light,stop_sign",
            "COCO基本": "person,bicycle,car,motorcycle,airplane,bus,train,truck",
            "自動運転基本": "car,person,bicycle,motorcycle,truck,bus,traffic_light,stop_sign,cone",
            "ミニカー用": "car,person,sign,cone,obstacle,barrier",
            "屋内ロボット": "person,chair,table,laptop,cell_phone,book,bottle,cup",
            "建設現場": "person,truck,excavator,cone,barrier,hard_hat,safety_vest",
            "カスタム": ""  # 空文字列でカスタム入力を示す
        }
        
        # プリセットボタンを作成
        preset_buttons = QButtonGroup(dialog)
        preset_buttons.setExclusive(True)
        
        for preset_name, preset_classes in presets.items():
            radio = QRadioButton(preset_name)
            radio.setProperty("preset_classes", preset_classes)
            
            # 説明を追加
            if preset_classes:
                description = f"({preset_classes})"
                radio.setToolTip(description)
            else:
                radio.setToolTip("現在の入力内容を保持")
            
            preset_buttons.addButton(radio)
            layout.addWidget(radio)
            
            # 現在の設定と一致するものがあれば選択
            current_classes = self.classes_input.text().strip()
            if preset_classes == current_classes:
                radio.setChecked(True)
        
        # カスタム入力フィールド
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("カスタム:"))
        custom_input = QLineEdit()
        custom_input.setPlaceholderText("カンマ区切りでクラス名を入力")
        custom_input.setText(self.classes_input.text())
        custom_layout.addWidget(custom_input)
        layout.addLayout(custom_layout)
        
        # ボタン
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # ダイアログを表示
        if dialog.exec_():
            # 選択されたプリセットを適用
            for button in preset_buttons.buttons():
                if button.isChecked():
                    preset_classes = button.property("preset_classes")
                    if preset_classes is None:  # カスタムが選択された場合
                        preset_classes = custom_input.text().strip()
                    if preset_classes:
                        self.classes_input.setText(preset_classes)
                    break

    def apply_classes(self):
        """クラス設定を確認してから反映"""
        text = self.classes_input.text().strip()
        if not text:
            QMessageBox.warning(self, "警告", "クラス名が入力されていません。")
            return
        
        classes = [cls.strip() for cls in text.split(',') if cls.strip()]
        
        if not classes:
            QMessageBox.warning(self, "警告", "有効なクラス名がありません。")
            return
        
        # 重複チェック
        unique_classes = list(set(classes))
        if len(unique_classes) != len(classes):
            duplicates = [cls for cls in classes if classes.count(cls) > 1]
            QMessageBox.warning(
                self, "警告", 
                f"重複するクラス名があります: {', '.join(set(duplicates))}"
            )
            return
        
        # 文字チェック
        invalid_chars = []
        for cls in classes:
            if not cls.replace('_', '').replace('-', '').isalnum():
                invalid_chars.append(cls)
        
        if invalid_chars:
            QMessageBox.warning(
                self, "警告",
                f"無効な文字を含むクラス名があります: {', '.join(invalid_chars)}\n"
                "英数字、アンダースコア(_)、ハイフン(-)のみ使用可能です。"
            )
            return
        
        # 確認メッセージ
        reply = QMessageBox.question(
            self, "クラス確認",
            f"以下のクラス設定を反映しますか？\n\n"
            f"クラス数: {len(classes)}\n"
            f"クラス: {', '.join(classes)}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # クラスを反映
            self._apply_class_changes(classes)
            QMessageBox.information(self, "完了", "クラス設定が反映されました。")
    
    def _apply_class_changes(self, classes):
        """クラス変更を実際に適用し、色を初期化"""
        # グローバル色設定を更新
        global CLASS_COLORS, SEGMENTATION_CLASS_COLORS, DETECTION_INFERENCE_TEXT_COLORS, DETECTION_INFERENCE_CLASS_COLORS
        
        # デフォルト色リスト
        default_colors = [
            (255, 0, 0, 180),    # 赤
            (0, 255, 0, 180),    # 緑
            (0, 0, 255, 180),    # 青
            (255, 255, 0, 180),  # 黄
            (255, 0, 255, 180),  # マゼンタ
            (0, 255, 255, 180),  # シアン
            (255, 128, 0, 180),  # オレンジ
            (128, 0, 255, 180),  # 紫
        ]
        
        default_seg_colors = [
            (255, 0, 0, 120),    # 赤
            (0, 255, 0, 120),    # 緑
            (0, 0, 255, 120),    # 青
            (255, 255, 0, 120),  # 黄
            (255, 0, 255, 120),  # マゼンタ
            (0, 255, 255, 120),  # シアン
            (255, 128, 0, 120),  # オレンジ
            (128, 0, 255, 120),  # 紫
        ]
        
        default_text_colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
            "#FF00FF", "#00FFFF", "#FF8000", "#8000FF"
        ]
        
        # 色辞書を初期化
        CLASS_COLORS = {}
        SEGMENTATION_CLASS_COLORS = {}
        DETECTION_INFERENCE_TEXT_COLORS = {}
        DETECTION_INFERENCE_CLASS_COLORS = {}
        
        # 各クラスに色を割り当て
        for i, class_name in enumerate(classes):
            color_idx = i % len(default_colors)
            CLASS_COLORS[class_name] = default_colors[color_idx]
            SEGMENTATION_CLASS_COLORS[class_name] = default_seg_colors[color_idx]
            DETECTION_INFERENCE_CLASS_COLORS[class_name] = default_seg_colors[color_idx]
            DETECTION_INFERENCE_TEXT_COLORS[class_name] = default_text_colors[color_idx]
        
        # 画面を再描画
        self.display_current_image()

    def get_current_classes(self):
        """現在設定されているクラスリストを取得"""
        text = self.classes_input.text().strip()
        if not text:
            return []
        return [cls.strip() for cls in text.split(',') if cls.strip()]

    def select_object_class(self):
            """物体クラスを選択するダイアログを表示 - 動的クラス対応"""
            classes = self.get_current_classes()
            
            if not classes:
                QMessageBox.warning(self, "警告", "検知クラスが設定されていません。\n先にクラス設定を行ってください。")
                return None
            
            # 前回選択したクラスのインデックスを取得
            default_index = 0
            if self.last_selected_bbox_class and self.last_selected_bbox_class in classes:
                default_index = classes.index(self.last_selected_bbox_class)
            
            class_name, ok = QInputDialog.getItem(
                self, 
                "クラス選択", 
                "オブジェクトのクラスを選択してください:",
                classes, 
                default_index,
                False
            )
            
            if ok and class_name:
                self.last_selected_bbox_class = class_name
                return class_name
            return None

    def show_display_settings(self):
        """ウィンドウサイズとフォントサイズの設定ダイアログを表示"""
        dialog = QDialog(self)
        dialog.setWindowTitle("表示設定")
        dialog.setMinimumWidth(450)
        dialog.setMinimumHeight(400)
        
        layout = QVBoxLayout(dialog)
        
        # ウィンドウサイズ設定
        window_group = QGroupBox("ウィンドウサイズ")
        window_layout = QVBoxLayout(window_group)
        
        # 現在のウィンドウサイズを表示
        current_size_label = QLabel(f"現在のサイズ: {self.width()} x {self.height()}")
        window_layout.addWidget(current_size_label)
        
        # 幅設定
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("幅:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(800, 3840)
        self.width_spin.setValue(self.width())
        self.width_spin.setSuffix(" px")
        width_layout.addWidget(self.width_spin)
        window_layout.addLayout(width_layout)
        
        # 高さ設定
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("高さ:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(600, 2160)
        self.height_spin.setValue(self.height())
        self.height_spin.setSuffix(" px")
        height_layout.addWidget(self.height_spin)
        window_layout.addLayout(height_layout)
        
        # プリセットボタン
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("プリセット:"))
        
        preset_buttons = [
            ("1280x720", 1280, 720),
            ("1600x900", 1600, 900),
            ("1920x1080", 1920, 1080),
            ("2560x1440", 2560, 1440)
        ]
        
        for text, w, h in preset_buttons:
            btn = QPushButton(text)
            # Lambda関数内でwidth, heightを正しくキャプチャするための修正
            btn.clicked.connect(lambda checked=False, width=w, height=h: self.set_window_size_preset(width, height))
            preset_layout.addWidget(btn)
        
        window_layout.addLayout(preset_layout)
        layout.addWidget(window_group)
        
        # フォントサイズ設定
        font_group = QGroupBox("フォントサイズ")
        font_layout = QVBoxLayout(font_group)
        
        # 現在のフォントサイズ
        current_font = self.font()
        current_font_label = QLabel(f"現在のフォントサイズ: {current_font.pointSize()}pt")
        font_layout.addWidget(current_font_label)
        
        # フォントサイズスライダー
        font_size_layout = QHBoxLayout()
        font_size_layout.addWidget(QLabel("サイズ:"))
        
        self.font_size_slider = QSlider(Qt.Horizontal)
        self.font_size_slider.setRange(8, 20)
        self.font_size_slider.setValue(current_font.pointSize())
        self.font_size_slider.setTickPosition(QSlider.TicksBelow)
        self.font_size_slider.setTickInterval(2)
        
        self.font_size_label = QLabel(f"{current_font.pointSize()}pt")
        self.font_size_slider.valueChanged.connect(lambda v: self.font_size_label.setText(f"{v}pt"))
        
        font_size_layout.addWidget(self.font_size_slider)
        font_size_layout.addWidget(self.font_size_label)
        font_layout.addLayout(font_size_layout)
        
        # プレビューテキスト
        preview_label = QLabel("プレビュー: アノテーションツール")
        preview_label.setFrameStyle(QFrame.Box)
        preview_label.setAlignment(Qt.AlignCenter)
        self.font_size_slider.valueChanged.connect(
            lambda v: preview_label.setFont(QFont(preview_label.font().family(), v))
        )
        font_layout.addWidget(preview_label)
        
        layout.addWidget(font_group)
        
        # 設定の保存オプション
        save_group = QGroupBox("設定の保存")
        save_layout = QVBoxLayout(save_group)
        
        self.save_settings_check = QCheckBox("次回起動時にこの設定を適用する")
        self.save_settings_check.setChecked(True)
        save_layout.addWidget(self.save_settings_check)
        
        layout.addWidget(save_group)
        
        # ボタン
        button_layout = QHBoxLayout()
        
        apply_button = QPushButton("適用")
        apply_button.clicked.connect(lambda: self.apply_display_settings(dialog, False))
        button_layout.addWidget(apply_button)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(lambda: self.apply_display_settings(dialog, True))
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("キャンセル")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()
    
    def set_window_size_preset(self, width, height):
        """ウィンドウサイズのプリセットを設定"""
        self.width_spin.setValue(width)
        self.height_spin.setValue(height)
    
    def apply_display_settings(self, dialog, close_dialog):
        """表示設定を適用"""
        # ウィンドウサイズを変更
        new_width = self.width_spin.value()
        new_height = self.height_spin.value()
        self.resize(new_width, new_height)
        
        # フォントサイズを変更
        new_font_size = self.font_size_slider.value()
        font = self.font()
        font.setPointSize(new_font_size)
        self.setFont(font)
        
        # すべての子ウィジェットにフォントを適用
        self.apply_font_to_children(self, font)
        
        # 設定を保存
        if self.save_settings_check.isChecked():
            self.save_display_settings(new_width, new_height, new_font_size, self.is_dark_mode)
        
        # ステータスバーに通知
        self.statusBar().showMessage(
            f"表示設定を適用しました - ウィンドウ: {new_width}x{new_height}, フォント: {new_font_size}pt", 
            3000
        )
        
        if close_dialog:
            dialog.accept()
    
    def apply_font_to_children(self, widget, font):
        """すべての子ウィジェットにフォントを適用"""
        for child in widget.findChildren(QWidget):
            if hasattr(child, 'setFont'):
                # 特定のウィジェットタイプごとにフォントサイズを調整
                if isinstance(child, (QPushButton, QLabel, QLineEdit, QComboBox)):
                    child_font = QFont(font)
                    child.setFont(child_font)
    
    def save_display_settings(self, width, height, font_size, dark_mode=None):
        """表示設定をファイルに保存"""
        settings_path = os.path.join(session_dir, "display_settings.json")
        settings = {
            "window_width": width,
            "window_height": height,
            "font_size": font_size
        }
        
        if dark_mode is not None:
            settings["dark_mode"] = dark_mode
        elif hasattr(self, 'is_dark_mode'):
            settings["dark_mode"] = self.is_dark_mode
            
        try:
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"表示設定の保存エラー: {e}")
    
    def load_display_settings(self):
        """保存された表示設定を読み込んで適用"""
        settings_path = os.path.join(session_dir, "display_settings.json")
        
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # ウィンドウサイズを適用
                if "window_width" in settings and "window_height" in settings:
                    self.resize(settings["window_width"], settings["window_height"])
                
                # フォントサイズを適用
                if "font_size" in settings:
                    font = self.font()
                    font.setPointSize(settings["font_size"])
                    self.setFont(font)
                    self.apply_font_to_children(self, font)
                
                # ダークモード設定を適用
                if "dark_mode" in settings:
                    self.is_dark_mode = settings["dark_mode"]
                    if hasattr(self, 'dark_mode_button'):
                        self.dark_mode_button.setChecked(self.is_dark_mode)
                    self.apply_dark_mode(self.is_dark_mode)
                    
            except Exception as e:
                print(f"表示設定の読み込みエラー: {e}")

    # ===========================================
    # Databricks連携機能
    # ===========================================

    def _check_databricks_connection_on_startup(self):
        """起動時にDatabricks接続を確認"""
        try:
            # Databricks連携が有効な場合のみ接続確認
            if self.mlflow_manager.use_databricks:
                print("起動時Databricks接続確認中...")

                # 接続を試みる（ダイアログなしでバックグラウンドで実行）
                self.mlflow_manager.is_initialized = False
                success = self.mlflow_manager.initialize(self.mlflow_manager.folder_path, parent_widget=None)

                if self.mlflow_manager._databricks_connected:
                    print("Databricks接続成功")
                else:
                    print("Databricks接続失敗 - ローカルモードで動作")

                # ステータスラベルを更新
                self._update_databricks_status_label()
        except Exception as e:
            print(f"起動時Databricks接続確認エラー: {e}")

    def _on_databricks_toggle(self, state):
        """Databricks連携のON/OFF切り替え"""
        enabled = state == Qt.Checked

        # MLflowManagerのモードを切り替え
        self.mlflow_manager.set_databricks_mode(enabled)

        # 有効にした場合は接続を試みる
        if enabled:
            self.mlflow_manager.is_initialized = False
            success = self.mlflow_manager.initialize(self.mlflow_manager.folder_path, parent_widget=self)
            if not success:
                QMessageBox.warning(
                    self,
                    "Databricks接続エラー",
                    "Databricksへの接続に失敗しました。\n\n"
                    "環境変数の設定を確認してください：\n"
                    "- DATABRICKS_HOST\n"
                    "- DATABRICKS_TOKEN\n\n"
                    "ローカルMLflowモードにフォールバックします。"
                )
                self.databricks_checkbox.setChecked(False)
        else:
            # ローカルモードに戻す
            self.mlflow_manager.is_initialized = False
            self.mlflow_manager.initialize(self.mlflow_manager.folder_path, parent_widget=self)

        # 状態ラベルを更新
        self._update_databricks_status_label()

    def _update_databricks_status_label(self):
        """Databricks状態ラベルを更新"""
        backend_info = self.mlflow_manager.get_backend_info()

        if backend_info["type"] == "databricks+local":
            self.databricks_status_label.setText(f"✓ Databricks+ローカル併用")
            self.databricks_status_label.setStyleSheet("color: green; font-size: 10px;")
        elif backend_info["type"] == "databricks":
            if backend_info["status"] == "未接続":
                self.databricks_status_label.setText("✗ Databricks: 未接続")
                self.databricks_status_label.setStyleSheet("color: orange; font-size: 10px;")
            else:
                self.databricks_status_label.setText(f"✓ Databricks: {backend_info['host'][:30]}...")
                self.databricks_status_label.setStyleSheet("color: green; font-size: 10px;")
        else:
            self.databricks_status_label.setText("ローカルMLflow使用中")
            self.databricks_status_label.setStyleSheet("color: gray; font-size: 10px;")

    def _open_local_mlflow_ui(self):
        """ローカルMLflow UIを開く"""
        try:
            import subprocess
            import sys
            from config import mlflow_dir

            # パスの正規化
            normalized_path = os.path.normpath(mlflow_dir).replace('\\', '/')
            if sys.platform.startswith('win'):
                tracking_uri = f"file:///{normalized_path}"
            else:
                tracking_uri = f"file://{normalized_path}"

            # MLflow UIを起動
            if sys.platform.startswith('win'):
                cmd = f'start cmd /k "mlflow ui --backend-store-uri {tracking_uri}"'
                subprocess.Popen(cmd, shell=True)
            else:
                cmd = f'mlflow ui --backend-store-uri {tracking_uri}'
                subprocess.Popen(cmd, shell=True)

            QMessageBox.information(
                self,
                "MLflow UI",
                "ローカルMLflow UIを起動しました。\n\n"
                "ブラウザで http://localhost:5000 にアクセスして実験結果を確認できます。\n\n"
                "UIを終了するには、コマンドウィンドウを閉じてください。"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "エラー",
                f"MLflow UIの起動に失敗しました:\n\n{str(e)}\n\n"
                "MLflowがインストールされているか確認してください: pip install mlflow"
            )

    def _open_databricks_ui(self):
        """Databricks MLflow UIを開く"""
        # Databricksモードが有効か確認
        if not self.mlflow_manager.use_databricks:
            QMessageBox.warning(
                self,
                "Databricks未有効",
                "Databricks連携が有効になっていません。\n\n"
                "「Databricks連携」チェックボックスをONにしてください。"
            )
            return

        # 未接続の場合、接続を試みる
        if not self.mlflow_manager._databricks_connected:
            self.mlflow_manager.is_initialized = False
            success = self.mlflow_manager.initialize(self.mlflow_manager.folder_path, parent_widget=self)

            self._update_databricks_status_label()

            if not self.mlflow_manager._databricks_connected:
                QMessageBox.warning(
                    self,
                    "Databricks接続失敗",
                    "Databricksへの接続に失敗しました。\n\n"
                    "環境変数の設定を確認してください。"
                )
                return

            QMessageBox.information(
                self,
                "Databricks接続成功",
                "Databricksへの接続に成功しました。"
            )

        # Databricks UIを開く
        self.mlflow_manager._open_databricks_ui(self)

        self._update_databricks_status_label()

    def _sync_to_databricks(self):
        """ローカルの学習記録をDatabricksに同期"""
        # Databricksモードが有効か確認
        if not self.mlflow_manager.use_databricks:
            QMessageBox.warning(
                self,
                "Databricks未有効",
                "Databricks連携が有効になっていません。\n\n"
                "「Databricks連携」チェックボックスをONにしてください。"
            )
            return

        # 同期状態を取得
        sync_status = self.mlflow_manager.get_sync_status()

        # 孤立Run数を取得（ローカルで削除されたがDatabricksに残っているRun）
        orphaned_count = self.mlflow_manager.get_orphaned_runs_count()

        # 同期オプションダイアログを表示
        dialog = QDialog(self)
        dialog.setWindowTitle("Databricks同期設定")
        dialog.setMinimumWidth(450)

        layout = QVBoxLayout(dialog)

        # 状態表示
        status_group = QGroupBox("現在の状態")
        status_layout = QVBoxLayout()
        status_layout.addWidget(QLabel(f"ローカルのRun数: {sync_status['local_runs']}"))
        status_layout.addWidget(QLabel(f"DatabricksのRun数: {sync_status['databricks_runs']}"))
        status_layout.addWidget(QLabel(f"推定未同期Run数: {sync_status['unsynced_runs']}"))
        if orphaned_count > 0:
            orphaned_label = QLabel(f"Databricksにのみ存在するRun数: {orphaned_count}")
            orphaned_label.setStyleSheet("color: orange;")
            status_layout.addWidget(orphaned_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 同期オプション
        options_group = QGroupBox("同期オプション")
        options_layout = QVBoxLayout()

        # アップロード同期
        upload_check = QCheckBox("ローカル→Databricks（新規Runをアップロード）")
        upload_check.setChecked(True)
        upload_check.setToolTip("ローカルにあってDatabricksにないRunをアップロードします")
        options_layout.addWidget(upload_check)

        # 削除同期
        delete_check = QCheckBox("ローカルで削除したRunをDatabricksからも削除")
        delete_check.setChecked(False)
        delete_check.setToolTip("ローカルに存在しないRunをDatabricksから削除します（注意: 元に戻せません）")
        if orphaned_count > 0:
            delete_check.setText(f"ローカルで削除したRunをDatabricksからも削除 ({orphaned_count}件)")
        options_layout.addWidget(delete_check)

        # 警告ラベル
        warning_label = QLabel("※ 削除オプションを有効にすると、Databricks上のRunが削除されます。\n   この操作は元に戻せません。")
        warning_label.setStyleSheet("color: red; font-size: 10px;")
        options_layout.addWidget(warning_label)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # ボタン
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return

        # オプションを取得
        do_upload = upload_check.isChecked()
        do_delete = delete_check.isChecked()

        if not do_upload and not do_delete:
            QMessageBox.information(self, "同期", "同期オプションが選択されていません。")
            return

        # 削除確認
        if do_delete and orphaned_count > 0:
            confirm = QMessageBox.warning(
                self,
                "削除確認",
                f"Databricksから{orphaned_count}件のRunを削除します。\n\n"
                "この操作は元に戻せません。続行しますか？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if confirm != QMessageBox.Yes:
                return

        # 進捗ダイアログを作成
        progress = QProgressDialog("Databricksに同期中...", "キャンセル", 0, 100, self)
        progress.setWindowTitle("同期中")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        # キャンセルフラグ
        cancelled = [False]

        def progress_callback(current, total, message):
            QApplication.processEvents()
            if progress.wasCanceled():
                cancelled[0] = True
                return
            percent = int((current / total) * 100) if total > 0 else 0
            progress.setValue(percent)
            progress.setLabelText(f"{message}\n({current}/{total})")
            QApplication.processEvents()

        def cancel_check():
            QApplication.processEvents()
            return progress.wasCanceled() or cancelled[0]

        # 同期実行
        try:
            if do_upload or do_delete:
                result = self.mlflow_manager.sync_local_to_databricks(
                    parent_widget=self,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    delete_orphaned=do_delete
                )
            else:
                result = {"synced": 0, "skipped": 0, "failed": 0, "deleted": 0, "errors": [], "cancelled": False}

            progress.close()

            # キャンセルされた場合
            if result.get("cancelled"):
                result_message = (
                    f"同期がキャンセルされました。\n\n"
                    f"キャンセル前に同期成功: {result['synced']} 件\n"
                    f"スキップ（既存）: {result['skipped']} 件\n"
                    f"削除: {result.get('deleted', 0)} 件"
                )
                QMessageBox.information(self, "同期キャンセル", result_message)
            elif result.get("message"):
                QMessageBox.information(self, "同期完了", result["message"])
            else:
                result_message = (
                    f"同期が完了しました。\n\n"
                    f"同期成功: {result['synced']} 件\n"
                    f"スキップ（既存）: {result['skipped']} 件\n"
                    f"失敗: {result['failed']} 件"
                )
                if result.get('deleted', 0) > 0:
                    result_message += f"\n削除: {result['deleted']} 件"

                if result['errors']:
                    result_message += f"\n\nエラー詳細:\n" + "\n".join(result['errors'][:5])
                    if len(result['errors']) > 5:
                        result_message += f"\n... 他 {len(result['errors']) - 5} 件"

                if result['failed'] > 0:
                    QMessageBox.warning(self, "同期完了（一部エラー）", result_message)
                else:
                    QMessageBox.information(self, "同期完了", result_message)

            # 状態を更新
            self._update_databricks_status_label()

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "同期エラー",
                f"同期中にエラーが発生しました:\n\n{str(e)}"
            )

    def _transfer_to_databricks(self):
        """現在のアノテーションをDatabricksに転送"""
        # Databricksモードが有効か確認
        if not self.mlflow_manager.use_databricks:
            QMessageBox.warning(
                self,
                "Databricks未有効",
                "Databricks連携が有効になっていません。\n\n"
                "「Databricks連携」チェックボックスをONにしてください。"
            )
            return

        # アノテーションがあるか確認
        if not self.annotations:
            QMessageBox.information(
                self,
                "情報",
                "転送するアノテーションがありません。\n\n"
                "先にアノテーションを作成してください。"
            )
            return

        # ZIPファイル名を入力
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"annotation_{timestamp}"

        zip_name, ok = QInputDialog.getText(
            self,
            "ZIPファイル名",
            "Databricksに転送するZIPファイル名を入力してください:\n"
            "（.zipは自動で付加されます）",
            QLineEdit.Normal,
            default_name
        )

        if not ok or not zip_name.strip():
            return

        zip_name = zip_name.strip()
        if not zip_name.endswith('.zip'):
            zip_name += '.zip'

        # 転送先を確認
        from config_databricks import DATABRICKS_VOLUMES_PATH

        # Volumesパスの存在確認
        print("[転送] Volumesパスの存在確認中...")
        temp_manager = DatabricksTransferManager()
        path_exists, path_message = temp_manager.check_volumes_path()

        if not path_exists:
            # パスが存在しない場合、作成を試みるか確認
            create_confirm = QMessageBox.question(
                self,
                "Volumesパスが存在しません",
                f"転送先のVolumesパスが存在しません:\n\n"
                f"{DATABRICKS_VOLUMES_PATH}\n\n"
                f"詳細: {path_message}\n\n"
                "Databricksでこのパスを作成してから再度お試しください。\n\n"
                "環境変数 DATABRICKS_VOLUMES_PATH で\n"
                "別のパスを指定することもできます。\n\n"
                "例: /Volumes/workspace/default/test",
                QMessageBox.Ok
            )
            return

        confirm = QMessageBox.question(
            self,
            "転送確認",
            f"以下の内容でDatabricksに転送します:\n\n"
            f"アノテーション数: {len(self.annotations)}\n"
            f"ファイル名: {zip_name}\n"
            f"転送先: {DATABRICKS_VOLUMES_PATH}/{zip_name}\n\n"
            "続行しますか？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if confirm != QMessageBox.Yes:
            return

        # 進捗ダイアログを作成
        progress = QProgressDialog("転送準備中...", "キャンセル", 0, 100, self)
        progress.setWindowTitle("Databricksへ転送中")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(400)
        progress.setValue(0)
        progress.show()

        # キャンセルフラグ
        cancelled = [False]

        def progress_callback(stage, current, total, message):
            QApplication.processEvents()
            if progress.wasCanceled():
                cancelled[0] = True
                return

            # ステージに応じた進捗計算
            # export: 0-20%, zip: 20-60%, upload: 60-100%
            if stage == 'export':
                percent = int((current / max(total, 1)) * 20)
                stage_name = "エクスポート中"
            elif stage == 'zip':
                percent = 20 + int((current / max(total, 1)) * 40)
                stage_name = "ZIP圧縮中"
            elif stage == 'upload':
                percent = 60 + int((current / max(total, 1)) * 40)
                stage_name = "アップロード中"
            else:
                percent = 0
                stage_name = ""

            progress.setValue(percent)
            progress.setLabelText(f"{stage_name}\n{message}")
            QApplication.processEvents()

        def cancel_check():
            QApplication.processEvents()
            return progress.wasCanceled() or cancelled[0]

        # 転送実行
        try:
            print("[転送] DatabricksTransferManager を作成中...")
            # DatabricksTransferManagerを作成
            transfer_manager = DatabricksTransferManager()

            print("[転送] image_mapとvariant_keysを構築中...")
            # image_mapとvariant_keysを構築
            image_map = {}
            variant_keys = {}

            # source_images_mapから構築
            if hasattr(self, 'source_images_map') and self.source_images_map:
                print(f"[転送] source_images_map を使用（バリアント数: {len(self.source_images_map)}）")
                for variant, images_list in self.source_images_map.items():
                    # variant_keysを設定
                    if variant == 'cam':
                        variant_keys[variant] = 'cam/image_array'
                    elif variant == 'cam0':
                        variant_keys[variant] = 'cam0/image_array'
                    elif variant == 'lidar':
                        variant_keys[variant] = 'lidar/image_array'
                    else:
                        variant_keys[variant] = f'{variant}/image_array'

                    # image_mapを構築
                    for idx, img_path in enumerate(images_list):
                        if idx not in image_map:
                            image_map[idx] = {}
                        image_map[idx][variant] = img_path
                print(f"[転送] image_map 構築完了（エントリ数: {len(image_map)}）")
            else:
                # 単一ソースの場合
                if hasattr(self, 'images') and self.images:
                    print(f"[転送] 単一ソースを使用（画像数: {len(self.images)}）")
                    variant_keys['cam'] = 'cam/image_array'
                    for idx, img_path in enumerate(self.images):
                        image_map[idx] = {'cam': img_path}

            # 削除インデックスを取得
            deleted_indexes = getattr(self, 'deleted_indexes', [])
            print(f"[転送] 削除インデックス数: {len(deleted_indexes)}")

            # 差分ベクトルとウェイポイントを取得
            diff_vectors = getattr(self, 'inference_diff_vectors', None)
            waypoint_annotations = getattr(self, 'waypoint_annotations', None)
            print(f"[転送] 差分ベクトル: {'あり' if diff_vectors else 'なし'}")
            print(f"[転送] ウェイポイント: {'あり' if waypoint_annotations else 'なし'}")

            print("[転送] transfer_annotations 呼び出し...")
            # 転送実行
            result = transfer_manager.transfer_annotations(
                annotations=self.annotations,
                inference_results=self.inference_results if hasattr(self, 'inference_results') else None,
                image_map=image_map,
                variant_keys=variant_keys,
                zip_name=zip_name,
                deleted_indexes=deleted_indexes,
                diff_vectors=diff_vectors,
                waypoint_annotations=waypoint_annotations,
                progress_callback=progress_callback,
                cancel_check=cancel_check
            )

            print(f"[転送] transfer_annotations 完了: {result}")
            progress.close()

            if result['success']:
                # サイズをMBに変換
                size_mb = result['zip_size'] / (1024 * 1024)
                QMessageBox.information(
                    self,
                    "転送完了",
                    f"Databricksへの転送が完了しました。\n\n"
                    f"アノテーション数: {result['annotation_count']}\n"
                    f"ZIPサイズ: {size_mb:.2f} MB\n"
                    f"転送先: {result['remote_path']}"
                )
            else:
                error_msg = result.get('error', '不明なエラー')
                if 'キャンセル' in error_msg:
                    QMessageBox.information(self, "転送キャンセル", "転送がキャンセルされました。")
                else:
                    QMessageBox.critical(
                        self,
                        "転送エラー",
                        f"転送中にエラーが発生しました:\n\n{error_msg}"
                    )

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "転送エラー",
                f"転送中にエラーが発生しました:\n\n{str(e)}\n\n{traceback.format_exc()}"
            )

    def _show_databricks_settings(self):
        """Databricks設定ダイアログを表示"""
        try:
            from config_databricks import (
                DATABRICKS_ENABLED, DATABRICKS_HOST, DATABRICKS_TOKEN,
                DATABRICKS_EXPERIMENT_PREFIX, get_databricks_status, get_env_template
            )
            config_available = True
        except ImportError:
            config_available = False

        dialog = QDialog(self)
        dialog.setWindowTitle("Databricks設定")
        dialog.setMinimumWidth(550)
        layout = QVBoxLayout(dialog)

        # 現在の状態を表示
        status_group = QGroupBox("接続状態")
        status_layout = QVBoxLayout()

        if config_available:
            status = get_databricks_status()
            status_text = f"状態: {status['status']}\n{status['message']}"
        else:
            status_text = "config_databricks.py が見つかりません"

        status_label = QLabel(status_text)
        status_label.setWordWrap(True)
        status_layout.addWidget(status_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 環境変数の状態を表示
        env_group = QGroupBox("環境変数の状態")
        env_layout = QFormLayout()

        env_enabled = os.environ.get("DATABRICKS_ENABLED", "")
        env_host = os.environ.get("DATABRICKS_HOST", "")
        env_token = os.environ.get("DATABRICKS_TOKEN", "")
        env_prefix = os.environ.get("DATABRICKS_EXPERIMENT_PREFIX", "")

        env_layout.addRow("DATABRICKS_ENABLED:", QLabel(env_enabled or "(未設定)"))
        env_layout.addRow("DATABRICKS_HOST:", QLabel(env_host[:40] + "..." if len(env_host) > 40 else env_host or "(未設定)"))
        env_layout.addRow("DATABRICKS_TOKEN:", QLabel("****" + env_token[-4:] if env_token else "(未設定)"))
        env_layout.addRow("EXPERIMENT_PREFIX:", QLabel(env_prefix or "(デフォルト使用)"))

        env_group.setLayout(env_layout)
        layout.addWidget(env_group)

        # 設定方法の説明
        help_group = QGroupBox("環境変数の設定方法")
        help_layout = QVBoxLayout()
        help_text = QLabel(
            "セキュリティのため、認証情報は環境変数で設定してください:\n\n"
            "Windows (PowerShell):\n"
            '  $env:DATABRICKS_ENABLED = "true"\n'
            '  $env:DATABRICKS_HOST = "https://..."\n'
            '  $env:DATABRICKS_TOKEN = "dapi..."\n\n'
            "Linux/Mac:\n"
            '  export DATABRICKS_ENABLED="true"\n'
            '  export DATABRICKS_HOST="https://..."\n'
            '  export DATABRICKS_TOKEN="dapi..."'
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("font-family: monospace;")
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        layout.addWidget(help_group)

        # ボタンレイアウト
        button_layout = QHBoxLayout()

        # テンプレートをコピーボタン
        if config_available:
            copy_template_button = QPushButton("設定テンプレートをコピー")
            copy_template_button.clicked.connect(lambda: self._copy_env_template(get_env_template()))
            button_layout.addWidget(copy_template_button)

        # READMEを開くボタン
        open_readme_button = QPushButton("READMEを開く")
        open_readme_button.clicked.connect(self._open_databricks_readme)
        button_layout.addWidget(open_readme_button)

        layout.addLayout(button_layout)

        # 閉じるボタン
        close_button = QPushButton("閉じる")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.exec_()

    def _copy_env_template(self, template: str):
        """環境変数テンプレートをクリップボードにコピー"""
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(template)
        QMessageBox.information(self, "コピー完了", "環境変数設定テンプレートをクリップボードにコピーしました")

    def _open_databricks_readme(self):
        """README_DATABRICKS.md を開く"""
        import subprocess
        import sys

        readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README_DATABRICKS.md")

        if not os.path.exists(readme_path):
            QMessageBox.warning(self, "エラー", f"READMEが見つかりません:\n{readme_path}")
            return

        try:
            if sys.platform.startswith('win'):
                os.startfile(readme_path)
            elif sys.platform.startswith('darwin'):
                subprocess.Popen(['open', readme_path])
            else:
                subprocess.Popen(['xdg-open', readme_path])
        except Exception as e:
            QMessageBox.warning(self, "エラー", f"ファイルを開けませんでした:\n{e}")

    # ========================================
    # Google Colab連携メソッド
    # ========================================

    def _is_colab_enabled(self) -> bool:
        """Colab連携が有効かどうかを返す"""
        try:
            from config_colab import COLAB_ENABLED
            return COLAB_ENABLED
        except ImportError:
            return False

    def _on_colab_toggle(self, state):
        """Colabチェックボックスの状態変更"""
        # 現在は環境変数で制御するため、チェックボックスは情報表示のみ
        self._update_colab_status_label()

    def _update_colab_status_label(self):
        """Colabステータスラベルを更新"""
        try:
            from config_colab import get_colab_status
            status = get_colab_status()
            if status['enabled']:
                if status.get('authenticated'):
                    self.colab_status_label.setText("認証済み")
                    self.colab_status_label.setStyleSheet("color: green;")
                else:
                    self.colab_status_label.setText("未認証")
                    self.colab_status_label.setStyleSheet("color: orange;")
            else:
                self.colab_status_label.setText("無効")
                self.colab_status_label.setStyleSheet("color: gray;")
        except ImportError:
            self.colab_status_label.setText("設定ファイルなし")
            self.colab_status_label.setStyleSheet("color: red;")

    def _open_colab_ui(self):
        """Google Colabを開く"""
        import webbrowser
        webbrowser.open("https://colab.research.google.com/")

    def _transfer_to_colab(self):
        """現在のアノテーションをGoogle Driveに転送してColabで学習"""
        # Colabモードが有効か確認
        if not self._is_colab_enabled():
            QMessageBox.warning(
                self,
                "Google Colab未有効",
                "Google Colab連携が有効になっていません。\n\n"
                "有効にするには環境変数を設定してください:\n"
                "  COLAB_ENABLED=true\n"
                "  GOOGLE_CLIENT_SECRETS=path/to/client_secrets.json\n\n"
                "設定ボタンから詳細を確認できます。"
            )
            return

        # アノテーションがあるか確認
        if not self.annotations:
            QMessageBox.information(
                self,
                "情報",
                "転送するアノテーションがありません。\n\n"
                "先にアノテーションを作成してください。"
            )
            return

        # ZIPファイル名を入力
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"annotation_{timestamp}"

        zip_name, ok = QInputDialog.getText(
            self,
            "ZIPファイル名",
            "Google Driveに転送するZIPファイル名を入力してください:\n"
            "（.zipは自動で付加されます）",
            QLineEdit.Normal,
            default_name
        )

        if not ok or not zip_name.strip():
            return

        zip_name = zip_name.strip()
        if not zip_name.endswith('.zip'):
            zip_name += '.zip'

        # 転送確認ダイアログ
        try:
            from config_colab import COLAB_DRIVE_FOLDER_NAME
        except ImportError:
            COLAB_DRIVE_FOLDER_NAME = "annotation_data"

        # オプションダイアログを表示
        dialog = QDialog(self)
        dialog.setWindowTitle("Google Colab転送設定")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # 転送内容
        info_group = QGroupBox("転送内容")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel(f"アノテーション数: {len(self.annotations)}"))
        info_layout.addWidget(QLabel(f"ファイル名: {zip_name}"))
        info_layout.addWidget(QLabel(f"転送先: Google Drive/{COLAB_DRIVE_FOLDER_NAME}/"))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # オプション
        options_group = QGroupBox("オプション")
        options_layout = QVBoxLayout()

        generate_notebook_check = QCheckBox("学習用Notebookを生成")
        generate_notebook_check.setChecked(True)
        generate_notebook_check.setToolTip("転送後にGoogle Colabで使用できるNotebookを生成します")
        options_layout.addWidget(generate_notebook_check)

        open_colab_check = QCheckBox("転送後にColabを開く")
        open_colab_check.setChecked(True)
        open_colab_check.setToolTip("転送完了後にブラウザでColabを開きます")
        options_layout.addWidget(open_colab_check)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # 認証に関する注意
        note_label = QLabel(
            "注意: 初回転送時はGoogleアカウントの認証が必要です。\n"
            "ブラウザが開きますので、アカウントを選択して認証してください。"
        )
        note_label.setStyleSheet("color: gray; font-size: 10px;")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)

        # ボタン
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return

        generate_notebook = generate_notebook_check.isChecked()
        open_colab = open_colab_check.isChecked()

        # 認証確認
        try:
            from utils.colab_transfer import ColabTransferManager

            # 認証中ダイアログを表示
            auth_progress = QProgressDialog("Google Driveに認証中...", "キャンセル", 0, 0, self)
            auth_progress.setWindowTitle("認証中")
            auth_progress.setWindowModality(Qt.WindowModal)
            auth_progress.setMinimumDuration(0)
            auth_progress.setMinimumWidth(300)
            auth_progress.show()
            QApplication.processEvents()

            # ColabTransferManagerを作成して認証
            transfer_manager = ColabTransferManager()
            success, message = transfer_manager.test_connection()

            auth_progress.close()

            if not success:
                QMessageBox.critical(
                    self,
                    "認証失敗",
                    f"Google Driveへの認証に失敗しました:\n\n{message}"
                )
                return

            # 認証成功 - 転送確認ダイアログを表示
            try:
                from config_colab import COLAB_DRIVE_FOLDER_NAME
            except ImportError:
                COLAB_DRIVE_FOLDER_NAME = "annotation_data"

            confirm_reply = QMessageBox.question(
                self,
                "認証完了 - 転送確認",
                f"Google Driveへの認証が完了しました。\n\n"
                f"以下の内容で転送を開始しますか？\n\n"
                f"  転送先: Google Drive/{COLAB_DRIVE_FOLDER_NAME}/\n"
                f"  ファイル名: {zip_name}\n"
                f"  アノテーション数: {len(self.annotations)}\n"
                f"  Notebook生成: {'あり' if generate_notebook else 'なし'}\n",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if confirm_reply != QMessageBox.Yes:
                return

        except ImportError as e:
            QMessageBox.critical(
                self,
                "インポートエラー",
                f"必要なライブラリがインストールされていません:\n\n{str(e)}\n\n"
                "pip install pydrive2 google-auth google-auth-oauthlib pyyaml でインストールしてください。"
            )
            return
        except Exception as e:
            QMessageBox.critical(
                self,
                "認証エラー",
                f"認証中にエラーが発生しました:\n\n{str(e)}"
            )
            return

        # 進捗ダイアログを作成
        progress = QProgressDialog("転送準備中...", "キャンセル", 0, 100, self)
        progress.setWindowTitle("Google Colabへ転送中")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(400)
        progress.setValue(0)
        progress.show()

        # キャンセルフラグ
        cancelled = [False]

        # 最後に更新した進捗値を記録（不要な更新を避ける）
        last_percent = [-1]

        def progress_callback(stage, current, total, message):
            # ステージに応じた進捗計算
            if stage == 'export':
                percent = int((current / max(total, 1)) * 15)
                stage_header = "--- ステージ1: エクスポート ---"
                if current == 0:
                    detail = "アノテーションをエクスポート中..."
                else:
                    detail = f"エクスポート完了"
            elif stage == 'zip':
                percent = 15 + int((current / max(total, 1)) * 30)
                stage_header = "--- ステージ2: ZIP圧縮 ---"
                progress_pct = int((current / max(total, 1)) * 100)
                detail = f"ZIP圧縮進捗: {current}/{total} ({progress_pct}%)"
            elif stage == 'upload':
                percent = 45 + int((current / max(total, 1)) * 35)
                stage_header = "--- ステージ3: アップロード ---"
                if current == 0:
                    detail = f"Google Driveにアップロード中...\nファイルサイズ: {total / (1024*1024):.2f} MB"
                else:
                    detail = f"アップロード中: {current // (1024*1024)} MB / {total // (1024*1024)} MB"
            elif stage == 'notebook':
                percent = 80 + int((current / max(total, 1)) * 20)
                stage_header = "--- ステージ4: Notebook生成 ---"
                if current == 0:
                    detail = "Colabノートブックを生成中..."
                else:
                    detail = "ノートブックアップロード完了"
            else:
                percent = 0
                stage_header = ""
                detail = message

            # 進捗値が変わった場合のみGUI更新（パフォーマンス最適化）
            if percent != last_percent[0]:
                last_percent[0] = percent
                progress.setValue(percent)
                progress.setLabelText(f"{stage_header}\n{detail}")
                QApplication.processEvents()

                # キャンセルチェック
                if progress.wasCanceled():
                    cancelled[0] = True

        def cancel_check():
            # cancelled[0]がすでにTrueならprocessEventsをスキップ
            if cancelled[0]:
                return True
            # GUIイベント処理してキャンセル状態を確認
            QApplication.processEvents()
            if progress.wasCanceled():
                cancelled[0] = True
            return cancelled[0]

        # 転送実行
        try:
            # image_mapとvariant_keysを構築（Databricksと同じロジック）
            image_map = {}
            variant_keys = {}

            if hasattr(self, 'source_images_map') and self.source_images_map:
                for variant, images_list in self.source_images_map.items():
                    if variant == 'cam':
                        variant_keys[variant] = 'cam/image_array'
                    elif variant == 'cam0':
                        variant_keys[variant] = 'cam0/image_array'
                    elif variant == 'lidar':
                        variant_keys[variant] = 'lidar/image_array'
                    else:
                        variant_keys[variant] = f'{variant}/image_array'

                    for idx, img_path in enumerate(images_list):
                        if idx not in image_map:
                            image_map[idx] = {}
                        image_map[idx][variant] = img_path
            else:
                if hasattr(self, 'images') and self.images:
                    variant_keys['cam'] = 'cam/image_array'
                    for idx, img_path in enumerate(self.images):
                        image_map[idx] = {'cam': img_path}

            deleted_indexes = getattr(self, 'deleted_indexes', [])
            diff_vectors = getattr(self, 'inference_diff_vectors', None)
            waypoint_annotations = getattr(self, 'waypoint_annotations', None)

            # 転送実行（認証済みのtransfer_managerを使用）
            result = transfer_manager.transfer_annotations(
                annotations=self.annotations,
                inference_results=self.inference_results if hasattr(self, 'inference_results') else None,
                image_map=image_map,
                variant_keys=variant_keys,
                zip_name=zip_name,
                deleted_indexes=deleted_indexes,
                diff_vectors=diff_vectors,
                waypoint_annotations=waypoint_annotations,
                generate_notebook=generate_notebook,
                open_colab=open_colab,
                progress_callback=progress_callback,
                cancel_check=cancel_check
            )

            progress.close()

            if result['success']:
                size_mb = result['zip_size'] / (1024 * 1024)
                message = (
                    f"Google Driveへの転送が完了しました。\n\n"
                    f"アノテーション数: {result['annotation_count']}\n"
                    f"ZIPサイズ: {size_mb:.2f} MB"
                )
                if 'colab_url' in result:
                    message += f"\n\nColab URL:\n{result['colab_url']}"

                QMessageBox.information(self, "転送完了", message)
                self._update_colab_status_label()
            else:
                error_msg = result.get('error', '不明なエラー')
                if 'キャンセル' in error_msg:
                    QMessageBox.information(self, "転送キャンセル", "転送がキャンセルされました。")
                else:
                    QMessageBox.critical(
                        self,
                        "転送エラー",
                        f"転送中にエラーが発生しました:\n\n{error_msg}"
                    )

        except ImportError as e:
            progress.close()
            QMessageBox.critical(
                self,
                "インポートエラー",
                f"必要なライブラリがインストールされていません:\n\n{str(e)}\n\n"
                "pip install pydrive2 google-auth google-auth-oauthlib pyyaml でインストールしてください。"
            )
        except Exception as e:
            progress.close()
            import traceback
            QMessageBox.critical(
                self,
                "転送エラー",
                f"転送中にエラーが発生しました:\n\n{str(e)}\n\n{traceback.format_exc()}"
            )

    def _download_model_from_colab(self):
        """Colabで学習したモデルをGoogle Driveからダウンロード"""
        # Colabモードが有効か確認
        if not self._is_colab_enabled():
            QMessageBox.warning(
                self,
                "Google Colab未有効",
                "Google Colab連携が有効になっていません。\n\n"
                "有効にするには環境変数を設定してください:\n"
                "  COLAB_ENABLED=true\n"
                "  GOOGLE_CLIENT_SECRETS=path/to/client_secrets.json"
            )
            return

        try:
            from utils.colab_transfer import ColabTransferManager

            # 進捗ダイアログを表示
            progress = QProgressDialog("Google Driveに接続中...", "キャンセル", 0, 100, self)
            progress.setWindowTitle("モデル一覧を取得中")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(10)
            progress.show()
            QApplication.processEvents()

            # モデル一覧を取得
            transfer_manager = ColabTransferManager()
            models = transfer_manager.list_models()

            progress.close()

            if not models:
                QMessageBox.information(
                    self,
                    "モデルなし",
                    "Google Driveにモデルファイルが見つかりませんでした。\n\n"
                    "Colabでモデルを学習し、Google Driveに保存してください。"
                )
                return

            # モデル選択ダイアログを表示
            dialog = QDialog(self)
            dialog.setWindowTitle("モデルをダウンロード")
            dialog.setMinimumWidth(500)

            layout = QVBoxLayout(dialog)

            # 説明ラベル
            info_label = QLabel(f"Google Drive上のモデル: {len(models)}件")
            layout.addWidget(info_label)

            # モデルリスト
            list_widget = QListWidget()
            for m in models:
                size_mb = m['size'] / (1024 * 1024)
                # 作成日時をフォーマット
                created = m['createdDate'][:10] if m['createdDate'] else "不明"
                item_text = f"{m['name']}  ({size_mb:.2f} MB, {created})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, m)
                list_widget.addItem(item)

            list_widget.setCurrentRow(0)
            layout.addWidget(list_widget)

            # 保存先
            save_group = QGroupBox("保存先")
            save_layout = QHBoxLayout()
            save_path_edit = QLineEdit()
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
            save_path_edit.setText(models_dir)
            save_path_edit.setReadOnly(True)
            save_layout.addWidget(save_path_edit)

            browse_button = QPushButton("参照...")
            def browse_folder():
                folder = QFileDialog.getExistingDirectory(dialog, "保存先フォルダを選択", models_dir)
                if folder:
                    save_path_edit.setText(folder)
            browse_button.clicked.connect(browse_folder)
            save_layout.addWidget(browse_button)
            save_group.setLayout(save_layout)
            layout.addWidget(save_group)

            # MLflowデータダウンロードオプション
            mlruns_checkbox = QCheckBox("MLflow実験データ(mlruns)もダウンロードしてマージする")
            mlruns_checkbox.setChecked(True)
            mlruns_checkbox.setToolTip("Colabで記録されたMLflow実験データをローカルにマージします")
            layout.addWidget(mlruns_checkbox)

            # ボタン
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            if dialog.exec_() != QDialog.Accepted:
                return

            # 選択されたモデルを取得
            current_item = list_widget.currentItem()
            if not current_item:
                return

            selected_model = current_item.data(Qt.UserRole)
            save_dir = save_path_edit.text()
            download_mlruns = mlruns_checkbox.isChecked()

            # 同じ名前のモデルが既に存在するかチェック
            local_path = os.path.join(save_dir, selected_model['name'])
            if os.path.exists(local_path):
                local_size = os.path.getsize(local_path)
                remote_size = selected_model['size']

                if local_size == remote_size:
                    # ファイルサイズが一致 → 同一ファイルの可能性が高い
                    reply = QMessageBox.question(
                        self,
                        "同名モデルが存在",
                        f"同じ名前のモデルが既に存在します:\n"
                        f"  {selected_model['name']}\n\n"
                        f"ローカル: {local_size / (1024*1024):.2f} MB\n"
                        f"リモート: {remote_size / (1024*1024):.2f} MB\n\n"
                        f"ファイルサイズが同じため、同一のモデルと思われます。\n"
                        f"ダウンロードをスキップしますか？",
                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                        QMessageBox.Yes
                    )
                    if reply == QMessageBox.Cancel:
                        return
                    elif reply == QMessageBox.Yes:
                        # スキップしてmlrunsのみダウンロード
                        if download_mlruns:
                            progress = QProgressDialog("MLflow実験データをダウンロード中...", "キャンセル", 0, 100, self)
                            progress.setWindowTitle("ダウンロード中")
                            progress.setWindowModality(Qt.WindowModal)
                            progress.setMinimumDuration(0)
                            progress.setValue(50)
                            progress.show()
                            QApplication.processEvents()

                            try:
                                mlruns_merged = self._download_and_merge_mlruns(transfer_manager, progress)
                            except Exception as e:
                                print(f"[Colab] mlrunsのダウンロードに失敗: {e}")
                            progress.close()

                        QMessageBox.information(
                            self,
                            "スキップ",
                            f"モデルのダウンロードをスキップしました。\n"
                            f"既存のモデルを使用: {local_path}"
                        )
                        return
                    # Noの場合は上書きダウンロードを続行
                else:
                    # ファイルサイズが異なる → 異なるバージョン
                    reply = QMessageBox.question(
                        self,
                        "同名モデルが存在",
                        f"同じ名前のモデルが既に存在しますが、サイズが異なります:\n"
                        f"  {selected_model['name']}\n\n"
                        f"ローカル: {local_size / (1024*1024):.2f} MB\n"
                        f"リモート: {remote_size / (1024*1024):.2f} MB\n\n"
                        f"上書きダウンロードしますか？",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return

            # ダウンロード実行
            progress = QProgressDialog("モデルをダウンロード中...", "キャンセル", 0, 100, self)
            progress.setWindowTitle("ダウンロード中")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(20)
            progress.show()
            QApplication.processEvents()

            def download_progress(current, total):
                if total > 0:
                    percent = int((current / total) * 80) + 20
                    progress.setValue(percent)
                    progress.setLabelText(f"ダウンロード中: {current // (1024*1024)} MB / {total // (1024*1024)} MB")
                    QApplication.processEvents()

            result_path = transfer_manager.download_file(
                selected_model['id'],
                local_path,
                progress_callback=download_progress
            )

            if result_path:
                # mlrunsをダウンロードしてマージ
                mlruns_merged = False
                if download_mlruns:
                    progress.setLabelText("MLflow実験データをダウンロード中...")
                    progress.setValue(50)
                    QApplication.processEvents()

                    try:
                        mlruns_merged = self._download_and_merge_mlruns(transfer_manager, progress)
                    except Exception as e:
                        print(f"[Colab] mlrunsのダウンロードに失敗: {e}")
                        # mlrunsのエラーは警告のみで続行

                progress.close()

                # ダウンロード成功
                message = f"モデルをダウンロードしました:\n{result_path}"
                if mlruns_merged:
                    message += "\n\nMLflow実験データもマージしました。"
                message += "\n\nこのモデルを読み込みますか？"

                reply = QMessageBox.question(
                    self,
                    "ダウンロード完了",
                    message,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    # モデルを読み込み
                    self._load_downloaded_model(result_path)
            else:
                progress.close()
                QMessageBox.warning(self, "エラー", "モデルのダウンロードに失敗しました。")

        except Exception as e:
            if 'progress' in locals():
                progress.close()
            import traceback
            QMessageBox.critical(
                self,
                "エラー",
                f"モデルのダウンロード中にエラーが発生しました:\n\n{str(e)}\n\n{traceback.format_exc()}"
            )

    def _download_and_merge_mlruns(self, transfer_manager, progress) -> bool:
        """Colabからmlrunsをダウンロードしてローカルにマージ

        Args:
            transfer_manager: ColabTransferManagerインスタンス
            progress: QProgressDialog

        Returns:
            マージ成功した場合True
        """
        import tempfile
        import shutil

        print("[Colab] mlrunsダウンロード開始...")

        # Google Driveからmlrunsをダウンロード
        temp_dir = tempfile.mkdtemp(prefix="mlruns_download_")

        try:
            def mlruns_progress(filename, current, total):
                progress.setLabelText(f"MLflow実験データをダウンロード中: {filename}")
                if total > 0:
                    percent = 50 + int((current / total) * 30)
                    progress.setValue(percent)
                QApplication.processEvents()

            # ローカルのmlrunsパスを取得
            local_mlruns = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlruns')

            download_result = transfer_manager.download_mlruns(
                local_dir=temp_dir,
                progress_callback=mlruns_progress,
                compare_with_local=local_mlruns  # ローカルのmlrunsと比較してスキップ
            )

            if not download_result:
                print("[Colab] mlrunsフォルダが見つかりませんでした")
                return False

            remote_mlruns = download_result['path']
            downloaded = download_result['downloaded']
            skipped = download_result['skipped']
            total = download_result['total']

            # スキップ情報を表示
            if skipped > 0:
                if downloaded == 0:
                    print(f"[Colab] 全ファイルが既にダウンロード済みです ({skipped}件)")
                else:
                    print(f"[Colab] ダウンロード: {downloaded}件, スキップ: {skipped}件 (既存)")

            # マージ処理
            progress.setLabelText("MLflow実験データをマージ中...")
            progress.setValue(85)
            QApplication.processEvents()

            merged_count = self._merge_mlruns_folders(remote_mlruns, local_mlruns)

            print(f"[Colab] mlrunsマージ完了: {merged_count}件の実験データをマージ")
            progress.setValue(95)
            QApplication.processEvents()

            return merged_count > 0

        except Exception as e:
            print(f"[Colab] mlrunsマージエラー: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # 一時ディレクトリをクリーンアップ
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[Colab] 一時ディレクトリの削除に失敗: {e}")

    def _merge_mlruns_folders(self, source_mlruns: str, dest_mlruns: str) -> int:
        """mlrunsフォルダをマージ

        Args:
            source_mlruns: マージ元のmlrunsパス（Colabからダウンロードしたもの）
            dest_mlruns: マージ先のmlrunsパス（ローカル）

        Returns:
            マージされた実験数
        """
        import shutil

        # ローカルのmlrunsがなければ作成
        os.makedirs(dest_mlruns, exist_ok=True)

        merged_count = 0

        # source_mlruns内の実験フォルダを取得
        if not os.path.exists(source_mlruns):
            return 0

        for item in os.listdir(source_mlruns):
            source_item = os.path.join(source_mlruns, item)

            # .trashフォルダはスキップ
            if item == '.trash':
                continue

            if os.path.isdir(source_item):
                dest_item = os.path.join(dest_mlruns, item)

                if item == 'models':
                    # modelsフォルダは特別扱い（上書きマージ）
                    if os.path.exists(dest_item):
                        # 既存のmodelsフォルダに追加
                        for model_file in os.listdir(source_item):
                            src_model = os.path.join(source_item, model_file)
                            dst_model = os.path.join(dest_item, model_file)
                            if not os.path.exists(dst_model):
                                if os.path.isdir(src_model):
                                    shutil.copytree(src_model, dst_model)
                                else:
                                    shutil.copy2(src_model, dst_model)
                    else:
                        shutil.copytree(source_item, dest_item)
                    continue

                # 実験フォルダ（数字のID）
                if os.path.exists(dest_item):
                    # 既存の実験フォルダがある場合、run単位でマージ
                    for run_id in os.listdir(source_item):
                        src_run = os.path.join(source_item, run_id)
                        dst_run = os.path.join(dest_item, run_id)

                        if os.path.isdir(src_run) and not os.path.exists(dst_run):
                            # 新しいrunをコピー
                            shutil.copytree(src_run, dst_run)
                            print(f"[Colab] run追加: {item}/{run_id}")
                            merged_count += 1
                        elif run_id == 'meta.yaml' and not os.path.exists(dst_run):
                            # meta.yamlをコピー
                            shutil.copy2(src_run, dst_run)
                else:
                    # 新しい実験フォルダをそのままコピー
                    shutil.copytree(source_item, dest_item)
                    # run数をカウント
                    runs = [d for d in os.listdir(dest_item) if os.path.isdir(os.path.join(dest_item, d))]
                    merged_count += len(runs)
                    print(f"[Colab] 実験追加: {item} ({len(runs)} runs)")

        return merged_count

    def _load_downloaded_model(self, model_path: str):
        """ダウンロードしたモデルを読み込む"""
        try:
            import torch

            # モデルをロード
            checkpoint = torch.load(model_path, map_location='cpu')

            # configを取得
            if 'config' in checkpoint:
                config = checkpoint['config']
                model_name = config.get('model_name', config.get('backbone', 'unknown'))
                print(f"[Colab] モデルを読み込みました: {model_name}")

                QMessageBox.information(
                    self,
                    "モデル読み込み完了",
                    f"モデルを保存しました:\n\n"
                    f"ファイル: {os.path.basename(model_path)}\n"
                    f"モデル: {model_name}\n\n"
                    "「モデル読込」ボタンから推論に使用できます。"
                )
            else:
                QMessageBox.information(
                    self,
                    "モデル保存完了",
                    f"モデルファイルを保存しました:\n{model_path}\n\n"
                    "「モデル読込」ボタンから読み込んでください。"
                )

        except Exception as e:
            QMessageBox.warning(
                self,
                "確認エラー",
                f"モデルの確認に失敗しました:\n{str(e)}\n\n"
                "ファイルは保存されています。「モデル読込」ボタンから読み込んでください。"
            )

    def _show_colab_settings(self):
        """Colab設定ダイアログを表示"""
        try:
            from config_colab import (
                COLAB_ENABLED, GOOGLE_CLIENT_SECRETS,
                COLAB_DRIVE_FOLDER_NAME, get_colab_status, get_env_template,
                get_oauth_setup_guide
            )
            config_available = True
        except ImportError:
            config_available = False

        dialog = QDialog(self)
        dialog.setWindowTitle("Google Colab設定")
        dialog.setMinimumWidth(600)
        layout = QVBoxLayout(dialog)

        # 現在の状態を表示
        status_group = QGroupBox("接続状態")
        status_layout = QVBoxLayout()

        if config_available:
            status = get_colab_status()
            status_text = f"状態: {status['status']}\n{status['message']}"
        else:
            status_text = "config_colab.py が見つかりません"

        status_label = QLabel(status_text)
        status_label.setWordWrap(True)
        status_layout.addWidget(status_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 設定方法を表示（タブで切り替え）
        tab_widget = QTabWidget()

        # 環境変数タブ
        env_tab = QWidget()
        env_layout = QVBoxLayout(env_tab)
        env_text = QPlainTextEdit()
        env_text.setReadOnly(True)
        if config_available:
            env_text.setPlainText(get_env_template())
        else:
            env_text.setPlainText("config_colab.py が見つかりません")
        env_text.setMinimumHeight(150)
        env_layout.addWidget(env_text)

        # コピーボタン
        if config_available:
            copy_env_button = QPushButton("環境変数テンプレートをコピー")
            copy_env_button.clicked.connect(lambda: self._copy_env_template(get_env_template()))
            env_layout.addWidget(copy_env_button)

        tab_widget.addTab(env_tab, "環境変数")

        # OAuth設定ガイドタブ
        oauth_tab = QWidget()
        oauth_layout = QVBoxLayout(oauth_tab)
        oauth_text = QPlainTextEdit()
        oauth_text.setReadOnly(True)
        if config_available:
            oauth_text.setPlainText(get_oauth_setup_guide())
        else:
            oauth_text.setPlainText(
                "Google Colab連携を有効にするには、以下の手順を実行してください:\n\n"
                "1. Google Cloud Consoleでプロジェクトを作成\n"
                "2. Google Drive APIを有効化\n"
                "3. OAuth 2.0クライアントIDを作成し、client_secrets.jsonをダウンロード\n"
                "4. 環境変数を設定:\n"
                "   COLAB_ENABLED=true\n"
                "   GOOGLE_CLIENT_SECRETS=path/to/client_secrets.json"
            )
        oauth_text.setMinimumHeight(200)
        oauth_layout.addWidget(oauth_text)
        tab_widget.addTab(oauth_tab, "OAuth設定ガイド")

        layout.addWidget(tab_widget)

        # 認証テストボタン
        if config_available and COLAB_ENABLED:
            test_button = QPushButton("接続テスト")
            test_button.clicked.connect(lambda: self._test_colab_connection(dialog))
            layout.addWidget(test_button)

        # 閉じるボタン
        close_button = QPushButton("閉じる")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.exec_()

    def _test_colab_connection(self, parent_dialog):
        """Google Drive接続テスト"""
        try:
            from utils.colab_transfer import ColabTransferManager

            # 認証が必要な場合の事前通知
            try:
                from config_colab import GOOGLE_CREDENTIALS_PATH
                import os
                if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
                    reply = QMessageBox.question(
                        parent_dialog,
                        "認証が必要",
                        "初回接続のため、ブラウザでGoogleアカウント認証が必要です。\n\n"
                        "ブラウザが開いたら、Googleアカウントを選択して認証を完了してください。\n"
                        "（タイムアウト: 60秒）\n\n"
                        "続行しますか？",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    if reply != QMessageBox.Yes:
                        return
            except ImportError:
                pass

            # 進捗表示
            parent_dialog.setEnabled(False)
            self.statusBar().showMessage("Google認証中... ブラウザで認証を完了してください（タイムアウト: 60秒）")
            QApplication.processEvents()

            manager = ColabTransferManager()
            success, message = manager.test_connection()

            parent_dialog.setEnabled(True)
            self.statusBar().clearMessage()

            if success:
                QMessageBox.information(parent_dialog, "接続テスト", message)
            else:
                QMessageBox.warning(parent_dialog, "接続テスト", message)

            # ステータスを更新
            self._update_colab_status_label()

        except ImportError as e:
            parent_dialog.setEnabled(True)
            self.statusBar().clearMessage()
            QMessageBox.critical(
                parent_dialog,
                "インポートエラー",
                f"必要なライブラリがインストールされていません:\n\n{str(e)}\n\n"
                "pip install pydrive2 google-auth google-auth-oauthlib pyyaml でインストールしてください。"
            )
        except TimeoutError as e:
            parent_dialog.setEnabled(True)
            self.statusBar().clearMessage()
            QMessageBox.warning(
                parent_dialog,
                "認証タイムアウト",
                f"{str(e)}\n\n"
                "ブラウザを閉じた場合や、認証に時間がかかりすぎた場合に発生します。\n"
                "再度「接続テスト」ボタンをクリックしてお試しください。"
            )
        except Exception as e:
            parent_dialog.setEnabled(True)
            self.statusBar().clearMessage()
            import traceback
            QMessageBox.critical(
                parent_dialog,
                "接続テストエラー",
                f"接続テスト中にエラーが発生しました:\n\n{str(e)}"
            )

    def toggle_dark_mode(self):
        """ダークモードを切り替える"""
        self.is_dark_mode = not self.is_dark_mode
        self.dark_mode_button.setChecked(self.is_dark_mode)
        self.apply_dark_mode(self.is_dark_mode)
        
        # 設定を保存
        if hasattr(self, 'width_spin') and hasattr(self, 'height_spin'):
            # 設定ダイアログが開いている場合
            self.save_display_settings(
                self.width_spin.value(), 
                self.height_spin.value(), 
                self.font().pointSize(),
                self.is_dark_mode
            )
        else:
            # 通常の場合
            self.save_display_settings(
                self.width(), 
                self.height(), 
                self.font().pointSize(),
                self.is_dark_mode
            )
    
    def apply_dark_mode(self, is_dark):
        """ダークモードのスタイルシートを適用"""
        if is_dark:
            # ダークモードのスタイル
            dark_style = """
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #606060;
            }
            QPushButton:checked {
                background-color: #0078d4;
                border-color: #106ebe;
            }
            QLabel {
                background-color: transparent;
                color: #ffffff;
            }
            QLineEdit {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
                color: #ffffff;
            }
            QComboBox {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
                color: #ffffff;
            }
            QComboBox QAbstractItemView {
                background-color: #404040;
                border: 1px solid #555555;
                selection-background-color: #0078d4;
                color: #ffffff;
            }
            QScrollArea {
                background-color: #2b2b2b;
                border: none;
            }
            QGroupBox {
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            QSpinBox {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
                color: #ffffff;
            }
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #404040;
                border: 1px solid #555555;
                border-bottom: none;
                padding: 6px;
                color: #ffffff;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            """
            self.setStyleSheet(dark_style)
            self.dark_mode_button.setText("ライトモード")
            
            # ラベルの色を明示的に更新
            if hasattr(self, 'idx_label'):
                self.idx_label.update()
            if hasattr(self, 'current_image_label'):
                self.current_image_label.update()
            if hasattr(self, 'current_image_info'):
                self.current_image_info.setStyleSheet("color: #ffffff; font-weight: bold;")
            if hasattr(self, 'graph_title'):
                self.graph_title.setStyleSheet("font-weight: bold; color: #ffffff;")
        else:
            # ライトモードのスタイル（デフォルト）
            self.setStyleSheet("")
            self.dark_mode_button.setText("ダークモード")
            
            # ラベルの色を明示的に更新
            if hasattr(self, 'idx_label'):
                self.idx_label.update()
            if hasattr(self, 'current_image_label'):
                self.current_image_label.update()
            if hasattr(self, 'current_image_info'):
                self.current_image_info.setStyleSheet("color: #333333; font-weight: bold;")
            if hasattr(self, 'graph_title'):
                self.graph_title.setStyleSheet("font-weight: bold; color: #333333;")

    def add_bbox_annotation(self, bbox):
        """バウンディングボックスアノテーションを追加"""
        if not self.images:
            return
        
        # インデックスベースに変更
        current_index = self.current_index
        
        if current_index not in self.bbox_annotations:
            self.bbox_annotations[current_index] = []
        
        self.bbox_annotations[current_index].append(bbox)
        
        ###
        print(bbox)

        # 前回のバウンディングボックスとして保存
        self.last_bbox = bbox.copy()
        
        # 現在のすべてのバウンディングボックスを保存
        self.last_bboxes = [box.copy() for box in self.bbox_annotations[current_index]]
        
        # 統計情報更新
        self.update_bbox_stats()
        
        # 画面更新
        self.main_image_view.update()
        self.update_gallery()

    def add_session_check_to_init_ui(self):
        """init_uiメソッドの最後に追加する初期セッション確認コード"""
        # メインウィンドウを最前面にアクティブ化
        self.show()
        self.raise_()
        self.activateWindow()
        
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
                # 確認ダイアログを表示（最前面表示）
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("前回のセッションを復元")
                msg_box.setText(f"前回の作業フォルダ（{len(valid_paths)}個）を読み込みますか？\n\n"
                              f"最初のフォルダ: {valid_paths[0]}\n" +
                              (f"他 {len(valid_paths)-1} フォルダ" if len(valid_paths) > 1 else ""))
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg_box.setDefaultButton(QMessageBox.Yes)
                msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                msg_box.activateWindow()
                msg_box.raise_()
                reply = msg_box.exec_()
                
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
                # 確認ダイアログを表示（最前面表示）
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("前回のセッションを復元")
                msg_box.setText(f"前回の作業フォルダを読み込みますか？\n\nフォルダ: {last_folder}")
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg_box.setDefaultButton(QMessageBox.Yes)
                msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                msg_box.activateWindow()
                msg_box.raise_()
                reply = msg_box.exec_()
                
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
            # self.model_refresh_button,         # モデル一覧更新ボタン
            # 推論結果表示チェックボックスは除外（モデル読み込み状態で制御）
        ]
        
        # 検索してボタン追加（UIから見つける方法）
        additional_buttons = []
        for button in self.findChildren(QPushButton):
            # ボタンのテキストで判断
            button_text = button.text()
            if any(keyword in button_text for keyword in [
                "Donkeycar", "Jetracer", "アノテーション動画作成", 
                "オートアノテーション実行", "一括推論実行",
                "モデルを学習・保存", "全画像を推論"
            ]):
                additional_buttons.append(button)
        
        # すべてのボタンリストを統合
        all_buttons = annotation_buttons + additional_buttons
        
        # ボタンの有効/無効を設定
        for button in all_buttons:
            if button:  # Noneでない場合のみ設定
                # 全画像推論ボタンは除外（モデル依存制御）
                if hasattr(self, 'batch_inference_button') and button == self.batch_inference_button:
                    continue
                # オートアノテーションボタンは除外（モデル依存制御）
                if hasattr(self, 'auto_annotate_button') and button == self.auto_annotate_button:
                    continue
                if hasattr(self, 'yolo_auto_annotate_btn') and button == self.yolo_auto_annotate_btn:
                    continue
                button.setEnabled(enabled)
        
        # ボタンの色も状態に応じて変更
        button_style = "" if enabled else "QPushButton:disabled { color: #aaaaaa; }"
        for button in all_buttons:
            if button and not isinstance(button, QCheckBox):  # チェックボックス以外のボタンにスタイル適用
                # 全画像推論ボタンは除外（モデル依存制御）
                if hasattr(self, 'batch_inference_button') and button == self.batch_inference_button:
                    continue
                # オートアノテーションボタンは除外（モデル依存制御）
                if hasattr(self, 'auto_annotate_button') and button == self.auto_annotate_button:
                    continue
                if hasattr(self, 'yolo_auto_annotate_btn') and button == self.yolo_auto_annotate_btn:
                    continue
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
        """現在表示中の画像を削除済みとしてマークする（アノテーションの有無を問わない）"""
        if not self.images:
            return
                    
        current_img_path = self.images[self.current_index]
        
        # 確認ダイアログ
        reply = QMessageBox.question(
            self, 
            "削除確認", 
            f"現在の画像（インデックス: {self.current_index}）を削除済みとしてマークしますか？\n"
            f"ファイル: {os.path.basename(current_img_path)}",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # 削除したインデックスを記録（統合リスト上のactual_indexを使用）
        self.deleted_indexes.append(self.current_index)
            
        # 削除したインデックスをソートして保持（重複を排除）
        self.deleted_indexes = sorted(list(set(self.deleted_indexes)))

        # UI更新 - 重い処理は遅延実行
        self.display_current_image()
        self._schedule_gallery_update()  # 遅延更新
        if hasattr(self, 'update_location_button_counts'):
            self.update_location_button_counts()
        self._schedule_distribution_graph_update()  # 遅延更新
        self.update_slider_deleted_indexes()
        
        QMessageBox.information(
                self, 
                "削除完了", 
                f"インデックス {self.current_index} を削除済みとしてマークしました。\n"
                f"アノテーションデータは保持されています。\n"
                f"\n削除済みインデックス数: {len(self.deleted_indexes)}"
            )

    def delete_clip_range(self):
        """指定範囲の画像を削除済みとしてマークする（アノテーションの有無を問わない）"""
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
        
        # 範囲内の画像数をカウント
        target_paths = self.images[start_idx:end_idx+1]
        
        # 範囲内のアノテーション数をカウント（情報表示用）
        annotations_in_range = sum(1 for idx in range(start_idx, end_idx + 1) if idx in self.annotations)

        # 確認ダイアログ
        reply = QMessageBox.question(
            self, 
            "範囲削除確認", 
            f"インデックス {start_idx} から {end_idx} までの"
            f"\n{len(target_paths)}個の画像を削除済みとしてマークします。"
            f"\n（このうち{annotations_in_range}個には既にアノテーションがあります）"
            f"\n\nこの操作は「復元」ボタンで元に戻せます。続行しますか？",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # 削除済みとして登録するインデックスのリスト
        indices_to_delete = list(range(start_idx, end_idx + 1))
        
        # アノテーション削除カウント用
        marked_as_deleted_count = 0
        
        # 各インデックスを処理
        for idx in indices_to_delete:
            # すでに削除済みの場合はスキップ
            if idx in self.deleted_indexes:
                continue

            # 削除したインデックスを記録（統合リスト上のactual_indexを使用）
            self.deleted_indexes.append(idx)
            marked_as_deleted_count += 1

        # 削除したインデックスをソートして重複を排除
        self.deleted_indexes = sorted(list(set(self.deleted_indexes)))

        # アノテーション数を更新
        self.annotated_count = len(self.annotations)

        # UI更新 - 重い処理は遅延実行
        self.display_current_image()
        self._schedule_gallery_update()  # 遅延更新
        self.update_location_button_counts()
        self._schedule_distribution_graph_update()  # 遅延更新
        self.update_slider_deleted_indexes()
        
        QMessageBox.information(
            self,
            "範囲削除完了",
            f"インデックス {start_idx} から {end_idx} までの範囲から"
            f"\n{marked_as_deleted_count}個の画像を削除済みとしてマークしました。"
            f"\nアノテーションデータは保持されています。"
            f"\n\n削除済みインデックスの合計数: {len(self.deleted_indexes)}"
        )

    def detect_downsampling_targets(self):
        """直進時（angle値が一定範囲内で連続）のデータを検出してダウンサンプリング対象に設定"""
        if not self.images or not self.annotations:
            QMessageBox.warning(self, "警告", "画像とアノテーションデータを読み込んでください。")
            return

        # パラメータ取得
        angle_min = self.downsample_angle_min.value()
        angle_max = self.downsample_angle_max.value()
        min_consecutive = self.downsample_consecutive.value()
        keep_every = self.downsample_keep_every.value()

        if angle_min >= angle_max:
            QMessageBox.warning(self, "警告", "angle範囲の最小値は最大値より小さくしてください。")
            return

        # 連続区間の検出
        consecutive_runs = []  # [(start_idx, end_idx), ...]
        current_run_start = None

        # インデックス順にソートしてチェック
        sorted_indices = sorted(self.annotations.keys())

        for i, idx in enumerate(sorted_indices):
            ann = self.annotations[idx]
            angle = ann.get('angle')

            # 削除済みはスキップ
            if idx in self.deleted_indexes:
                if current_run_start is not None:
                    # 連続区間終了
                    run_end = sorted_indices[i - 1] if i > 0 else current_run_start
                    if run_end - current_run_start + 1 >= min_consecutive:
                        consecutive_runs.append((current_run_start, run_end))
                    current_run_start = None
                continue

            if angle is not None and angle_min <= angle <= angle_max:
                # 範囲内
                if current_run_start is None:
                    current_run_start = idx
            else:
                # 範囲外 - 連続区間終了
                if current_run_start is not None:
                    run_end = sorted_indices[i - 1] if i > 0 else current_run_start
                    if run_end - current_run_start + 1 >= min_consecutive:
                        consecutive_runs.append((current_run_start, run_end))
                    current_run_start = None

        # 最後の連続区間をチェック
        if current_run_start is not None:
            run_end = sorted_indices[-1]
            if run_end - current_run_start + 1 >= min_consecutive:
                consecutive_runs.append((current_run_start, run_end))

        # ダウンサンプリング対象を決定（連続区間内でkeep_every枚ごとに1枚を残す、0なら全て対象）
        new_downsampled = []
        for run_start, run_end in consecutive_runs:
            count = 0
            for idx in range(run_start, run_end + 1):
                if idx in self.deleted_indexes:
                    continue
                if idx not in self.annotations:
                    continue
                count += 1
                if keep_every == 0:
                    # 間隔0: 全てダウンサンプリング対象
                    new_downsampled.append(idx)
                else:
                    # keep_every枚ごとに1枚残す（1, keep_every+1, 2*keep_every+1, ...を残す）
                    if count % keep_every != 1:
                        new_downsampled.append(idx)

        # 既存のダウンサンプリング対象と統合
        self.downsampled_indexes = sorted(list(set(self.downsampled_indexes + new_downsampled)))

        # UI更新
        self.update_slider_downsampled_indexes()
        self.downsample_count_label.setText(f"{len(self.downsampled_indexes)}件")

        # angle検出ボタンを「再検出」に変更し水色にする
        redetect_style = """
            QPushButton {
                background-color: #5bc0de;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }
            QPushButton:hover {
                background-color: #46b8da;
            }
        """
        if hasattr(self, 'detect_downsample_button'):
            self.detect_downsample_button.setText("再検出")
            self.detect_downsample_button.setStyleSheet(redetect_style)

        # 分布グラフを更新
        if hasattr(self, 'update_distribution_graph'):
            self.update_distribution_graph()

        # 現在の画像表示を更新（DS対象バッジ表示のため）
        if hasattr(self, '_set_annotation_point_on_canvas'):
            self._set_annotation_point_on_canvas()

        # 分析ウィンドウが開いている場合は更新
        if hasattr(self, 'data_analysis_dialog') and self.data_analysis_dialog is not None:
            if self.data_analysis_dialog.isVisible():
                self.data_analysis_dialog.update_analysis()

        # 結果表示
        QMessageBox.information(
            self,
            "ダウンサンプリング検出完了",
            f"検出条件:\n"
            f"・angle範囲: {angle_min:.2f} 〜 {angle_max:.2f}\n"
            f"・連続フレーム数: {min_consecutive}以上\n"
            f"・残す間隔: {keep_every}枚に1枚\n\n"
            f"検出された連続区間: {len(consecutive_runs)}箇所\n"
            f"ダウンサンプリング対象: {len(self.downsampled_indexes)}件"
        )

    def detect_throttle_downsampling_targets(self):
        """throttle値が一定範囲内で連続するデータを検出してダウンサンプリング対象に設定"""
        if not self.images or not self.annotations:
            QMessageBox.warning(self, "警告", "画像とアノテーションデータを読み込んでください。")
            return

        # パラメータ取得
        throttle_min = self.downsample_throttle_min.value()
        throttle_max = self.downsample_throttle_max.value()
        min_consecutive = self.downsample_throttle_consecutive.value()
        keep_every = self.downsample_throttle_keep_every.value()

        if throttle_min >= throttle_max:
            QMessageBox.warning(self, "警告", "throttle範囲の最小値は最大値より小さくしてください。")
            return

        # 連続区間の検出
        consecutive_runs = []  # [(start_idx, end_idx), ...]
        current_run_start = None

        # インデックス順にソートしてチェック
        sorted_indices = sorted(self.annotations.keys())

        for i, idx in enumerate(sorted_indices):
            ann = self.annotations[idx]
            throttle = ann.get('throttle')

            # 削除済みはスキップ
            if idx in self.deleted_indexes:
                if current_run_start is not None:
                    # 連続区間終了
                    run_end = sorted_indices[i - 1] if i > 0 else current_run_start
                    if run_end - current_run_start + 1 >= min_consecutive:
                        consecutive_runs.append((current_run_start, run_end))
                    current_run_start = None
                continue

            if throttle is not None and throttle_min <= throttle <= throttle_max:
                # 範囲内
                if current_run_start is None:
                    current_run_start = idx
            else:
                # 範囲外 - 連続区間終了
                if current_run_start is not None:
                    run_end = sorted_indices[i - 1] if i > 0 else current_run_start
                    if run_end - current_run_start + 1 >= min_consecutive:
                        consecutive_runs.append((current_run_start, run_end))
                    current_run_start = None

        # 最後の連続区間をチェック
        if current_run_start is not None:
            run_end = sorted_indices[-1]
            if run_end - current_run_start + 1 >= min_consecutive:
                consecutive_runs.append((current_run_start, run_end))

        # ダウンサンプリング対象を決定（連続区間内でkeep_every枚ごとに1枚を残す、0なら全て対象）
        new_downsampled = []
        for run_start, run_end in consecutive_runs:
            count = 0
            for idx in range(run_start, run_end + 1):
                if idx in self.deleted_indexes:
                    continue
                if idx not in self.annotations:
                    continue
                count += 1
                if keep_every == 0:
                    # 間隔0: 全てダウンサンプリング対象
                    new_downsampled.append(idx)
                else:
                    # keep_every枚ごとに1枚残す（1, keep_every+1, 2*keep_every+1, ...を残す）
                    if count % keep_every != 1:
                        new_downsampled.append(idx)

        # 既存のダウンサンプリング対象と統合
        self.downsampled_indexes = sorted(list(set(self.downsampled_indexes + new_downsampled)))

        # UI更新
        self.update_slider_downsampled_indexes()
        self.downsample_count_label.setText(f"{len(self.downsampled_indexes)}件")
        self.throttle_downsample_count_label.setText(f"(+{len(new_downsampled)}件)")

        # throttle検出ボタンを「再検出」に変更し水色にする
        redetect_style = """
            QPushButton {
                background-color: #5bc0de;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }
            QPushButton:hover {
                background-color: #46b8da;
            }
        """
        if hasattr(self, 'detect_throttle_downsample_button'):
            self.detect_throttle_downsample_button.setText("再検出")
            self.detect_throttle_downsample_button.setStyleSheet(redetect_style)

        # 分布グラフを更新
        if hasattr(self, 'update_distribution_graph'):
            self.update_distribution_graph()

        # 現在の画像表示を更新（DS対象バッジ表示のため）
        if hasattr(self, '_set_annotation_point_on_canvas'):
            self._set_annotation_point_on_canvas()

        # 分析ウィンドウが開いている場合は更新
        if hasattr(self, 'data_analysis_dialog') and self.data_analysis_dialog is not None:
            if self.data_analysis_dialog.isVisible():
                self.data_analysis_dialog.update_analysis()

        # 結果表示
        QMessageBox.information(
            self,
            "Throttleダウンサンプリング検出完了",
            f"検出条件:\n"
            f"・throttle範囲: {throttle_min:.2f} 〜 {throttle_max:.2f}\n"
            f"・連続フレーム数: {min_consecutive}以上\n"
            f"・残す間隔: {keep_every}枚に1枚\n\n"
            f"検出された連続区間: {len(consecutive_runs)}箇所\n"
            f"今回追加: {len(new_downsampled)}件\n"
            f"ダウンサンプリング対象合計: {len(self.downsampled_indexes)}件"
        )

    def clear_downsampling_targets(self):
        """ダウンサンプリング対象をすべて解除"""
        if not self.downsampled_indexes:
            QMessageBox.information(self, "情報", "ダウンサンプリング対象はありません。")
            return

        reply = QMessageBox.question(
            self,
            "確認",
            f"{len(self.downsampled_indexes)}件のダウンサンプリング対象をすべて解除しますか？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.downsampled_indexes = []
            self.update_slider_downsampled_indexes()
            self.downsample_count_label.setText("0件")
            if hasattr(self, 'throttle_downsample_count_label'):
                self.throttle_downsample_count_label.setText("(0件)")

            # 検出ボタンを「検出」に戻し青色にリセット
            detect_style = """
                QPushButton {
                    background-color: #4a90d9;
                    color: white;
                    font-weight: bold;
                    border-radius: 4px;
                    padding: 6px 12px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #3a7fc8;
                }
            """
            if hasattr(self, 'detect_downsample_button'):
                self.detect_downsample_button.setText("検出")
                self.detect_downsample_button.setStyleSheet(detect_style)
            if hasattr(self, 'detect_throttle_downsample_button'):
                self.detect_throttle_downsample_button.setText("検出")
                self.detect_throttle_downsample_button.setStyleSheet(detect_style)

            # 分布グラフを更新
            if hasattr(self, 'update_distribution_graph'):
                self.update_distribution_graph()

            # 現在の画像表示を更新（DS対象バッジ表示解除のため）
            if hasattr(self, '_set_annotation_point_on_canvas'):
                self._set_annotation_point_on_canvas()

            # 分析ウィンドウが開いている場合は更新
            if hasattr(self, 'data_analysis_dialog') and self.data_analysis_dialog is not None:
                if self.data_analysis_dialog.isVisible():
                    self.data_analysis_dialog.update_analysis()

            QMessageBox.information(self, "完了", "ダウンサンプリング対象を解除しました。")

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
                
        # アノテーションデータの検索と読み込みを実行
        annotations_loaded = False
        
        try:
            # 読み込み前に既存のデータをクリア（安全のため）
            if self.annotations:
                self.clear_annotations()
                
            # フォルダ直下（imagesフォルダと同じ階層）のマニフェストファイルを確認
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
            self.display_current_image()
            self.update_gallery()
            self.update_slider_deleted_indexes()
            
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
            # waypointモードの場合、現在の画像のwaypoint数をチェック
            if not self._check_waypoint_count_before_transition():
                # チェックに失敗した場合、スライダーを元の位置に戻す
                self.image_slider.blockSignals(True)
                self.image_slider.setValue(self.current_index)
                self.image_slider.blockSignals(False)
                return

            # 画像を移動する前に、作成途中のアノテーション頂点をクリア
            self.clear_incomplete_annotations(show_message=True)

            # スライダー移動前に現在の画像のwaypoint情報を保存
            old_index = self.current_index
            if (old_index is not None and
                old_index in self.waypoint_annotations and
                self.waypoint_annotations[old_index]):
                self.last_waypoints = self.waypoint_annotations[old_index].copy()

            self.current_index = value

            # 前回waypoint自動適用機能
            if (hasattr(self, 'auto_apply_last_waypoint') and
                self.auto_apply_last_waypoint and
                self.last_waypoints and
                hasattr(self, 'current_mode') and
                self.current_mode == 3):  # waypointモードの場合のみ

                # 新しい画像にwaypointがない場合のみ適用
                if (self.current_index not in self.waypoint_annotations or
                    not self.waypoint_annotations[self.current_index]):

                    self.waypoint_annotations[self.current_index] = self.last_waypoints.copy()

                    # ステータスメッセージ
                    if hasattr(self, 'statusBar'):
                        self.statusBar().showMessage(f"前回のwaypoint {len(self.last_waypoints)}個を自動適用しました", 2000)

            self.display_current_image()

            # タイマーを再開して推論実行をデバウンス
            self.inference_debounce_timer.stop()
            self.inference_debounce_timer.start(self.inference_debounce_delay)

            self.update_ui()

            # データ分析ダイアログが開いている場合、現在位置を更新
            if hasattr(self, 'data_analysis_dialog') and self.data_analysis_dialog is not None:
                if self.data_analysis_dialog.isVisible():
                    self.data_analysis_dialog.update_current_position(value)

    def execute_slider_inference(self):
        """スライダー変更後のデバウンス処理で推論を実行"""
        if not self.images:
            return

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

        # セグメンテーション推論表示の更新
        if hasattr(self, 'segmentation_inference_checkbox') and self.segmentation_inference_checkbox.isChecked():
            current_img_path = self.images[self.current_index]
            # 推論結果がまだない場合のみ推論を実行
            if current_img_path not in self.segmentation_inference_results:
                self.run_single_yolo_segmentation_inference()
            else:
                # 既にある結果の表示を更新
                self.update_segmentation_inference_display()

    def toggle_future_annotation_display(self, state):
        """将来アノテーション表示の切り替え"""
        show_future = (state == Qt.Checked)
        self.main_image_view.show_future_annotations = show_future
        self.main_image_view.update()

        if show_future:
            self.statusBar().showMessage("将来アノテーション表示をオンにしました", 3000)
        else:
            self.statusBar().showMessage("将来アノテーション表示をオフにしました", 3000)

    def toggle_gradcam_display(self, state):
        """CAM表示の切り替え"""
        show_gradcam = (state == Qt.Checked)
        self.main_image_view.show_gradcam = show_gradcam

        if show_gradcam:
            # CAMを生成して表示
            self.update_gradcam_visualization()
            self.statusBar().showMessage("CAM表示をオンにしました", 3000)
        else:
            # CAMオーバーレイをクリア
            self.main_image_view.gradcam_overlay = None
            self.main_image_view.update()
            self.statusBar().showMessage("CAM表示をオフにしました", 3000)

    def change_gradcam_target(self, target):
        """CAM対象出力の変更"""
        self.main_image_view.gradcam_target = target

        # CAM表示中なら更新
        if self.main_image_view.show_gradcam:
            self.update_gradcam_visualization()

    def change_gradcam_method(self, method):
        """CAM手法の変更"""
        self.main_image_view.gradcam_method = method

        # CAM表示中なら更新
        if self.main_image_view.show_gradcam:
            self.update_gradcam_visualization()

    def change_gradcam_direction(self, direction):
        """勾配方向の変更"""
        self.main_image_view.gradcam_direction = direction

        # CAM表示中なら更新
        if self.main_image_view.show_gradcam:
            self.update_gradcam_visualization()

    def update_gradcam_visualization(self):
        """CAM可視化を更新"""
        if not self.images or not hasattr(self, 'model') or self.model is None:
            return

        try:
            from utils.gradcam_utils import GradCAM, apply_colormap, generate_bidirectional_cam
            import cv2

            # 現在の画像パス
            img_path = self.images[self.current_index]

            # 画像を読み込み
            original_image = Image.open(img_path).convert('RGB')
            original_np = np.array(original_image)

            # 前処理
            transform = self.model.get_preprocess()
            input_tensor = transform(original_image).unsqueeze(0)

            # デバイスに移動
            device = self.model.device if hasattr(self.model, 'device') else torch.device('cpu')
            input_tensor = input_tensor.to(device)

            # CAMインスタンス作成
            gradcam = GradCAM(self.model)

            try:
                # 対象出力のインデックスを取得
                target_map = {'angle': 0, 'throttle': 1, 'speed': 2}
                target_idx = target_map.get(self.main_image_view.gradcam_target, 0)

                # CAM手法を取得
                cam_method = getattr(self.main_image_view, 'gradcam_method', 'gradcam')

                # 勾配方向を取得
                cam_direction = getattr(self.main_image_view, 'gradcam_direction', 'both')

                # 画像サイズ
                h, w = original_np.shape[:2]

                if cam_direction == 'both':
                    # 正負両方向を赤青で可視化
                    positive_heatmap, negative_heatmap, combined_rgb = generate_bidirectional_cam(
                        gradcam,
                        input_tensor,
                        target_output_index=target_idx,
                        method=cam_method
                    )

                    # ヒートマップをリサイズ
                    combined_rgb_resized = cv2.resize(combined_rgb, (w, h))
                    positive_resized = cv2.resize(positive_heatmap, (w, h))
                    negative_resized = cv2.resize(negative_heatmap, (w, h))

                    # RGBからRGBAに変換
                    heatmap_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    heatmap_rgba[:, :, :3] = combined_rgb_resized

                    # アルファチャンネル: 正負どちらかの強度が高い部分ほど不透明に
                    combined_intensity = np.maximum(positive_resized, negative_resized)
                    alpha_value = int(255 * self.main_image_view.gradcam_alpha)
                    heatmap_rgba[:, :, 3] = (combined_intensity * alpha_value).astype(np.uint8)

                else:
                    # 単一方向（従来の処理）
                    heatmap = gradcam.generate_cam(
                        input_tensor,
                        target_output_index=target_idx,
                        method=cam_method,
                        direction=cam_direction
                    )

                    # ヒートマップを画像サイズにリサイズ
                    heatmap_resized = cv2.resize(heatmap, (w, h))

                    # カラーマップを適用（BGRで返る）
                    heatmap_colored = apply_colormap(heatmap_resized, cv2.COLORMAP_JET)
                    # BGRからRGBAに変換（アルファチャンネル付き）
                    heatmap_rgba = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGBA)

                    # アルファチャンネルをヒートマップの強度に基づいて設定
                    alpha_value = int(255 * self.main_image_view.gradcam_alpha)
                    heatmap_rgba[:, :, 3] = (heatmap_resized * alpha_value).astype(np.uint8)

                # QPixmapに変換（RGBA）
                bytes_per_line = 4 * w
                q_image = QImage(heatmap_rgba.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
                self.main_image_view.gradcam_overlay = QPixmap.fromImage(q_image.copy())

                # 画面を更新
                self.main_image_view.update()

            finally:
                # フックを削除してメモリリークを防ぐ
                gradcam.remove_hooks()

        except Exception as e:
            print(f"CAM生成エラー: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"CAM生成エラー: {e}", 5000)

    def toggle_inference_display(self, state):
        """自動運転推論表示の切り替え"""
        show_inference = (state == Qt.Checked)
        self.main_image_view.show_inference = show_inference
        
        # 画面更新
        #if hasattr(self, 'main_image_view'):
        self.main_image_view.update()
        
        # 表示情報の更新
        if show_inference:
            self.update_inference_display()
            self.statusBar().showMessage("自動運転推論結果表示をオンにしました", 3000)
        else:
            # 表示をクリア
            if hasattr(self, 'inference_info_label'):
                self.inference_info_label.setText(" ")  # スペースで高さを維持
            self.statusBar().showMessage("自動運転推論結果表示をオフにしました", 3000)
        
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
            # models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
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
            
            # 推論結果を保存（インデックスベースに変換）
            old_count = len(self.inference_results)
            
            # 画像パスからインデックスに変換して保存
            for img_path, result in inference_results.items():
                # 画像パスから対応するインデックスを取得
                try:
                    img_index = self.images.index(img_path)
                    self.inference_results[img_index] = result
                    print(f"推論結果保存: インデックス{img_index} <- {os.path.basename(img_path)}")
                    
                    # 差分ベクトルの計算と保存
                    self.calculate_and_store_diff_vector(img_index)
                    
                except ValueError:
                    print(f"警告: 画像パス {img_path} がself.imagesに見つかりません")
                    # パスでも保存（後方互換性のため）
                    self.inference_results[img_path] = result
            
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
            import traceback
            traceback.print_exc()  # エラーの詳細を表示
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
            batch_size = 50
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
                    # インデックスベースでチェック
                    batch_to_process = []
                    for img_path in current_batch:
                        try:
                            img_index = self.images.index(img_path)
                            if img_index not in self.inference_results:
                                batch_to_process.append(img_path)
                            else:
                                skipped_count += 1
                        except ValueError:
                            # インデックスが見つからない場合は処理対象に含める
                            batch_to_process.append(img_path)
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
                    
                    # 結果をインデックスベースで保存（ここが重要な修正点）
                    for img_path, result in batch_results.items():
                        # 画像パスから対応するインデックスを取得
                        try:
                            img_index = self.images.index(img_path)
                            self.inference_results[img_index] = result
                            success_count += 1
                            print(f"推論結果保存: インデックス{img_index} <- {os.path.basename(img_path)}")
                            
                            # 差分ベクトルの計算と保存
                            self.calculate_and_store_diff_vector(img_index)
                            
                        except ValueError:
                            print(f"警告: 画像パス {img_path} がself.imagesに見つかりません")
                            # パスでも保存（後方互換性のため）
                            self.inference_results[img_path] = result
                            success_count += 1
                    
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
            # models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
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
            
            # 推論結果を保存（インデックスベースに変換）
            old_count = len(self.location_inference_results)
            for img_path, result in inference_results.items():
                # 画像パスからインデックスを取得
                try:
                    img_index = self.images.index(img_path)
                except ValueError:
                    continue  # 画像がリストにない場合はスキップ
                    
                # 結果から位置情報を抽出
                if "pilot/loc" in result:
                    location = result["pilot/loc"]
                elif "loc" in result:
                    location = result["loc"]
                else:
                    # 位置情報がない場合、何らかのロジックで判断（例：角度から推定）
                    location = estimate_location_from_angle(result.get("angle", 0))
                
                # 位置情報付きの結果を保存（インデックスベース）
                self.location_inference_results[img_index] = {
                    "loc": location,
                    "x": result.get("x", 0),
                    "y": result.get("y", 0)
                }
            
            new_count = len(self.location_inference_results)
            
            # 位置推論表示チェックボックスを自動的にオンにする
            was_checked = self.location_inference_checkbox.isChecked()
            self.location_inference_checkbox.setChecked(True)
            
            # 表示を更新
            self.update_ui()

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
                
        current_index = self.current_index
        
        # 自動運転推論表示がOFFの場合は表示をクリア
        if not hasattr(self, 'inference_checkbox') or not self.inference_checkbox.isChecked():
            if hasattr(self, 'inference_info_label'):
                self.inference_info_label.setText(" ")  # スペースで高さを維持
            
            # 推論ポイントをクリア
            if hasattr(self, 'main_image_view'):
                self.main_image_view.inference_point = None
            
            return
                
        # 推論結果がある場合、表示を更新（インデックスベースで探す）
        inference = None
        if current_index in self.inference_results:
            inference = self.inference_results[current_index]
        
        if inference:
            # 新しいキー形式があればそれを使い、なければ古い形式を使う
            if "pilot/angle" in inference and "pilot/throttle" in inference:
                angle = inference["pilot/angle"]
                throttle = inference["pilot/throttle"]
            else:
                angle = inference["angle"]
                throttle = inference["throttle"]

            # 推論情報のリッチテキスト
            inference_text = f"<b>推論結果:</b><br>"
            inference_text += f"angle = <span style='color: #009999;'>{angle:.4f}</span><br>"
            inference_text += f"throttle = <span style='color: #009999;'>{throttle:.4f}</span>"


            # 追加: 差分ベクトルの計算と表示
            if current_index in self.annotations:
                self.calculate_and_store_diff_vector(current_index)
                
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
            # 推論結果がない場合はクリア（スペースで高さを維持）
            self.inference_info_label.setText(" ")
            self.main_image_view.inference_point = None
        
        # 推論表示のチェック状態を反映
        self.main_image_view.show_inference = self.inference_checkbox.isChecked()

    ### 読み込み関連         
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
        if not dialog.exec_():
            # キャンセルまたは×ボタンが押された場合は何もせずに終了
            return

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
                # 有効なフォルダがある場合のみ画像読み込みを実行
                # 少し遅延させてから画像読み込みを実行（UIが更新される時間を確保）
                QTimer.singleShot(100, self.load_images)
            else:
                self.folder_input.setText("")
                QMessageBox.critical(
                    self,
                    "エラー",
                    "選択されたフォルダのいずれにもimagesフォルダが含まれていません。\n処理を中止します。"
                )
                return

    def load_images(self):
        """
        選択した各フォルダの下のimagesフォルダから画像を読み込む
        アノテーションは自動では読み込まない
        Jetracer形式のファイル名にも対応（修正版）
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
        
        # プログレスダイアログを作成
        progress = QProgressDialog("フォルダを読み込み中...", "キャンセル", 0, 100, self)
        progress.setWindowTitle("読み込み進捗")
        progress.setModal(True)
        progress.show()
        
        # 全画像フォルダの画像を集める（各フォルダごとにソートしてから連結）
        all_images = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        print(f"{len(image_folders)}個のimagesフォルダを検索中...")
        progress.setLabelText(f"{len(image_folders)}個のフォルダを検索中...")
        progress.setValue(10)
        QApplication.processEvents()

        # 各フォルダを順次処理し、各フォルダ内でソートしてから連結
        for folder_idx, img_folder in enumerate(image_folders):
            if progress.wasCanceled():
                return

            folder_name = os.path.basename(os.path.dirname(img_folder))
            progress.setLabelText(f"フォルダ '{folder_name}' を読み込み中... ({folder_idx + 1}/{len(image_folders)})")
            progress.setValue(10 + (folder_idx * 70 // len(image_folders)))
            QApplication.processEvents()

            print(f"画像フォルダを検索中: {img_folder}")

            # このフォルダの画像を集める
            folder_images = []
            try:
                files = os.listdir(img_folder)
                for file_idx, file in enumerate(files):
                    if progress.wasCanceled():
                        return

                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        folder_images.append(os.path.join(img_folder, file))

                    # ファイル処理の進捗を細かく更新
                    if file_idx % 10 == 0:  # 10ファイルごとに更新
                        file_progress = 10 + (folder_idx * 70 // len(image_folders)) + (file_idx * 10 // len(files) if len(files) > 0 else 0)
                        progress.setValue(min(file_progress, 80))
                        QApplication.processEvents()

            except Exception as e:
                print(f"画像フォルダ {img_folder} の読み込みエラー: {e}")
                continue

            # このフォルダの画像をファイル名のインデックスでソート
            if folder_images:
                print(f"フォルダ '{folder_name}': {len(folder_images)}枚の画像をソート中...")
                folder_image_with_indices = []

                for img_path in folder_images:
                    basename = os.path.basename(img_path)
                    # ファイル名からインデックスを抽出
                    try:
                        # Jetracer形式を優先的にチェック: x_y_index_cam_image_array_.jpg -> index
                        # 例: 200_100_2_cam_image_array_.jpg -> 2
                        jetracer_match = re.match(r'^\d+_\d+_(\d+)_', basename)
                        if jetracer_match:
                            index = int(jetracer_match.group(1))
                            folder_image_with_indices.append((img_path, index))
                        else:
                            # 通常形式: 10900_cam_image_array_.jpg -> 10900
                            normal_match = re.match(r'^(\d+)_', basename)
                            if normal_match:
                                index = int(normal_match.group(1))
                                folder_image_with_indices.append((img_path, index))
                            else:
                                # インデックスが抽出できない場合は、ファイル名でソート
                                folder_image_with_indices.append((img_path, basename))
                    except Exception as e:
                        print(f"ファイル名からインデックス抽出エラー: {basename} - {e}")
                        # エラーの場合はファイル名でソート
                        folder_image_with_indices.append((img_path, basename))

                # このフォルダの画像をソート
                folder_image_with_indices.sort(key=lambda x: x[1])
                sorted_folder_images = [img_path for img_path, _ in folder_image_with_indices]

                # フォルダの順番を維持して連結
                all_images.extend(sorted_folder_images)
                print(f"フォルダ '{folder_name}': ソート完了、全体に追加 (現在の合計: {len(all_images)}枚)")

        if progress.wasCanceled():
            return

        progress.setLabelText("画像ファイルを確認中...")
        progress.setValue(85)
        QApplication.processEvents()

        if not all_images:
            progress.close()
            QMessageBox.warning(self, "エラー", "選択されたフォルダ内のimagesフォルダに画像ファイルがありません。")
            return

        print(f"全{len(image_folders)}フォルダから合計{len(all_images)}枚の画像を読み込みました（フォルダ順序維持）")

        # ソート済みの画像パスリストを作成（既に各フォルダ内でソート済み＋フォルダ順で連結済み）
        images = all_images

        # --- 画像グルーピング（インデックス＆キー単位） ---
        progress.setLabelText("画像データを整理中...")
        progress.setValue(90)
        QApplication.processEvents()
        
        self.image_groups = {}  # { index: { variant: path, ... } }
        self.variant_images = {}  # 各キーの画像リスト

        for img_idx, img_path in enumerate(images):
            if progress.wasCanceled():
                return
            basename = os.path.basename(img_path)
            
            # Jetracer形式を優先的にチェック: 200_100_2_cam_image_array_.jpg
            jetracer_match = re.match(r'^\d+_\d+_(\d+)_([A-Za-z0-9]+)_image_array', basename)
            if jetracer_match:
                # Jetracer形式の場合
                idx = int(jetracer_match.group(1))
                variant = jetracer_match.group(2)
                print(f"Jetracer形式グルーピング: {basename} -> インデックス {idx}, バリアント {variant}")
            else:
                # 通常形式: 10900_cam_image_array_.jpg
                normal_match = re.match(r'^(\d+)_([A-Za-z0-9]+)_image_array', basename)
                if normal_match:
                    # 通常形式の場合
                    idx = int(normal_match.group(1))
                    variant = normal_match.group(2)
                    print(f"通常形式グルーピング: {basename} -> インデックス {idx}, バリアント {variant}")
                else:
                    # どちらにもマッチしない場合はスキップまたはデフォルト値を使用
                    print(f"警告: ファイル名パターンにマッチしません: {basename}")
                    continue
            
            # 画像グループに追加
            self.image_groups.setdefault(idx, {})[variant] = img_path
        
            # キー別に画像リストを作成
            if variant not in self.variant_images:
                self.variant_images[variant] = []
            self.variant_images[variant].append(img_path)
        
        self.sorted_indices = sorted(self.image_groups.keys())

        # キー一覧を更新
        self.available_variants = sorted(self.variant_images.keys())
        
        # available_variantsが空の場合のエラーハンドリングを追加
        if not self.available_variants:
            # バリアントが見つからない場合、全画像を'unknown'キーとして処理
            print("警告: 有効なバリアントが見つかりません。全画像を'unknown'キーとして処理します。")
            self.available_variants = ['unknown']
            self.variant_images = {'unknown': images}
            self.current_variant = 'unknown'
        else:
            # 新しいフォルダを読み込んだときは常にcamを優先
            if 'cam' in self.available_variants:
                # camが存在すれば最優先で選択
                self.current_variant = 'cam'
                print(f"[INFO] 新しいフォルダ読み込み: 'cam' バリアントを選択")
            else:
                # camがなければ最初のキー
                self.current_variant = self.available_variants[0]
                print(f"[INFO] 新しいフォルダ読み込み: '{self.current_variant}' バリアントを選択 (cam なし)")

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
        
        # 最終的な処理
        progress.setLabelText("画面を更新中...")
        progress.setValue(95)
        QApplication.processEvents()
        
        # Update UI
        self.display_current_image()
        self.update_gallery()
        self.update_slider_deleted_indexes()
        
        # モデルリストを更新
        self.refresh_model_list()
        #self.refresh_yolo_model_list()
        self.refresh_yolo_unified_model_list()  # 追加

        # 位置ボタンのカウント表示を更新
        self.update_location_button_counts()
        
        # アノテーション分布を初期化
        if hasattr(self, 'distribution_label'):
            self.distribution_label.clear()
            self.distribution_label.setText("アノテーションがありません")
        
        # 統計情報を更新（画像読み込み時）
        self.update_driving_annotation_stats()
        
        # アノテーション関連ボタンをアクティブ化
        self.set_annotation_buttons_enabled(True)
        
        # オートアノテーションボタンは各モデルが読み込まれるまで無効化
        if hasattr(self, 'auto_annotate_button'):
            self.auto_annotate_button.setEnabled(False)
        if hasattr(self, 'yolo_auto_annotate_btn'):
            self.yolo_auto_annotate_btn.setEnabled(False)
        
        # 推論結果表示チェックボックスの状態を正しく設定
        self.update_inference_checkboxes_status()
        
        # 画像ソース切り替え時の推論実行
        self.run_inference_after_image_source_change()
        
        # プログレスダイアログを閉じる
        progress.setValue(100)
        progress.close()
        
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
        
        # セッション情報を保存
        self.save_session_info()

    def run_inference_after_image_source_change(self):
        """画像ソース切り替え後の推論実行"""
        if not self.images:
            return
            
        # 自動運転推論の実行
        if (hasattr(self, 'inference_checkbox') and 
            self.inference_checkbox.isChecked() and 
            hasattr(self, 'model') and self.model is not None):
            
            print("画像ソース切り替え検出: 自動運転推論を実行します")
            self.statusBar().showMessage("画像ソース切り替えのため推論を実行中...", 3000)
            try:
                # 現在の画像のみ推論を実行
                self.run_inference_check(all_images=False)
            except Exception as e:
                print(f"自動運転推論エラー: {e}")
        
        # 位置推論の実行
        if (hasattr(self, 'location_inference_checkbox') and 
            self.location_inference_checkbox.isChecked() and 
            hasattr(self, 'location_model_manager') and 
            self.location_model_manager.is_model_loaded()):
            
            print("画像ソース切り替え検出: 位置推論を実行します")
            try:
                # 現在の画像のみ位置推論を実行
                self.run_location_inference_check(all_images=False)
            except Exception as e:
                print(f"位置推論エラー: {e}")
        
        # YOLO物体検知推論の実行
        if (hasattr(self, 'detection_inference_checkbox') and 
            self.detection_inference_checkbox.isChecked() and 
            hasattr(self, 'yolo_model') and self.yolo_model is not None):
            
            print("画像ソース切り替え検出: YOLO物体検知推論を実行します")
            try:
                self.run_single_yolo_inference()
            except Exception as e:
                print(f"YOLO物体検知推論エラー: {e}")
        
        # YOLOセグメンテーション推論の実行
        if (hasattr(self, 'segmentation_inference_checkbox') and 
            self.segmentation_inference_checkbox.isChecked() and 
            hasattr(self, 'yolo_seg_model') and self.yolo_seg_model is not None):
            
            print("画像ソース切り替え検出: YOLOセグメンテーション推論を実行します")
            try:
                self.run_single_yolo_segmentation_inference()
            except Exception as e:
                print(f"YOLOセグメンテーション推論エラー: {e}")

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
                    except Exception as e:
                        print(f"カタログファイル検索エラー {parent_dir}: {e}")
                            
            progress.setValue(len(self.folder_paths))
            progress.close()
            
            if annotations_loaded:
                # Update UI
                self.display_current_image()
                self.update_gallery()
                self.update_distribution_graph()  # 追加：分布グラフを更新
                self.update_slider_deleted_indexes()
                
                # 位置ボタンのカウント表示を更新
                self.update_location_button_counts()
                
                # 運転アノテーションの統計情報を更新
                self.update_driving_annotation_stats()
                
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

        except Exception as e:
            if 'progress' in locals():
                progress.close()
            traceback.print_exc()
            QMessageBox.critical(
                self, 
                "エラー", 
                f"アノテーションの読み込み中にエラーが発生しました: {str(e)}"
            )

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
            self.display_current_image()
            self.update_gallery()
            self.update_slider_deleted_indexes()
            
            # 位置ボタンのカウント表示を更新
            self.update_location_button_counts()
            
            print(f"サブフォルダアノテーション読み込み完了: {len(self.annotations)}個のアノテーション")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "エラー",
                f"サブフォルダアノテーションの読み込み中にエラーが発生しました: {str(e)}"
            )

    def open_data_analysis(self):
        """データ分析ダイアログを開く"""
        if not self.annotations:
            QMessageBox.warning(self, "警告", "アノテーションデータがありません。")
            return

        # 利用可能なセンサーキーを取得
        available_keys = getattr(self, 'available_sensor_keys', set())

        # ダイアログを作成
        self.data_analysis_dialog = DataAnalysisDialog(
            parent=self,
            annotations=self.annotations,
            images=self.images,
            deleted_indexes=getattr(self, 'deleted_indexes', []),
            downsampled_indexes=getattr(self, 'downsampled_indexes', []),
            available_sensor_keys=available_keys
        )

        # ジャンプシグナルを接続
        self.data_analysis_dialog.jump_to_image.connect(self.jump_to_index_from_analysis)

        # 非モーダルで表示（メインウィンドウと並行操作可能）
        self.data_analysis_dialog.show()

    def jump_to_index_from_analysis(self, index):
        """データ分析ダイアログからのジャンプ要求を処理"""
        if 0 <= index < len(self.images):
            self.current_index = index
            self.display_current_image()
            self.update_gallery()
            self.image_slider.setValue(index)
            self.slider_value_label.setText(f"{index + 1}/{len(self.images)}")
            self.statusBar().showMessage(f"インデックス {index} にジャンプしました", 3000)

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
        # models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
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
            
            # モデルを明示的に読み込み（self.modelに保存）
            from model_catalog import get_model, load_model_weights, detect_num_outputs_from_checkpoint

            # モデルファイル名から実際のモデルタイプを判定
            model_filename = os.path.basename(model_path)
            actual_model_type = model_type  # デフォルトは選択されたタイプ

            # ファイル名に基づいてモデルタイプを調整
            if "_location_" in model_filename:
                # 位置推論モデルの場合は適切なエラーメッセージを表示
                raise ValueError(f"選択されたファイルは位置推論モデルです。自動運転モデルを選択してください。\nファイル: {model_filename}")
            elif "yolo" in model_filename.lower():
                # YOLOモデルの場合
                raise ValueError(f"選択されたファイルはYOLOモデルです。自動運転モデルを選択してください。\nファイル: {model_filename}")

            # チェックポイントから出力数を検出
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_outputs = detect_num_outputs_from_checkpoint(model_path, device)

            # モデルインスタンスを作成（検出した出力数で）
            self.model = get_model(actual_model_type, pretrained=False, num_outputs=num_outputs)

            # 重みを読み込み
            load_model_weights(self.model, model_path, device)
            self.model.eval()
            
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
            
            # 推論結果を保存（インデックスベースに変換）
            old_count = len(self.inference_results)
            
            # 画像パスからインデックスに変換して保存
            for img_path, result in inference_results.items():
                # 画像パスから対応するインデックスを取得
                try:
                    img_index = self.images.index(img_path)
                    self.inference_results[img_index] = result
                    print(f"推論結果保存: インデックス{img_index} <- {os.path.basename(img_path)}")
                    
                    # 差分ベクトルの計算と保存
                    self.calculate_and_store_diff_vector(img_index)
                    
                except ValueError:
                    print(f"警告: 画像パス {img_path} がself.imagesに見つかりません")
                    # パスでも保存（後方互換性のため）
                    self.inference_results[img_path] = result
            
            # モデル変更を検出するための状態を保持
            self._last_model_info = (model_type, model_path)
            
            # 推論表示チェックボックスを有効にして自動的にオンにする
            progress.setLabelText("推論表示を更新中...")
            progress.setValue(90)
            QApplication.processEvents()
            
            self.inference_checkbox.setEnabled(True)
            self.inference_checkbox.setToolTip("自動運転モデルが読み込まれています")
            self.inference_checkbox.setChecked(True)
            
            # 差分ベクトル表示チェックボックスも有効にする
            self.diff_vector_checkbox.setEnabled(True)
            self.diff_vector_checkbox.setToolTip("自動運転モデルが読み込まれています")
            
            # 全画像推論ボタンも有効にする
            if hasattr(self, 'batch_inference_button'):
                self.batch_inference_button.setEnabled(True)
                self.batch_inference_button.setToolTip("全ての画像に対して推論を実行します")
            
            # 各モデルの状態を更新
            self.update_inference_checkboxes_status()
            
            # 推論表示を更新
            self.update_inference_display()
            self.update_ui()
            
            # ダークモードの状態を再適用（モデル読み込み後のスタイルリセット対策）
            if hasattr(self, 'is_dark_mode') and self.is_dark_mode:
                if hasattr(self, 'current_image_info'):
                    self.current_image_info.setStyleSheet("color: #ffffff; font-weight: bold;")
                if hasattr(self, 'graph_title'):
                    self.graph_title.setStyleSheet("font-weight: bold; color: #ffffff;")

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
            
            # セッション情報を保存
            self.save_session_info()
            
            # 自動運転モデル読み込み完了後、オートアノテーションボタンを有効化
            if hasattr(self, 'auto_annotate_button'):
                self.auto_annotate_button.setEnabled(True)
            
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
                        
                        # 削除済みインデックスをインスタンス変数に初期化（後で実際の画像インデックスも追加される）
                        if not hasattr(self, 'deleted_indexes'):
                            self.deleted_indexes = []
                        print(f"削除済みエントリインデックス: {deleted_indexes}")
            
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
            
            # 削除されたエントリに対応する実際の画像インデックスを記録するための辞書
            deleted_actual_indexes = []
            
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
                            # 安全な進捗計算
                            try:
                                if total_entries > 0:
                                    sub_progress = 30 + int(i * progress_step) + int((entry_count / total_entries) * progress_step)
                                else:
                                    sub_progress = 30 + int(i * progress_step)
                                # sub_progressがNoneまたは不正な値でないか確認
                                if sub_progress is not None and isinstance(sub_progress, (int, float)):
                                    progress.setValue(min(80, int(sub_progress)))
                            except Exception as progress_error:
                                # 進捗更新エラーは無視（致命的ではない）
                                pass
                            QApplication.processEvents()
                        
                        entry = json.loads(line)
                        
                        # エントリのインデックスを取得
                        entry_index = entry.get('_index', None)
                        
                        # 削除されたインデックスかどうかをチェック
                        is_deleted = entry_index in deleted_indexes
                        
                        # 画像ファイル名を取得
                        # */image_array パターンを検索
                        img_name = ''
                        for key in entry.keys():
                            if key.endswith('/image_array'):
                                img_name = entry[key]
                                break
                        if not img_name:
                            continue
                        
                        # 画像パスの処理 - 複数のパターンを試す
                        img_path = None
                        
                        # 様々なパターンで画像を検索
                        path_patterns = [
                            os.path.join(images_folder, img_name),
                            os.path.join(catalog_folder, img_name),
                            os.path.join(os.path.dirname(catalog_path), img_name),
                            os.path.join(catalog_folder, "images", img_name),
                            # catalog_folderの親ディレクトリも検索
                            os.path.join(os.path.dirname(catalog_folder), img_name),
                            os.path.join(os.path.dirname(catalog_folder), "images", img_name)
                        ]
                        
                        for path in path_patterns:
                            if os.path.exists(path):
                                # パスが存在する場合、self.imagesに含まれているか確認
                                if path in self.images:
                                    img_path = path
                                    break
                                # 含まれていない場合でも、カタログフォルダ内の画像なら使用
                                elif path.startswith(catalog_folder) or path.startswith(images_folder):
                                    img_path = path
                                    # self.imagesに追加（後で使用できるように）
                                    if path not in self.images:
                                        self.images.append(path)
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
                        
                        # 画像のインデックスを self.images リストから取得
                        actual_index = None
                        try:
                            actual_index = self.images.index(img_path)
                        except ValueError:
                            print(f"警告: 画像 {img_path} が self.images に見つかりません")
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

                            # Speed情報を取得
                            speed = entry.get('speed', entry.get('user/speed', entry.get('pilot/speed', None)))

                            # 座標に変換
                            x = int((angle + 1) / 2 * img_width)
                            y = int((1 - throttle) / 2 * img_height)

                            # 範囲内に収める
                            x = max(0, min(x, img_width - 1))
                            y = max(0, min(y, img_height - 1))

                            # アノテーションを保存 - actual_indexを使用
                            self.annotations[actual_index] = {
                                "angle": angle,
                                "throttle": throttle,
                                "x": x,
                                "y": y,
                                "original_index": entry_index  # 元のインデックスを保存（保存時の復元用）
                            }

                            # 位置情報があれば追加
                            if location is not None:
                                self.annotations[actual_index]["loc"] = location
                                self.location_annotations[actual_index] = location

                                # 位置情報ボタンがまだなければ追加
                                self.ensure_location_button_exists(location)

                            # Speed情報があれば追加
                            if speed is not None:
                                self.annotations[actual_index]["speed"] = speed

                            # その他のセンサーデータを保存（数値データのみ）
                            for key, value in entry.items():
                                # 既に処理済みのキーやメタデータはスキップ
                                if key.startswith('_') or key.endswith('/image_array'):
                                    continue
                                if key in ['user/angle', 'pilot/angle', 'user/throttle', 'pilot/throttle',
                                           'user/loc', 'pilot/loc', 'speed', 'user/speed', 'pilot/speed']:
                                    continue
                                # 数値データのみ保存
                                if isinstance(value, (int, float)):
                                    self.annotations[actual_index][key] = value
                                    # 利用可能なキーを記録
                                    if not hasattr(self, 'available_sensor_keys'):
                                        self.available_sensor_keys = set()
                                    self.available_sensor_keys.add(key)

                            # タイムスタンプを保存
                            self.annotation_timestamps[actual_index] = entry.get('_timestamp_ms', int(time.time() * 1000))
                            
                            # 削除されたエントリの場合、削除インデックスリストに追加
                            if is_deleted:
                                deleted_actual_indexes.append(actual_index)

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
                                self.inference_results[actual_index] = {
                                    "angle": pilot_angle,
                                    "throttle": pilot_throttle,
                                    "pilot/angle": pilot_angle,
                                    "pilot/throttle": pilot_throttle,
                                    "x": pilot_x,
                                    "y": pilot_y
                                }

                                # 推論結果に位置情報があれば追加
                                if "pilot/loc" in entry:
                                    self.inference_results[actual_index]["pilot/loc"] = entry["pilot/loc"]
                                    self.inference_results[actual_index]["loc"] = entry["pilot/loc"]

                                # 推論結果にspeed情報があれば追加
                                if "pilot/speed" in entry:
                                    self.inference_results[actual_index]["pilot/speed"] = entry["pilot/speed"]
                                    self.inference_results[actual_index]["speed"] = entry["pilot/speed"]
                                    
                        except Exception as e:
                            print(f"画像 {img_path} の処理中にエラー: {e}")
                            continue
            
            # 削除されたインデックスを設定（実際の画像インデックスのみを使用）
            if deleted_actual_indexes:
                # 実際の画像インデックスを設定
                self.deleted_indexes.extend(deleted_actual_indexes)
                # 重複を削除してソート
                self.deleted_indexes = sorted(list(set(self.deleted_indexes)))
                print(f"削除済みインデックスを設定: 実際の画像インデックス {len(deleted_actual_indexes)}個、総計 {len(self.deleted_indexes)}個")
            
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
            
            # 運転アノテーションの統計情報を更新
            self.update_driving_annotation_stats()
            
            # スライダーの削除インデックスを更新
            self.update_slider_deleted_indexes()
            return self.annotated_count > 0
                
        except Exception as e:
            # エラー発生時も進捗ダイアログを閉じる
            if 'progress' in locals():
                progress.close()
                
            print(f"カタログフォルダ {catalog_folder} の読み込み中にエラー: {str(e)}")
            traceback.print_exc()
            return False


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
        self.display_current_image()
        self.update_gallery()
        self.update_slider_deleted_indexes()
        
        # 位置ボタンのカウント表示を更新
        self.update_location_button_counts()
        
        # 分布グラフを更新
        if hasattr(self, 'distribution_label'):
            self.distribution_label.clear()
            self.distribution_label.setText("アノテーションがありません")

        print("アノテーションデータをクリアしました")



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
    
    def update_annotation_info_label(self):
        """物体検知アノテーション情報を表示する"""
        if not self.images:
            return ""

        # インデックスベースに変更
        current_index = self.current_index
        is_deleted = hasattr(self, 'deleted_indexes') and current_index in self.deleted_indexes

        # 物体検知アノテーション情報
        bbox_info = ""
        # 修正: パスベース → インデックスベース
        if (current_index in self.bbox_annotations and
            self.bbox_annotations[current_index]):
            bboxes = self.bbox_annotations[current_index]
            bbox_info = f"<b>物体検知アノテーション:</b><br>"

            # 削除済みの場合は表示を追加
            if is_deleted:
                bbox_info = f"<span style='color: #FF5555;'>[削除済み]</span> " + bbox_info

            # クラスごとのカウント辞書
            class_counts = {}
            for bbox in bboxes:
                class_name = bbox.get('class', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # クラスカウント情報のフォーマット
            bbox_info += "検出オブジェクト:<br>"
            for class_name, count in class_counts.items():
                # このクラスの色を取得
                class_colors = {
                    'car': "#FF0000",     # 赤
                    'person': "#00FF00",  # 緑
                    'sign': "#0000FF",    # 青
                    'cone': "#FFFF00",    # 黄
                    'unknown': "#808080"  # グレー
                }
                color = class_colors.get(class_name, "#FF0000")

                bbox_info += f"<span style='color: {color}; font-weight: bold;'>● {class_name}</span>: {count}個<br>"

            bbox_info += f"合計: {len(bboxes)}個のオブジェクト<br>"

        return bbox_info

    def display_current_image(self):
        """現在の画像を表示（YOLOアノテーションも含む）"""
        if not self.images or self.current_index >= len(self.images):
            return
        
        current_image_path = self.images[self.current_index]
        
        try:
            # 画像を読み込み
            image = load_image_safely(current_image_path)
            if image is None:
                print(f"画像読み込み失敗: {current_image_path}")
                return
            
            # PIL画像をQImageに変換してQPixmapに設定
            qimage = pil_to_qimage(image)
            pixmap = QPixmap.fromImage(qimage)
            
            # main_image_viewに画像を設定
            if hasattr(self, 'main_image_view'):
                self.main_image_view.setPixmap(pixmap)
                
                # バウンディングボックスアノテーションがあれば表示用データを設定
                if hasattr(self, 'bbox_annotations') and self.current_index in self.bbox_annotations:
                    # ObjectDetectionImageLabelがあればバウンディングボックスを設定
                    if hasattr(self.main_image_view, 'set_boxes'):
                        boxes = []
                        for bbox_data in self.bbox_annotations[self.current_index]:
                            class_name = bbox_data.get('class', 'unknown')
                            # 座標を適切に取得し、明示的にfloat型に変換
                            try:
                                x1 = float(bbox_data.get('x1', 0.0))
                                y1 = float(bbox_data.get('y1', 0.0))
                                x2 = float(bbox_data.get('x2', x1))
                                y2 = float(bbox_data.get('y2', y1))
                                
                                # 座標を0-1範囲内に制限
                                x1 = max(0.0, min(1.0, x1))
                                y1 = max(0.0, min(1.0, y1))
                                x2 = max(0.0, min(1.0, x2))
                                y2 = max(0.0, min(1.0, y2))
                                
                                # ObjectDetectionImageLabelが期待するピクセル座標に変換
                                img_width = pixmap.width()
                                img_height = pixmap.height()
                                
                                pixel_x1 = int(x1 * img_width)
                                pixel_y1 = int(y1 * img_height)
                                pixel_x2 = int(x2 * img_width)
                                pixel_y2 = int(y2 * img_height)
                                
                                bbox = (pixel_x1, pixel_y1, pixel_x2, pixel_y2)
                                boxes.append((class_name, bbox))
                            except (ValueError, TypeError) as e:
                                print(f"Error converting bbox coordinates: {e}")
                                print(f"Problematic bbox_data: {bbox_data}")
                                continue
                        self.main_image_view.set_boxes(boxes)
                
                # セグメンテーション推論結果があれば表示更新
                if hasattr(self, 'segmentation_inference_results') and current_image_path in self.segmentation_inference_results:
                    result = self.segmentation_inference_results[current_image_path]
                    if result and result.get('segments'):
                        # セグメンテーション推論結果の表示を更新
                        self.update_segmentation_inference_display()
                
                # セグメンテーション手動アノテーション結果があれば表示
                if hasattr(self, 'segmentation_annotations') and current_image_path in self.segmentation_annotations:
                    # セグメンテーションデータを設定する処理を追加（今後の拡張用）
                    pass

                # 運転アノテーションポイント（赤丸）の設定
                self._set_annotation_point_on_canvas()

                # 推論結果ポイント（青丸）の設定
                self._set_inference_point_on_canvas()

                # CAM表示が有効な場合は更新
                if hasattr(self, 'gradcam_checkbox') and self.gradcam_checkbox.isChecked():
                    self.update_gradcam_visualization()

            # Enhanced annotations display processing - UI要素の更新
            self._update_enhanced_ui_elements()

        except Exception as e:
            error_message = str(e)
            print(f"Error loading image {current_image_path}: {error_message}")
            print(f"画像パス: {current_image_path}")
            if hasattr(self, 'bbox_annotations') and self.current_index in self.bbox_annotations:
                print(f"バウンディングボックス数: {len(self.bbox_annotations[self.current_index])}")
                for i, bbox in enumerate(self.bbox_annotations[self.current_index]):
                    print(f"  BBox {i}: {bbox}")
                    print(f"    座標型チェック: x1={bbox.get('x1')}({type(bbox.get('x1'))}), y1={bbox.get('y1')}({type(bbox.get('y1'))}), x2={bbox.get('x2')}({type(bbox.get('x2'))}), y2={bbox.get('y2')}({type(bbox.get('y2'))})")
            import traceback
            traceback.print_exc()
            return

    def update_gallery(self):
        """ギャラリー表示を更新する - 位置情報の問題を根本的に修正"""
        # Clear current gallery - メモリリーク防止のため即座に削除
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
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
        col_count = GALLERY_COL_COUNT 
        
        # ギャラリーにサムネイルを追加
        for i, idx in enumerate(display_indices):
            if 0 <= idx < total_images:
                img_path = self.images[idx]
                
                # 削除されたインデックスの場合、削除済みフラグをセット
                is_deleted = hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes
                
                # アノテーション情報を取得（インデックスベースに統一）
                annotation = None
                location_value = None
                bbox_annotations = None
                segmentation_annotations = None

                # インデックスベースでアノテーションを取得
                if idx in self.annotations:
                    annotation = self.annotations[idx]
                    # 位置情報を事前に特定
                    if annotation and 'loc' in annotation:
                        location_value = annotation['loc']

                # 位置情報専用の辞書をインデックスベースで確認
                if location_value is None and idx in self.location_annotations:
                    location_value = self.location_annotations[idx]

                # 修正: インデックスベースでバウンディングボックスを取得
                if idx in self.bbox_annotations:
                    bbox_annotations = self.bbox_annotations[idx]

                # 修正: インデックスベースでセグメンテーションアノテーションを取得
                if (hasattr(self, 'segmentation_annotations') and
                    idx in self.segmentation_annotations):
                    segmentation_annotations = self.segmentation_annotations[idx]

                # waypointアノテーションを取得
                waypoint_annotations = None
                if (hasattr(self, 'waypoint_annotations') and
                    idx in self.waypoint_annotations):
                    waypoint_annotations = self.waypoint_annotations[idx]

                # 拡張サムネイルウィジェットを作成
                thumb = ThumbnailWidget(
                    img_path=img_path,
                    index=idx,
                    is_selected=(idx == current_idx),
                    annotation=annotation,
                    on_click=self.select_image,
                    location_value=location_value,
                    is_deleted=is_deleted,
                    bbox_annotations=bbox_annotations,
                    segmentation_annotations=segmentation_annotations,
                    waypoint_annotations=waypoint_annotations
                )
                
                # col_count列のグリッドで配置
                row = i // col_count
                col = i % col_count
                
                self.gallery_layout.addWidget(thumb, row, col)

    def _update_enhanced_ui_elements(self):
        """Enhanced annotations display UI更新処理"""
        if not self.images:
            return

        current_index = self.current_index
        current_img_path = self.images[current_index]
        is_deleted = hasattr(self, 'deleted_indexes') and current_index in self.deleted_indexes

        # スライダーの表示を更新
        if hasattr(self, 'slider_value_label'):
            self.slider_value_label.setText(f"{current_index + 1}/{len(self.images)}")

        # 画像情報表示の更新
        if hasattr(self, 'current_image_info'):
            filename = os.path.basename(current_img_path)
            status_text = " [削除済み]" if is_deleted else ""
            self.current_image_info.setText(
                f"画像 {current_index + 1} of {len(self.images)}:{status_text}\n{filename}"
            )

            # 削除済みの場合は赤字で表示
            if is_deleted:
                self.current_image_info.setStyleSheet("color: #FF5555; font-weight: bold;")
            else:
                self.current_image_info.setStyleSheet("color: #333333; font-weight: bold;")

        # アノテーション情報の表示
        self._update_annotation_info_display(current_index, is_deleted)

        # 位置情報ラベルの更新
        self._update_location_info_display(current_index, is_deleted)

    def _update_annotation_info_display(self, current_index, is_deleted):
        """アノテーション情報表示の更新"""
        if not hasattr(self, 'annotation_info_label'):
            return

        if current_index in self.annotations:
            anno = self.annotations[current_index]

            # アノテーション辞書が存在するがデータが空の場合をチェック
            has_driving_annotation = 'angle' in anno or 'throttle' in anno or 'loc' in anno

            if has_driving_annotation:
                # 基本的なアノテーション情報
                annotation_text = f"<b>運転アノテーション情報:</b><br>"
                if is_deleted:
                    annotation_text = f"<span style='color: #FF5555;'><b>削除済み</b></span><br>" + annotation_text

                # angleとthrottleの存在チェックを追加
                if 'angle' in anno:
                    annotation_text += f"angle = <span style='color: #FF6666;'>{anno['angle']:.4f}</span><br>"
                else:
                    annotation_text += f"angle = <span style='color: #999999;'>未設定</span><br>"

                if 'throttle' in anno:
                    annotation_text += f"throttle = <span style='color: #FF6666;'>{anno['throttle']:.4f}</span>"
                else:
                    annotation_text += f"throttle = <span style='color: #999999;'>未設定</span>"

                # 位置情報があれば追加して強調表示
                if 'loc' in anno:
                    loc_value = anno['loc']
                    loc_color = get_location_color(loc_value)

                    # 位置情報を色付きのバッジとして表示
                    annotation_text += f"<br><div style='margin-top: 10px;'>"
                    annotation_text += f"<div style='display: inline-block; background-color: {loc_color.name()}; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'>"
                    annotation_text += f"位置 {loc_value}</div></div>"

                # 物体検知アノテーション情報を追加
                bbox_info = self.update_annotation_info_label()
                if bbox_info:
                    annotation_text += f"<br><br>{bbox_info}"

                # リッチテキストとして設定
                self.annotation_info_label.setText(annotation_text)
                self.annotation_info_label.setTextFormat(Qt.RichText)
            else:
                # 運転アノテーションデータが空の場合、物体検知アノテーションのみ表示
                bbox_info = self.update_annotation_info_label()
                if bbox_info:
                    if is_deleted:
                        bbox_info = f"<span style='color: #FF5555;'><b>削除済み</b></span><br>" + bbox_info
                    self.annotation_info_label.setText(bbox_info)
                    self.annotation_info_label.setTextFormat(Qt.RichText)
                else:
                    self.annotation_info_label.setText("")
        # 修正: インデックスベースでバウンディングボックスをチェック
        elif (current_index in self.bbox_annotations and
              self.bbox_annotations[current_index]):
            # 自動運転アノテーションはないが、物体検知アノテーションはある場合
            bbox_info = self.update_annotation_info_label()

            # 削除済みの場合は削除済み表示を追加
            if is_deleted:
                bbox_info = f"<span style='color: #FF5555;'><b>削除済み</b></span><br>" + bbox_info

            self.annotation_info_label.setText(bbox_info)
            self.annotation_info_label.setTextFormat(Qt.RichText)
        elif is_deleted:
            # 削除済みの場合のメッセージ
            self.annotation_info_label.setText(
                "<span style='color: #FF5555;'>この画像は削除済みです。<br>"
                "画像をクリックするか「削除状態を復元」ボタンを押して<br>"
                "再度アノテーションを行えます。</span>"
            )
            self.annotation_info_label.setTextFormat(Qt.RichText)
        else:
            self.annotation_info_label.setText("")

    def _update_location_info_display(self, current_index, is_deleted):
        """位置情報表示の更新"""
        if not hasattr(self, 'current_location_label'):
            return

        location_value = None
        # アノテーションの位置情報を確認
        if current_index in self.annotations and 'loc' in self.annotations[current_index]:
            location_value = self.annotations[current_index]['loc']
        # 位置情報専用の辞書を確認
        elif current_index in self.location_annotations:
            location_value = self.location_annotations[current_index]

        # 位置情報ラベルの更新
        if location_value is not None and not is_deleted:
            # 位置情報ラベルの更新（self.current_locationは更新しない）
            self.current_location_label.setText(f"現在の位置情報: {location_value}")

            # 位置情報に基づいた色を取得
            loc_color = get_location_color(location_value)
            self.current_location_label.setStyleSheet(f"color: {loc_color.name()}; font-weight: bold;")

            # ボタンの選択状態を更新
            if hasattr(self, 'location_buttons'):
                for button in self.location_buttons:
                    button_value = button.property("location_value")
                    button.setChecked(button_value == location_value)
        else:
            # 位置情報がない場合
            self.current_location_label.setText("現在の位置情報: なし")
            self.current_location_label.setStyleSheet("")

            # すべてのボタンの選択を解除
            if hasattr(self, 'location_buttons'):
                for button in self.location_buttons:
                    button.setChecked(False)

    def _set_annotation_point_on_canvas(self):
        """キャンバス上に運転アノテーションポイント（赤丸）を設定"""
        if not hasattr(self, 'main_image_view'):
            return

        # アノテーションポイントの設定（画像読み込みの成功/失敗に関係なく実行）
        if (self.current_index in self.annotations and
            'x' in self.annotations[self.current_index] and
            'y' in self.annotations[self.current_index]):
            anno = self.annotations[self.current_index]
            self.main_image_view.annotation_point = QPoint(anno['x'], anno['y'])
        else:
            self.main_image_view.annotation_point = None

        # 削除済みの場合
        is_deleted = hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes
        if is_deleted:
            # 削除済みフラグを設定
            self.main_image_view.is_deleted = True
        else:
            self.main_image_view.is_deleted = False

        # ダウンサンプリング対象の場合
        is_downsampled = hasattr(self, 'downsampled_indexes') and self.current_index in self.downsampled_indexes
        self.main_image_view.is_downsampled = is_downsampled

        # UIを更新（画像読み込みの成功/失敗に関係なく実行）
        self.main_image_view.update()

    def _set_inference_point_on_canvas(self):
        """キャンバス上に推論結果ポイント（青丸）を設定"""
        if not hasattr(self, 'main_image_view'):
            return

        # 推論ポイントの設定（画像読み込みの成功/失敗に関係なく実行）
        if self.inference_checkbox.isChecked() and self.current_index in self.inference_results:
            inference = self.inference_results[self.current_index]
            self.main_image_view.inference_point = QPoint(inference['x'], inference['y'])
        else:
            self.main_image_view.inference_point = None

        # UIを更新
        self.main_image_view.update()

    def select_image(self, index):
        if 0 <= index < len(self.images):
            # インデックスが変わらない場合は何もしない
            if index == self.current_index:
                return

            # 画像を移動する前に、作成途中のアノテーション頂点をクリア
            self.clear_incomplete_annotations(show_message=True)

            # 前回のwaypointを保存（変更前のインデックス用）
            if (self.current_index is not None and
                self.current_index in self.waypoint_annotations and
                self.waypoint_annotations[self.current_index]):
                self.last_waypoints = self.waypoint_annotations[self.current_index].copy()

            # 現在の画像に変更
            self.current_index = index
            
            # スライダーの値を更新
            self.image_slider.setValue(index)
            self.slider_value_label.setText(f"{index + 1}/{len(self.images)}")
            
            # 画像表示を更新
            self.display_current_image()

            # 前回waypoint自動適用機能
            if (hasattr(self, 'auto_apply_last_waypoint') and
                self.auto_apply_last_waypoint and
                self.last_waypoints and
                hasattr(self, 'current_mode') and
                self.current_mode == 3):  # waypointモードの場合のみ

                # 新しい画像にwaypointがない場合のみ適用
                if (self.current_index not in self.waypoint_annotations or
                    not self.waypoint_annotations[self.current_index]):

                    self.waypoint_annotations[self.current_index] = self.last_waypoints.copy()

                    # ステータスメッセージ
                    if hasattr(self, 'statusBar'):
                        self.statusBar().showMessage(f"前回のwaypoint {len(self.last_waypoints)}個を自動適用しました", 2000)

                    # 画面を再更新
                    self.display_current_image()

            # 推論表示チェックボックスがONの場合、推論結果を表示
            # デバウンス処理により遅延実行（連打対応）
            if self.inference_checkbox.isChecked():
                self._schedule_inference()

            # ギャラリー更新
            self.update_gallery()
            
            # サムネイルクリック時は自動スキップしない（ユーザーが明示的に選択した画像で停止）
            # 自動スキップ機能は矢印キーなどの他の操作でのみ有効

    def clear_incomplete_annotations(self, show_message=True):
        """作成途中のアノテーションをクリアする共通関数"""
        cleared = False
        message_parts = []

        # セグメンテーションの作成途中の頂点をクリア
        if hasattr(self.main_image_view, 'current_segmentation_polygon') and self.main_image_view.current_segmentation_polygon:
            vertex_count = len(self.main_image_view.current_segmentation_polygon)
            self.main_image_view.current_segmentation_polygon = []
            if hasattr(self.main_image_view, 'is_drawing_segmentation'):
                self.main_image_view.is_drawing_segmentation = False
            message_parts.append(f"セグメンテーション（{vertex_count}個の頂点）")
            cleared = True

        # 選択状態をクリア
        if hasattr(self.main_image_view, 'selected_polygon_index') and self.main_image_view.selected_polygon_index is not None:
            self.main_image_view.selected_polygon_index = None
            self.main_image_view.selected_vertex_index = None
            cleared = True

        if hasattr(self.main_image_view, 'selected_segmentation_index') and self.main_image_view.selected_segmentation_index is not None:
            self.main_image_view.selected_segmentation_index = None
            cleared = True

        # 画面を更新
        if cleared and hasattr(self.main_image_view, 'update'):
            self.main_image_view.update()

        # ステータスバーに通知
        if cleared and show_message and message_parts:
            message = "作成途中の" + "、".join(message_parts) + "をクリアしました"
            self.statusBar().showMessage(message, 2000)

        return cleared

    def skip_images(self, count):
        """指定した数だけ画像をスキップする - 位置推論の自動実行も追加"""
        # waypointモードの場合、現在の画像のwaypoint数をチェック
        if not self._check_waypoint_count_before_transition():
            return

        new_index = self.current_index + count

        # 自動再生中に境界に到達した場合は停止
        is_auto_playing = hasattr(self, 'auto_play_timer') and self.auto_play_timer.isActive()

        # Ensure the new index is within bounds
        if new_index < 0:
            new_index = 0
            # 自動再生中で最初の画像に到達した場合は停止
            if is_auto_playing:
                self.stop_auto_play()
                self.statusBar().showMessage("最初の画像に到達したため自動再生を停止しました", 2000)
        elif new_index >= len(self.images):
            new_index = len(self.images) - 1
            # 自動再生中で最後の画像に到達した場合は停止
            if is_auto_playing:
                self.stop_auto_play()
                self.statusBar().showMessage("最後の画像に到達したため自動再生を停止しました", 2000)

        # インデックスが変わらない場合は何もしない
        if new_index == self.current_index:
            return

        # 画像を移動する前に、作成途中のアノテーション頂点をクリア
        self.clear_incomplete_annotations(show_message=True)

        # スキップ前に現在の画像のwaypoint情報を保存
        current_index = self.current_index
        if (current_index is not None and
            current_index in self.waypoint_annotations and
            self.waypoint_annotations[current_index]):
            self.last_waypoints = self.waypoint_annotations[current_index].copy()

        # スキップ前に現在の画像のバウンディングボックス情報を確認し、すべてのボックスを記録
        if hasattr(self, 'bbox_annotations') and len(self.images) > 0:
            # インデックスベースに変更
            current_index = self.current_index
            if (current_index is not None and 
                isinstance(current_index, int) and 
                current_index in self.bbox_annotations and 
                self.bbox_annotations[current_index]):
                # すべてのバウンディングボックスをリストとして保存
                self.last_bboxes = [bbox.copy() for bbox in self.bbox_annotations[current_index]]
                
                # 互換性のため、最後のボックスも個別に保存
                if self.last_bboxes:
                    self.last_bbox = self.last_bboxes[-1].copy()

        # スキップ前に現在の画像のセグメンテーション情報を確認し、すべてのセグメンテーションを記録
        if hasattr(self, 'segmentation_annotations') and len(self.images) > 0:
            # インデックスベースに変更
            current_index = self.current_index
            if (current_index is not None and 
                isinstance(current_index, int) and 
                current_index in self.segmentation_annotations and 
                self.segmentation_annotations[current_index]):
                # すべてのセグメンテーションをリストとして保存
                self.last_segmentations = [seg.copy() for seg in self.segmentation_annotations[current_index]]
                print(f"スキップ時にセグメンテーション情報を更新: {len(self.last_segmentations)}個のセグメンテーション")
                
                # 互換性のため、最後のセグメンテーションも個別に保存
                if self.last_segmentations:
                    self.last_segmentation = self.last_segmentations[-1].copy()

        # 自動位置設定をする前の現在の位置情報を保存
        old_current_location = self.current_location
        
        # 現在のインデックスを更新
        self.current_index = new_index

        # 新しい画像パスを取得（後で複数回使うので変数に格納）
        new_img_path = self.images[self.current_index]

        # スライダーの値を更新（valueChangedシグナルが発生し、slider_changedが呼ばれる）
        self.image_slider.setValue(new_index)
        self.slider_value_label.setText(f"{new_index + 1}/{len(self.images)}")
                
        # 自動チェックオン
        if self.auto_apply_location:
            # 前の値がある場合は上書き
            if old_current_location is not None:
                self.current_location = old_current_location
                # インデックスとパスの両方に対応するため、必要に応じて初期化
                if new_index not in self.annotations:
                    self.annotations[new_index] = {}
                self.annotations[new_index]['loc'] = self.current_location
                self.location_annotations[new_index] = self.current_location
        else:
            if self.current_index in self.location_annotations:
                self.current_location = self.location_annotations[self.current_index]
            else:
                self.current_location = None

        # 前回のバウンディングボックスを適用（画像表示の前に先に処理）
        if hasattr(self, 'auto_apply_last_bbox') and self.auto_apply_last_bbox:
            # インデックスベースに変更: 現在の画像にボックスがない場合に適用
            if (new_index not in self.bbox_annotations or 
                not self.bbox_annotations[new_index]):
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

        # 前回のセグメンテーションを自動適用（最後に追加）
        if hasattr(self, 'auto_apply_last_segmentation') and self.auto_apply_last_segmentation:
            # インデックスベースに変更: 現在の画像にセグメンテーションがない場合に適用
            if (new_index not in self.segmentation_annotations or 
                not self.segmentation_annotations[new_index]):
                # last_segmentationsが存在すればそれを使用、なければlast_segmentationを使用
                if hasattr(self, 'last_segmentations') and self.last_segmentations:
                    # すべてのセグメンテーションを適用
                    for seg in self.last_segmentations:
                        self.add_segmentation_annotation(seg.copy())
                    
                    # ステータスバーに表示
                    self.statusBar().showMessage(f"前回の {len(self.last_segmentations)}個のセグメンテーションを適用しました", 3000)
                
                elif hasattr(self, 'last_segmentation') and self.last_segmentation:
                    # 後方互換性のため、単一セグメンテーションの場合も処理
                    self.add_segmentation_annotation(self.last_segmentation.copy())
                    self.statusBar().showMessage(f"前回の '{self.last_segmentation['class']}' セグメンテーションを適用しました", 3000)

        # 前回waypoint自動適用機能
        if (hasattr(self, 'auto_apply_last_waypoint') and
            self.auto_apply_last_waypoint and
            self.last_waypoints and
            hasattr(self, 'current_mode') and
            self.current_mode == 3):  # waypointモードの場合のみ

            # 新しい画像にwaypointがない場合のみ適用
            if (new_index not in self.waypoint_annotations or
                not self.waypoint_annotations[new_index]):

                self.waypoint_annotations[new_index] = self.last_waypoints.copy()

                # ステータスメッセージ
                if hasattr(self, 'statusBar'):
                    self.statusBar().showMessage(f"前回のwaypoint {len(self.last_waypoints)}個を自動適用しました", 2000)

        # 画像表示を更新
        self.display_current_image()
        
        # ここから追加: 位置推論表示チェックボックスがONの場合、自動的に推論実行
        if hasattr(self, 'location_inference_checkbox') and self.location_inference_checkbox.isChecked():
            # 推論結果がまだない場合のみ推論を実行
            if new_img_path not in self.location_inference_results:
                self.run_location_inference()
            # 表示を更新
            self.update_location_inference_display()

        # waypoint推論表示チェックボックスがONの場合、自動的に推論実行
        if hasattr(self, 'waypoint_inference_checkbox') and self.waypoint_inference_checkbox.isChecked():
            # 推論結果がまだない場合のみ推論を実行
            self.update_waypoint_inference_display()

        # 推論表示チェックボックスがONの場合、推論結果がなければ実行
        # デバウンス処理により遅延実行（連打対応）
        if self.inference_checkbox.isChecked():
            self._schedule_inference()

        # 物体検知推論表示の更新
        self.update_detection_info_panel()

        # セグメンテーション推論表示の更新
        if hasattr(self, 'segmentation_inference_checkbox') and self.segmentation_inference_checkbox.isChecked():
            # 推論結果がまだない場合のみ推論を実行
            if new_img_path not in self.segmentation_inference_results:
                self.run_single_yolo_segmentation_inference()
            else:
                # 既にある結果の表示を更新
                self.update_segmentation_inference_display()

        # 画面を更新 - ギャラリーは遅延更新でパフォーマンス改善
        # 再生中はデバウンスをスキップして直接更新（タイマーリセット問題を回避）
        if is_auto_playing:
            self.update_gallery()
        else:
            self._schedule_gallery_update()  # デバウンスで遅延更新
        self.update_inference_display()

        # 位置アノテーション数の表示を更新
        self.update_location_button_counts()

        # データ分析ダイアログが開いている場合、現在位置を更新
        if hasattr(self, 'data_analysis_dialog') and self.data_analysis_dialog is not None:
            if self.data_analysis_dialog.isVisible():
                self.data_analysis_dialog.update_current_position(new_index)

    def handle_annotation(self, x, y):
        """画像のアノテーションを処理する - 削除済み画像への再アノテーションをサポート（パフォーマンス最適化版）"""
        if not self.images:
            return

        current_img_path = self.images[self.current_index]

        # Get image dimensions (キャッシュを使用してImage.open()を削減)
        if current_img_path in self.image_size_cache:
            width, height = self.image_size_cache[current_img_path]
        else:
            img = Image.open(current_img_path)
            width, height = img.size
            self.image_size_cache[current_img_path] = (width, height)

        # Get normalized coordinates
        angle, throttle = normalize_coordinates(x, y, width, height)

        # Store current state in history before changing
        # 履歴のサイズ制限を追加してメモリリークを防止
        if self.current_index in self.annotations:
            # 変更前の状態のみを保存（辞書全体ではなく）
            previous_annotation = self.annotations[self.current_index].copy()
            self.annotation_history.append({
                'index': self.current_index,
                'annotation': previous_annotation
            })
            # 履歴を最新100件に制限
            if len(self.annotation_history) > 100:
                self.annotation_history = self.annotation_history[-100:]

        # 削除済みインデックスの場合、削除リストから削除
        if hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes:
            # 現在のインデックスを削除済みリストから除外
            self.deleted_indexes.remove(self.current_index)

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
            self.location_annotations[self.current_index] = self.current_location

        # 位置ボタンのカウント表示を更新（軽い処理なので同期実行）
        self.update_location_button_counts()

        # Update UI - 軽い処理は即座に実行、重い処理は非同期化
        self.update_ui()  # 軽量なので即座に実行
        self.display_current_image()  # 現在の画像表示は即座に更新（重要）

        # 重い処理はデバウンスタイマーで遅延実行（連打対応）
        # Note: ギャラリー更新はskip_images内で呼ばれるため、ここでは呼ばない
        # self._schedule_gallery_update()  # ギャラリー更新を遅延
        self._schedule_distribution_graph_update()  # グラフ更新を遅延
        self.update_slider_deleted_indexes()  # スライダーは即座に更新（軽量）

    def restore_deleted_annotation(self):
        """現在表示中の削除済みの画像を復元する（削除状態を解除する）"""
        if not self.images or not hasattr(self, 'deleted_indexes'):
            return

        # 削除済みリストから削除
        self.deleted_indexes.remove(self.current_index)

        # UI更新 - 重い処理は遅延実行
        self.display_current_image()
        self._schedule_gallery_update()  # 遅延更新
        self._schedule_distribution_graph_update()  # 遅延更新

    def restore_all_deleted_annotations(self):
        """全ての削除済みアノテーションの状態を復元する"""
        if not self.images or not hasattr(self, 'deleted_indexes') or not self.deleted_indexes:
            QMessageBox.information(
                self,
                "情報",
                "復元する削除済みのアノテーションがありません。"
            )
            return

        # 確認ダイアログを表示
        deleted_count = len(self.deleted_indexes)
        reply = QMessageBox.question(
            self,
            "確認",
            f"全ての削除状態をクリアします。よろしいですか？\n\n"
            f"削除済みインデックス数: {deleted_count}個",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 削除済みリストをクリア
        self.deleted_indexes = []

        # UI更新 - 重い処理は遅延実行
        self.display_current_image()
        self._schedule_gallery_update()  # 遅延更新
        self._schedule_distribution_graph_update()  # 遅延更新
        self.update_slider_deleted_indexes()
    

    ### エクスポート関連
    def export_to_donkey(self):
        """Donkeycar形式でエクスポートする - 共通ダイアログを使用"""
        if not self.annotations:
            QMessageBox.information(self, "情報", "エクスポートするアノテーションがありません。")
            return
        
        # 共通ダイアログを表示
        export_config = self.show_unified_export_dialog("donkey")
        if not export_config:
            return  # キャンセルされた場合
        
        try:
            # deleted_indexesはactual_index（GUIでのインデックス）をそのまま渡す
            # export_to_donkey内でassigned_indexに変換される
            actual_deleted_indexes = list(self.deleted_indexes) if hasattr(self, 'deleted_indexes') and self.deleted_indexes else []

            # エクスポート実行
            catalog_path = export_to_donkey(
                export_config['output_folder'],
                self.annotations,
                inference_results=self.inference_results,
                deleted_indexes=actual_deleted_indexes,
                image_map=export_config['image_map'],
                variant_keys=export_config['variant_keys'],
                diff_vectors=self.inference_diff_vectors if hasattr(self, 'inference_diff_vectors') else None,
                waypoint_annotations=self.waypoint_annotations if hasattr(self, 'waypoint_annotations') else None
            )
            
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
                f"選択画像ソース: {', '.join(export_config['selected_variants'])}\n"
                f"保存先: {export_config['output_folder']}\n"
                f"エクスポート数: {len(self.annotations)}個"
            )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "エラー", 
                f"Donkeycarエクスポート中にエラーが発生しました: {str(e)}\n\n"
                f"詳細: {traceback.format_exc()}"
            )

    def export_to_jetracer(self):
        """Jetracer形式でエクスポートする - 共通ダイアログを使用"""
        if not self.annotations:
            QMessageBox.information(self, "情報", "エクスポートするアノテーションがありません。")
            return
        
        # 共通ダイアログを表示
        export_config = self.show_unified_export_dialog("jetracer")
        if not export_config:
            return  # キャンセルされた場合
        
        try:
            # Jetracer形式でエクスポート
            # インデックスベースのアノテーションを画像パスベースに変換
            path_based_annotations = {}
            for idx, annotation in self.annotations.items():
                if isinstance(idx, int) and 0 <= idx < len(self.images):
                    img_path = self.images[idx]
                    path_based_annotations[img_path] = annotation
                elif isinstance(idx, str):
                    path_based_annotations[idx] = annotation
            
            # 推論結果も同様に変換
            path_based_inference = {}
            if hasattr(self, 'inference_results') and self.inference_results:
                for idx, inference in self.inference_results.items():
                    if isinstance(idx, int) and 0 <= idx < len(self.images):
                        img_path = self.images[idx]
                        path_based_inference[img_path] = inference
                    elif isinstance(idx, str):
                        path_based_inference[idx] = inference
            
            catalog_path = export_to_jetracer(
                export_config['output_folder'], 
                path_based_annotations,
                inference_results=path_based_inference
            )
            
            QMessageBox.information(
                self, 
                "完了", 
                f"アノテーションをJetracer形式でエクスポートしました。\n"
                f"保存先: {export_config['output_folder']}\n"
                f"エクスポート数: {len(path_based_annotations)}個"
            )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "エラー", 
                f"Jetracerエクスポート中にエラーが発生しました: {str(e)}\n\n"
                f"詳細: {traceback.format_exc()}"
            )

    def export_annotations_to_yolo(self, train_dir, val_dir, classes):
        """バウンディングボックスアノテーションをYOLO形式でエクスポート - インデックスベース対応"""
        
        if not self.bbox_annotations:
            print("バウンディングボックスアノテーションがありません")
            return
        
        print(f"バウンディングボックスアノテーションエクスポート開始")
        print(f"アノテーション数: {len(self.bbox_annotations)}")
        
        # クラス名からインデックスへのマッピングを作成
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        print(f"クラス-インデックスマッピング: {class_to_index}")
        
        # アノテーションをランダムに学習用と検証用に分割
        annotation_indices = list(self.bbox_annotations.keys())
        random.shuffle(annotation_indices)
        
        split_point = int(len(annotation_indices) * 0.7)  # 70%を学習用
        train_indices = annotation_indices[:split_point]
        val_indices = annotation_indices[split_point:]
        
        print(f"学習用: {len(train_indices)}枚, 検証用: {len(val_indices)}枚")
        
        # 学習用データのエクスポート
        train_success = 0
        for index in train_indices:
            try:
                # インデックスから画像パスを取得
                if index >= len(self.images):
                    print(f"警告: インデックス {index} が画像リストの範囲外です")
                    continue
                    
                success = self._export_single_bbox_annotation(
                    index, train_dir, class_to_index
                )
                if success:
                    train_success += 1
            except Exception as e:
                print(f"アノテーション処理エラー {index}: {str(e)}")
        
        print(f"処理成功: {train_success}/{len(train_indices)}")
        
        # 検証用データのエクスポート
        val_success = 0
        for index in val_indices:
            try:
                # インデックスから画像パスを取得
                if index >= len(self.images):
                    print(f"警告: インデックス {index} が画像リストの範囲外です")
                    continue
                    
                success = self._export_single_bbox_annotation(
                    index, val_dir, class_to_index
                )
                if success:
                    val_success += 1
            except Exception as e:
                print(f"アノテーション処理エラー {index}: {str(e)}")
        
        print(f"処理成功: {val_success}/{len(val_indices)}")
        print(f"バウンディングボックスアノテーションエクスポート完了")


    def _export_single_bbox_annotation(self, index, output_dir, class_to_index):
        """単一のバウンディングボックスアノテーションをエクスポート（インデックスベース）"""
        
        # インデックスから画像パスを取得
        if index >= len(self.images):
            print(f"インデックス {index} が範囲外です")
            return False
        
        img_path = self.images[index]
        
        # 画像ファイルの存在確認
        if not os.path.exists(img_path):
            print(f"画像ファイルが見つかりません: {img_path}")
            return False
        
        # バウンディングボックスデータを取得
        bboxes = self.bbox_annotations.get(index, [])
        if not bboxes:
            print(f"インデックス {index} にバウンディングボックスデータがありません")
            return False
        
        # 画像ファイルをコピー
        img_filename = os.path.basename(img_path)
        dest_img_path = os.path.join(output_dir, "images", img_filename)
        
        try:
            import shutil
            shutil.copy2(img_path, dest_img_path)
        except Exception as e:
            print(f"画像コピーエラー {img_path}: {str(e)}")
            return False
        
        # ラベルファイルの作成
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(output_dir, "labels", label_filename)
        
        try:
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    # クラス名の取得（複数の形式に対応）
                    class_name = None
                    if isinstance(bbox, dict):
                        class_name = bbox.get('class') or bbox.get('class_name')
                    else:
                        class_name = getattr(bbox, 'class', None) or getattr(bbox, 'class_name', None)
                    
                    # クラスインデックスを取得
                    if not class_name or class_name not in class_to_index:
                        print(f"警告: 未知のクラス '{class_name}' をスキップします")
                        continue
                    
                    class_idx = class_to_index[class_name]
                    
                    # バウンディングボックス座標を取得（既に正規化済み）
                    if isinstance(bbox, dict):
                        # 現在の辞書形式（正規化済み）
                        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    else:
                        # 古いオブジェクト形式（ピクセル座標の場合）
                        from PIL import Image
                        with Image.open(img_path) as img:
                            img_width, img_height = img.size
                        
                        x1 = bbox.x / img_width
                        y1 = bbox.y / img_height
                        x2 = (bbox.x + bbox.width) / img_width
                        y2 = (bbox.y + bbox.height) / img_height
                        print(x1,x2,"old")
                    # YOLO形式に変換: center_x, center_y, width, height
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # YOLO形式で書き込み
                    f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            return True
            
        except Exception as e:
            print(f"ラベルファイル作成エラー {label_path}: {str(e)}")
            return False

    def export_segmentation_annotations_to_yolo(self, train_dir, val_dir, classes):
        """セグメンテーションアノテーションをYOLO形式でエクスポート - 正しいセグメンテーション形式"""
        
        if not self.segmentation_annotations:
            print("セグメンテーションアノテーションがありません")
            return
        
        print(f"セグメンテーションアノテーションエクスポート開始")
        print(f"アノテーション数: {len(self.segmentation_annotations)}")
        
        # クラス名からインデックスへのマッピングを作成
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        print(f"クラス-インデックスマッピング: {class_to_index}")
        
        # アノテーションをランダムに学習用と検証用に分割
        annotation_indices = list(self.segmentation_annotations.keys())
        import random
        random.shuffle(annotation_indices)
        
        split_point = int(len(annotation_indices) * 0.7)  # 70%を学習用
        train_indices = annotation_indices[:split_point]
        val_indices = annotation_indices[split_point:]
        
        print(f"学習用: {len(train_indices)}枚, 検証用: {len(val_indices)}枚")
        
        # 学習用データのエクスポート
        train_success = 0
        for index in train_indices:
            try:
                # インデックスから画像パスを取得
                if index >= len(self.images):
                    print(f"警告: インデックス {index} が画像リストの範囲外です")
                    continue
                    
                success = self._export_single_segmentation_annotation(
                    index, train_dir, class_to_index
                )
                if success:
                    train_success += 1
            except Exception as e:
                print(f"アノテーション処理エラー {index}: {str(e)}")
        
        print(f"学習用処理成功: {train_success}/{len(train_indices)}")
        
        # 検証用データのエクスポート
        val_success = 0
        for index in val_indices:
            try:
                # インデックスから画像パスを取得
                if index >= len(self.images):
                    print(f"警告: インデックス {index} が画像リストの範囲外です")
                    continue
                    
                success = self._export_single_segmentation_annotation(
                    index, val_dir, class_to_index
                )
                if success:
                    val_success += 1
            except Exception as e:
                print(f"アノテーション処理エラー {index}: {str(e)}")
        
        print(f"検証用処理成功: {val_success}/{len(val_indices)}")
        print(f"セグメンテーションアノテーションエクスポート完了")

    def _export_single_segmentation_annotation(self, index, output_dir, class_to_index):
        """単一のセグメンテーションアノテーションをYOLO形式でエクスポート（インデックスベース）"""
        
        # インデックスから画像パスを取得
        if index >= len(self.images):
            print(f"インデックス {index} が範囲外です")
            return False
        
        img_path = self.images[index]
        
        # 画像ファイルの存在確認
        if not os.path.exists(img_path):
            print(f"画像ファイルが見つかりません: {img_path}")
            return False
        
        # セグメンテーションデータを取得
        segmentations = self.segmentation_annotations.get(index, [])
        if not segmentations:
            print(f"インデックス {index} にセグメンテーションデータがありません")
            return False
        
        # 画像のサイズを取得
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"画像サイズ取得エラー {img_path}: {str(e)}")
            return False
        
        print(f"処理中: {os.path.basename(img_path)} ({img_width}x{img_height})")
        
        # 画像ファイルをコピー
        img_filename = os.path.basename(img_path)
        dest_img_path = os.path.join(output_dir, "images", img_filename)
        
        try:
            import shutil
            os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
            shutil.copy2(img_path, dest_img_path)
        except Exception as e:
            print(f"画像コピーエラー {img_path}: {str(e)}")
            return False
        
        # ラベルファイルの作成
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(output_dir, "labels", label_filename)
        
        try:
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            valid_annotations = 0
            with open(label_path, 'w') as f:
                for seg_idx, seg_data in enumerate(segmentations):
                    class_name = seg_data.get('class', 'unknown')
                    points = seg_data.get('points', [])
                    
                    # クラスインデックスを取得
                    if class_name not in class_to_index:
                        print(f"警告: 未知のクラス '{class_name}' をスキップします")
                        continue
                    
                    class_idx = class_to_index[class_name]
                    
                    # ポイント数の確認
                    if len(points) < 3:
                        print(f"警告: ポイント数が不足しています ({len(points)}点) - セグメンテーション {seg_idx}")
                        continue
                    
                    print(f"  セグメンテーション {seg_idx}: クラス={class_name}({class_idx}), ポイント数={len(points)}")
                    
                    # YOLO セグメンテーション形式のコーディネートに変換
                    normalized_coords = []
                    for point_idx, (px, py) in enumerate(points):
                        # 座標を正規化 (0-1の範囲)
                        norm_x = float(px) / float(img_width)
                        norm_y = float(py) / float(img_height)
                        
                        # 範囲チェック
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))
                        
                        normalized_coords.extend([norm_x, norm_y])
                        
                        if point_idx < 3:  # 最初の3点だけログ出力
                            print(f"    ポイント{point_idx}: ({px}, {py}) -> ({norm_x:.4f}, {norm_y:.4f})")
                    
                    # 最低6個の値（3点）が必要
                    if len(normalized_coords) < 6:
                        print(f"警告: 正規化座標が不足 ({len(normalized_coords)}個)")
                        continue
                    
                    # YOLO セグメンテーション形式で書き込み
                    # 形式: class_id x1 y1 x2 y2 x3 y3 ... (正規化座標)
                    coords_str = ' '.join(f"{coord:.6f}" for coord in normalized_coords)
                    line = f"{class_idx} {coords_str}\n"
                    f.write(line)
                    valid_annotations += 1
                    
                    print(f"    書き込み: クラス{class_idx} + {len(normalized_coords)}個の座標")
            
            print(f"  有効なアノテーション: {valid_annotations}個")
            
            # ファイルが空でないことを確認
            if valid_annotations == 0:
                print(f"警告: {label_filename} に有効なアノテーションがありません")
                return False
            
            return True
            
        except Exception as e:
            print(f"ラベルファイル作成エラー {label_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def browse_output_folder(self, folder_input):
        """出力フォルダを選択するダイアログを表示"""
        current_path = folder_input.text().strip()
        if not current_path:
            current_path = annotation_folder
        
        selected_folder = QFileDialog.getExistingDirectory(
            self, "エクスポート先フォルダを選択", 
            current_path,
            QFileDialog.ShowDirsOnly
        )
        
        if selected_folder:
            folder_input.setText(selected_folder)

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
        inference_check = QCheckBox("推論結果を表示する（水色丸）")
        inference_check.setChecked(self.inference_checkbox.isChecked())  # UIの設定を初期値に
        dialog_layout.addWidget(inference_check)
        
        # 差分ベクトル表示設定
        diff_vector_check = QCheckBox("差分ベクトルを表示（緑矢印）")
        diff_vector_check.setChecked(self.diff_vector_checkbox.isChecked())  # UIの設定を初期値に
        dialog_layout.addWidget(diff_vector_check)

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
        show_diff_vectors = diff_vector_check.isChecked()  # 追加
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
                    progress_callback=update_progress,
                    diff_vectors=self.inference_diff_vectors if (hasattr(self, 'inference_diff_vectors') and show_diff_vectors) else None  # 修正
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
                    images_list=selected_images,
                    diff_vectors=self.inference_diff_vectors if (hasattr(self, 'inference_diff_vectors') and show_diff_vectors) else None  # 修正
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


    ### mlflow 修正版学習ロジック自動運転モデル
    def train_and_save_model(self):
        if not self.annotations:
            QMessageBox.warning(self, "警告", "モデルを学習するにはアノテーションが必要です。")
            return
        
        # 現在選択されているモデルを取得
        model_type = self.auto_method_combo.currentText()
        
        # データにspeedキーがあるかチェック
        has_speed_data = False
        speed_count = 0
        for idx, ann in self.annotations.items():
            if 'speed' in ann or 'user/speed' in ann or 'pilot/speed' in ann:
                speed_count += 1

        if speed_count > 0:
            has_speed_data = True
            print(f"Speed data detected: {speed_count} annotations with speed information")

        # 学習設定ダイアログを表示
        training_settings = QDialog(self)
        training_settings.setWindowTitle("学習設定")
        training_settings.setMinimumWidth(700)  # 横並び用に幅を広げる
        training_settings.setMinimumHeight(600)

        settings_layout = QVBoxLayout(training_settings)

        # タブウィジェットを作成
        tabs = QTabWidget()

        # 基本設定タブ
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)

        # 初期化設定グループ（モデル選択と重みの読み込み）
        init_group = QGroupBox("初期化設定")
        init_layout = QVBoxLayout()

        # モデルアーキテクチャ選択（初期化設定の先頭に配置）
        model_select_layout = QHBoxLayout()
        model_select_label = QLabel("モデルアーキテクチャ:")
        model_type_combo = QComboBox()
        model_type_combo.setMinimumWidth(200)

        # 利用可能なモデルアーキテクチャのリストを取得
        available_architectures = list_available_models()
        model_type_combo.addItems(available_architectures)

        # 現在選択されているモデルタイプをデフォルトにする
        if model_type in available_architectures:
            model_type_combo.setCurrentText(model_type)

        model_select_layout.addWidget(model_select_label)
        model_select_layout.addWidget(model_type_combo)
        model_select_layout.addStretch()
        init_layout.addLayout(model_select_layout)

        # セパレーター的なスペース
        init_layout.addSpacing(10)

        # 利用可能なモデルファイルのリストを取得
        all_available_models = []
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pth'):
                    model_path_check = os.path.join(models_dir, f)
                    if os.path.exists(model_path_check):
                        all_available_models.append(f)
        all_available_models.sort(reverse=True)  # 新しいものが上に来るようにソート

        # 選択されたモデルタイプに基づいてフィルタリングする関数
        def get_filtered_models(selected_model_type):
            """選択されたモデルタイプに基づいてモデルファイルをフィルタリング"""
            filtered = []
            for model_file in all_available_models:
                # モデルファイル名のプレフィックスをチェック（例: ResNet18_xxx.pth, MobileNetV3_xxx.pth）
                model_name_lower = model_file.lower()
                selected_type_lower = selected_model_type.lower().replace('_', '').replace('-', '')

                # モデルタイプ名のバリエーションをチェック
                if selected_type_lower in model_name_lower.replace('_', '').replace('-', ''):
                    filtered.append(model_file)
            return filtered

        # 現在のモデルタイプに基づいてフィルタリングされたモデルリスト
        filtered_models = get_filtered_models(model_type)
        has_valid_models = len(filtered_models) > 0

        # 事前学習済みの重みがないモデルのリスト
        models_without_pretrained = ["donkeycar", "donkey_fcn"]

        def has_pretrained_weights(model_name):
            """指定されたモデルが事前学習済みの重みを持つかどうかを判定"""
            return model_name.lower() not in [m.lower() for m in models_without_pretrained]

        # ラジオボタングループ
        weights_button_group = QButtonGroup(training_settings)

        # 現在選択されているモデルが事前学習済みの重みを持つかどうか
        initial_has_pretrained = has_pretrained_weights(model_type)

        weights_radio_pretrained = QRadioButton("事前学習済みの重みを使用（推奨）")
        if initial_has_pretrained:
            weights_radio_pretrained.setChecked(True)
            weights_radio_pretrained.setToolTip("ImageNetで事前学習済みの重みを使用して学習します（転移学習）")
        else:
            weights_radio_pretrained.setEnabled(False)
            weights_radio_pretrained.setToolTip("このモデルには事前学習済みの重みがありません")
        weights_button_group.addButton(weights_radio_pretrained)
        init_layout.addWidget(weights_radio_pretrained)

        weights_radio_random = QRadioButton("ランダム初期化（スクラッチから学習）")
        weights_radio_random.setToolTip("重みをランダムに初期化して最初から学習します（学習に時間がかかります）")
        # 事前学習済みがない場合はランダム初期化をデフォルトに
        if not initial_has_pretrained:
            weights_radio_random.setChecked(True)
        weights_button_group.addButton(weights_radio_random)
        init_layout.addWidget(weights_radio_random)

        weights_radio_finetune = QRadioButton("既存モデルの重みを使用（ファインチューニング）")
        if has_valid_models:
            weights_radio_finetune.setToolTip("選択したモデルの重みを使用してファインチューニングします")
        else:
            weights_radio_finetune.setEnabled(False)
            weights_radio_finetune.setToolTip("選択したモデルタイプに対応するモデルがありません")
        weights_button_group.addButton(weights_radio_finetune)
        init_layout.addWidget(weights_radio_finetune)

        # モデル選択用のコンボボックス
        finetune_model_layout = QHBoxLayout()
        finetune_model_layout.setContentsMargins(20, 0, 0, 0)  # 左インデント
        finetune_model_label = QLabel("ベースモデル:")
        finetune_model_combo = QComboBox()
        finetune_model_combo.setMinimumWidth(300)

        # フィルタリングされたモデルでコンボボックスを初期化
        def update_finetune_model_list():
            """モデルタイプに基づいてファインチューニング用モデルリストと事前学習済みオプションを更新"""
            selected_type = model_type_combo.currentText()
            filtered = get_filtered_models(selected_type)

            # 事前学習済みの重みの有無をチェック
            has_pretrained = has_pretrained_weights(selected_type)
            if has_pretrained:
                weights_radio_pretrained.setEnabled(True)
                weights_radio_pretrained.setToolTip("ImageNetで事前学習済みの重みを使用して学習します（転移学習）")
            else:
                weights_radio_pretrained.setEnabled(False)
                weights_radio_pretrained.setToolTip("このモデルには事前学習済みの重みがありません")
                # 事前学習済みが選択されていたら、ランダム初期化に切り替え
                if weights_radio_pretrained.isChecked():
                    weights_radio_random.setChecked(True)

            finetune_model_combo.clear()
            if filtered:
                finetune_model_combo.addItems(filtered)
                # 現在選択されているモデルがリストにあればデフォルトにする
                current_model = self.model_combo.currentText()
                if current_model in filtered:
                    finetune_model_combo.setCurrentText(current_model)
                # ファインチューニングオプションを有効化
                weights_radio_finetune.setEnabled(True)
                weights_radio_finetune.setToolTip("選択したモデルの重みを使用してファインチューニングします")
            else:
                finetune_model_combo.addItem(f"{selected_type}のモデルがありません")
                # ファインチューニングオプションを無効化
                weights_radio_finetune.setEnabled(False)
                weights_radio_finetune.setToolTip(f"{selected_type}に対応するモデルがありません")
                # ファインチューニングが選択されていたら、適切なオプションに切り替え
                if weights_radio_finetune.isChecked():
                    if has_pretrained:
                        weights_radio_pretrained.setChecked(True)
                    else:
                        weights_radio_random.setChecked(True)

            # コンボボックスの有効/無効状態を更新
            toggle_finetune_model_combo()

        # 初期リストを設定
        if has_valid_models:
            finetune_model_combo.addItems(filtered_models)
            current_model = self.model_combo.currentText()
            if current_model in filtered_models:
                finetune_model_combo.setCurrentText(current_model)
        else:
            finetune_model_combo.addItem(f"{model_type}のモデルがありません")

        finetune_model_combo.setEnabled(False)  # 初期状態では無効
        finetune_model_label.setEnabled(False)

        finetune_model_layout.addWidget(finetune_model_label)
        finetune_model_layout.addWidget(finetune_model_combo)
        finetune_model_layout.addStretch()
        init_layout.addLayout(finetune_model_layout)

        # ラジオボタンの状態に応じてコンボボックスの有効/無効を切り替え
        def toggle_finetune_model_combo():
            is_finetune = weights_radio_finetune.isChecked()
            selected_type = model_type_combo.currentText()
            filtered = get_filtered_models(selected_type)
            has_models = len(filtered) > 0
            finetune_model_combo.setEnabled(is_finetune and has_models)
            finetune_model_label.setEnabled(is_finetune and has_models)

        weights_radio_pretrained.toggled.connect(toggle_finetune_model_combo)
        weights_radio_random.toggled.connect(toggle_finetune_model_combo)
        weights_radio_finetune.toggled.connect(toggle_finetune_model_combo)

        # モデルタイプが変更されたらファインチューニング用モデルリストを更新
        model_type_combo.currentIndexChanged.connect(update_finetune_model_list)

        init_group.setLayout(init_layout)
        basic_layout.addWidget(init_group)

        # 出力設定グループ（Speed出力と将来予測を統合）
        output_settings_group = QGroupBox("出力設定")
        output_settings_layout = QVBoxLayout()

        # Speed出力オプション（データにspeedがある場合のみ表示）
        speed_output_check = None
        speed_normalize_spin = None
        if has_speed_data:
            # チェックボックスと正規化設定を横並びで配置
            speed_row_layout = QHBoxLayout()

            speed_output_check = QCheckBox("Speed（速度）を出力に追加")
            speed_output_check.setChecked(False)
            speed_row_layout.addWidget(speed_output_check)

            # 正規化設定（チェックボックスの右側）
            speed_normalize_label = QLabel("正規化値:")
            speed_row_layout.addWidget(speed_normalize_label)
            speed_normalize_spin = QDoubleSpinBox()
            speed_normalize_spin.setRange(0.1, 100.0)
            speed_normalize_spin.setValue(10.0)
            speed_normalize_spin.setDecimals(1)
            speed_normalize_spin.setSingleStep(1.0)
            speed_normalize_spin.setToolTip("Speed値を正規化する際の除数（デフォルト: 10.0）")
            speed_normalize_spin.setFixedWidth(70)
            speed_row_layout.addWidget(speed_normalize_spin)

            speed_normalize_info = QLabel("※ Speed値はこの値で除算されます")
            speed_normalize_info.setStyleSheet("color: #666; font-size: 11px;")
            speed_row_layout.addWidget(speed_normalize_info)

            speed_row_layout.addStretch()
            output_settings_layout.addLayout(speed_row_layout)

            speed_info_label = QLabel(f"※ {speed_count}個のアノテーションにspeedデータが含まれています")
            speed_info_label.setStyleSheet("color: #666; font-size: 11px;")
            output_settings_layout.addWidget(speed_info_label)

            # セクション間のスペース
            output_settings_layout.addSpacing(10)

        future_output_check = QCheckBox("将来フレームの予測を出力に追加")
        future_output_check.setChecked(False)
        future_output_check.setToolTip("5, 10フレーム先のangle, throttle(, speed)を追加出力")
        output_settings_layout.addWidget(future_output_check)

        future_info_label = QLabel("※ 5フレーム先と10フレーム先のangle, throttle(, speed)を追加出力")
        future_info_label.setStyleSheet("color: #666; font-size: 11px;")
        output_settings_layout.addWidget(future_info_label)

        future_detail_label = QLabel("出力例（speed有）: [angle, throttle, speed, t+5_angle, t+5_throttle, t+5_speed, t+10_angle, t+10_throttle, t+10_speed]")
        future_detail_label.setStyleSheet("color: #888; font-size: 11px;")
        future_detail_label.setWordWrap(True)
        output_settings_layout.addWidget(future_detail_label)

        output_settings_layout.addStretch()
        output_settings_group.setLayout(output_settings_layout)
        basic_layout.addWidget(output_settings_group)

        # 学習パラメータグループ
        training_params_group = QGroupBox("学習パラメータ")
        training_params_layout = QVBoxLayout()

        # エポック数・学習率設定（同じ行に配置）
        epoch_lr_layout = QHBoxLayout()
        epoch_lr_layout.addWidget(QLabel("学習エポック数:"))
        epoch_spin = QSpinBox()
        epoch_spin.setRange(1, 1000)
        epoch_spin.setValue(30)  # デフォルト: 30エポック
        epoch_lr_layout.addWidget(epoch_spin)

        epoch_lr_layout.addWidget(QLabel("学習率:"))
        lr_combo = QComboBox()
        learning_rates = ["0.001", "0.0005", "0.0001", "0.00005", "0.00001"]
        lr_combo.addItems(learning_rates)
        lr_combo.setCurrentIndex(0)  # デフォルト: 0.001
        epoch_lr_layout.addWidget(lr_combo)

        epoch_lr_layout.addStretch()
        training_params_layout.addLayout(epoch_lr_layout)

        # Early Stopping設定（チェックボックスと忍耐エポック数を同じ行に配置）
        early_stopping_layout = QHBoxLayout()
        early_stopping_check = QCheckBox("Early Stopping")
        early_stopping_check.setChecked(False)  # デフォルト: 無効
        early_stopping_layout.addWidget(early_stopping_check)

        patience_label = QLabel("忍耐エポック数:")
        patience_label.setEnabled(False)  # 初期状態では無効
        early_stopping_layout.addWidget(patience_label)
        patience_spin = QSpinBox()
        patience_spin.setRange(1, 50)
        patience_spin.setValue(5)
        patience_spin.setEnabled(False)  # 初期状態では無効
        early_stopping_layout.addWidget(patience_spin)

        min_delta_label = QLabel("最小改善量:")
        min_delta_label.setEnabled(False)  # 初期状態では無効
        early_stopping_layout.addWidget(min_delta_label)
        min_delta_spin = QDoubleSpinBox()
        min_delta_spin.setRange(0.0, 1.0)
        min_delta_spin.setSingleStep(0.0001)
        min_delta_spin.setDecimals(4)
        min_delta_spin.setValue(0.0001)  # デフォルト: 0.0001
        min_delta_spin.setEnabled(False)  # 初期状態では無効
        early_stopping_layout.addWidget(min_delta_spin)

        early_stopping_layout.addStretch()
        training_params_layout.addLayout(early_stopping_layout)

        # Early Stoppingチェックボックスの状態に応じて設定の有効/無効を切り替え
        def update_early_stopping_ui():
            enabled = early_stopping_check.isChecked()
            patience_label.setEnabled(enabled)
            patience_spin.setEnabled(enabled)
            min_delta_label.setEnabled(enabled)
            min_delta_spin.setEnabled(enabled)

        early_stopping_check.toggled.connect(update_early_stopping_ui)

        # バッチサイズ・検証データ割合・Weight Decay設定（同じ行に配置）
        batch_val_wd_layout = QHBoxLayout()

        batch_val_wd_layout.addWidget(QLabel("バッチサイズ:"))
        batch_size_combo = QComboBox()
        batch_sizes = ["8", "16", "32", "64", "128", "256"]
        batch_size_combo.addItems(batch_sizes)
        batch_size_combo.setCurrentIndex(2)  # デフォルト: 32
        batch_val_wd_layout.addWidget(batch_size_combo)

        batch_val_wd_layout.addWidget(QLabel("検証データ割合:"))
        val_split_spin = QDoubleSpinBox()
        val_split_spin.setRange(0.1, 0.5)
        val_split_spin.setSingleStep(0.05)
        val_split_spin.setDecimals(2)
        val_split_spin.setValue(0.2)  # デフォルト: 20%
        val_split_spin.setToolTip("学習データから検証用に分割する割合")
        batch_val_wd_layout.addWidget(val_split_spin)

        batch_val_wd_layout.addWidget(QLabel("Weight Decay:"))
        weight_decay_combo = QComboBox()
        weight_decays = ["0", "1e-5", "1e-4", "1e-3", "1e-2"]
        weight_decay_combo.addItems(weight_decays)
        weight_decay_combo.setCurrentIndex(2)  # デフォルト: 1e-4
        weight_decay_combo.setToolTip("L2正則化の強さ（過学習防止）")
        batch_val_wd_layout.addWidget(weight_decay_combo)

        batch_val_wd_layout.addStretch()
        training_params_layout.addLayout(batch_val_wd_layout)

        # Optimizer・Scheduler設定（同じ行に配置）
        optimizer_scheduler_layout = QHBoxLayout()

        optimizer_scheduler_layout.addWidget(QLabel("Optimizer:"))
        optimizer_combo = QComboBox()
        optimizers = ["Adam", "AdamW", "SGD"]
        optimizer_combo.addItems(optimizers)
        optimizer_combo.setCurrentIndex(0)  # デフォルト: Adam
        optimizer_combo.setToolTip("Adam: 汎用的, AdamW: Weight Decay改良版, SGD: 古典的だが安定")
        optimizer_scheduler_layout.addWidget(optimizer_combo)

        optimizer_scheduler_layout.addWidget(QLabel("Scheduler:"))
        scheduler_combo = QComboBox()
        schedulers = ["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "None"]
        scheduler_combo.addItems(schedulers)
        scheduler_combo.setCurrentIndex(0)  # デフォルト: ReduceLROnPlateau
        scheduler_combo.setToolTip("ReduceLROnPlateau: 損失停滞時に学習率低下, StepLR: 固定ステップで低下, CosineAnnealing: コサイン曲線で調整")
        optimizer_scheduler_layout.addWidget(scheduler_combo)

        optimizer_scheduler_layout.addStretch()
        training_params_layout.addLayout(optimizer_scheduler_layout)

        training_params_group.setLayout(training_params_layout)
        basic_layout.addWidget(training_params_group)

        # 学習対象データ選択グループボックス
        data_selection_group = QGroupBox("学習データ選択")
        data_selection_layout = QVBoxLayout()

        # データ選択オプション
        data_radio_all = QRadioButton("すべてのアノテーションデータを使用")
        data_radio_all.setChecked(True)  # デフォルトですべて使用
        data_selection_layout.addWidget(data_radio_all)

        # スキップ設定（ラジオボタンとスピンボックスを同じ行に配置）
        skip_layout = QHBoxLayout()
        data_radio_skip = QRadioButton("スキップ設定でデータを間引く")
        skip_layout.addWidget(data_radio_skip)
        skip_layout.addWidget(QLabel("スキップ枚数:"))
        custom_skip_spin = QSpinBox()
        custom_skip_spin.setRange(2, 100)
        custom_skip_spin.setValue(5)  # デフォルト: 5枚
        custom_skip_spin.setEnabled(False)  # 初期状態では無効
        skip_layout.addWidget(custom_skip_spin)
        skip_layout.addStretch()
        data_selection_layout.addLayout(skip_layout)

        # インデックス範囲指定（ラジオボタンとスピンボックスを同じ行に配置）
        range_layout = QHBoxLayout()
        data_radio_range = QRadioButton("インデックス範囲を指定")
        range_layout.addWidget(data_radio_range)

        # インデックス範囲の入力フィールド
        range_start_spin = QSpinBox()
        range_start_spin.setRange(0, 99999)
        range_start_spin.setValue(0)
        range_start_spin.setEnabled(False)  # 初期状態では無効

        range_end_spin = QSpinBox()
        range_end_spin.setRange(0, 99999)
        # 画像の最大インデックスを設定
        max_index = len(self.images) - 1 if hasattr(self, 'images') and self.images else 0
        range_end_spin.setValue(max_index)
        range_end_spin.setEnabled(False)  # 初期状態では無効

        range_layout.addWidget(range_start_spin)
        range_layout.addWidget(QLabel("〜"))
        range_layout.addWidget(range_end_spin)
        range_layout.addStretch()
        data_selection_layout.addLayout(range_layout)

        # データサンプル数の表示ラベル
        data_sample_label = QLabel("")
        data_selection_layout.addWidget(data_sample_label)

        # ダウンサンプリング除外チェックボックス
        exclude_downsampled_check = QCheckBox("ダウンサンプリング対象を除外")
        exclude_downsampled_check.setChecked(True)  # デフォルトでON
        downsampled_count = len(getattr(self, 'downsampled_indexes', []))
        exclude_downsampled_check.setToolTip(f"直進時などのダウンサンプリング対象データ（現在{downsampled_count}件）を学習から除外します")
        if downsampled_count == 0:
            exclude_downsampled_check.setEnabled(False)
            exclude_downsampled_check.setText("ダウンサンプリング対象を除外 (0件)")
        else:
            exclude_downsampled_check.setText(f"ダウンサンプリング対象を除外 ({downsampled_count}件)")
        data_selection_layout.addWidget(exclude_downsampled_check)

        # ラジオボタンの状態に応じて各設定欄の有効/無効を切り替える
        def update_data_selection_ui():
            # スキップ設定の有効/無効
            custom_skip_spin.setEnabled(data_radio_skip.isChecked())

            # インデックス範囲設定の有効/無効
            range_start_spin.setEnabled(data_radio_range.isChecked())
            range_end_spin.setEnabled(data_radio_range.isChecked())

            # ダウンサンプリング除外設定を取得
            exclude_downsampled = exclude_downsampled_check.isChecked()
            downsampled_set = set(getattr(self, 'downsampled_indexes', []))

            # サンプル数の計算と表示（削除済み・ダウンサンプリングを考慮）
            if data_radio_all.isChecked():
                # アノテーション総数と除外数を計算
                total_annotations = len(self.annotations)

                # 削除済み・ダウンサンプリングをカウント
                deleted_count = 0
                ds_count = 0
                for idx in self.annotations.keys():
                    if hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes:
                        deleted_count += 1
                    elif exclude_downsampled and idx in downsampled_set:
                        ds_count += 1
                excluded_count = deleted_count + ds_count
                sample_count = total_annotations - excluded_count

                exclude_info = f"削除済み{deleted_count}枚"
                if exclude_downsampled and ds_count > 0:
                    exclude_info += f" + DS{ds_count}枚"
                data_sample_label.setText(f"<b>使用データ数: {sample_count}枚</b> (全{total_annotations}枚 - {exclude_info})")
                data_sample_label.setStyleSheet("color: #2E7D32; font-weight: bold; font-size: 13px;")
            elif data_radio_skip.isChecked():
                skip = custom_skip_spin.value()
                total_skipped = 0
                deleted_count = 0
                ds_count = 0

                for idx in self.annotations.keys():
                    if idx % skip == 0:
                        total_skipped += 1
                        if hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes:
                            deleted_count += 1
                        elif exclude_downsampled and idx in downsampled_set:
                            ds_count += 1
                excluded_count = deleted_count + ds_count
                sample_count = total_skipped - excluded_count

                exclude_info = f"削除済み{deleted_count}枚"
                if exclude_downsampled and ds_count > 0:
                    exclude_info += f" + DS{ds_count}枚"
                data_sample_label.setText(f"<b>使用データ数: {sample_count}枚</b> ({skip}枚ごと、対象{total_skipped}枚 - {exclude_info})")
                data_sample_label.setStyleSheet("color: #2E7D32; font-weight: bold; font-size: 13px;")
            elif data_radio_range.isChecked():
                start = range_start_spin.value()
                end = range_end_spin.value()
                total_in_range = 0
                deleted_count = 0
                ds_count = 0

                for idx in self.annotations:
                    if start <= idx <= end:
                        total_in_range += 1
                        if hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes:
                            deleted_count += 1
                        elif exclude_downsampled and idx in downsampled_set:
                            ds_count += 1
                excluded_count = deleted_count + ds_count
                sample_count = total_in_range - excluded_count

                exclude_info = f"削除済み{deleted_count}枚"
                if exclude_downsampled and ds_count > 0:
                    exclude_info += f" + DS{ds_count}枚"
                data_sample_label.setText(f"<b>使用データ数: {sample_count}枚</b> (範囲{start}-{end}、対象{total_in_range}枚 - {exclude_info})")
                data_sample_label.setStyleSheet("color: #2E7D32; font-weight: bold; font-size: 13px;")

        # ラジオボタンの状態変更イベントを接続
        data_radio_all.toggled.connect(update_data_selection_ui)
        data_radio_skip.toggled.connect(update_data_selection_ui)
        data_radio_range.toggled.connect(update_data_selection_ui)

        # スピンボックスの値変更イベントを接続
        custom_skip_spin.valueChanged.connect(update_data_selection_ui)
        range_start_spin.valueChanged.connect(update_data_selection_ui)
        range_end_spin.valueChanged.connect(update_data_selection_ui)

        # ダウンサンプリング除外チェックボックスの変更イベントを接続
        exclude_downsampled_check.toggled.connect(update_data_selection_ui)

        # 初期表示を設定
        update_data_selection_ui()

        data_selection_group.setLayout(data_selection_layout)
        basic_layout.addWidget(data_selection_group)

        # タブに追加
        tabs.addTab(basic_tab, "基本設定")
        
        # データオーグメンテーションタブ
        aug_tab = QWidget()
        aug_layout = QVBoxLayout(aug_tab)
        
        # データオーグメンテーション有効化チェックボックス
        aug_enable_check = QCheckBox("データオーグメンテーションを有効にする")
        aug_enable_check.setChecked(False)  # デフォルトオフ
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

        # モデル名とコメント欄を追加
        settings_layout.addWidget(QLabel(""))  # スペース追加

        # モデル名編集欄
        model_name_group = QGroupBox("モデル名設定")
        model_name_layout = QVBoxLayout(model_name_group)

        # プレフィックス（固定）とサフィックス（編集可能）を分離
        # 自動運転モデルの場合はmodel_typeをプレフィックスとする
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # プレフィックスとサフィックスを横並びで表示
        name_input_layout = QHBoxLayout()
        name_input_layout.addWidget(QLabel("モデル名:"))

        # プレフィックス（固定、編集不可）- 動的に更新される
        prefix_label = QLabel(f"{model_type}_")
        prefix_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; font-family: monospace;")
        name_input_layout.addWidget(prefix_label)

        # サフィックス（編集可能）
        model_name_suffix_input = QLineEdit()
        model_name_suffix_input.setText(timestamp)
        model_name_suffix_input.setPlaceholderText("カスタム名を入力")
        name_input_layout.addWidget(model_name_suffix_input)

        model_name_layout.addLayout(name_input_layout)

        model_name_note = QLabel(f"※ モデルタイプ ({model_type}) のプレフィックスは変更できません。.pthは自動的に付与されます")
        model_name_note.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        model_name_layout.addWidget(model_name_note)

        # モデルタイプが変更されたらプレフィックスと注釈を更新
        def update_model_name_prefix():
            selected_type = model_type_combo.currentText()
            prefix_label.setText(f"{selected_type}_")
            model_name_note.setText(f"※ モデルタイプ ({selected_type}) のプレフィックスは変更できません。.pthは自動的に付与されます")

        model_type_combo.currentIndexChanged.connect(update_model_name_prefix)

        settings_layout.addWidget(model_name_group)

        # コメント欄
        comment_group = QGroupBox("学習コメント (MLflowに記録)")
        comment_layout = QVBoxLayout(comment_group)

        comment_layout.addWidget(QLabel("コメント:"))
        comment_input = QPlainTextEdit()
        comment_input.setPlaceholderText("この学習についてのメモやコメントを入力してください (任意)")
        comment_input.setMaximumHeight(80)
        comment_layout.addWidget(comment_input)

        settings_layout.addWidget(comment_group)

        # ボタンの配置
        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        start_button = button_box.addButton("Start training", QDialogButtonBox.AcceptRole)
        button_box.accepted.connect(training_settings.accept)
        button_box.rejected.connect(training_settings.reject)
        settings_layout.addWidget(button_box)

        # ダイアログを表示
        if not training_settings.exec_():
            return
        
        # 設定値の取得
        # ダイアログで選択されたモデルタイプを使用
        model_type = model_type_combo.currentText()
        autonomous_prefix = f"{model_type}_"

        num_epochs = epoch_spin.value()
        use_early_stopping = early_stopping_check.isChecked()
        patience = patience_spin.value() if use_early_stopping else 0
        min_delta = min_delta_spin.value() if use_early_stopping else 0.0
        learning_rate = float(lr_combo.currentText())

        # バッチサイズ
        user_batch_size = int(batch_size_combo.currentText())

        # 検証データ割合
        val_split = val_split_spin.value()

        # Weight Decay
        weight_decay_text = weight_decay_combo.currentText()
        weight_decay = float(weight_decay_text) if weight_decay_text != "0" else 0.0

        # Optimizer・Scheduler
        optimizer_name = optimizer_combo.currentText()
        scheduler_name = scheduler_combo.currentText()

        model_name = autonomous_prefix + model_name_suffix_input.text().strip()
        comment = comment_input.toPlainText().strip()

        # モデル重み設定の取得
        use_pretrained = weights_radio_pretrained.isChecked()  # 事前学習済みの重みを使用
        use_random_init = weights_radio_random.isChecked()  # ランダム初期化
        load_weights = weights_radio_finetune.isChecked()  # ファインチューニング
        selected_finetune_model = finetune_model_combo.currentText() if load_weights else None
        # ファインチューニング用モデルが有効かどうかを確認
        filtered_models_for_check = get_filtered_models(model_type)
        has_valid_finetune_models = len(filtered_models_for_check) > 0 and selected_finetune_model in filtered_models_for_check
        model_path = os.path.join(models_dir, selected_finetune_model) if load_weights and has_valid_finetune_models else None

        # Speed出力設定の取得
        use_speed_output = False
        speed_normalize_value = 10.0
        if has_speed_data and speed_output_check is not None:
            use_speed_output = speed_output_check.isChecked()
            if speed_normalize_spin is not None:
                speed_normalize_value = speed_normalize_spin.value()

        # 将来予測出力設定の取得
        use_future_output = future_output_check.isChecked()

        # データ選択設定の取得
        use_all = data_radio_all.isChecked()
        use_skip = data_radio_skip.isChecked()
        use_range = data_radio_range.isChecked()

        skip_count = custom_skip_spin.value() if use_skip else 1
        range_start = range_start_spin.value() if use_range else 0
        range_end = range_end_spin.value() if use_range else (len(self.images) - 1)

        # ダウンサンプリング除外設定の取得
        exclude_downsampled = exclude_downsampled_check.isChecked()

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

        try:
            # 学習データの準備（データ選択設定を適用）
            image_paths = []
            downsampled_set = set(getattr(self, 'downsampled_indexes', []))

            for idx in self.annotations.keys():
                # 削除済みインデックスをスキップ（actual_indexで判定）
                if hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes:
                    continue

                # ダウンサンプリング対象をスキップ（チェックボックスがONの場合）
                if exclude_downsampled and idx in downsampled_set:
                    continue

                if isinstance(idx, int) and 0 <= idx < len(self.images):
                    # データ選択条件に基づいてフィルタリング
                    if use_all:
                        # 全データ使用
                        image_paths.append(self.images[idx])
                    elif use_skip:
                        # スキップ設定による間引き
                        if idx % skip_count == 0:
                            image_paths.append(self.images[idx])
                    elif use_range:
                        # インデックス範囲フィルタリング
                        if range_start <= idx <= range_end:
                            image_paths.append(self.images[idx]) 
                                            
            if not image_paths:
                QMessageBox.warning(self, "警告", "学習データがありません。")
                return
            
            # 対応するアノテーション値を取得
            annotation_values = []
            for img_path in image_paths:
                # パスからインデックスを逆引き
                idx = self.images.index(img_path)
                if idx in self.annotations:
                    # ディープコピーして元のデータを変更しないようにする
                    annotation_values.append(deepcopy(self.annotations[idx]))

            # Speed値の正規化（speedが含まれている場合）
            if use_speed_output and speed_normalize_value > 0:
                for annotation in annotation_values:
                    if 'speed' in annotation:
                        # speed値を正規化値で除算
                        annotation['speed'] = annotation['speed'] / speed_normalize_value

            # データ数の確認とバッチサイズの調整
            batch_size = min(user_batch_size, len(image_paths))
            if batch_size < 2:
                QMessageBox.warning(self, "警告", "データ数が不足しています。最低2枚の画像が必要です。")
                return

            # デバッグ出力: 学習設定
            print("\n" + "="*60)
            print("【学習開始】トレーニング設定")
            print("="*60)
            print(f"[モデル設定]")
            print(f"  モデルアーキテクチャ: {model_type}")
            print(f"  モデル名: {model_name}")
            print(f"  初期化方法: {'事前学習済み' if use_pretrained else ('ランダム初期化' if use_random_init else 'ファインチューニング')}")
            if load_weights and selected_finetune_model:
                print(f"  ファインチューニング元: {selected_finetune_model}")
            print(f"[学習パラメータ]")
            print(f"  エポック数: {num_epochs}")
            print(f"  学習率: {learning_rate}")
            print(f"  Weight Decay: {weight_decay}")
            print(f"  バッチサイズ: {batch_size}")
            print(f"  検証データ割合: {val_split:.0%}")
            print(f"  Optimizer: {optimizer_name}")
            print(f"  Scheduler: {scheduler_name}")
            print(f"[Early Stopping]")
            print(f"  有効: {use_early_stopping}")
            if use_early_stopping:
                print(f"  忍耐エポック数: {patience}")
                print(f"  最小改善量: {min_delta}")
            print(f"[出力設定]")
            print(f"  Speed出力: {use_speed_output}")
            print(f"  将来予測出力: {use_future_output}")
            print(f"[データ設定]")
            print(f"  学習データ数: {len(image_paths)}枚")
            print(f"  データ選択: {'全て' if use_all else ('スキップ' if use_skip else 'インデックス範囲')}")
            if use_skip:
                print(f"  スキップ枚数: {skip_count}")
            if use_range:
                print(f"  範囲: {range_start} - {range_end}")
            print(f"[オーグメンテーション]")
            print(f"  有効: {augmentation_params['enabled']}")
            if augmentation_params['enabled']:
                print(f"  設定: {augmentation_params}")
            print("="*60 + "\n")

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
                        
            # 出力数を決定
            # 基本: angle, throttle = 2
            # speed追加: +1 = 3
            # 将来予測追加: use_speed時は+6、use_speed無し時は+4
            base_outputs = 3 if use_speed_output else 2
            if use_future_output:
                # 将来予測を含める場合
                # use_speed=True: 5フレーム先(angle,throttle,speed) + 10フレーム先(angle,throttle,speed) = +6
                # use_speed=False: 5フレーム先(angle,throttle) + 10フレーム先(angle,throttle) = +4
                future_outputs_per_frame = 3 if use_speed_output else 2
                num_outputs = base_outputs + (future_outputs_per_frame * 2)  # 2フレーム分（t+5, t+10）
            else:
                num_outputs = base_outputs

            # データセットの作成（バッチサイズと詳細オーグメンテーション設定を明示的に指定）
            train_loader, val_loader, dataset_info = create_datasets(
                image_paths=image_paths,
                annotations=annotation_values,
                model_name=model_type,
                use_augmentation=augmentation_params if augmentation_params['enabled'] else False,
                batch_size=batch_size,  # バッチサイズ
                val_split=val_split,  # 検証データ割合
                use_speed=use_speed_output,  # Speed出力を使用するかどうか
                use_future=use_future_output,  # 将来予測出力を使用するかどうか
                num_outputs=num_outputs  # 出力数を指定
            )

            # 最初の画像から実際のサイズを取得
            sample_img_path = image_paths[0]
            sample_img = Image.open(sample_img_path)
            input_size = (sample_img.height, sample_img.width)  # 高さ、幅の順

            progress.setLabelText(f"入力サイズ: {input_size} で学習準備中...")
            progress.setValue(10)
            QApplication.processEvents()

            # モデルの学習 - 初期化設定に基づいてパラメータを設定
            # - 事前学習済み: pretrained=True, model_path=None
            # - ランダム初期化: pretrained=False, model_path=None
            # - ファインチューニング: pretrained=False, model_path=指定されたパス
            training_results = train_model(
                model_name=model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                save_dir=models_dir,
                progress_callback=update_progress,
                pretrained=use_pretrained,  # 事前学習済みの重みを使用するか
                model_path=model_path if load_weights else None,  # ファインチューニングの場合はパスを指定
                num_epochs=num_epochs,  # 指定されたエポック数
                learning_rate=learning_rate,  # 指定された学習率
                weight_decay=weight_decay,  # L2正則化
                use_early_stopping=use_early_stopping,  # Early Stoppingの有効/無効
                patience=patience,  # 忍耐値
                min_delta=min_delta,  # 最小改善量
                optimizer_name=optimizer_name,  # 最適化アルゴリズム
                scheduler_name=scheduler_name,  # 学習率スケジューラ
                custom_model_name=model_name if model_name else None,  # カスタムモデル名
                num_outputs=num_outputs  # 出力数を指定
            )
            
            progress.close()

            # キャンセルされた場合の処理
            if training_results.get('cancelled', False):
                QMessageBox.information(
                    self,
                    "学習キャンセル",
                    f"モデル学習がキャンセルされました。\n\n"
                    f"完了したエポック数: {training_results.get('completed_epochs', 0)}/{num_epochs}"
                )
                self.statusBar().showMessage("学習がキャンセルされました", 5000)
                return

            # MLflowに結果を記録 - 統合版
            mlflow_info = self._log_autonomous_driving_training(
                model_type=model_type,
                training_results=training_results,
                training_params={
                    "model_type": model_type,
                    "num_epochs": num_epochs,
                    "completed_epochs": training_results.get('completed_epochs', num_epochs),
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "val_split": val_split,
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "use_early_stopping": use_early_stopping,
                    "patience": patience if use_early_stopping else 0,
                    "min_delta": min_delta if use_early_stopping else 0.0,
                    "early_stopped": training_results.get('early_stopped', False),
                    "initial_weights": "fine-tuned" if load_weights else ("random" if use_random_init else "pretrained"),
                    "pretrained_model_name": selected_finetune_model if load_weights else None,
                    "augmentation_enabled": augmentation_params['enabled'],
                    "sampling_strategy": self._get_sampling_strategy_name(use_all, use_skip, use_range, skip_count),
                    "augmentation_params": augmentation_params,
                    "data_folder": self.folder_path if hasattr(self, 'folder_path') and self.folder_path else "unknown",
                    "model_name": model_name,
                    "comment": comment
                },
                dataset_info={
                    "total_annotations": len(self.annotations),
                    "used_samples": len(image_paths),
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "input_shape": input_size,
                    "deleted_samples": len(getattr(self, 'deleted_indexes', []))
                },
                image_paths=image_paths
            )
            
            # 成功メッセージを表示
            self._show_training_success_message(
                model_type=model_type,
                training_results=training_results,
                training_params={
                    "model_type": model_type,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "val_split": val_split,
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "use_early_stopping": use_early_stopping,
                    "patience": patience,
                    "min_delta": min_delta,
                    "load_weights": load_weights,
                    "use_random_init": use_random_init,
                    "selected_model": selected_finetune_model if load_weights else None,
                    "data_folder": os.path.basename(self.folder_path) if hasattr(self, 'folder_path') and self.folder_path else "unknown"
                },
                dataset_info={
                    "image_paths_count": len(image_paths),
                    "input_size": input_size,
                    "sampling_info": self._get_sampling_info(use_all, use_skip, use_range, skip_count, range_start, range_end)
                },
                augmentation_params=augmentation_params,
                mlflow_info=mlflow_info
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

    def _log_autonomous_driving_training(self, model_type, training_results, training_params, dataset_info, image_paths):
        """自動運転モデルの学習結果をMLflowに記録"""
        
        try:
            # MLflowManagerが初期化されていない場合は初期化
            if not hasattr(self, 'mlflow_manager'):
                self.mlflow_manager = MLflowManager(self.folder_path)
            
            # メトリクスを準備
            metrics = {
                "best_val_loss": training_results.get('best_val_loss', 0.0),
                "final_train_loss": training_results['train_losses'][-1] if 'train_losses' in training_results else 0.0,
                "final_val_loss": training_results['val_losses'][-1] if 'val_losses' in training_results else 0.0,
                "train_losses": training_results.get('train_losses', []),
                "val_losses": training_results.get('val_losses', []),
                "status": "early_stopped" if training_results.get('early_stopped', False) else "completed"
            }
            
            # 自動運転特有のメトリクスを追加（可能であれば）
            if 'steering_accuracy' in training_results:
                metrics["steering_accuracy"] = training_results['steering_accuracy']
            if 'throttle_accuracy' in training_results:
                metrics["throttle_accuracy"] = training_results['throttle_accuracy']
            if 'steering_mae' in training_results:
                metrics["steering_mae"] = training_results['steering_mae']
            if 'throttle_mae' in training_results:
                metrics["throttle_mae"] = training_results['throttle_mae']
            
            # MLflowに記録
            success = self.mlflow_manager.log_autonomous_driving_model(
                model_path=training_results['best_model_path'],
                training_params=training_params,
                metrics=metrics,
                dataset_info=dataset_info
            )
            
            if success:
                return "MLflowに学習履歴を記録しました。\n「MLflow比較」ボタンで結果を確認できます。"
            else:
                return "MLflowへの記録中にエラーが発生しました。"
                
        except ImportError:
            return "MLflowがインストールされていないため、学習履歴は記録されませんでした。\npip install mlflow でインストールできます。"
        except Exception as e:
            print(f"MLflow記録エラー: {e}")
            return f"MLflowへの記録中にエラーが発生しました: {str(e)}"

    def _get_sampling_strategy_name(self, use_all, use_skip, use_range, skip_count):
        """サンプリング戦略の名前を取得"""
        if use_all:
            return "all"
        elif use_skip:
            return f"skip_{skip_count}"
        elif use_range:
            return "range"
        return "unknown"

    def _get_sampling_info(self, use_all, use_skip, use_range, skip_count, range_start, range_end):
        """サンプリング情報の文字列を取得"""
        if use_all:
            total_annotations = len(self.annotations)
            excluded_count = sum(1 for idx in self.annotations if idx in getattr(self, 'deleted_indexes', []))
            used_count = total_annotations - excluded_count
            return f"すべて使用 ({used_count}/{total_annotations}枚使用, 削除済み{excluded_count}枚を除外)"
        elif use_skip:
            total_annotations = len(self.annotations)
            valid_indices = [idx for idx in self.annotations if idx not in getattr(self, 'deleted_indexes', [])]
            sampled_count = len([idx for idx in valid_indices if idx % skip_count == 0])
            excluded_count = sum(1 for idx in self.annotations if idx in getattr(self, 'deleted_indexes', []))
            return f"{skip_count}枚ごとに1枚 ({sampled_count}/{total_annotations}枚使用, 削除済み{excluded_count}枚を除外)"
        elif use_range:
            in_range_count = sum(1 for idx in self.annotations if range_start <= idx <= range_end)
            excluded_count = sum(1 for idx in self.annotations if range_start <= idx <= range_end and idx in getattr(self, 'deleted_indexes', []))
            sample_count = in_range_count - excluded_count
            total_count = len(self.annotations)
            return f"インデックス範囲 {range_start}～{range_end} ({sample_count}/{total_count}枚使用, 削除済み{excluded_count}枚を除外)"
        return "不明"

    def _show_training_success_message(self, model_type, training_results, training_params, dataset_info, augmentation_params, mlflow_info):
        """学習成功メッセージを表示"""
        
        # オーグメンテーション情報を生成
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
        
        # 初期重みの情報
        weights_info = ""
        if training_params['load_weights']:
            weights_info = f"初期重み: {training_params['selected_model']} から読み込み\n"
        elif training_params.get('use_random_init', False):
            weights_info = "初期重み: ランダム初期化（スクラッチ）\n"
        else:
            weights_info = "初期重み: 事前学習済みモデル\n"
        
        # Early Stopping情報
        early_stopping_info = ""
        if training_params['use_early_stopping']:
            if training_results.get('early_stopped', False):
                early_stopping_info = f"Early Stopping: {training_results.get('stopped_epoch', 0)}エポックで停止\n"
            else:
                early_stopping_info = f"Early Stopping: 発動せず (忍耐値: {training_params['patience']})\n"
        
        # 入力サイズ情報
        input_size = dataset_info['input_size']
        input_size_info = f"入力サイズ: {input_size[0]}x{input_size[1]} (H x W)\n"
        
        # 学習時間情報
        time_info = ""
        if 'total_training_time' in training_results:
            from model_training import format_time
            total_time_str = format_time(training_results['total_training_time'])
            avg_epoch_time_str = format_time(training_results.get('avg_epoch_time', 0))
            time_info = f"学習時間: {total_time_str} (平均エポック時間: {avg_epoch_time_str})\n"
        
        # 成功メッセージを表示
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("学習完了")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(
            f"{model_type} モデルを学習し保存しました: {os.path.basename(training_results['model_path'])}\n" +
            f"最良検証損失: {training_results['best_val_loss']:.6f}\n" +
            f"実施エポック数: {training_results.get('completed_epochs', training_params['num_epochs'])}/{training_params['num_epochs']}\n" +
            early_stopping_info +
            time_info +
            f"学習データ数: {dataset_info['image_paths_count']}枚 {dataset_info['sampling_info']}\n" +
            input_size_info +
            weights_info +
            f"学習率: {training_params['learning_rate']}, Weight Decay: {training_params['weight_decay']}\n" +
            f"バッチサイズ: {training_params['batch_size']}, 検証データ割合: {training_params['val_split']:.0%}\n" +
            f"Optimizer: {training_params['optimizer']}, Scheduler: {training_params['scheduler']}\n" +
            aug_details +
            f"\n{mlflow_info}"
        )

        # OKボタン
        ok_button = msg_box.addButton(QMessageBox.Ok)

        # MLflow を開くボタンを追加
        mlflow_button = msg_box.addButton("MLflowを開く", QMessageBox.ActionRole)

        msg_box.exec_()

        # MLflowボタンがクリックされた場合
        if msg_box.clickedButton() == mlflow_button:
            self.mlflow_manager.open_ui()
        

    ###

    #TODO:annotationsのpathからindexへ変更
    def auto_annotate(self):
        """オートアノテーションを実行する - 詳細な進捗表示付き"""
        if not self.annotations:
            QMessageBox.warning(self, "警告", "オートアノテーションを実行するには、まず数枚の画像に手動でアノテーションを行ってください。")
            return

        # 範囲指定ダイアログを表示
        range_dialog = QDialog(self)
        range_dialog.setWindowTitle("オートアノテーション範囲指定")
        range_dialog.setMinimumWidth(400)

        layout = QVBoxLayout(range_dialog)

        # 説明ラベル
        info_label = QLabel("オートアノテーションを実行する範囲を指定してください。")
        layout.addWidget(info_label)

        # 範囲選択
        range_group = QGroupBox("範囲")
        range_layout = QVBoxLayout(range_group)

        # ラジオボタン
        all_radio = QRadioButton("すべてのアノテーションされていない画像")
        all_radio.setChecked(True)
        range_radio = QRadioButton("インデックス範囲を指定")

        range_layout.addWidget(all_radio)
        range_layout.addWidget(range_radio)

        # インデックス範囲入力
        index_layout = QHBoxLayout()
        index_layout.addWidget(QLabel("開始:"))
        start_spin = QSpinBox()
        start_spin.setRange(0, len(self.images) - 1)
        start_spin.setValue(0)
        start_spin.setEnabled(False)
        index_layout.addWidget(start_spin)

        # 現在位置ボタン（開始）
        start_current_button = QPushButton("現在位置")
        start_current_button.setEnabled(False)
        start_current_button.setToolTip("現在表示中の画像インデックスを設定")
        start_current_button.clicked.connect(lambda: start_spin.setValue(self.current_index))
        index_layout.addWidget(start_current_button)

        index_layout.addWidget(QLabel("終了:"))
        end_spin = QSpinBox()
        end_spin.setRange(0, len(self.images) - 1)
        end_spin.setValue(len(self.images) - 1)
        end_spin.setEnabled(False)
        index_layout.addWidget(end_spin)

        # 現在位置ボタン（終了）
        end_current_button = QPushButton("現在位置")
        end_current_button.setEnabled(False)
        end_current_button.setToolTip("現在表示中の画像インデックスを設定")
        end_current_button.clicked.connect(lambda: end_spin.setValue(self.current_index))
        index_layout.addWidget(end_current_button)

        range_layout.addLayout(index_layout)
        layout.addWidget(range_group)

        # ラジオボタンの状態変化でスピンボックスとボタンを有効/無効化
        def on_range_radio_toggled(checked):
            start_spin.setEnabled(checked)
            end_spin.setEnabled(checked)
            start_current_button.setEnabled(checked)
            end_current_button.setEnabled(checked)

        range_radio.toggled.connect(on_range_radio_toggled)

        # OK/キャンセルボタン
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(range_dialog.accept)
        button_box.rejected.connect(range_dialog.reject)
        layout.addWidget(button_box)

        # ダイアログ実行
        if range_dialog.exec_() != QDialog.Accepted:
            return

        # 範囲に基づいてアノテーション対象画像を取得
        if all_radio.isChecked():
            # すべてのアノテーションされていない画像
            unannotated_images = [img for img in self.images if img not in self.annotations]
        else:
            # 指定範囲のアノテーションされていない画像
            start_idx = start_spin.value()
            end_idx = end_spin.value()
            if start_idx > end_idx:
                QMessageBox.warning(self, "警告", "開始インデックスは終了インデックス以下である必要があります。")
                return

            range_images = self.images[start_idx:end_idx + 1]
            unannotated_images = [img for img in range_images if img not in self.annotations]

        if not unannotated_images:
            QMessageBox.information(self, "情報", "指定範囲にアノテーションされていない画像がありません。")
            return
        
        # 選択された学習方法（モデル）を取得
        model_type = self.auto_method_combo.currentText()
        selected_model = self.model_combo.currentText()
        
        # モデルのパスを取得
        model_path = None
        if hasattr(self, 'model_combo') and selected_model not in ["モデルが見つかりません", "フォルダを選択してください"] and "が見つかりません" not in selected_model:
            # models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
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
                            self.location_annotations[self.current_index] = loc_value
                            
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
                self.display_current_image()
                self.update_gallery()
                self.update_distribution_graph()

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

    def yolo_auto_annotate(self):
        """YOLOを使用した物体検知・セグメンテーションのオートアノテーション"""
        from utils.yolo_utils import get_yolo_model, batch_detect_objects_and_segments

        if not self.images:
            QMessageBox.warning(self, "警告", "画像が読み込まれていません。")
            return

        # モデルタイプを先に判定
        model_type = "detect"
        if hasattr(self, 'yolo_seg_model') and self.yolo_seg_model is not None:
            model_type = "segment"
        elif hasattr(self, 'yolo_model') and self.yolo_model is not None:
            model_type = "detect"

        # 既存のアノテーションがある場合は確認（モデルタイプに応じて）
        if model_type == "segment":
            existing_count = len(self.segmentation_annotations) if hasattr(self, 'segmentation_annotations') else 0
            if existing_count > 0:
                msg = f"既存のセグメンテーションアノテーションがあります:\n"
                msg += f"・{existing_count}個の画像\n"
                msg += "\nどのように処理しますか？"
        else:
            existing_count = len(self.bbox_annotations) if hasattr(self, 'bbox_annotations') else 0
            if existing_count > 0:
                msg = f"既存のバウンディングボックスアノテーションがあります:\n"
                msg += f"・{existing_count}個の画像\n"
                msg += "\nどのように処理しますか？"

        if 'msg' in locals():

            msgBox = QMessageBox()
            msgBox.setWindowTitle("既存アノテーションの処理")
            msgBox.setText(msg)
            msgBox.addButton("上書き", QMessageBox.AcceptRole)
            msgBox.addButton("追加", QMessageBox.AcceptRole)
            msgBox.addButton("キャンセル", QMessageBox.RejectRole)

            result = msgBox.exec_()

            if result == 2:  # キャンセル
                return
            elif result == 0:  # 上書き
                # モデルタイプに応じてクリア
                if model_type == "segment":
                    if hasattr(self, 'segmentation_annotations'):
                        self.segmentation_annotations.clear()
                else:
                    if hasattr(self, 'bbox_annotations'):
                        self.bbox_annotations.clear()

        # 範囲指定ダイアログを表示
        range_dialog = QDialog(self)
        range_dialog.setWindowTitle("YOLOオートアノテーション範囲指定")
        range_dialog.setMinimumWidth(400)

        layout = QVBoxLayout(range_dialog)

        # 説明ラベル
        info_label = QLabel(f"YOLO{model_type}のオートアノテーションを実行する範囲を指定してください。")
        layout.addWidget(info_label)

        # 範囲選択
        range_group = QGroupBox("範囲")
        range_layout = QVBoxLayout(range_group)

        # ラジオボタン
        all_radio = QRadioButton("すべての画像")
        all_radio.setChecked(True)
        range_radio = QRadioButton("インデックス範囲を指定")

        range_layout.addWidget(all_radio)
        range_layout.addWidget(range_radio)

        # インデックス範囲入力
        index_layout = QHBoxLayout()
        index_layout.addWidget(QLabel("開始:"))
        start_spin = QSpinBox()
        start_spin.setRange(0, len(self.images) - 1)
        start_spin.setValue(0)
        start_spin.setEnabled(False)
        index_layout.addWidget(start_spin)

        # 現在位置ボタン（開始）
        start_current_button = QPushButton("現在位置")
        start_current_button.setEnabled(False)
        start_current_button.setToolTip("現在表示中の画像インデックスを設定")
        start_current_button.clicked.connect(lambda: start_spin.setValue(self.current_index))
        index_layout.addWidget(start_current_button)

        index_layout.addWidget(QLabel("終了:"))
        end_spin = QSpinBox()
        end_spin.setRange(0, len(self.images) - 1)
        end_spin.setValue(len(self.images) - 1)
        end_spin.setEnabled(False)
        index_layout.addWidget(end_spin)

        # 現在位置ボタン（終了）
        end_current_button = QPushButton("現在位置")
        end_current_button.setEnabled(False)
        end_current_button.setToolTip("現在表示中の画像インデックスを設定")
        end_current_button.clicked.connect(lambda: end_spin.setValue(self.current_index))
        index_layout.addWidget(end_current_button)

        range_layout.addLayout(index_layout)
        layout.addWidget(range_group)

        # ラジオボタンの状態変化でスピンボックスとボタンを有効/無効化
        def on_range_radio_toggled(checked):
            start_spin.setEnabled(checked)
            end_spin.setEnabled(checked)
            start_current_button.setEnabled(checked)
            end_current_button.setEnabled(checked)

        range_radio.toggled.connect(on_range_radio_toggled)

        # OK/キャンセルボタン
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(range_dialog.accept)
        button_box.rejected.connect(range_dialog.reject)
        layout.addWidget(button_box)

        # ダイアログ実行
        if range_dialog.exec_() != QDialog.Accepted:
            return

        # 範囲に基づいて処理対象画像を取得
        if all_radio.isChecked():
            # すべての画像
            target_images = self.images[:]
        else:
            # 指定範囲の画像
            start_idx = start_spin.value()
            end_idx = end_spin.value()
            if start_idx > end_idx:
                QMessageBox.warning(self, "警告", "開始インデックスは終了インデックス以下である必要があります。")
                return

            target_images = self.images[start_idx:end_idx + 1]

        if not target_images:
            QMessageBox.information(self, "情報", "処理対象の画像がありません。")
            return
        
        # 信頼度の設定を取得
        conf_threshold = 0.25
        if hasattr(self, 'yolo_conf_spinbox'):
            conf_threshold = self.yolo_conf_spinbox.value()
        
        # 処理するクラスの設定を取得
        target_classes = []
        if hasattr(self, 'classes_input') and self.classes_input.text():
            target_classes = [cls.strip() for cls in self.classes_input.text().split(',') if cls.strip()]
        
        # スキップ枚数の設定ダイアログ
        dialog = QDialog(self)
        dialog.setWindowTitle("オートアノテーション設定")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)

        # 説明ラベル
        info_label = QLabel("オートアノテーションの実行方法を選択してください")
        layout.addWidget(info_label)
        
        # ラジオボタングループ
        radio_group = QButtonGroup(dialog)
        
        # 全画像オプション
        all_radio = QRadioButton("すべての画像を処理")
        all_radio.setChecked(True)
        radio_group.addButton(all_radio, 0)
        layout.addWidget(all_radio)
        
        # スキップオプション
        skip_radio = QRadioButton("指定枚数ごとに処理")
        radio_group.addButton(skip_radio, 1)
        layout.addWidget(skip_radio)
        
        # スキップ枚数入力
        skip_layout = QHBoxLayout()
        skip_layout.addSpacing(20)
        skip_label = QLabel("スキップ枚数:")
        skip_layout.addWidget(skip_label)
        
        skip_spinbox = QSpinBox()
        skip_spinbox.setMinimum(1)
        skip_spinbox.setMaximum(100)
        skip_spinbox.setValue(5)
        skip_spinbox.setEnabled(False)
        skip_layout.addWidget(skip_spinbox)
        
        skip_layout.addWidget(QLabel("枚ごと"))
        skip_layout.addStretch()
        layout.addLayout(skip_layout)
        
        # スキップラジオボタンが選択されたときにスピンボックスを有効化
        skip_radio.toggled.connect(skip_spinbox.setEnabled)

        # 処理枚数の見積もり表示
        estimate_label = QLabel()
        def update_estimate():
            if all_radio.isChecked():
                count = len(target_images)
                estimate_label.setText(f"処理予定: {count}枚")
            else:
                skip = skip_spinbox.value()
                count = (len(target_images) + skip - 1) // skip
                estimate_label.setText(f"処理予定: 約{count}枚（{skip}枚ごと）")

        all_radio.toggled.connect(update_estimate)
        skip_spinbox.valueChanged.connect(update_estimate)
        update_estimate()
        layout.addWidget(estimate_label)
        
        # ボタン
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        # 設定を取得
        process_all = all_radio.isChecked()
        skip_count = skip_spinbox.value() if not process_all else 1

        # 処理する画像リストを作成（範囲指定されたtarget_imagesに対してスキップ処理を適用）
        if process_all:
            images_to_process = target_images
        else:
            images_to_process = target_images[::skip_count]
        
        # 進捗ダイアログを表示
        progress = QProgressDialog(
            f"YOLO オートアノテーション準備中... ({len(images_to_process)}枚の画像)",
            "キャンセル", 0, 100, self
        )
        progress.setWindowTitle("YOLO オートアノテーション実行中")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        try:
            # モデル読み込み
            progress.setLabelText("YOLOモデルを読み込み中...")
            progress.setValue(10)
            QApplication.processEvents()
            
            # 現在読み込まれているモデルを使用するか、デフォルトモデルを使用
            model = None
            model_type = "detect"  # デフォルトは物体検知
            
            # 既に読み込まれているモデルがあるか確認
            if hasattr(self, 'yolo_model') and self.yolo_model is not None:
                # 物体検知モデルが読み込まれている
                model = self.yolo_model
                model_type = "detect"
                progress.setLabelText("物体検知モデルを使用します...")
            elif hasattr(self, 'yolo_seg_model') and self.yolo_seg_model is not None:
                # セグメンテーションモデルが読み込まれている
                model = self.yolo_seg_model
                model_type = "segment"
                progress.setLabelText("セグメンテーションモデルを使用します...")
            else:
                # モデルが読み込まれていない場合はデフォルトモデルを使用
                model_path = None
                if os.path.exists("yolo11n.pt"):
                    model_path = "yolo11n.pt"
                    model_type = "detect"
                elif os.path.exists("yolov8n-seg.pt"):
                    model_path = "yolov8n-seg.pt"
                    model_type = "segment"
                
                from utils.yolo_utils import get_yolo_model
                model = get_yolo_model(model_path)
            
            if model is None:
                QMessageBox.warning(self, "警告", "YOLOモデルが読み込まれていません。")
                progress.close()
                return
            
            # 進捗コールバック関数
            def progress_callback(current, total, message):
                if progress.wasCanceled():
                    return False
                
                progress_value = 10 + int((current / total) * 85)
                progress.setValue(progress_value)
                progress.setLabelText(message)
                QApplication.processEvents()
                return True
            
            # バッチ処理実行
            if model_type == "segment":
                progress.setLabelText("セグメンテーションを実行中...")
            else:
                progress.setLabelText("物体検知を実行中...")
            progress.setValue(15)
            QApplication.processEvents()
            
            from utils.yolo_utils import batch_detect_objects_and_segments
            results = batch_detect_objects_and_segments(
                images_to_process, model, conf_threshold, progress_callback
            )
            
            if progress.wasCanceled():
                return
            
            # 結果を既存のアノテーションデータに統合
            progress.setLabelText("アノテーションデータを統合中...")
            progress.setValue(95)
            QApplication.processEvents()
            
            # 手動アノテーションと同じ辞書を使用
            if not hasattr(self, 'bbox_annotations'):
                self.bbox_annotations = {}
            if not hasattr(self, 'segmentation_annotations'):
                self.segmentation_annotations = {}
            
            detection_count = 0
            segmentation_count = 0
            
            # 画像パスからインデックスへのマッピングを作成
            img_path_to_index = {path: idx for idx, path in enumerate(self.images)}
            
            for img_path, result in results.items():
                # 画像インデックスを取得
                if img_path not in img_path_to_index:
                    continue
                img_index = img_path_to_index[img_path]
                
                # バウンディングボックス処理（物体検知モデルの場合のみ）
                if model_type == "detect" and result['detections']:
                    if target_classes:
                        # 指定されたクラスのみフィルタリング
                        filtered_detections = [
                            det for det in result['detections'] 
                            if det['class'] in target_classes
                        ]
                    else:
                        filtered_detections = result['detections']
                    
                    if filtered_detections:
                        # 既存のアノテーションを取得（重複チェック用）
                        existing_bboxes = self.bbox_annotations.get(img_index, [])

                        # 手動アノテーションと同じ形式に変換（既に正規化座標）
                        bbox_annotations = []
                        skipped_bbox_count = 0
                        for det in filtered_detections:
                            # bboxは既に正規化座標（0-1）で受け取り、型の一貫性を確保
                            x1 = float(det['bbox'][0])
                            y1 = float(det['bbox'][1])
                            x2 = float(det['bbox'][2])
                            y2 = float(det['bbox'][3])

                            # 範囲チェック（0-1の間に収まることを確認）し、明示的にfloat型で保存
                            x1 = float(max(0.0, min(1.0, x1)))
                            y1 = float(max(0.0, min(1.0, y1)))
                            x2 = float(max(0.0, min(1.0, x2)))
                            y2 = float(max(0.0, min(1.0, y2)))

                            new_bbox = {
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'class': det['class'],
                                'confidence': float(det.get('confidence', 1.0))
                            }

                            # 既存のアノテーションとの重複チェック
                            if existing_bboxes:
                                is_overlap = self.check_bbox_overlap(
                                    new_bbox, existing_bboxes, iou_threshold=0.5
                                )
                                if is_overlap:
                                    # 重複している場合はスキップ
                                    skipped_bbox_count += 1
                                    continue

                            # 重複していない場合のみ追加
                            bbox_annotations.append(new_bbox)

                        # アノテーションを統合
                        if bbox_annotations:
                            if img_index in self.bbox_annotations:
                                # 既存のアノテーションに追加
                                self.bbox_annotations[img_index].extend(bbox_annotations)
                            else:
                                self.bbox_annotations[img_index] = bbox_annotations

                            detection_count += len(bbox_annotations)

                        # スキップした数をログ出力
                        if skipped_bbox_count > 0:
                            print(f"画像 {img_index}: {skipped_bbox_count}個の重複バウンディングボックスをスキップしました")
                
                # セグメンテーション処理（セグメンテーションモデルの場合のみ）
                if model_type == "segment" and result['segments']:
                    if target_classes:
                        # 指定されたクラスのみフィルタリング
                        filtered_segments = [
                            seg for seg in result['segments'] 
                            if seg['class'] in target_classes
                        ]
                    else:
                        filtered_segments = result['segments']
                    
                    if filtered_segments:
                        # 手動アノテーションと同じ形式に変換
                        seg_annotations = []
                        # 現在の画像サイズを取得
                        current_img_path = self.images[img_index]
                        try:
                            from PIL import Image
                            with Image.open(current_img_path) as img:
                                img_width, img_height = img.size
                        except Exception as e:
                            print(f"画像サイズ取得エラー {current_img_path}: {e}")
                            continue
                        
                        # 既存のアノテーションを取得（重複チェック用）
                        existing_segs = self.segmentation_annotations.get(img_index, [])

                        skipped_count = 0
                        for seg in filtered_segments:
                            # 正規化座標をピクセル座標に変換（手動セグメンテーションと同じ形式）
                            pixel_points = []
                            for point in seg['points']:
                                # 正規化座標からピクセル座標へ変換
                                norm_x = float(max(0.0, min(1.0, float(point[0]))))
                                norm_y = float(max(0.0, min(1.0, float(point[1]))))

                                pixel_x = int(norm_x * img_width)
                                pixel_y = int(norm_y * img_height)

                                # ピクセル座標を画像境界内に制限
                                pixel_x = max(0, min(img_width - 1, pixel_x))
                                pixel_y = max(0, min(img_height - 1, pixel_y))

                                pixel_points.append((pixel_x, pixel_y))

                            new_seg = {
                                'class': seg['class'],
                                'points': pixel_points,  # ピクセル座標で保存
                                'confidence': float(seg.get('confidence', 1.0))
                            }

                            # 既存のアノテーションとの重複チェック
                            if existing_segs:
                                is_overlap = self.check_segmentation_overlap(
                                    new_seg, existing_segs, img_width, img_height, iou_threshold=0.5
                                )
                                if is_overlap:
                                    # 重複している場合はスキップ
                                    skipped_count += 1
                                    continue

                            # 重複していない場合のみ追加
                            seg_annotations.append(new_seg)

                        # アノテーションを統合
                        if seg_annotations:
                            if img_index in self.segmentation_annotations:
                                # 既存のアノテーションに追加
                                self.segmentation_annotations[img_index].extend(seg_annotations)
                            else:
                                self.segmentation_annotations[img_index] = seg_annotations

                            segmentation_count += len(seg_annotations)

                        # スキップした数をログ出力
                        if skipped_count > 0:
                            print(f"画像 {img_index}: {skipped_count}個の重複セグメンテーションをスキップしました")
            
            # UI更新
            progress.setLabelText("表示を更新中...")
            progress.setValue(98)
            QApplication.processEvents()
            
            # 現在の画像表示を更新
            self.display_current_image()
            
            # ギャラリー更新
            self.update_gallery()
            
            # 統計情報更新
            self.update_ui()
            
            progress.setValue(100)
            progress.close()
            
            # 完了メッセージ
            if model_type == "segment":
                message = f"セグメンテーション オートアノテーションが完了しました。\n"
                message += f"処理画像数: {len(images_to_process)}枚"
                if not process_all:
                    message += f"（{skip_count}枚ごと）"
                message += f"\n検出されたセグメンテーション: {segmentation_count}個"
            else:
                message = f"物体検知 オートアノテーションが完了しました。\n"
                message += f"処理画像数: {len(images_to_process)}枚"
                if not process_all:
                    message += f"（{skip_count}枚ごと）"
                message += f"\n検出されたバウンディングボックス: {detection_count}個"
            
            if target_classes:
                message += f"\n対象クラス: {', '.join(target_classes)}"
            
            QMessageBox.information(self, "完了", message)
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "エラー", 
                f"YOLO オートアノテーション中にエラーが発生しました: {str(e)}"
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
            original_pixmap = QPixmap.fromImage(pil_to_qimage(original_img))
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
                sample_pixmap = QPixmap.fromImage(pil_to_qimage(img))
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

    def update_detection_info_panel(self):
        """物体検知推論結果の情報パネルを更新する"""
        if not self.images:
            return
            
        current_index = self.current_index
        
        if hasattr(self, 'detection_inference_checkbox') and self.detection_inference_checkbox.isChecked():
            if current_index in self.detection_inference_results:
                # クラスごとのカウント辞書を作成
                class_counts = {}
                inference_bboxes = self.detection_inference_results[current_index]
                
                for bbox in inference_bboxes:
                    class_name = bbox.get('class', 'unknown')
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # 情報テキストを構築
                inference_text = "<b>物体検知推論結果:</b><br>"
                inference_text += "検出オブジェクト:<br>"
                
                for class_name, count in class_counts.items():
                    # クラスに応じた色を設定
                    class_colors = DETECTION_INFERENCE_TEXT_COLORS
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
                self.detection_inference_info_label.setText(" ")  # スペースで高さを維持
        
        return False


    ### 位置モデル関連
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

    def add_location_model_section(self):
        """位置推論モデルのセクションを追加する"""
        # 位置モデルマネージャーの初期化
        self.location_model_manager = LocationModelManager(APP_DIR_PATH, MODELS_DIR_NAME)

        # YOLOモデル領域の上に位置推論モデルセクションを配置するため、
        # オリジナルのレイアウトを取得
        left_layout = self.get_left_layout()
        if left_layout is None:
            print("警告: left_layoutが見つかりません")
            return
        
        # 位置推論モデルを現在のレイアウトの末尾に追加する
        # （呼び出し順序を調整済みなので、物体検知コンテナより前に配置される）
        insert_index = left_layout.count()
        
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
        
        # 位置モデル学習ボタン
        train_location_button = QPushButton("モデル学習・保存")
        train_location_button.clicked.connect(self.train_and_save_location_model)
        apply_style(train_location_button, 'training')
        location_model_buttons_layout.addWidget(train_location_button)

        # モデル読み込みボタン
        self.location_load_button = QPushButton("モデル読込")
        self.location_load_button.setToolTip("modelsフォルダのモデルを読込む")
        self.location_load_button.clicked.connect(self.load_location_model)
        apply_style(self.location_load_button, 'model')
        location_model_buttons_layout.addWidget(self.location_load_button)
        
        location_model_layout.addLayout(location_model_buttons_layout)
                
        # 位置推論表示チェックボックス
        location_inference_layout = QHBoxLayout()
        self.location_inference_checkbox = QCheckBox("位置推論結果表示")
        self.location_inference_checkbox.setChecked(False)
        self.location_inference_checkbox.setEnabled(False)  # 初期状態は無効
        self.location_inference_checkbox.setToolTip("位置モデルが読み込まれていません")
        self.location_inference_checkbox.stateChanged.connect(self.toggle_location_inference_display)
        location_inference_layout.addWidget(self.location_inference_checkbox)
        location_model_layout.addLayout(location_inference_layout)
        
        # 位置モデルコンテナを追加（物体検知コンテナより前に配置される）
        left_layout.addWidget(self.location_model_container)
        
        # 推論結果格納用の辞書を初期化
        self.location_inference_results = {}
        
        # モデルリストの取得は新しいマネージャーを使用
        self.refresh_location_model_list()

        # 推論結果格納用の辞書を初期化
        self.location_inference_results = {}

    def add_waypoint_model_section(self):
        """ウェイポイントモデルのセクションを追加する"""
        # ウェイポイントモデルマネージャーの初期化
        self.waypoint_model_manager = LocationModelManager(APP_DIR_PATH, MODELS_DIR_NAME)

        # left_layoutを取得
        left_layout = self.get_left_layout()
        if left_layout is None:
            print("警告: left_layoutが見つかりません")
            return

        # ウェイポイントモデルコンテナを作成
        self.waypoint_model_container = QWidget()
        waypoint_model_layout = QVBoxLayout(self.waypoint_model_container)

        # ヘッダータイトル
        waypoint_model_label = QLabel("ウェイポイント推論モデル:")
        waypoint_model_label.setStyleSheet("font-weight: bold")
        waypoint_model_layout.addWidget(waypoint_model_label)

        # モデル選択
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("モデルタイプ:"))
        self.waypoint_model_combo = QComboBox()
        self.waypoint_model_combo.addItems(["donkey_waypoint", "resnet18_waypoint"])
        self.waypoint_model_combo.currentIndexChanged.connect(self.on_waypoint_model_type_changed)
        model_type_layout.addWidget(self.waypoint_model_combo)
        waypoint_model_layout.addLayout(model_type_layout)

        # 事前学習済みモデル選択
        self.waypoint_saved_model_combo = QComboBox()
        self.waypoint_saved_model_combo.setMinimumWidth(180)
        self.waypoint_saved_model_combo.setStyleSheet("combobox-popup: 0;")
        waypoint_model_layout.addWidget(self.waypoint_saved_model_combo)

        # モデル操作ボタン
        waypoint_model_buttons_layout = QHBoxLayout()

        # ウェイポイントモデル学習ボタン
        train_waypoint_button = QPushButton("モデル学習・保存")
        train_waypoint_button.clicked.connect(self.train_and_save_waypoint_model)
        apply_style(train_waypoint_button, 'training')
        waypoint_model_buttons_layout.addWidget(train_waypoint_button)

        # モデル読み込みボタン
        self.waypoint_load_button = QPushButton("モデル読込")
        self.waypoint_load_button.setToolTip("modelsフォルダのモデルを読込む")
        self.waypoint_load_button.clicked.connect(self.load_waypoint_model)
        apply_style(self.waypoint_load_button, 'model')
        waypoint_model_buttons_layout.addWidget(self.waypoint_load_button)

        waypoint_model_layout.addLayout(waypoint_model_buttons_layout)

        # ウェイポイント推論表示チェックボックス
        waypoint_inference_layout = QHBoxLayout()
        self.waypoint_inference_checkbox = QCheckBox("ウェイポイント推論結果表示")
        self.waypoint_inference_checkbox.setChecked(False)
        self.waypoint_inference_checkbox.setEnabled(False)  # 初期状態は無効
        self.waypoint_inference_checkbox.setToolTip("ウェイポイントモデルが読み込まれていません")
        self.waypoint_inference_checkbox.stateChanged.connect(self.toggle_waypoint_inference_display)
        waypoint_inference_layout.addWidget(self.waypoint_inference_checkbox)
        waypoint_model_layout.addLayout(waypoint_inference_layout)

        # ウェイポイントモデルコンテナを追加
        left_layout.addWidget(self.waypoint_model_container)

        # 推論結果格納用の辞書を初期化
        self.waypoint_inference_results = {}

        # モデルリストの取得
        self.refresh_waypoint_model_list()

    def refresh_waypoint_model_list(self):
        """保存されているウェイポイントモデルのリストを更新"""
        self.waypoint_saved_model_combo.clear()

        # 更新開始のメッセージを表示
        self.statusBar().showMessage("ウェイポイントモデルリストを更新中...")

        # 現在選択されているモデルタイプを取得
        selected_model_type = self.waypoint_model_combo.currentText()

        # モデルマネージャーからモデルリストを取得
        model_files = self.waypoint_model_manager.get_model_list(model_type=selected_model_type)

        if not model_files:
            self.waypoint_saved_model_combo.addItem(f"{selected_model_type}のウェイポイントモデルが見つかりません")
            self.statusBar().showMessage(f"{selected_model_type}のウェイポイントモデルが見つかりません。モデルを学習してください", 3000)
            return

        # モデルファイルをコンボボックスに追加
        for model_file in model_files:
            display_name = os.path.basename(model_file).replace('.pth', '')
            self.waypoint_saved_model_combo.addItem(display_name, model_file)

        self.statusBar().showMessage(f"{len(model_files)}個のウェイポイントモデルが見つかりました", 2000)

    def on_waypoint_model_type_changed(self, index):
        """ウェイポイントモデルタイプが変更された時の処理"""
        # モデルリストを更新
        self.refresh_waypoint_model_list()

        # 現在のウェイポイントモデルをクリア
        if hasattr(self, 'waypoint_model'):
            del self.waypoint_model
            self.waypoint_model = None

        # ウェイポイント推論を無効化
        if hasattr(self, 'waypoint_inference_checkbox'):
            self.waypoint_inference_checkbox.setChecked(False)
            self.waypoint_inference_checkbox.setEnabled(False)
            self.waypoint_inference_checkbox.setToolTip("ウェイポイントモデルが読み込まれていません")

    def load_waypoint_model(self):
        """ウェイポイントモデルを読み込む"""
        current_data = self.waypoint_saved_model_combo.currentData()
        if not current_data:
            QMessageBox.warning(self, "警告", "読み込むウェイポイントモデルが選択されていません。")
            return

        model_path = current_data
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "エラー", f"ウェイポイントモデルファイルが見つかりません: {model_path}")
            return

        try:
            # モデルタイプを取得
            model_type = self.waypoint_model_combo.currentText()

            print(f"\n{'='*60}")
            print(f"[Waypointモデル読み込み] 開始")
            print(f"{'='*60}")
            print(f"モデルタイプ: {model_type}")
            print(f"モデルパス: {model_path}")

            # 既存のモデルがある場合はクリア
            if hasattr(self, 'waypoint_model') and self.waypoint_model is not None:
                print(f"既存のモデルをクリアします")
                del self.waypoint_model
                self.waypoint_model = None

            # 既存の推論結果をクリア
            old_results_count = len(self.waypoint_inference_results)
            if old_results_count > 0:
                print(f"既存の推論結果 {old_results_count}件をクリアします")
                self.waypoint_inference_results.clear()

            # ウェイポイントモデルをロード
            from model_catalog import get_model

            # チェックポイントをロード
            print(f"チェックポイントを読み込み中...")
            checkpoint = torch.load(model_path, map_location='cpu')
            num_waypoints = checkpoint.get('num_waypoints', 4)
            print(f"ウェイポイント数: {num_waypoints}")

            # モデルを初期化
            print(f"モデルを初期化中...")
            self.waypoint_model = get_model(model_type, pretrained=False, input_size=(224, 224))
            self.waypoint_model.num_waypoints = num_waypoints
            self.waypoint_model.load_state_dict(checkpoint['model_state_dict'])
            self.waypoint_model.eval()
            print(f"モデルの評価モードに設定完了")

            # 推論チェックボックスを有効化して自動でONにする
            self.waypoint_inference_checkbox.setEnabled(True)
            self.waypoint_inference_checkbox.setChecked(True)
            self.waypoint_inference_checkbox.setToolTip(f"ウェイポイントモデル ({model_type}, {num_waypoints}ポイント) が読み込まれています")

            print(f"推論チェックボックスを有効化しました")

            # 現在の画像で推論を実行（チェックボックスがONの場合）
            if self.waypoint_inference_checkbox.isChecked() and self.images and self.current_index is not None:
                print(f"現在の画像(index={self.current_index})で推論を実行します")
                self.update_waypoint_inference_display()

            # 画面を再描画
            if hasattr(self, 'main_image_view'):
                self.main_image_view.update()
                print(f"画面を再描画しました")

            print(f"{'='*60}")
            print(f"[Waypointモデル読み込み] 成功")
            print(f"{'='*60}\n")

            QMessageBox.information(self, "成功", f"ウェイポイントモデルを読み込みました\nモデル: {os.path.basename(model_path)}\nウェイポイント数: {num_waypoints}")

        except Exception as e:
            print(f"{'='*60}")
            print(f"[Waypointモデル読み込み] エラー")
            print(f"{'='*60}")
            print(f"エラー内容: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            QMessageBox.critical(self, "エラー", f"ウェイポイントモデルの読み込みに失敗しました:\n{str(e)}")

    def toggle_waypoint_inference_display(self):
        """ウェイポイント推論表示のON/OFFを切り替える"""
        if not hasattr(self, 'waypoint_model') or self.waypoint_model is None:
            self.waypoint_inference_checkbox.setChecked(False)
            QMessageBox.warning(self, "警告", "ウェイポイントモデルが読み込まれていません。")
            return

        if self.waypoint_inference_checkbox.isChecked():
            # ウェイポイント推論を有効化
            self.statusBar().showMessage("ウェイポイント推論表示を有効化しました", 2000)
        else:
            # ウェイポイント推論を無効化
            self.statusBar().showMessage("ウェイポイント推論表示を無効化しました", 2000)

        # 現在の画像の推論表示を更新
        self.update_waypoint_inference_display()

        # 画面更新
        if hasattr(self, 'main_image_view'):
            self.main_image_view.update()

    def update_waypoint_inference_display(self):
        """ウェイポイント推論表示を更新する"""
        if not self.images:
            return

        current_index = self.current_index

        if hasattr(self, 'waypoint_inference_checkbox') and self.waypoint_inference_checkbox.isChecked():
            if current_index in self.waypoint_inference_results:
                # 既に推論済みの結果がある場合は画面更新のみ
                if hasattr(self, 'main_image_view'):
                    self.main_image_view.update()
                return

            # ウェイポイント推論を実行
            try:
                if hasattr(self, 'waypoint_model') and self.waypoint_model is not None:
                    # 現在の画像を取得
                    img_path = self.images[current_index]
                    img = Image.open(img_path).convert('RGB')

                    # 推論実行（結果は正規化座標で返される）
                    waypoint_coords_raw = self.waypoint_model.run(img)

                    # 座標を0-1の範囲にクリップ
                    waypoint_coords = []
                    clipped_count = 0
                    for wx, wy in waypoint_coords_raw:
                        wx_clipped = max(0.0, min(1.0, wx))
                        wy_clipped = max(0.0, min(1.0, wy))
                        waypoint_coords.append([wx_clipped, wy_clipped])
                        if wx != wx_clipped or wy != wy_clipped:
                            clipped_count += 1

                    # 結果を保存
                    self.waypoint_inference_results[current_index] = waypoint_coords

                    # 元の画像サイズを取得
                    original_img = Image.open(img_path)
                    img_width, img_height = original_img.size

                    # 推論結果をターミナルに表示（簡潔版）
                    pixel_coords = [(int(wx * img_width), int(wy * img_height)) for wx, wy in waypoint_coords]
                    clip_info = f" ({clipped_count}個クリップ)" if clipped_count > 0 else ""
                    print(f"[Waypoint推論] index={current_index}, waypoints={len(waypoint_coords)}個{clip_info}, pixel={pixel_coords}")

            except Exception as e:
                print(f"ウェイポイント推論エラー: {e}")
                import traceback
                traceback.print_exc()
                self.waypoint_inference_results[current_index] = []

        # 画面を更新（推論実行の有無に関わらず）
        if hasattr(self, 'main_image_view'):
            self.main_image_view.update()

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

        # コンボボックスに追加（モデル名のみを表示、フルパスはユーザーデータとして保持）
        for model_file in model_files:
            display_name = os.path.basename(model_file).replace('.pth', '')
            self.location_saved_model_combo.addItem(display_name, model_file)

        # 更新完了メッセージ
        self.statusBar().showMessage(f"{len(model_files)}個の{selected_model_type}位置モデルを読み込みました", 3000)

    def load_location_model(self):
        """選択された位置モデルを読み込む"""
        if not self.images:
            QMessageBox.warning(self, "警告", "画像が読み込まれていません。")
            return

        # モデル情報を取得
        model_type = self.location_model_combo.currentText()
        selected_model_path = self.location_saved_model_combo.currentData()

        if not selected_model_path:
            QMessageBox.warning(self, "警告", "有効な位置モデルが選択されていません。")
            return

        # モデルのパスを取得（フルパスがユーザーデータに保存されている）
        model_path = selected_model_path
        
        # モデルが存在するか確認
        if not os.path.exists(model_path):
            model_name = os.path.basename(model_path)
            QMessageBox.warning(self, "警告", f"選択されたモデルが見つかりません: {model_name}")
            return

        # 進捗ダイアログを表示
        model_name = os.path.basename(model_path).replace('.pth', '')
        progress = QProgressDialog(
            f"位置モデル '{model_type} ({model_name})' を読み込み中...", 
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
            
            # 推論表示チェックボックスを有効にして自動的にオンにする
            self.location_inference_checkbox.setEnabled(True)
            self.location_inference_checkbox.setToolTip("位置モデルが読み込まれています")
            self.location_inference_checkbox.setChecked(True)
            
            # show_location_inferenceフラグを設定
            self.show_location_inference = True
            
            # 情報パネルを先に更新（基本的な情報表示）
            self.update_location_info_panel()
            
            # 位置推論表示を更新（推論結果の詳細表示）- これを最後にして上書きを防ぐ
            self.update_location_inference_display()
            
            # 各モデルの状態を更新
            self.update_inference_checkboxes_status()
            
            # 画面描画を更新
            if hasattr(self, 'main_image_view'):
                self.main_image_view.update()
            
            update_progress(100)
            progress.close()

            # 成功メッセージ
            self.statusBar().showMessage(f"位置モデル '{model_type} ({model_name})' を読み込みました (クラス数: {num_classes})", 5000)
            
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
        current_index = self.current_index
        
        # マネージャーを使用して推論を実行
        result = self.location_model_manager.run_inference(current_img_path)
        
        if result:
            # 推論結果を保存（インデックスベース）
            self.location_inference_results[current_index] = result
            
            # 表示更新は呼び出し元で行うため、ここでは呼ばない
            
            return True
        
        return False

    def train_and_save_location_model(self):
        """位置推論モデルの学習"""
        
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
        num_classes = LOCATION_DEFAULT_NUM_CLASSES
        
        # 選択されたモデル
        model_type = self.location_model_combo.currentText()
        
        # アノテーション統計情報を収集（削除済みマークを考慮）
        total_images = len(self.images)
        
        # 削除済みでない位置アノテーションをカウント
        valid_location_annotations = 0
        deleted_count = 0
        for idx in self.location_annotations.keys():
            # 削除マークされていないかチェック（actual_indexで判定）
            if hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes:
                deleted_count += 1
                continue
            valid_location_annotations += 1

        annotated_images = len(self.location_annotations)  # 全体数
        
        # 学習設定ダイアログを表示
        training_settings = self._create_location_training_dialog(model_type, actual_classes, unique_locations, num_classes, total_images, annotated_images, valid_location_annotations, deleted_count)
        
        if not training_settings.exec_():
            return
        
        # 設定値の取得
        training_config = self._get_location_training_config(training_settings)
        
        try:
            # データの準備（インデックスベース修正版）
            image_data = self._prepare_location_training_data(unique_locations)
            
            if not image_data['image_paths']:
                QMessageBox.warning(self, "警告", "有効な位置アノテーションがありません。")
                return
            
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
                image_paths=image_data['image_paths'],
                location_labels=image_data['location_indices'],
                val_split=0.2,
                model_name=model_type,
                batch_size=training_config['batch_size'],
                use_augmentation=training_config['use_augmentation']
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
            
            # モデル学習（統合版）
            training_results = self._train_location_model_internal(
                model_type=model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                training_config=training_config,
                progress_callback=update_progress
            )
            
            # モデルのメタデータ保存
            self._save_location_model_metadata(
                training_results['best_model_path'],
                num_classes,
                actual_classes,
                image_data['location_to_index']
            )
            
            progress.setValue(95)
            progress.setLabelText("MLflowに学習結果を記録中...")
            QApplication.processEvents()
            
            # MLflowに結果を記録
            mlflow_info = self._log_location_training(
                model_type=model_type,
                training_results=training_results,
                training_config=training_config,
                dataset_info={
                    "total_annotations": len(self.location_annotations),
                    "used_samples": len(image_data['image_paths']),
                    "train_samples": dataset_info['train_samples'],
                    "val_samples": dataset_info['val_samples'],
                    "input_shape": dataset_info['actual_image_size'],
                    "num_classes": num_classes,
                    "actual_classes": actual_classes,
                    "unique_locations": unique_locations,
                    "location_mapping": image_data['location_to_index']
                }
            )
            
            # モデルリストを更新
            self.refresh_location_model_list()
            
            progress.setValue(100)
            progress.close()
            
            # 学習完了メッセージ
            self._show_location_training_success(
                model_type=model_type,
                training_results=training_results,
                training_config=training_config,
                dataset_info={
                    "image_paths_count": len(image_data['image_paths']),
                    "num_classes": num_classes,
                    "actual_classes": actual_classes
                },
                mlflow_info=mlflow_info
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

    def _create_location_training_dialog(self, model_type, actual_classes, unique_locations, num_classes, total_images, annotated_images, valid_location_annotations, deleted_count):
        """位置モデル学習設定ダイアログを作成"""
        
        training_settings = QDialog(self)
        training_settings.setWindowTitle("位置モデル学習設定")
        training_settings.setMinimumWidth(500)
        
        settings_layout = QVBoxLayout(training_settings)
        
        # アノテーション統計情報を表示（削除済みマークを考慮）
        stats_label = QLabel(f"<b>学習データ統計:</b><br>"
                           f"総読み込み画像数: {total_images}枚<br>"
                           f"位置アノテーション済み画像数: {annotated_images}枚<br>"
                           f"<b style='color: #2E7D32; font-size: 14px;'>実際の学習使用枚数: {valid_location_annotations}枚</b><br>"
                           f"({annotated_images}枚 - 削除済み{deleted_count}枚)<br>"
                           f"<span style='color: #FF6600;'>※ 削除マークされた画像は学習対象から除外されます</span>")
        stats_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px;")
        settings_layout.addWidget(stats_label)
        
        settings_layout.addWidget(QLabel(""))  # スペース追加
        
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
        training_settings.epoch_spin = QSpinBox()
        training_settings.epoch_spin.setRange(1, 1000)
        training_settings.epoch_spin.setValue(30)
        epoch_layout.addWidget(training_settings.epoch_spin)
        settings_layout.addLayout(epoch_layout)
        
        # バッチサイズ設定
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("バッチサイズ:"))
        training_settings.batch_spin = QSpinBox()
        training_settings.batch_spin.setRange(1, 128)
        training_settings.batch_spin.setValue(16)
        batch_layout.addWidget(training_settings.batch_spin)
        settings_layout.addLayout(batch_layout)
        
        # データオーグメンテーション設定
        training_settings.aug_check = QCheckBox("データオーグメンテーションを有効にする")
        training_settings.aug_check.setChecked(True)
        settings_layout.addWidget(training_settings.aug_check)
        
        # Early Stopping設定
        training_settings.early_stopping_check = QCheckBox("Early Stopping を有効にする")
        training_settings.early_stopping_check.setChecked(True)
        settings_layout.addWidget(training_settings.early_stopping_check)
        
        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("忍耐エポック数:"))
        training_settings.patience_spin = QSpinBox()
        training_settings.patience_spin.setRange(1, 20)
        training_settings.patience_spin.setValue(5)
        patience_layout.addWidget(training_settings.patience_spin)
        settings_layout.addLayout(patience_layout)
        
        # 学習率設定
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学習率:"))
        training_settings.lr_combo = QComboBox()
        learning_rates = ["0.001", "0.0005", "0.0001", "0.00005", "0.00001"]
        training_settings.lr_combo.addItems(learning_rates)
        training_settings.lr_combo.setCurrentIndex(0)
        lr_layout.addWidget(training_settings.lr_combo)
        settings_layout.addLayout(lr_layout)

        # モデル名とコメント欄を追加
        settings_layout.addWidget(QLabel(""))  # スペース追加

        # モデル名編集欄
        model_name_group = QGroupBox("モデル名設定")
        model_name_layout = QVBoxLayout(model_name_group)

        # プレフィックス（固定）とサフィックス（編集可能）を分離
        # 位置モデルの場合はmodel_typeをプレフィックスとする
        location_prefix = f"{model_type}_"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # プレフィックスとサフィックスを横並びで表示
        name_input_layout = QHBoxLayout()
        name_input_layout.addWidget(QLabel("モデル名:"))

        # プレフィックス（固定、編集不可）
        prefix_label = QLabel(location_prefix)
        prefix_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; font-family: monospace;")
        name_input_layout.addWidget(prefix_label)

        # サフィックス（編集可能）
        training_settings.model_name_suffix_input = QLineEdit()
        training_settings.model_name_suffix_input.setText(timestamp)
        training_settings.model_name_suffix_input.setPlaceholderText("カスタム名を入力")
        name_input_layout.addWidget(training_settings.model_name_suffix_input)

        model_name_layout.addLayout(name_input_layout)

        # プレフィックスを保存（後で使用）
        training_settings.model_name_prefix = location_prefix

        model_name_note = QLabel(f"※ モデルタイプ ({model_type}) のプレフィックスは変更できません。.pthは自動的に付与されます")
        model_name_note.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        model_name_layout.addWidget(model_name_note)

        settings_layout.addWidget(model_name_group)

        # コメント欄
        comment_group = QGroupBox("学習コメント (MLflowに記録)")
        comment_layout = QVBoxLayout(comment_group)

        comment_layout.addWidget(QLabel("コメント:"))
        training_settings.comment_input = QPlainTextEdit()
        training_settings.comment_input.setPlaceholderText("この学習についてのメモやコメントを入力してください (任意)")
        training_settings.comment_input.setMaximumHeight(80)
        comment_layout.addWidget(training_settings.comment_input)

        settings_layout.addWidget(comment_group)

        # ボタンの配置
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(training_settings.accept)
        button_box.rejected.connect(training_settings.reject)
        settings_layout.addWidget(button_box)

        return training_settings

    def _create_waypoint_training_dialog(self, model_type, most_common_waypoint_count, total_images, annotated_images, valid_waypoint_count, deleted_count):
        """ウェイポイントモデル学習設定ダイアログを作成"""

        training_settings = QDialog(self)
        training_settings.setWindowTitle("ウェイポイントモデル学習設定")
        training_settings.setMinimumWidth(500)

        settings_layout = QVBoxLayout(training_settings)

        # アノテーション統計情報を表示
        stats_label = QLabel(f"<b>学習データ統計:</b><br>"
                           f"総読み込み画像数: {total_images}枚<br>"
                           f"ウェイポイントアノテーション済み画像数: {annotated_images}枚<br>"
                           f"<b style='color: #2E7D32; font-size: 14px;'>実際の学習使用枚数: {valid_waypoint_count}枚</b><br>"
                           f"({annotated_images}枚 - 削除済み{deleted_count}枚)<br>"
                           f"<span style='color: #FF6600;'>※ 削除マークされた画像は学習対象から除外されます</span>")
        stats_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px;")
        settings_layout.addWidget(stats_label)

        settings_layout.addWidget(QLabel(""))  # スペース追加

        # ウェイポイント数設定
        waypoint_layout = QHBoxLayout()
        waypoint_layout.addWidget(QLabel("ウェイポイント数:"))
        training_settings.waypoint_spin = QSpinBox()
        training_settings.waypoint_spin.setRange(2, 10)
        training_settings.waypoint_spin.setValue(most_common_waypoint_count)
        waypoint_layout.addWidget(training_settings.waypoint_spin)
        settings_layout.addLayout(waypoint_layout)

        # エポック数設定
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("学習エポック数:"))
        training_settings.epoch_spin = QSpinBox()
        training_settings.epoch_spin.setRange(1, 1000)
        training_settings.epoch_spin.setValue(50)
        epoch_layout.addWidget(training_settings.epoch_spin)
        settings_layout.addLayout(epoch_layout)

        # バッチサイズ設定
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("バッチサイズ:"))
        training_settings.batch_spin = QSpinBox()
        training_settings.batch_spin.setRange(1, 128)
        training_settings.batch_spin.setValue(8)
        batch_layout.addWidget(training_settings.batch_spin)
        settings_layout.addLayout(batch_layout)

        # データオーグメンテーション設定
        training_settings.aug_check = QCheckBox("データオーグメンテーションを有効にする")
        training_settings.aug_check.setChecked(True)
        settings_layout.addWidget(training_settings.aug_check)

        # Early Stopping設定
        training_settings.early_stopping_check = QCheckBox("Early Stopping を有効にする")
        training_settings.early_stopping_check.setChecked(True)
        settings_layout.addWidget(training_settings.early_stopping_check)

        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("忍耐エポック数:"))
        training_settings.patience_spin = QSpinBox()
        training_settings.patience_spin.setRange(1, 20)
        training_settings.patience_spin.setValue(10)
        patience_layout.addWidget(training_settings.patience_spin)
        settings_layout.addLayout(patience_layout)

        # 学習率設定
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学習率:"))
        training_settings.lr_combo = QComboBox()
        learning_rates = ["0.001", "0.0005", "0.0001", "0.00005", "0.00001"]
        training_settings.lr_combo.addItems(learning_rates)
        training_settings.lr_combo.setCurrentIndex(1)  # 0.0005をデフォルト
        lr_layout.addWidget(training_settings.lr_combo)
        settings_layout.addLayout(lr_layout)

        # モデル名とコメント欄を追加
        settings_layout.addWidget(QLabel(""))  # スペース追加

        # モデル名編集欄
        model_name_group = QGroupBox("モデル名設定")
        model_name_layout = QVBoxLayout(model_name_group)

        # プレフィックス（固定）とサフィックス（編集可能）を分離
        # ウェイポイントモデルの場合はmodel_typeをプレフィックスとする
        waypoint_prefix = f"{model_type}_"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # プレフィックスとサフィックスを横並びで表示
        name_input_layout = QHBoxLayout()
        name_input_layout.addWidget(QLabel("モデル名:"))

        # プレフィックス（固定、編集不可）
        prefix_label = QLabel(waypoint_prefix)
        prefix_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; font-family: monospace;")
        name_input_layout.addWidget(prefix_label)

        # サフィックス（編集可能）
        training_settings.model_name_suffix_input = QLineEdit()
        training_settings.model_name_suffix_input.setText(timestamp)
        training_settings.model_name_suffix_input.setPlaceholderText("カスタム名を入力")
        name_input_layout.addWidget(training_settings.model_name_suffix_input)

        model_name_layout.addLayout(name_input_layout)

        # プレフィックスを保存（後で使用）
        training_settings.model_name_prefix = waypoint_prefix

        model_name_note = QLabel(f"※ モデルタイプ ({model_type}) のプレフィックスは変更できません。.pthは自動的に付与されます")
        model_name_note.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        model_name_layout.addWidget(model_name_note)

        settings_layout.addWidget(model_name_group)

        # コメント欄
        comment_group = QGroupBox("学習コメント (MLflowに記録)")
        comment_layout = QVBoxLayout(comment_group)

        comment_layout.addWidget(QLabel("コメント:"))
        training_settings.comment_input = QPlainTextEdit()
        training_settings.comment_input.setPlaceholderText("この学習についてのメモやコメントを入力してください (任意)")
        training_settings.comment_input.setMaximumHeight(80)
        comment_layout.addWidget(training_settings.comment_input)

        settings_layout.addWidget(comment_group)

        # ボタンの配置
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(training_settings.accept)
        button_box.rejected.connect(training_settings.reject)
        settings_layout.addWidget(button_box)

        return training_settings

    def _get_waypoint_training_config(self, dialog):
        """ウェイポイント学習設定ダイアログから設定値を取得"""

        return {
            'num_waypoints': dialog.waypoint_spin.value(),
            'num_epochs': dialog.epoch_spin.value(),
            'batch_size': dialog.batch_spin.value(),
            'use_augmentation': dialog.aug_check.isChecked(),
            'use_early_stopping': dialog.early_stopping_check.isChecked(),
            'patience': dialog.patience_spin.value() if dialog.early_stopping_check.isChecked() else 0,
            'learning_rate': float(dialog.lr_combo.currentText()),
            'model_name': dialog.model_name_prefix + dialog.model_name_suffix_input.text().strip(),
            'comment': dialog.comment_input.toPlainText().strip()
        }

    def _prepare_waypoint_training_data(self, num_waypoints):
        """ウェイポイント学習データの準備"""

        # データを準備
        image_paths = []
        waypoint_coordinates = []
        skipped_images = []  # スキップされた画像情報

        for idx, waypoints in self.waypoint_annotations.items():
            # インデックスが有効範囲内かチェック
            if isinstance(idx, int) and 0 <= idx < len(self.images):
                # 削除マークされたアノテーションは学習データから除外（actual_indexで判定）
                is_deleted = False
                if hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes:
                    is_deleted = True

                # 削除マークされていない場合のみ学習データに追加
                if not is_deleted and waypoints:
                    img_path = self.images[idx]

                    if len(waypoints) == num_waypoints:
                        # 正しいwaypoint数の場合、学習データに追加
                        image_paths.append(img_path)

                        # 画像サイズを取得して座標を正規化
                        img = Image.open(img_path)
                        img_width, img_height = img.size

                        # ウェイポイント座標を正規化してフラットなリストに変換 [x1, y1, x2, y2, ...]
                        flattened_coords = []
                        for x, y in waypoints:
                            # ピクセル座標を0-1の正規化座標に変換
                            x_norm = float(x) / img_width
                            y_norm = float(y) / img_height
                            flattened_coords.extend([x_norm, y_norm])
                        waypoint_coordinates.append(flattened_coords)
                    else:
                        # waypoint数が一致しない場合、スキップ情報を記録
                        skipped_images.append({
                            'index': idx,
                            'path': img_path,
                            'waypoint_count': len(waypoints),
                            'reason': f'{len(waypoints)}点（必要: {num_waypoints}点）'
                        })

        return {
            'image_paths': image_paths,
            'waypoint_coordinates': waypoint_coordinates,
            'skipped_images': skipped_images
        }

    def _get_location_training_config(self, dialog):
        """学習設定ダイアログから設定値を取得"""

        return {
            'num_epochs': dialog.epoch_spin.value(),
            'batch_size': dialog.batch_spin.value(),
            'use_augmentation': dialog.aug_check.isChecked(),
            'use_early_stopping': dialog.early_stopping_check.isChecked(),
            'patience': dialog.patience_spin.value() if dialog.early_stopping_check.isChecked() else 0,
            'learning_rate': float(dialog.lr_combo.currentText()),
            'model_name': dialog.model_name_prefix + dialog.model_name_suffix_input.text().strip(),
            'comment': dialog.comment_input.toPlainText().strip()
        }

    def _prepare_location_training_data(self, unique_locations):
        """位置学習データの準備（インデックスベース修正版）"""
        
        # 位置ラベルのマッピングを作成（実際の位置値をインデックスに変換）
        location_to_index = {loc: i for i, loc in enumerate(unique_locations)}
        
        # データを準備（インデックスベースに修正）
        image_paths = []
        location_labels = []
        
        for idx, location in self.location_annotations.items():
            # インデックスが有効範囲内かチェック
            if isinstance(idx, int) and 0 <= idx < len(self.images):
                # 削除マークされたアノテーションは学習データから除外（actual_indexで判定）
                is_deleted = False
                if hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes:
                    is_deleted = True
                
                # 削除マークされていない場合のみ学習データに追加
                if not is_deleted:
                    # インデックスから画像パスを取得
                    img_path = self.images[idx]
                    image_paths.append(img_path)
                    location_labels.append(location)
        
        # ラベルをインデックスに変換
        location_indices = [location_to_index[label] for label in location_labels]
        
        return {
            'image_paths': image_paths,
            'location_labels': location_labels,
            'location_indices': location_indices,
            'location_to_index': location_to_index
        }

    def _train_location_model_internal(self, model_type, train_loader, val_loader, num_classes, training_config, progress_callback):
        """位置モデル学習の内部実装（外部依存を排除）"""
        
        # デバイスの設定
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # モデルのロード（get_model関数を使用）
        if progress_callback:
            progress_callback(0, training_config['num_epochs'], "モデルをロード中...")
        
        model = self._initialize_location_model(model_type, num_classes, device)
        
        # 損失関数と最適化アルゴリズム
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        
        # トレーニングループ
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        best_val_acc = 0.0
        
        # Early Stopping用の変数
        early_stopping_counter = 0
        early_stopped = False
        stopped_epoch = 0
        
        # 保存ディレクトリとファイル名
        models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
        os.makedirs(models_dir, exist_ok=True)

        # カスタムモデル名が指定されていればそれを使用、ただしモデルタイプを必ず含める
        custom_name = training_config.get('model_name', '').strip()
        if custom_name:
            # カスタム名がある場合: モデルタイプ_カスタム名 の形式
            save_name = f"{model_type}_{custom_name}"
        else:
            # カスタム名がない場合: モデルタイプのみ
            save_name = model_type

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(models_dir, f'{save_name}.pth')
        best_model_path = os.path.join(models_dir, f'{save_name}_best.pth')
        
        completed_epochs = 0
        for epoch in range(training_config['num_epochs']):
            # 進捗コールバック - エポック開始
            if progress_callback:
                message = f"エポック {epoch+1}/{training_config['num_epochs']} 開始"
                should_continue = progress_callback(epoch, training_config['num_epochs'], message)
                if not should_continue:
                    break
            
            # トレーニングフェーズ
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # 統計情報を更新
                epoch_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            # エポック損失と精度の計算
            epoch_loss /= len(train_loader.dataset)
            epoch_accuracy = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            
            # 検証フェーズ
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_loss /= len(val_loader.dataset)
            val_accuracy = 100 * correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # 学習率の調整
            scheduler.step(val_loss)
            
            completed_epochs = epoch + 1
            
            # 進捗コールバック - エポック終了
            if progress_callback:
                message = f"エポック {epoch+1}/{training_config['num_epochs']}, 学習損失: {epoch_loss:.4f}, 検証損失: {val_loss:.4f}, "
                message += f"学習精度: {epoch_accuracy:.2f}%, 検証精度: {val_accuracy:.2f}%"
                should_continue = progress_callback(epoch + 1, training_config['num_epochs'], message)
                if not should_continue:
                    break
            
            # 最良モデルの保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'accuracy': best_val_acc,
                    'num_classes': num_classes
                }, best_model_path)
                
            elif val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'accuracy': best_val_acc,
                    'num_classes': num_classes
                }, best_model_path)
            else:
                # Early Stopping判定
                if training_config['use_early_stopping']:
                    early_stopping_counter += 1
                    if early_stopping_counter >= training_config['patience']:
                        early_stopped = True
                        stopped_epoch = epoch + 1
                        break
        
        # 最終モデルの保存
        torch.save({
            'epoch': completed_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'early_stopped': early_stopped,
            'stopped_epoch': stopped_epoch if early_stopped else completed_epochs,
            'num_classes': num_classes
        }, model_path)
        
        return {
            'model_name': model_type,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'model_path': model_path,
            'best_model_path': best_model_path,
            'completed_epochs': completed_epochs,
            'early_stopped': early_stopped,
            'stopped_epoch': stopped_epoch if early_stopped else completed_epochs
        }

    def _initialize_location_model(self, model_type, num_classes, device):
        """位置モデルの初期化"""
        
        # モデルを初期化（既存のget_model関数を使用）
        if 'donkey_location' in model_type:
            model = get_model(model_type, pretrained=True)
            model.classifier = nn.Linear(50, num_classes)  # 出力層を置き換え
        elif 'resnet18_location' in model_type:
            model = get_model(model_type, pretrained=True)
            if hasattr(model, 'regressor'):
                in_features = model.regressor.in_features if hasattr(model.regressor, 'in_features') else model.regressor[0].in_features
                model.regressor = nn.Linear(in_features, num_classes)
        else:
            model = get_model(model_type, pretrained=True)
        
        return model.to(device)

    def _save_location_model_metadata(self, model_path, num_classes, actual_classes, location_mapping):
        """位置モデルのメタデータを保存"""
        
        checkpoint = torch.load(model_path, map_location='cpu')
        checkpoint['num_classes'] = num_classes
        checkpoint['actual_classes'] = actual_classes
        checkpoint['location_mapping'] = location_mapping
        torch.save(checkpoint, model_path)

    def _log_location_training(self, model_type, training_results, training_config, dataset_info):
        """位置推論モデルの学習結果をMLflowに記録"""
        
        try:
            # MLflowManagerが初期化されていない場合は初期化
            if not hasattr(self, 'mlflow_manager'):
                self.mlflow_manager = MLflowManager(self.folder_path)
            
            # メトリクスを準備
            metrics = {
                "best_val_loss": training_results.get('best_val_loss', 0.0),
                "best_val_acc": training_results.get('best_val_acc', 0.0),
                "final_train_loss": training_results['train_losses'][-1] if 'train_losses' in training_results else 0.0,
                "final_val_loss": training_results['val_losses'][-1] if 'val_losses' in training_results else 0.0,
                "final_train_acc": training_results['train_accuracies'][-1] if 'train_accuracies' in training_results else 0.0,
                "final_val_acc": training_results['val_accuracies'][-1] if 'val_accuracies' in training_results else 0.0,
                "train_losses": training_results.get('train_losses', []),
                "val_losses": training_results.get('val_losses', []),
                "train_accuracies": training_results.get('train_accuracies', []),
                "val_accuracies": training_results.get('val_accuracies', []),
                "status": "early_stopped" if training_results.get('early_stopped', False) else "completed"
            }
            
            # 学習パラメータを準備
            training_params = {
                "model_type": model_type,
                "num_epochs": training_config['num_epochs'],
                "completed_epochs": training_results.get('completed_epochs', training_config['num_epochs']),
                "learning_rate": training_config['learning_rate'],
                "batch_size": training_config['batch_size'],
                "use_early_stopping": training_config['use_early_stopping'],
                "patience": training_config['patience'],
                "early_stopped": training_results.get('early_stopped', False),
                "augmentation_enabled": training_config['use_augmentation'],
                "coordinate_system": "classification",  # 位置推論特有
                "estimation_method": "cnn_classification",  # 位置推論特有
                "fixed_classes": dataset_info['num_classes'],
                "actual_classes": dataset_info['actual_classes'],
                "data_folder": self.folder_path if hasattr(self, 'folder_path') and self.folder_path else "unknown"
            }

            # MLflowに記録
            success = self.mlflow_manager.log_position_estimation_model(
                model_path=training_results['best_model_path'],
                training_params=training_params,
                metrics=metrics,
                dataset_info=dataset_info
            )
            
            if success:
                return "MLflowに学習履歴を記録しました。\n「MLflow比較」ボタンで結果を確認できます。"
            else:
                return "MLflowへの記録中にエラーが発生しました。"
                
        except ImportError:
            return "MLflowがインストールされていないため、学習履歴は記録されませんでした。\npip install mlflow でインストールできます。"
        except Exception as e:
            print(f"MLflow記録エラー: {e}")
            return f"MLflowへの記録中にエラーが発生しました: {str(e)}"

    def _show_location_training_success(self, model_type, training_results, training_config, dataset_info, mlflow_info):
        """位置モデル学習成功メッセージを表示"""
        
        # Early Stopping情報
        early_stopping_info = ""
        if training_config['use_early_stopping']:
            if training_results.get('early_stopped', False):
                early_stopping_info = f"Early Stopping: {training_results.get('stopped_epoch', 0)}エポックで停止\n"
            else:
                early_stopping_info = f"Early Stopping: 発動せず (忍耐値: {training_config['patience']})\n"
        
        # 学習時間情報
        time_info = ""
        if 'total_training_time' in training_results:
            from model_training import format_time
            total_time_str = format_time(training_results['total_training_time'])
            avg_epoch_time_str = format_time(training_results.get('avg_epoch_time', 0))
            time_info = f"学習時間: {total_time_str} (平均エポック時間: {avg_epoch_time_str})\n"
        
        # 学習完了メッセージ
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("学習完了")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(
            f"{model_type} 位置モデルを学習し保存しました: {os.path.basename(training_results['best_model_path'])}\n" +
            f"最良検証損失: {training_results['best_val_loss']:.6f}\n" +
            f"最良検証精度: {training_results['best_val_acc']:.2f}%\n" +
            f"実施エポック数: {training_results['completed_epochs']}/{training_config['num_epochs']}\n" +
            early_stopping_info +
            time_info +
            f"出力クラス数: {dataset_info['num_classes']} (実際の位置クラス数: {dataset_info['actual_classes']})\n" +
            f"学習データ数: {dataset_info['image_paths_count']}枚\n" +
            f"学習率: {training_config['learning_rate']}\n" +
            f"バッチサイズ: {training_config['batch_size']}\n" +
            f"データオーグメンテーション: {'有効' if training_config['use_augmentation'] else '無効'}\n\n" +
            f"{mlflow_info}"
        )
        ok_button = msg_box.addButton(QMessageBox.Ok)
        mlflow_button = msg_box.addButton("MLflowを開く", QMessageBox.ActionRole)
        msg_box.exec_()

        if msg_box.clickedButton() == mlflow_button:
            self.mlflow_manager.open_ui()


    def train_and_save_waypoint_model(self):
        """ウェイポイント推論モデルの学習"""

        if not self.images or not self.waypoint_annotations:
            QMessageBox.warning(self, "警告", "ウェイポイントモデルを学習するにはウェイポイントアノテーションが必要です。")
            return

        # 有効なウェイポイントアノテーションをチェック
        valid_waypoint_count = 0
        waypoint_counts = []
        deleted_count = 0

        for idx, waypoints in self.waypoint_annotations.items():
            # 削除マークされていないかチェック（actual_indexで判定）
            if hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes:
                deleted_count += 1
                continue

            if waypoints and len(waypoints) > 0:
                valid_waypoint_count += 1
                waypoint_counts.append(len(waypoints))

        if valid_waypoint_count < 5:
            QMessageBox.warning(self, "警告", f"ウェイポイントモデルを学習するには少なくとも5枚のアノテーションが必要です。現在: {valid_waypoint_count}枚")
            return

        # ウェイポイント数の統計
        if not waypoint_counts:
            QMessageBox.warning(self, "警告", "有効なウェイポイントアノテーションがありません。")
            return

        most_common_waypoint_count = max(set(waypoint_counts), key=waypoint_counts.count)

        # 選択されたモデル
        model_type = self.waypoint_model_combo.currentText()

        # アノテーション統計情報を収集（削除済みマークを考慮）
        total_images = len(self.images)
        annotated_images = len(self.waypoint_annotations)

        # 学習設定ダイアログを表示
        training_settings = self._create_waypoint_training_dialog(
            model_type, most_common_waypoint_count, total_images, annotated_images,
            valid_waypoint_count, deleted_count
        )

        if not training_settings.exec_():
            return

        # 設定値の取得
        training_config = self._get_waypoint_training_config(training_settings)

        try:
            # データの準備
            image_data = self._prepare_waypoint_training_data(training_config['num_waypoints'])

            if not image_data['image_paths']:
                QMessageBox.warning(self, "警告", "有効なウェイポイントアノテーションがありません。")
                return

            # スキップされた画像がある場合、ユーザーに通知
            if image_data['skipped_images']:
                skipped_count = len(image_data['skipped_images'])
                skipped_msg = f"以下の{skipped_count}枚の画像はwaypoint数が一致しないためスキップされます:\n\n"

                # 最初の10件のみ表示
                for i, skipped in enumerate(image_data['skipped_images'][:10]):
                    img_name = os.path.basename(skipped['path'])
                    skipped_msg += f"  {skipped['index']}: {img_name} - {skipped['reason']}\n"

                if skipped_count > 10:
                    skipped_msg += f"\n...他 {skipped_count - 10}件"

                skipped_msg += f"\n\n学習に使用される画像: {len(image_data['image_paths'])}枚\n続行しますか？"

                reply = QMessageBox.question(self, "確認", skipped_msg,
                                            QMessageBox.Yes | QMessageBox.No,
                                            QMessageBox.Yes)
                if reply == QMessageBox.No:
                    return

            # 進捗ダイアログを表示
            progress = QProgressDialog(
                f"ウェイポイントモデル '{model_type}' の学習データを準備中...",
                "キャンセル", 0, 100, self
            )
            progress.setWindowTitle("ウェイポイントモデル学習")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()

            # データセット作成
            train_loader, val_loader, dataset_info = create_waypoint_datasets(
                image_paths=image_data['image_paths'],
                waypoint_labels=image_data['waypoint_coordinates'],
                val_split=0.2,
                model_name=model_type,
                batch_size=training_config['batch_size'],
                use_augmentation=training_config['use_augmentation'],
                num_waypoints=training_config['num_waypoints']
            )

            progress.setValue(20)
            progress.setLabelText(f"モデル '{model_type}' を初期化中... ({training_config['num_waypoints']}ウェイポイント)")
            QApplication.processEvents()

            # 進捗コールバック関数
            def update_progress(current, total, message=None):
                if message:
                    progress.setLabelText(message)
                progress.setValue(20 + int(current * 70 / total))
                QApplication.processEvents()
                return not progress.wasCanceled()

            # モデル学習
            training_results = self._train_waypoint_model_internal(
                model_type=model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                num_waypoints=training_config['num_waypoints'],
                training_config=training_config,
                progress_callback=update_progress
            )

            # モデルのメタデータ保存
            self._save_waypoint_model_metadata(
                training_results['best_model_path'],
                training_config['num_waypoints']
            )

            # 学習曲線グラフを保存
            progress.setValue(92)
            progress.setLabelText("学習曲線を保存中...")
            QApplication.processEvents()

            self._save_waypoint_training_curve(
                training_results['best_model_path'],
                training_results['training_history']
            )

            progress.setValue(95)
            progress.setLabelText("MLflowに学習結果を記録中...")
            QApplication.processEvents()

            # MLflowに結果を記録
            mlflow_info = self._log_waypoint_training(
                model_type=model_type,
                training_results=training_results,
                training_config=training_config,
                dataset_info={
                    "total_annotations": len(self.waypoint_annotations),
                    "used_samples": len(image_data['image_paths']),
                    "train_samples": dataset_info['train_samples'],
                    "val_samples": dataset_info['val_samples'],
                    "input_shape": dataset_info['actual_image_size'],
                    "num_waypoints": training_config['num_waypoints']
                }
            )

            # モデルリストを更新
            self.refresh_waypoint_model_list()

            progress.setValue(100)
            progress.close()

            # 学習完了メッセージ
            self._show_waypoint_training_success(
                model_type=model_type,
                training_results=training_results,
                training_config=training_config,
                dataset_info={
                    "image_paths_count": len(image_data['image_paths']),
                    "num_waypoints": training_config['num_waypoints']
                },
                mlflow_info=mlflow_info
            )

        except Exception as e:
            if 'progress' in locals():
                progress.close()
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"ウェイポイントモデルの学習中にエラーが発生しました:\n{str(e)}")

    def _train_waypoint_model_internal(self, model_type, train_loader, val_loader, num_waypoints, training_config, progress_callback):
        """ウェイポイントモデルの内部学習ロジック"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # モデル初期化
        model = self._initialize_waypoint_model(model_type, num_waypoints, device)

        # 学習パラメータ設定
        criterion = nn.MSELoss()  # 回帰問題のMSE損失
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Early Stopping設定
        early_stopping = None
        if training_config['use_early_stopping']:
            from model_training import EarlyStopping
            early_stopping = EarlyStopping(patience=training_config['patience'], min_delta=0.001)

        # 学習ループ
        best_val_loss = float('inf')
        best_model_path = None
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'waypoint_losses': []  # 各waypointごとのloss履歴
        }

        # 保存ディレクトリとタイムスタンプ
        models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        import time
        start_time = time.time()
        epoch_times = []

        for epoch in range(training_config['num_epochs']):
            epoch_start_time = time.time()

            # トレーニング
            model.train()
            train_loss = 0.0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 検証
            model.eval()
            val_loss = 0.0
            waypoint_losses = [0.0] * (num_waypoints * 2)  # 各x, y座標のloss

            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    # 各waypoint座標ごとのlossを計算
                    for i in range(num_waypoints * 2):
                        coord_loss = ((outputs[:, i] - targets[:, i]) ** 2).mean().item()
                        waypoint_losses[i] += coord_loss

            val_loss /= len(val_loader)
            waypoint_losses = [wl / len(val_loader) for wl in waypoint_losses]

            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['waypoint_losses'].append(waypoint_losses)

            # 学習率スケジューラー更新
            scheduler.step(val_loss)

            # ベストモデル保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 最初のエポックで保存パスを設定（タイムスタンプを固定）
                if best_model_path is None:
                    # カスタムモデル名が指定されていればそれを使用、ただしモデルタイプを必ず含める
                    custom_name = training_config.get('model_name', '').strip()
                    if custom_name:
                        # カスタム名がある場合: モデルタイプ_カスタム名 の形式
                        save_name = f"{model_type}_{custom_name}"
                    else:
                        # カスタム名がない場合: モデルタイプのみ
                        save_name = model_type
                    model_filename = f"{save_name}.pth"
                    best_model_path = os.path.join(models_dir, model_filename)

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_type': model_type,
                    'num_waypoints': num_waypoints,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, best_model_path)

            # 進捗更新
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            message = f"Epoch {epoch+1}/{training_config['num_epochs']}: 訓練損失={train_loss:.6f}, 検証損失={val_loss:.6f}"
            if not progress_callback(epoch + 1, training_config['num_epochs'], message):
                break

            # Early Stoppingチェック
            if early_stopping and early_stopping(val_loss):
                break

        total_time = time.time() - start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

        return {
            'best_model_path': best_model_path,
            'best_val_loss': best_val_loss,
            'completed_epochs': len(training_history['train_loss']),
            'training_history': training_history,
            'total_training_time': total_time,
            'avg_epoch_time': avg_epoch_time,
            'early_stopped': early_stopping.early_stop if early_stopping else False,
            'stopped_epoch': len(training_history['train_loss']) if early_stopping and early_stopping.early_stop else None
        }

    def _initialize_waypoint_model(self, model_type, num_waypoints, device):
        """ウェイポイントモデルを初期化"""
        from model_catalog import get_model

        # モデルのロード
        model = get_model(model_type, pretrained=True, input_size=(224, 224))
        model.num_waypoints = num_waypoints  # ウェイポイント数を設定
        model.to(device)

        return model

    def _save_waypoint_model_metadata(self, model_path, num_waypoints):
        """ウェイポイントモデルのメタデータを保存"""
        checkpoint = torch.load(model_path, map_location='cpu')
        checkpoint['num_waypoints'] = num_waypoints
        torch.save(checkpoint, model_path)

    def _save_waypoint_training_curve(self, model_path, training_history):
        """ウェイポイントモデルの学習曲線をグラフとして保存"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # グラフのファイル名を生成
            graph_path = model_path.replace('.pth', '_training_curve.png')

            epochs = range(1, len(training_history['train_loss']) + 1)

            # waypoint詳細lossがあるかチェック
            has_waypoint_losses = 'waypoint_losses' in training_history and training_history['waypoint_losses']

            if has_waypoint_losses:
                # 3列のサブプロット: 全体loss、waypointごとのloss、座標ごとのloss
                fig = plt.figure(figsize=(20, 6))
                gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[0, 2])
            else:
                # waypoint詳細lossがない場合は全体lossのみ
                fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

            # ===== グラフ1: 全体の学習曲線 =====
            ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

            # 最小検証損失のエポックをマーク
            min_val_loss_epoch = training_history['val_loss'].index(min(training_history['val_loss'])) + 1
            min_val_loss = min(training_history['val_loss'])
            ax1.axvline(x=min_val_loss_epoch, color='g', linestyle='--', alpha=0.7,
                       label=f'Best (Epoch {min_val_loss_epoch})')

            ax1.set_title('Overall Training Curve', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=11)
            ax1.set_ylabel('Loss (MSE)', fontsize=11)
            ax1.legend(fontsize=10, loc='upper right')
            ax1.grid(True, alpha=0.3)

            # ===== グラフ2 & 3: 各waypointごとの詳細loss =====
            if has_waypoint_losses:
                waypoint_losses_array = np.array(training_history['waypoint_losses'])
                num_coords = waypoint_losses_array.shape[1]
                num_waypoints = num_coords // 2

                colors = plt.cm.tab10(np.linspace(0, 1, num_waypoints))

                # グラフ2: 各waypointの合計loss (x + y)
                for wp_idx in range(num_waypoints):
                    x_idx = wp_idx * 2
                    y_idx = wp_idx * 2 + 1
                    waypoint_total_loss = waypoint_losses_array[:, x_idx] + waypoint_losses_array[:, y_idx]
                    ax2.plot(epochs, waypoint_total_loss, color=colors[wp_idx],
                            label=f'Waypoint {wp_idx + 1}', linewidth=2)

                ax2.set_title('Loss per Waypoint (X + Y)', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Epoch', fontsize=11)
                ax2.set_ylabel('Loss (MSE)', fontsize=11)
                ax2.legend(fontsize=10, loc='upper right')
                ax2.grid(True, alpha=0.3)

                # グラフ3: 各座標(x, y)のloss
                for wp_idx in range(num_waypoints):
                    x_idx = wp_idx * 2
                    y_idx = wp_idx * 2 + 1
                    ax3.plot(epochs, waypoint_losses_array[:, x_idx], color=colors[wp_idx],
                            linestyle='-', label=f'WP{wp_idx + 1}-X', linewidth=1.5, alpha=0.8)
                    ax3.plot(epochs, waypoint_losses_array[:, y_idx], color=colors[wp_idx],
                            linestyle='--', label=f'WP{wp_idx + 1}-Y', linewidth=1.5, alpha=0.8)

                ax3.set_title('Loss per Coordinate (X: solid, Y: dashed)', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Epoch', fontsize=11)
                ax3.set_ylabel('Loss (MSE)', fontsize=11)
                ax3.legend(fontsize=9, loc='upper right', ncol=2)
                ax3.grid(True, alpha=0.3)

            # 全体のタイトル
            fig.suptitle('Waypoint Model Training Analysis', fontsize=16, fontweight='bold', y=0.98)

            # グラフを保存
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(graph_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"学習曲線を保存しました: {graph_path}")

        except Exception as e:
            print(f"学習曲線の保存に失敗しました: {e}")
            import traceback
            traceback.print_exc()

    def _log_waypoint_training(self, model_type, training_results, training_config, dataset_info):
        """ウェイポイント推論モデルの学習結果をMLflowに記録"""

        try:
            # MLflowManagerが初期化されていない場合は初期化
            if not hasattr(self, 'mlflow_manager'):
                self.mlflow_manager = MLflowManager(self.folder_path)

            # 学習パラメータを記録
            params = {
                'model_type': model_type,
                'num_waypoints': training_config['num_waypoints'],
                'learning_rate': training_config['learning_rate'],
                'batch_size': training_config['batch_size'],
                'num_epochs': training_config['num_epochs'],
                'use_augmentation': training_config['use_augmentation'],
                'use_early_stopping': training_config['use_early_stopping']
            }

            if training_config['use_early_stopping']:
                params['patience'] = training_config['patience']

            # メトリクスを記録
            metrics = {
                'best_val_loss': training_results['best_val_loss'],
                'completed_epochs': training_results['completed_epochs'],
                'total_training_time': training_results['total_training_time'],
                'avg_epoch_time': training_results['avg_epoch_time'],
                'train_samples': dataset_info['train_samples'],
                'val_samples': dataset_info['val_samples'],
                'total_annotations': dataset_info['total_annotations'],
                'used_samples': dataset_info['used_samples']
            }

            # アーティファクトを記録
            artifacts = {
                'model_path': training_results['best_model_path']
            }

            # MLflowに記録 - ウェイポイント専用メソッドを使用
            run_info = self.mlflow_manager.log_waypoint_regression_model(
                model_path=training_results['best_model_path'],
                training_params={
                    'model_type': model_type,
                    'num_waypoints': training_config['num_waypoints'],
                    'learning_rate': training_config['learning_rate'],
                    'batch_size': training_config['batch_size'],
                    'num_epochs': training_config['num_epochs'],
                    'completed_epochs': training_results['completed_epochs'],
                    'use_early_stopping': training_config['use_early_stopping'],
                    'patience': training_config['patience'] if training_config['use_early_stopping'] else 0,
                    'use_augmentation': training_config['use_augmentation'],
                    'data_folder': self.folder_path
                },
                metrics={
                    'best_val_loss': training_results['best_val_loss'],
                    'completed_epochs': training_results['completed_epochs'],
                    'total_training_time': training_results['total_training_time'],
                    'avg_epoch_time': training_results['avg_epoch_time'],
                    'status': 'completed'
                },
                dataset_info={
                    'train_samples': dataset_info['train_samples'],
                    'val_samples': dataset_info['val_samples'],
                    'total_annotations': dataset_info['total_annotations'],
                    'used_samples': dataset_info['used_samples']
                }
            )

            return f"MLflow Run ID: {run_info['run_id']}"

        except Exception as e:
            print(f"MLflow記録エラー: {e}")
            return "MLflow記録に失敗しました"

    def _show_waypoint_training_success(self, model_type, training_results, training_config, dataset_info, mlflow_info):
        """ウェイポイントモデル学習成功メッセージを表示"""

        # Early Stopping情報
        early_stopping_info = ""
        if training_config['use_early_stopping']:
            if training_results.get('early_stopped', False):
                early_stopping_info = f"Early Stopping: {training_results.get('stopped_epoch', 0)}エポックで停止\n"
            else:
                early_stopping_info = f"Early Stopping: 発動せず (忍耐値: {training_config['patience']})\n"

        # 学習時間情報
        time_info = ""
        if 'total_training_time' in training_results:
            from model_training import format_time
            total_time_str = format_time(training_results['total_training_time'])
            avg_epoch_time_str = format_time(training_results.get('avg_epoch_time', 0))
            time_info = f"学習時間: {total_time_str} (平均エポック時間: {avg_epoch_time_str})\n"

        # 学習完了メッセージ
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("学習完了")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(
            f"{model_type} ウェイポイントモデルを学習し保存しました: {os.path.basename(training_results['best_model_path'])}\n" +
            f"最良検証損失: {training_results['best_val_loss']:.6f}\n" +
            f"実施エポック数: {training_results['completed_epochs']}/{training_config['num_epochs']}\n" +
            early_stopping_info +
            time_info +
            f"ウェイポイント数: {dataset_info['num_waypoints']}\n" +
            f"学習データ数: {dataset_info['image_paths_count']}枚\n" +
            f"学習率: {training_config['learning_rate']}\n" +
            f"バッチサイズ: {training_config['batch_size']}\n" +
            f"データオーグメンテーション: {'有効' if training_config['use_augmentation'] else '無効'}\n\n" +
            f"{mlflow_info}"
        )
        ok_button = msg_box.addButton(QMessageBox.Ok)
        mlflow_button = msg_box.addButton("MLflowを開く", QMessageBox.ActionRole)
        msg_box.exec_()

        if msg_box.clickedButton() == mlflow_button:
            self.mlflow_manager.open_ui()

    def update_location_inference_display(self):
        """位置推論表示を更新する - 上位3クラスのみ表示"""
        if not self.images:
            return
        
        current_index = self.current_index
    
        if hasattr(self, 'location_inference_checkbox') and self.location_inference_checkbox.isChecked():
            if current_index in self.location_inference_results:
                # 推論結果を取得
                result = self.location_inference_results[current_index]
                pred_class = result['pred_class']
                confidence = result['confidence']
                all_probs = result.get('all_probs', [])
                
                # 情報テキストを構築（インライン表示）
                inference_text = "<b>位置推論結果:</b> "
                
                # 一番高いクラスは背景色付きで表示
                loc_color = get_location_color(pred_class)
                
                # 予測クラスを背景色付きで表示
                inference_text += f"<div style='background-color: {loc_color.name()};'>"
                #inference_text += f"<div style='background-color: {loc_color.name()}; color: white; font-weight: bold; padding: 5px; border-radius: 5px; margin: 5px 0;'>"
                #inference_text += f"予測位置: {pred_class} (確信度: {confidence:.4f})</div>"
                
                # 上位3クラスの予測結果を表示（すでに確率でソートされている前提）
                if all_probs:
                    # 確率が高い順にインデックスをソート
                    sorted_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)
                    
                    # 上位3つ（または全部、少ない方）を取得
                    top_k = min(3, len(sorted_indices))
                    top_indices = sorted_indices[:top_k]
                    
                    inference_text += "" #"<br>上位クラス:<br>"
                    
                    for i, idx in enumerate(top_indices):
                        # すでに表示したクラスは飛ばす
                        if idx == pred_class and i > 0:
                            continue
                        
                        # 位置によって色を変える
                        if i == 0:
                            color = "white"
                        else:
                            color = get_location_color(idx).name()
                        
                        # 各クラスの予測確率
                        inference_text += f"<span style='color: {color}; font-weight: bold;'>{i+1}. 位置 {idx}: {all_probs[idx]:.4f}</span><br>"
                
                # HTMLタグを閉じる
                inference_text += "</div>"
                
                # テキストをラベルに設定（ラベルは初期化時に作成済み）
                if hasattr(self, 'location_inference_info_label'):
                    # ラベルを確実に表示
                    self.location_inference_info_label.show()
                    self.location_inference_info_label.setVisible(True)
                    
                    self.location_inference_info_label.setText(inference_text)
                    self.location_inference_info_label.setTextFormat(Qt.RichText)
                    
                    # 即座に更新を強制
                    self.location_inference_info_label.repaint()
                    self.location_inference_info_label.update()
                    
                    # 親ウィジェットと情報パネル全体も更新
                    if self.location_inference_info_label.parent():
                        self.location_inference_info_label.parent().update()
                    
                    # 情報パネル全体を更新
                    if hasattr(self, 'info_scroll'):
                        self.info_scroll.update()
                        
                    # メインウィンドウも更新
                    self.update()
                    
                    QApplication.processEvents()
                
                return True
                    
            # モデルがロードされていて推論結果がない場合は実行
            elif hasattr(self, 'location_model_manager') and self.location_model_manager.is_model_loaded():
                self.run_location_inference()
                # 再帰的に呼び出して表示を更新
                return self.update_location_inference_display()
        else:
            # 表示がオフの場合はラベルをクリア（スペースで高さを維持）
            if hasattr(self, 'location_inference_info_label'):
                self.location_inference_info_label.setText(" ")
        
        return False
    
    ###

    def update_slider_deleted_indexes(self):
        """スライダーの削除済みインデックス表示を更新"""
        if hasattr(self, 'image_slider') and isinstance(self.image_slider, DeletedIndexesSlider):
            self.image_slider.setDeletedIndexes(self.deleted_indexes, len(self.images))

    def update_slider_downsampled_indexes(self):
        """スライダーのダウンサンプリング対象インデックス表示を更新"""
        if hasattr(self, 'image_slider') and isinstance(self.image_slider, DeletedIndexesSlider):
            self.image_slider.setDownsampledIndexes(self.downsampled_indexes, len(self.images))

    ###
    def _export_segmentation_subset(self, indices, output_dir, class_to_index):
        """セグメンテーションサブセットのエクスポート - クラス名修正版"""
        
        success_count = 0
        
        for idx in indices:
            if idx in self.segmentation_annotations:
                try:
                    # 画像をコピー
                    source_image_path = self.images[idx]
                    image_filename = os.path.basename(source_image_path)
                    dest_image_path = os.path.join(output_dir, "images", image_filename)
                    
                    import shutil
                    shutil.copy2(source_image_path, dest_image_path)
                    
                    # セグメンテーションアノテーションを処理
                    label_filename = os.path.splitext(image_filename)[0] + ".txt"
                    label_path = os.path.join(output_dir, "labels", label_filename)
                    
                    # 画像サイズを取得
                    from PIL import Image
                    with Image.open(source_image_path) as img:
                        img_width, img_height = img.size
                    
                    print(f"処理中: {image_filename} (サイズ: {img_width}x{img_height})")
                    
                    # ラベルファイルを作成（セグメンテーション形式）
                    label_lines = []
                    valid_segments = 0
                    
                    for seg_idx, seg in enumerate(self.segmentation_annotations[idx]):
                        # クラス名を取得 - 複数のキーを試す
                        class_name = None
                        points = []
                        
                        if isinstance(seg, dict):
                            # 複数のキー名で試行
                            class_name = seg.get('class_name') or seg.get('class') or seg.get('label')
                            points = seg.get('points', [])
                            
                            print(f"  セグメント {seg_idx}: 辞書キー={list(seg.keys())}")
                            print(f"    class_name={seg.get('class_name')}, class={seg.get('class')}")
                            
                        else:
                            # オブジェクト形式の場合
                            class_name = (getattr(seg, 'class_name', None) or 
                                        getattr(seg, 'class', None) or 
                                        getattr(seg, 'label', None))
                            points = getattr(seg, 'points', [])
                        
                        print(f"  セグメント {seg_idx}: 最終クラス名={class_name}, ポイント数={len(points)}")
                        
                        # クラス名が有効で、期待されるクラスリストに含まれているかチェック
                        if class_name and class_name in class_to_index and points and len(points) >= 3:
                            class_id = class_to_index[class_name]
                            
                            # ポイントを正規化座標に変換
                            normalized_points = []
                            valid_points = 0
                            
                            for point_idx, point in enumerate(points):
                                try:
                                    # タプル形式の座標を処理
                                    if isinstance(point, tuple) and len(point) >= 2:
                                        x = float(point[0])
                                        y = float(point[1])
                                    elif isinstance(point, dict):
                                        x = float(point.get('x', 0))
                                        y = float(point.get('y', 0))
                                    elif hasattr(point, 'x') and hasattr(point, 'y'):
                                        x = float(point.x)
                                        y = float(point.y)
                                    else:
                                        print(f"    警告: ポイント {point_idx} の形式が不正: {type(point)}, 値: {point}")
                                        continue
                                    
                                    # 正規化（0-1の範囲）
                                    x_norm = x / img_width
                                    y_norm = y / img_height
                                    
                                    # 座標を0-1の範囲にクランプ
                                    x_norm = max(0.0, min(1.0, x_norm))
                                    y_norm = max(0.0, min(1.0, y_norm))
                                    
                                    normalized_points.extend([x_norm, y_norm])
                                    valid_points += 1
                                    
                                except (ValueError, TypeError) as e:
                                    print(f"    エラー: ポイント {point_idx} の変換失敗: {e}")
                                    continue
                            
                            print(f"    有効ポイント数: {valid_points}")
                            
                            # YOLO セグメンテーション形式で書き込み
                            if len(normalized_points) >= 6:  # 最低3点 (6座標)
                                points_str = ' '.join([f"{coord:.6f}" for coord in normalized_points])
                                label_line = f"{class_id} {points_str}"
                                label_lines.append(label_line)
                                valid_segments += 1
                                print(f"    ✓ ラベル行作成: クラス{class_id} ({class_name}), {valid_points}点")
                            else:
                                print(f"    警告: 有効ポイント不足 ({len(normalized_points)//2}点)")
                        else:
                            print(f"    スキップ: クラス名='{class_name}', 有効クラス={class_name in class_to_index if class_name else False}, ポイント数={len(points)}")
                    
                    # ラベルファイルに書き込み
                    if label_lines:
                        with open(label_path, 'w') as f:
                            for line in label_lines:
                                f.write(line + '\n')
                        print(f"  ✓ ラベルファイル作成成功: {len(label_lines)}行")
                    else:
                        # 空のラベルファイルを作成
                        with open(label_path, 'w') as f:
                            pass
                        print(f"  ❌ 警告: 有効なセグメンテーションなし、空ファイル作成")
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"セグメンテーション インデックス {idx} の処理中にエラー: {e}")
                    import traceback
                    traceback.print_exc()
        
        return success_count

    def debug_segmentation_data(self):
        """セグメンテーションデータのデバッグ用メソッド - 強化版"""
        
        print("\n=== セグメンテーションデータ詳細確認 ===")
        
        for idx, segments in list(self.segmentation_annotations.items())[:3]:
            print(f"\nインデックス {idx}:")
            print(f"  セグメント数: {len(segments) if segments else 0}")
            
            if segments:
                for seg_idx, seg in enumerate(segments):
                    print(f"  セグメント {seg_idx}:")
                    print(f"    タイプ: {type(seg)}")
                    
                    if isinstance(seg, dict):
                        print(f"    全キー: {list(seg.keys())}")
                        # 複数のクラス名キーを確認
                        for key in ['class_name', 'class', 'label', 'category']:
                            if key in seg:
                                print(f"    {key}: {seg[key]}")
                        
                        points = seg.get('points', [])
                        print(f"    ポイント数: {len(points)}")
                        if points and len(points) > 0:
                            print(f"    最初のポイント: {points[0]} (タイプ: {type(points[0])})")
                            if isinstance(points[0], tuple):
                                print(f"      タプル座標: x={points[0][0]}, y={points[0][1]}")
                            elif isinstance(points[0], dict):
                                print(f"      辞書座標: x={points[0].get('x')}, y={points[0].get('y')}")
                    else:
                        # オブジェクト属性を確認
                        attrs = [attr for attr in dir(seg) if not attr.startswith('_')]
                        print(f"    利用可能な属性: {attrs}")
                        
                        for attr in ['class_name', 'class', 'label']:
                            if hasattr(seg, attr):
                                print(f"    {attr}: {getattr(seg, attr)}")
                        
                        points = getattr(seg, 'points', [])
                        print(f"    ポイント数: {len(points)}")

    # アノテーション検証の改善
    def _validate_yolo_annotations(self, task_type):
        """YOLOアノテーションの検証 - クラス名確認強化版（削除マーク除外対応）"""
        
        if task_type == "detect":
            if not hasattr(self, 'bbox_annotations') or not self.bbox_annotations:
                QMessageBox.warning(self, "警告", "物体検知アノテーションがありません。")
                return None, None
            
            # 削除マークされていないアノテーションのみ抽出
            valid_annotations = {}
            excluded_count = 0
            
            for idx, boxes in self.bbox_annotations.items():
                # 削除マークされていないかチェック（actual_indexで判定）
                if hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes:
                    excluded_count += 1
                    continue
                
                # 有効なアノテーションのみ追加
                if boxes:
                    valid_annotations[idx] = boxes
            
            if not valid_annotations:
                QMessageBox.warning(self, "警告", f"有効な物体検知アノテーションがありません。\n（削除マーク済み: {excluded_count}件）")
                return None, None
            
            annotations = valid_annotations
            total_boxes = sum(len(boxes) for boxes in annotations.values())
            return annotations, {"total_count": total_boxes, "image_count": len(annotations), "excluded_count": excluded_count}
        
        if task_type == "segment":
            if not self.segmentation_annotations:
                QMessageBox.warning(self, "警告", "セグメンテーションアノテーションがありません。")
                return None, None
            
            # 削除マークされていないアノテーションのみ抽出
            valid_annotations = {}
            excluded_count = 0
            
            for idx, segments in self.segmentation_annotations.items():
                # 削除マークされていないかチェック（actual_indexで判定）
                if hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes:
                    excluded_count += 1
                    continue
                
                # 有効なアノテーションのみ追加
                if segments:
                    valid_annotations[idx] = segments
            
            annotations = valid_annotations
            
            # セグメンテーションデータの詳細検証
            total_segments = 0
            valid_segments = 0
            valid_images = 0
            class_names_found = set()
            
            for index, segments in annotations.items():
                image_has_valid_segments = False
                if segments and len(segments) > 0:
                    total_segments += len(segments)
                    for seg in segments:
                        # クラス名を取得
                        class_name = None
                        if isinstance(seg, dict):
                            class_name = seg.get('class_name') or seg.get('class') or seg.get('label')
                        else:
                            class_name = (getattr(seg, 'class_name', None) or 
                                        getattr(seg, 'class', None) or 
                                        getattr(seg, 'label', None))
                        
                        if class_name:
                            class_names_found.add(class_name)
                        
                        # ポイント数チェック
                        points = None
                        if isinstance(seg, dict):
                            points = seg.get('points', [])
                        else:
                            points = getattr(seg, 'points', [])
                        
                        if class_name and points and len(points) >= 3:
                            valid_segments += 1
                            image_has_valid_segments = True
                
                if image_has_valid_segments:
                    valid_images += 1
            
            print(f"\n=== セグメンテーションアノテーション確認 ===")
            print(f"アノテーション辞書数: {len(annotations)}")
            print(f"総セグメンテーション数: {total_segments}")
            print(f"有効なセグメンテーション数: {valid_segments}")
            print(f"有効なセグメンテーションがある画像数: {valid_images}")
            print(f"発見されたクラス名: {sorted(class_names_found)}")
            print("=" * 50)
            
            # クラス名の不一致チェック
            expected_classes = set(self.get_current_classes())
            missing_classes = expected_classes - class_names_found
            extra_classes = class_names_found - expected_classes
            
            if missing_classes:
                print(f"警告: 期待されるクラスが見つからない: {missing_classes}")
            if extra_classes:
                print(f"警告: 予期しないクラスが見つかった: {extra_classes}")
            
            if valid_segments == 0:
                QMessageBox.critical(
                    self,
                    "セグメンテーションデータなし",
                    f"有効なセグメンテーションアノテーションが見つかりません。\n\n"
                    f"発見されたクラス名: {sorted(class_names_found)}\n"
                    f"期待されるクラス名: {sorted(expected_classes)}\n\n"
                    f"クラス名が一致していることを確認してください。"
                )
                return None, None
            
            return annotations, {"total_count": valid_segments, "image_count": valid_images, "excluded_count": excluded_count}
        
        # ... 検出タスクの処理は既存と同じ ...
        
        return None, None

# 追加機能切り出し　削除表示
class DeletedIndexesSlider(QSlider):
    """削除済みインデックスとダウンサンプリングインデックスを視覚的に表示するカスタムスライダー"""

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.deleted_indexes = []  # 削除済みインデックスのリスト
        self.downsampled_indexes = []  # ダウンサンプリング対象インデックスのリスト
        self.total_count = 0       # 総インデックス数

    def setDeletedIndexes(self, deleted_indexes, total_count):
        """削除済みインデックスを設定"""
        self.deleted_indexes = deleted_indexes
        self.total_count = total_count
        self.update()  # スライダーを再描画

    def setDownsampledIndexes(self, downsampled_indexes, total_count):
        """ダウンサンプリング対象インデックスを設定"""
        self.downsampled_indexes = downsampled_indexes
        self.total_count = total_count
        self.update()  # スライダーを再描画

    def paintEvent(self, event):
        """削除インデックス（赤）とダウンサンプリングインデックス（青）を表示するスライダー"""
        # 最初にスライダー全体を通常通り描画（ハンドルも含む）
        super().paintEvent(event)

        # カスタム描画開始
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        try:
            # スタイルオプション初期化
            option = QStyleOptionSlider()
            self.initStyleOption(option)

            # トラックの矩形を取得
            groove_rect = self.style().subControlRect(
                QStyle.CC_Slider, option, QStyle.SC_SliderGroove, self
            )

            track_length = groove_rect.width()
            track_start = groove_rect.x()
            track_height = groove_rect.height()

            # ダウンサンプリングインデックス描画（青マーク）- 先に描画（下層）
            if self.downsampled_indexes and self.total_count > 0:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(50, 100, 255, 180)))

                for idx in self.downsampled_indexes:
                    if 0 <= idx < self.total_count:
                        position = track_start + (idx / (self.total_count - 1)) * track_length
                        mark_width = max(3, track_length / self.total_count)
                        mark_height = track_height + 6

                        painter.drawRect(
                            int(position - mark_width / 2),
                            int(groove_rect.center().y()),
                            int(mark_width),
                            int(mark_height / 2)
                        )

            # 削除インデックス描画（赤マーク）- 後に描画（上層）
            if self.deleted_indexes and self.total_count > 0:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(255, 50, 50, 180)))

                for idx in self.deleted_indexes:
                    if 0 <= idx < self.total_count:
                        position = track_start + (idx / (self.total_count - 1)) * track_length
                        mark_width = max(3, track_length / self.total_count)
                        mark_height = track_height + 6

                        painter.drawRect(
                            int(position - mark_width / 2),
                            int(groove_rect.center().y()),
                            int(mark_width),
                            int(mark_height / 2)
                        )

        finally:
            painter.end()


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