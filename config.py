# config.py
"""
アプリケーション設定定数
"""
import os

# ===========================================
# アプリケーション基本設定
# ===========================================
APP_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# ディレクトリ名
SESSION_DIR_NAME = "sessions"
MODELS_DIR_NAME = "models"
ANNOTATION_DIR_NAME = "annotation"
DATA_DONKEY_DIR_NAME = "data_donkey"
DATA_JETRACER_DIR_NAME = "data_jetracer"
DATA_YOLO_DIR_NAME = "data_yolo"
VIDEO_DIR_NAME = "video"

# ===========================================
# パス設定（動的生成）
# ===========================================
annotation_folder = os.path.join(APP_DIR_PATH, ANNOTATION_DIR_NAME)
os.makedirs(annotation_folder, exist_ok=True)
donkey_dataset_dir = os.path.join(annotation_folder, DATA_DONKEY_DIR_NAME)
os.makedirs(donkey_dataset_dir, exist_ok=True)
jetracer_dataset_dir = os.path.join(annotation_folder, DATA_JETRACER_DIR_NAME)
os.makedirs(jetracer_dataset_dir, exist_ok=True)
yolo_dataset_dir = os.path.join(annotation_folder, DATA_YOLO_DIR_NAME)
os.makedirs(yolo_dataset_dir, exist_ok=True)
video_folder = os.path.join(annotation_folder, VIDEO_DIR_NAME)
os.makedirs(video_folder, exist_ok=True)

models_dir = os.path.join(APP_DIR_PATH, MODELS_DIR_NAME)
os.makedirs(models_dir, exist_ok=True)
mlflow_dir = os.path.join(APP_DIR_PATH, "mlruns")
os.makedirs(mlflow_dir, exist_ok=True)
# パスの正規化 - すべてのバックスラッシュをフォワードスラッシュに変換
normalized_path = mlflow_dir.replace("\\", "/")

session_dir = os.path.join(APP_DIR_PATH, SESSION_DIR_NAME)
os.makedirs(session_dir, exist_ok=True)

# ===========================================
# UI関連設定
# ===========================================
# メインウィンドウ
MAIN_WINDOW_WIDTH = 1600
MAIN_WINDOW_HEIGHT = 900
MAIN_WINDOW_X = 100
MAIN_WINDOW_Y = 100

# 左パネル
LEFT_PANEL_MAX_WIDTH = 300

# 画像表示
DEFAULT_ZOOM_FACTOR = 2.5
MAIN_IMAGE_MIN_WIDTH = 1000
MAIN_IMAGE_MIN_HEIGHT = 800
MAIN_IMAGE_VIEW_MIN_WIDTH = 800
MAIN_IMAGE_VIEW_MIN_HEIGHT = 600

# グリッド表示
DEFAULT_GRID_SIZE = 10

# ギャラリー表示
GALLERY_MIN_HEIGHT = 175
GALLERY_COL_COUNT = 5
GALLERY_GRID_SPACING = 2
GALLERY_GRID_MARGIN = 0

# サムネイル
THUMBNAIL_WIDTH = 210
THUMBNAIL_HEIGHT = 170
THUMBNAIL_IMAGE_WIDTH = 150
THUMBNAIL_IMAGE_HEIGHT = 140
THUMBNAIL_MIN_IMAGE_WIDTH = 150
THUMBNAIL_MIN_IMAGE_HEIGHT = 120
THUMBNAIL_INFO_PANEL_WIDTH = 70
THUMBNAIL_FILENAME_HEIGHT = 10

# 情報パネル
INFO_PANEL_WIDTH = 280
INFO_PANEL_MARGIN = 20
INFO_PANEL_MIN_WIDTH = 200
DISTRIBUTION_GRAPH_HEIGHT = 400

# スライダー
SLIDER_TICK_INTERVAL = 10

# ===========================================
# アノテーション関連設定
# ===========================================
# 描画サイズ
ANNOTATION_CIRCLE_SIZE = 15
ANNOTATION_PEN_WIDTH = 4
INFERENCE_CIRCLE_SIZE = 15
INFERENCE_PEN_WIDTH = 4

# バウンディングボックス
BBOX_MIN_SIZE = 10
BBOX_HANDLE_SIZE = 8
BBOX_HOVER_HANDLE_SIZE = 10
BBOX_PEN_WIDTH_NORMAL = 2
BBOX_PEN_WIDTH_HOVERED = 2.5
BBOX_PEN_WIDTH_SELECTED = 3

# セグメンテーション
SEGMENTATION_CLOSE_THRESHOLD = 15
SEGMENTATION_VERTEX_RADIUS = 8
SEGMENTATION_VERTEX_HANDLE_RADIUS = 8
SEGMENTATION_HOVER_VERTEX_RADIUS = 5

# ===========================================
# 色設定（QColorは使用側で生成）
# ===========================================
# 基本色
COLOR_RED = (255, 0, 0, 180)
COLOR_GREEN = (0, 255, 0, 180)
COLOR_BLUE = (0, 0, 255, 180)
COLOR_YELLOW = (255, 255, 0, 180)
COLOR_GRAY = (128, 128, 128, 180)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# クラス別色（バウンディングボックス）
CLASS_COLORS = {
    'car': COLOR_RED,
    'person': COLOR_GREEN,
    'sign': COLOR_BLUE,
    'cone': COLOR_YELLOW,
    'unknown': COLOR_GRAY
}

# クラス別色（セグメンテーション）
SEGMENTATION_CLASS_COLORS = {
    'car': (255, 0, 0, 120),
    'person': (0, 255, 0, 120),
    'sign': (0, 0, 255, 120),
    'cone': (255, 255, 0, 120),
    'unknown': (128, 128, 128, 120)
}

# 推論結果色（物体検知）
DETECTION_INFERENCE_CLASS_COLORS = {
    'car': (255, 0, 0, 120),
    'person': (0, 255, 0, 120),
    'sign': (0, 0, 255, 120),
    'cone': (255, 255, 0, 120),
    'unknown': (128, 128, 128, 120)
}

# 推論結果色（文字表示用）
DETECTION_INFERENCE_TEXT_COLORS = {
    'car': "#FF0000",
    'person': "#00FF00",
    'sign': "#0000FF",
    'cone': "#FFFF00",
    'unknown': "#808080"
}

# ===========================================
# モデル学習関連設定
# ===========================================
# デフォルト学習パラメータ
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_PATIENCE = 5
DEFAULT_EARLY_STOPPING = True

# YOLO学習パラメータ
YOLO_DEFAULT_EPOCHS = 30
YOLO_DEFAULT_BATCH_SIZE = 16
YOLO_DEFAULT_IMG_SIZE = 640
YOLO_DEFAULT_CONFIDENCE = 0.6
YOLO_DEFAULT_LEARNING_RATE = 0.001
YOLO_DEFAULT_PATIENCE = 10

# 位置モデル学習パラメータ
LOCATION_DEFAULT_EPOCHS = 30
LOCATION_DEFAULT_BATCH_SIZE = 16
LOCATION_DEFAULT_LEARNING_RATE = 0.001
LOCATION_DEFAULT_PATIENCE = 5
LOCATION_DEFAULT_NUM_CLASSES = 8

# オーグメンテーション設定
DEFAULT_AUGMENTATION_ENABLED = True
DEFAULT_FLIP_PROB = 0.5
DEFAULT_BRIGHTNESS = 0.5
DEFAULT_CONTRAST = 0.5
DEFAULT_SATURATION = 0.5
DEFAULT_ROTATION_DEGREES = 5
DEFAULT_TRANSLATE_RATIO = 0.1
DEFAULT_ERASE_PROB = 0.2
DEFAULT_ERASE_MIN_RATIO = 0.02
DEFAULT_ERASE_MAX_RATIO = 0.2

# ===========================================
# ファイル関連設定
# ===========================================
# サポートする画像拡張子
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# ファイル名パターン
FILENAME_PATTERN_JETRACER = r'^\d+_\d+_(\d+)_'
FILENAME_PATTERN_NORMAL = r'^(\d+)_'
FILENAME_PATTERN_JETRACER_FULL = r'^\d+_\d+_(\d+)_([A-Za-z0-9]+)_image_array'
FILENAME_PATTERN_NORMAL_FULL = r'^(\d+)_([A-Za-z0-9]+)_image_array'

# ===========================================
# 動画作成関連設定
# ===========================================
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_SKIP_COUNT = 10
VIDEO_BATCH_SIZE = 50

# ===========================================
# 推論関連設定
# ===========================================
INFERENCE_BATCH_SIZE = 50
YOLO_INFERENCE_BATCH_SIZE = 64
LOCATION_INFERENCE_BATCH_SIZE = 64

# ===========================================
# エクスポート関連設定
# ===========================================
TRAIN_VAL_SPLIT_RATIO = 0.8

# ===========================================
# UI要素のサイズ設定
# ===========================================
# スピンボックス
SPINBOX_MIN_RANGE = 0
SPINBOX_MAX_RANGE = 99999
SKIP_COUNT_MIN = 1
SKIP_COUNT_MAX = 1000
SKIP_COUNT_DEFAULT = 10

# ダイアログサイズ
TRAINING_DIALOG_MIN_WIDTH = 550
TRAINING_DIALOG_MIN_HEIGHT = 600
YOLO_TRAINING_DIALOG_MIN_WIDTH = 500
YOLO_TRAINING_DIALOG_MIN_HEIGHT = 600
LOCATION_TRAINING_DIALOG_MIN_WIDTH = 500
EXPORT_DIALOG_MIN_WIDTH = 500
EXPORT_DIALOG_MIN_HEIGHT = 400
YOLO_UNIFIED_EXPORT_DIALOG_MIN_WIDTH = 550
YOLO_UNIFIED_EXPORT_DIALOG_MIN_HEIGHT = 400
VIDEO_CREATION_DIALOG_MIN_WIDTH = 500
VIDEO_CREATION_DIALOG_MIN_HEIGHT = 500
AUGMENTATION_PREVIEW_DIALOG_MIN_WIDTH = 800
AUGMENTATION_PREVIEW_DIALOG_MIN_HEIGHT = 500

# ===========================================
# アプリケーション動作設定
# ===========================================
# スレッド設定
TORCH_NUM_THREADS = 2

# 自動再生設定
AUTO_PLAY_INTERVAL_NORMAL = 20  # ミリ秒
AUTO_PLAY_INTERVAL_INFERENCE = 100  # ミリ秒（推論表示時）

# プログレス表示設定
PROGRESS_UPDATE_INTERVAL = 100  # エントリごと
PROGRESS_MIN_DURATION = 0