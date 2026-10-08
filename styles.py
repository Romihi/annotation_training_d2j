# styles.py
"""
アプリケーションのスタイル定義
テーマとスタイルを統合して管理するモジュール
"""
import os
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

class Styles:
    """アプリケーションのスタイルとテーマを管理するクラス"""
    
    def __init__(self, theme_name="light"):
        # テーマを初期化
        self.theme_name = theme_name

        # 位置情報の色を定義（これを先に定義）
        self.location_colors = [
            QColor(255, 0, 0),      # 赤
            QColor(0, 150, 0),      # 緑
            QColor(0, 0, 255),      # 青
            QColor(255, 165, 0),    # オレンジ
            QColor(128, 0, 128),    # 紫
            QColor(0, 128, 128),    # ティール
            QColor(255, 0, 255),    # マゼンタ
            QColor(128, 128, 0)     # オリーブ
        ]

        self.load_theme(theme_name)
        # テーマカラーに基づいてスタイルを生成
        self.generate_styles()
    
    def load_theme(self, theme_name):
        """指定されたテーマの色とスタイルを読み込む"""
        self.theme_name = theme_name
        
        if theme_name == "dark":
            self.colors = {
                'background': '#1F2937',
                'surface': '#374151',
                'surface_alt': '#4B5563',
                'primary': '#3B82F6',
                'primary_hover': '#2563EB',
                'primary_pressed': '#1D4ED8',
                'primary_disabled': '#93C5FD',
                'secondary': '#8B5CF6',
                'secondary_hover': '#7C3AED',
                'secondary_pressed': '#6D28D9',
                'secondary_disabled': '#C4B5FD',
                'success': '#10B981',
                'success_hover': '#059669',
                'success_pressed': '#047857',
                'success_disabled': '#6EE7B7',
                'warning': '#F59E0B',
                'warning_hover': '#D97706',
                'warning_pressed': '#B45309',
                'warning_disabled': '#FCD34D',
                'error': '#EF4444',
                'error_hover': '#DC2626',
                'error_pressed': '#B91C1C',
                'error_disabled': '#FCA5A5',
                'text': '#F9FAFB',
                'text_secondary': '#D1D5DB',
                'text_disabled': '#9CA3AF',
                'border': '#4B5563',
                'border_hover': '#6B7280',
                'nav': '#6B7280',
                'nav_hover': '#4B5563',
                'nav_pressed': '#374151',
                'nav_disabled': '#9CA3AF',
                # 特殊アクションボタン用の色を追加 - ティール/ターコイズ系
                'special': '#0EA5E9',  # 明るいティール 
                'special_hover': '#0284C7',  # やや濃いティール
                'special_pressed': '#0369A1',  # 濃いティール
                'special_disabled': '#7DD3FC',  # 薄いティール
            }
        else:  # light theme (default)
            self.colors = {
                'background': '#F9FAFB',
                'surface': '#FFFFFF',
                'surface_alt': '#F3F4F6',
                'primary': '#2563EB',
                'primary_hover': '#1D4ED8',
                'primary_pressed': '#1E40AF',
                'primary_disabled': '#93C5FD',
                'secondary': '#7C3AED',
                'secondary_hover': '#6D28D9',
                'secondary_pressed': '#5B21B6',
                'secondary_disabled': '#C4B5FD',
                'success': '#059669',
                'success_hover': '#047857',
                'success_pressed': '#065F46',
                'success_disabled': '#6EE7B7',
                'warning': '#D97706',
                'warning_hover': '#B45309',
                'warning_pressed': '#92400E',
                'warning_disabled': '#FCD34D',
                'error': '#E11D48',
                'error_hover': '#BE123C',
                'error_pressed': '#9F1239',
                'error_disabled': '#FDA4AF',
                'text': '#111827',
                'text_secondary': '#4B5563',
                'text_disabled': '#9CA3AF',
                'border': '#E5E7EB',
                'border_hover': '#D1D5DB',
                'nav': '#6B7280',
                'nav_hover': '#4B5563',
                'nav_pressed': '#374151',
                'nav_disabled': '#D1D5DB',
                # 特殊アクションボタン用の色を追加 - ティール/ターコイズ系
                'special': '#0EA5E9',  # 明るいティール 
                'special_hover': '#0284C7',  # やや濃いティール
                'special_pressed': '#0369A1',  # 濃いティール
                'special_disabled': '#7DD3FC',  # 薄いティール
            }
    
    def generate_styles(self):
        """テーマカラーを使用して各種スタイルを生成"""
        
        # ======= ボタンスタイル =======
        
        # 主要アクションボタン（読み込み、保存など）
        self.PRIMARY_STYLE = f"""
            QPushButton {{
                background-color: {self.colors['primary']};
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary_hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['primary_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['primary_disabled']};
                color: #475569;  /* 淡い無効時背景でも読めるよう暗めの文字にする */
            }}
        """

        # モデル操作ボタン（モデル読み込み、リスト更新）
        self.MODEL_STYLE = f"""
            QPushButton {{
                background-color: {self.colors['secondary']};
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self.colors['secondary_hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['secondary_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['secondary_disabled']};
                color: #475569;  /* 淡い無効時背景でも読めるよう暗めの文字にする */
            }}
        """

        # 学習・トレーニングボタン
        self.TRAINING_STYLE = f"""
            QPushButton {{
                background-color: {self.colors['success']};
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self.colors['success_hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['success_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['success_disabled']};
                color: #475569;  /* 淡い無効時背景でも読めるよう暗めの文字にする */
            }}
        """

        # エクスポートボタン
        self.EXPORT_STYLE = f"""
            QPushButton {{
                background-color: {self.colors['warning']};
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self.colors['warning_hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['warning_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['warning_disabled']};
                color: #475569;  /* 淡い無効時背景でも読めるよう暗めの文字にする */
            }}
        """

        # 特殊アクションボタン（MLflow比較、一括処理など）
        self.SPECIAL_STYLE = f"""
            QPushButton {{
                background-color: {self.colors['special']};
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self.colors['special_hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['special_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['special_disabled']};
                color: #475569;  /* 淡い無効時背景でも読めるよう暗めの文字にする */
            }}
        """
        # 削除などの破壊的アクションボタン
        self.DESTRUCTIVE_STYLE = f"""
            QPushButton {{
                background-color: {self.colors['error']};
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self.colors['error_hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['error_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['error_disabled']};
                color: #475569;  /* 淡い無効時背景でも読めるよう暗めの文字にする */
            }}
        """

        # ナビゲーションボタン
        self.NAV_STYLE = f"""
            QPushButton {{
                background-color: {self.colors['nav']};
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self.colors['nav_hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['nav_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['nav_disabled']};
                color: #475569;  /* 淡い無効時背景でも読めるよう暗めの文字にする */
            }}
        """

        # ======= コンテナ・パネルスタイル =======
        
        # 左側パネル
        self.LEFT_PANEL_STYLE = f"""
            QWidget {{
                background-color: {self.colors['background']};
                border-right: 1px solid {self.colors['border']};
            }}
        """

        # 情報パネル
        self.INFO_PANEL_STYLE = f"""
            QWidget#info_panel {{
                background-color: {self.colors['surface_alt']};
                border-radius: 8px;
                padding: 10px;
            }}
        """

        # ギャラリーコンテナ
        self.GALLERY_CONTAINER_STYLE = f"""
            QWidget {{
                background-color: {self.colors['surface_alt']};
                border-top: 1px solid {self.colors['border']};
            }}
        """

        # グループボックス
        self.GROUP_BOX_STYLE = f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {self.colors['border']};
                border-radius: 6px;
                margin-top: 1ex;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: {self.colors['background']};
                color: {self.colors['text']};
            }}
        """

        # ======= 入力コントロールスタイル =======
        
        # テキスト入力フィールド
        self.TEXT_INPUT_STYLE = f"""
            QLineEdit {{
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 5px;
                background-color: {self.colors['surface']};
                color: {self.colors['text']};
            }}
            QLineEdit:focus {{
                border: 1px solid {self.colors['primary']};
            }}
            QLineEdit:disabled {{
                background-color: {self.colors['surface_alt']};
                color: {self.colors['text_disabled']};
            }}
        """

        # スピンボックス
        self.SPIN_BOX_STYLE = f"""
            QSpinBox, QDoubleSpinBox {{
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 5px;
                background-color: {self.colors['surface']};
                color: {self.colors['text']};
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {self.colors['primary']};
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                width: 16px;
                border-left: 1px solid {self.colors['border']};
                border-bottom: 1px solid {self.colors['border']};
                border-top-right-radius: 3px;
                subcontrol-origin: border;
                subcontrol-position: top right;
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                width: 16px;
                border-left: 1px solid {self.colors['border']};
                border-top-right-radius: 0px;
                border-bottom-right-radius: 3px;
                subcontrol-origin: border;
                subcontrol-position: bottom right;
            }}
        """

        # コンボボックス（ドロップダウン）
        self.COMBO_BOX_STYLE = f"""
            QComboBox {{
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 5px;
                background-color: {self.colors['surface']};
                color: {self.colors['text']};
            }}
            QComboBox:focus {{
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox::drop-down {{
                width: 20px;
                border-left: 1px solid {self.colors['border']};
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }}
            QComboBox::down-arrow {{
                width: 10px;
                height: 10px;
            }}
        """

        # チェックボックス
        self.CHECKBOX_STYLE = f"""
            QCheckBox {{
                spacing: 5px;
                color: {self.colors['text']};
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
            }}
            QCheckBox::indicator:unchecked {{
                border: 1px solid {self.colors['border']};
                background-color: {self.colors['surface']};
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                border: 1px solid {self.colors['primary']};
                background-color: {self.colors['primary']};
                border-radius: 3px;
            }}
        """

        # ラジオボタン
        self.RADIO_BUTTON_STYLE = f"""
            QRadioButton {{
                spacing: 5px;
                color: {self.colors['text']};
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
            }}
            QRadioButton::indicator:unchecked {{
                border: 1px solid {self.colors['border']};
                background-color: {self.colors['surface']};
                border-radius: 9px;
            }}
            QRadioButton::indicator:checked {{
                border: 1px solid {self.colors['primary']};
                background-color: {self.colors['primary']};
                border-radius: 9px;
            }}
        """

        # ======= スライダーとプログレスバー =======
        
        # スライダー
        self.SLIDER_STYLE = f"""
            QSlider::groove:horizontal {{
                border: 1px solid {self.colors['border']};
                height: 8px;
                background: {self.colors['surface_alt']};
                margin: 2px 0;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {self.colors['primary']};
                border: 1px solid {self.colors['primary']};
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {self.colors['primary_hover']};
            }}
        """

        # プログレスバー
        self.PROGRESS_BAR_STYLE = f"""
            QProgressBar {{
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                text-align: center;
                background-color: {self.colors['surface_alt']};
                height: 20px;
                color: {self.colors['text']};
            }}
            QProgressBar::chunk {{
                background-color: {self.colors['primary']};
                width: 1px;
            }}
        """

        # ======= ラベルとテキスト表示 =======
        
        # タイトルラベル
        self.TITLE_LABEL_STYLE = f"""
            QLabel {{
                font-size: 16px;
                font-weight: bold;
                color: {self.colors['text']};
            }}
        """

        # サブタイトルラベル
        self.SUBTITLE_LABEL_STYLE = f"""
            QLabel {{
                font-size: 14px;
                font-weight: bold;
                color: {self.colors['text']};
            }}
        """

        # 通常のラベル
        self.NORMAL_LABEL_STYLE = f"""
            QLabel {{
                font-size: 12px;
                color: {self.colors['text_secondary']};
            }}
        """

        # 強調ラベル
        self.EMPHASIS_LABEL_STYLE = f"""
            QLabel {{
                font-size: 12px;
                font-weight: bold;
                color: {self.colors['text']};
            }}
        """

        # ヒントテキスト
        self.HINT_LABEL_STYLE = f"""
            QLabel {{
                font-size: 11px;
                font-style: italic;
                color: {self.colors['text_secondary']};
            }}
        """

        # エラーラベル
        self.ERROR_LABEL_STYLE = f"""
            QLabel {{
                font-size: 12px;
                color: {self.colors['error']};
            }}
        """

        # ======= ダイアログとポップアップ =======
        
        # 標準ダイアログ
        self.STANDARD_DIALOG_STYLE = f"""
            QDialog {{
                background-color: {self.colors['surface']};
                border: 1px solid {self.colors['border']};
                border-radius: 8px;
            }}
        """

        # タイトルバー
        self.DIALOG_TITLE_STYLE = f"""
            QLabel#dialog_title {{
                font-size: 16px;
                font-weight: bold;
                color: {self.colors['text']};
                padding-bottom: 10px;
                border-bottom: 1px solid {self.colors['border']};
            }}
        """

        # ダイアログボタン（OK/キャンセル）
        self.DIALOG_BUTTON_STYLE = f"""
            QPushButton {{
                min-width: 80px;
                padding: 6px 12px;
                border-radius: 4px;
            }}
            QPushButton#okButton {{
                background-color: {self.colors['primary']};
                color: white;
                border: none;
            }}
            QPushButton#okButton:hover {{
                background-color: {self.colors['primary_hover']};
            }}
            QPushButton#cancelButton {{
                background-color: {self.colors['surface_alt']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['border']};
            }}
            QPushButton#cancelButton:hover {{
                background-color: {self.colors['surface']};
            }}
        """

        # ======= メインウィンドウ要素 =======
        
        # メインウィンドウ
        self.MAIN_WINDOW_STYLE = f"""
            QMainWindow {{
                background-color: {self.colors['background']};
            }}
        """

        # ステータスバー
        self.STATUS_BAR_STYLE = f"""
            QStatusBar {{
                background-color: {self.colors['surface_alt']};
                border-top: 1px solid {self.colors['border']};
                color: {self.colors['text_secondary']};
            }}
            QStatusBar::item {{
                border: none;
            }}
        """

        # ツールバー
        self.TOOLBAR_STYLE = f"""
            QToolBar {{
                background-color: {self.colors['background']};
                border-bottom: 1px solid {self.colors['border']};
                spacing: 6px;
            }}
            QToolBar::separator {{
                width: 1px;
                background-color: {self.colors['border']};
                margin: 6px 4px;
            }}
        """

        # スクロールエリア
        self.SCROLL_AREA_STYLE = f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                border: none;
                background-color: {self.colors['surface_alt']};
                width: 12px;
                margin: 12px 0 12px 0;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {self.colors['nav']};
                min-height: 20px;
                border-radius: 6px;
            }}
            QScrollBar::add-line:vertical {{
                height: 12px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }}
            QScrollBar::sub-line:vertical {{
                height: 12px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }}
        """

        # ======= サムネイル関連 =======
        
        # サムネイルコンテナ
        self.THUMBNAIL_CONTAINER_STYLE = f"""
            QWidget {{
                background-color: {self.colors['surface']};
                border: 1px solid {self.colors['border']};
                border-radius: 6px;
            }}
        """

        # 選択中サムネイル
        self.THUMBNAIL_SELECTED_STYLE = f"""
            QFrame {{
                border: 2px solid {self.colors['primary']};
                border-radius: 6px;
            }}
        """

        # 削除済みサムネイル
        self.THUMBNAIL_DELETED_STYLE = f"""
            QFrame {{
                border: 2px solid {self.colors['error']};
                border-radius: 6px;
                opacity: 0.7;
            }}
        """
    
    def apply_style(self, widget, style_type):
        """指定されたスタイルをウィジェットに適用する関数"""
        style_map = {
            # ボタンスタイル
            'primary': self.PRIMARY_STYLE,
            'model': self.MODEL_STYLE,
            'training': self.TRAINING_STYLE,
            'export': self.EXPORT_STYLE,
            'special': self.SPECIAL_STYLE,
            'destructive': self.DESTRUCTIVE_STYLE,
            'nav': self.NAV_STYLE,
            
            # コンテナスタイル
            'left_panel': self.LEFT_PANEL_STYLE,
            'info_panel': self.INFO_PANEL_STYLE,
            'gallery': self.GALLERY_CONTAINER_STYLE,
            'group_box': self.GROUP_BOX_STYLE,
            
            # 入力コントロール
            'text_input': self.TEXT_INPUT_STYLE,
            'spin_box': self.SPIN_BOX_STYLE,
            'combo_box': self.COMBO_BOX_STYLE,
            'checkbox': self.CHECKBOX_STYLE,
            'radio': self.RADIO_BUTTON_STYLE,
            
            # スライダーとプログレスバー
            'slider': self.SLIDER_STYLE,
            'progress': self.PROGRESS_BAR_STYLE,
            
            # ラベル
            'title': self.TITLE_LABEL_STYLE,
            'subtitle': self.SUBTITLE_LABEL_STYLE,
            'label': self.NORMAL_LABEL_STYLE,
            'emphasis': self.EMPHASIS_LABEL_STYLE,
            'hint': self.HINT_LABEL_STYLE,
            'error': self.ERROR_LABEL_STYLE,
            
            # ダイアログ
            'dialog': self.STANDARD_DIALOG_STYLE,
            'dialog_title': self.DIALOG_TITLE_STYLE,
            'dialog_button': self.DIALOG_BUTTON_STYLE,
            
            # ウィンドウ要素
            'main_window': self.MAIN_WINDOW_STYLE,
            'status_bar': self.STATUS_BAR_STYLE,
            'toolbar': self.TOOLBAR_STYLE,
            'scroll': self.SCROLL_AREA_STYLE,
            
            # サムネイル
            'thumbnail': self.THUMBNAIL_CONTAINER_STYLE,
            'thumbnail_selected': self.THUMBNAIL_SELECTED_STYLE,
            'thumbnail_deleted': self.THUMBNAIL_DELETED_STYLE,
        }
        
        if style_type in style_map:
            widget.setStyleSheet(style_map[style_type])
        else:
            print(f"警告: 未定義のスタイルタイプ '{style_type}' が指定されました")
    
    def set_theme(self, theme_name):
        """テーマを切り替える"""
        self.load_theme(theme_name)
        self.generate_styles()
        return self.theme_name
    
    def get_current_theme(self):
        """現在のテーマ名を取得"""
        return self.theme_name
    
    def get_color(self, color_name):
        """指定された名前のテーマカラーを取得"""
        if color_name in self.colors:
            return self.colors[color_name]
        else:
            print(f"警告: 未定義のカラー名 '{color_name}' が指定されました")
            return "#000000"  # デフォルト色（黒）

    def get_location_color(self, location_value):
            """位置情報の値から色を取得する
            
            Args:
                location_value: 位置情報の値（整数）または None
                
            Returns:
                QColor: 位置情報に対応する色
            """
            # 位置情報の値に基づいて色を選択（8で割った余りを使用）
            if location_value is None:
                return QColor(200, 200, 200)  # グレー（位置情報なし）
            
            # 色インデックスを取得（0〜7の範囲）
            color_index = location_value % 8
            return self.location_colors[color_index]
    
    def get_location_color_hex(self, location_value):
        """位置情報の値から16進数カラーコードを取得する
        
        Args:
            location_value: 位置情報の値（整数）または None
            
        Returns:
            str: 16進数カラーコード
        """
        color = self.get_location_color(location_value)
        return color.name()



# グローバルなスタイルインスタンスを作成（シングルトン）
app_styles = Styles()

# 直接インポート可能なスタイル変数（便宜上）
PRIMARY_STYLE = app_styles.PRIMARY_STYLE
MODEL_STYLE = app_styles.MODEL_STYLE
TRAINING_STYLE = app_styles.TRAINING_STYLE
EXPORT_STYLE = app_styles.EXPORT_STYLE
SPECIAL_STYLE = app_styles.SPECIAL_STYLE
DESTRUCTIVE_STYLE = app_styles.DESTRUCTIVE_STYLE
NAV_STYLE = app_styles.NAV_STYLE

# スタイル適用関数
def apply_style(widget, style_type):
    """指定されたスタイルをウィジェットに適用するグローバル関数"""
    app_styles.apply_style(widget, style_type)

# テーマ切り替え関数
def set_theme(theme_name):
    """テーマを切り替えるグローバル関数"""
    return app_styles.set_theme(theme_name)

def get_current_theme():
    """現在のテーマ名を取得するグローバル関数"""
    return app_styles.get_current_theme()

def get_color(color_name):
    """指定された名前のテーマカラーを取得するグローバル関数"""
    return app_styles.get_color(color_name)

def get_location_color(location_value):
    """位置情報の値から色を取得するグローバル関数"""
    return app_styles.get_location_color(location_value)

def get_location_color_hex(location_value):
    """位置情報の値から16進数カラーコードを取得するグローバル関数"""
    return app_styles.get_location_color_hex(location_value)


# =====================================================================
# ダークモード（アプリ全体テーマ）
# =====================================================================
# ダークモードでは QSS でウィジェット全体の文字色を白にするため、個々の
# ウィジェットに "color: #333" のような固定色を書くと背景と同化して読めなく
# なる。固定色の代わりに「役割」だけをプロパティで付け（set_text_role など）、
# 実際の色はモードごとのアプリ全体QSSで決める。切替時は QSS を貼り直すだけで
# 既存ウィジェットにも反映される。

_is_dark_mode = False
_light_palette = None  # 初回適用時にデフォルトパレットを退避する

# 文字色の役割: 役割名 -> (ライト, ダーク)
TEXT_ROLE_COLORS = {
    'strong':  ('#333333', '#E6E8EB'),   # 本文より少し強調（旧 #333/#444）
    'muted':   ('#666666', '#A8B0B8'),   # 補足説明（旧 #666/#777/#555）
    'faint':   ('#888888', '#8E979F'),   # さらに薄い注記（旧 #888/gray/#aaa）
    'success': ('#2E7D32', '#81C784'),   # 成功・件数などの緑
    'error':   ('#D32F2F', '#FF6B6B'),   # エラー・削除済みの赤
    'warning': ('#E65100', '#FFB74D'),   # 注意のオレンジ
    'info':    ('#1565C0', '#64B5F6'),   # 情報の青
    'accent':  ('#6A1B9A', '#CE93D8'),   # 紫（TogiVAD・セグメンテーション）
    'teal':    ('#009999', '#4DD0E1'),   # 推論値の青緑
}

# 面（背景）の役割: 役割名 -> ((ライト背景, ライト枠), (ダーク背景, ダーク枠))
PANEL_ROLE_COLORS = {
    'box':    (('#f8f8f8', '#dddddd'), ('#383838', '#555555')),  # 枠付きの薄い箱
    'strip':  (('#f8f8f8', None),      ('#383838', None)),       # 枠なしの帯
    'code':   (('#f0f0f0', '#cccccc'), ('#3a3a3a', '#555555')),  # 固定文字列の表示欄
    'footer': (('#e8eaed', None),      ('#333333', None)),       # ダイアログ下部の固定エリア
}

_DARK_BASE_QSS = """
QMainWindow, QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
}
QPushButton {
    background-color: #404040;
    color: #ffffff;
}
QPushButton:hover { background-color: #505050; }
QPushButton:pressed { background-color: #606060; }
QPushButton:checked { background-color: #0078d4; }
QPushButton:disabled { color: #8E979F; }
QToolButton { background-color: transparent; color: #ffffff; }
QToolButton:hover { background-color: #505050; }
QLabel {
    background-color: transparent;
    color: #ffffff;
}
QLineEdit, QTextEdit, QPlainTextEdit, QTextBrowser,
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #404040;
    color: #ffffff;
    selection-background-color: #0078d4;
    selection-color: #ffffff;
}
QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    color: #8E979F;
}
QComboBox QAbstractItemView {
    background-color: #404040;
    color: #ffffff;
    selection-background-color: #0078d4;
    selection-color: #ffffff;
}
QListView, QTreeView, QTableView {
    background-color: #353535;
    alternate-background-color: #3d3d3d;
    color: #ffffff;
    selection-background-color: #0078d4;
    selection-color: #ffffff;
    gridline-color: #555555;
}
QHeaderView::section {
    background-color: #404040;
    color: #ffffff;
    border: 1px solid #2b2b2b;
}
QTableCornerButton::section { background-color: #404040; }
QScrollArea { background-color: #2b2b2b; }
QGroupBox { color: #ffffff; }
QGroupBox::title { color: #ffffff; }
QCheckBox, QRadioButton { color: #ffffff; background-color: transparent; }
QCheckBox:disabled, QRadioButton:disabled { color: #8E979F; }
/* 標準スタイルの枠はダークパレットだと背景に溶けるので、枠と塗りを明示する */
QCheckBox::indicator, QRadioButton::indicator {
    width: 13px;
    height: 13px;
    border: 1px solid #9AA3AC;
    background-color: #404040;
}
QCheckBox::indicator { border-radius: 2px; }
QRadioButton::indicator { border-radius: 7px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #64B5F6; }
QCheckBox::indicator:checked {
    border-color: #0078d4;
    background-color: #0078d4;
    image: url(%(check_icon)s);
}
QRadioButton::indicator:checked {
    border: 4px solid #0078d4;
    background-color: #ffffff;
}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    border-color: #5a5a5a;
    background-color: #333333;
}
QCheckBox::indicator:checked:disabled { background-color: #4a5a6a; border-color: #4a5a6a; }
QRadioButton::indicator:checked:disabled { border-color: #4a5a6a; background-color: #8E979F; }
QDialog {
    background-color: #2b2b2b;
    color: #ffffff;
}
QTabWidget::pane { background-color: #2b2b2b; }
QTabBar::tab {
    background-color: #404040;
    color: #ffffff;
}
QTabBar::tab:selected { background-color: #0078d4; }
QMenuBar { background-color: #2b2b2b; color: #ffffff; }
QMenuBar::item:selected { background-color: #0078d4; }
QMenu { background-color: #353535; color: #ffffff; }
QMenu::item:selected { background-color: #0078d4; }
QMenu::item:disabled { color: #8E979F; }
QToolTip {
    background-color: #404040;
    color: #ffffff;
    border: 1px solid #555555;
}
QStatusBar { color: #ffffff; }
QProgressBar { color: #ffffff; }
QProgressBar::chunk { background-color: #0078d4; }
"""

# 排他選択（セグメンテッドコントロール）ボタン。set_segmented() で付ける。
_SEGMENTED_QSS_LIGHT = (
    'QPushButton[segmented="true"]{border:1px solid #aaa;border-radius:3px;padding:4px 8px;background:#f0f0f0;}'
    'QPushButton[segmented="true"]:hover{background:#e6e6e6;}'
    'QPushButton[segmented="true"]:checked{background:#4a90d9;color:white;border-color:#2a70b9;font-weight:bold;}'
)
_SEGMENTED_QSS_DARK = (
    'QPushButton[segmented="true"]{border:1px solid #666;border-radius:3px;padding:4px 8px;background:#404040;color:#ffffff;}'
    'QPushButton[segmented="true"]:hover{background:#505050;}'
    'QPushButton[segmented="true"]:checked{background:#4a90d9;color:white;border-color:#6ab0f9;font-weight:bold;}'
)


def _role_qss(is_dark):
    """役割プロパティ -> 色 の QSS を組み立てる（ライト/ダーク共通の仕組み）"""
    idx = 1 if is_dark else 0
    rules = []
    for role, colors in TEXT_ROLE_COLORS.items():
        rules.append(f'*[textRole="{role}"] {{ color: {colors[idx]}; }}')
    for role, variants in PANEL_ROLE_COLORS.items():
        bg, border = variants[idx]
        border_qss = f' border: 1px solid {border}; border-radius: 4px;' if border else ''
        rules.append(f'*[panelRole="{role}"] {{ background-color: {bg};{border_qss} }}')
    rules.append(_SEGMENTED_QSS_DARK if is_dark else _SEGMENTED_QSS_LIGHT)
    return "\n".join(rules)


def _dark_base_qss():
    check_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'assets', 'check_white.png').replace(os.sep, '/')
    return _DARK_BASE_QSS % {'check_icon': check_icon}


def build_app_qss(is_dark):
    """アプリ全体に貼る QSS を返す。ライトはネイティブ見た目を保ち役割色のみ定義する"""
    base = _dark_base_qss() if is_dark else ""
    return base + "\n" + _role_qss(is_dark)


def _build_dark_palette():
    """QSS の効かない部分（palette() 参照、ネイティブ描画）用のダークパレット"""
    white = QColor('#ffffff')
    p = QPalette()
    p.setColor(QPalette.Window, QColor('#2b2b2b'))
    p.setColor(QPalette.WindowText, white)
    p.setColor(QPalette.Base, QColor('#404040'))
    p.setColor(QPalette.AlternateBase, QColor('#353535'))
    p.setColor(QPalette.ToolTipBase, QColor('#404040'))
    p.setColor(QPalette.ToolTipText, white)
    p.setColor(QPalette.Text, white)
    p.setColor(QPalette.Button, QColor('#404040'))
    p.setColor(QPalette.ButtonText, white)
    p.setColor(QPalette.BrightText, white)
    p.setColor(QPalette.Highlight, QColor('#0078d4'))
    p.setColor(QPalette.HighlightedText, white)
    p.setColor(QPalette.Link, QColor('#64B5F6'))
    p.setColor(QPalette.Light, QColor('#5a5a5a'))
    p.setColor(QPalette.Midlight, QColor('#4a4a4a'))
    p.setColor(QPalette.Mid, QColor('#555555'))
    p.setColor(QPalette.Dark, QColor('#1e1e1e'))
    p.setColor(QPalette.Shadow, QColor('#000000'))
    if hasattr(QPalette, 'PlaceholderText'):
        p.setColor(QPalette.PlaceholderText, QColor('#8E979F'))
    disabled = QColor('#8E979F')
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, disabled)
    return p


def apply_app_theme(is_dark):
    """ダーク/ライトをアプリ全体（全ウィンドウ・ダイアログ）に適用する"""
    global _is_dark_mode, _light_palette
    _is_dark_mode = bool(is_dark)
    set_theme('dark' if _is_dark_mode else 'light')
    app = QApplication.instance()
    if app is None:
        return
    if _light_palette is None:
        _light_palette = QPalette(app.palette())
    app.setPalette(_build_dark_palette() if _is_dark_mode else _light_palette)
    app.setStyleSheet(build_app_qss(_is_dark_mode))


def is_dark_mode():
    """現在ダークモードかどうか"""
    return _is_dark_mode


def theme_color(role):
    """役割名から現在のモードの文字色（16進文字列）を返す。リッチテキスト用"""
    colors = TEXT_ROLE_COLORS.get(role)
    if colors is None:
        print(f"警告: 未定義の文字色役割 '{role}' が指定されました")
        return '#ffffff' if _is_dark_mode else '#000000'
    return colors[1 if _is_dark_mode else 0]


def set_style_role(widget, prop, value):
    """QSS 用の動的プロパティを設定し、作成済みウィジェットにも即反映させる"""
    widget.setProperty(prop, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_text_role(widget, role):
    """文字色の役割を付ける（None で解除）。色は setStyleSheet に書かないこと"""
    if role is not None and role not in TEXT_ROLE_COLORS:
        print(f"警告: 未定義の文字色役割 '{role}' が指定されました")
    set_style_role(widget, 'textRole', role)


def set_panel_role(widget, role):
    """面（背景・枠）の役割を付ける（None で解除）"""
    if role is not None and role not in PANEL_ROLE_COLORS:
        print(f"警告: 未定義の面役割 '{role}' が指定されました")
    set_style_role(widget, 'panelRole', role)


def set_segmented(button):
    """排他選択ボタンの見た目を付ける"""
    set_style_role(button, 'segmented', True)


def location_button_qss(color=None):
    """位置ボタンの QSS。color(QColor) を渡すとその位置の色で、None ならグレー表示"""
    if color is not None:
        checked = f"""
            QPushButton:checked {{
                background-color: {color.name()};
                color: white;
                font-weight: bold;
            }}"""
        if _is_dark_mode:
            normal_bg, normal_fg = color.darker(250).name(), '#ffffff'
        else:
            normal_bg, normal_fg = color.lighter(140).name(), 'black'
        border = color.name()
    else:
        checked = """
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }"""
        if _is_dark_mode:
            normal_bg, normal_fg, border = '#3a3a3a', '#A8B0B8', '#555555'
        else:
            normal_bg, normal_fg, border = '#f0f0f0', '#888888', '#cccccc'
    return f"""
        QPushButton {{
            padding: 8px;
            border: 1px solid {border};
            border-radius: 4px;
            background-color: {normal_bg};
            color: {normal_fg};
        }}{checked}
    """


def location_text_color(location_value):
    """位置情報の色を文字用に返す。ダークでは暗い色（青・紫）が沈むので明るくする"""
    color = get_location_color(location_value)
    if _is_dark_mode:
        h, s, v, a = color.getHsv()
        color = QColor.fromHsv(h, min(s, 170), max(v, 230), a)
    return color


def tint_group_qss(bg_color, border_color):
    """設定パネル（QGroupBox）を淡く塗り分ける QSS。ダークでは同系色の暗い面にする"""
    if _is_dark_mode:
        bg_color = QColor(bg_color).darker(400).name()
        border_color = QColor(border_color).darker(250).name()
    return (
        "QGroupBox { background-color: %s; border: 1px solid %s;"
        " border-radius: 6px; margin-top: 8px; padding-top: 6px; font-weight: bold; }"
        " QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
        % (bg_color, border_color))
