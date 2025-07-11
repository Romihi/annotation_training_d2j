# styles.py
"""
アプリケーションのスタイル定義
テーマとスタイルを統合して管理するモジュール
"""
from PyQt5.QtGui import QColor

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
                color: #F1F5F9;
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
                color: #F1F5F9;
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
                color: #F1F5F9;
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
                color: #F1F5F9;
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
                color: #F1F5F9;
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
                color: #F1F5F9;
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
                color: #F1F5F9;
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
