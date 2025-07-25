###未移管
"""色関連のユーティリティ関数"""
from PyQt5.QtGui import QColor
from typing import Tuple
import sys
sys.path.append('..')
from config import LOCATION_COLORS, CLASS_COLORS, SEGMENTATION_CLASS_COLORS

def get_location_color(location_value: int) -> QColor:
    """
    位置情報に対応する色を返す
    
    Args:
        location_value: 位置情報の値（0-7）
        
    Returns:
        QColor: 対応する色
    """
    colors = [QColor(*rgb) for rgb in LOCATION_COLORS]
    return colors[location_value % len(colors)]

def get_class_color(class_name: str, alpha: int = 180) -> QColor:
    """
    クラス名に対応する色を返す
    
    Args:
        class_name: オブジェクトクラス名
        alpha: 透明度（0-255）
        
    Returns:
        QColor: 対応する色
    """
    color_tuple = CLASS_COLORS.get(class_name, CLASS_COLORS['unknown'])
    return QColor(*color_tuple[:3], alpha)

def get_segmentation_color(class_name: str) -> QColor:
    """
    セグメンテーション用のクラス色を返す
    
    Args:
        class_name: オブジェクトクラス名
        
    Returns:
        QColor: 対応する色（透明度含む）
    """
    color_tuple = SEGMENTATION_CLASS_COLORS.get(
        class_name, 
        SEGMENTATION_CLASS_COLORS['unknown']
    )
    return QColor(*color_tuple)