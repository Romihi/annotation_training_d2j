# utils/geometry_utils.py
"""座標変換・幾何学関連のユーティリティ関数"""
from typing import Tuple

def estimate_location_from_angle(angle):
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

def normalize_coordinates(x: int, y: int, width: int, height: int) -> Tuple[float, float]:
    """
    ピクセル座標を正規化座標（-1～1）に変換
    
    Args:
        x, y: ピクセル座標
        width, height: 画像サイズ
        
    Returns:
        Tuple[float, float]: (angle, throttle) の正規化座標
    """
    # X座標を-1（左）から1（右）に変換
    angle = (x / width) * 2 - 1
    
    # Y座標を1（上）から-1（下）に変換
    throttle = -((y / height) * 2 - 1)
    
    return angle, throttle

def denormalize_coordinates(angle: float, throttle: float, 
                          width: int, height: int) -> Tuple[int, int]:
    """
    正規化座標をピクセル座標に変換
    
    Args:
        angle, throttle: 正規化座標（-1～1）
        width, height: 画像サイズ
        
    Returns:
        Tuple[int, int]: (x, y) のピクセル座標
    """
    # angleを0～widthに変換
    x = int((angle + 1) / 2 * width)
    
    # throttleを0～heightに変換
    y = int((1 - throttle) / 2 * height)
    
    # 範囲内に収める
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    
    return x, y

def is_point_in_polygon(x: float, y: float, polygon_points: list) -> bool:
    """
    点がポリゴン内にあるかを判定（Ray casting algorithm）
    
    Args:
        x, y: チェックする点の座標
        polygon_points: ポリゴンの頂点リスト [(x1,y1), (x2,y2), ...]
        
    Returns:
        bool: 点がポリゴン内にある場合True
    """
    if len(polygon_points) < 3:
        return False
    
    n = len(polygon_points)
    inside = False
    
    p1x, p1y = polygon_points[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def calculate_bbox_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """
    バウンディングボックスの中心座標を計算
    
    Args:
        x1, y1, x2, y2: バウンディングボックスの座標
        
    Returns:
        Tuple[float, float]: 中心座標(center_x, center_y)
    """
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_bbox_dimensions(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """
    バウンディングボックスの幅と高さを計算
    
    Args:
        x1, y1, x2, y2: バウンディングボックスの座標
        
    Returns:
        Tuple[float, float]: (幅, 高さ)
    """
    return (abs(x2 - x1), abs(y2 - y1))