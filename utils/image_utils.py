# utils/image_utils.py
"""画像処理関連のユーティリティ関数"""
import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from typing import Optional, Tuple

def pil_to_qimage(pil_image: Image.Image) -> QImage:
    """
    PIL ImageをQImageに変換
    
    Args:
        pil_image: PIL Image
        
    Returns:
        QImage: 変換されたQImage
    """
    # RGBに変換して確実にフォーマットを統一
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # NumPy配列に変換
    img_array = np.array(pil_image)
    
    # QImageに変換（RGBフォーマット）
    height, width, channels = img_array.shape
    bytes_per_line = channels * width
    
    return QImage(img_array.data, width, height, 
                  bytes_per_line, QImage.Format_RGB888)

def pil_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    """
    PIL ImageをQPixmapに変換
    
    Args:
        pil_image: PIL Image
        
    Returns:
        QPixmap: 変換されたQPixmap
    """
    qimage = pil_to_qimage(pil_image)
    return QPixmap.fromImage(qimage)

def load_image_safely(image_path: str) -> Optional[Image.Image]:
    """
    画像を安全に読み込む
    
    Args:
        image_path: 画像ファイルパス
        
    Returns:
        Optional[Image.Image]: 読み込まれた画像、失敗時はNone
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"画像読み込みエラー: {image_path} - {e}")
        return None

def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    """
    画像のサイズを取得
    
    Args:
        image_path: 画像ファイルパス
        
    Returns:
        Optional[Tuple[int, int]]: (width, height)、失敗時はNone
    """
    img = load_image_safely(image_path)
    if img:
        return img.size
    return None