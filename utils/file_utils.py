# utils/file_utils.py
"""ファイル操作関連のユーティリティ関数"""
import os
import re
from typing import List, Tuple, Optional
import sys
sys.path.append('..')
from config import (
    IMAGE_EXTENSIONS, 
    FILENAME_PATTERN_JETRACER, 
    FILENAME_PATTERN_NORMAL,
    FILENAME_PATTERN_JETRACER_FULL,
    FILENAME_PATTERN_NORMAL_FULL
)

def get_image_files(directory: str) -> List[str]:
    """
    指定ディレクトリから画像ファイルを取得
    
    Args:
        directory: 検索対象ディレクトリ
        
    Returns:
        List[str]: 画像ファイルパスのリスト
    """
    image_files = []
    
    if not os.path.exists(directory):
        return image_files
        
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            image_files.append(os.path.join(directory, file))
            
    return image_files

def extract_index_from_filename(filename: str) -> Optional[int]:
    """
    ファイル名からインデックスを抽出
    
    Args:
        filename: ファイル名
        
    Returns:
        Optional[int]: 抽出されたインデックス、抽出できない場合はNone
    """
    basename = os.path.basename(filename)
    
    # Jetracer形式を優先的にチェック
    jetracer_match = re.match(FILENAME_PATTERN_JETRACER, basename)
    if jetracer_match:
        return int(jetracer_match.group(1))
    
    # 通常形式をチェック
    normal_match = re.match(FILENAME_PATTERN_NORMAL, basename)
    if normal_match:
        return int(normal_match.group(1))
        
    return None

def extract_variant_info(filename: str) -> Tuple[Optional[int], Optional[str]]:
    """
    ファイル名からインデックスとバリアント情報を抽出
    
    Args:
        filename: ファイル名
        
    Returns:
        Tuple[Optional[int], Optional[str]]: (インデックス, バリアント名)
    """
    basename = os.path.basename(filename)
    
    # Jetracer形式フルパターン
    jetracer_match = re.match(FILENAME_PATTERN_JETRACER_FULL, basename)
    if jetracer_match:
        return int(jetracer_match.group(1)), jetracer_match.group(2)
    
    # 通常形式フルパターン
    normal_match = re.match(FILENAME_PATTERN_NORMAL_FULL, basename)
    if normal_match:
        return int(normal_match.group(1)), normal_match.group(2)
        
    return None, None

def ensure_directory_exists(directory: str) -> bool:
    """
    ディレクトリが存在することを保証する
    
    Args:
        directory: ディレクトリパス
        
    Returns:
        bool: 作成成功またはすでに存在する場合True
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"ディレクトリ作成エラー: {directory} - {e}")
        return False