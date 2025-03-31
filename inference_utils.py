"""
推論ユーティリティ - モデルなどを使用して推論を行う関数
"""

import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
import torchvision.transforms as transforms

from models import get_model, list_available_models

_MODEL_CACHE = {}

def batch_inference(
    image_paths: List[str], 
    method: str = "model", 
    model_type: Optional[str] = None,
    model_path: Optional[str] = None,
    force_reload: bool = False  # 必要な場合にのみTrueに設定
) -> Dict[str, Dict[str, Any]]:
    """画像バッチに対して推論を実行する"""
    results = {}
    
    if method == "model" and model_type:
        # モデルを使用した推論
        results = _infer_with_model(image_paths, model_type, model_path, force_reload)
    else:
        raise ValueError(f"サポートされていない推論方法: {method}")
    
    return results

def _infer_with_model(
    image_paths: List[str], 
    model_type: str, 
    model_path: Optional[str] = None,
    force_reload: bool = False
) -> Dict[str, Dict[str, Any]]:
    """モデルを使用して推論する"""
    global _MODEL_CACHE
    results = {}

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # キャッシュキーを作成
        cache_key = (model_type, model_path)
        
        # キャッシュからモデルを取得するか、新しくロードする
        if not force_reload and cache_key in _MODEL_CACHE:
            #print(f"キャッシュからモデルを使用: {model_type}")
            model = _MODEL_CACHE[cache_key]
        else:
            print(f"新しくモデルをロード: {model_type}, パス: {model_path}")
            # モデルの初期化
            model = get_model(model_type, pretrained=False)
            
            # モデルパスが指定されていない場合は、最新のモデルファイルを探す
            if not model_path:
                # ... (モデル検索ロジックは変更なし) ...
                pass
                
            # 保存済みモデルをロード
            if model_path and os.path.exists(model_path):
                try:
                    # モデルの状態を読み込む
                    checkpoint = torch.load(model_path, map_location=device)
                    
                    # モデル状態の辞書が直接保存されている場合
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # モデルの状態が直接保存されている古い形式の場合
                        model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"モデル読み込みエラー: {e}")
                    # 読み込みに失敗した場合は事前学習済みモデルを使用
                    model = get_model(model_type, pretrained=True)
            else:
                # モデルパスが指定されていないか存在しない場合は事前学習済みを使用
                model = get_model(model_type, pretrained=True)
            
            model = model.to(device)
            model.eval()  # 評価モードに設定
            
            # キャッシュに保存
            _MODEL_CACHE[cache_key] = model
        
        # モデルの前処理を取得
        transform = model.get_preprocess()
        
        # バッチ処理
        with torch.no_grad():
            for img_path in image_paths:
                try:
                    # 画像を読み込む
                    img = Image.open(img_path).convert('RGB')
                    img_width, img_height = img.size
                    
                    # 前処理
                    img_tensor = transform(img)
                    img_tensor = img_tensor.unsqueeze(0).to(device)
                    
                    # 推論
                    output = model(img_tensor)
                    angle, throttle = output[0].cpu().numpy()
                    
                    # 座標に変換
                    x = int((angle + 1) / 2 * img_width)
                    y = int((1 - throttle) / 2 * img_height)
                    
                    # 範囲内に収める
                    x = max(0, min(x, img_width - 1))
                    y = max(0, min(y, img_height - 1))
                    
                    # 結果を保存
                    results[img_path] = {
                        "angle": float(angle),
                        "throttle": float(throttle),
                        "x": x,
                        "y": y
                    }
                    
                except Exception as e:
                    print(f"画像 {img_path} の推論中にエラー: {e}")
                    
    except Exception as e:
        print(f"モデル推論エラー: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    return results

def clear_model_cache():
    """モデルキャッシュをクリアする - メモリ解放が必要な場合に使用"""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    import gc
    gc.collect()
    torch.cuda.empty_cache()  # GPUメモリも解放
