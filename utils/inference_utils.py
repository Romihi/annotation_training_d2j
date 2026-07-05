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

from model_catalog import get_model, list_available_models

_MODEL_CACHE = {}

def batch_inference(
    image_paths: List[str],
    method: str = "model",
    model_type: Optional[str] = None,
    model_path: Optional[str] = None,
    force_reload: bool = False,
    downscale_factor: float = 1.0
) -> Dict[str, Dict[str, Any]]:
    """画像バッチに対して推論を実行する"""
    results = {}

    if method == "model" and model_type:
        # モデルを使用した推論
        results = _infer_with_model(image_paths, model_type, model_path, force_reload, downscale_factor)
    else:
        raise ValueError(f"サポートされていない推論方法: {method}")
    
    return results

def _infer_with_model(
    image_paths: List[str],
    model_type: str,
    model_path: Optional[str] = None,
    force_reload: bool = False,
    downscale_factor: float = 1.0
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

            # チェックポイントから出力数と入力サイズを検出
            num_outputs = 2  # デフォルトは2出力
            input_size = None
            if model_path and os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    state_dict = checkpoint.get('model_state_dict', checkpoint)

                    # regressor.biasまたはregressor.weightから出力数を検出
                    if 'regressor.bias' in state_dict:
                        num_outputs = state_dict['regressor.bias'].shape[0]
                        print(f"チェックポイントから出力数を検出: {num_outputs}")
                    elif 'regressor.weight' in state_dict:
                        num_outputs = state_dict['regressor.weight'].shape[0]
                        print(f"チェックポイントから出力数を検出: {num_outputs}")

                    # 入力サイズを検出
                    if isinstance(checkpoint, dict) and 'input_size' in checkpoint:
                        input_size = tuple(checkpoint['input_size'])
                        print(f"チェックポイントから入力サイズを検出: {input_size}")
                except Exception as e:
                    print(f"出力数/入力サイズの検出に失敗: {e}")

            # モデルの初期化（検出した出力数と入力サイズで）
            model = get_model(model_type, pretrained=False, input_size=input_size, num_outputs=num_outputs)

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

                    # 解像度ダウンスケール（ピクセレーション）
                    if downscale_factor < 1.0:
                        sw = max(1, int(img_width * downscale_factor))
                        sh = max(1, int(img_height * downscale_factor))
                        img = img.resize((sw, sh), Image.NEAREST).resize((img_width, img_height), Image.NEAREST)

                    # 前処理
                    img_tensor = transform(img)
                    img_tensor = img_tensor.unsqueeze(0).to(device)
                    
                    # 推論
                    output = model(img_tensor)
                    output_values = output[0].cpu().numpy()

                    # 推論値を[-1, 1]にクリッピング
                    output_values = np.clip(output_values, -1.0, 1.0)

                    # 出力数に応じて値を取得
                    # データセットの順序: [angle, throttle, ...]
                    # 出力パターン:
                    #   2出力: [angle, throttle]
                    #   3出力: [angle, throttle, speed]
                    #   6出力: [angle, throttle, t+5_angle, t+5_throttle, t+10_angle, t+10_throttle]
                    #   9出力: [angle, throttle, speed, t+5_angle, t+5_throttle, t+5_speed, t+10_angle, t+10_throttle, t+10_speed]
                    num_outputs = len(output_values)
                    angle = output_values[0]
                    throttle = output_values[1]

                    # speedの判定: 3出力または9出力の場合のみspeedが存在
                    speed = None
                    if num_outputs == 3 or num_outputs >= 9:
                        speed = output_values[2]

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

                    # speedがある場合は追加
                    if speed is not None:
                        results[img_path]["speed"] = float(speed)

                    # 将来予測の出力がある場合
                    # 9出力モデル（speed有り）: [angle, throttle, speed, t+5_angle, t+5_throttle, t+5_speed, t+10_angle, t+10_throttle, t+10_speed]
                    # 6出力モデル（speed無し）: [angle, throttle, t+5_angle, t+5_throttle, t+10_angle, t+10_throttle]
                    if num_outputs >= 9:
                        # speed有りの将来予測（9出力）
                        # t+5の値（インデックス3,4,5: angle, throttle, speed）
                        future_5_angle = output_values[3]
                        future_5_throttle = output_values[4]
                        future_5_speed = output_values[5]
                        future_5_x = int((future_5_angle + 1) / 2 * img_width)
                        future_5_y = int((1 - future_5_throttle) / 2 * img_height)
                        future_5_x = max(0, min(future_5_x, img_width - 1))
                        future_5_y = max(0, min(future_5_y, img_height - 1))

                        results[img_path]["future_5"] = {
                            "angle": float(future_5_angle),
                            "throttle": float(future_5_throttle),
                            "speed": float(future_5_speed),
                            "x": future_5_x,
                            "y": future_5_y
                        }

                        # t+10の値（インデックス6,7,8: angle, throttle, speed）
                        future_10_angle = output_values[6]
                        future_10_throttle = output_values[7]
                        future_10_speed = output_values[8]
                        future_10_x = int((future_10_angle + 1) / 2 * img_width)
                        future_10_y = int((1 - future_10_throttle) / 2 * img_height)
                        future_10_x = max(0, min(future_10_x, img_width - 1))
                        future_10_y = max(0, min(future_10_y, img_height - 1))

                        results[img_path]["future_10"] = {
                            "angle": float(future_10_angle),
                            "throttle": float(future_10_throttle),
                            "speed": float(future_10_speed),
                            "x": future_10_x,
                            "y": future_10_y
                        }
                    elif num_outputs == 6:
                        # speed無しの将来予測（6出力）
                        # t+5の値（インデックス2,3: angle, throttle）
                        future_5_angle = output_values[2]
                        future_5_throttle = output_values[3]
                        future_5_x = int((future_5_angle + 1) / 2 * img_width)
                        future_5_y = int((1 - future_5_throttle) / 2 * img_height)
                        future_5_x = max(0, min(future_5_x, img_width - 1))
                        future_5_y = max(0, min(future_5_y, img_height - 1))

                        results[img_path]["future_5"] = {
                            "angle": float(future_5_angle),
                            "throttle": float(future_5_throttle),
                            "x": future_5_x,
                            "y": future_5_y
                        }

                        # t+10の値（インデックス4,5: angle, throttle）
                        future_10_angle = output_values[4]
                        future_10_throttle = output_values[5]
                        future_10_x = int((future_10_angle + 1) / 2 * img_width)
                        future_10_y = int((1 - future_10_throttle) / 2 * img_height)
                        future_10_x = max(0, min(future_10_x, img_width - 1))
                        future_10_y = max(0, min(future_10_y, img_height - 1))

                        results[img_path]["future_10"] = {
                            "angle": float(future_10_angle),
                            "throttle": float(future_10_throttle),
                            "x": future_10_x,
                            "y": future_10_y
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
