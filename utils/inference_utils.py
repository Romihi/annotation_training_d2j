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

from model_catalog import get_model, list_available_models, apply_vehicle_mask

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
                        # 学習時のspeed正規化値（保存されていれば表示側で利用）
                        if checkpoint.get('speed_normalize'):
                            model._speed_normalize = float(checkpoint['speed_normalize'])
                        # 学習時の車両マスク（保存されていれば推論時にも同じマスクを適用）
                        if checkpoint.get('vehicle_mask'):
                            model._vehicle_mask = [tuple(p) for p in checkpoint['vehicle_mask']]
                        # 学習時の将来予測フレームオフセット（推論結果のキー・表示に利用）
                        if checkpoint.get('future_offsets'):
                            model._future_offsets = [int(v) for v in checkpoint['future_offsets']]
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

                    # 学習時に車両マスクを使ったモデルは推論時にも同じマスクを適用
                    img = apply_vehicle_mask(img, getattr(model, '_vehicle_mask', None))

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
                        # 学習時のspeed正規化値（m/s換算・表示位置の計算用）
                        _speed_norm = getattr(model, '_speed_normalize', None)
                        if _speed_norm:
                            results[img_path]["speed_normalize"] = float(_speed_norm)

                    # 将来予測の出力がある場合
                    # 9出力モデル（speed有り）: [angle, throttle, speed, t+o1_angle, t+o1_throttle, t+o1_speed, t+o2_angle, t+o2_throttle, t+o2_speed]
                    # 6出力モデル（speed無し）: [angle, throttle, t+o1_angle, t+o1_throttle, t+o2_angle, t+o2_throttle]
                    # フレームオフセット(o1, o2)は学習時の設定（チェックポイント保存値、既定は5,10）
                    future_offsets = (getattr(model, '_future_offsets', None) or [5, 10])[:2]
                    if num_outputs >= 9:
                        # speed有りの将来予測（9出力）
                        for fi, offset in enumerate(future_offsets):
                            base = 3 + fi * 3
                            f_angle = output_values[base]
                            f_throttle = output_values[base + 1]
                            f_speed = output_values[base + 2]
                            f_x = max(0, min(int((f_angle + 1) / 2 * img_width), img_width - 1))
                            f_y = max(0, min(int((1 - f_throttle) / 2 * img_height), img_height - 1))
                            results[img_path][f"future_{offset}"] = {
                                "angle": float(f_angle),
                                "throttle": float(f_throttle),
                                "speed": float(f_speed),
                                "x": f_x,
                                "y": f_y
                            }
                        results[img_path]["future_offsets"] = list(future_offsets)
                    elif num_outputs == 6:
                        # speed無しの将来予測（6出力）
                        for fi, offset in enumerate(future_offsets):
                            base = 2 + fi * 2
                            f_angle = output_values[base]
                            f_throttle = output_values[base + 1]
                            f_x = max(0, min(int((f_angle + 1) / 2 * img_width), img_width - 1))
                            f_y = max(0, min(int((1 - f_throttle) / 2 * img_height), img_height - 1))
                            results[img_path][f"future_{offset}"] = {
                                "angle": float(f_angle),
                                "throttle": float(f_throttle),
                                "x": f_x,
                                "y": f_y
                            }
                        results[img_path]["future_offsets"] = list(future_offsets)
                    
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
