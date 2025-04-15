"""
推論ユーティリティ - モデルを使用した推論処理（リファクタリング版）
"""

import os
import torch
from PIL import Image
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from model_catalog import get_model, is_dual_input_model

# モデルキャッシュ： (model_type, model_path) -> (model, preprocess, device, is_dual)
_MODEL_CACHE = {}
# デバイスを一度だけ初期化
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_inference(
    image_paths: List[str], 
    method: str = "model", 
    model_type: Optional[str] = None,
    model_path: Optional[str] = None,
    force_reload: bool = False,
    batch_size: int = 16,
    second_image_paths: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    画像バッチに対する推論を実行する - 2画像入力モデル対応
    
    Args:
        image_paths: 推論を実行する画像パスのリスト
        method: 推論方法 ("model" または "random")
        model_type: 使用するモデルタイプ
        model_path: モデルのファイルパス（modelメソッドでのみ使用）
        force_reload: モデルを強制的に再ロードするか
        batch_size: バッチサイズ
        second_image_paths: 2画像目のパスリスト（2画像モデルの場合）
        
    Returns:
        dict: キーが画像パス、値が推論結果の辞書
    """
    results = {}
    
    # ランダム推論の場合（テスト・デバッグ用）
    if method == "random":
        import random
        for img_path in image_paths:
            angle = random.uniform(-1, 1)
            throttle = random.uniform(-1, 1)
            results[img_path] = {
                "angle": angle,
                "throttle": throttle,
                "pilot/angle": angle,
                "pilot/throttle": throttle,
                "x": int((angle + 1) / 2 * 320),
                "y": int((1 - throttle) / 2 * 240)
            }
        return results
    
    # モデルを使った推論
    elif method == "model" and model_type:
        # モデルと前処理を取得（キャッシュがあれば再利用）
        model, preprocess, is_dual = _get_model(model_type, model_path, force_reload)
        
        # 2画像入力モデルだが2画像目がない場合の処理
        if is_dual and (second_image_paths is None or len(second_image_paths) == 0):
            print("警告: 2画像入力モデルですが2画像目が提供されていません。最初の画像の前の画像を使用します。")
            # 前の画像を2画像目として使用
            second_image_paths = []
            for i, img_path in enumerate(image_paths):
                if i > 0:
                    second_image_paths.append(image_paths[i-1])
                else:
                    second_image_paths.append(image_paths[i])  # 最初の画像は自分自身を使用
        
        # 推論実行
        with torch.no_grad():
            # バッチ処理
            for start_idx in range(0, len(image_paths), batch_size):
                # バッチのスライスを取得
                end_idx = min(start_idx + batch_size, len(image_paths))
                batch_paths = image_paths[start_idx:end_idx]
                
                # 2画像入力モデルの場合
                if is_dual and second_image_paths:
                    # 2画像目のバッチを取得
                    batch_second_paths = second_image_paths[start_idx:end_idx]
                    
                    # バッチ内の各画像ペアを処理
                    batch_results = _process_dual_image_batch(
                        batch_paths, 
                        batch_second_paths, 
                        model, 
                        preprocess
                    )
                    results.update(batch_results)
                else:
                    # 通常の1画像入力処理
                    batch_results = _process_single_image_batch(
                        batch_paths,
                        model,
                        preprocess
                    )
                    results.update(batch_results)
                    
        return results
    else:
        raise ValueError(f"サポートされていない推論方法: {method}")


def _get_model(model_type, model_path, force_reload=False):
    """モデルを取得またはロードする（キャッシュ対応）"""
    global _MODEL_CACHE, _DEVICE
    
    # キャッシュキーを作成
    cache_key = (model_type, model_path)
    
    # キャッシュからモデルを取得するか、新しくロードする
    if not force_reload and cache_key in _MODEL_CACHE:
        model, preprocess, is_dual = _MODEL_CACHE[cache_key]
    else:
        # モデルの初期化
        model = get_model(model_type, pretrained=False)
        
        # 保存済みモデルをロード
        if model_path and os.path.exists(model_path):
            try:
                # モデルの状態を読み込む
                checkpoint = torch.load(model_path, map_location=_DEVICE)
                
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
        
        # 前処理を取得
        preprocess = model.get_preprocess()
        
        # 2画像入力モデルかどうかを判定
        is_dual = is_dual_input_model(model)
        
        # デバイスへ転送
        model = model.to(_DEVICE)
        model.eval()  # 評価モードに設定
        
        # キャッシュに保存
        _MODEL_CACHE[cache_key] = (model, preprocess, is_dual)
    
    return model, preprocess, is_dual


def _process_single_image_batch(batch_paths, model, preprocess):
    """単一画像バッチの推論処理"""
    results = {}
    global _DEVICE
    
    # 画像バッチをテンソルに変換
    batch_tensors = []
    batch_sizes = []  # 各画像のサイズを保存
    
    for img_path in batch_paths:
        try:
            # 画像を読み込む
            img = Image.open(img_path).convert('RGB')
            batch_sizes.append((img.width, img.height))
            
            # 前処理
            img_tensor = preprocess(img)
            img_tensor = img_tensor.unsqueeze(0)
            batch_tensors.append(img_tensor)
        except Exception as e:
            print(f"画像の読み込みエラー {img_path}: {e}")
            # エラーの場合はこの画像をスキップ
            continue
    
    if not batch_tensors:
        return results  # 有効な画像がない場合は空の結果を返す
    
    # バッチをまとめる
    batch_tensor = torch.cat(batch_tensors, dim=0).to(_DEVICE)
    
    # 推論実行
    outputs = model(batch_tensor)
    outputs_np = outputs.cpu().numpy()
    
    # 各画像の結果を解析
    for i, (img_path, (img_width, img_height)) in enumerate(zip(batch_paths, batch_sizes)):
        if i >= len(outputs_np):
            continue  # 出力の数が入力より少ない場合の対策
        
        # 値の範囲を調整 (0-1 から -1,1 に変換)
        angle = float(outputs_np[i][0] * 2 - 1)
        throttle = float(outputs_np[i][1] * 2 - 1)
        
        # ピクセル座標に変換
        x = int((angle + 1) / 2 * img_width)
        y = int((1 - throttle) / 2 * img_height)
        
        # 画像サイズに合わせてクリップ
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        
        # 結果を格納
        results[img_path] = {
            "angle": angle,
            "throttle": throttle,
            "pilot/angle": angle,
            "pilot/throttle": throttle,
            "x": x,
            "y": y
        }
    
    return results


def _process_dual_image_batch(batch_paths, batch_second_paths, model, preprocess):
    """2画像入力モデル用のバッチ処理"""
    results = {}
    global _DEVICE
    
    # 両方の画像バッチをテンソルに変換
    batch_tensors1 = []
    batch_tensors2 = []
    batch_sizes = []  # 各画像のサイズを保存
    valid_indices = []  # 有効な画像のインデックス
    
    for i, (img_path, second_path) in enumerate(zip(batch_paths, batch_second_paths)):
        try:
            # 1枚目の画像
            img1 = Image.open(img_path).convert('RGB')
            batch_sizes.append((img1.width, img1.height))
            
            # 2枚目の画像
            img2 = Image.open(second_path).convert('RGB')
            
            # 前処理
            img_tensor1 = preprocess(img1).unsqueeze(0)
            img_tensor2 = preprocess(img2).unsqueeze(0)
            
            batch_tensors1.append(img_tensor1)
            batch_tensors2.append(img_tensor2)
            valid_indices.append(i)
        except Exception as e:
            print(f"画像ペアの読み込みエラー {img_path}, {second_path}: {e}")
            # エラーの場合はこの画像ペアをスキップ
            continue
    
    if not batch_tensors1 or not batch_tensors2:
        return results  # 有効な画像がない場合は空の結果を返す
    
    # バッチをまとめる
    batch_tensor1 = torch.cat(batch_tensors1, dim=0).to(_DEVICE)
    batch_tensor2 = torch.cat(batch_tensors2, dim=0).to(_DEVICE)
    
    # 推論実行（2画像入力）
    outputs = model(batch_tensor1, batch_tensor2)
    outputs_np = outputs.cpu().numpy()
    
    # 各画像の結果を解析
    for idx, orig_idx in enumerate(valid_indices):
        if idx >= len(outputs_np):
            continue  # 出力の数が入力より少ない場合の対策
        
        img_path = batch_paths[orig_idx]
        img_width, img_height = batch_sizes[idx]
        
        # 値の範囲を調整 (0-1 から -1,1 に変換)
        angle = float(outputs_np[idx][0] * 2 - 1)
        throttle = float(outputs_np[idx][1] * 2 - 1)
        
        # ピクセル座標に変換
        x = int((angle + 1) / 2 * img_width)
        y = int((1 - throttle) / 2 * img_height)
        
        # 画像サイズに合わせてクリップ
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        
        # 結果を格納
        results[img_path] = {
            "angle": angle,
            "throttle": throttle,
            "pilot/angle": angle,
            "pilot/throttle": throttle,
            "x": x,
            "y": y
        }
    
    return results


def clear_model_cache():
    """モデルキャッシュをクリアする - メモリ解放が必要な場合に使用"""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # GPUメモリも解放