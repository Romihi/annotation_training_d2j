#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import logging
import glob
from pathlib import Path

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_pytorch_to_tensorrt(model, input_size=(224, 224), batch_size=1, 
                               fp16_mode=True, max_workspace_size=1<<25, 
                               save_path=None, device='cuda'):
    """
    PyTorchモデルをTensorRTモデルに変換する
    
    Args:
        model: 変換するPyTorchモデル（すでにcudaデバイスに配置済みであること）
        input_size: 入力画像サイズ（height, width）
        batch_size: バッチサイズ
        fp16_mode: 半精度（FP16）を使用するかどうか
        max_workspace_size: TensorRTエンジンに割り当てる最大ワークスペースサイズ
        save_path: 変換したモデルを保存するパス（Noneの場合は保存しない）
        device: 使用するデバイス
        
    Returns:
        変換されたTensorRTモデル
    """
    try:
        # torch2trtがインポートできるか確認
        from torch2trt import torch2trt, TRTModule
    except ImportError:
        logger.error("torch2trt がインストールされていません。pip install torch2trt でインストールしてください。")
        return None
    
    # CUDAが利用可能か確認
    if not torch.cuda.is_available():
        logger.error("CUDA が利用できないため、TensorRTへの変換ができません。")
        return None
    
    # モデルをCUDAに移動し、評価モードに設定
    model = model.to(device)
    model.eval()
    
    # 入力サイズに基づいてダミー入力を作成
    x = torch.ones((batch_size, 3, input_size[0], input_size[1])).to(device)
    
    # モデルをTensorRTに変換
    logger.info("PyTorchモデルをTensorRTに変換しています...")
    try:
        model_trt = torch2trt(
            model, 
            [x], 
            fp16_mode=fp16_mode,
            max_workspace_size=max_workspace_size
        )
        logger.info("TensorRTへの変換が完了しました。")
        
        # 変換したモデルを保存
        if save_path:
            torch.save(model_trt.state_dict(), save_path)
            logger.info(f"TensorRTモデルを {save_path} に保存しました。")
        
        return model_trt
    
    except Exception as e:
        logger.error(f"TensorRTへの変換中にエラーが発生しました: {e}")
        return None


def load_tensorrt_model(model_path, device='cuda'):
    """
    保存されたTensorRTモデルを読み込む
    
    Args:
        model_path: TensorRTモデルへのパス
        device: モデルを配置するデバイス
        
    Returns:
        読み込まれたTensorRTモデル
    """
    try:
        from torch2trt import TRTModule
    except ImportError:
        logger.error("torch2trt がインストールされていません。")
        return None
    
    if not os.path.exists(model_path):
        logger.error(f"モデルファイル {model_path} が見つかりません。")
        return None
    
    try:
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"TensorRTモデルを {model_path} から読み込みました。")
        return model_trt
    
    except Exception as e:
        logger.error(f"TensorRTモデル読み込み中にエラーが発生しました: {e}")
        return None


def load_model_weights(model, weights_path, device):
    """
    モデルに重みを読み込む（チェックポイント形式か通常の形式かを自動判定）
    
    Args:
        model: 重みを読み込むモデル
        weights_path: 重みファイルのパス
        device: 使用するデバイス
        
    Returns:
        重みを読み込んだモデル
    """
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("チェックポイント形式のモデルを読み込みました。")
    else:
        model.load_state_dict(checkpoint)
        logger.info("state_dict形式のモデルを読み込みました。")
    return model


def find_pytorch_models(models_dir):
    """
    指定されたディレクトリ内のPyTorchモデル（.pthファイル）を探す
    
    Args:
        models_dir: 検索するディレクトリ
        
    Returns:
        見つかったPyTorchモデルのリスト
    """
    # モデルディレクトリが存在するか確認
    if not os.path.exists(models_dir):
        logger.error(f"ディレクトリ {models_dir} が見つかりません。")
        return []
    
    # .pthファイルを検索
    pth_files = glob.glob(os.path.join(models_dir, "**", "*.pth"), recursive=True)
    
    # _trtが含まれているファイルを除外（すでに変換済みのモデル）
    pth_files = [f for f in pth_files if "_trt" not in f]
    
    return pth_files


def main():
    parser = argparse.ArgumentParser(description='PyTorchモデルをTensorRTモデルに変換するツール')
    parser.add_argument('--models_dir', type=str, default='models', help='PyTorchモデルを含むディレクトリ')
    parser.add_argument('--model_type', type=str, default=None, help='モデルタイプ (例: resnet18)')
    parser.add_argument('--width', type=int, default=224, help='入力画像の幅')
    parser.add_argument('--height', type=int, default=224, help='入力画像の高さ')
    parser.add_argument('--batch_size', type=int, default=1, help='バッチサイズ')
    parser.add_argument('--fp16', action='store_true', help='FP16モードを有効にする')
    
    args = parser.parse_args()
    
    # CUDAが利用可能か確認
    if not torch.cuda.is_available():
        logger.error("CUDA が利用できないため、TensorRTへの変換ができません。")
        return
    
    try:
        from torch2trt import TRTModule
    except ImportError:
        logger.error("torch2trt がインストールされていません。")
        return
    
    # PyTorchモデルを検索
    pth_files = find_pytorch_models(args.models_dir)
    
    if not pth_files:
        logger.error(f"{args.models_dir} 内にPyTorchモデル（.pthファイル）が見つかりませんでした。")
        return
    
    # 見つかったモデルを表示
    print("\n=== 変換可能なPyTorchモデル ===")
    for i, model_path in enumerate(pth_files):
        print(f"{i+1}. {model_path}")
    
    # ユーザーにモデルを選択してもらう
    while True:
        try:
            choice = input("\n変換するモデルの番号を入力してください（qで終了）: ")
            if choice.lower() == 'q':
                return
            
            idx = int(choice) - 1
            if 0 <= idx < len(pth_files):
                selected_model_path = pth_files[idx]
                break
            else:
                print("有効な番号を入力してください。")
        except ValueError:
            print("数字または 'q' を入力してください。")
    
    # 選択されたモデルのモデルタイプを取得
    if args.model_type is None:
        model_type = input("モデルタイプを入力してください（例: resnet18）: ")
    else:
        model_type = args.model_type
    
    # モデルカタログからモデルを取得
    try:
        # モデルカタログをインポート
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model_catalog import get_model
        
        # モデルを作成
        model = get_model(model_type, pretrained=False, input_size=(args.height, args.width))
        
        # 選択したモデルの重みを読み込む
        device = torch.device('cuda')
        model = load_model_weights(model, selected_model_path, device)
        model = model.to(device)
        model.eval()
        
        # TensorRTモデルのパスを作成
        trt_model_path = selected_model_path.replace('.pth', '_trt.pth')
        
        # ユーザーに変換の確認
        print(f"\n選択したモデル: {selected_model_path}")
        print(f"変換後のモデル: {trt_model_path}")
        print(f"入力サイズ: 高さ={args.height}, 幅={args.width}")
        print(f"FP16モード: {'有効' if args.fp16 else '無効'}")
        
        confirm = input("\nこの設定でモデルを変換しますか？ (y/n): ")
        if confirm.lower() != 'y':
            print("変換をキャンセルしました。")
            return
        
        # モデルを変換
        model_trt = convert_pytorch_to_tensorrt(
            model, 
            input_size=(args.height, args.width),
            batch_size=args.batch_size,
            fp16_mode=args.fp16,
            save_path=trt_model_path,
            device=device
        )
        
        if model_trt is not None:
            print(f"\nモデルの変換に成功しました。変換後のモデルは {trt_model_path} に保存されました。")
        else:
            print("\nモデルの変換に失敗しました。")
    
    except ImportError:
        logger.error("model_catalog モジュールをインポートできませんでした。")
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()