#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# TensorRTのPythonバインディングにアクセスするためのパス追加
sys.path.insert(0, '/usr/lib/python3.10/dist-packages')

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


def get_available_model_types():
    """annotation_training_d2j/model_catalog.pyから利用可能なモデルタイプを取得"""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        annotation_path = os.path.join(project_root, 'annotation_training_d2j')
        sys.path.insert(0, annotation_path)
        from model_catalog import list_all_available_models

        all_models = list_all_available_models()
        categorized = {"driving": [], "position": [], "waypoint": []}

        for model_name in all_models:
            if model_name.endswith("_location"):
                categorized["position"].append(model_name)
            elif model_name.endswith("_waypoint"):
                categorized["waypoint"].append(model_name)
            else:
                categorized["driving"].append(model_name)

        return categorized
    except Exception as e:
        logger.warning(f"モデルカタログの読み込みに失敗: {e}")
        return {
            "driving": ["donkeycar", "resnet18", "resnet34", "mobilevit_xxs", "edgenext_xx_small"],
            "position": ["donkey_location", "resnet18_location"],
            "waypoint": ["donkey_waypoint", "resnet18_waypoint"]
        }


def infer_model_type_from_filename(model_path, available_models=None):
    """ファイル名からモデルタイプを推測する"""
    if available_models is None:
        available_models = get_available_model_types()

    basename = os.path.basename(model_path).lower()

    # 全モデル名を長い順にソート（部分一致を防ぐ）
    all_model_names = (
        available_models["position"] +
        available_models["waypoint"] +
        available_models["driving"]
    )
    sorted_models = sorted(all_model_names, key=len, reverse=True)

    # 完全一致を試みる
    for model_name in sorted_models:
        model_name_lower = model_name.lower()
        variants = [
            model_name_lower,
            model_name_lower.replace('_', '-'),
            model_name_lower.replace('_', '')
        ]

        for variant in variants:
            if variant in basename:
                if model_name in available_models["position"]:
                    return model_name, "position"
                elif model_name in available_models["waypoint"]:
                    return model_name, "waypoint"
                else:
                    return model_name, "driving"

    # キーワードベースの推測
    if "location" in basename:
        return ("resnet18_location" if "resnet18" in basename else "donkey_location"), "position"
    elif "waypoint" in basename:
        return ("resnet18_waypoint" if "resnet18" in basename else "donkey_waypoint"), "waypoint"
    else:
        return "donkeycar", "driving"


def main():
    parser = argparse.ArgumentParser(description='PyTorchモデルをTensorRTモデルに変換するツール')
    parser.add_argument('--models_dir', type=str, default='models', help='PyTorchモデルを含むディレクトリ')
    parser.add_argument('--model_type', type=str, default=None, help='モデルタイプ (例: resnet18)')
    parser.add_argument('--width', type=int, default=224, help='入力画像の幅')
    parser.add_argument('--height', type=int, default=224, help='入力画像の高さ')
    parser.add_argument('--batch_size', type=int, default=1, help='バッチサイズ')
    parser.add_argument('--fp16', action='store_true', default=True, help='FP16モードを有効にする（デフォルト: 有効）')
    parser.add_argument('--fp32', action='store_true', help='FP32モードで変換（FP16を無効化）')
    
    args = parser.parse_args()

    # FP32フラグが指定された場合、FP16を無効化
    if args.fp32:
        args.fp16 = False

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
    
    # ファイル名からモデルタイプを自動推測
    available_models = get_available_model_types()
    inferred_model_type, category = infer_model_type_from_filename(selected_model_path, available_models)

    # 選択されたモデルのモデルタイプを取得
    if args.model_type is None:
        print(f"\n推測されたモデルタイプ: {inferred_model_type} ({category}モデル)")
        confirm = input(f"このモデルタイプで変換しますか？ (y/n または別のモデル名を入力): ")
        if confirm.lower() == 'y' or confirm == '':
            model_type = inferred_model_type
        elif confirm.lower() == 'n':
            print("\n利用可能なモデルタイプ:")
            print(f"  自動運転: {', '.join(available_models['driving'])}")
            print(f"  位置推論: {', '.join(available_models['position'])}")
            print(f"  ウェイポイント: {', '.join(available_models['waypoint'])}")
            model_type = input("\nモデルタイプを入力してください: ")
            # カテゴリを再判定
            if model_type in available_models['position']:
                category = 'position'
            elif model_type in available_models['waypoint']:
                category = 'waypoint'
            else:
                category = 'driving'
        else:
            model_type = confirm
            # カテゴリを再判定
            if model_type in available_models['position']:
                category = 'position'
            elif model_type in available_models['waypoint']:
                category = 'waypoint'
            else:
                category = 'driving'
    else:
        model_type = args.model_type
        # カテゴリを判定
        if model_type in available_models['position']:
            category = 'position'
        elif model_type in available_models['waypoint']:
            category = 'waypoint'
        else:
            category = 'driving'

    # モデルカタログからモデルを取得
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # カテゴリに応じてモデルを読み込み
        if category in ['position', 'waypoint']:
            # annotation_training_d2j/model_catalogを使用
            annotation_path = os.path.join(project_root, 'annotation_training_d2j')
            sys.path.insert(0, annotation_path)
            from model_catalog import get_model
            logger.info(f"{category}モデルを読み込みます: {model_type}")
            model = get_model(model_type, pretrained=False, input_size=(args.height, args.width))
        else:
            # train_pytorch.pyのget_model_from_catalogを使用
            sys.path.insert(0, project_root)
            from train_pytorch import get_model_from_catalog
            logger.info(f"自動運転モデルを読み込みます: {model_type}")
            model = get_model_from_catalog(model_type)

        if model is None:
            logger.error(f"モデルタイプ '{model_type}' の読み込みに失敗しました")
            return
        
        # 選択したモデルの重みを読み込む
        device = torch.device('cuda')
        model = load_model_weights(model, selected_model_path, device)
        model = model.to(device)
        model.eval()
        
        # TensorRTモデルのパスを作成（_tensorrt.ptに統一）
        trt_model_path = selected_model_path.replace('.pth', '_tensorrt.pt')
        
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