#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
from pathlib import Path

def convert_pytorch_to_onnx(model_path, model_type, output_path=None, input_size=(224, 224), 
                           dynamic_axes=True, simplify=True, opset_version=12):
    """
    PyTorchモデルをONNX形式に変換する
    
    Args:
        model_path: 変換するPyTorchモデルのパス
        model_type: モデルタイプ（例: resnet18）
        output_path: 出力ファイルパス (Noneの場合は自動生成)
        input_size: 入力画像サイズ (高さ, 幅) - 保存されたモデルから自動検出される場合は上書きされる
        dynamic_axes: バッチサイズを動的にするかどうか
        simplify: ONNXモデルを単純化するかどうか
        opset_version: ONNXのopsetバージョン
        
    Returns:
        変換されたONNXモデルのパス
    """
    # 出力パスが指定されていない場合、元のファイル名を基に自動生成
    if output_path is None:
        base_path = os.path.splitext(model_path)[0]
        output_path = f"{base_path}.onnx"
    
    # モデルカタログのインポート
    try:
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.dirname(script_dir))
        from model_catalog import get_model, load_model_weights
    except ImportError as e:
        print(f"モデルカタログのインポートに失敗しました: {e}")
        print("このスクリプトはDonkeyCar環境内で実行する必要があります")
        return None
    
    print(f"PyTorchモデル '{model_path}' をONNX形式に変換しています...")
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    try:
        # まず保存されたモデルから入力サイズ情報を取得を試みる
        actual_input_size = input_size
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                # 保存された入力サイズがあるかチェック
                if 'input_size' in checkpoint:
                    actual_input_size = checkpoint['input_size']
                    print(f"保存されたモデルから入力サイズを検出: {actual_input_size}")
                elif 'model_input_size' in checkpoint:
                    actual_input_size = checkpoint['model_input_size']
                    print(f"保存されたモデルから入力サイズを検出: {actual_input_size}")
                # DonkeyModelなど、重みのサイズから入力サイズを推定
                elif 'model_state_dict' in checkpoint and model_type.lower() in ['donkey_model', 'donkey', 'donkey_location']:
                    state_dict = checkpoint['model_state_dict']
                    if 'dense_layers.0.weight' in state_dict:
                        # 最初の全結合層の重みサイズから入力サイズを逆算
                        first_layer_input_size = state_dict['dense_layers.0.weight'].shape[1]
                        print(f"最初の全結合層の入力サイズ: {first_layer_input_size}")
                        
                        # DonkeyModelの場合、畳み込み出力サイズから元の画像サイズを推定
                        # 一般的なサイズパターンを試行
                        size_patterns = [
                            (120, 160),  # 標準サイズ
                            (224, 224),  # 正方形サイズ
                            (240, 320),  # 2倍サイズ
                            (180, 240),  # 1.5倍サイズ
                            (96, 128),   # 0.8倍サイズ
                            (60, 80),    # 0.5倍サイズ
                        ]
                        
                        for test_size in size_patterns:
                            # DonkeyModelの畳み込み出力サイズを計算
                            # Conv1: (3,224,224) -> (32,112,112)
                            # Conv2: (32,112,112) -> (64,56,56) 
                            # Conv3: (64,56,56) -> (64,28,28)
                            # Flatten: 64*28*28 = 50176 (for 224x224 input)
                            h, w = test_size
                            conv_out_h = ((h // 2) // 2) // 2  # 3回のmax pooling
                            conv_out_w = ((w // 2) // 2) // 2
                            expected_features = 64 * conv_out_h * conv_out_w
                            
                            if expected_features == first_layer_input_size:
                                actual_input_size = test_size
                                print(f"重みサイズから入力サイズを推定: {actual_input_size}")
                                break
                        else:
                            print(f"警告: 重みサイズ {first_layer_input_size} に対応する入力サイズを特定できませんでした")
                            
        except Exception as e:
            print(f"入力サイズの自動検出に失敗しました（デフォルト値を使用）: {e}")
        
        print(f"使用する入力サイズ: {actual_input_size}")
        
        # 現在の入力サイズと学習済みモデルのサイズが異なる場合の警告
        if input_size != actual_input_size:
            print(f"警告: 指定された入力サイズ {input_size} と学習済みモデルのサイズ {actual_input_size} が異なります")
            print(f"ONNXモデルは学習済みモデルのサイズ {actual_input_size} で作成されます")
            print(f"推論時は入力画像を {actual_input_size} にリサイズしてください")
        
        # 位置モデルの場合、クラス数も検出
        num_classes = None
        if model_type.lower() == "donkey_location":
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    # 最終分類層からクラス数を検出
                    if 'classifier.weight' in state_dict:
                        num_classes = state_dict['classifier.weight'].shape[0]
                        print(f"検出されたクラス数: {num_classes}")
                    elif 'regressor.weight' in state_dict:
                        num_classes = state_dict['regressor.weight'].shape[0]
                        print(f"検出されたクラス数: {num_classes}")
            except Exception as e:
                print(f"クラス数の検出に失敗（デフォルト値8を使用）: {e}")
                num_classes = 8
        
        # モデルのロード
        if model_type.lower() == "donkey_location" and num_classes is not None:
            # 位置モデルの場合、クラス数も指定
            from model_catalog import DonkeyLocationModel
            model = DonkeyLocationModel(num_classes=num_classes, pretrained=False, input_size=actual_input_size)
        else:
            model = get_model(model_type, pretrained=False, input_size=actual_input_size)
        model = load_model_weights(model, model_path, device)
        model = model.to(device)
        model.eval()
        
        # ダミー入力の作成（検出された実際の入力サイズを使用）
        dummy_input = torch.randn(1, 3, actual_input_size[0], actual_input_size[1], device=device)
        
        # 動的軸の設定（バッチサイズを動的にする）
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
        # ONNXエクスポートの実行
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict
        )
        
        # ONNXモデルを単純化（onnx-simplifierが必要）
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnxsim_simplify
                
                print("ONNXモデルを単純化しています...")
                onnx_model = onnx.load(output_path)
                model_simplified, check = onnxsim_simplify(onnx_model)
                
                if check:
                    onnx.save(model_simplified, output_path)
                    print("ONNXモデルの単純化が成功しました")
                else:
                    print("警告: ONNXモデルの単純化に失敗しました")
            except ImportError:
                print("警告: onnx-simplifierがインストールされていないため、単純化をスキップします")
                print("pip install onnx-simplifier でインストールできます")
        
        print(f"変換が完了しました！ONNXモデルは '{output_path}' に保存されました")
        
        # ONNXモデルの検証
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNXモデルの検証に成功しました")
            
            # モデル情報の表示
            print(f"モデル情報:")
            print(f"  - 入力: {onnx_model.graph.input[0].name}, 形状: {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
            print(f"  - 出力: {onnx_model.graph.output[0].name}, 形状: {[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")
            print(f"  - 学習時の入力サイズ: {actual_input_size}")
            if input_size != actual_input_size:
                print(f"  - 注意: 現在の設定入力サイズ: {input_size}")
                print(f"  - 推論時は画像を {actual_input_size} にリサイズしてください")
            
            # ONNX Runtimeのテスト実行（インストールされている場合）
            try:
                import onnxruntime as ort
                
                print("\nONNX Runtimeで推論テストを実行しています...")
                # 利用可能なプロバイダの表示
                available_providers = ort.get_available_providers()
                print(f"利用可能なプロバイダ: {available_providers}")
                
                # 安定したプロバイダを優先して使用（TensorRTエラーを回避）
                preferred_providers = []
                if 'CUDAExecutionProvider' in available_providers:
                    preferred_providers.append('CUDAExecutionProvider')
                preferred_providers.append('CPUExecutionProvider')
                
                print(f"使用するプロバイダ: {preferred_providers}")
                
                # セッション作成
                session = ort.InferenceSession(output_path, providers=preferred_providers)
                
                # テスト入力の準備（検出された実際の入力サイズを使用）
                test_input = np.random.randn(1, 3, actual_input_size[0], actual_input_size[1]).astype(np.float32)
                input_name = session.get_inputs()[0].name
                
                # 推論実行
                result = session.run(None, {input_name: test_input})
                print(f"テスト推論の結果: 形状={result[0].shape}, 値={result[0][0]}")
                print("ONNX Runtime推論テストに成功しました")
                
            except ImportError:
                print("ONNX Runtimeがインストールされていないため、推論テストをスキップします")
                print("pip install onnxruntime でインストールできます")
            
        except Exception as e:
            print(f"ONNXモデルの検証中にエラーが発生しました: {e}")
        
        return output_path
        
    except Exception as e:
        print(f"変換中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='PyTorchモデルをONNX形式に変換するスクリプト')
    parser.add_argument('--model_path', type=str, required=True, help='変換するPyTorchモデルのパス')
    parser.add_argument('--model_type', type=str, required=True, help='モデルタイプ (例: resnet18)')
    parser.add_argument('--output_path', type=str, default=None, help='出力ONNXファイルのパス（省略可）')
    parser.add_argument('--width', type=int, default=224, help='入力画像の幅')
    parser.add_argument('--height', type=int, default=224, help='入力画像の高さ')
    parser.add_argument('--no-dynamic', action='store_true', help='動的バッチサイズを無効にする')
    parser.add_argument('--no-simplify', action='store_true', help='ONNXモデルの単純化を無効にする')
    parser.add_argument('--opset', type=int, default=12, help='ONNXのopsetバージョン')
    
    args = parser.parse_args()
    
    # PyTorchモデルをONNX形式に変換
    convert_pytorch_to_onnx(
        model_path=args.model_path,
        model_type=args.model_type,
        output_path=args.output_path,
        input_size=(args.height, args.width),
        dynamic_axes=not args.no_dynamic,
        simplify=not args.no_simplify,
        opset_version=args.opset
    )


if __name__ == "__main__":
    main()