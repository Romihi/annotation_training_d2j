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
        input_size: 入力画像サイズ (高さ, 幅)
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
        # モデルのロード
        model = get_model(model_type, pretrained=False, input_size=input_size)
        model = load_model_weights(model, model_path, device)
        model = model.to(device)
        model.eval()
        
        # ダミー入力の作成
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)
        
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
            
            # ONNX Runtimeのテスト実行（インストールされている場合）
            try:
                import onnxruntime as ort
                
                print("\nONNX Runtimeで推論テストを実行しています...")
                # 利用可能なプロバイダの表示
                providers = ort.get_available_providers()
                print(f"利用可能なプロバイダ: {providers}")
                
                # セッション作成
                session = ort.InferenceSession(output_path, providers=providers)
                
                # テスト入力の準備
                test_input = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)
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