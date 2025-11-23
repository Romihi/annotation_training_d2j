#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import time

def convert_onnx_to_openvino(onnx_path, output_path=None, precision='FP32', compress_to_fp16=False):
    """
    ONNXモデルをOpenVINO形式に変換する
    
    Args:
        onnx_path: 変換するONNXモデルのパス
        output_path: 出力ファイルパス (Noneの場合は自動生成)
        precision: モデルの精度 (FP32, FP16, INT8)
        compress_to_fp16: FP16に圧縮するかどうか
        
    Returns:
        変換されたOpenVINOモデルのパス
    """
    # 出力パスが指定されていない場合、元のファイル名を基に自動生成
    if output_path is None:
        base_path = os.path.splitext(onnx_path)[0]
        output_path = f"{base_path}_openvino"
    
    print(f"ONNXモデル '{onnx_path}' をOpenVINO形式に変換しています...")
    
    try:
        # OpenVINOへの変換
        try:
            from openvino.tools import mo
            from openvino.runtime import serialize
        except ImportError:
            print("OpenVINOがインストールされていません。以下のコマンドでインストールしてください:")
            print("pip install openvino-dev")
            return None
        
        # OpenVINOモデルへの変換
        print("OpenVINO形式に変換しています...")
        
        # Model Optimizerのパラメータ設定
        mo_args = [
            '--input_model', onnx_path,
            '--output_dir', os.path.dirname(output_path) or '.',
            '--model_name', os.path.basename(output_path),
            '--data_type', precision.upper()
        ]
        
        if compress_to_fp16 and precision == 'FP32':
            mo_args.extend(['--compress_to_fp16'])
        
        # Model Optimizerの実行
        try:
            # 新しいOpenVINO APIを使用
            from openvino.tools import mo
            from openvino.runtime import Core
            
            # Model Optimizerを直接実行
            mo_result = mo.convert_model(
                onnx_path,
                compress_to_fp16=(compress_to_fp16 and precision == 'FP32')
            )
            
            # モデルを保存
            from openvino.runtime import serialize
            serialize(mo_result, f"{output_path}.xml", f"{output_path}.bin")
            
        except Exception as e:
            # 古いAPIまたはCLI経由での実行を試みる
            import subprocess
            cmd = ['mo'] + mo_args
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                if result.stderr:
                    print("警告:", result.stderr)
            except subprocess.CalledProcessError as e:
                print(f"Model Optimizer実行エラー: {e}")
                print(f"標準出力: {e.stdout}")
                print(f"標準エラー: {e.stderr}")
                raise
        
        # 出力ファイルの確認
        xml_path = f"{output_path}.xml"
        bin_path = f"{output_path}.bin"
        
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            print(f"変換が完了しました！")
            print(f"  - XMLファイル: {xml_path}")
            print(f"  - BINファイル: {bin_path}")
            
            # ファイルサイズの表示
            onnx_size = os.path.getsize(onnx_path) / 1024 / 1024  # MB
            openvino_size = (os.path.getsize(xml_path) + os.path.getsize(bin_path)) / 1024 / 1024  # MB
            
            print(f"\nファイルサイズ:")
            print(f"  - 元のONNXモデル: {onnx_size:.1f} MB")
            print(f"  - OpenVINOモデル: {openvino_size:.1f} MB")
            print(f"  - サイズ削減: {(1 - openvino_size/onnx_size)*100:.1f}%")
            
            # OpenVINOモデルの検証
            try:
                from openvino.runtime import Core
                
                print("\nOpenVINOモデルの検証を実行しています...")
                ie = Core()
                
                # 利用可能なデバイスの表示
                available_devices = ie.available_devices
                print(f"利用可能なデバイス: {available_devices}")
                
                # モデルの読み込み
                network = ie.read_model(model=xml_path, weights=bin_path)
                
                # 入出力情報の表示
                print("\nモデル情報:")
                for input_info in network.inputs:
                    input_shape = input_info.shape
                    print(f"  - 入力: {input_info.any_name}, 形状: {input_shape}")
                
                for output_info in network.outputs:
                    output_shape = output_info.shape
                    print(f"  - 出力: {output_info.any_name}, 形状: {output_shape}")
                
                # テスト推論（CPU）
                print("\nCPUでテスト推論を実行しています...")
                compiled_model = ie.compile_model(network, "CPU")
                
                print("OpenVINOモデルの検証に成功しました")
                
            except Exception as e:
                print(f"OpenVINOモデルの検証中にエラーが発生しました: {e}")
            
            return xml_path
        else:
            print("エラー: OpenVINOモデルファイルが生成されませんでした")
            return None
        
    except Exception as e:
        print(f"変換中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_pytorch_to_openvino(model_path, model_type, output_path=None, input_size=(224, 224), 
                               precision='FP32', compress_to_fp16=False, dynamic_batch=True):
    """
    PyTorchモデルをOpenVINO形式に変換する
    
    Args:
        model_path: 変換するPyTorchモデルのパス
        model_type: モデルタイプ（例: resnet18）
        output_path: 出力ファイルパス (Noneの場合は自動生成)
        input_size: 入力画像サイズ (高さ, 幅) - 保存されたモデルから自動検出される場合は上書きされる
        precision: モデルの精度 (FP32, FP16, INT8)
        compress_to_fp16: FP16に圧縮するかどうか
        
    Returns:
        変換されたOpenVINOモデルのパス
    """
    # 出力パスが指定されていない場合、元のファイル名を基に自動生成
    if output_path is None:
        base_path = os.path.splitext(model_path)[0]
        output_path = f"{base_path}_openvino"
    
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
    
    print(f"PyTorchモデル '{model_path}' をOpenVINO形式に変換しています...")
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    try:
        # まず保存されたモデルから入力サイズ情報を取得を試みる
        actual_input_size = input_size
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
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
            print(f"OpenVINOモデルは学習済みモデルのサイズ {actual_input_size} で作成されます")
            print(f"推論時は入力画像を {actual_input_size} にリサイズしてください")
        
        # 位置モデルの場合、クラス数も検出
        num_classes = None
        if model_type.lower() == "donkey_location":
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
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
            model = get_model(model_type, pretrained=False, input_size=actual_input_size)
        else:
            model = get_model(model_type, pretrained=False, input_size=actual_input_size)
        model = load_model_weights(model, model_path, device)
        model = model.to(device)
        model.eval()
        
        # OpenVINOへの変換
        try:
            from openvino.tools import mo
            from openvino.runtime import serialize
        except ImportError:
            print("OpenVINOがインストールされていません。以下のコマンドでインストールしてください:")
            print("pip install openvino-dev")
            return None
        
        # 一時的にONNXファイルを作成（OpenVINOはONNX経由で変換）
        temp_onnx_path = os.path.splitext(model_path)[0] + "_temp.onnx"
        
        # ダミー入力の作成（検出された実際の入力サイズを使用）
        dummy_input = torch.randn(1, 3, actual_input_size[0], actual_input_size[1], device=device)
        
        # ONNXエクスポート
        if dynamic_batch:
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        else:
            dynamic_axes = None
            
        torch.onnx.export(
            model,
            dummy_input,
            temp_onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        print("ONNX中間ファイルを作成しました")
        
        # OpenVINOモデルへの変換
        print("OpenVINO形式に変換しています...")
        
        # Model Optimizerのパラメータ設定
        output_dir = os.path.dirname(output_path) or '.'
        mo_args = [
            '--input_model', temp_onnx_path,
            '--output_dir', output_dir,
            '--model_name', os.path.basename(output_path),
            '--input_shape', f'[1,3,{actual_input_size[0]},{actual_input_size[1]}]',
            '--data_type', precision.upper()
        ]
        
        if compress_to_fp16 and precision == 'FP32':
            mo_args.extend(['--compress_to_fp16'])
        
        # Model Optimizerの実行
        try:
            # 新しいOpenVINO APIを使用
            from openvino.tools import mo
            from openvino.runtime import Core
            
            # Model Optimizerを直接実行
            mo_result = mo.convert_model(
                temp_onnx_path,
                compress_to_fp16=(compress_to_fp16 and precision == 'FP32')
            )
            
            # モデルを保存
            from openvino.runtime import serialize
            serialize(mo_result, f"{output_path}.xml", f"{output_path}.bin")
            
        except Exception as e:
            # 古いAPIまたはCLI経由での実行を試みる
            import subprocess
            cmd = ['mo'] + mo_args
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                if result.stderr:
                    print("警告:", result.stderr)
            except subprocess.CalledProcessError as e:
                print(f"Model Optimizer実行エラー: {e}")
                print(f"標準出力: {e.stdout}")
                print(f"標準エラー: {e.stderr}")
                raise
        
        # 一時ファイルの削除
        if os.path.exists(temp_onnx_path):
            os.remove(temp_onnx_path)
            print("一時ファイルを削除しました")
        
        # 出力ファイルの確認
        xml_path = f"{output_path}.xml"
        bin_path = f"{output_path}.bin"
        
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            print(f"変換が完了しました！")
            print(f"  - XMLファイル: {xml_path}")
            print(f"  - BINファイル: {bin_path}")
            print(f"  - 学習時の入力サイズ: {actual_input_size}")
            if input_size != actual_input_size:
                print(f"  - 注意: 現在の設定入力サイズ: {input_size}")
                print(f"  - 推論時は画像を {actual_input_size} にリサイズしてください")
            
            # OpenVINOモデルの検証
            try:
                from openvino.runtime import Core
                
                print("\nOpenVINOモデルの検証を実行しています...")
                ie = Core()
                
                # 利用可能なデバイスの表示
                available_devices = ie.available_devices
                print(f"利用可能なデバイス: {available_devices}")
                
                # モデルの読み込み
                network = ie.read_model(model=xml_path, weights=bin_path)
                
                # 入出力情報の表示
                print("\nモデル情報:")
                for input_info in network.inputs:
                    input_name = input_info.any_name
                    try:
                        # 動的形状の場合は partial_shape を使用
                        if input_info.partial_shape.is_dynamic:
                            input_shape = str(input_info.partial_shape)
                        else:
                            input_shape = list(input_info.shape)
                    except:
                        input_shape = "動的形状"
                    print(f"  - 入力: {input_name}, 形状: {input_shape}")
                
                for output_info in network.outputs:
                    output_name = output_info.any_name
                    try:
                        # 動的形状の場合は partial_shape を使用
                        if output_info.partial_shape.is_dynamic:
                            output_shape = str(output_info.partial_shape)
                        else:
                            output_shape = list(output_info.shape)
                    except:
                        output_shape = "動的形状"
                    print(f"  - 出力: {output_name}, 形状: {output_shape}")
                
                # 変換前後の推論結果を比較
                print("\n変換前後の推論結果を比較しています...")

                # テスト入力の準備（固定のランダムシードで再現性を確保）
                np.random.seed(42)
                test_input = np.random.randn(1, 3, actual_input_size[0], actual_input_size[1]).astype(np.float32)

                # PyTorchモデルで推論
                print("  - PyTorchモデルで推論中...")
                with torch.no_grad():
                    torch_input = torch.from_numpy(test_input).to(device)
                    pytorch_output = model(torch_input).cpu().numpy()
                print(f"    PyTorch出力: 形状={pytorch_output.shape}, 範囲=[{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")

                # OpenVINOモデルで推論
                print("  - OpenVINOモデルで推論中...")
                compiled_model = ie.compile_model(network, "CPU")
                infer_request = compiled_model.create_infer_request()
                infer_request.infer({0: test_input})
                openvino_output = infer_request.get_output_tensor(0).data
                print(f"    OpenVINO出力: 形状={openvino_output.shape}, 範囲=[{openvino_output.min():.6f}, {openvino_output.max():.6f}]")

                # 結果の比較
                print("\n推論結果の比較:")
                diff = np.abs(pytorch_output - openvino_output)
                mse = np.mean((pytorch_output - openvino_output) ** 2)
                mae = np.mean(diff)
                max_diff = np.max(diff)

                print(f"  - 平均二乗誤差 (MSE): {mse:.10f}")
                print(f"  - 平均絶対誤差 (MAE): {mae:.10f}")
                print(f"  - 最大絶対誤差: {max_diff:.10f}")

                # 相対誤差も計算（ゼロ除算を避ける）
                pytorch_abs = np.abs(pytorch_output)
                nonzero_mask = pytorch_abs > 1e-10
                if np.any(nonzero_mask):
                    relative_diff = diff[nonzero_mask] / pytorch_abs[nonzero_mask]
                    mean_relative_error = np.mean(relative_diff) * 100
                    max_relative_error = np.max(relative_diff) * 100
                    print(f"  - 平均相対誤差: {mean_relative_error:.6f}%")
                    print(f"  - 最大相対誤差: {max_relative_error:.6f}%")

                # 精度の評価
                if max_diff < 1e-5:
                    print("\n✓ 変換結果: 非常に高精度（誤差 < 1e-5）")
                elif max_diff < 1e-4:
                    print("\n✓ 変換結果: 高精度（誤差 < 1e-4）")
                elif max_diff < 1e-3:
                    print("\n✓ 変換結果: 良好（誤差 < 1e-3）")
                else:
                    print(f"\n⚠ 変換結果: 誤差が大きい可能性があります（最大誤差: {max_diff:.6f}）")

                print("OpenVINO推論テストに成功しました")
                
            except Exception as e:
                print(f"OpenVINOモデルの検証中にエラーが発生しました: {e}")
            
            return xml_path
        else:
            print("エラー: OpenVINOモデルファイルが生成されませんでした")
            return None
        
    except Exception as e:
        print(f"変換中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
        # 一時ファイルのクリーンアップ
        temp_onnx_path = os.path.splitext(model_path)[0] + "_temp.onnx"
        if os.path.exists(temp_onnx_path):
            os.remove(temp_onnx_path)
        
        return None


def convert_yolo_to_openvino(model_path, output_path=None, input_size=640,
                             precision='FP32', compress_to_fp16=False, half=False):
    """
    YOLOモデル（Ultralytics）をOpenVINO形式に変換する

    Args:
        model_path: 変換するYOLOモデルのパス (.pt)
        output_path: 出力ディレクトリパス (Noneの場合は自動生成)
        input_size: 入力画像サイズ（デフォルト: 640）
        precision: モデルの精度 (FP32, FP16, INT8)
        compress_to_fp16: FP16に圧縮するかどうか
        half: FP16モードで変換するかどうか

    Returns:
        変換されたOpenVINOモデルのパス
    """
    # 出力パスが指定されていない場合、元のファイル名を基に自動生成
    if output_path is None:
        base_path = os.path.splitext(model_path)[0]
        output_path = f"{base_path}_openvino_model"

    print(f"YOLOモデル '{model_path}' をOpenVINO形式に変換しています...")

    try:
        # Ultralyticsのインポート
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Ultralyticsがインストールされていません。以下のコマンドでインストールしてください:")
            print("pip install ultralytics")
            return None

        # YOLOモデルの読み込み
        print("YOLOモデルを読み込んでいます...")
        model = YOLO(model_path)

        # モデル情報の表示
        print(f"\nモデル情報:")
        print(f"  - モデルタイプ: {model.model_name if hasattr(model, 'model_name') else 'YOLO'}")
        print(f"  - 入力サイズ: {input_size}")

        # テスト入力の準備（変換前の推論用）
        print("\n変換前のPyTorchモデルで推論を実行...")
        np.random.seed(42)
        test_input = np.random.rand(input_size, input_size, 3).astype(np.uint8)

        # PyTorchモデルで推論（時間計測）
        start_time = time.time()
        pytorch_results = model(test_input, verbose=False)
        pytorch_time = time.time() - start_time

        # PyTorch推論結果の取得
        pytorch_boxes = pytorch_results[0].boxes
        if pytorch_boxes is not None and len(pytorch_boxes) > 0:
            pytorch_detections = {
                'boxes': pytorch_boxes.xyxy.cpu().numpy() if pytorch_boxes.xyxy is not None else np.array([]),
                'scores': pytorch_boxes.conf.cpu().numpy() if pytorch_boxes.conf is not None else np.array([]),
                'classes': pytorch_boxes.cls.cpu().numpy() if pytorch_boxes.cls is not None else np.array([])
            }
            print(f"  - 検出数: {len(pytorch_detections['boxes'])}")
            print(f"  - 推論時間: {pytorch_time*1000:.2f}ms")
            if len(pytorch_detections['boxes']) > 0:
                print(f"  - 信頼度範囲: [{pytorch_detections['scores'].min():.4f}, {pytorch_detections['scores'].max():.4f}]")
        else:
            pytorch_detections = {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([])
            }
            print(f"  - 検出数: 0")
            print(f"  - 推論時間: {pytorch_time*1000:.2f}ms")

        # OpenVINOへの変換
        print("\nOpenVINO形式に変換しています...")
        export_args = {
            'format': 'openvino',
            'imgsz': input_size,
            'half': half or (precision == 'FP16') or compress_to_fp16,
            'int8': precision == 'INT8'
        }

        # エクスポート実行
        export_path = model.export(**export_args)

        # エクスポートされたモデルのパスを確認
        # Ultralyticsは通常 "model_name_openvino_model" というディレクトリを作成
        if os.path.isdir(export_path):
            openvino_dir = export_path
        else:
            # export_pathがファイルパスの場合、親ディレクトリを取得
            openvino_dir = os.path.dirname(export_path)

        # XMLファイルを探す
        xml_files = list(Path(openvino_dir).glob("*.xml"))
        if not xml_files:
            print("エラー: OpenVINOモデルファイル(.xml)が見つかりません")
            return None

        xml_path = str(xml_files[0])
        bin_path = xml_path.replace('.xml', '.bin')

        if not os.path.exists(bin_path):
            print("エラー: OpenVINOモデルファイル(.bin)が見つかりません")
            return None

        print(f"\n変換が完了しました！")
        print(f"  - XMLファイル: {xml_path}")
        print(f"  - BINファイル: {bin_path}")

        # ファイルサイズの表示
        model_size = os.path.getsize(model_path) / 1024 / 1024  # MB
        openvino_size = (os.path.getsize(xml_path) + os.path.getsize(bin_path)) / 1024 / 1024  # MB

        print(f"\nファイルサイズ:")
        print(f"  - 元のYOLOモデル: {model_size:.1f} MB")
        print(f"  - OpenVINOモデル: {openvino_size:.1f} MB")
        if model_size > 0:
            print(f"  - サイズ比率: {openvino_size/model_size*100:.1f}%")

        # OpenVINOモデルでの推論テスト
        try:
            from openvino.runtime import Core

            print("\n変換後のOpenVINOモデルで推論を実行...")
            ie = Core()

            # 利用可能なデバイスの表示
            available_devices = ie.available_devices
            print(f"利用可能なデバイス: {available_devices}")

            # OpenVINOモデルの読み込みと推論（ディレクトリパスを指定）
            openvino_model = YOLO(openvino_dir, task='detect')

            # OpenVINOモデルで推論（時間計測）
            start_time = time.time()
            openvino_results = openvino_model(test_input, verbose=False)
            openvino_time = time.time() - start_time

            # OpenVINO推論結果の取得
            openvino_boxes = openvino_results[0].boxes
            if openvino_boxes is not None and len(openvino_boxes) > 0:
                openvino_detections = {
                    'boxes': openvino_boxes.xyxy.cpu().numpy() if openvino_boxes.xyxy is not None else np.array([]),
                    'scores': openvino_boxes.conf.cpu().numpy() if openvino_boxes.conf is not None else np.array([]),
                    'classes': openvino_boxes.cls.cpu().numpy() if openvino_boxes.cls is not None else np.array([])
                }
                print(f"  - 検出数: {len(openvino_detections['boxes'])}")
                print(f"  - 推論時間: {openvino_time*1000:.2f}ms")
                print(f"  - 高速化: {pytorch_time/openvino_time:.2f}x")
                if len(openvino_detections['boxes']) > 0:
                    print(f"  - 信頼度範囲: [{openvino_detections['scores'].min():.4f}, {openvino_detections['scores'].max():.4f}]")
            else:
                openvino_detections = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'classes': np.array([])
                }
                print(f"  - 検出数: 0")
                print(f"  - 推論時間: {openvino_time*1000:.2f}ms")
                print(f"  - 高速化: {pytorch_time/openvino_time:.2f}x")

            # 検出結果の比較
            print("\n推論結果の比較:")
            pytorch_count = len(pytorch_detections['boxes'])
            openvino_count = len(openvino_detections['boxes'])

            print(f"  - PyTorch検出数: {pytorch_count}")
            print(f"  - OpenVINO検出数: {openvino_count}")

            if pytorch_count > 0 and openvino_count > 0:
                # 検出数が同じ場合は詳細比較
                if pytorch_count == openvino_count:
                    # バウンディングボックスの誤差
                    box_diff = np.abs(pytorch_detections['boxes'] - openvino_detections['boxes'])
                    max_box_diff = np.max(box_diff)
                    mean_box_diff = np.mean(box_diff)

                    # 信頼度の誤差
                    score_diff = np.abs(pytorch_detections['scores'] - openvino_detections['scores'])
                    max_score_diff = np.max(score_diff)
                    mean_score_diff = np.mean(score_diff)

                    # クラスの一致率
                    class_match = np.sum(pytorch_detections['classes'] == openvino_detections['classes']) / pytorch_count * 100

                    print(f"  - バウンディングボックス平均誤差: {mean_box_diff:.4f} ピクセル")
                    print(f"  - バウンディングボックス最大誤差: {max_box_diff:.4f} ピクセル")
                    print(f"  - 信頼度平均誤差: {mean_score_diff:.6f}")
                    print(f"  - 信頼度最大誤差: {max_score_diff:.6f}")
                    print(f"  - クラス一致率: {class_match:.1f}%")

                    # 精度の評価
                    if max_box_diff < 1.0 and max_score_diff < 0.01 and class_match == 100:
                        print("\n✓ 変換結果: 非常に高精度")
                    elif max_box_diff < 5.0 and max_score_diff < 0.05 and class_match >= 90:
                        print("\n✓ 変換結果: 高精度")
                    elif max_box_diff < 10.0 and max_score_diff < 0.1 and class_match >= 80:
                        print("\n✓ 変換結果: 良好")
                    else:
                        print(f"\n⚠ 変換結果: 誤差が大きい可能性があります")
                else:
                    diff_count = abs(pytorch_count - openvino_count)
                    diff_percent = diff_count / max(pytorch_count, openvino_count) * 100
                    print(f"  - 検出数の差: {diff_count} ({diff_percent:.1f}%)")

                    if diff_percent < 10:
                        print("\n✓ 変換結果: 良好（検出数の差は小さい）")
                    else:
                        print(f"\n⚠ 検出数に差があります（{diff_percent:.1f}%）")
            elif pytorch_count == 0 and openvino_count == 0:
                print("\n✓ 変換結果: 一致（両方とも検出なし）")
            else:
                print("\n⚠ 検出結果が大きく異なります")

            print("\nOpenVINO推論テストに成功しました")

        except Exception as e:
            print(f"OpenVINOモデルの検証中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

        return xml_path

    except Exception as e:
        print(f"変換中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='PyTorch、YOLO、またはONNXモデルをOpenVINO形式に変換するスクリプト')
    parser.add_argument('--model_path', type=str, required=True, help='変換するモデルのパス（.pth, .pt, .onnx）')
    parser.add_argument('--model_type', type=str, help='モデルタイプ (例: resnet18, yolo) - PyTorchモデルの場合は必須')
    parser.add_argument('--output_path', type=str, default=None, help='出力OpenVINOモデルのパス（省略可）')
    parser.add_argument('--width', type=int, default=224, help='入力画像の幅（PyTorchモデルの場合）')
    parser.add_argument('--height', type=int, default=224, help='入力画像の高さ（PyTorchモデルの場合）')
    parser.add_argument('--input_size', type=int, default=640, help='入力画像サイズ（YOLOモデルの場合）')
    parser.add_argument('--precision', type=str, default='FP32', choices=['FP32', 'FP16', 'INT8'], help='モデルの精度')
    parser.add_argument('--compress_to_fp16', action='store_true', help='FP32モデルをFP16に圧縮')

    args = parser.parse_args()

    # ファイル拡張子を確認
    file_ext = os.path.splitext(args.model_path)[1].lower()

    # パス全体からYOLOモデルかどうかを判定（ファイル名だけでなくディレクトリ名も含む）
    model_path_lower = args.model_path.lower()
    is_yolo = ('yolo' in model_path_lower or
               (args.model_type and 'yolo' in args.model_type.lower()))

    if file_ext == '.onnx':
        # ONNXモデルをOpenVINO形式に変換
        print("ONNXモデルを検出しました。")
        convert_onnx_to_openvino(
            onnx_path=args.model_path,
            output_path=args.output_path,
            precision=args.precision,
            compress_to_fp16=args.compress_to_fp16
        )
    elif file_ext in ['.pth', '.pt']:
        if is_yolo:
            # YOLOモデルをOpenVINO形式に変換
            print("YOLOモデルを検出しました。")
            convert_yolo_to_openvino(
                model_path=args.model_path,
                output_path=args.output_path,
                input_size=args.input_size,
                precision=args.precision,
                compress_to_fp16=args.compress_to_fp16
            )
        else:
            # PyTorchモデルをOpenVINO形式に変換
            print("PyTorchモデルを検出しました。")
            if not args.model_type:
                print("エラー: PyTorchモデルの場合は --model_type が必須です。")
                print("YOLOモデルの場合は --model_type yolo を指定するか、ファイル名に 'yolo' を含めてください。")
                parser.print_help()
                return

            convert_pytorch_to_openvino(
                model_path=args.model_path,
                model_type=args.model_type,
                output_path=args.output_path,
                input_size=(args.height, args.width),
                precision=args.precision,
                compress_to_fp16=args.compress_to_fp16
            )
    else:
        print(f"エラー: サポートされていないファイル形式です: {file_ext}")
        print("サポートされている形式: .pth, .pt, .onnx")


if __name__ == "__main__":
    main()