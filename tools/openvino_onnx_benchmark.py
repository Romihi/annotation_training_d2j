import time
import numpy as np
import onnxruntime as ort
from openvino import Core
import argparse
import statistics
from pathlib import Path

class ModelBenchmark:
    def __init__(self, model_path, input_shape=(1, 3, 224, 224)):
        self.model_path = Path(model_path)
        self.input_shape = input_shape
        self.input_data = np.random.randn(*input_shape).astype(np.float32)
        
    def benchmark_onnx(self, num_runs=100, warmup_runs=10):
        """ONNXランタイムでの推論速度を測定"""
        print(f"ONNX Runtime ベンチマーク開始: {self.model_path}")
        
        # セッション作成
        session = ort.InferenceSession(str(self.model_path))
        input_name = session.get_inputs()[0].name
        
        # ウォームアップ
        print("ウォームアップ中...")
        for _ in range(warmup_runs):
            session.run(None, {input_name: self.input_data})
        
        # 実際の測定
        print(f"推論時間測定中... ({num_runs}回)")
        times = []
        for i in range(num_runs):
            start_time = time.time()
            outputs = session.run(None, {input_name: self.input_data})
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ミリ秒に変換
            
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{num_runs} 完了")
        
        return times
    
    def benchmark_openvino(self, num_runs=100, warmup_runs=10):
        """OpenVINOでの推論速度を測定"""
        print(f"OpenVINO ベンチマーク開始: {self.model_path}")
        
        # Core作成とモデル読み込み
        core = Core()
        model = core.read_model(str(self.model_path))
        compiled_model = core.compile_model(model, "CPU")
        
        # 入力/出力情報取得
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        
        # ウォームアップ
        print("ウォームアップ中...")
        for _ in range(warmup_runs):
            compiled_model([self.input_data])
        
        # 実際の測定
        print(f"推論時間測定中... ({num_runs}回)")
        times = []
        for i in range(num_runs):
            start_time = time.time()
            outputs = compiled_model([self.input_data])
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ミリ秒に変換
            
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{num_runs} 完了")
        
        return times
    
    def convert_onnx_to_openvino(self, output_path):
        """ONNXモデルをOpenVINO IRフォーマットに変換"""
        try:
            from openvino import convert_model, save_model
            print(f"ONNXモデルをOpenVINO IRに変換中...")
            
            # 新しいAPIで変換
            ov_model = convert_model(str(self.model_path))
            
            # IRフォーマットで保存
            output_xml = output_path.with_suffix('.xml')
            save_model(ov_model, str(output_xml))
            
            print(f"変換完了: {output_xml}")
            return output_xml
            
        except ImportError:
            # 古いAPIを試す
            try:
                import subprocess
                import sys
                print("コマンドラインツールを使用してONNXをOpenVINOに変換中...")
                
                cmd = [
                    sys.executable, "-m", "openvino.tools.mo",
                    "--input_model", str(self.model_path),
                    "--output_dir", str(output_path.parent),
                    "--model_name", output_path.stem
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return output_path.with_suffix('.xml')
                else:
                    print(f"変換エラー: {result.stderr}")
                    return None
                    
            except Exception as e2:
                print(f"コマンドライン変換エラー: {e2}")
                return None
                
        except Exception as e:
            print(f"変換エラー: {e}")
            return None
    
    def print_results(self, onnx_times, openvino_times):
        """結果を表示"""
        print("\n" + "="*60)
        print("ベンチマーク結果")
        print("="*60)
        
        # ONNX結果
        onnx_mean = statistics.mean(onnx_times)
        onnx_std = statistics.stdev(onnx_times)
        onnx_min = min(onnx_times)
        onnx_max = max(onnx_times)
        
        print(f"ONNX Runtime:")
        print(f"  平均: {onnx_mean:.2f} ms")
        print(f"  標準偏差: {onnx_std:.2f} ms")
        print(f"  最小: {onnx_min:.2f} ms")
        print(f"  最大: {onnx_max:.2f} ms")
        print(f"  FPS: {1000/onnx_mean:.1f}")
        
        # OpenVINO結果
        openvino_mean = statistics.mean(openvino_times)
        openvino_std = statistics.stdev(openvino_times)
        openvino_min = min(openvino_times)
        openvino_max = max(openvino_times)
        
        print(f"\nOpenVINO:")
        print(f"  平均: {openvino_mean:.2f} ms")
        print(f"  標準偏差: {openvino_std:.2f} ms")
        print(f"  最小: {openvino_min:.2f} ms")
        print(f"  最大: {openvino_max:.2f} ms")
        print(f"  FPS: {1000/openvino_mean:.1f}")
        
        # 比較
        speedup = onnx_mean / openvino_mean
        print(f"\n速度比較:")
        if speedup > 1:
            print(f"  OpenVINOはONNXより {speedup:.2f}x 高速")
        else:
            print(f"  ONNXはOpenVINOより {1/speedup:.2f}x 高速")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='OpenVINO vs ONNX 速度比較')
    parser.add_argument('--onnx_model', required=True, help='ONNXモデルファイルのパス')
    parser.add_argument('--openvino_model', help='OpenVINO IRモデルファイルのパス（.xml）')
    parser.add_argument('--runs', type=int, default=100, help='推論回数（デフォルト: 100）')
    parser.add_argument('--warmup', type=int, default=10, help='ウォームアップ回数（デフォルト: 10）')
    parser.add_argument('--input_shape', nargs=4, type=int, default=[1, 3, 224, 224], 
                       help='入力形状 [batch, channel, height, width]')
    
    args = parser.parse_args()
    
    # ONNXモデルのベンチマーク
    onnx_benchmark = ModelBenchmark(args.onnx_model, tuple(args.input_shape))
    onnx_times = onnx_benchmark.benchmark_onnx(args.runs, args.warmup)
    
    # OpenVINOモデルの準備
    if args.openvino_model:
        openvino_model_path = Path(args.openvino_model)
    else:
        # ONNXからOpenVINOに変換
        openvino_model_path = Path(args.onnx_model).with_suffix('.xml')
        converted_path = onnx_benchmark.convert_onnx_to_openvino(openvino_model_path)
        if not converted_path:
            print("OpenVINOモデルの変換に失敗しました")
            return
        openvino_model_path = converted_path
    
    # OpenVINOモデルのベンチマーク
    openvino_benchmark = ModelBenchmark(openvino_model_path, tuple(args.input_shape))
    openvino_times = openvino_benchmark.benchmark_openvino(args.runs, args.warmup)
    
    # 結果表示
    onnx_benchmark.print_results(onnx_times, openvino_times)

if __name__ == "__main__":
    main()
