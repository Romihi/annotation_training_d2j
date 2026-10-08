#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch → ONNX → TensorRT 変換ツール

変換フロー:
  1. PyTorch (.pth) → ONNX (.onnx)
  2. ONNX → TensorRT エンジン (.trt)  ← tensorrt がインストールされている場合のみ

TensorRT のインストール:
  CUDA 12.x: pip install tensorrt==10.x.x  (NVIDIA公式ページ参照)
  または NVIDIA TensorRT SDK から whl を直接インストール

MultiSourceModel (マルチカメラ・attention 融合) にも対応。

TogiVAD 系（togivad_*.pth。Tier1/ベクトル化/Fusion/Pilot の全フラグ構成）にも対応:
  - ONNX 化は togivad/export_onnx.py に委譲（入出力名 = config.output_spec）
  - trt サブコマンドの出力は TogiVADRuntime の規約に合わせ **.engine** 拡張子で、
    語彙 sidecar <name>_vocab.npy も同ディレクトリに生成される
  例: python tools/torch2trt_converter.py trt models/togivad_xxx.pth
"""

import os
import sys
import argparse
import re
import glob as glob_module
import time
import io

# Windows コンソール (cp932) で絵文字を含む torch.onnx ログが UnicodeEncodeError になるのを防ぐ
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TOOLS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# togivad パッケージはリポジトリ直下
_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# TogiVAD 対応
# ---------------------------------------------------------------------------

def _is_togivad_ckpt(model_path: str, quick: bool = False) -> bool:
    """TogiVAD 系チェックポイントかを判定する。

    ファイル名 `togivad` プレフィックス（アノテーションツールの保存規約）を優先。
    quick=False ならチェックポイントのキー構成（state_dict + config + vocab =
    togivad/train.py 互換形式）でも判定する（list 等の一括処理は quick=True）。
    """
    if os.path.basename(model_path).lower().startswith('togivad'):
        return True
    if quick:
        return False
    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        return isinstance(ckpt, dict) and \
            {'state_dict', 'config', 'vocab'} <= set(ckpt.keys())
    except Exception:
        return False


def _load_togivad(model_path: str):
    """TogiVAD チェックポイント → (model(eval,cpu), TogiVADConfig)。"""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from togivad.config import TogiVADConfig
    from togivad.model.net import TogiVADNano
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = TogiVADConfig.from_dict(ckpt['config'])
    model = TogiVADNano(cfg, vocab=ckpt['vocab'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, cfg


def _togivad_flags(cfg) -> str:
    flags = [n for n in ('use_residual', 'use_temporal', 'use_map',
                         'use_agent_motion', 'use_lidar', 'use_control')
             if getattr(cfg, n, False)]
    return ', '.join(flags) if flags else 'なし（基本構成）'


def export_togivad_to_onnx(model_path: str, output_path: str = None) -> str:
    """TogiVAD を ONNX に変換する（togivad.export_onnx に委譲）。

    入出力名・順序は config.output_spec が単一情報源（use_lidar/use_temporal の
    追加入力、residual/map/agent/control/bev_state の追加出力を自動で含む）。
    語彙 sidecar <base>_vocab.npy も同時に生成される（.engine 読込規約と同名）。
    """
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from togivad.export_onnx import export as togivad_export

    if output_path is None:
        output_path = os.path.splitext(model_path)[0] + '.onnx'

    model, cfg = _load_togivad(model_path)
    print(f"\n[ONNX 変換 (TogiVAD)]")
    print(f"  入力モデル : {model_path}")
    print(f"  出力ONNX   : {output_path}")
    print(f"  構成       : {cfg.num_cameras}cam × {cfg.image_h}x{cfg.image_w}, "
          f"ego_dim={cfg.ego_dim}, vocab_k={cfg.vocab_k}, horizon={cfg.horizon}")
    print(f"  拡張フラグ : {_togivad_flags(cfg)}")

    # nn.MultiheadAttention fast-path を OFF（一般 ops への展開用）
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass
    with torch.no_grad():
        togivad_export(model, cfg, output_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ONNX 変換成功: {size_mb:.2f} MB")
    print(f"  語彙 sidecar : {output_path.replace('.onnx', '_vocab.npy')}")
    return output_path


# ---------------------------------------------------------------------------
# MultiSourceModel 用 ONNX ラッパー
# ---------------------------------------------------------------------------

class _MultiSourceONNXWrapper(nn.Module):
    """
    MultiSourceModel は [B, num_sources*3, H, W] の連結テンソルを受け取るが、
    ONNX 推論側では各カメラを個別テンソルとして渡したい場合のラッパー。
    *args (cam0[B,3,H,W], cam1[B,3,H,W], ...) を channel 方向に連結して渡す。
    """
    def __init__(self, model: nn.Module, num_sources: int):
        super().__init__()
        self.model = model
        self.num_sources = num_sources

    def forward(self, *args):
        # 各カメラ画像をチャネル方向に連結: [B, N*3, H, W]
        x = torch.cat(args, dim=1)
        return self.model(x)


# ---------------------------------------------------------------------------
# ユーティリティ関数
# ---------------------------------------------------------------------------

def find_pytorch_models(search_dir: str = None) -> list:
    """指定ディレクトリ以下の .pth ファイルを列挙する"""
    if search_dir is None:
        search_dir = os.path.join(_PROJECT_ROOT, 'models')
    pattern = os.path.join(search_dir, '**', '*.pth')
    return sorted(glob_module.glob(pattern, recursive=True), reverse=True)


def get_available_model_types() -> list:
    """model_catalog から利用可能なモデルタイプ一覧を返す"""
    try:
        from model_catalog import list_available_models
        return list_available_models()
    except ImportError:
        return []


def infer_model_type_from_filename(filename: str) -> tuple:
    """
    ファイル名からモデルタイプ・num_sources・fusion_method を推定する。

    例:
      multi3_attention_mobilenetv3_small_100_xxx.pth
        → ('mobilenetv3_small_100', 3, 'attention')
      mobilenetv3_small_100_xxx.pth
        → ('mobilenetv3_small_100', 1, None)
    """
    name = os.path.basename(filename)

    # マルチソースパターン: multi{N}_{fusion}_{base}
    m = re.match(r'multi(\d+)_(\w+?)_(.*?)(?:_\d{8}|\.pth)', name)
    if m:
        num_sources = int(m.group(1))
        fusion = m.group(2)
        base = m.group(3)
        return base, num_sources, fusion

    # シングルソースパターン
    available = get_available_model_types()
    for mt in sorted(available, key=len, reverse=True):
        if mt in name:
            return mt, 1, None

    return None, 1, None


def load_model_weights(model_path: str, device: torch.device = None):
    """
    チェックポイントをロードしてメタ情報を返す。

    Returns:
        dict: {
            'model_state_dict': OrderedDict,
            'num_outputs': int,
            'input_size': tuple,
            'num_sources': int,
            'fusion_method': str | None,
            'base_model_name': str | None,
        }
    """
    if device is None:
        device = torch.device('cpu')

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        sd = ckpt.get('model_state_dict', ckpt)
    else:
        sd = ckpt
        ckpt = {}

    # 出力数の検出
    if 'regressor.bias' in sd:
        num_outputs = sd['regressor.bias'].shape[0]
    elif 'regressor.3.bias' in sd:
        num_outputs = sd['regressor.3.bias'].shape[0]
    elif 'regressor.0.bias' in sd:
        num_outputs = sd['regressor.0.bias'].shape[0]
    else:
        num_outputs = 2

    return {
        'model_state_dict': sd,
        'num_outputs': num_outputs,
        'input_size': tuple(ckpt.get('input_size', (224, 224))),
        'num_sources': ckpt.get('num_sources', 1),
        'fusion_method': ckpt.get('fusion_method', None),
        'base_model_name': ckpt.get('base_model_name', None),
    }


def _build_model(meta: dict, model_type: str, device: torch.device) -> nn.Module:
    """メタ情報からモデルを構築して重みをロードする"""
    from model_catalog import get_model, MultiSourceModel

    num_sources = meta['num_sources']
    fusion = meta['fusion_method']
    input_size = meta['input_size']
    num_outputs = meta['num_outputs']
    base_name = meta['base_model_name'] or model_type

    if num_sources > 1:
        model = MultiSourceModel(
            base_model_name=base_name,
            num_sources=num_sources,
            fusion_method=fusion or 'concat',
            num_outputs=num_outputs,
            input_size=input_size,
        )
    else:
        model = get_model(model_type, pretrained=False,
                          input_size=input_size, num_outputs=num_outputs)

    model.load_state_dict(meta['model_state_dict'])
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# ONNX エクスポート
# ---------------------------------------------------------------------------

def export_to_onnx(model_path: str, model_type: str,
                   output_path: str = None,
                   opset: int = 17,
                   dynamic_batch: bool = False) -> str:
    """
    PyTorch モデルを ONNX に変換する。

    Returns:
        str: 出力 ONNX ファイルパス (失敗時は None)
    """
    if output_path is None:
        output_path = os.path.splitext(model_path)[0] + '.onnx'

    print(f"\n[ONNX 変換]")
    print(f"  入力モデル : {model_path}")
    print(f"  出力ONNX   : {output_path}")

    device = torch.device('cpu')

    meta = load_model_weights(model_path, device)
    print(f"  num_outputs: {meta['num_outputs']}")
    print(f"  input_size : {meta['input_size']}")
    print(f"  num_sources: {meta['num_sources']}")
    print(f"  fusion     : {meta['fusion_method']}")

    model = _build_model(meta, model_type, device)

    H, W = meta['input_size']
    num_sources = meta['num_sources']

    if num_sources > 1:
        # MultiSourceModel: 各カメラを個別テンソルとして受け取るラッパー
        # ラッパー内で channel 連結して本体モデルに渡す
        wrapper = _MultiSourceONNXWrapper(model, num_sources)
        dummy_inputs = tuple(torch.randn(1, 3, H, W) for _ in range(num_sources))
        input_names = [f'cam{i}' for i in range(num_sources)]
        print(f"  ダミー入力 : {num_sources}×(1,3,{H},{W})")

        # MultiheadAttention は dynamo エクスポーターでのみ正しく変換できる
        # (レガシー TorchScript エクスポーターは _native_multi_head_attention 非対応)
        onnx_prog = torch.onnx.export(
            wrapper,
            dummy_inputs,
            dynamo=True,
            input_names=input_names,
            output_names=['output'],
        )
        onnx_prog.save(output_path)
    else:
        # シングルソース: 単一テンソル入力 (レガシーエクスポーターで十分)
        dummy = (torch.randn(1, 3, H, W),)
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}

        print(f"  ダミー入力 : (1,3,{H},{W})")
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                output_path,
                opset_version=opset,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ONNX 変換成功: {size_mb:.2f} MB")
    return output_path


# ---------------------------------------------------------------------------
# TensorRT 変換
# ---------------------------------------------------------------------------

def convert_onnx_to_trt(onnx_path: str,
                         output_path: str = None,
                         precision: str = 'FP16',
                         workspace_gb: int = 4,
                         num_sources: int = 1) -> str:
    """
    ONNX → TensorRT エンジン変換。

    Returns:
        str: .trt エンジンファイルパス (失敗/未インストール時は None)
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("\n[TensorRT 未インストール]")
        print("  TensorRT が見つかりません。以下のコマンドでインストールしてください:")
        print("  pip install tensorrt  (CUDA バージョンに合ったものを選択)")
        print("  または NVIDIA Developer サイトから TRT SDK をダウンロード")
        return None

    if output_path is None:
        suffix = '_fp16' if precision == 'FP16' else '_fp32'
        output_path = os.path.splitext(onnx_path)[0] + suffix + '.trt'

    print(f"\n[TensorRT 変換]")
    print(f"  ONNX      : {onnx_path}")
    print(f"  出力      : {output_path}")
    print(f"  精度      : {precision}")
    print(f"  ワークスペース: {workspace_gb} GB")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX パースエラー: {parser.get_error(i)}")
            return None
    print("  ONNX パース完了")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                  workspace_gb * (1 << 30))

    if precision == 'FP16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 モードを有効化")
    elif precision == 'INT8':
        config.set_flag(trt.BuilderFlag.INT8)
        print("  INT8 モードを有効化 (キャリブレーションなし - 精度低下の可能性あり)")

    print("  エンジンビルド中 (数分かかることがあります)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  エンジンビルド失敗")
        return None

    with open(output_path, 'wb') as f:
        f.write(serialized)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  TRT 変換成功: {size_mb:.2f} MB  ({elapsed:.1f}秒)")
    return output_path


def convert_pytorch_to_tensorrt(model_path: str,
                                 model_type: str,
                                 output_path: str = None,
                                 input_size: tuple = (224, 224),
                                 precision: str = 'FP16',
                                 workspace_gb: int = 4,
                                 opset: int = 17,
                                 keep_onnx: bool = False) -> str:
    """
    PyTorch → ONNX → TRT の一括変換。

    Returns:
        str: TRT エンジンパス (TRT 未インストール時は ONNX パス)
    """
    # 1. ONNX 変換
    if output_path:
        onnx_path = os.path.splitext(output_path)[0] + '.onnx'
    else:
        onnx_path = None

    onnx_result = export_to_onnx(model_path, model_type,
                                  output_path=onnx_path, opset=opset)
    if not onnx_result:
        return None

    # 2. TRT 変換
    meta = load_model_weights(model_path)
    trt_out = os.path.splitext(onnx_result)[0]
    trt_out += '_fp16.trt' if precision == 'FP16' else '_fp32.trt'

    trt_result = convert_onnx_to_trt(
        onnx_result,
        output_path=trt_out,
        precision=precision,
        workspace_gb=workspace_gb,
        num_sources=meta['num_sources'],
    )

    if trt_result and not keep_onnx:
        os.remove(onnx_result)
        print(f"  中間 ONNX を削除: {onnx_result}")

    return trt_result or onnx_result


def convert_sequence_to_tensorrt(model_path: str, output_path: str = None,
                                   precision: str = 'FP16') -> str:
    """シーケンス/GRU モデルの TRT 変換（将来拡張用スタブ）"""
    print("警告: シーケンスモデルの TRT 変換は未実装です。ONNX 変換のみ実行します。")
    return export_to_onnx(model_path, model_type='gru', output_path=output_path)


# ---------------------------------------------------------------------------
# ベンチマーク
# ---------------------------------------------------------------------------

def benchmark_inference(model_path: str = None,
                         engine_path: str = None,
                         model=None,
                         input_size: tuple = (224, 224),
                         num_sources: int = 1,
                         device: str = 'cuda',
                         num_warmup: int = 10,
                         num_iterations: int = 100) -> dict:
    """
    PyTorch モデルまたは TRT エンジンの推論速度をベンチマーク。

    Returns:
        dict: {'mean_ms', 'std_ms', 'fps'}
    """
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    H, W = input_size

    results = {}

    # --- PyTorch ベンチマーク ---
    if model_path or model is not None:
        fn = None
        if model is None and model_path and _is_togivad_ckpt(model_path):
            # TogiVAD: cfg のフラグ構成に応じた複数入力（images/ego + 追加入力）
            model, tv_cfg = _load_togivad(model_path)
            model = model.to(dev)
            imgs = torch.randn(1, tv_cfg.num_cameras, 3, tv_cfg.image_h,
                               tv_cfg.image_w, device=dev)
            ego = torch.zeros(1, tv_cfg.ego_dim, device=dev)
            kw = {}
            if getattr(tv_cfg, 'use_lidar', False):
                kw['lidar_bev'] = torch.zeros(
                    1, 1, tv_cfg.bev_size, tv_cfg.bev_size, device=dev)
            if getattr(tv_cfg, 'use_temporal', False):
                kw['prev_bev'] = torch.zeros(
                    1, tv_cfg.bev_dim, tv_cfg.bev_size, tv_cfg.bev_size,
                    device=dev)
                kw['ego_dpose'] = torch.zeros(1, 3, device=dev)
            fn = lambda: model(imgs, ego, **kw)
        elif model is None:
            meta = load_model_weights(model_path, dev)
            mt, ns, fm = infer_model_type_from_filename(model_path)
            model = _build_model(meta, mt or 'mobilenetv3_small_100', dev)
            num_sources = meta['num_sources']

        if fn is None:
            if num_sources > 1:
                dummy = [torch.randn(1, 3, H, W, device=dev) for _ in range(num_sources)]
                fn = lambda: model(dummy)
            else:
                dummy = torch.randn(1, 3, H, W, device=dev)
                fn = lambda: model(dummy)

        with torch.no_grad():
            for _ in range(num_warmup):
                fn()
            if dev.type == 'cuda':
                torch.cuda.synchronize()

            times = []
            for _ in range(num_iterations):
                t0 = time.perf_counter()
                fn()
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        results['pytorch'] = {
            'mean_ms': mean_ms,
            'std_ms': std_ms,
            'fps': 1000.0 / mean_ms,
        }
        print(f"[PyTorch]  {mean_ms:.2f} ± {std_ms:.2f} ms  ({1000/mean_ms:.1f} FPS)")

    # --- TRT エンジンベンチマーク ---
    if engine_path:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()

            # 入出力バッファ準備
            bindings = []
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = engine.get_tensor_shape(name)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                buf = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dtype).itemsize)
                bindings.append(int(buf))

            for _ in range(num_warmup):
                context.execute_v2(bindings)
            cuda.Context.synchronize()

            times = []
            for _ in range(num_iterations):
                t0 = time.perf_counter()
                context.execute_v2(bindings)
                cuda.Context.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

            mean_ms = float(np.mean(times))
            std_ms = float(np.std(times))
            results['tensorrt'] = {
                'mean_ms': mean_ms,
                'std_ms': std_ms,
                'fps': 1000.0 / mean_ms,
            }
            print(f"[TensorRT] {mean_ms:.2f} ± {std_ms:.2f} ms  ({1000/mean_ms:.1f} FPS)")

        except ImportError:
            print("TensorRT/PyCUDA 未インストールのため TRT ベンチマークをスキップ")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description='PyTorch → ONNX → TensorRT 変換ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # ONNX のみ
  python tools/torch2trt_converter.py onnx models/mobilenetv3_small_100_xxx.pth

  # ONNX + TRT (TRT インストール済みの場合)
  python tools/torch2trt_converter.py trt models/mobilenetv3_small_100_xxx.pth --precision FP16

  # マルチソースモデル (モデルタイプ指定不要 - チェックポイントから自動検出)
  python tools/torch2trt_converter.py trt models/multi3_attention_mobilenetv3_small_100_xxx.pth

  # ベンチマーク
  python tools/torch2trt_converter.py bench --model models/xxx.pth

  # モデル一覧
  python tools/torch2trt_converter.py list
""")
    sub = p.add_subparsers(dest='cmd', required=True)

    # onnx サブコマンド
    p_onnx = sub.add_parser('onnx', help='PyTorch → ONNX 変換')
    p_onnx.add_argument('model_path', help='.pth ファイルパス')
    p_onnx.add_argument('--model-type', default=None,
                        help='モデルタイプ (省略時はファイル名から自動推定)')
    p_onnx.add_argument('--output', default=None, help='出力 ONNX パス')
    p_onnx.add_argument('--opset', type=int, default=17)
    p_onnx.add_argument('--dynamic-batch', action='store_true')

    # trt サブコマンド
    p_trt = sub.add_parser('trt', help='PyTorch → ONNX → TRT 変換')
    p_trt.add_argument('model_path', help='.pth ファイルパス')
    p_trt.add_argument('--model-type', default=None)
    p_trt.add_argument('--output', default=None, help='出力ベースパス')
    p_trt.add_argument('--precision', choices=['FP32', 'FP16', 'INT8'], default='FP16')
    p_trt.add_argument('--workspace', type=int, default=4, help='ワークスペース (GB)')
    p_trt.add_argument('--opset', type=int, default=17)
    p_trt.add_argument('--keep-onnx', action='store_true', help='中間 ONNX を保持')

    # bench サブコマンド
    p_bench = sub.add_parser('bench', help='推論速度ベンチマーク')
    p_bench.add_argument('--model', default=None, help='.pth パス')
    p_bench.add_argument('--engine', default=None, help='.trt エンジンパス')
    p_bench.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p_bench.add_argument('--iterations', type=int, default=100)

    # list サブコマンド
    sub.add_parser('list', help='models/ 以下の .pth ファイルを一覧表示')

    return p.parse_args()


def main():
    args = _parse_args()

    if args.cmd == 'list':
        models = find_pytorch_models()
        if not models:
            print("モデルファイルが見つかりません")
            return
        print(f"\n{len(models)} 件のモデルファイル:")
        for p in models:
            if _is_togivad_ckpt(p, quick=True):
                tag = "togivad"
            else:
                mt, ns, fm = infer_model_type_from_filename(p)
                tag = f"multi{ns}_{fm}" if ns > 1 else "single"
            print(f"  [{tag:20s}] {os.path.relpath(p)}")

    elif args.cmd == 'onnx':
        if _is_togivad_ckpt(args.model_path):
            export_togivad_to_onnx(args.model_path, output_path=args.output)
            return
        mt = args.model_type
        if mt is None:
            mt, ns, fm = infer_model_type_from_filename(args.model_path)
            print(f"モデルタイプを自動推定: {mt} (sources={ns}, fusion={fm})")
        export_to_onnx(args.model_path, mt or '',
                       output_path=args.output,
                       opset=args.opset,
                       dynamic_batch=args.dynamic_batch)

    elif args.cmd == 'trt':
        if _is_togivad_ckpt(args.model_path):
            # TogiVAD: ONNX → engine（TogiVADRuntime の .engine 規約で出力。
            # 語彙 sidecar <base>_vocab.npy は export が生成済み＝engine と同名）
            onnx_path = (os.path.splitext(args.output)[0] + '.onnx'
                         if args.output else
                         os.path.splitext(args.model_path)[0] + '.onnx')
            onnx_result = export_togivad_to_onnx(args.model_path, onnx_path)
            if not onnx_result:
                return
            engine_out = os.path.splitext(onnx_result)[0] + '.engine'
            trt_result = convert_onnx_to_trt(
                onnx_result, output_path=engine_out,
                precision=args.precision, workspace_gb=args.workspace)
            if trt_result and not args.keep_onnx:
                os.remove(onnx_result)
                print(f"  中間 ONNX を削除: {onnx_result}")
            if trt_result:
                print(f"\nTogiVADRuntime でそのまま読めます: "
                      f"TogiVADRuntime('{trt_result}')")
            return
        mt = args.model_type
        if mt is None:
            mt, ns, fm = infer_model_type_from_filename(args.model_path)
            print(f"モデルタイプを自動推定: {mt} (sources={ns}, fusion={fm})")
        convert_pytorch_to_tensorrt(
            args.model_path, mt or '',
            output_path=args.output,
            precision=args.precision,
            workspace_gb=args.workspace,
            opset=args.opset,
            keep_onnx=args.keep_onnx,
        )

    elif args.cmd == 'bench':
        benchmark_inference(
            model_path=args.model,
            engine_path=args.engine,
            device=args.device,
            num_iterations=args.iterations,
        )


if __name__ == '__main__':
    main()
