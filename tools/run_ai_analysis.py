"""MLflow実験データをLLMに分析させるCLI。

ローカルLLM(Ollama等)またはクラウド(Claude)で、指定実験タイプのRun群を分析する。

使い方:
    # ローカルLLM(Ollama, 既定)で自動運転モデルの実験を分析
    python tools/run_ai_analysis.py --type autonomous --model qwen2.5:7b-instruct

    # LM Studio を使う
    python tools/run_ai_analysis.py --type sequence --base-url lmstudio --model <model>

    # クラウドClaude（要 ANTHROPIC_API_KEY）
    python tools/run_ai_analysis.py --type autonomous --backend anthropic

    # 結果をMLflowに記録しない（標準出力のみ）
    python tools/run_ai_analysis.py --type autonomous --no-log
"""
import argparse
import os
import sys

# プロジェクトルートを import パスに追加
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from managers import MLflowManager, ModelType
from managers.mlflow_ai_analyzer import MLflowAIAnalyzer
from config import mlflow_dir

TYPE_MAP = {
    "autonomous": ModelType.AUTONOMOUS_DRIVING,
    "position": ModelType.POSITION_ESTIMATION,
    "waypoint": ModelType.WAYPOINT_REGRESSION,
    "yolo": ModelType.YOLO_DETECTION,
    "yolo_seg": ModelType.YOLO_SEGMENTATION,
    "sequence": ModelType.SEQUENCE,
}


def main():
    p = argparse.ArgumentParser(description="MLflow実験をLLMで分析")
    p.add_argument("--type", choices=TYPE_MAP.keys(), default="autonomous",
                   help="分析する実験タイプ")
    p.add_argument("--backend", choices=["local", "anthropic"], default="local",
                   help="LLMバックエンド（local=Ollama等 / anthropic=Claude）")
    p.add_argument("--base-url", default="ollama",
                   help="localバックエンドのエンドポイント（ollama/lmstudio/vllm/llamacpp または完全URL）")
    p.add_argument("--model", default=None, help="モデルID（未指定はバックエンド既定）")
    p.add_argument("--max-runs", type=int, default=40, help="分析対象の最大Run数")
    p.add_argument("--dir", default=None, help="MLflowトラッキングのベースディレクトリ（既定 config.mlflow_dir）")
    p.add_argument("--no-log", action="store_true", help="分析結果をMLflowに記録しない")
    args = p.parse_args()

    base_dir = args.dir or mlflow_dir

    # MLflow初期化（Databricksは使わずローカルのみ）
    manager = MLflowManager(base_dir, use_databricks=False)
    if not manager.initialize(base_dir):
        print("MLflowの初期化に失敗しました。")
        sys.exit(1)

    analyzer = MLflowAIAnalyzer(
        manager,
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
    )

    print(f"分析中... (backend={args.backend}, model={analyzer.model}"
          + (f", url={analyzer.base_url}" if args.backend == "local" else "") + ")")

    result = analyzer.analyze_experiment(
        TYPE_MAP[args.type],
        max_runs=args.max_runs,
        log_to_mlflow=not args.no_log,
    )

    if result["status"] == "success":
        print("\n" + "=" * 60)
        print(result["analysis"])
        print("=" * 60)
        if not args.no_log:
            print("\n（分析結果は MLflow に ai_analysis.md として記録されました）")
    else:
        print(f"\nエラー: {result['message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
