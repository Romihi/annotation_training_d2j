"""MLflow実験データを生成AI(LLM)に分析させる機能。

2種類のバックエンドに対応:
  - "anthropic": Claude API（クラウド、従量課金。要 anthropic SDK + ANTHROPIC_API_KEY）
  - "local"    : ローカルLLM（Ollama / LM Studio / vLLM / llama.cpp 等の
                 OpenAI互換エンドポイント。要 openai SDK。オフライン・無料）

MLflow 3 の GenAI 機能を活用:
  - "anthropic" は `mlflow.anthropic.autolog()`、"local" は `mlflow.openai.autolog()`
    により LLM 呼び出しを MLflow トレースとして記録（プロンプト・応答・
    トークン使用量・レイテンシが MLflow UI で観測可能）。

いずれの SDK も遅延インポートし、未導入でもアプリ本体が壊れないようにしている。

用途:
  各実験タイプ（自動運転/位置推定/時系列…）の Run 群を集約し、
  「どのハイパーパラメータがベスト指標と相関するか」「次に試すべき設定」を
  自然言語で提案させる。
"""
import os
import io
from datetime import datetime

import mlflow

# MLflow 3: LLM呼び出しのトレース連携（存在すれば利用）
try:
    import mlflow.anthropic as _mlflow_anthropic
    _MLFLOW_ANTHROPIC_AVAILABLE = True
except Exception:
    _MLFLOW_ANTHROPIC_AVAILABLE = False

try:
    import mlflow.openai as _mlflow_openai
    _MLFLOW_OPENAI_AVAILABLE = True
except Exception:
    _MLFLOW_OPENAI_AVAILABLE = False

# 既定モデル
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_LOCAL_MODEL = "qwen2.5:7b-instruct"  # Ollama例。導入済みモデル名に合わせて変更

# ローカルLLMサーバのOpenAI互換エンドポイント（プリセット）
LOCAL_PRESETS = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "vllm": "http://localhost:8000/v1",
    "llamacpp": "http://localhost:8080/v1",
}

SYSTEM_PROMPT = (
    "あなたは自動運転・機械学習の専門家です。MLflowの実験結果を読み、"
    "実務的で再現性のある改善提案を行います。"
)


def is_available(backend: str = "anthropic") -> bool:
    """指定バックエンドの SDK が利用可能か（APIキー/サーバ起動は別途チェック）"""
    if backend == "anthropic":
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False
    elif backend == "local":
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False
    return False


class MLflowAIAnalyzer:
    """MLflow の実験 Run を LLM に分析させるアナライザ（クラウド/ローカル両対応）"""

    def __init__(self, mlflow_manager, backend: str = "anthropic",
                 model: str = None, base_url: str = None, api_key: str = None,
                 enable_tracing: bool = True):
        """
        Args:
            mlflow_manager: MLflowManager インスタンス（Run取得・記録に使用）
            backend: "anthropic"（Claude API）または "local"（ローカルLLM）
            model: 使用するモデルID（未指定ならバックエンド既定値）
            base_url: localバックエンド時のエンドポイント。
                      LOCAL_PRESETS のキー("ollama"等)または完全なURLを指定可。
                      未指定なら ollama を使用。
            api_key: localバックエンド時のAPIキー（ローカルサーバは通常不要なため
                     ダミー可。未指定なら "local"）。
            enable_tracing: MLflow に LLM 呼び出しをトレース記録するか
        """
        self.mlflow_manager = mlflow_manager
        self.backend = backend
        self.enable_tracing = enable_tracing

        if backend == "anthropic":
            self.model = model or DEFAULT_ANTHROPIC_MODEL
            self.base_url = None
            self.api_key = api_key
        else:  # local
            self.model = model or DEFAULT_LOCAL_MODEL
            preset = base_url or "ollama"
            self.base_url = LOCAL_PRESETS.get(preset, preset)
            self.api_key = api_key or "local"  # OpenAI SDKは非空キーが必要

    # ------------------------------------------------------------------
    # データ整形
    # ------------------------------------------------------------------
    def _build_runs_summary(self, runs_df, max_runs: int = 40) -> str:
        """search_runs の DataFrame を LLM 用のコンパクトな表に整形"""
        if runs_df is None or len(runs_df) == 0:
            return ""

        keep_cols = []
        rename = {}
        for col in runs_df.columns:
            if col == "tags.mlflow.runName":
                keep_cols.append(col)
                rename[col] = "run_name"
            elif col.startswith("params.") or col.startswith("metrics."):
                keep_cols.append(col)
                rename[col] = col.split(".", 1)[1]

        subset = runs_df[keep_cols].rename(columns=rename).head(max_runs)
        buf = io.StringIO()
        subset.to_csv(buf, index=False)
        return buf.getvalue()

    def _build_prompt(self, experiment_label: str, runs_csv: str) -> str:
        return (
            f"以下は自動運転RCカーのモデル学習における「{experiment_label}」実験の"
            f"MLflow Run 一覧です（CSV形式: 各行が1つの学習試行、列はハイパーパラメータと評価指標）。\n\n"
            f"```csv\n{runs_csv}\n```\n\n"
            "次の観点で日本語で分析してください:\n"
            "1. ベスト性能の Run と、その特徴（どのパラメータ設定が効いたか）\n"
            "2. ハイパーパラメータと評価指標（val_loss / accuracy 等）の傾向・相関\n"
            "3. 過学習や学習不足の兆候があれば指摘\n"
            "4. 次に試すべき具体的な設定（学習率・バッチサイズ・エポック数・データ拡張など）を3案、根拠付きで\n\n"
            "簡潔で実用的に。表が有効なら Markdown 表を使ってください。"
        )

    # ------------------------------------------------------------------
    # バックエンド別 LLM 呼び出し
    # ------------------------------------------------------------------
    def _call_anthropic(self, prompt: str) -> str:
        import anthropic

        if self.enable_tracing and _MLFLOW_ANTHROPIC_AVAILABLE:
            try:
                _mlflow_anthropic.autolog()
            except Exception as e:
                print(f"MLflow autolog(anthropic) 有効化に失敗（トレースなしで続行）: {e}")

        client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic.Anthropic()
        with client.messages.stream(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()
        return "".join(b.text for b in message.content if b.type == "text")

    def _call_local(self, prompt: str) -> str:
        import openai

        if self.enable_tracing and _MLFLOW_OPENAI_AVAILABLE:
            try:
                _mlflow_openai.autolog()
            except Exception as e:
                print(f"MLflow autolog(openai) 有効化に失敗（トレースなしで続行）: {e}")

        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # メイン
    # ------------------------------------------------------------------
    def analyze_experiment(self, model_type, max_runs: int = 40,
                           log_to_mlflow: bool = True) -> dict:
        """指定モデルタイプの実験 Run を LLM に分析させる

        Returns:
            dict: {"status": "success"|"error", "analysis": str, "message": str}
        """
        # SDK チェック
        if not is_available(self.backend):
            pkg = "anthropic" if self.backend == "anthropic" else "openai"
            return {"status": "error", "analysis": "",
                    "message": f"{pkg} SDK が未インストールです。`pip install {pkg}` を実行してください。"}

        # 認証/接続前提のチェック
        if self.backend == "anthropic" and not (self.api_key or os.environ.get("ANTHROPIC_API_KEY")):
            return {"status": "error", "analysis": "",
                    "message": "環境変数 ANTHROPIC_API_KEY が設定されていません。"}

        # Run 取得
        runs_df = self.mlflow_manager.get_experiment_runs(model_type)
        runs_csv = self._build_runs_summary(runs_df, max_runs=max_runs)
        if not runs_csv:
            return {"status": "error", "analysis": "", "message": "分析対象の Run がありません。"}

        experiment_label = self.mlflow_manager.EXPERIMENT_NAMES.get(model_type, str(model_type))
        prompt = self._build_prompt(experiment_label, runs_csv)

        try:
            if self.backend == "anthropic":
                analysis = self._call_anthropic(prompt)
            else:
                analysis = self._call_local(prompt)
        except Exception as e:
            hint = ""
            if self.backend == "local":
                hint = f"（ローカルLLMサーバ {self.base_url} が起動しモデル '{self.model}' が利用可能か確認してください）"
            return {"status": "error", "analysis": "", "message": f"分析エラー: {e} {hint}"}

        if log_to_mlflow and analysis:
            try:
                self._log_analysis(model_type, experiment_label, analysis)
            except Exception as e:
                print(f"分析結果の記録に失敗: {e}")

        return {"status": "success", "analysis": analysis, "message": ""}

    def _log_analysis(self, model_type, experiment_label: str, analysis: str):
        """分析結果を MLflow の Run として記録（成果物=Markdown）"""
        if not self.mlflow_manager.set_experiment(model_type, "local"):
            return
        run_name = f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("analysis_type", "ai_experiment_review")
            mlflow.set_tag("analysis_backend", self.backend)
            mlflow.set_tag("analysis_model", self.model)
            mlflow.set_tag("analyzed_experiment", experiment_label)
            mlflow.log_text(analysis, "ai_analysis.md")
        if self.mlflow_manager.local_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_manager.local_tracking_uri)
