# MLflow 3 対応まとめ

調査日: 2026-06-21 / 対象環境のインストール済みバージョン: **mlflow 3.11.1**

## 1. 結論

本プロジェクトが使用している MLflow API は **ほぼ全てが MLflow 3 でもそのまま動作する**（コアAPIのスモークテスト合格）。
ハードな破壊的変更は **1箇所のみ**（`log_model` の引数）で、それ以外は「非推奨警告」レベル。

使用中の主なAPI（いずれも MLflow 3 で有効）:
`set_tracking_uri` / `set_experiment` / `get_experiment_by_name` / `create_experiment` /
`start_run` / `log_param` / `log_metric` / `set_tag(s)` / `log_artifact(s)` /
`search_runs` / `MlflowClient` / `delete_run` / `list_artifacts` / `download_artifacts`

## 2. 修正箇所（実施済み）

| # | ファイル | 内容 | 種別 |
|---|---------|------|------|
| 1 | `requirements.txt` | `mlflow>=1.20.0` → `mlflow>=3.1,<4.0` | バージョン固定 |
| 2 | `setup/rtx5060/requirements_cuda.txt` | 同上 + `nvidia-ml-py` / `psutil`（システムメトリクス用） | バージョン固定 |
| 3 | `databricks/03_train_model.py` | `mlflow.pytorch.log_model(model, "model")` → `log_model(model, name="model", signature=..., input_example=...)` | **破壊的変更対応** |
| 4 | `databricks/03_train_model.py` | `start_run(log_system_metrics=True)` に変更（学習中のGPU/CPU/メモリを自動記録） | 新機能 |

### 補足: `log_model` の破壊的変更
MLflow 3 では `log_model` の第2位置引数 `artifact_path` が **非推奨**となり、キーワード引数 `name=` に置き換えられた。
位置引数のままでも当面は動作するが将来削除予定のため修正済み。あわせて MLflow 3 推奨の
`signature` / `input_example` を付与し、モデルサービング・入力検証に対応した。

## 2.5 実施済み（追加対応）

| # | ファイル | 内容 |
|---|---------|------|
| 5 | `managers/mlflow_manager.py` | ローカルバックエンドを **ファイルストア → SQLite** に移行（`sqlite:///{dir}/mlflow.db` + `file://{dir}/mlartifacts`）。`_build_local_uris` ヘルパー追加、`_initialize_local`/`_get_default_mlflow_uri`/`open_ui` をsqlite対応に変更。`config.py` の `mlflow_dir` はベースディレクトリとしてそのまま利用（変更不要） |
| 6 | `tools/migrate_mlflow_to_sqlite.py` | 既存ファイルストアのRunをSQLiteへ移行するツール（パラメータ/タグ/メトリクス履歴/アーティファクトを保全、冪等） |
| 7 | `managers/mlflow_ai_analyzer.py` | **MLflow 3 GenAI連携**: 実験RunをClaudeに分析させる新機能。`mlflow.anthropic.autolog()` でClaude呼び出しをMLflowトレースとして記録 |

## 2.6 Python 3.14 + mlflow 3.14 の既知問題（パッチ対応済み）

venv環境は **Python 3.14 / mlflow 3.14** だが、mlflow 3.14 のFastAPIサーバ
（`mlflow ui`）が `from importlib.abc import Traversable` を実行する。
`Traversable` は **Python 3.14 で `importlib.abc` から削除**されたため、
`mlflow ui` 起動時に `ImportError` で停止する。

- 対策: `mlflow/assistant/skill_installer.py` の該当importを新旧両対応に修正（適用済み・動作確認済み）。
- **再インストールで消える**ため、`pip install` でmlflowを入れ直したら再適用すること:
  ```bash
  venv\Scripts\python tools/patch_mlflow_py314.py
  ```
- あわせて、アプリの **2つ目のUI起動経路** `main.py: _open_local_mlflow_ui` も
  `file://` → `sqlite:///` に修正（`open_ui` と同様）。

## 3. ファイルストアバックエンドの非推奨 → SQLite移行（実施済み）

MLflow 3 では **ファイルストア（`file://` / `./mlruns`）が 2026年2月をもって非推奨**となった
（実行時に `FutureWarning` が出る）。本プロジェクトのローカル記録（`managers/mlflow_manager.py`）は
全面的に `file:///{path}` を使用している。

- **現状**: 動作はする（警告のみ）。すぐ壊れるわけではない。
- **推奨移行先**: DBバックエンド（例 `sqlite:///mlflow.db`）

### SQLiteバックエンドへ移行する場合の影響範囲
| 箇所 | 変更内容 |
|------|---------|
| `config.py` | tracking URI を `sqlite:///{path}/mlflow.db` 化、artifact ルートを別途定義 |
| `mlflow_manager._initialize_local` | tracking_uri / artifact_location の設定変更 |
| `mlflow_manager.open_ui` | `mlflow ui --backend-store-uri sqlite:///... --default-artifact-root file:///...` |
| `mlflow_manager._get_default_mlflow_uri` / 同期系メソッド | 同期元URIをsqlite化 |
| 既存データ | 既存 `mlruns`（ファイルストア）は別形式。履歴を引き継ぐには移行が必要 |

> SQLite化により **Model Registry / エイリアス（champion/challenger）が使えるようになる**
> （ファイルストアでは Model Registry 非対応）。Windows実機で動作確認済み。

### 既存データの移行手順
```bash
# 既定(config.mlflow_dir)を対象
python tools/migrate_mlflow_to_sqlite.py
# 任意ディレクトリ（学習時に使ったデータフォルダ）を指定
python tools/migrate_mlflow_to_sqlite.py --dir "C:/path/to/data_folder"
# 内容確認のみ（書き込みなし）
python tools/migrate_mlflow_to_sqlite.py --dry-run
```

## 3.5 AI実験分析機能（生成AI連携）の課金について

`managers/mlflow_ai_analyzer.py` は **Anthropic API（開発者プラットフォーム）** を
`anthropic` SDK 経由で呼び出す（`ANTHROPIC_API_KEY` 必須）。

> ⚠️ **Claude Pro/Max の月額サブスクリプションとは課金が別**。
> 月額サブスクは claude.ai / デスクトップ / Claude Code 用で、SDKからのAPI呼び出しは
> **従量課金（APIクレジット）** になる。1回の分析（Run数十件）の概算コストは
> claude-opus-4-8 で **約 $0.05〜0.20 程度**（入力 $5 / 出力 $25 per 1M tokens）。
> コスト重視なら `model="claude-sonnet-4-6"`（$3/$15）や `"claude-haiku-4-5"`（$1/$5）も指定可能。
> サブスク内で済ませたい場合は、自動化せず実験CSVを claude.ai / Claude Code に
> 手動で貼り付けて分析する運用になる（本モジュールは使わない）。

### ローカルLLM連携（オフライン・無料）
`mlflow_ai_analyzer.py` は **ローカルLLMバックエンド** にも対応（API課金なし）。
Ollama / LM Studio / vLLM / llama.cpp など **OpenAI互換API** を公開するものなら接続可能。
MLflow 3 の `mlflow.openai.autolog()` でローカルLLM呼び出しもトレース記録される。

```python
from managers.mlflow_ai_analyzer import MLflowAIAnalyzer
from managers import ModelType

# ローカルLLM（Ollama例）。要 pip install openai + ollama サーバ起動
analyzer = MLflowAIAnalyzer(
    mlflow_manager,
    backend="local",
    base_url="ollama",            # "ollama"/"lmstudio"/"vllm"/"llamacpp" or 完全URL
    model="qwen2.5:7b-instruct",  # 導入済みモデル名
)
result = analyzer.analyze_experiment(ModelType.AUTONOMOUS_DRIVING)
print(result["analysis"])
```

プリセット: Ollama `:11434/v1` / LM Studio `:1234/v1` / vLLM `:8000/v1` / llama.cpp `:8080/v1`。
クラウドClaudeを使う場合は `backend="anthropic"`（既定）。

## 4. MLflow 3 新機能を活かした提案

| 優先 | 機能 | 内容 | 前提 |
|------|------|------|------|
| ★★★ | **システムメトリクス記録** | 学習中のGPU使用率・VRAM・CPU・メモリを自動記録。RTX5060環境の学習監視に有用 | `start_run(log_system_metrics=True)` + `nvidia-ml-py`。**学習ループを `start_run` で囲む箇所**でのみ有効（現ローカル実装は学習後に記録するため要リファクタ） |
| ★★★ | **モデルsignature/input_example** | 入出力スキーマを記録。サービング・検証・再現性向上 | `infer_signature` |
| ★★☆ | **Model Registry + エイリアス** | 各実験のベストモデルに `@champion` を付与し、車両デプロイ対象を一元管理 | **SQLite/DBバックエンド必須** |
| ★★☆ | **データセット追跡（`mlflow.data`）** | 学習に使ったアノテーションデータのバージョン/件数/ハッシュを Run の入力として記録 | `mlflow.data.from_pandas` 等 |
| ★☆☆ | **モデル中心のトラッキング（Logged Models）** | MLflow 3 でモデルが第一級エンティティ化。複数Runにまたがるモデル比較 | — |

### 注意点
- ローカルマネージャは「学習完了後にまとめて記録」する後処理型のため、システムメトリクスを
  活かすには学習ループを `start_run` 内に入れるリファクタが必要（databricks側は学習ループを
  囲んでいるため対応済み）。
- Model Registry / エイリアスはファイルストアでは動作しない → §3 のSQLite移行が前提。
