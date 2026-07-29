# AGENTS.md

## プロジェクト概要

日本各地の AMeDAS 気象データを取得し、LightGBM による複数 horizon の**風速予測**を行うポートフォリオプロジェクト。
データ取得 → 前処理 → SQLite 保存 → 学習 → 予測 の一連のパイプラインを実装している。

## 環境・セットアップ

- OS: Windows / PowerShell
- Python: 3.13
- パッケージ管理: `uv`

```powershell
python -m pip install -U uv
python -m uv venv
python -m uv pip install -e .
```

## 主要ライブラリ

| 用途 | ライブラリ |
|---|---|
| データ取得 | `requests` (気象庁 API) |
| データ処理 | `pandas`, `polars`, `pyarrow` |
| 機械学習 | `lightgbm`, `scikit-learn` |
| 可視化 | `plotly` |
| 開発 | `ruff`, `mypy`, `pytest` |

## データソース

**気象庁 AMeDAS** (1時間粒度、複数地点)

- API エンドポイント: 気象庁の公開 JSON API
- **リクエスト間隔は必ず 2 秒以上空ける**（スクレイピングポリシー遵守）
- 風速の単位は m/s（変換不要）

## ディレクトリ構成

```
world_weather_project/
├── forecast_core/             # 予測パイプライン本体（import 名 = forecast_core）
│   ├── config/                # 設定（frozen dataclass）
│   │   ├── config.py          #   AmedasFetch / DataBase / PredictionDataBase / PreProcess / Pipeline
│   │   └── params.py          #   LightGBMParams
│   ├── preprocessing/
│   │   └── amedas_processor.py# 前処理（カラム選択・型変換）
│   ├── features/
│   │   └── feature_preprocessor.py  # 日平均化・型変換
│   ├── data/
│   │   ├── amedas_fetcher.py  # AMeDAS データ取得
│   │   ├── database_manager.py# SQLite 保存・読み込み
│   │   └── dataset.py         # Dataset / DatasetGenerator（horizon 展開）
│   ├── models/
│   │   ├── model_store.py     # pickle 保存・読み込み（TrainedModelArtifact）
│   │   └── multi/
│   │       ├── builder.py     # MultiModelBuilder
│   │       └── multioutputmodel.py  # LightGBM multi-output ラッパー
│   ├── evaluation/
│   │   └── evaluator.py
│   ├── utils/
│   │   └── pd_pl_utils.py     # pandas / polars 変換ヘルパ
│   └── pipeline/
│       ├── download.py        # ステップ1: 取得 → DB 保存
│       ├── train.py           # ステップ2: 学習 → モデル保存
│       └── predict.py         # ステップ3: 予測 → DB 保存（CSV は任意）
├── app_platform/              # 可視化アプリ（標準ライブラリ platform と衝突するため app_ 接頭辞）
│   ├── data/                  # DB・Parquet の読み込み処理
│   ├── extract/               # 抽出・結合・集計
│   ├── figures/               # Plotly Figure 生成
│   ├── components/            # 地図・入力欄などの共通部品
│   └── pages/                 # 画面構成・画面遷移
├── scripts/                   # 手動実行スクリプト（バックフィル・最新取得）
├── data/
│   ├── raw/amedas/            # 生データ（JSON → CSV）
│   ├── processed/amedas/      # 加工済み SQLite DB
│   └── predictions/           # 予測結果 SQLite DB
├── outputs/                   # 予測結果 CSV
├── test_models/               # 学習済みモデル pickle
└── tests/
    └── integration/           # パイプライン統合テスト
```

`forecast_core` / `app_platform` はどちらも editable install 済みパッケージなので、
`sys.path` 操作は不要。`from forecast_core.data.dataset import Dataset` のように import する。
データパスは cwd 相対のため、実行は必ずリポジトリルートで行うこと。

## パイプライン実行コマンド

```powershell
# 1. データ取得 → DB 保存
python -m uv run python -m forecast_core.pipeline.download

# 2. 学習 → モデル保存
python -m uv run python -m forecast_core.pipeline.train --test-size 0.2

# 3. 予測 → CSV 出力
python -m uv run python -m forecast_core.pipeline.predict `
  --model-path test_models/test_model.pkl `
  --start "2026-04-01 00:00:00" `
  --end "2026-04-21 00:00:00" `
  --output-path outputs/predictions.csv
```

## Config 構造

`forecast_core/config/config.py` の frozen dataclass でパスや列名を一元管理。ハードコード禁止。

```python
Config
├── amedas_fetch:        AmedasFetch         # 取得対象地点・保存先・リクエスト間隔
├── database:            DataBase            # DB パス・テーブル名・タイムスタンプ列名
├── prediction_database: PredictionDataBase  # 予測結果 DB パス・テーブル名
├── preprocess:          PreProcess          # 特徴量列・風速列名
└── pipeline:            Pipeline            # horizon リスト（例: [1, 2, 3]）
```

モデルハイパーパラメータは `forecast_core/config/params.py` の `LightGBMParams` で管理。

## コーディング規約

- **型ヒント必須**（すべての関数シグネチャ）
- フォーマッタ: `ruff`、型チェック: `mypy`
- 設定値はすべて `forecast_core/config/` に外出しし、コード内にハードコードしない
- 可読性を重視（短絡的な省略より明示的な記述）
- コメントは「なぜそう書いたか」が自明でない箇所にのみ付ける
- ヘルパー関数は使用する関数より前に置く

## テスト方針

- テストは**パイプライン単位の統合テスト**を基本とする（`tests/integration/`）
- ダミーデータを生成して DB 構築 → 学習 → 予測 の一連を通す
- 外部 API 呼び出し（AMeDAS 取得）は `monkeypatch` でスキップする
- ユニットテストが必要な場合は `tests/unit/` に追加する

## コード品質チェック

実装後は必ず以下を順に実行し、すべて通るまで修正を繰り返すこと:

1. `ruff check . --fix`
2. `ruff format .`
3. `mypy .`

すべて通ってから完了報告すること。
