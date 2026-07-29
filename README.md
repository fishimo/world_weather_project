# world_weather_project

日本の AMeDAS 気象データを対象に、データ取得から学習・予測までを一通り回せる実験用プロジェクトです。
気象庁公開データを取得し、東京（北の丸公園/旧大手町）の風速を 1〜3 日先まで予測します。

## 現時点の実装内容

- 気象庁 AMeDAS JSON API からの最新データ取得（1時間粒度）
- 気象庁過去データ HTML からの履歴データ取得（月単位バックフィル）
- rawデータ CSV への追記・重複排除保存
- 加工済みデータの SQLite 保存
- DB からの全件読み込み / 期間指定読み込み
- 学習用 Dataset の作成
- horizon 展開と train/test 分割
- LightGBM ベースの multi-output 学習
- 学習済みモデルの pickle 保存
- 保存済みモデルを使った推論
- 予測結果の CSV 出力

## セットアップ

- 想定環境: Windows / PowerShell
- Python: `3.13`
- パッケージ管理: `uv`

```powershell
python -m pip install -U uv
python -m uv venv
python -m uv sync
```

## ディレクトリと主な出力

- raw データ: `data/raw/amedas/amedas_raw.csv`
- 加工済み DB: `data/processed/amedas/processed_amedas.db`
- 学習済みモデル: `test_models/test_model.pkl`
- 予測結果: `outputs/predictions.csv`

## 実行方法

### 0. 過去データのバックフィル（初回のみ）

学習に使う期間の履歴データを気象庁HTMLから取得して raw CSV に保存します。

```powershell
# 単月
python scripts/fetch_historical.py --year 2025 --month 4

# 期間指定
python scripts/fetch_historical.py --from 2024-01 --to 2025-04
```

> リクエスト間隔は 2 秒以上（config.request_interval_sec）で気象庁への負荷を配慮しています。

### 1. データ取得と DB 更新

最新の AMeDAS JSON データを取得し、raw CSV に追記・前処理・SQLite 保存まで行います。

```powershell
python -m uv run python -m forecast_core.pipeline.download
```

### 2. 学習

DB に入っている加工済みデータを読み込み、前処理・Dataset 作成・学習・簡易評価・モデル保存までを行います。

```powershell
python -m uv run python -m forecast_core.pipeline.train --test-size 0.2
```

学習済みモデルはデフォルトで `test_models/test_model.pkl` に保存されます。

### 3. 予測

保存済みモデルを読み込み、指定期間のデータから最新日時のレコードを使って予測し、CSV に保存します。

```powershell
python -m uv run python -m forecast_core.pipeline.predict `
  --model-path test_models/test_model.pkl `
  --start "2026-04-01 00:00:00" `
  --end "2026-04-21 00:00:00" `
  --output-path outputs/predictions.csv
```

### 最新データの定期取得（参考: cron 設定例）

毎時 25 分に最新データを取得して raw CSV に追記する場合:

```
25 * * * * cd /path/to/world_weather_project && python scripts/fetch_latest.py
```

予測も合わせて行う場合は `forecast_core.pipeline.predict` を実行してください（内部で最新データ取得を行います）。

## 補足

- 予測で使う horizon は `forecast_core/config/config.py` の `pipeline.horizon = [1, 2, 3]`（日単位）です。
- 観測地点の追加は `forecast_core/config/config.py` の `AmedasFetch.stations` リストに `AmedasStation` を追加します。
- 物理量（気温・湿度・気圧等）の追加は `amedas_fetcher.py` の `_JSON_VALUE_FIELDS` / `_HTML_VALUE_FIELDS` に1行追記します。
- 学習時の特徴量一覧はモデル保存時に保持されており、推論時に列順を自動で揃えます。
