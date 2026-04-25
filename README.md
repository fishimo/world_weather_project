# world_weather_project

日本の地点別気象データを対象に、データ取得から学習・予測までを一通り回せる実験用プロジェクトです。
現在は Kaggle の `global-weather-repository` を取得し、日本データに絞って前処理し、SQLite に保存したうえで、複数 horizon の気温予測を行うところまで実装しています。

## 現時点の実装内容

- Kaggle API を使った気象データのダウンロード
- 日本のデータの抽出
- 必要カラムの選択と風速 `wind_kph -> wind_mps` 変換
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
python -m uv pip install -e .
```

Kaggle からデータ取得するため、事前に Kaggle API の認証設定が必要です。
`kaggle.json` を使う場合は、一般的な Kaggle CLI の配置場所に置いてから実行してください。

## ディレクトリと主な出力

- 生データ: `data/raw/kaggle_world_weather/`
- 加工済み DB: `data/processed/kaggle_world_weather/processed_GDR.db`
- 学習済みモデル: `test_models/test_model.pkl`
- 予測結果: `outputs/predictions.csv`

## 実行方法

### 1. データダウンロードと DB 更新

生データのダウンロード、前処理、日本データの抽出、SQLite への保存までをまとめて実行します。

```powershell
python -m uv run python -m portfolio.pipeline.download
```

### 2. 学習

DB に入っている加工済みデータを読み込み、前処理、Dataset 作成、学習、簡易評価、モデル保存までを行います。

```powershell
python -m uv run python -m portfolio.pipeline.train
```

`--test-size` を指定すると train/test の分割比率を変更できます。

```powershell
python -m uv run python -m portfolio.pipeline.train --test-size 0.2
```

学習済みモデルはデフォルトで `test_models/test_model.pkl` に保存されます。

### 3. 予測

保存済みモデルを読み込み、指定期間のデータから最新日時のレコードを使って予測し、CSV に保存します。

```powershell
python -m uv run python -m portfolio.pipeline.prediction --start "2026-04-01 00:00:00" --end "2026-04-21 00:00:00"
```

モデルパスや出力先を変えたい場合は引数で指定できます。

```powershell
python -m uv run python -m portfolio.pipeline.prediction `
  --model-path test_models/test_model.pkl `
  --start "2026-04-01 00:00:00" `
  --end "2026-04-21 00:00:00" `
  --output-path outputs/predictions.csv
```

## 補足

- 予測で使う horizon は現在 `configs/config.py` の `pipeline.horizon = [1, 2, 3]` です。
- 学習時の特徴量一覧はモデル保存時に一緒に持たせており、推論時はその特徴量順にそろえて予測します。
- `prediction` は内部的に `predict.py` の処理を呼び出しており、指定期間のデータを読み込んだあと、その期間内の最新日時のレコードだけを使って予測します。
