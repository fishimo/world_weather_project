# world_weather_project

## 目的
日本（東京）の天気・風速データを定期取得し、蓄積したデータから日平均風速の1～5日先予測を行う。
またこの時系列予測は、最終的に気象庁GPVデータMSMおよびLFMの風速予測を行ううえでの土台とする予定である。

---

## 環境構築

### 前提（uv / pyproject.toml）
- Windows / PowerShell（想定）
- Python 3.13

### 1) uv のインストール
```powershell
python -m pip install -U uv
python -m uv --version
```

### 2) 仮想環境の作成
```powershell
python -m uv init
python -m uv venv
```

### 3) 依存関係のインストール
```powershell
python -m uv pip install -e .
```

---

## 全体像

### データ保存

- kaggle datasetからWorld Weather Repositoryのデータをkaggle APIをたたいて生データを取得・CSVのまま保存
- 取得後欲しいデータのみ抽出し、DB（SQLite）として保存。

実行
```powershell
python -m uv run python -m scripts.check_data_save
```

### 