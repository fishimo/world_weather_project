from __future__ import annotations

from pathlib import Path

import pandas as pd

# DBに保存するカラム順（後段の PreProcessor が参照する列名と一致させる）
_PROCESSED_COLS: list[str] = [
    "location_name",
    "latitude",
    "longitude",
    "last_updated",
    "temperature_celsius",
    "wind_mps",
    "wind_degree",
    "pressure_mb",
]


class AmedasProcessor:
    """AMeDAS rawデータCSVを読み込み、後段DBと同じスキーマに整形する。

    raw CSV の "timestamp" を "last_updated" にリネームし、
    DBに保存する列のみ抽出する。単位変換は不要（風速は元々m/s）。
    """

    def __init__(self, config) -> None:
        fetch = config.amedas_fetch
        self._rawdata_path = Path(fetch.raw_dir) / fetch.raw_filename

    @property
    def rawdata_path(self) -> Path:
        return self._rawdata_path

    def read_data(self) -> pd.DataFrame:
        if not self._rawdata_path.is_file():
            raise FileNotFoundError(f"生データが見つかりません: {self._rawdata_path}")
        return pd.read_csv(self._rawdata_path)

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """rawデータをDBに保存できるスキーマに整形する。"""
        df = df.rename(columns={"timestamp": "last_updated"})
        missing = [c for c in _PROCESSED_COLS if c not in df.columns]
        if missing:
            print(f"[WARN] raw データに存在しないカラム: {missing}")
        existing = [c for c in _PROCESSED_COLS if c in df.columns]
        df = df[existing].copy()
        self._log_missing(df)
        return df

    def _log_missing(self, df: pd.DataFrame) -> None:
        nan_count = df.isna().sum()
        nan_rate = df.isna().mean()
        for col in df.columns:
            print(f"{col}: missing={nan_count[col]} ({nan_rate[col] * 100:.2f} %)")
