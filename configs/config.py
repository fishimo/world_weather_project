from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class AmedasStation:
    """AMeDAS観測地点の設定"""
    station_id: str = "44132"
    name: str = "Tokyo"
    latitude: float = 35.6944
    longitude: float = 139.7529
    # 気象庁過去データ(hourly_s1.php)用パラメータ
    historical_prec_no: str = "44"     # 都道府県番号
    historical_block_no: str = "47662" # 地点番号（気象官署）


@dataclass(frozen=True)
class AmedasFetch:
    """AMeDASデータ取得の設定"""
    stations: list = field(default_factory=lambda: [AmedasStation()])
    raw_dir: Path = field(default_factory=lambda: Path("data/raw/amedas"))
    raw_filename: str = "amedas_raw.csv"
    # 将来10/30分粒度に変更する場合はここを変える
    granularity_minutes: int = 60
    request_interval_sec: float = 2.0


@dataclass(frozen=True)
class DataBase:
    """DBManager用の設定"""
    table_name: str = "processed_amedas_data"
    ts_col: str = "last_updated"
    db_path: Path = field(
        default_factory=lambda: Path("data/processed/amedas/processed_amedas.db")
    )


@dataclass(frozen=True)
class PreProcess:
    """preprocessor用の設定"""
    ts_col: str = "last_updated"
    wind_col: str = "wind_mps"
    require_cols: list = field(default_factory=lambda: [
        "location_name",
        "last_updated",
        "temperature_celsius",
        "wind_mps",
        "wind_degree",
        "pressure_mb",
    ])


@dataclass(frozen=True)
class Pipeline:
    """pipeline用の設定"""
    horizon: List[int] = field(default_factory=lambda: [1, 2, 3])


class Config:
    """全体の設定を管理"""
    def __init__(self) -> None:
        self.amedas_fetch = AmedasFetch()
        self.database = DataBase()
        self.preprocess = PreProcess()
        self.pipeline = Pipeline()


config = Config()
