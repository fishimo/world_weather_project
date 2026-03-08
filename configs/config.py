from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataDownload:
    """datadownload用の設定"""
    datasets: str = "nelgiriyewithana/global-weather-repository"
    out_dir: Path = field(default_factory=lambda: Path("data/raw/kaggle_world_weather"))

@dataclass(frozen=True)
class DataProcess:
    """dataprocessor用の設定"""
    rawdata_path: Path = field(default_factory=lambda: Path("data/raw/kaggle_world_weather/GlobalWeatherRepository.csv"))
    selected_cols: list = field(default_factory=lambda: [
        "country", 
        "location_name",
        "latitude",
        "longitude",
        "last_updated",
        "temperature_celsius",
        "wind_kph",
        "wind_degree",
        "pressure_mb",
        ])
    selected_country: list = field(default_factory=lambda: ["Japan"])

@dataclass(frozen=True)
class DataBase:
    """DBManager用の設定"""
    table_name: str = "processed_GWR_data"
    ts_col: str = "last_updated"
    db_path: Path = field(default_factory=lambda: Path("data/processed/kaggle_world_weather/processed_GDR.db"))

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
    horizon: List[int] = field(
        default_factory=lambda: [1, 2, 3]
    ) 

class Config:
    """全体の設定を管理"""
    def __init__(self) -> None:
        """設定オブジェクトを初期化"""
        self.data_download = DataDownload()
        self.data_process = DataProcess()
        self.database = DataBase()
        self.preprocess = PreProcess()
        self.pipeline = Pipeline()

config = Config()