from dataclasses import dataclass, field
from pathlib import Path

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
    db_path: Path = field(default_factory=lambda: Path("data/processed/kaggle_world_weather/processed_GDR.db"))

class Config:
    """全体の設定を管理"""
    def __init__(self) -> None:
        """設定オブジェクトを初期化"""
        self.data_download = DataDownload()
        self.data_process = DataProcess()
        self.database = DataBase()

config = Config()