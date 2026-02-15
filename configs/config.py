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
    rawdata_path: Path = field(default_factory=lambda: Path("data/raw/kaggle_world_weather/global-weather-repository.csv"))
    selected_cols: list = field(default_factory=lambda: [
        "country", 
        "location_name",
        "latitude",
        "longitude",
        "location_name",
        "last_updated_epoch",
        "temperature_celsius",
        "wind_kph",
        "wind_degree",
        "pressure_mb",
        ])
    selected_country: list = field(default_factory=lambda: ["Japan"])
    



class Config:
    """全体の設定を管理"""
    def __init__(self) -> None:
        """設定オブジェクトを初期化"""
        self.data_download = DataDownload()
