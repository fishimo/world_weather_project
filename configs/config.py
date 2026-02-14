from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class DataDownload:
    """datadownload用の設定"""
    datasets: str = "nelgiriyewithana/global-weather-repository"
    dir_download: Path = field(default_factory=lambda: Path("data/raw/kaggle_world_weather"))

class Config:
    """全体の設定を管理"""
    def __init__(self) -> None:
        """設定オブジェクトを初期化"""
        self.data_download = DataDownload()