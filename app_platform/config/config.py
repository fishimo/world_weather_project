from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class JapanMapView:
    """日本地図の表示設定"""

    scope: str = "asia"
    resolution: int = 50  # 海岸線の解像度。50 が最も細かい
    lataxis_range: List[float] = field(default_factory=lambda: [24.0, 46.0])
    lonaxis_range: List[float] = field(default_factory=lambda: [122.0, 148.0])
    land_color: str = "#eaeaea"
    ocean_color: str = "#cfe8f3"
    marker_size: int = 12
    marker_color: str = "#d62728"


class Config:
    """アプリ側の設定を管理"""

    def __init__(self) -> None:
        self.japan_map_view = JapanMapView()


config = Config()
