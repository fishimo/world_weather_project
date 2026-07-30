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


@dataclass(frozen=True)
class WindSpeedChart:
    """風速の時系列グラフの表示設定

    色は検証済みパレットから取っている。line_color は明るい背景に対して
    コントラスト比 4.30:1（3:1 以上が必要）。grid / axis はデータより
    目立たないよう意図的に薄い
    """

    line_color: str = "#2a78d6"
    line_width: int = 2
    surface_color: str = "#fcfcfb"
    grid_color: str = "#e1e0d9"
    axis_color: str = "#c3c2b7"
    tick_color: str = "#898781"
    title_color: str = "#52514e"
    y_axis_title: str = "風速 (m/s)"
    x_axis_title: str = "時刻"


@dataclass(frozen=True)
class Path:
    """データファイルの場所。パスは cwd 相対なので実行はリポジトリルートで行う"""

    pred_path: str = "data/predictions/predictions.csv"


class Config:
    """アプリ側の設定を管理"""

    def __init__(self) -> None:
        self.japan_map_view = JapanMapView()
        self.wind_speed_chart = WindSpeedChart()
        self.path = Path()


config = Config()
