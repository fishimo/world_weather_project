from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app_platform.config.config import config


def create_japan_map(stations: pd.DataFrame) -> go.Figure:
    """観測地点を日本地図上にプロットした Figure を返す。

    Args:
        stations: latitude / longitude / name 列を持つ観測地点の DataFrame。
    """
    view = config.japan_map_view

    fig = px.scatter_geo(
        stations,
        lat="latitude",
        lon="longitude",
        hover_name="name",
    )
    fig.update_traces(marker={"size": view.marker_size, "color": view.marker_color})

    # 地点が1つだけだと fitbounds="locations" はズームしすぎて地図が破綻するため、
    # 表示範囲を明示的に指定して日本全域に固定している
    fig.update_geos(
        scope=view.scope,
        resolution=view.resolution,
        lataxis_range=view.lataxis_range,
        lonaxis_range=view.lonaxis_range,
        showcountries=True,
        showocean=True,
        landcolor=view.land_color,
        oceancolor=view.ocean_color,
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig
