from __future__ import annotations

from typing import Any

import dash
from dash import dcc, html

from app_platform.config.config import config
from app_platform.data.stations import load_stations
from app_platform.extract.extract_data import extract_predictions
from app_platform.figures.wind_speed import create_wind_speed_chart


def layout(station_id: str | None = None, **kwargs: Any) -> html.Div:
    """観測地点ごとの詳細page。

    Args:
        station_id: URL から渡される観測地点ID。直接 /station/ を開かれた場合は None。
    """
    if station_id is None:
        return html.Div("観測地点が指定されていません。")

    # URL は文字列で渡ってくるので、DataFrame 側の型に合わせて比較する
    stations = load_stations()
    matched = stations[stations["station_id"].astype(str) == str(station_id)]
    if matched.empty:
        return html.Div(f"観測地点 {station_id} は登録されていません。")

    station = matched.iloc[0]
    station_name = station["name"]

    df = extract_predictions(config.path.pred_path, station_id)
    fig = create_wind_speed_chart(df, station_name)

    return html.Div(
        [
            html.H2(f"{station_name}（{station_id}）"),
            html.P(f"緯度 {station['latitude']} / 経度 {station['longitude']}"),
            dcc.Graph(figure=fig),
            dcc.Link("← 地図へ戻る", href="/"),
        ]
    )


# <station_id> の部分が layout() に同名のキーワード引数として渡ってくる。
# 先頭のスラッシュは必須（"./<station_id>" と書くと "/./..." になる）。
# layout= の明示が必要な理由は home.py のコメントを参照
dash.register_page(
    __name__,
    path_template="/station/<station_id>",
    name="観測地点の詳細",
    layout=layout,
)
