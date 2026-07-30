from __future__ import annotations

from typing import Any

import dash
from dash import Input, Output, callback, dcc, html

from app_platform.data.stations import load_stations
from app_platform.figures.japan_map import create_japan_map

MAP_GRAPH_ID = "japan-map"


def layout(**kwargs: Any) -> html.Div:
    """トップpage。日本地図に観測地点をプロットする。"""
    return html.Div(
        [
            dcc.Graph(
                id=MAP_GRAPH_ID,
                figure=create_japan_map(load_stations()),
                style={"height": "80vh"},
            ),
        ]
    )


# layout= を明示的に渡す。Dash が layout を自動で拾うのは pages_folder 経由で
# import した場合だけなので、手動 import する構成では明示が必須
# （省略すると NoLayoutException になる）
dash.register_page(__name__, path="/", name="地図", layout=layout)


@callback(
    Output("url", "pathname"),
    Input(MAP_GRAPH_ID, "clickData"),
    prevent_initial_call=True,  # 起動時の発火（clickData が None）を防ぐ
)
def go_to_station_detail(click_data: dict[str, Any] | None) -> Any:
    """地図上の地点がクリックされたら、その地点の詳細pageへ遷移する。"""
    if click_data is None:
        return dash.no_update

    # points はクリックされた点の配列、customdata は custom_data に渡した列の配列。
    # どちらも要素1つでもリストで来るため [0] が2回必要
    station_id = click_data["points"][0]["customdata"][0]
    return f"/station/{station_id}"
