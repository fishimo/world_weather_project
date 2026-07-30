from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app_platform.config.config import config


def _empty_figure(message: str) -> go.Figure:
    """データが無いときに、空のグラフ枠とメッセージだけを返す。"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def create_wind_speed_chart(
    predictions: pd.DataFrame,
    station_name: str | None = None,
) -> go.Figure:
    """風速予測の時系列グラフを返す。

    Args:
        predictions: extract_predictions() で1地点に絞り込み済みの DataFrame。
            target_at / wind_speed 列を持つこと。
        station_name: グラフタイトルに出す地点名。
    """
    view = config.wind_speed_chart

    if predictions.empty:
        return _empty_figure("表示できる予測データがありません。")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions["target_at"],
            y=predictions["wind_speed"],
            mode="lines",  # 48点あるのでマーカーは付けない。付けるなら size>=8
            line={"color": view.line_color, "width": view.line_width},
            name="予測風速",
            hovertemplate="%{x|%m/%d %H:%M}<br>%{y:.2f} m/s<extra></extra>",
        )
    )

    # 系列が1本なので凡例は出さず、タイトルで何のグラフかを示す
    title = "風速予測" if station_name is None else f"{station_name} の風速予測"

    fig.update_layout(
        title={"text": title, "font": {"color": view.title_color}},
        showlegend=False,
        # 縦線カーソル + ツールチップ。時系列はこれが無いと値を読み取れない
        hovermode="x unified",
        plot_bgcolor=view.surface_color,
        paper_bgcolor=view.surface_color,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif'},
        margin={"r": 20, "t": 60, "l": 60, "b": 40},
    )

    # グリッドと軸はデータより手前に出ないよう薄く保つ
    fig.update_xaxes(
        title_text=view.x_axis_title,
        showgrid=False,
        linecolor=view.axis_color,
        tickfont={"color": view.tick_color},
    )
    fig.update_yaxes(
        title_text=view.y_axis_title,
        showgrid=True,
        gridcolor=view.grid_color,
        linecolor=view.axis_color,
        tickfont={"color": view.tick_color},
        # 風速は絶対量なので 0 基準にしている。変化を強調したいなら消す
        rangemode="tozero",
    )

    return fig
