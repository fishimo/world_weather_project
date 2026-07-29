from __future__ import annotations

import pandas as pd

from forecast_core.config.config import config as core_config


def load_stations() -> pd.DataFrame:
    """AMeDAS観測地点の一覧を DataFrame で返す。

    地点定義は forecast_core 側の config を正とし、アプリ側では複製しない。
    地点を追加した際にアプリ側の修正が不要になる。
    """
    return pd.DataFrame(
        [
            {
                "station_id": s.station_id,
                "name": s.name,
                "latitude": s.latitude,
                "longitude": s.longitude,
            }
            for s in core_config.amedas_fetch.stations
        ]
    )
