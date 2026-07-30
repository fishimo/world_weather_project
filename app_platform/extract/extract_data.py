import pandas as pd

from app_platform.data.load_data import load_predictions


def extract_predictions(path: str, station_id: str) -> pd.DataFrame:
    """予測結果を1地点ぶんに絞り込み、時刻順に並べて返す。"""
    df = load_predictions(path)
    mask = df["station_id"].astype(str) == str(station_id)
    df_station = df[mask]
    return df_station.sort_values("target_at", ignore_index=True)
