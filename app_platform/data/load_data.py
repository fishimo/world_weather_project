import pandas as pd


def load_predictions(path: str) -> pd.DataFrame:
    """予測結果のCSVを読み込む。

    station_id は DB 側が TEXT のため、int に推論されないよう明示的に str で読む。
    """
    df = pd.read_csv(
        path,
        dtype={"station_id": str},
        parse_dates=["origin_at", "target_at", "created_at"],
    )

    return df
