from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl


# polars変換
def ensure_polars(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    # pandas.DataFrame想定
    return pl.from_pandas(df)


# pandas変換
def ensure_pandas(df: Any) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError(f"Unsupported df type: {type(df)}")
