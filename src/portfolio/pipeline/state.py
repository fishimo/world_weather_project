from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from portfolio.data.dataset import Dataset


@dataclass
class BaseState:
    """Train / Predict 共通の状態"""

    horizon: Optional[List[int]] = None

    df: Optional[pl.DataFrame] = None
    model: Optional[Any] = None
    ds: Optional[Dataset] = None
    _model_exists: bool = False

    def __repr__(self) -> str:
        df_info = f"shape: {self.df.shape}" if self.df is not None else "None"
        ds_info = f"Dataset(len={len(self.ds)})" if self.ds is not None else "None"
        return (
            f"{self.__class__.__name__}("
            f"horizon={self.horizon}, "
            f"df={df_info}, "
            f"ds={ds_info})"
        )


@dataclass
class TrainState(BaseState):
    """全量学習・ホールドアウト学習などに使う State（CV 専用フィールドは含まない）"""

    test_ds: Optional[Dataset] = None
    train_ds: Optional[Dataset] = None
    score: Optional[dict] = None
    test_predictions: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        df_info = f"shape: {self.df.shape}" if self.df is not None else "None"
        ds_info = f"Dataset(len={len(self.ds)})" if self.ds is not None else "None"
        test_ds_info = (
            f"Dataset(len={len(self.test_ds)})" if self.test_ds is not None else "None"
        )
        train_ds_info = (
            f"Dataset(len={len(self.train_ds)})"
            if self.train_ds is not None
            else "None"
        )
        predictions_info = (
            f"array(shape={self.test_predictions.shape})"
            if self.test_predictions is not None
            else "None"
        )
        return (
            f"{self.__class__.__name__}("
            f"horizon={self.horizon}, "
            f"df={df_info}, "
            f"ds={ds_info}, "
            f"test_ds={test_ds_info}, "
            f"train_ds={train_ds_info}, "
            f"score={self.score}, "
            f"test_predictions={predictions_info})"
        )


@dataclass
class PredictState(BaseState):
    """推論時専用の情報を追加した State"""

    target_date: Optional[date] = None
    predictions: Optional[np.ndarray] = None
    pred_df: Optional[pd.DataFrame] = None
    origin_ts: Optional[datetime] = None
    experiment_name: Optional[str] = None
    as_of_ts: Optional[datetime] = None

    def __repr__(self) -> str:
        df_info = f"shape: {self.df.shape}" if self.df is not None else "None"
        ds_info = f"Dataset(len={len(self.ds)})" if self.ds is not None else "None"
        predictions_info = (
            f"array(shape={self.predictions.shape})"
            if self.predictions is not None
            else "None"
        )
        pred_df_info = (
            f"DataFrame(shape={self.pred_df.shape})"
            if self.pred_df is not None
            else "None"
        )
        return (
            f"{self.__class__.__name__}("
            f"horizon={self.horizon}, "
            f"df={df_info}, "
            f"ds={ds_info}, "
            f"predictions={predictions_info}, "
            f"pred_df={pred_df_info})"
        )


StateLike = Union[TrainState, PredictState]
