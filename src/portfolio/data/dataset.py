from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import polars as pl
import pandas as pd
import re

from portfolio.utils.pd_pl_utils import ensure_pandas, ensure_polars


@dataclass
class Dataset:
    raw: Optional[pl.DataFrame] = None
    idx: Optional[pl.DataFrame] = None
    y: Optional[pd.Series] = None
    X: Optional[pd.DataFrame] = None
    y_expanded: Optional[pd.DataFrame] = None
    original_expanded: Optional[pd.DataFrame] = None
    feature_list: Optional[List[str]] = None

    @classmethod
    def make_dataset(
        cls,
        df: pl.DataFrame,
        y_col: str = "wind_mps_daily_mean",
        idx_col: Optional[List[str]] = None,
    ) -> "Dataset":
        "生データからDatasetを作成するファクトリメソッド"
        if idx_col is None:
            idx_col = ["date", "location_name"]
        
        df_pd = ensure_pandas(df).copy()
        if not isinstance(df_pd, pd.DataFrame):
            raise TypeError("df には pandas DataFrame が必須です")
        
        df_pd["date"] = pd.to_datetime(df_pd["date"])
        df_pd = df_pd.sort_values(["date", "location_name"]).reset_index(drop=True)

        feature_cols = [c for c in df_pd.columns if c not in idx_col + [y_col]]
        X_cols = [col for col in feature_cols if df_pd[col].dtype != "object"]
        X_df = df_pd[X_cols] if X_cols else pd.DataFrame(index=df_pd.index)

        return cls(
            raw = df,
            idx = df_pd[idx_col].reset_index(drop=True),
            y = df_pd[y_col].reset_index(drop=True),
            X = X_df.reset_index(drop=True),
            feature_list=X_cols,
        )
    
    @property
    def date(self) -> pd.Series:
        return self.idx["date"] if self.idx is not None else pd.Series([])

    def __getitem__(self, key: pd.Series | slice) -> "Dataset":
        """スライシングまたはブールインデックスでフィルタリング"""
        if self.idx is None:
            raise ValueError("idx が必要です")
        
        if isinstance(key, pd.Series):
            mask = key.reindex(self.idx.index, fill_value=False)
        elif isinstance(key, slice):
            mask = pd.Series(False, index=self.idx.index)
            mask.iloc[key] = True
        else:
            raise TypeError(f"indexのtypeがサポートされていません: {type(key)}")
        
        def get_masked(
                df_or_series: pd.DataFrame | pd.Series | None,
                mask_series: pd.Series,
        ) -> pd.DataFrame | pd.Series | None:
            if df_or_series is None:
                return None
            mask_aligned = mask_series.reindex(df_or_series.index, fill_value=False)
            return df_or_series[mask_aligned].reset_index(drop=True)
        
        return Dataset(
            raw=self.raw,
            idx=get_masked(self.idx, mask),
            y=get_masked(self.y, mask),
            y_expanded=get_masked(self.y_expanded, mask),
            X=get_masked(self.X, mask),
            feature_list=(
                self.feature_list.copy() if self.feature_list is not None else None
            ),
        )
    
    def __len__(self) -> int:
        if self.idx is None:
            return 0
        return len(self.idx) 


class DatasetGenerator:
    def __init__(
            self,
            horizon: List[int],
            y_col: str = "wind_mps_daily_mean",
            idx_col: List[str] | None = None,
    ):
        self.horizon = horizon
        self.y_col = y_col
        self.idx_col = (
            idx_col if idx_col is not None else ["date", "location_name"]
        )

        self.regex_y = rf"^{re.escape(y_col)}_\d+$"

    def match_y(self, df: pd.DataFrame) -> List[str]:
        """horizon展開後のy列抽出"""
        pattern = re.compile(self.regex_y)
        return [col for col in df.columns if pattern.match(col)]
    
    def prepare_dataset(self, df: pl.DataFrame) -> Dataset:
        """polars DataFrameをdatasetに変換"""
        df_pd = ensure_pandas(df).copy()
        if not isinstance(df_pd, pd.DataFrame):
            raise TypeError("df には pandas DataFrame が必須です")
        
        if len(self.match_y(df_pd)) > 0:
            raise ValueError("特徴量に y が含まれています")
        
        df_pd["date"] = pd.to_datetime(df_pd["date"])
        df_pd = df_pd.sort_values(["date", "location_name"]).reset_index(drop=True)

        feature_cols = [
            c for c in df_pd.columns if c not in self.idx_col + [self.y_col]
        ]
        # objectは特徴量に入れない
        X_cols = []
        for col in feature_cols:
            if df_pd[col].dtype == "object":
                continue
            X_cols.append(col)
        
        X_df = df_pd[X_cols] if X_cols else pd.DataFrame(index=df_pd.index)

        return Dataset(
            raw=df,
            idx=df_pd[self.idx_col].reset_index(drop=True),
            y=df_pd[self.y_col].reset_index(drop=True),
            X=X_df.reset_index(drop=True),
            feature_list=X_cols,
        )
    
    def expand_horizon(
            self, ds: Dataset, targets: Optional[Sequence[str]] = None
    ) -> Dataset:
        """
        yについて "date", "location_name" ごとに
        horizonの数だけshift(-h)する

        args: 
            - dataset: 展開対象のDataset

        return:
            - horizon展開後のDataset 
        """
        if ds.idx is None or ds.y is None:
            raise ValueError("idx と y は必須です")

        df_pd = pd.concat([ds.idx, ds.y.to_frame(name=self.y_col)], axis=1)

        # pandas df -> polars df
        df_pl = ensure_polars(df_pd)
        df_pl = df_pl.sort(["date", "location_name"])

        df_sorted = ensure_pandas(df_pl)

        # y_expandedを作成
        y_expanded_cols = []
        for h in self.horizon:
            col_name = f"{self.y_col}_{h}"
            y_expanded_cols.append(
                pl.col(self.y_col).shift(-h).over("location_name").alias(col_name)
            )
        df_y_expanded = df_pl.select(y_expanded_cols)
        y_expanded = df_y_expanded.to_pandas()
        
        # TO DO: 残差学習したらここにoriginalのexpandも入れる

        # datasetの更新
        ds.idx = df_sorted[self.idx_col].reset_index(drop=True)
        ds.y = df_sorted[self.y_col].reset_index(drop=True)
        if ds.X is not None:
            # TO DO: ここでXがdf_sotedに入っていないので更新の意味がない
            # Xの列が存在するときのみ更新
            X_cols = [col for col in ds.X.columns if col in df_sorted.columns]
            if X_cols:
                ds.X = df_sorted[X_cols].reset_index(drop=True)
        ds.y_expanded = y_expanded.reset_index(drop=True)
        
        return ds
    
    def split(self, ds: Dataset, test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        """
        Datasetを分割してtrain_dsとtest_dsを返す

        args: 
            - ds: 分割対象のDataset
            - test_size: test_dのサイズ(0.0～1.0)

        return:
            - train_ds: 訓練用Dataset
            - test_ds: テスト用Dataset
        """
        if test_size < 0.0 or test_size > 1.0:
            raise ValueError("testサイズは0.0～1.0内で指定してください")
        if ds.idx is None or ds.y is None:
            raise ValueError("idx と y が必要です")
        if "date" not in ds.idx.columns:
            raise ValueError("idx に date 列が必要です")
        
        unique_ts = ds.date.unique()
        test_n = int(len(unique_ts) * test_size)
        split_ts = unique_ts[-test_n]

        train_ds = ds[ds.date < split_ts]
        test_ds = ds[ds.date >= split_ts]

        return train_ds, test_ds
    
    def filter_trainable_rows(self, ds: Dataset) -> Dataset:
        if ds.idx is None or ds.X is None or ds.y is None or ds.y_expanded is None:
            raise ValueError("dataset must have idx, X, y and y_expanded")

        mask = ~ds.idx.isna().any(axis=1)
        mask &= ~ds.y.isna()
        mask &= ~ds.y_expanded.isna().any(axis=1)
        return ds[mask]

