import polars as pl



class PreProcessor:
    def __init__(self, config):
        self.ts_col = str(config.preprocess.ts_col)
        self.wind_col = str(config.preprocess.wind_col)
        self.require_cols = config.preprocess.require_cols

    def _ensure_datetime(self, df: pl.DataFrame) -> pl.DataFrame:
        """ts_colがdatetimeであることを保障する"""
        dtype = df.schema[self.ts_col]

        if dtype == pl.Datetime:
            return df

        if dtype == pl.Utf8:
            return df.with_columns(
                pl.col(self.ts_col)
                .str.strptime(pl.Datetime, strict=False)
                .alias(self.ts_col)
            )

        raise TypeError(
            f"{self.ts_col} must be pl.Datetime or pl.Utf8, got {dtype}"
        )
    
    def select_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """使うカラムを選択"""
        if self.require_cols is None:
            return df
        missing_cols = [col for col in self.require_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"require_colsに指定されているカラムが見つかりません: {missing_cols}. "
                f"指定されているカラム: {df.columns}"
            )
        
        return df.select(self.require_cols)

    # def ave_day_wind(self, df: pl.DataFrame) -> pl.DataFrame:
    #     """風速を日平均する（1日1行に集約）"""
    #     if self.ts_col not in df.columns:
    #         raise ValueError(f"timestamp カラムが見つかりません: {self.ts_col}")
    #     if self.wind_col not in df.columns:
    #         raise ValueError(f"wind カラムが見つかりません: {self.wind_col}")
        
    #     out = self._ensure_datetime(df)
    #     out = (
    #         out.with_columns(
    #             pl.col(self.ts_col).dt.date().alias("date")
    #         )
    #         .group_by("date")
    #         .agg(
    #             pl.col(self.wind_col).mean().alias(f"{self.wind_col}_daily_mean")
    #         )
    #         .sort("date")
    #     )
    #     return out
    
    def ave_day_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        数値カラムを日平均して、1日1地点1行に集約する。
        """
        key_cols = ["date", "location_name"]
        out = self._ensure_datetime(df).with_columns(
        pl.col(self.ts_col).dt.date().alias("date")
        )

        # 平均対象の数値列を選ぶ
        candidate_cols = self.require_cols if self.require_cols is not None else out.columns
        numeric_cols = [
            col for col in candidate_cols
            if col not in key_cols + [self.ts_col]
            and out.schema[col] in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            )
        ]

        if not numeric_cols:
            raise ValueError("日平均できる数値列がありません")
        
        agg_exprs = [
            pl.col(col).mean().alias(f"{col}_daily_mean")
            for col in numeric_cols
        ]

        return (
            out.group_by(key_cols)
            .agg(agg_exprs)
            .sort(key_cols)
        )

    def convert_type(self, df: pl.DataFrame) -> pl.DataFrame:
        """文字列(Utf8)カラムを Categorical に変換"""
        return df.with_columns(
            pl.col(pl.Utf8).cast(pl.Categorical)
        )
