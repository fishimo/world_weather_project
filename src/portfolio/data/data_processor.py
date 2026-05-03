from __future__ import annotations

from pathlib import Path
import pandas as pd


class DataProcessor:
    def __init__(self, config):
        self.rawdata_path = Path(config.data_process.rawdata_path)
        self.selected_cols = config.data_process.selected_cols
        self.selected_country = config.data_process.selected_country

    def _select_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """欲しい国だけ抽出するヘルパー関数"""
        if self.selected_country is None:
            raise ValueError("選択されている国がありません。")
        df_selected_country = df[df["country"].isin(self.selected_country)]
        return df_selected_country

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欲しいカラムを抽出するヘルパー関数
        カラム名はconfigで指定
        念のためないカラムも表示

        args:
            df: 加工対象df

        return:
            df_selected_cols: 加工後df
        """
        # 念のため無い対象カラムも表示 
        missing = [c for c in self.selected_cols if c not in df.columns]
        if missing:
            print(f"[WARN] missing columns (not found in df): {missing}")
        # 対象カラムのみ抽出    
        existing_cols = [c for c in self.selected_cols if c in df.columns]
        df_selected_cols = df.loc[:, existing_cols]
        return df_selected_cols
    
    def _convert_windspeed(self, df: pd.DataFrame) -> pd.DataFrame:
        """wind_kphをwind_mpsに変換するヘルパー関数"""
        if "wind_kph" not in df.columns:
            raise KeyError("風速km/hのカラムが見つかりません。")
        df["wind_mps"] = df["wind_kph"]*1000/3600  # km=1000 m / h=3600 s より
        
        return df 
    
    def _check_missing(self, df: pd.DataFrame) -> None:
        """カラムごとの欠損値を表示するヘルパー関数"""
        nan_count = df.isna().sum()
        nan_rate = df.isna().mean()

        for col in df.columns:
            print(f"{col}: missing={nan_count[col]} ({nan_rate[col]*100:.2f} %)")

    def read_data(self) -> pd.DataFrame:
        """
        raw_dataをdfに変換
        
        return:
            pd.DataFrame: 生データdf
        """
        if not self.rawdata_path.is_file():
            raise FileNotFoundError(f"生データのファイルが見つかりません: {self.rawdata_path}")
        return pd.read_csv(self.rawdata_path)

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データの加工を行う
        中身は国選択->カラム選択->風速単位変換->欠損値表示
        """
        df_processed = self._select_country(df)
        df_processed = self._select_columns(df_processed)
        df_processed = self._convert_windspeed(df_processed)
        self._check_missing(df_processed)
        return df_processed