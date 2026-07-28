from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

from portfolio.utils.pd_pl_utils import ensure_polars


class DBManager:
    """DB操作全般を管理するクラス"""

    def __init__(self, config):
        """
        初期化
        
        args:
            table_name: db用テーブル名
            db_path: dbを保存するpath
            ts_col: TimeStampを表すカラム名
        """
        self.table_name = str(config.database.table_name)
        self.db_path = Path(config.database.db_path)
        self.ts_col = str(config.database.ts_col)

    def processed_data_save(
        self,
        df: pd.DataFrame,
        *,
        if_exists:str = "replace", # 将来"append"や"fail"なども使い分けたい
        index: bool = False,
        ) -> None:
        """DataProcessorで加工済みのデータをdbに保存"""
        # db用のディレクトリ作成
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 入力チェック
        if df is None or df.empty:
            raise ValueError("保存するデータがありません")
        
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(self.table_name, conn, if_exists=if_exists, index=index)

    def load_all(self) -> pl.DataFrame:
        """加工データをDBから全て取り出す"""
        # db_pathの確認
        if not self.db_path.exists():
            raise FileNotFoundError(f"DB file not found: {self.db_path.resolve()}")
        
        print(f"Loading all data from {self.table_name}")
        # カラム名/テーブル名はパラメータ化できないので、存在確認してから使う
        with sqlite3.connect(self.db_path) as conn:
            # テーブル存在確認
            tbl = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=(self.table_name,),
            )
            if tbl.empty:
                raise ValueError(f"Table not found: {self.table_name}")

            # ts_col 存在確認
            schema = pd.read_sql_query(f"PRAGMA table_info({self.table_name})", conn)
            if self.ts_col not in set(schema["name"].tolist()):
                raise ValueError(
                    f"Column not found: {self.ts_col}. Available: {schema['name'].tolist()}"
                )

            sql = f"""
            SELECT *
            FROM {self.table_name}
            ORDER BY {self.ts_col}
            """

            pd_df = pd.read_sql_query(sql, conn)

        # pandas -> polars
        return ensure_polars(pd_df)

    def load_processed_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame:
        """
        欲しい日付を選択して、加工データをDBから取り出す

        args:
            start date: ロード開始日
            end date: ロード終了日

        return:
            pl.DataFrame: polarsのデータフレーム（パイプラインをpolarsで作成する予定のため）
        """
        # 指定日付の確認
        if start_date >= end_date:
            raise ValueError(
                f"start_date must be earlier than end_date. start_date={start_date}, end_date={end_date}"
            )
        print(f"Loading data slice: start={start_date}, end={end_date}")

        if not self.db_path.exists():
            raise FileNotFoundError(f"DB file not found: {self.db_path.resolve()}")

        # SQLiteのTEXT日時比較を想定：ISO形式に寄せる
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

        # カラム名/テーブル名はパラメータ化できないので、存在確認してから使う
        with sqlite3.connect(self.db_path) as conn:
            # テーブル存在確認
            tbl = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=(self.table_name,),
            )
            if tbl.empty:
                raise ValueError(f"Table not found: {self.table_name}")

            # ts_col 存在確認
            schema = pd.read_sql_query(f"PRAGMA table_info({self.table_name})", conn)
            if self.ts_col not in set(schema["name"].tolist()):
                raise ValueError(
                    f"Column not found: {self.ts_col}. Available: {schema['name'].tolist()}"
                )

            sql = f"""
            SELECT *
            FROM {self.table_name}
            WHERE {self.ts_col} >= ?
              AND {self.ts_col} < ?
            ORDER BY {self.ts_col}
            """

            pd_df = pd.read_sql_query(sql, conn, params=(start_str, end_str))

        if pd_df.empty:
            raise ValueError(
                "No processed data found in the requested range: "
                f"start={start_str}, end={end_str}, table={self.table_name}"
            )

        # pandas -> polars
        return ensure_polars(pd_df)

        

        

    
