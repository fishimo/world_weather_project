from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd


class DBManager:
    """DB操作全般を管理するクラス"""

    def __init__(self, config):
        """
        初期化
        
        args:
            table_name: db用テーブル名
            db_path: dbを保存するpath
        """
        self.table_name = str(config.database.table_name)
        self.db_path = Path(config.database.db_path)

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

        

    