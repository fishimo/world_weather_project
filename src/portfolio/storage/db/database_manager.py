

from pathlib import Path


class DBManager:
    """DB操作全般を管理するクラス"""

    def __init__(self, table_name: str, db_path: Path=None):
        """
        初期化
        
        args:
            table_name: db用テーブル名
            db_path: dbを保存するpath
        """
        self.config = config.db_manager

    def processed_data_save(self):
        """DataProcessorで加工済みのデータをdbに保存"""
        

    