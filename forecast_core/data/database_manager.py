from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

from forecast_core.utils.pd_pl_utils import ensure_polars

# predictionsテーブルの列順（DDL・INSERT・戻り値で共通に使う）
PREDICTION_COLUMNS: list[str] = [
    "station_id",
    "model_id",
    "origin_at",
    "target_at",
    "wind_speed",
    "created_at",
]

# 複合主キー。origin_at を含めることで「いつ時点の予測か」を区別して蓄積できる。
# 含めないと同じ target_at への再予測が上書きになり、リードタイム別の履歴が残らない。
# horizon は target_at - origin_at で導出できるため列には持たない。
PREDICTION_KEY_COLUMNS: list[str] = [
    "station_id",
    "model_id",
    "origin_at",
    "target_at",
]

# station_id / model_id は将来 stations / models テーブルへの FK にする想定。
# 現時点では参照先テーブルが無いため REFERENCES は張らない
# （張ると PRAGMA foreign_keys=ON 時に INSERT が必ず失敗するため）
_PREDICTION_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    station_id  TEXT      NOT NULL,
    model_id    TEXT      NOT NULL,
    origin_at   TIMESTAMP NOT NULL,
    target_at   TIMESTAMP NOT NULL,
    wind_speed  REAL      NOT NULL,
    created_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (station_id, model_id, origin_at, target_at)
)
"""


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
        self.prediction_table_name = str(config.prediction_database.table_name)
        self.prediction_db_path = Path(config.prediction_database.db_path)

    def processed_data_save(
        self,
        df: pd.DataFrame,
        *,
        if_exists: str = "replace",  # 将来"append"や"fail"なども使い分けたい
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
                    f"Column not found: {self.ts_col}. "
                    f"Available: {schema['name'].tolist()}"
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
            pl.DataFrame: polarsのデータフレーム
                （パイプラインをpolarsで作成する予定のため）
        """
        # 指定日付の確認
        if start_date >= end_date:
            raise ValueError(
                "start_date must be earlier than end_date. "
                f"start_date={start_date}, end_date={end_date}"
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
                    f"Column not found: {self.ts_col}. "
                    f"Available: {schema['name'].tolist()}"
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

    # ---------- predictions ----------

    def _assert_prediction_schema(self, conn: sqlite3.Connection) -> None:
        """既存テーブルが現行スキーマかを確認する。

        CREATE TABLE IF NOT EXISTS は既存テーブルを黙って素通りするため、
        旧スキーマのDBに書くと INSERT で分かりにくいエラーになる。ここで先に落とす。
        """
        schema = pd.read_sql_query(
            f"PRAGMA table_info({self.prediction_table_name})", conn
        )
        existing = set(schema["name"].tolist())
        missing = [col for col in PREDICTION_COLUMNS if col not in existing]
        if missing:
            raise ValueError(
                f"{self.prediction_table_name} テーブルのスキーマが古いです"
                f"（不足カラム: {missing}）。"
                f"{self.prediction_db_path} を削除して作り直してください。"
            )

    def _normalize_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """予測DataFrameをpredictionsテーブルのスキーマに整形・検証する"""
        missing = [col for col in PREDICTION_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"予測データに必要なカラムがありません: {missing}. "
                f"指定されているカラム: {list(df.columns)}"
            )

        out = df.loc[:, PREDICTION_COLUMNS].copy()

        # NOT NULL 制約を DB に到達する前に検出する
        null_cols = [col for col in PREDICTION_COLUMNS if out[col].isna().any()]
        if null_cols:
            raise ValueError(f"NOT NULL 列に欠損があります: {null_cols}")

        # TIMESTAMP は SQLite の TEXT 比較が効くよう ISO 形式に揃える
        for col in ("origin_at", "target_at", "created_at"):
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # 予測は基点より先の時刻を指すはずで、逆転はデータ組み立て側のバグ
        inverted = out["target_at"] <= out["origin_at"]
        if bool(inverted.any()):
            sample = out.loc[inverted, ["origin_at", "target_at"]].head(3)
            raise ValueError(
                "target_at が origin_at 以前になっている行があります: "
                f"{sample.to_dict('records')}"
            )

        out["station_id"] = out["station_id"].astype(str)
        out["model_id"] = out["model_id"].astype(str)
        out["wind_speed"] = out["wind_speed"].astype(float)

        # 同一バッチ内に主キー重複があると UPSERT が自己衝突するため先に潰す
        return out.drop_duplicates(subset=PREDICTION_KEY_COLUMNS, keep="last")

    def predictions_data_save(self, df: pd.DataFrame) -> int:
        """
        予測結果をpredictionsテーブルに保存する。

        同一の (station_id, model_id, origin_at, target_at) が既にある場合は
        wind_speed / created_at を上書きする（同じ条件での再実行を想定）。
        origin_at が違えば別行として蓄積されるため、同じ target_at に対する
        リードタイム別の予測履歴が残る。

        args:
            df: PREDICTION_COLUMNS を含むDataFrame

        return:
            int: 保存した行数
        """
        if df is None or df.empty:
            raise ValueError("保存するデータがありません")

        rows = self._normalize_predictions(df)

        self.prediction_db_path.parent.mkdir(parents=True, exist_ok=True)

        cols = ", ".join(PREDICTION_COLUMNS)
        placeholders = ", ".join("?" for _ in PREDICTION_COLUMNS)
        conflict = ", ".join(PREDICTION_KEY_COLUMNS)
        sql = f"""
        INSERT INTO {self.prediction_table_name} ({cols})
        VALUES ({placeholders})
        ON CONFLICT({conflict}) DO UPDATE SET
            wind_speed = excluded.wind_speed,
            created_at = excluded.created_at
        """

        with sqlite3.connect(self.prediction_db_path) as conn:
            conn.execute(_PREDICTION_DDL.format(table=self.prediction_table_name))
            self._assert_prediction_schema(conn)
            conn.executemany(sql, rows.itertuples(index=False, name=None))

        return len(rows)

    def load_predictions(
        self,
        *,
        station_id: str | None = None,
        model_id: str | None = None,
        origin_at: datetime | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """
        predictionsテーブルから予測結果を読み出す。

        args:
            station_id / model_id: 指定時はその値で絞り込む
            origin_at: 指定時はその基点日時の予測のみ取り出す
            start / end: target_at の範囲（start <= target_at < end）

        return:
            pl.DataFrame: (origin_at, target_at)昇順の予測結果
        """
        if not self.prediction_db_path.exists():
            raise FileNotFoundError(
                f"Prediction DB file not found: {self.prediction_db_path.resolve()}"
            )

        conditions: list[str] = []
        params: list[str] = []
        if station_id is not None:
            conditions.append("station_id = ?")
            params.append(station_id)
        if model_id is not None:
            conditions.append("model_id = ?")
            params.append(model_id)
        if origin_at is not None:
            conditions.append("origin_at = ?")
            params.append(origin_at.strftime("%Y-%m-%d %H:%M:%S"))
        if start is not None:
            conditions.append("target_at >= ?")
            params.append(start.strftime("%Y-%m-%d %H:%M:%S"))
        if end is not None:
            conditions.append("target_at < ?")
            params.append(end.strftime("%Y-%m-%d %H:%M:%S"))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
        SELECT {", ".join(PREDICTION_COLUMNS)}
        FROM {self.prediction_table_name}
        {where}
        ORDER BY origin_at, target_at
        """

        with sqlite3.connect(self.prediction_db_path) as conn:
            tbl = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=(self.prediction_table_name,),
            )
            if tbl.empty:
                raise ValueError(f"Table not found: {self.prediction_table_name}")

            pd_df = pd.read_sql_query(sql, conn, params=tuple(params))

        return ensure_polars(pd_df)
