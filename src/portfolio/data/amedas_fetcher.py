from __future__ import annotations

import calendar
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

_JST = timezone(timedelta(hours=9))
_HISTORICAL_BASE = "https://www.data.jma.go.jp/stats/etrn/view/hourly_s1.php"
_AMEDAS_JSON_BASE = "https://www.jma.go.jp/bosai/amedas/data/point"

# AMeDAS JSON 風向コード(1-16) → 度数（0=静穏, 16=N→0°）
_WIND_CODE_TO_DEG: dict[int, float] = {k: k * 22.5 for k in range(1, 16)}
_WIND_CODE_TO_DEG[0] = 0.0
_WIND_CODE_TO_DEG[16] = 0.0

# 気象庁HTML 16方位テキスト → 度数
_WIND_TEXT_TO_DEG: dict[str, float] = {
    "静穏": 0.0,   "北北東": 22.5,  "北東": 45.0,   "東北東": 67.5,
    "東": 90.0,    "東南東": 112.5, "南東": 135.0,  "南南東": 157.5,
    "南": 180.0,   "南南西": 202.5, "南西": 225.0,  "西南西": 247.5,
    "西": 270.0,   "西北西": 292.5, "北西": 315.0,  "北北西": 337.5,
    "北": 0.0,
}

# AMeDAS JSON の3時間ブロック開始時刻
_BLOCK_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)

# ---------- 物理量マッピング ----------
# 新フィールド追加時はここと RAW_COLUMNS を合わせて更新する

# AMeDAS JSON フィールド名 → raw CSV カラム名
# windDirection/wind は風向・風速として別途処理するためここには含めない
_JSON_VALUE_FIELDS: dict[str, str] = {
    "temp": "temperature_celsius",
    "humidity": "humidity_pct",
    "pressure": "pressure_mb",
    # 拡張例: "precipitation1h": "precipitation_1h_mm",
}

# HTML カラムキーワード → raw CSV カラム名
# wind 系は別途処理するためここには含めない
# value: (include_keywords, exclude_keywords)
_HTML_VALUE_FIELDS: dict[str, tuple[list[str], list[str] | None]] = {
    "temperature_celsius": (["気温"], None),
    "humidity_pct":        (["湿度"], None),
    # 気圧は現地優先; _HTML_PRESSURE_FALLBACK で海面を代替
    "pressure_mb":         (["気圧", "現地"], None),
}
_HTML_PRESSURE_FALLBACK: list[str] = ["気圧"]

# raw CSV のカラム順（両fetcher共通）
RAW_COLUMNS: list[str] = [
    "timestamp", "station_id", "location_name", "latitude", "longitude",
    "temperature_celsius", "humidity_pct", "wind_mps", "wind_degree",
    "pressure_mb", "missing_flag",
]


# ---------- helpers ----------

def _to_float(value: object) -> float | None:
    """任意の値をfloatに変換。変換不能またはNaNの場合はNoneを返す。"""
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


def _json_value(field: object) -> float | None:
    """AMeDAS JSONフィールド [value, quality_flag] から値を取得する。"""
    if field is None:
        return None
    if isinstance(field, (list, tuple)) and len(field) >= 1:
        return _to_float(field[0])
    return _to_float(field)


def _find_html_col(
    flat_cols: list[str],
    include: list[str],
    exclude: list[str] | None = None,
) -> str | None:
    """フラット化されたHTMLカラム名リストからキーワードマッチで列名を返す。"""
    for col in flat_cols:
        if all(k in col for k in include):
            if exclude and any(e in col for e in exclude):
                continue
            return col
    return None


# ---------- fetchers ----------

class AmedasHistoricalFetcher:
    """気象庁過去データ(HTML)から日単位で取得し、月単位にまとめて返す。

    リクエストは1日分ずつ行い、間隔は config.request_interval_sec に従う。
    当月を指定した場合は今日以降の日付をスキップする。
    """

    def __init__(self, config) -> None:
        self._station = config.amedas_fetch.stations[0]
        self._interval = config.amedas_fetch.request_interval_sec

    def fetch_month(self, year: int, month: int) -> pd.DataFrame:
        """指定年月の全日分データを取得して返す。"""
        today = datetime.now(_JST).date()
        n_days = calendar.monthrange(year, month)[1]
        # 当月は今日まで、未来日付はスキップ
        last_day = today.day if (year, month) == (today.year, today.month) else n_days

        frames: list[pd.DataFrame] = []
        for day in range(1, last_day + 1):
            print(f"    {year}/{month:02d}/{day:02d} を取得中 ...")
            frames.append(self._fetch_day(year, month, day))
            if day < last_day:
                time.sleep(self._interval)
        empty = pd.DataFrame(columns=RAW_COLUMNS)
        return pd.concat(frames, ignore_index=True) if frames else empty

    def fetch_range(
        self,
        from_year: int,
        from_month: int,
        to_year: int,
        to_month: int,
    ) -> pd.DataFrame:
        """指定期間の月別データを結合して返す。"""
        frames: list[pd.DataFrame] = []
        y, m = from_year, from_month
        while (y, m) <= (to_year, to_month):
            print(f"  {y}/{m:02d} を取得中 ...")
            frames.append(self.fetch_month(y, m))
            m += 1
            if m > 12:
                m = 1
                y += 1
        empty = pd.DataFrame(columns=RAW_COLUMNS)
        return pd.concat(frames, ignore_index=True) if frames else empty

    def _fetch_day(self, year: int, month: int, day: int) -> pd.DataFrame:
        url = (
            f"{_HISTORICAL_BASE}"
            f"?prec_no={self._station.historical_prec_no}"
            f"&block_no={self._station.historical_block_no}"
            f"&year={year}&month={month}&day={day}&view="
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        # 気象庁 etrn ページは EUC-JP が多い
        try:
            html = resp.content.decode("euc-jp")
        except UnicodeDecodeError:
            html = resp.content.decode("utf-8", errors="replace")
        return self._parse_day_html(html, year, month, day)

    def _parse_day_html(
        self, html: str, year: int, month: int, day: int
    ) -> pd.DataFrame:
        try:
            tables = pd.read_html(
                StringIO(html),
                na_values=["--", "×", "///", " ", ""],
                header=[0, 1],
                flavor="lxml",
            )
        except Exception as exc:
            raise ValueError(
                f"{year}/{month:02d}/{day:02d} のHTMLパースに失敗: {exc}"
            ) from exc

        # 20行以上のテーブルを気象データとみなす
        candidates = [t for t in tables if len(t) >= 20]
        if not candidates:
            raise ValueError(
                f"{year}/{month:02d}/{day:02d} に時別テーブルが見つかりません "
                f"(table sizes={[len(t) for t in tables]})"
            )
        table = max(candidates, key=len)

        # MultiIndex カラムをフラット文字列に変換（"Unnamed" 部分は除外）
        if isinstance(table.columns, pd.MultiIndex):
            flat = [
                " ".join(str(c) for c in col if "Unnamed" not in str(c)).strip()
                for col in table.columns
            ]
        else:
            flat = [str(c) for c in table.columns]
        table.columns = flat

        hour_col = flat[0]
        wind_spd_col = _find_html_col(flat, ["風速"])
        wind_dir_col = _find_html_col(flat, ["風向"])
        if wind_spd_col is None or wind_dir_col is None:
            raise ValueError(
                f"{year}/{month:02d}/{day:02d}: 風速/風向カラムが見つかりません "
                f"(columns={flat})"
            )

        # _HTML_VALUE_FIELDS に基づいて各物理量のカラムを特定
        value_col_map: dict[str, str | None] = {}
        for raw_col, (include, exclude) in _HTML_VALUE_FIELDS.items():
            col = _find_html_col(flat, include, exclude)
            # 気圧は現地が見つからなければ海面で代替
            if col is None and raw_col == "pressure_mb":
                col = _find_html_col(flat, _HTML_PRESSURE_FALLBACK)
            value_col_map[raw_col] = col

        rows: list[dict] = []
        for _, row in table.iterrows():
            try:
                hour = int(float(str(row[hour_col])))
            except (ValueError, TypeError):
                continue
            if not 1 <= hour <= 24:
                continue

            # 時=24 は翌日0時
            if hour == 24:
                ts = (
                    datetime(year, month, day, 0, 0, 0, tzinfo=_JST)
                    + timedelta(days=1)
                )
            else:
                ts = datetime(year, month, day, hour, 0, 0, tzinfo=_JST)

            wind_mps = _to_float(row.get(wind_spd_col))
            wind_dir_text = str(row.get(wind_dir_col, "")).strip()
            wind_degree = _WIND_TEXT_TO_DEG.get(wind_dir_text)

            record: dict = {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "station_id": self._station.station_id,
                "location_name": self._station.name,
                "latitude": self._station.latitude,
                "longitude": self._station.longitude,
                "wind_mps": wind_mps,
                "wind_degree": wind_degree,
                "missing_flag": wind_mps is None,
            }
            for raw_col, src_col in value_col_map.items():
                record[raw_col] = _to_float(row.get(src_col) if src_col else None)

            rows.append(record)

        return pd.DataFrame(rows, columns=RAW_COLUMNS)


class AmedasLatestFetcher:
    """AMeDAS JSON APIから現在時刻を含む3時間ブロックのデータを取得する。

    毎時00分レコードのみ抽出し、1時間粒度で返す。
    """

    def __init__(self, config) -> None:
        self._stations = config.amedas_fetch.stations
        self._interval = config.amedas_fetch.request_interval_sec

    def fetch_latest(self) -> pd.DataFrame:
        """現在時刻を含む3時間ブロックの全地点データを取得して返す。"""
        now = datetime.now(_JST)
        block_hour = max(h for h in _BLOCK_HOURS if h <= now.hour)
        block_dt = now.replace(hour=block_hour, minute=0, second=0, microsecond=0)
        return self._fetch_block(block_dt)

    def _fetch_block(self, block_dt: datetime) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for i, station in enumerate(self._stations):
            url = (
                f"{_AMEDAS_JSON_BASE}/{station.station_id}/"
                f"{block_dt.strftime('%Y%m%d')}_{block_dt.strftime('%H')}.json"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            frames.append(self._parse_block(resp.json(), station))
            if i < len(self._stations) - 1:
                time.sleep(self._interval)
        empty = pd.DataFrame(columns=RAW_COLUMNS)
        return pd.concat(frames, ignore_index=True) if frames else empty

    def _parse_block(self, records: dict, station) -> pd.DataFrame:
        """JSONブロックから毎時00分のレコードのみ抽出してrawスキーマに整形する。"""
        rows: list[dict] = []
        for ts_str, obs in records.items():
            # キー形式: YYYYMMDDHHmmss (14桁)
            if len(ts_str) < 12:
                continue
            if int(ts_str[10:12]) != 0:  # 毎時00分のみ
                continue
            try:
                ts = datetime(
                    int(ts_str[0:4]), int(ts_str[4:6]), int(ts_str[6:8]),
                    int(ts_str[8:10]), 0, 0, tzinfo=_JST,
                )
            except ValueError:
                continue

            wind_mps = _json_value(obs.get("wind"))
            wind_code = _json_value(obs.get("windDirection"))
            wind_degree = (
                _WIND_CODE_TO_DEG.get(int(wind_code))
                if wind_code is not None
                else None
            )

            record: dict = {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "station_id": station.station_id,
                "location_name": station.name,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "wind_mps": wind_mps,
                "wind_degree": wind_degree,
                "missing_flag": wind_mps is None,
            }
            for json_field, raw_col in _JSON_VALUE_FIELDS.items():
                record[raw_col] = _json_value(obs.get(json_field))

            rows.append(record)

        return pd.DataFrame(rows, columns=RAW_COLUMNS)


class AmedasRawDataWriter:
    """AMeDAS rawデータCSVを管理する（追記・重複排除・時刻順ソート）。"""

    _DEDUP_KEYS = ["timestamp", "station_id"]

    def __init__(self, config) -> None:
        raw_dir = Path(config.amedas_fetch.raw_dir)
        self._path = raw_dir / config.amedas_fetch.raw_filename

    @property
    def raw_data_path(self) -> Path:
        return self._path

    def has_raw_data(self) -> bool:
        return self._path.is_file()

    def save_raw_data(self, df: pd.DataFrame) -> None:
        """新データを既存CSVに追記し、重複排除・時刻順ソートで保存する。"""
        if df.empty:
            return
        if self.has_raw_data():
            existing = pd.read_csv(self._path)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            combined = df.copy()

        combined = (
            combined
            .drop_duplicates(subset=self._DEDUP_KEYS, keep="last")
            .sort_values(["station_id", "timestamp"])
            .reset_index(drop=True)
        )
        combined.to_csv(self._path, index=False)
        print(f"raw CSV 保存: {self._path} ({len(combined)} 行)")
