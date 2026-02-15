from __future__ import annotations

import time

from configs.config import config
from src.portfolio.ingest.data_downloader import DataDownloader
from src.portfolio.ingest.data_processor import DataProcessor
from src.portfolio.storage.db.database_manager import DBManager

def _fmt_sec(sec: float) -> str:
    """timeを分かりやすく表示するヘルパー関数"""
    # 例: 12.3s / 3m21s / 1h02m03s
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main() -> None:
    t0 = time.perf_counter()

    # download
    t = time.perf_counter()
    print("[1/4] data download start")
    downloader = DataDownloader(config)
    downloader.data_download()
    print(f"[1/4] data download done({_fmt_sec(time.perf_counter() - t)})")

    # read
    t = time.perf_counter()
    print("[2/4] data read start")
    processor = DataProcessor(config)
    df = processor.read_data()
    print(f"[2/4] data read done \nrows={len(df):,} \ncols={df.shape[1]} \n({_fmt_sec(time.perf_counter() - t)}")


    # process
    t = time.perf_counter()
    print("[3/4] data process start")
    df_processed = processor.process_data(df)
    print(f"[3/4] data process done \nrows={len(df_processed):,} \ncols={df_processed.shape[1]} \n({_fmt_sec(time.perf_counter() - t)})")

    # db save
    t = time.perf_counter()
    print("[4/4] db save start")
    db = DBManager(config)
    db.processed_data_save(df_processed)
    print(f"[4/4] db save done({_fmt_sec(time.perf_counter() - t)})")

    print(f"[ALL] finished ({_fmt_sec(time.perf_counter() - t0)})")


if __name__ == "__main__":
    main()