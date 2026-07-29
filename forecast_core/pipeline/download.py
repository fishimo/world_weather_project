from __future__ import annotations

from forecast_core.config.config import Config
from forecast_core.data.amedas_fetcher import AmedasLatestFetcher, AmedasRawDataWriter
from forecast_core.data.database_manager import DBManager
from forecast_core.preprocessing.amedas_processor import AmedasProcessor


def main() -> None:
    config = Config()

    print("[1/4] fetch latest AMeDAS data")
    fetcher = AmedasLatestFetcher(config)
    writer = AmedasRawDataWriter(config)
    try:
        df_raw = fetcher.fetch_latest()
        writer.save_raw_data(df_raw)
    except Exception as e:
        if writer.has_raw_data():
            print(f"[WARN] fetch 失敗、既存 raw データで続行: {e}")
        else:
            raise

    print("[2/4] read raw data")
    processor = AmedasProcessor(config)
    raw_df = processor.read_data()
    print(f"loaded rows={len(raw_df)}, cols={raw_df.shape[1]}")

    print("[3/4] process data")
    processed_df = processor.process_data(raw_df)
    print(f"processed rows={len(processed_df)}, cols={processed_df.shape[1]}")

    print("[4/4] save to DB")
    db_manager = DBManager(config)
    db_manager.processed_data_save(processed_df, if_exists="replace")
    print(f"saved DB: {config.database.db_path}")


if __name__ == "__main__":
    main()
