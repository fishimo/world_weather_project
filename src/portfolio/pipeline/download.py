from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from configs.config import Config
from portfolio.data.data_downloader import DataDownloader
from portfolio.data.data_processor import DataProcessor
from portfolio.data.database_manager import DBManager


def main() -> None:
    config = Config()

    print("[1/4] download raw data")
    downloader = DataDownloader(config)
    downloader.data_download()

    print("[2/4] read raw data")
    processor = DataProcessor(config)
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
