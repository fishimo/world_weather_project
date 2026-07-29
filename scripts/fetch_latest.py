"""AMeDAS JSON APIから最新の3時間ブロックデータを取得して rawデータCSVに保存する。

毎時00分レコードのみ抽出して保存するため、cronで毎時25分頃に実行することを想定。

使用例:
    python scripts/fetch_latest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from configs.config import Config
from portfolio.data.amedas_fetcher import AmedasLatestFetcher, AmedasRawDataWriter


def main() -> None:
    config = Config()
    fetcher = AmedasLatestFetcher(config)
    writer = AmedasRawDataWriter(config)

    print("AMeDAS 最新データを取得中 ...")
    df = fetcher.fetch_latest()
    print(f"取得完了: {len(df)} 行")
    writer.save_raw_data(df)


if __name__ == "__main__":
    main()
