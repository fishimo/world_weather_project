"""気象庁過去データ(HTML)から指定期間のAMeDASデータを取得して rawデータCSVに保存する。

使用例:
    # 単月
    python scripts/fetch_historical.py --year 2025 --month 4

    # 期間指定
    python scripts/fetch_historical.py --from 2024-01 --to 2025-04
"""

from __future__ import annotations

import argparse
import sys

from forecast_core.config.config import Config
from forecast_core.data.amedas_fetcher import (
    AmedasHistoricalFetcher,
    AmedasRawDataWriter,
)


def _parse_ym(ym: str) -> tuple[int, int]:
    parts = ym.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"年月は YYYY-MM 形式で指定してください: {ym}")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="気象庁過去データ(HTML)から指定期間のAMeDASデータを取得してCSVに保存する。"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="対象年 (--month と合わせて使用)")
    group.add_argument(
        "--from",
        dest="from_ym",
        metavar="YYYY-MM",
        help="取得開始年月 (例: 2024-01)",
    )
    parser.add_argument("--month", type=int, help="対象月 (--year と合わせて使用)")
    parser.add_argument(
        "--to",
        dest="to_ym",
        metavar="YYYY-MM",
        help="取得終了年月 (例: 2025-04、省略時は --from と同じ月)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    fetcher = AmedasHistoricalFetcher(config)
    writer = AmedasRawDataWriter(config)

    if args.year is not None:
        if args.month is None:
            print("[ERROR] --year を使う場合は --month も指定してください")
            sys.exit(1)
        print(f"取得対象: {args.year}/{args.month:02d}")
        df = fetcher.fetch_month(args.year, args.month)
    else:
        from_year, from_month = _parse_ym(args.from_ym)
        to_ym = args.to_ym if args.to_ym else args.from_ym
        to_year, to_month = _parse_ym(to_ym)
        print(f"取得対象: {from_year}/{from_month:02d} ～ {to_year}/{to_month:02d}")
        df = fetcher.fetch_range(from_year, from_month, to_year, to_month)

    print(f"取得完了: {len(df)} 行")
    writer.save_raw_data(df)


if __name__ == "__main__":
    main()
