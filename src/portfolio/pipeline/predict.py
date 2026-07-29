from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from configs.config import Config
from portfolio.data.amedas_fetcher import AmedasLatestFetcher, AmedasRawDataWriter
from portfolio.data.amedas_processor import AmedasProcessor
from portfolio.data.database_manager import DBManager
from portfolio.data.dataset import Dataset, DatasetGenerator
from portfolio.features.feature_preprocessor import PreProcessor
from portfolio.models.model_store import ModelStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download latest raw data, build DB, and run prediction."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("test_models/test_model.pkl"),
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default=None,
        help=(
            "Base date to predict from. Example: 2026-07-01. "
            "Defaults to the latest date available in the loaded slice."
        ),
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help=(
            "Prediction data start datetime. Example: 2026-04-01 00:00:00. "
            "Derived from --origin when omitted."
        ),
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help=(
            "Prediction data end datetime. Example: 2026-04-21 00:00:00. "
            "Derived from --origin when omitted."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model identifier stored in predictions. Defaults to model file stem.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip the AMeDAS fetch step and predict from the existing DB only.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional CSV dump of the prediction output (DB save always runs).",
    )
    return parser.parse_args()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _resolve_window(
    start: str | None,
    end: str | None,
    origin: datetime | None,
) -> tuple[datetime, datetime]:
    """DBから読み出す期間を決める。

    --start/--end が両方指定されていればそれを使う。省略された場合は
    --origin の当日1日分（origin 00:00 <= t < 翌日 00:00）を読む。
    日平均は origin 当日の時別データだけで確定するため、これで足りる。
    """
    if start is not None and end is not None:
        return _parse_datetime(start), _parse_datetime(end)

    if start is not None or end is not None:
        raise ValueError("--start と --end は両方指定してください")

    if origin is None:
        raise ValueError("--origin か、--start と --end の組を指定してください")

    origin_day = origin.replace(hour=0, minute=0, second=0, microsecond=0)
    return origin_day, origin_day + timedelta(days=1)


def _to_prediction_records(
    output_df: pd.DataFrame,
    station_id_map: dict[str, str],
    model_id: str,
    created_at: datetime,
) -> pd.DataFrame:
    """予測結果を predictions テーブルのスキーマに変換する。"""
    required = ("location_name", "origin_date", "timestamp", "y_pred")
    missing = [col for col in required if col not in output_df.columns]
    if missing:
        raise ValueError(f"Prediction output requires columns: {missing}")

    names = output_df["location_name"].astype(str)
    unknown = sorted(set(names) - set(station_id_map))
    if unknown:
        raise ValueError(
            f"config に定義されていない観測地点です: {unknown}. "
            f"既知の地点: {sorted(station_id_map)}"
        )

    return pd.DataFrame(
        {
            "station_id": names.map(station_id_map),
            "model_id": model_id,
            "origin_at": output_df["origin_date"],
            "target_at": output_df["timestamp"],
            "wind_speed": output_df["y_pred"],
            "created_at": created_at,
        }
    )


def _align_features(df: pd.DataFrame, feature_list: list[str] | None) -> pd.DataFrame:
    if feature_list is None:
        return df

    aligned = df.copy()
    for col in feature_list:
        if col not in aligned.columns:
            aligned[col] = 0.0

    return aligned.loc[:, feature_list]


def _build_prediction_output(
    idx_df: pd.DataFrame | None,
    predictions: pd.DataFrame,
    horizon: list[int],
) -> pd.DataFrame:
    if idx_df is None:
        idx_df = pd.DataFrame(index=range(len(predictions)))

    output_frames: list[pd.DataFrame] = []
    base_df = idx_df.reset_index(drop=True).copy()

    if "date" in base_df.columns:
        base_ts = pd.to_datetime(base_df["date"])
    else:
        raise ValueError("Prediction output requires 'date' column in ds.idx.")

    for h in horizon:
        col_name = f"pred_h{h}"
        if col_name not in predictions.columns:
            raise ValueError(f"Prediction column not found: {col_name}")

        out_h = base_df.copy()
        out_h["origin_date"] = base_ts
        out_h["timestamp"] = base_ts + pd.to_timedelta(h, unit="D")
        out_h["horizon"] = h
        out_h["y_pred"] = predictions[col_name].to_numpy()
        output_frames.append(out_h)

    output_df = pd.concat(output_frames, axis=0, ignore_index=True)
    sort_cols = ["timestamp"]
    if "location_name" in output_df.columns:
        sort_cols.append("location_name")
    return output_df.sort_values(sort_cols, ignore_index=True)


def _select_origin_rows(ds: Dataset, origin: datetime | None) -> Dataset:
    """予測の基点となる日の行だけを取り出す。

    origin が None のときは読み込んだ範囲の最新日を使う。
    """
    idx_df = ds.idx
    if idx_df is None or "date" not in idx_df.columns:
        raise ValueError("Prediction dataset requires 'date' column in ds.idx.")

    dates = pd.to_datetime(idx_df["date"])
    if origin is None:
        origin_date = dates.max()
    else:
        origin_date = pd.Timestamp(origin).normalize()
        if not bool((dates == origin_date).any()):
            days = sorted(dates.dt.strftime("%Y-%m-%d").unique())
            raise ValueError(
                f"origin={origin_date.date()} のデータがありません。"
                f"読み込んだ範囲: {days[0]} 〜 {days[-1]} ({len(days)}日)"
            )

    return ds[(dates == origin_date).reset_index(drop=True)]


def _fetch_and_update_db(config: Config, db_manager: DBManager) -> None:
    """最新のAMeDASデータを取得し、raw CSV と DB を更新する。

    取得に失敗しても既存DBで予測を続行できるよう、例外は握り潰して警告に留める。
    """
    print("[2/6] fetch latest AMeDAS data and update DB")
    fetcher = AmedasLatestFetcher(config)
    writer = AmedasRawDataWriter(config)
    try:
        df_raw = fetcher.fetch_latest()
        if df_raw.empty:
            print("[WARN] 新規データなし、既存 DB で続行")
            return
        writer.save_raw_data(df_raw)
        processor = AmedasProcessor(config)
        # 取得した新規行のみ処理・追記する（raw CSV 全体を再度 append しない）
        processed_new = processor.process_data(df_raw)
        db_manager.processed_data_save(processed_new, if_exists="append")
    except Exception as e:
        print(f"[WARN] fetch/process 失敗、既存 DB で続行: {e}")


def main() -> None:
    args = parse_args()
    origin_dt = _parse_datetime(args.origin) if args.origin else None
    start_dt, end_dt = _resolve_window(args.start, args.end, origin_dt)

    config = Config()

    print("[1/6] load model")
    artifact = ModelStore().load(args.model_path)
    model = artifact.model
    horizon = list(artifact.metadata.get("horizon", config.pipeline.horizon))
    print(f"loaded model={artifact.model_name}, horizon={horizon}")

    db_manager = DBManager(config)
    if args.skip_fetch:
        print("[2/6] skip fetch (--skip-fetch), use existing DB")
    else:
        _fetch_and_update_db(config, db_manager)

    print(f"[3/6] load slice from DB start={start_dt} end={end_dt}")
    df = db_manager.load_processed_data(start_dt, end_dt)
    print(f"loaded rows={df.height}, cols={df.width}")

    print("[4/6] preprocess and build dataset")
    preprocessor = PreProcessor(config)
    df = preprocessor.select_columns(df)
    df = preprocessor.ave_day_columns(df)
    df = preprocessor.convert_type(df)

    dataset_generator = DatasetGenerator(horizon)
    ds = dataset_generator.prepare_dataset(df)
    ds = _select_origin_rows(ds, origin_dt)
    if ds.X is None or ds.idx is None:
        raise ValueError("Prediction dataset has no feature columns.")
    ds.X = _align_features(ds.X, artifact.feature_list)

    print(
        f"dataset ready rows={len(ds)}, features={ds.X.shape[1]}, "
        f"origin={pd.to_datetime(ds.idx['date']).max():%Y-%m-%d}"
    )

    print("[5/6] predict")
    predictions = model.predict(ds, horizons=horizon)
    print(f"predictions shape={predictions.shape}")

    print("[6/6] save output")
    pred_cols = [f"pred_h{h}" for h in horizon]
    pred_values = pd.DataFrame(predictions, columns=pred_cols)
    output_df = _build_prediction_output(ds.idx, pred_values, horizon)

    model_id = args.model_id or args.model_path.stem
    records = _to_prediction_records(
        output_df,
        config.amedas_fetch.station_id_by_name,
        model_id,
        datetime.now(),
    )
    saved = db_manager.predictions_data_save(records)
    print(
        f"saved predictions: {db_manager.prediction_db_path} "
        f"table={db_manager.prediction_table_name} rows={saved} model_id={model_id} "
        f"origin_at={records['origin_at'].max():%Y-%m-%d} "
        f"target_at={records['target_at'].min():%Y-%m-%d} -> "
        f"{records['target_at'].max():%Y-%m-%d}"
    )

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.output_path, index=False)
        print(f"saved CSV dump: {args.output_path}")


if __name__ == "__main__":
    main()
