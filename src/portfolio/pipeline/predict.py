from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

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
from portfolio.data.dataset import DatasetGenerator
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
        "--start",
        type=str,
        required=True,
        help="Prediction data start datetime. Example: 2026-04-01 00:00:00",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="Prediction data end datetime. Example: 2026-04-21 00:00:00",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/predictions.csv"),
        help="Path to save prediction output CSV.",
    )
    return parser.parse_args()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


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


def _select_latest_rows(ds: object) -> object:
    idx_df = getattr(ds, "idx", None)
    if idx_df is None or "date" not in idx_df.columns:
        raise ValueError("Prediction dataset requires 'date' column in ds.idx.")

    latest_date = pd.to_datetime(idx_df["date"]).max()
    latest_mask = pd.to_datetime(idx_df["date"]) == latest_date
    return ds[latest_mask.reset_index(drop=True)]


def main() -> None:
    args = parse_args()
    start_dt = _parse_datetime(args.start)
    end_dt = _parse_datetime(args.end)

    config = Config()

    print("[1/8] load model")
    artifact = ModelStore().load(args.model_path)
    model = artifact.model
    horizon = list(artifact.metadata.get("horizon", config.pipeline.horizon))
    print(f"loaded model={artifact.model_name}, horizon={horizon}")

    print("[2/8] download latest raw data")
    downloader = DataDownloader(config)
    try:
        downloader.data_download()
    except RuntimeError as e:
        if downloader.has_raw_data():
            print(f"[WARN] download failed, fallback to local raw data: {e}")
            print(f"[WARN] using existing file: {downloader.raw_data_path()}")
        else:
            raise

    print("[3/8] read/process raw data")
    processor = DataProcessor(config)
    raw_df = processor.read_data()
    processed_df = processor.process_data(raw_df)

    print("[4/8] save processed data to DB")
    db_manager = DBManager(config)
    db_manager.processed_data_save(processed_df, if_exists="append")

    print(f"[5/8] load slice from DB start={start_dt} end={end_dt}")
    df = db_manager.load_processed_data(start_dt, end_dt)
    print(f"loaded rows={df.height}, cols={df.width}")

    print("[6/8] preprocess and build dataset")
    preprocessor = PreProcessor(config)
    df = preprocessor.select_columns(df)
    df = preprocessor.ave_day_columns(df)
    df = preprocessor.convert_type(df)

    dataset_generator = DatasetGenerator(horizon)
    ds = dataset_generator.prepare_dataset(df)
    ds = _select_latest_rows(ds)
    if ds.X is None:
        raise ValueError("Prediction dataset has no feature columns.")
    ds.X = _align_features(ds.X, artifact.feature_list)

    print(f"dataset ready rows={len(ds)}, features={ds.X.shape[1]}, latest_date={ds.idx['date'].max()}")

    print("[7/8] predict")
    predictions = model.predict(ds, horizons=horizon)
    print(f"predictions shape={predictions.shape}")

    print("[8/8] save output")
    pred_cols = [f"pred_h{h}" for h in horizon]
    pred_values = pd.DataFrame(predictions, columns=pred_cols)
    output_df = _build_prediction_output(ds.idx, pred_values, horizon)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_path, index=False)
    print(f"saved predictions: {args.output_path}")


if __name__ == "__main__":
    main()

