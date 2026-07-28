from __future__ import annotations

import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from configs.config import Config
from configs.params import LightGBMParams
from portfolio.data.amedas_processor import AmedasProcessor
from portfolio.data.database_manager import DBManager
from portfolio.pipeline import predict as predict_pipeline
from portfolio.pipeline import train as train_pipeline


class _SmallParams:
    lightgbm = LightGBMParams(
        n_estimators=20,
        num_leaves=8,
        min_data_in_leaf=1,
    )


def _write_dummy_amedas_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    start = datetime(2026, 1, 1, 12, 0, 0)
    stations = [
        ("44132", "Tokyo", 35.6944, 139.7529, 3.0),
        ("44136", "Osaka", 34.6937, 135.5023, 4.0),
    ]

    for day in range(12):
        current = start + timedelta(days=day)
        for station_id, location_name, lat, lon, base_wind in stations:
            rows.append(
                {
                    "timestamp": current.strftime("%Y-%m-%d %H:%M:%S"),
                    "station_id": station_id,
                    "location_name": location_name,
                    "latitude": lat,
                    "longitude": lon,
                    "temperature_celsius": 18.0 + day,
                    "humidity_pct": 60.0,
                    "wind_mps": base_wind + day * 0.1,
                    "wind_degree": 90.0 + day * 3,
                    "pressure_mb": 1013.0 + day * 0.1,
                    "missing_flag": False,
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def _prepare_processed_db(workspace: Path) -> Path:
    raw_csv = workspace / "data" / "raw" / "amedas" / "amedas_raw.csv"
    _write_dummy_amedas_csv(raw_csv)

    config = Config()
    processor = AmedasProcessor(config)
    processed_df = processor.process_data(processor.read_data())

    db_manager = DBManager(config)
    db_manager.processed_data_save(processed_df, if_exists="replace")
    return raw_csv


def _run_train(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(train_pipeline, "Params", _SmallParams)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--test-size", "0.25"],
    )
    train_pipeline.main()
    return Path("test_models") / "test_model.pkl"


@pytest.fixture
def integration_workspace(monkeypatch: pytest.MonkeyPatch) -> Path:
    project_root = Path.cwd()
    workspace = project_root / ".tmp_integration" / f"case_{uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(workspace)
    _prepare_processed_db(workspace)

    yield workspace

    shutil.rmtree(workspace, ignore_errors=True)


def test_train_pipeline_saves_model(
    integration_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _run_train(monkeypatch)

    assert model_path.exists()

    artifact = train_pipeline.ModelStore().load(model_path)
    assert artifact.model_name == "MultiOutputModel"
    assert artifact.metadata["horizon"] == [1, 2, 3]
    assert artifact.feature_list is not None
    assert len(artifact.feature_list) > 0


def test_predict_pipeline_uses_saved_model(
    integration_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _run_train(monkeypatch)

    # fetch_latest をスキップ（空DataFrameを返す → DB更新ステップをバイパス）
    monkeypatch.setattr(
        predict_pipeline.AmedasLatestFetcher,
        "fetch_latest",
        lambda _: pd.DataFrame(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--model-path",
            str(model_path),
            "--start",
            "2026-01-08",
            "--end",
            "2026-01-12",
            "--output-path",
            "outputs/predictions.csv",
        ],
    )

    predict_pipeline.main()

    output_path = Path("outputs") / "predictions.csv"
    assert output_path.exists()

    output_df = pd.read_csv(output_path)
    assert not output_df.empty
    assert {"origin_date", "timestamp", "horizon", "y_pred"}.issubset(output_df.columns)
    assert sorted(output_df["horizon"].unique().tolist()) == [1, 2, 3]
    assert len(output_df) == 6
