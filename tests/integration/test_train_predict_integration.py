from __future__ import annotations

import shutil
import sqlite3
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from configs.config import AmedasFetch, AmedasStation, Config, Pipeline
from configs.params import LightGBMParams
from portfolio.data.amedas_processor import AmedasProcessor
from portfolio.data.database_manager import DBManager
from portfolio.pipeline import predict as predict_pipeline
from portfolio.pipeline import train as train_pipeline

# ダミーデータの地点定義。config とテストデータで同じものを使う
_STATIONS = [
    ("44132", "Tokyo", 35.6944, 139.7529, 3.0),
    ("44136", "Osaka", 34.6937, 135.5023, 4.0),
]

_HORIZON = [1, 2, 3]
_N_DAYS = 12
_START = datetime(2026, 1, 1, 12, 0, 0)


class _TestConfig(Config):
    """テスト用 Config。

    horizon を本番設定（1〜95日）から切り離す。本番値のままだと 95 日先まで
    揃った行が必要になり、少量のダミーデータでは学習可能行が 0 になるため。
    地点もダミーデータに合わせて 2 箇所定義する。
    """

    def __init__(self) -> None:
        super().__init__()
        self.pipeline = Pipeline(horizon=list(_HORIZON))
        self.amedas_fetch = AmedasFetch(
            stations=[
                AmedasStation(
                    station_id=station_id,
                    name=name,
                    latitude=lat,
                    longitude=lon,
                )
                for station_id, name, lat, lon, _ in _STATIONS
            ]
        )


class _SmallParams:
    lightgbm = LightGBMParams(
        n_estimators=20,
        num_leaves=8,
        min_data_in_leaf=1,
    )


def _write_dummy_amedas_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for day in range(_N_DAYS):
        current = _START + timedelta(days=day)
        for station_id, location_name, lat, lon, base_wind in _STATIONS:
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

    config = _TestConfig()
    processor = AmedasProcessor(config)
    processed_df = processor.process_data(processor.read_data())

    db_manager = DBManager(config)
    db_manager.processed_data_save(processed_df, if_exists="replace")
    return raw_csv


def _run_train(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(train_pipeline, "Config", _TestConfig)
    monkeypatch.setattr(train_pipeline, "Params", _SmallParams)
    monkeypatch.setattr(sys, "argv", ["train.py", "--test-size", "0.25"])
    train_pipeline.main()
    return Path("test_models") / "test_model.pkl"


def _run_predict(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(predict_pipeline, "Config", _TestConfig)
    monkeypatch.setattr(sys, "argv", argv)
    predict_pipeline.main()


def _read_predictions() -> pd.DataFrame:
    db_path = Path("data") / "predictions" / "predictions.db"
    assert db_path.exists(), f"predictions DB が作られていません: {db_path}"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY origin_at, target_at, station_id",
            conn,
        )


@pytest.fixture
def integration_workspace(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
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
    assert artifact.metadata["horizon"] == _HORIZON
    assert artifact.feature_list is not None
    assert len(artifact.feature_list) > 0


def test_predict_pipeline_saves_to_prediction_db(
    integration_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _run_train(monkeypatch)

    _run_predict(
        monkeypatch,
        [
            "predict.py",
            "--model-path",
            str(model_path),
            "--skip-fetch",
            "--origin",
            "2026-01-11",
        ],
    )

    preds = _read_predictions()

    # 2地点 x 3horizon
    assert len(preds) == len(_STATIONS) * len(_HORIZON)
    assert set(preds["station_id"]) == {"44132", "44136"}
    assert set(preds["model_id"]) == {"test_model"}
    assert set(preds["origin_at"]) == {"2026-01-11 00:00:00"}
    assert set(preds["target_at"]) == {
        "2026-01-12 00:00:00",
        "2026-01-13 00:00:00",
        "2026-01-14 00:00:00",
    }
    assert preds["wind_speed"].notna().all()
    assert preds["created_at"].notna().all()


def test_predict_pipeline_accumulates_by_origin(
    integration_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """origin が違えば別行として貯まり、同じ origin の再実行は上書きされる。"""
    model_path = _run_train(monkeypatch)

    base_argv = ["predict.py", "--model-path", str(model_path), "--skip-fetch"]
    for origin in ("2026-01-10", "2026-01-11"):
        _run_predict(monkeypatch, [*base_argv, "--origin", origin])

    preds = _read_predictions()
    per_run = len(_STATIONS) * len(_HORIZON)
    assert len(preds) == per_run * 2
    assert set(preds["origin_at"]) == {
        "2026-01-10 00:00:00",
        "2026-01-11 00:00:00",
    }

    # 同じ origin を再実行しても行は増えない（UPSERT）
    _run_predict(monkeypatch, [*base_argv, "--origin", "2026-01-11"])
    assert len(_read_predictions()) == per_run * 2

    # 同じ target_at が origin 違いで両方残っている
    overlap = preds[preds["target_at"] == "2026-01-12 00:00:00"]
    assert set(overlap["origin_at"]) == {
        "2026-01-10 00:00:00",
        "2026-01-11 00:00:00",
    }


def test_predict_pipeline_writes_optional_csv(
    integration_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _run_train(monkeypatch)

    output_path = Path("outputs") / "predictions.csv"
    _run_predict(
        monkeypatch,
        [
            "predict.py",
            "--model-path",
            str(model_path),
            "--skip-fetch",
            "--origin",
            "2026-01-11",
            "--output-path",
            str(output_path),
        ],
    )

    assert output_path.exists()
    output_df = pd.read_csv(output_path)
    assert not output_df.empty
    assert {"origin_date", "timestamp", "horizon", "y_pred"}.issubset(output_df.columns)
    assert sorted(output_df["horizon"].unique().tolist()) == _HORIZON
    assert len(output_df) == len(_STATIONS) * len(_HORIZON)
