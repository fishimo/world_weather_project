from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from configs.config import Config
from configs.params import Params
from portfolio.data.database_manager import DBManager
from portfolio.data.dataset import DatasetGenerator
from portfolio.eval.evaluator import Evaluator
from portfolio.features.feature_preprocessor import PreProcessor
from portfolio.models.model_store import ModelStore, TrainedModelArtifact
from portfolio.models.multi.builder import MultiModelBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check DB load -> dataset build -> train"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test dataset ratio",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    config = Config()
    params = Params()
    horizon = list(config.pipeline.horizon)
    print(f"horizon={horizon}")

    db_manager = DBManager(config)
    df = db_manager.load_all()
    print(f"loaded db rows={df.height}, cols={df.width}")

    preprocessor = PreProcessor(config)
    df = preprocessor.select_columns(df)
    df = preprocessor.ave_day_columns(df)
    df = preprocessor.convert_type(df)
    print(f"preprocessed rows={df.height}, cols={df.width}")

    dataset_generator = DatasetGenerator(horizon)
    ds = dataset_generator.prepare_dataset(df)
    ds = dataset_generator.expand_horizon(ds)
    ds = dataset_generator.filter_trainable_rows(ds)

    if len(ds) == 0:
        raise ValueError("No trainable rows found")

    train_ds, test_ds = dataset_generator.split(ds, test_size=args.test_size)
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            f"train/test is empty: train={len(train_ds)}, test={len(test_ds)}"
        )

    feature_count = 0 if train_ds.X is None else train_ds.X.shape[1]
    print(
        f"dataset ready train={len(train_ds)}, test={len(test_ds)}, "
        f"features={feature_count}"
    )

    builder = MultiModelBuilder(params)
    model = builder.base_model()
    model.fit(train_ds, horizons=horizon)
    print("train finished")

    preds = model.predict(test_ds, horizons=horizon)
    print(f"predict finished shape={preds.shape}")
    
    evaluator = Evaluator()
    score = evaluator.evaluate(
        y_true=test_ds.y_expanded.to_numpy(),
        y_pred=preds,
    )
    print(f"train score: {score}")

    artifact = TrainedModelArtifact(
        model=model,
        model_name=type(model).__name__,
        feature_list=train_ds.feature_list,
        params=asdict(params.lightgbm),
        metadata={
            "horizon": horizon,
        },
    )

    save_path = Path("test_models/test_model.pkl")
    model_store = ModelStore()
    model_store.save(artifact, save_path)
    print(f"saved model: {save_path}")


if __name__ == "__main__":
    main()
