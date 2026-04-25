from pathlib import Path
import pandas as pd

from configs.config import Config
from src.portfolio.data.database_manager import DBManager
from src.portfolio.data.dataset import Dataset, DatasetGenerator
from src.portfolio.features.feature_preprocessor import PreProcessor


def main() -> None:
    def dataset_to_frame(ds: Dataset) -> pd.DataFrame:
        parts = []
        if ds.idx is not None:
            parts.append(ds.idx)
        if ds.y is not None:
            parts.append(ds.y)
        if ds.X is not None:
            parts.append(ds.X)
        if ds.y_expanded is not None:
            parts.append(ds.y_expanded)
        return pd.concat(parts, axis=1)
    
    # 設定
    config = Config()
    horizon = config.pipeline.horizon
    output_dir = Path("./data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_manager = DBManager(config)
    df = db_manager.load_all()
    print(df.height)

    preprocessor = PreProcessor(config)
    df = preprocessor.select_columns(df)
    df = preprocessor.ave_day_columns(df)
    df = preprocessor.convert_type(df)
    print(df.height)

    dataset_generator = DatasetGenerator(horizon)
    ds = dataset_generator.prepare_dataset(df)
    ds = dataset_generator.expand_horizon(ds)
    train_ds, test_ds = dataset_generator.split(ds, test_size=0.2)

    train_df = dataset_to_frame(train_ds)
    test_df = dataset_to_frame(test_ds)

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("train_ds")
    print(train_df.head())
    print(train_df.shape)

    print("test_ds")
    print(test_df.head())
    print(test_df.shape)

if __name__ == "__main__":
    main()