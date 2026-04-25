from __future__ import annotations

from portfolio.data.dataset import DatasetGenerator
from portfolio.features.feature_preprocessor import PreProcessor
from portfolio.utils.pd_pl_utils import ensure_polars


class PreProcess(BaseStep):
    def __init__(self, preprocessor: PreProcessor) -> None:
        super().__init__()
        self.preprocessor = preprocessor

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        self._require_state(
            state,
            require_attrs=("df", ),
            require_non_none=("df", ),
        )

        pl_df = ensure_polars(state.df)
        pl_df = self.preprocessor.select_columns(pl_df)
        pl_df = self.preprocessor.ave_day_columns(pl_df)
        pl_df = self.preprocessor.convert_type(pl_df)
        state.df = pl_df

        return state
    

class PrepareDataset(BaseStep):
    def __init__(self, dataset_generator: DatasetGenerator) -> None:
        super().__init__()
        self.dataset_generator = dataset_generator

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        self._require_state(
            state,
            require_attrs=("df", "horizon", "ds"),
            require_non_none=("df", "horizon"),
        )

        state.ds = self.dataset_generator.prepare_dataset(state.df)

        return state


class ExpandHorizon(BaseStep):
    def __init__(
            self, 
            dataset_generator: DatasetGenerator,
            targets: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.dataset_generator = dataset_generator
        self.targets = targets
    
    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        self._require_state(
            state,
            require_attrs=("ds"),
            require_non_none=("ds", ),
        )
        state.ds = self.dataset_generator.expand_horizon(state.ds, targets=self.targets)

        return state


class DatasetFilterNan(BaseStep):
    def __init__(self, dataset_generator: DatasetGenerator) -> None:
        super().__init__()
        self.dataset_generator = dataset_generator

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        self._require_state(
            state,
            require_attrs=("ds"),
            require_non_none=("ds", ),
        )

        state.ds = self.dataset_generator.filter_trainable_rows(state.ds)

        return state
    

class SplitDataset(BaseStep):
    def __init__(
        self, dataset_generator: DatasetGenerator, test_size: float = 0.2
    ) -> None:
        super().__init__()
        self.dataset_generator = dataset_generator
        self.test_size = test_size

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        self._require_state(
            state,
            require_attrs=(
                "ds",
                "test_ds",
                "train_ds",
            ),
            require_non_none=("ds",),
        )
        state.train_ds, state.test_ds = self.dataset_generator.split(
            state.ds, test_size=self.test_size
        )
        train_n = (
            state.train_ds.X.shape[0]
            if state.train_ds and state.train_ds.X is not None
            else 0
        )
        test_n = (
            state.test_ds.X.shape[0]
            if state.test_ds and state.test_ds.X is not None
            else 0
        )
        self.log_out = f"train: {train_n}, test: {test_n}"
        return state
