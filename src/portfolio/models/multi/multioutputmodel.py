from typing import Any, Callable, Optional, Protocol, Self

import numpy as np
from tqdm import tqdm

from portfolio.data.dataset import Dataset


class RegressorLike(Protocol):
    """
    fit/predictを持つ回帰モデルなら何でもこの Protocol を満たす。
    LightGBMなどを抽象化するためのインターフェイス。
    """

    def fit(self, X: Any, y: Any) -> Any: ...

    def predict(self, X:Any) -> Any: ...
    

class MultiOutputModel:
    """RegressorLikeをhorizonごとにfitするラッパー"""

    def __init__(self, base_model_factory: Callable[[], RegressorLike]):
        self.base_model_factory = base_model_factory
        self.models = []
        self.n_horizons: Optional[int] = None

    def fit(self, ds: Dataset, **kwargs: Any) -> Self:
        """Datasetを受け取って学習する"""
        if ds.X is None or ds.y_expanded is None:
            raise ValueError("ds.X and ds.y_expanded must not be None")

        X = ds.X.copy()
        Y = ds.y_expanded.copy()
        # horizonは外から渡す
        horizons = kwargs.get("horizons")

        if horizons is None:
            raise ValueError("horizon must not be None")

        n_samples, n_horizons = Y.shape
        self.n_horizons = n_horizons
        if n_horizons != len(horizons):
            raise ValueError(
                f"len(horizons) must equal number of targets. "
                f"got len(horizons)={len(horizons)}, n_horizons={n_horizons}"
            )

        # いったんまとめて学習
        self.models = [self.base_model_factory() for _ in range(n_horizons)]

        print(f"starting model trainning: {n_horizons} horizon/ {n_samples} samples")

        for h_idx in tqdm(
            range(n_horizons),
            desc="Training horizons (sequential)",
            unit="horizon",
            leave=False,
        ):
            self.models[h_idx].fit(X, Y.iloc[:, h_idx])

        return self
    
    def predict(self, ds: Dataset, **kwargs: Any) -> np.ndarray:
        """Datasetを受け取って予測する"""
        if self.n_horizons is None or not self.models:
            raise RuntimeError("Model is not fitted yet.")
        
        horizons = kwargs.get("horizons")
        if horizons is None:
            raise ValueError("horizon must not be None")

        if ds.X is None:
            raise ValueError("ds must not be None")
        
        X = ds.X.copy()
        predictions = [
            np.array(self.models[h_idx].predict(X)) for h_idx in range(len(horizons))
        ]

        return np.column_stack(predictions)
