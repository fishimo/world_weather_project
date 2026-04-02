from typing import Any, Protocol, Self

import numpy as np
from portfolio.data.dataset import Dataset


class BasicModel(Protocol):
    """
    パイプラインで統一的に扱うための最小インターフェイス。

    - fit: Datasetを受け取り学習してselfを返す
    - predict: 特徴量Xまたはdsを受け取って予測配列(n_samoles, n_horizons)を返す
    """

    def fit(self, ds: Dataset, **kwargs: Any) -> Self: ...

    def predict(self, X_or_ds: Any, **kwargs: Any) -> np.ndarray: ...