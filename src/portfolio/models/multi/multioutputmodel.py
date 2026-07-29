from typing import Any, Callable, Dict, List, Optional, Protocol, Self

import numpy as np
from tqdm import tqdm

from portfolio.data.dataset import Dataset


class RegressorLike(Protocol):
    """
    fit/predictを持つ回帰モデルなら何でもこの Protocol を満たす。
    LightGBMなどを抽象化するためのインターフェイス。
    """

    def fit(self, X: Any, y: Any) -> Any: ...

    def predict(self, X: Any) -> Any: ...


class MultiOutputModel:
    """RegressorLikeをhorizonごとにfitするラッパー"""

    def __init__(self, base_model_factory: Callable[[], RegressorLike]):
        self.base_model_factory = base_model_factory
        self.models: List[RegressorLike] = []
        self.n_horizons: Optional[int] = None
        # feature_importance で使う。全horizonで同じ特徴量を使うため1本で足りる
        self.feature_names: Optional[List[str]] = None

    @property
    def feature_importance(self) -> Dict[str, List[float]]:
        """特徴量名 -> horizonごとの重要度リスト。

        horizon別モデルの重要度を特徴量ごとに横に並べたもので、
        VizEvaluation の重要度プロットが期待する形。
        """
        if not self.models:
            raise RuntimeError("Model is not fitted yet.")

        # feature_names 導入前に pickle 化されたモデルには属性自体が無いため
        # getattr で受ける（AttributeError ではなく説明付きの例外にする）
        feature_names = getattr(self, "feature_names", None)
        if feature_names is None:
            raise ValueError(
                "feature names are not available. "
                "feature_names を保存していない古い形式のモデルの可能性があります"
            )

        importances = []
        for model in self.models:
            values = getattr(model, "feature_importances_", None)
            if values is None:
                raise AttributeError(
                    f"{type(model).__name__} は feature_importances_ を持ちません"
                )
            importances.append(np.asarray(values, dtype=float))

        return {
            name: [float(row[i]) for row in importances]
            for i, name in enumerate(feature_names)
        }

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
        self.feature_names = list(X.columns) if hasattr(X, "columns") else None

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
