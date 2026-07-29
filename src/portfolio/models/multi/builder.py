from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

from lightgbm import LGBMRegressor

from configs.params import Params
from portfolio.models.multi.multioutputmodel import MultiOutputModel, RegressorLike


class MultiModelBuilder:
    """MultiOutPutModelを生成するBuilder"""

    def __init__(
        self,
        params: Params,
        base_model_factory: Optional[Callable[[], RegressorLike]] = None,
    ):
        self.params = params
        self._external_factory = base_model_factory

    def _build_lgb_params(self) -> Dict[str, Any]:
        """LightGBMパラメータを構築"""
        lgb_params = asdict(self.params.lightgbm)
        return lgb_params

    def _default_base_model_factory(self) -> RegressorLike:
        lgb_params = self._build_lgb_params()
        return LGBMRegressor(**lgb_params)

    @property
    def base_model_factory(self) -> Callable[[], RegressorLike]:
        return self._external_factory or self._default_base_model_factory

    def base_model(self) -> MultiOutputModel:
        return MultiOutputModel(base_model_factory=self.base_model_factory)
