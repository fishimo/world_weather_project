from dataclasses import dataclass


@dataclass(frozen=True)
class LightGBMParams:
    """LightGBMのパラメータ"""
    
    objective:str = "regression"
    metric:str = "rmse"
    boosting_type:str = "gbdt"
    learning_rate:float = 0.03
    num_leaves:int = 31
    min_data_in_leaf:int = 30
    feature_fraction:float = 0.8
    bagging_fraction:float = 0.8
    bagging_freq:int = 1
    lambda_l2:float = 1.0
    verbosity:int = -1
    random_state:int = 42
    n_estimators:int = 3000

class Params:
    """モデルパラメータの統合クラス"""

    lightgbm = LightGBMParams()
