class ModelTrain(BaseStep):
    """BasicModelを学習するステップ（統一インターフェイス）"""

    def __init__(self, builder: Any, mode: Literal["train", "overall"]):
        super().__init__()
        self.builder = builder
        self.mode = mode

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        ds = None
        if self.mode == "train":
            if not isinstance(state, TrainState):
                raise TypeError("state must be TrainState for mode='train'")
            self._require_state(
                state, 
                require_attrs=("train_ds", "horizon"), 
                require_non_none=("train_ds", "horizon")
            )
            ds = state.train_ds
        elif self.mode == "overall":
            self._require_state(
                state, 
                require_attrs=("ds", "horizon"), 
                require_non_none=("ds", "horizon")
            )
            ds = state.ds
        if ds is None:
            raise ValueError("ds must not be None")

        # overall モードの場合、y_expanded に NaN がある行を除外
        # (ExpandHorizon で末尾の行に NaN が発生するため)
        if self.mode == "overall" and ds.y_expanded is not None:
            valid_mask: pd.Series = ~ds.y_expanded.isna().any(axis=1)  # type: ignore[assignment]
            if not bool(valid_mask.all()):
                ds = ds[valid_mask]

        state.model = self.builder.build_model()

        fit_kwargs: dict[str, Any] = {"horizons": state.horizon}
        state.model.fit(ds, **fit_kwargs)

        if ds.X is not None:
            self.log_out = f"X.shape: {ds.X.shape}"
        return state
    
    def depict(self, prefix: str = "[0]") -> Tuple[List[str], str]:
        lines = [f"{prefix} ModelTrain: \033[34m{self.mode}\033[0m"]
        return lines, self._gen_next_prefix(prefix)
    

class ModelPredict(BaseStep):
    """BasicModelで予測するステップ（統一インターフェイス）"""

    def __init__(self, mode: Literal["train", "predict"] = "train") -> None:
        super().__init__()
        self.mode = mode

    @staticmethod
    def _predict_with_optional_context(
        model: Any,
        X: np.ndarray,
        idx: Any,
        horizons: Any,
    ) -> Any:
        try:
            return model.predict(X, idx=idx, horizons=horizons)
        except TypeError as e:
            # mlflow.pyfunc.PyFuncModel.predict accepts only model_input/params.
            if "unexpected keyword argument" in str(e):
                return model.predict(X)
            raise

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        if self.mode == "train":
            if not isinstance(state, TrainState):
                raise TypeError("state must be TrainState for mode='train'")
            self._require_state(
                state,
                require_attrs=("model", "test_ds", "horizon"),
                require_non_none=("model", "test_ds", "horizon"),
            )
            if state.test_ds is None:
                raise ValueError("test_ds must not be None")

            predictions = state.model.predict(  # type: ignore[union-attr]
                state.test_ds, idx=state.test_ds.idx, horizons=state.horizon
            )
            state.test_predictions = predictions
            self.log_out = f"predictions.shape: {predictions.shape}"

        elif self.mode == "predict":
            if not isinstance(state, PredictState):
                raise TypeError("state must be PredictState for mode='predict'")
            self._require_state(
                state,
                require_attrs=("model", "ds", "horizon"),
                require_non_none=("model", "ds", "horizon"),
            )
            if state.ds is None:
                raise ValueError("ds must not be None")

            predictions = state.model.predict(  # type: ignore[union-attr]
                state.ds, idx=state.ds.idx, horizons=state.horizon
            )
            state.predictions = predictions
            self.log_out = f"predictions.shape: {predictions.shape}"

        return state
    

class ModelPlotEvaluation(BaseStep):
    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        if not isinstance(state, TrainState):
            raise TypeError("state must be TrainState for ModelPlotEvaluation")
        self._require_state(
            state,
            require_attrs=("score", "horizon", "test_ds", "test_predictions"),
            require_non_none=("score", "horizon", "test_ds", "test_predictions"),
        )
        if (
            state.score is None
            or state.horizon is None
            or state.test_ds is None
            or state.test_ds.idx is None
            or state.test_ds.y_expanded is None
            or state.test_predictions is None
        ):
            raise ValueError("Required attributes must not be None")
        viz_evaluation = VizEvaluation(state.score, state.horizon)  # type: ignore[arg-type]
        idx_pl = pl.from_pandas(state.test_ds.idx)
        y_true_pl = pl.from_pandas(state.test_ds.y_expanded)

        viz_evaluation.plot_series(idx_pl, y_true_pl, state.test_predictions)
        viz_evaluation.plot_all_metrics()
        if state.model is not None and hasattr(state.model, "feature_importance"):
            viz_evaluation.plot_mean_feature_importance(state.model)  # type: ignore[arg-type]
            viz_evaluation.feature_importance_map(state.model)  # type: ignore[arg-type]
        return state


class BuildPredDf(BaseStep):
    """予測結果をlong形式のDataFrameに整形するステップ"""

    def __init__(
        self,
        *,
        freq_minutes: int = 30,
        origin_ts_col: str = "timestamp",
        area_col: str = "area",
        biz_col: str = "biz_code",
        y_pred_col: str = "y_pred",
    ) -> None:
        super().__init__()
        self.freq_minutes = freq_minutes
        self.origin_ts_col = origin_ts_col
        self.area_col = area_col
        self.biz_col = biz_col
        self.y_pred_col = y_pred_col

    def _get_dataset(self, state: Any) -> Any:
        for name in ("ds", "pred_ds", "test_ds"):
            ds = getattr(state, name, None)
            if ds is not None:
                return ds
        return None

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        if not isinstance(state, PredictState):
            raise TypeError("state must be PredictState for BuildPredDf")
        self._require_state(
            state,
            require_attrs=("predictions",),
            require_non_none=("predictions",),
        )

        ds = self._get_dataset(state)
        if ds is None or getattr(ds, "idx", None) is None:
            raise ValueError("BuildPredDf: dataset idx not found")

        idx_df = ds.idx
        if not isinstance(idx_df, pd.DataFrame):
            raise TypeError("BuildPredDf: ds.idx must be pandas DataFrame")

        for c in (self.origin_ts_col, self.area_col, self.biz_col):
            if c not in idx_df.columns:
                raise ValueError(f"BuildPredDf: ds.idx must have column '{c}'")

        pred = np.asarray(state.predictions)
        if pred.ndim == 1:
            pred = pred[:, None]
        if pred.ndim != 2:
            raise ValueError(
                f"BuildPredDf: predictions must be 1D or 2D, got shape={pred.shape}"
            )

        N, H = pred.shape

        state_h = getattr(state, "horizon", None)
        if isinstance(state_h, (list, tuple)) and len(state_h) == H:
            h_list = list(map(int, state_h))
        else:
            if state_h is None:
                raise ValueError("BuildPredDf: state.horizon is None")
            raise ValueError(
                f"BuildPredDf: state.horizon length mismatch. Expected {H}, got {len(state_h) if state_h else 0}"
            )

        origin_ts = pd.to_datetime(idx_df[self.origin_ts_col]).to_numpy()
        areas = idx_df[self.area_col].to_numpy()
        biz = idx_df[self.biz_col].to_numpy()

        if len(origin_ts) != N:
            raise ValueError(
                f"BuildPredDf: N mismatch. idx rows={len(origin_ts)} vs predictions N={N}"
            )

        origin_rep = np.repeat(origin_ts, H)
        h_rep = np.tile(np.array(h_list, dtype=np.int32), N)
        pred_vals = pred.reshape(-1).astype(np.float64, copy=False)

        forecast_ts = origin_rep + pd.to_timedelta(h_rep * self.freq_minutes, unit="m")

        area_rep = np.repeat(areas, H)
        biz_rep = np.repeat(biz, H)

        out = pd.DataFrame(
            {
                "origin_timestamp": origin_rep,
                "forecast_timestamp": forecast_ts,
                "horizon": h_rep,
                self.area_col: area_rep,
                self.biz_col: biz_rep,
                self.y_pred_col: pred_vals,
            }
        )

        state.pred_df = out
        self.log_out = f"pred_df.shape: {out.shape}"
        return state