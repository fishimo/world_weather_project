from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from portfolio.models.multi.multioutputmodel import MultiOutputModel

# 図の出力先ルート。artifact_path はこの下に相対で展開される
FIGURE_OUTPUT_ROOT = Path("outputs")


def log_figure(fig: Any, artifact_path: str) -> Path:
    """plotly の図を HTML として保存する。

    元は MLflow の log_figure を呼ぶ想定だったが、本プロジェクトは MLflow を
    使っていないためローカル保存に置き換えている。
    """
    path = FIGURE_OUTPUT_ROOT / artifact_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    return path


@dataclass
class EvaluationMetrics:
    """
    予測評価のメトリクスを柔軟に扱えるデータクラス

    features:
        metrics["mae"]["overall"]
        metrics["mae"]["per_horizon"]
    """

    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_samples: int = 0
    n_horizons: int = 0

    def add_metric(
        self, name: str, overall: float, per_horizon: Optional[np.ndarray] = None
    ) -> None:
        """新しい指標を追加する"""
        self.metrics[name] = {
            "overall": float(overall),
            "per_horizon": np.array(per_horizon) if per_horizon is not None else None,
        }


class Evaluator:
    """モデル評価クラス

    予測結果の評価指標を計算します。
    対応する評価指標:
    - bias
    - relative_bias
    - MAE
    - RMSE (Root Mean Squared Error)
    """

    def caluculate_bias(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        biasを計算（y_pred - y_true）

        Args:
            y_true: 真の値 (n_samples, n_horizons)
            y_pred: 予測値 (n_samples, n_horizons)

        Returns:
            Tuple[float, np.ndarray]: (全体MAE, ホライズン毎MAE)
        """
        bias = y_pred - y_true

        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))

        if len(bias.shape) > 1:
            # 各horizonごとに有効なデータのみでbiasを計算
            bias_per_h = np.array(
                [
                    (
                        float(np.mean(bias[valid_mask[:, h], h]))
                        if np.any(valid_mask[:, h])
                        else np.nan
                    )
                    for h in range(bias.shape[1])
                ]
            )
            valid_all = np.any(valid_mask, axis=1)
            if np.any(valid_all):
                bias_valid = bias[valid_all]
                bias_overall = float(np.nanmean(bias_valid))
            else:
                bias_overall = float(np.nan)
        else:
            valid_mask_1d = valid_mask.flatten()
            bias_overall = (
                float(np.mean(bias[valid_mask_1d]))
                if np.any(valid_mask_1d)
                else float(np.nan)
            )
            bias_per_h = np.array([bias_overall])

        return bias_overall, bias_per_h

    def calculate_relative_bias(
        self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 10.0
    ) -> Tuple[float, np.ndarray]:
        """
        相対biasを計算

        Args:
            y_true: 真の値 (n_samples, n_horizons)
            y_pred: 予測値 (n_samples, n_horizons)
            epsilon: y_trueがこの値以下のサンプルを相対biasの計算から除外する閾値

        Returns:
            Tuple[float, np.ndarray]: (全体MAPE, ホライズン毎相対bias)
        """
        bias = y_pred - y_true
        denom = np.abs(y_true)
        denom = np.where(denom > epsilon, denom, np.nan)

        ratio = bias / denom * 100

        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        valid_mask = valid_mask & ~np.isnan(denom)

        # overall
        if ratio.ndim > 1:
            valid_all = np.any(valid_mask, axis=1)
            if np.any(valid_all):
                ratio_valid = ratio[valid_all]
                relative_bias_overall = float(np.nanmean(ratio_valid))
            else:
                relative_bias_overall = float(np.nan)
        else:
            valid_mask_1d = valid_mask.flatten()
            relative_bias_overall = (
                float(np.mean(ratio[valid_mask_1d]))
                if np.any(valid_mask_1d)
                else float(np.nan)
            )

        # per horizon
        if ratio.ndim > 1:
            relative_bias_per_h = np.array(
                [
                    (
                        float(np.mean(ratio[valid_mask[:, h], h]))
                        if np.any(valid_mask[:, h])
                        else np.nan
                    )
                    for h in range(ratio.shape[1])
                ]
            )
        else:
            relative_bias_per_h = np.array([relative_bias_overall])

        return relative_bias_overall, relative_bias_per_h

    def calculate_mae(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        MAEを計算

        Args:
            y_true: 真の値 (n_samples, n_horizons)
            y_pred: 予測値 (n_samples, n_horizons)

        Returns:
            Tuple[float, np.ndarray]: (全体MAE, ホライズン毎MAE)
        """
        abs_err = np.abs(y_pred - y_true)

        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))

        if len(abs_err.shape) > 1:
            # 各horizonごとに有効なデータのみでMAEを計算
            mae_per_h = np.array(
                [
                    (
                        float(np.mean(abs_err[valid_mask[:, h], h]))
                        if np.any(valid_mask[:, h])
                        else np.nan
                    )
                    for h in range(abs_err.shape[1])
                ]
            )
            valid_all = np.any(valid_mask, axis=1)
            if np.any(valid_all):
                abs_err_valid = abs_err[valid_all]
                mae_overall = float(np.nanmean(abs_err_valid))
            else:
                mae_overall = float(np.nan)
        else:
            valid_mask_1d = valid_mask.flatten()
            mae_overall = (
                float(np.mean(abs_err[valid_mask_1d]))
                if np.any(valid_mask_1d)
                else float(np.nan)
            )
            mae_per_h = np.array([mae_overall])

        return mae_overall, mae_per_h

    def calculate_rmse(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        RMSEを計算

        Args:
            y_true: 真の値 (n_samples, n_horizons)
            y_pred: 予測値 (n_samples, n_horizons)

        Returns:
            Tuple[float, np.ndarray]: (全体RMSE, ホライズン毎RMSE)
        """
        squared_err = (y_pred - y_true) ** 2

        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))

        if len(squared_err.shape) > 1:
            # 各horizonごとに有効なデータのみでRMSEを計算
            rmse_per_h = np.array(
                [
                    (
                        float(np.sqrt(np.mean(squared_err[valid_mask[:, h], h])))
                        if np.any(valid_mask[:, h])
                        else np.nan
                    )
                    for h in range(squared_err.shape[1])
                ]
            )
            valid_all = np.any(valid_mask, axis=1)
            if np.any(valid_all):
                squared_err_valid = squared_err[valid_all]
                rmse_overall = float(np.sqrt(np.nanmean(squared_err_valid)))
            else:
                rmse_overall = float(np.nan)
        else:
            valid_mask_1d = valid_mask.flatten()
            rmse_overall = (
                float(np.sqrt(np.mean(squared_err[valid_mask_1d])))
                if np.any(valid_mask_1d)
                else float(np.nan)
            )
            rmse_per_h = np.array([rmse_overall])

        return rmse_overall, rmse_per_h

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Union[List[str], str] = "all",
        epsilon: float = 1e-10,
    ) -> EvaluationMetrics:
        """
        全ての評価指標を計算

        Args:
            timestamp: 30分刻みのタイムスタンプ (n_samples,)
            y_true: 真の値 (n_samples, n_horizons)
            y_pred: 予測値 (n_samples, n_horizons)
            metrics: "all" または ["mae", "mape", ...]
            epsilon: ゼロ除算を防ぐための小さな値

        Returns:
            EvaluationMetrics: 評価指標をまとめたオブジェクト
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )

        n_samples = y_true.shape[0]
        n_horizons = y_true.shape[1] if y_true.ndim > 1 else 1

        # 引数を正規化
        if metrics == "all":
            metrics = ["bias", "relative_bias", "mae", "rmse"]
        elif isinstance(metrics, str):
            metrics = [metrics.lower()]
        else:
            metrics = [m.lower() for m in metrics]

        result = EvaluationMetrics(n_samples=n_samples, n_horizons=n_horizons)

        dispatch = {
            "bias": lambda yt, yp: self.caluculate_bias(yt, yp),
            "relative_bias": lambda yt, yp: self.calculate_relative_bias(
                yt, yp, epsilon
            ),
            "mae": lambda yt, yp: self.calculate_mae(yt, yp),
            "rmse": lambda yt, yp: self.calculate_rmse(yt, yp),
        }

        for name in metrics:
            if name not in dispatch:
                raise ValueError(f"Unknown metric: {name}")

            overall, per_h = dispatch[name](y_true, y_pred)

            result.add_metric(name, overall, per_h)

        return result


class VizEvaluation:
    def __init__(
        self, score: Union[EvaluationMetrics, Dict[str, Any]], horizon: List[int]
    ):
        self.score = score
        self.horizon = horizon

    def plot_all_metrics(self) -> None:
        """
        登録されている全メトリクスの per_horizon を折れ線でプロットし、MLflowに保存する
        Args:
            figsize: 1つの subplot の大きさ（plotlyでは使用されません）
        """
        # scoreが辞書の場合は"metrics"キーから取得、
        # EvaluationMetricsオブジェクトの場合は.metrics属性から取得
        if isinstance(self.score, dict):
            metrics = self.score.get("metrics", {})
        else:
            metrics = self.score.metrics
        if not metrics:
            raise ValueError("No metrics found in EvaluationMetrics.")

        metric_names = list(metrics.keys())
        n_metrics = len(metric_names)

        # plotlyのsubplotを作成
        fig = make_subplots(
            rows=n_metrics,
            cols=1,
            subplot_titles=[
                f"{name}  (overall={metrics[name].get('overall', 0):.3f})"
                for name in metric_names
            ],
            vertical_spacing=0.1,
        )

        for i, name in enumerate(metric_names, start=1):
            data = metrics[name]
            per_h = data.get("per_horizon", None)

            if per_h is None:
                print(
                    f"[Warning] '{name}' に per_horizon がありません。スキップします。"
                )
                continue

            fig.add_trace(
                go.Scatter(
                    x=self.horizon,
                    y=per_h,
                    mode="lines+markers",
                    name=name.upper(),
                    showlegend=False,
                ),
                row=i,
                col=1,
            )

        fig.update_xaxes(title_text="Horizon")
        fig.update_yaxes(title_text="Value")
        fig.update_layout(
            height=300 * n_metrics,
            title_text="Metrics per Horizon",
            showlegend=False,
        )

        log_figure(fig, artifact_path="figures/metrics.html")

    def plot_series(
        self,
        idx: pl.DataFrame,
        y_true: pl.DataFrame,
        y_pred: np.ndarray,
    ) -> None:
        """
        y_true と y_pred の最初と最後の horizon を重ねてプロットし、MLflowに保存する

        - y_pred[:, 0]  : horizon = 0
        - y_pred[:, -1] : horizon = max horizon
        - x-axis        : idx['timestamp']（ラベル非表示）
        """
        ts = idx["timestamp"]
        true_series = y_true.to_series(0)

        pred_h0 = y_pred[:, 0]
        pred_hmax = y_pred[:, -1]

        n_ts = len(ts)
        n_true = len(true_series)
        n_pred_h0 = len(pred_h0)
        n_pred_hmax = len(pred_hmax)

        if not (n_ts == n_true == n_pred_h0 == n_pred_hmax):
            raise ValueError(
                f"Length mismatch detected:\n"
                f"  len(timestamp) = {n_ts}\n"
                f"  len(y_true)    = {n_true}\n"
                f"  len(y_pred[h=0])   = {n_pred_h0}\n"
                f"  len(y_pred[h=max]) = {n_pred_hmax}\n"
                "All series must have the same length."
            )

        df_all = pl.DataFrame(
            {
                "timestamp": ts,
                "y_true": true_series,
                "y_pred_h0": pred_h0,
                "y_pred_hmax": pred_hmax,
            }
        )

        df_mean = (
            df_all.group_by("timestamp")
            .agg(pl.exclude("timestamp").mean())
            .sort("timestamp")
        )
        ts_mean = df_mean["timestamp"]
        true_mean = df_mean["y_true"]
        pred_h0_mean = df_mean["y_pred_h0"]
        pred_hmax_mean = df_mean["y_pred_hmax"]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=ts_mean,
                y=true_mean,
                mode="lines",
                name="y_true",
                line=dict(width=1.2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ts_mean,
                y=pred_h0_mean,
                mode="lines",
                name=f"y_pred (h={min(self.horizon)})",
                line=dict(width=1.2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ts_mean,
                y=pred_hmax_mean,
                mode="lines",
                name=f"y_pred (h={max(self.horizon)})",
                line=dict(width=1.2),
            )
        )

        fig.update_layout(
            title="Actual vs Predicted Series (averaged by timestamp)",
            xaxis_title="Timestamp",
            yaxis_title="Value",
            height=400,
            hovermode="x unified",
        )

        log_figure(fig, artifact_path="figures/series.html")

    def plot_mean_feature_importance(
        self,
        model: MultiOutputModel,
    ) -> None:
        """
        feature importance をプロットし、MLflowに保存する

        Args:
            model: MultiOutputModelインスタンス
        """
        feature_importance_dict = model.feature_importance  # Dict[str, List[float]]

        # 各特徴量について平均を計算
        feature_importance = {
            name: float(np.mean(importance_list))
            for name, importance_list in feature_importance_dict.items()
        }

        # 重要度でソート
        sorted_items = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        features = [item[0] for item in sorted_items]
        importances = [item[1] for item in sorted_items]

        fig = go.Figure(
            data=go.Bar(
                x=importances,
                y=features,
                orientation="h",
            )
        )

        fig.update_layout(
            title=f"Feature Importance ({len(features)} features)",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=max(800, len(features) * 20),  # 特徴量数に応じて高さを調整
            yaxis=dict(autorange="reversed"),  # 上位が上に来るように
        )

        log_figure(fig, artifact_path="figures/feature_importance.html")

    def feature_importance_map(
        self,
        model: MultiOutputModel,
    ) -> None:
        """
        特徴量重要度をheatmapでプロットし、MLflowに保存する

        Args:
            model: MultiOutputModelインスタンス
        """
        feature_importance_dict = model.feature_importance  # Dict[str, List[float]]

        # 特徴量名と重要度リストを取得
        feature_names = list(feature_importance_dict.keys())
        importance_matrix = np.array(
            [feature_importance_dict[name] for name in feature_names]
        )

        # horizonの数
        n_horizons = importance_matrix.shape[1]
        # horizonのリストを使用（self.horizonが設定されている場合）
        if self.horizon is not None and len(self.horizon) == n_horizons:
            horizons = self.horizon
        else:
            # フォールバック: horizonリストが利用できない場合は連番を使用
            horizons = list(range(n_horizons))

        # 最後のhorizonの重要度でソート（降順）
        last_horizon_importances = importance_matrix[:, -1]
        sorted_indices = np.argsort(last_horizon_importances)[::-1]
        feature_names_sorted = [feature_names[i] for i in sorted_indices]
        importance_matrix_sorted = importance_matrix[sorted_indices]

        # heatmapを作成
        fig = go.Figure(
            data=go.Heatmap(
                z=importance_matrix_sorted,
                x=horizons,
                y=feature_names_sorted,
                colorscale=[
                    [0, "#ffffff"],
                    [1, "#0066cc"],
                ],  # 白から濃い青の2色スケール
                colorbar=dict(title="Importance"),
            )
        )

        fig.update_layout(
            title=(
                f"Feature Importance Map "
                f"({len(feature_names)} features, {n_horizons} horizons)"
            ),
            xaxis_title="Horizon",
            yaxis_title="Features",
            height=max(800, len(feature_names) * 20),  # 特徴量数に応じて高さを調整
            yaxis=dict(autorange="reversed"),  # 上位が上に来るように
        )

        log_figure(fig, artifact_path="figures/feature_importance_map.html")
