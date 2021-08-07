from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # matplotlibの日本語表示対応


@dataclass
class PartialDependence:
    """Partial Dependence (PD)

    Args:
        estimator: 学習済みモデル
        X: 特徴量
    """

    estimator: Any
    X: np.ndarray

    def _counterfactual_prediction(
        self, X: np.ndarray, j: int, xj: float
    ) -> np.ndarray:
        """ある特徴量の値を置き換えたときの予測値を求める

        Args:
            j: 値を置き換える特徴量のインデックス
            xj: 置き換える値
        """

        # 特徴量の値を置き換える際、元データが上書きされないようコピー
        X_replaced = X.copy()

        # 特徴量の値を置き換えて予測し、平均をとって返す
        X_replaced[:, j] = xj
        average_prediction = self.estimator.predict(X_replaced).mean()

        return average_prediction

    def partial_dependence(self, j: int, n_grid: int = 30) -> None:
        """PDを求める

        Args:
            j: PDを計算したい特徴量の名前
            n_grid: グリッドを何分割するか
        """

        # ターゲットの変数を、取りうる値の最大値から最小値まで動かせるようにする
        xjs = np.linspace(self.X[:, j].min(), self.X[:, j].max(), num=n_grid)

        # インスタンスごとのモデルの予測値を平均
        average_prediction = np.array(
            [self._counterfactual_prediction(self.X, j, xj) for xj in xjs]
        )

        self.j = j
        self.result = {"values": xjs, "pred": average_prediction}

    def plot(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """可視化を行う"""
        
        # 真の関数形
        y = f(self.result["values"])

        fig, ax = plt.subplots()
        
        # 真の関係を可視化
        ax.plot(self.result["values"], y, zorder=1, c=".7", label="真の関係")
        # 推定された関係を可視化
        ax.plot(self.result["values"], self.result["pred"], zorder=2, label="推定された関係")
        
        ax.set(xlabel=f"X{self.j}", ylabel=ylabel)
        ax.legend()
        fig.suptitle(title)

        fig.show()


class Marginal(PartialDependence):
    """Marginal Plot"""

    def marginal(self, j: int, n_grid: int = 30) -> None:
        """Marginal Plotのためのデータを求める

        Args:
            j:
        """

        # ターゲットの変数を、取りうる値の最大値から最小値まで動かせるようにする
        xjks = np.quantile(self.X[:, j], q=np.arange(0, 1, 1 / n_grid))

        marginals = np.zeros(n_grid)
        for k in range(1, n_grid):
            mask = (self.X[:, j] >= xjks[k - 1]) & (self.X[:, j] <= xjks[k])

            marginals[k] = self._counterfactual_prediction(
                self.X[mask], j, (xjks[k - 1] + xjks[k]) / 2
            )

        self.j = j
        self.result = {"values": xjks, "pred": marginals}


class AccumulatedLocalEffects(Marginal):
    """Accumulated Local Effects (ALE)"""

    def accumulated_local_effects(self, j: int, n_grid: int = 30) -> None:
        """ALEを求める

        Args:
            j:
        """

        # ターゲットの変数を、取りうる値の最大値から最小値まで動かせるようにする
        xjks = np.quantile(self.X[:, j], q=np.arange(0, 1, 1 / n_grid))

        local_effects = np.zeros(n_grid)
        for k in range(1, n_grid):
            mask = (self.X[:, j] >= xjks[k - 1]) & (self.X[:, j] <= xjks[k])

            local_effects[k] = self._counterfactual_prediction(
                self.X[mask], j, xjks[k]
            ) - self._counterfactual_prediction(self.X[mask], j, xjks[k - 1])

        ale = np.cumsum(local_effects)

        self.j = j
        self.result = {"values": xjks, "pred": ale}