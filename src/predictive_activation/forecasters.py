"""Congestion-onset forecasters. All expose `risk(features) -> P(onset within HORIZON)`.

scikit-learn models are the default (fast, dependency-light, run anywhere). A PyTorch LSTM/GRU
implementation lives in `torch_models.py` and plugs into the same interface.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


class ScoreBaseline:
    """Heuristic risk scores that need no training (what a threshold alarm effectively uses)."""

    def __init__(self, feature: str, scale: float = 1.0):
        self.feature, self.scale, self.name = feature, scale, f"threshold on {feature}"

    def fit(self, X, y):
        return self

    def risk(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X[:, _IDX[self.feature]] * self.scale, 0.0, 1.0)


class SkForecaster:
    def __init__(self, kind: str = "gbm", seed: int = 0):
        self.kind = kind
        if kind == "logreg":
            self.est = make_pipeline(SimpleImputer(strategy="constant", fill_value=-1.0), StandardScaler(),
                                     LogisticRegression(max_iter=500, C=1.0))
            self.name = "Logistic regression"
        elif kind == "gbm":
            self.est = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_depth=5,
                                                      l2_regularization=1.0, random_state=seed)
            self.name = "Gradient-boosted trees"
        else:
            raise ValueError(kind)

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def risk(self, X: np.ndarray) -> np.ndarray:
        return self.est.predict_proba(X)[:, 1]
