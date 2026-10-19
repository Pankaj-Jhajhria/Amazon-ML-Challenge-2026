"""LightGBM-based pairwise match classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from .utils import get_logger, timer

LOGGER = get_logger(__name__)

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "verbosity": -1,
    "seed": 42,
}


class LGBMMatcher:
    """MIT-licensed (LightGBM) tabular classifier -- well within the challenge's
    "MIT/Apache-2.0, <=8B parameters" model constraint."""

    def __init__(self, feature_columns: List[str], params: Optional[dict] = None):
        self.feature_columns = feature_columns
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.booster: Optional[lgb.Booster] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 30,
        num_threads: int = 4,
    ) -> "LGBMMatcher":
        params = {**self.params, "num_threads": num_threads}
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        if n_pos > 0:
            params["scale_pos_weight"] = max(n_neg / n_pos, 1.0)

        train_set = lgb.Dataset(X_train[self.feature_columns], label=y_train)
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=0)]
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_set = lgb.Dataset(X_val[self.feature_columns], label=y_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

        with timer("LightGBM training", LOGGER):
            self.booster = lgb.train(
                params,
                train_set,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model has not been trained or loaded.")
        best_iter = self.booster.best_iteration or self.booster.current_iteration()
        return self.booster.predict(X[self.feature_columns], num_iteration=best_iter)

    def feature_importance(self) -> pd.DataFrame:
        if self.booster is None:
            raise RuntimeError("Model has not been trained or loaded.")
        importance = self.booster.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": self.feature_columns, "gain": importance})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        if self.booster is None:
            raise RuntimeError("No trained booster to save.")
        self.booster.save_model(str(model_dir / "model.txt"))
        with open(model_dir / "feature_columns.json", "w", encoding="utf-8") as f:
            json.dump(self.feature_columns, f)
        with open(model_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(self.params, f)

    @classmethod
    def load(cls, model_dir: Path) -> "LGBMMatcher":
        model_dir = Path(model_dir)
        with open(model_dir / "feature_columns.json", "r", encoding="utf-8") as f:
            feature_columns = json.load(f)
        with open(model_dir / "params.json", "r", encoding="utf-8") as f:
            params = json.load(f)
        obj = cls(feature_columns=feature_columns, params=params)
        obj.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        return obj
