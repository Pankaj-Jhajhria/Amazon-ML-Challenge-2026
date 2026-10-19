"""Hard-negative mining: select difficult negative training pairs from the
candidate pool rather than relying only on random negatives, since false
merges are heavily penalized by the F0.5 metric.
"""
from __future__ import annotations

from typing import Set, Tuple

import pandas as pd

from .utils import get_logger

LOGGER = get_logger(__name__)


def label_pairs(feature_df: pd.DataFrame, positive_pairs: Set[Tuple[str, str]]) -> pd.DataFrame:
    out = feature_df.copy()
    keys = list(zip(out["source1_entity_id"], out["candidate_entity_id"]))
    out["label"] = [1 if k in positive_pairs else 0 for k in keys]
    return out


def select_training_pairs(
    labeled_df: pd.DataFrame, negative_ratio: float = 5.0, hard_fraction: float = 0.7, random_state: int = 42
) -> pd.DataFrame:
    """Down-sample negatives to `negative_ratio` * n_positives, biased towards
    'hard' negatives (high name/address similarity) so the classifier learns a
    sharp decision boundary instead of separating obviously-different pairs."""
    positives = labeled_df[labeled_df["label"] == 1]
    negatives = labeled_df[labeled_df["label"] == 0]

    n_pos = len(positives)
    if n_pos == 0:
        LOGGER.warning("No positive pairs found in labeled_df; returning as-is.")
        return labeled_df

    n_neg_target = max(int(n_pos * negative_ratio), 100)
    if len(negatives) <= n_neg_target:
        return pd.concat([positives, negatives], ignore_index=True)

    n_hard = int(n_neg_target * hard_fraction)
    n_random = n_neg_target - n_hard

    hardness = 0.5 * negatives["name_wratio"] + 0.3 * negatives["addr_wratio"] + 0.2 * negatives["name_token_set"]
    negatives = negatives.assign(_hardness=hardness)
    hard_negatives = negatives.sort_values("_hardness", ascending=False).head(n_hard)

    remaining = negatives.drop(hard_negatives.index)
    random_negatives = remaining.sample(n=min(n_random, len(remaining)), random_state=random_state)

    negatives_selected = pd.concat([hard_negatives, random_negatives], ignore_index=True)
    negatives_selected = negatives_selected.drop(columns=["_hardness"], errors="ignore")

    result = pd.concat([positives, negatives_selected], ignore_index=True)
    return result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
