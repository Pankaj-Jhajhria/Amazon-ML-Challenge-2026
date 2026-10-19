"""Assemble labeled training pairs and perform grouped train/validation splits."""
from __future__ import annotations

from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .io import parse_id_list


def ground_truth_to_positive_pairs(ground_truth_df: pd.DataFrame) -> Set[Tuple[str, str]]:
    pairs = set()
    for s1_id, matched in zip(ground_truth_df["source1_entity_id"], ground_truth_df["matched_entity_ids"]):
        for target in parse_id_list(matched):
            pairs.add((s1_id, target))
    return pairs


def ground_truth_to_true_matches(ground_truth_df: pd.DataFrame) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for s1_id, matched in zip(ground_truth_df["source1_entity_id"], ground_truth_df["matched_entity_ids"]):
        out[s1_id] = set(parse_id_list(matched))
    return out


def group_train_val_split(
    s1_ids: np.ndarray, val_fraction: float = 0.15, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Split unique Source1 entity ids into train/val groups (no S1 entity
    appears in both splits)."""
    unique_ids = np.unique(s1_ids)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    train_idx, val_idx = next(splitter.split(unique_ids, groups=unique_ids))
    return unique_ids[train_idx], unique_ids[val_idx]
