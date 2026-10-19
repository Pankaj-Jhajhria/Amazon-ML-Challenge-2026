"""Threshold selection that directly optimizes macro F0.5 on a validation split."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metric import macro_f05_from_predictions
from .utils import get_logger, timer

LOGGER = get_logger(__name__)

THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.01)]


def _source_prefix(entity_id: str) -> str:
    return entity_id.split("-")[0] if "-" in entity_id else entity_id


def _predictions_at_threshold(
    val_df: pd.DataFrame, threshold: float, threshold_by_source: Optional[Dict[str, float]] = None
) -> Dict[str, set]:
    preds: Dict[str, set] = {}
    for row in val_df.itertuples(index=False):
        thr = threshold
        if threshold_by_source is not None:
            thr = threshold_by_source.get(_source_prefix(row.candidate_entity_id), threshold)
        if row.prob >= thr:
            preds.setdefault(row.source1_entity_id, set()).add(row.candidate_entity_id)
    return preds


def optimize_global_threshold(val_df: pd.DataFrame, true_matches: Dict[str, set], all_s1_ids: List[str]) -> Dict:
    best = {"threshold": 0.5, "macro_f05": -1.0}
    with timer("global threshold sweep", LOGGER):
        for t in tqdm(THRESHOLD_GRID, desc="threshold sweep", leave=False):
            preds = _predictions_at_threshold(val_df, t)
            result = macro_f05_from_predictions(true_matches, preds, all_s1_ids)
            if result["macro_f05"] > best["macro_f05"]:
                best = {"threshold": float(t), **result}
    return best


def optimize_per_source_threshold(
    val_df: pd.DataFrame, true_matches: Dict[str, set], all_s1_ids: List[str], global_threshold: float
) -> Dict:
    """Try independent thresholds for each candidate source prefix (S2-/S3-);
    only used if it beats a single global threshold on validation."""
    if val_df.empty:
        return {"use_per_source": False}
    sources = sorted(set(_source_prefix(x) for x in val_df["candidate_entity_id"].unique()))
    if len(sources) < 2:
        return {"use_per_source": False}

    best_by_source: Dict[str, float] = {s: global_threshold for s in sources}
    with timer("per-source threshold sweep", LOGGER):
        for source in sources:
            best_local = {"threshold": global_threshold, "score": -1.0}
            for t in THRESHOLD_GRID:
                trial = dict(best_by_source)
                trial[source] = float(t)
                preds = _predictions_at_threshold(val_df, global_threshold, trial)
                result = macro_f05_from_predictions(true_matches, preds, all_s1_ids)
                if result["macro_f05"] > best_local["score"]:
                    best_local = {"threshold": float(t), "score": result["macro_f05"]}
            best_by_source[source] = best_local["threshold"]

    preds = _predictions_at_threshold(val_df, global_threshold, best_by_source)
    final = macro_f05_from_predictions(true_matches, preds, all_s1_ids)
    return {"use_per_source": True, "thresholds": best_by_source, "macro_f05": final["macro_f05"]}


def select_thresholds(val_df: pd.DataFrame, true_matches: Dict[str, set], all_s1_ids: List[str]) -> Dict:
    global_result = optimize_global_threshold(val_df, true_matches, all_s1_ids)
    per_source_result = optimize_per_source_threshold(val_df, true_matches, all_s1_ids, global_result["threshold"])
    if per_source_result.get("use_per_source") and per_source_result["macro_f05"] > global_result["macro_f05"]:
        return {
            "mode": "per_source",
            "global_threshold": global_result["threshold"],
            "per_source_thresholds": per_source_result["thresholds"],
            "macro_f05": per_source_result["macro_f05"],
        }
    return {
        "mode": "global",
        "global_threshold": global_result["threshold"],
        "per_source_thresholds": None,
        "macro_f05": global_result["macro_f05"],
    }


def apply_thresholds(val_df: pd.DataFrame, threshold_config: Dict) -> Dict[str, set]:
    if threshold_config["mode"] == "per_source":
        return _predictions_at_threshold(
            val_df, threshold_config["global_threshold"], threshold_config["per_source_thresholds"]
        )
    return _predictions_at_threshold(val_df, threshold_config["global_threshold"])
