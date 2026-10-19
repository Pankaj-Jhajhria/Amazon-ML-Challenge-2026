"""Diagnostic reporting: dataset stats, candidate recall, class balance, memory."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .blocking import measure_candidate_recall
from .utils import get_logger, memory_usage_mb

LOGGER = get_logger(__name__)


def dataset_stats(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame, ground_truth_df: pd.DataFrame = None) -> Dict:
    stats = {
        "n_source1": len(s1_df),
        "n_source2": len(s2_df),
        "n_source3": len(s3_df),
        "countries": sorted(
            set(s1_df["country"].str.lower()) | set(s2_df["country"].str.lower()) | set(s3_df["country"].str.lower())
        ),
    }
    if ground_truth_df is not None:
        from .io import parse_id_list
        n_pos_pairs = sum(len(parse_id_list(m)) for m in ground_truth_df["matched_entity_ids"])
        n_singletons = sum(1 for m in ground_truth_df["matched_entity_ids"] if not parse_id_list(m))
        stats.update({
            "n_ground_truth_rows": len(ground_truth_df),
            "n_positive_pairs": n_pos_pairs,
            "n_singletons": n_singletons,
        })
    LOGGER.info("Dataset stats: %s", stats)
    return stats


def candidate_report(candidates_df: pd.DataFrame, ground_truth_df: pd.DataFrame = None) -> Dict:
    if candidates_df.empty:
        report = {"n_candidate_pairs": 0, "n_source1_with_candidates": 0, "avg_candidates_per_s1": 0.0, "max_candidates_per_s1": 0}
    else:
        per_s1_counts = candidates_df.groupby("source1_entity_id").size()
        report = {
            "n_candidate_pairs": len(candidates_df),
            "n_source1_with_candidates": candidates_df["source1_entity_id"].nunique(),
            "avg_candidates_per_s1": float(per_s1_counts.mean()),
            "max_candidates_per_s1": int(per_s1_counts.max()),
        }
    if ground_truth_df is not None:
        report.update(measure_candidate_recall(candidates_df, ground_truth_df))
    LOGGER.info("Candidate report: %s", report)
    return report


def class_balance(labeled_df: pd.DataFrame) -> Dict:
    n_pos = int((labeled_df["label"] == 1).sum())
    n_neg = int((labeled_df["label"] == 0).sum())
    report = {"n_positive": n_pos, "n_negative": n_neg, "ratio_neg_to_pos": (n_neg / n_pos) if n_pos else float("inf")}
    LOGGER.info("Class balance: %s", report)
    return report


def report_memory(label: str) -> None:
    mem = memory_usage_mb()
    if mem is not None:
        LOGGER.info("[MEMORY] %s: %.0f MB RSS", label, mem)
