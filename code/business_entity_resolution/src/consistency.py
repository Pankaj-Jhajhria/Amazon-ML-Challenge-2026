"""Conservative post-processing: resolve conflicting target assignments.

A Source-2/Source-3 record describes exactly one real-world business, so it
should not end up matched to two *different* Source-1 entities in the final
output. One Source-1 entity legitimately having many matches is untouched;
only cross-Source1 duplicate claims on the same target are pruned, keeping
the highest-confidence claim(s). This never forces one-to-one matching.
"""
from __future__ import annotations

import pandas as pd

from .utils import get_logger

LOGGER = get_logger(__name__)


def resolve_target_conflicts(final_df: pd.DataFrame, margin: float = 0.03) -> pd.DataFrame:
    """final_df columns: source1_entity_id, candidate_entity_id, prob.
    For each candidate_entity_id claimed by more than one source1_entity_id,
    keep only the claim(s) within `margin` of the top probability for that
    target."""
    if final_df.empty:
        return final_df

    best_per_target = final_df.groupby("candidate_entity_id")["prob"].transform("max")
    keep_mask = final_df["prob"] >= (best_per_target - margin)
    dropped = int((~keep_mask).sum())
    if dropped:
        LOGGER.info("Consistency post-processing dropped %d conflicting low-confidence claims", dropped)
    return final_df[keep_mask].reset_index(drop=True)
