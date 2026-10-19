"""Re-rank and trim blocking candidates using fast string-similarity scoring.

This keeps the final candidate pool (candidate_pairs.tsv) small and
precision-friendly without ever building a dense similarity matrix over the
whole dataset: only the (already blocked) candidates for each Source-1
entity are scored, using RapidFuzz's C-accelerated ratio functions.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from .utils import get_logger, timer

LOGGER = get_logger(__name__)


def _combined_text(name_core: str, addr_char: str) -> str:
    return f"{name_core} {addr_char}".strip()


def build_text_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return {
        eid: _combined_text(name, addr)
        for eid, name, addr in zip(df["entity_id"], df["name_core"], df["addr_char"])
    }


def rerank_and_trim(
    candidates_df: pd.DataFrame, s1_lookup: Dict[str, str], s23_lookup: Dict[str, str], top_k: int = 40
) -> pd.DataFrame:
    if candidates_df.empty:
        out = candidates_df.copy()
        out["retrieval_score"] = pd.Series(dtype=float)
        return out

    scores = []
    pairs = list(zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]))
    with timer(f"rapidfuzz re-ranking ({len(pairs)} pairs)", LOGGER):
        for s1_id, cand_id in tqdm(pairs, desc="retrieval scoring", leave=False):
            s1_text = s1_lookup.get(s1_id, "")
            cand_text = s23_lookup.get(cand_id, "")
            scores.append(0.0 if not s1_text or not cand_text else fuzz.WRatio(s1_text, cand_text))

    out = candidates_df.copy()
    out["retrieval_score"] = scores
    out = (
        out.sort_values(["source1_entity_id", "retrieval_score"], ascending=[True, False])
        .groupby("source1_entity_id", sort=False, group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return out
