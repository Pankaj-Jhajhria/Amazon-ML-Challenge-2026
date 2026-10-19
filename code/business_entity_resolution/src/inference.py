"""End-to-end inference pipeline: raw sources -> candidates -> features ->
model scores -> thresholds -> consistency -> final match sets.

Candidate generation runs over the whole source set at once (DuckDB handles
the heavy join out-of-core), but feature extraction, scoring and assembly run
in chunks of Source-1 entities to keep peak RAM bounded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .blocking import BlockingConfig, generate_candidate_pairs
from .consistency import resolve_target_conflicts
from .features import build_meta_lookup, compute_features_batch, compute_rare_tokens
from .model import LGBMMatcher
from .normalize import normalize_dataframe
from .retrieval import build_text_lookup, rerank_and_trim
from .threshold import apply_thresholds
from .utils import chunked, get_logger, timer

LOGGER = get_logger(__name__)


@dataclass
class PipelineConfig:
    max_candidates_per_s1: int = 150
    retrieval_top_k: int = 40
    duckdb_memory_limit: str = "4GB"
    duckdb_threads: int = 4
    chunk_size: int = 2000


def prepare_normalized(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
    with timer("normalization", LOGGER):
        n1 = normalize_dataframe(s1_df)
        n2 = normalize_dataframe(s2_df)
        n3 = normalize_dataframe(s3_df)
    return n1, n2, n3


def generate_candidates(n1: pd.DataFrame, n2: pd.DataFrame, n3: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    blocking_config = BlockingConfig(
        max_candidates_per_s1=config.max_candidates_per_s1,
        duckdb_memory_limit=config.duckdb_memory_limit,
        duckdb_threads=config.duckdb_threads,
    )
    return generate_candidate_pairs(n1, n2, n3, blocking_config)


def retrieve_and_trim(n1: pd.DataFrame, n23: pd.DataFrame, candidates: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    s1_lookup = build_text_lookup(n1)
    s23_lookup = build_text_lookup(n23)
    return rerank_and_trim(candidates, s1_lookup, s23_lookup, top_k=config.retrieval_top_k)


def score_candidates(
    trimmed_candidates: pd.DataFrame, n1: pd.DataFrame, n23: pd.DataFrame, model: LGBMMatcher, config: PipelineConfig
) -> pd.DataFrame:
    """Extract features and score candidates in chunks of Source1 ids."""
    s1_meta = build_meta_lookup(n1)
    s23_meta = build_meta_lookup(n23)
    rare_tokens = compute_rare_tokens(n1, n23)

    s1_ids = trimmed_candidates["source1_entity_id"].unique().tolist()
    scored_chunks: List[pd.DataFrame] = []
    with timer("chunked feature extraction + scoring", LOGGER):
        for id_chunk in chunked(s1_ids, config.chunk_size):
            chunk_pairs = trimmed_candidates[trimmed_candidates["source1_entity_id"].isin(id_chunk)]
            feats = compute_features_batch(chunk_pairs, s1_meta, s23_meta, rare_tokens)
            if feats.empty:
                continue
            feats["prob"] = model.predict_proba(feats)
            scored_chunks.append(feats[["source1_entity_id", "candidate_entity_id", "prob"]])
    if not scored_chunks:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "prob"])
    return pd.concat(scored_chunks, ignore_index=True)


def assemble_final_matches(scored_df: pd.DataFrame, threshold_config: Dict, consistency_margin: float = 0.03) -> pd.DataFrame:
    predicted = apply_thresholds(scored_df, threshold_config)
    keep_pairs = set()
    for s1_id, targets in predicted.items():
        for t in targets:
            keep_pairs.add((s1_id, t))
    if not keep_pairs or scored_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "prob"])
    mask = [
        (s1, c) in keep_pairs
        for s1, c in zip(scored_df["source1_entity_id"], scored_df["candidate_entity_id"])
    ]
    final_df = scored_df[mask].reset_index(drop=True)
    return resolve_target_conflicts(final_df, margin=consistency_margin)


def run_full_pipeline(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    model: LGBMMatcher,
    threshold_config: Dict,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, pd.DataFrame]:
    config = config or PipelineConfig()
    n1, n2, n3 = prepare_normalized(s1_df, s2_df, s3_df)
    n23 = pd.concat([n2, n3], ignore_index=True)

    candidates = generate_candidates(n1, n2, n3, config)
    trimmed = retrieve_and_trim(n1, n23, candidates, config)
    scored = score_candidates(trimmed, n1, n23, model, config)
    final_matches = assemble_final_matches(scored, threshold_config)
    return {"candidates": trimmed, "scored": scored, "final_matches": final_matches}
