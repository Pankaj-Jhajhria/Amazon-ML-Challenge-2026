"""Write and validate the two required output TSV files."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd

from .io import parse_id_list, write_tsv
from .utils import get_logger

LOGGER = get_logger(__name__)


def _dedupe_join(ids: Iterable[str]) -> str:
    seen: List[str] = []
    seen_set = set()
    for i in ids:
        if i not in seen_set:
            seen_set.add(i)
            seen.append(i)
    return ",".join(seen)


def build_output_table(pairs_df: pd.DataFrame, all_s1_ids: List[str], id_col_name: str) -> pd.DataFrame:
    """pairs_df columns: source1_entity_id, candidate_entity_id."""
    grouped: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ids}
    if not pairs_df.empty:
        for s1_id, cand_id in zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"]):
            if s1_id in grouped and cand_id != s1_id:
                grouped[s1_id].append(cand_id)
    rows = [(s1, _dedupe_join(v)) for s1, v in grouped.items()]
    return pd.DataFrame(rows, columns=["source1_entity_id", id_col_name])


def write_matching_results(final_matches_df: pd.DataFrame, all_s1_ids: List[str], path: Path) -> None:
    table = build_output_table(final_matches_df, all_s1_ids, "matched_entity_ids")
    write_tsv(table, path)


def write_candidate_pairs(candidates_df: pd.DataFrame, all_s1_ids: List[str], path: Path) -> None:
    table = build_output_table(candidates_df, all_s1_ids, "candidate_entity_ids")
    write_tsv(table, path)


def validate_outputs(matching_path: Path, candidates_path: Path, valid_target_ids: Set[str], all_s1_ids: List[str]) -> List[str]:
    """Re-implements the challenge's own format checks so problems are caught
    locally instead of costing a leaderboard submission."""
    issues: List[str] = []

    match_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    cand_df = pd.read_csv(candidates_path, sep="\t", dtype=str, keep_default_na=False)

    if set(match_df.columns) != {"source1_entity_id", "matched_entity_ids"}:
        issues.append(f"matching_results.tsv has unexpected columns: {list(match_df.columns)}")
    if set(cand_df.columns) != {"source1_entity_id", "candidate_entity_ids"}:
        issues.append(f"candidate_pairs.tsv has unexpected columns: {list(cand_df.columns)}")

    match_ids = match_df["source1_entity_id"].tolist()
    cand_ids = cand_df["source1_entity_id"].tolist()
    all_s1_set = set(all_s1_ids)

    if len(match_ids) != len(set(match_ids)):
        issues.append("Duplicate source1_entity_id rows in matching_results.tsv")
    if len(cand_ids) != len(set(cand_ids)):
        issues.append("Duplicate source1_entity_id rows in candidate_pairs.tsv")

    missing_in_match = all_s1_set - set(match_ids)
    if missing_in_match:
        issues.append(f"{len(missing_in_match)} Source1 entities missing from matching_results.tsv")
    missing_in_cand = all_s1_set - set(cand_ids)
    if missing_in_cand:
        issues.append(f"{len(missing_in_cand)} Source1 entities missing from candidate_pairs.tsv")

    match_lookup = dict(zip(match_df["source1_entity_id"], match_df["matched_entity_ids"]))
    cand_lookup = dict(zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"]))

    for s1_id, matched_str in match_lookup.items():
        matched = parse_id_list(matched_str)
        if len(matched) != len(set(matched)):
            issues.append(f"Duplicate matched IDs for {s1_id}")
        for m in matched:
            if m == s1_id or m.startswith("S1-"):
                issues.append(f"Invalid self/Source1 reference in matches for {s1_id}: {m}")
            elif valid_target_ids and m not in valid_target_ids:
                issues.append(f"Unknown target id {m} in matches for {s1_id}")
        candidate_set = set(parse_id_list(cand_lookup.get(s1_id, "")))
        if not set(matched).issubset(candidate_set):
            issues.append(f"Matched IDs for {s1_id} are not a subset of its candidates")

    if issues:
        LOGGER.warning("Output validation found %d issue(s)", len(issues))
        for issue in issues:
            LOGGER.warning(" - %s", issue)
    else:
        LOGGER.info("Output validation PASSED")
    return issues
