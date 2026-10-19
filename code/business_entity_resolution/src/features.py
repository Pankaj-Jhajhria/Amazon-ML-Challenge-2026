"""Pairwise feature engineering for the Source1 x candidate matching problem."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from .utils import get_logger, timer

LOGGER = get_logger(__name__)

FEATURE_COLUMNS: List[str] = [
    "name_exact", "name_core_exact", "name_wratio", "name_ratio",
    "name_token_sort", "name_token_set", "name_partial",
    "name_token_overlap", "name_token_coverage", "name_shared_tokens",
    "name_char_ngram_jaccard", "name_prefix_match", "name_suffix_match",
    "name_len_ratio", "name_rare_token_agree",
    "addr_exact", "addr_wratio", "addr_token_sort", "addr_token_set",
    "addr_token_overlap", "addr_token_coverage", "addr_char_ngram_jaccard",
    "addr_number_overlap", "addr_number_jaccard", "postal_match",
    "addr_len_ratio",
    "country_exact", "country_missing_either",
    "combo_number_agreement", "n_views",
]


def _tok_set(s: str) -> set:
    return set(t for t in (s or "").split(" ") if t)


def _char_ngrams(s: str, n: int = 3) -> set:
    s = (s or "").replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a or ""), len(b or "")
    if la == 0 and lb == 0:
        return 1.0
    return (min(la, lb) / max(la, lb)) if max(la, lb) else 0.0


def _num_set(s: str) -> set:
    return set(x for x in (s or "").split(",") if x)


def _row_pair_features(s1: Dict, s2: Dict, n_views: float, rare_tokens: set) -> Dict[str, float]:
    n1_core, n2_core = s1["name_core"], s2["name_core"]
    n1_tok, n2_tok = _tok_set(s1["name_tokens"]), _tok_set(s2["name_tokens"])
    a1_char, a2_char = s1["addr_char"], s2["addr_char"]
    a1_tok, a2_tok = _tok_set(s1["addr_tokens"]), _tok_set(s2["addr_tokens"])

    shared_name_tokens = n1_tok & n2_tok
    name_rare_agree = len(shared_name_tokens & rare_tokens)

    n1_nums, n2_nums = _num_set(s1["addr_numbers"]), _num_set(s2["addr_numbers"])
    number_overlap = len(n1_nums & n2_nums)

    postal1, postal2 = s1["addr_postal"], s2["addr_postal"]
    postal_match = 1.0 if postal1 and postal2 and postal1 == postal2 else 0.0

    country1, country2 = s1["country_norm"], s2["country_norm"]
    country_missing = 1.0 if (not country1 or not country2) else 0.0
    country_exact = 1.0 if country1 and country2 and country1 == country2 else 0.0

    return {
        "name_exact": 1.0 if s1["name_char"] == s2["name_char"] and s1["name_char"] else 0.0,
        "name_core_exact": 1.0 if n1_core == n2_core and n1_core else 0.0,
        "name_wratio": fuzz.WRatio(n1_core, n2_core) / 100.0,
        "name_ratio": fuzz.ratio(n1_core, n2_core) / 100.0,
        "name_token_sort": fuzz.token_sort_ratio(n1_core, n2_core) / 100.0,
        "name_token_set": fuzz.token_set_ratio(n1_core, n2_core) / 100.0,
        "name_partial": fuzz.partial_ratio(n1_core, n2_core) / 100.0,
        "name_token_overlap": float(len(shared_name_tokens)),
        "name_token_coverage": _jaccard(n1_tok, n2_tok),
        "name_shared_tokens": float(len(shared_name_tokens)),
        "name_char_ngram_jaccard": _jaccard(_char_ngrams(n1_core), _char_ngrams(n2_core)),
        "name_prefix_match": 1.0 if n1_core[:4] == n2_core[:4] and n1_core else 0.0,
        "name_suffix_match": 1.0 if n1_core[-4:] == n2_core[-4:] and n1_core else 0.0,
        "name_len_ratio": _len_ratio(n1_core, n2_core),
        "name_rare_token_agree": float(name_rare_agree),
        "addr_exact": 1.0 if a1_char == a2_char and a1_char else 0.0,
        "addr_wratio": fuzz.WRatio(a1_char, a2_char) / 100.0,
        "addr_token_sort": fuzz.token_sort_ratio(a1_char, a2_char) / 100.0,
        "addr_token_set": fuzz.token_set_ratio(a1_char, a2_char) / 100.0,
        "addr_token_overlap": float(len(a1_tok & a2_tok)),
        "addr_token_coverage": _jaccard(a1_tok, a2_tok),
        "addr_char_ngram_jaccard": _jaccard(_char_ngrams(a1_char, 4), _char_ngrams(a2_char, 4)),
        "addr_number_overlap": float(number_overlap),
        "addr_number_jaccard": _jaccard(n1_nums, n2_nums),
        "postal_match": postal_match,
        "addr_len_ratio": _len_ratio(a1_char, a2_char),
        "country_exact": country_exact,
        "country_missing_either": country_missing,
        "combo_number_agreement": 1.0 if number_overlap > 0 and postal_match else float(number_overlap > 0),
        "n_views": float(n_views),
    }


def compute_rare_tokens(*dfs: pd.DataFrame, df_threshold: int = 25) -> set:
    counter: Counter = Counter()
    for df in dfs:
        for tok_str in df["name_tokens"]:
            counter.update(set((tok_str or "").split(" ")))
    return {tok for tok, cnt in counter.items() if 0 < cnt <= df_threshold}


def build_meta_lookup(df: pd.DataFrame) -> Dict[str, Dict]:
    cols = [
        "entity_id", "name_char", "name_tokens", "name_core", "name_core_sorted",
        "addr_char", "addr_tokens", "addr_numbers", "addr_postal", "addr_unit", "country_norm",
    ]
    lookup: Dict[str, Dict] = {}
    for row in df[cols].itertuples(index=False):
        d = row._asdict()
        lookup[d["entity_id"]] = d
    return lookup


def compute_features_batch(
    pairs_df: pd.DataFrame, s1_meta: Dict[str, Dict], s23_meta: Dict[str, Dict], rare_tokens: set
) -> pd.DataFrame:
    """pairs_df must have columns source1_entity_id, candidate_entity_id[, n_views]."""
    empty_cols = ["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS
    if pairs_df.empty:
        return pd.DataFrame(columns=empty_cols)

    has_views = "n_views" in pairs_df.columns
    records = []
    with timer(f"feature extraction ({len(pairs_df)} pairs)", LOGGER):
        for row in tqdm(pairs_df.itertuples(index=False), total=len(pairs_df), desc="features", leave=False):
            s1_id = row.source1_entity_id
            cand_id = row.candidate_entity_id
            n_views = getattr(row, "n_views", 1) if has_views else 1
            s1 = s1_meta.get(s1_id)
            s2 = s23_meta.get(cand_id)
            if s1 is None or s2 is None:
                continue
            feats = _row_pair_features(s1, s2, n_views, rare_tokens)
            feats["source1_entity_id"] = s1_id
            feats["candidate_entity_id"] = cand_id
            records.append(feats)

    if not records:
        return pd.DataFrame(columns=empty_cols)
    out = pd.DataFrame.from_records(records)
    return out[empty_cols]
