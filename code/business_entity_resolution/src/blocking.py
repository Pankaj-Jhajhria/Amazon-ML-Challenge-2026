"""High-recall multi-view candidate generation using DuckDB SQL joins.

Rather than nested Python loops (which do not scale to millions of records),
each blocking "view" is expressed as a key table (entity_id, key_value), and
candidates are produced by joining the Source-1 key table against the
Source-2/Source-3 key table on matching key_value within each view, then
UNION-ing the results across views. DuckDB's out-of-core query engine does
the heavy lifting instead of Python, and country is only ever one signal
among several rather than a hard filter, so a mislabeled or unseen country
(e.g. France in the test set) cannot silently zero out a candidate set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import duckdb
import pandas as pd
from tqdm import tqdm

from .utils import get_logger, timer

LOGGER = get_logger(__name__)

MIN_TOKEN_LEN = 3
RARE_DF_THRESHOLD = 50  # key values shared by more records than this are dropped (too common to be useful)


def _explode_tokens(df: pd.DataFrame, id_col: str, token_col: str) -> pd.DataFrame:
    rows = []
    for eid, tok_str in tqdm(list(zip(df[id_col], df[token_col])), desc=f"tokens:{token_col}", leave=False):
        if not tok_str:
            continue
        for tok in set(tok_str.split(" ")):
            if len(tok) >= MIN_TOKEN_LEN:
                rows.append((eid, tok))
    return pd.DataFrame(rows, columns=["entity_id", "key_value"])


def _char_ngrams(text: str, n: int = 4) -> List[str]:
    text = text.replace(" ", "")
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def _explode_ngrams(df: pd.DataFrame, id_col: str, char_col: str, n: int = 4,
                     max_grams_per_record: int = 12) -> pd.DataFrame:
    rows = []
    for eid, text in tqdm(list(zip(df[id_col], df[char_col])), desc=f"ngrams:{char_col}", leave=False):
        grams = _char_ngrams(text, n=n)
        if not grams:
            continue
        if len(grams) > max_grams_per_record:
            step = len(grams) / max_grams_per_record
            grams = [grams[int(i * step)] for i in range(max_grams_per_record)]
        for g in set(grams):
            rows.append((eid, g))
    return pd.DataFrame(rows, columns=["entity_id", "key_value"])


def _exact_key(df: pd.DataFrame, id_col: str, value_col: str) -> pd.DataFrame:
    sub = df[[id_col, value_col]].rename(columns={id_col: "entity_id", value_col: "key_value"})
    sub = sub[sub["key_value"].astype(str).str.len() > 0]
    return sub.reset_index(drop=True)


def build_key_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build one key table per blocking view for a normalized source dataframe."""
    views: Dict[str, pd.DataFrame] = {}
    views["name_token"] = _explode_tokens(df, "entity_id", "name_tokens")
    views["addr_token"] = _explode_tokens(df, "entity_id", "addr_tokens")
    views["name_ngram"] = _explode_ngrams(df, "entity_id", "name_core", n=4)
    views["addr_ngram"] = _explode_ngrams(df, "entity_id", "addr_char", n=5)
    views["postal"] = _exact_key(df, "entity_id", "addr_postal")
    views["name_core_exact"] = _exact_key(df, "entity_id", "name_core_sorted")

    num_rows = []
    for eid, numbers in zip(df["entity_id"], df["addr_numbers"]):
        if not numbers:
            continue
        for num in set(numbers.split(",")):
            if num:
                num_rows.append((eid, num))
    views["addr_number"] = pd.DataFrame(num_rows, columns=["entity_id", "key_value"])
    return views


def _downweight_common_tokens(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """Drop key values so common they would blow up the candidate set (e.g. a
    name token shared by thousands of records)."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table_name}_freq AS
        SELECT key_value, COUNT(*) AS df
        FROM {table_name}
        GROUP BY key_value
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table_name}_filtered AS
        SELECT t.entity_id, t.key_value
        FROM {table_name} t
        JOIN {table_name}_freq f USING (key_value)
        WHERE f.df <= {RARE_DF_THRESHOLD}
        """
    )


@dataclass
class BlockingConfig:
    max_candidates_per_s1: int = 150
    duckdb_memory_limit: str = "4GB"
    duckdb_threads: int = 4


def generate_candidate_pairs(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame, config: BlockingConfig
) -> pd.DataFrame:
    """Return [source1_entity_id, candidate_entity_id, n_views], at most
    `config.max_candidates_per_s1` rows per Source-1 entity, produced by
    unioning several independent blocking views and ranking by how many
    independent views agree."""
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)

    with timer("build blocking key tables", LOGGER):
        s1_views = build_key_tables(s1_df)
        s23_views = build_key_tables(s23_df)

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA memory_limit='{config.duckdb_memory_limit}'")
    con.execute(f"PRAGMA threads={config.duckdb_threads}")

    view_names = sorted(set(s1_views) & set(s23_views))
    union_parts = []
    with timer("duckdb blocking joins", LOGGER):
        for view in tqdm(view_names, desc="blocking views"):
            s1_tbl, s23_tbl = s1_views[view], s23_views[view]
            if s1_tbl.empty or s23_tbl.empty:
                continue
            con.register("s1_keys", s1_tbl)
            con.register("s23_keys", s23_tbl)
            _downweight_common_tokens(con, "s1_keys")
            _downweight_common_tokens(con, "s23_keys")
            part = con.execute(
                """
                SELECT DISTINCT a.entity_id AS source1_entity_id,
                       b.entity_id AS candidate_entity_id,
                       ? AS view_name
                FROM s1_keys_filtered a
                JOIN s23_keys_filtered b USING (key_value)
                """,
                [view],
            ).fetchdf()
            union_parts.append(part)
            con.unregister("s1_keys")
            con.unregister("s23_keys")

    if not union_parts:
        con.close()
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "n_views"])

    all_pairs = pd.concat(union_parts, ignore_index=True)
    con.register("all_pairs", all_pairs)
    with timer("aggregate + cap candidates per Source1", LOGGER):
        capped = con.execute(
            """
            WITH agg AS (
                SELECT source1_entity_id, candidate_entity_id, COUNT(DISTINCT view_name) AS n_views
                FROM all_pairs
                GROUP BY source1_entity_id, candidate_entity_id
            ),
            ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY source1_entity_id ORDER BY n_views DESC, candidate_entity_id
                       ) AS rn
                FROM agg
            )
            SELECT source1_entity_id, candidate_entity_id, n_views
            FROM ranked
            WHERE rn <= ?
            """,
            [config.max_candidates_per_s1],
        ).fetchdf()
    con.close()
    return capped


def measure_candidate_recall(candidates_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict[str, float]:
    """Fraction of ground-truth positive pairs that survive candidate generation."""
    from .io import parse_id_list

    gt_pairs = set()
    gt_total = 0
    for s1_id, matched in zip(ground_truth_df["source1_entity_id"], ground_truth_df["matched_entity_ids"]):
        for target in parse_id_list(matched):
            gt_pairs.add((s1_id, target))
            gt_total += 1

    if gt_total == 0:
        return {"pair_recall": 1.0, "gt_pairs": 0, "found_pairs": 0}

    cand_pairs = set(zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]))
    found = len(gt_pairs & cand_pairs)
    return {"pair_recall": found / gt_total, "gt_pairs": gt_total, "found_pairs": found}
