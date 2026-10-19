"""IO helpers for reading challenge TSV files and writing output TSVs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import pandas as pd

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def read_source(path: Path) -> pd.DataFrame:
    """Read one source1/source2/source3 TSV file into a DataFrame of strings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_NONE,
        engine="python",
    )
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    for col in REQUIRED_SOURCE_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    return df[REQUIRED_SOURCE_COLUMNS]


def read_ground_truth(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_NONE,
        engine="python",
    )
    missing = [c for c in REQUIRED_GT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").astype(str)
    return df[REQUIRED_GT_COLUMNS]


def parse_id_list(cell: str) -> List[str]:
    cell = (cell or "").strip()
    if not cell:
        return []
    return [x.strip() for x in cell.split(",") if x.strip()]


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
