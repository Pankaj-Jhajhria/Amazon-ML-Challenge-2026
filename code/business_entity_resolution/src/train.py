"""Train the LightGBM matcher end-to-end from the raw challenge dataset.

Example:
    python src/train.py --data-root "C:/path/to/Amazon ML Hackthon/dataset"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import diagnostics
from .dataset import ground_truth_to_positive_pairs, ground_truth_to_true_matches, group_train_val_split
from .features import FEATURE_COLUMNS, build_meta_lookup, compute_features_batch, compute_rare_tokens
from .hard_negatives import label_pairs, select_training_pairs
from .inference import PipelineConfig, generate_candidates, prepare_normalized, retrieve_and_trim
from .io import read_ground_truth, read_source
from .model import LGBMMatcher
from .threshold import select_thresholds
from .utils import ensure_dir, get_logger, set_seed, timer

LOGGER = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Business Entity Resolution matcher.")
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root containing train/ and test/")
    parser.add_argument("--model-dir", type=Path, default=_PROJECT_ROOT / "artifacts")
    parser.add_argument("--max-candidates", type=int, default=150, help="Max candidates per Source1 after blocking")
    parser.add_argument("--retrieval-top-k", type=int, default=40, help="Max candidates per Source1 after retrieval trim")
    parser.add_argument("--negative-ratio", type=float, default=5.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--duckdb-memory-limit", type=str, default="4GB")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)

    train_dir = args.data_root / "train"
    s1_df = read_source(train_dir / "train_source1.tsv")
    s2_df = read_source(train_dir / "train_source2.tsv")
    s3_df = read_source(train_dir / "train_source3.tsv")
    gt_df = read_ground_truth(train_dir / "train_ground_truth.tsv")

    diagnostics.dataset_stats(s1_df, s2_df, s3_df, gt_df)

    config = PipelineConfig(
        max_candidates_per_s1=args.max_candidates,
        retrieval_top_k=args.retrieval_top_k,
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_threads=args.threads,
        chunk_size=args.chunk_size,
    )

    n1, n2, n3 = prepare_normalized(s1_df, s2_df, s3_df)
    n23 = pd.concat([n2, n3], ignore_index=True)

    candidates = generate_candidates(n1, n2, n3, config)
    diagnostics.candidate_report(candidates, gt_df)

    trimmed = retrieve_and_trim(n1, n23, candidates, config)
    diagnostics.candidate_report(trimmed, gt_df)

    s1_meta = build_meta_lookup(n1)
    s23_meta = build_meta_lookup(n23)
    rare_tokens = compute_rare_tokens(n1, n23)

    with timer("training feature extraction", LOGGER):
        feature_df = compute_features_batch(trimmed, s1_meta, s23_meta, rare_tokens)

    positive_pairs = ground_truth_to_positive_pairs(gt_df)
    labeled_df = label_pairs(feature_df, positive_pairs)
    diagnostics.class_balance(labeled_df)

    train_ids, val_ids = group_train_val_split(
        labeled_df["source1_entity_id"].values, val_fraction=args.val_fraction, random_state=args.seed
    )
    train_mask = labeled_df["source1_entity_id"].isin(train_ids)
    train_df = labeled_df[train_mask].reset_index(drop=True)
    val_df = labeled_df[~train_mask].reset_index(drop=True)

    train_df = select_training_pairs(train_df, negative_ratio=args.negative_ratio, random_state=args.seed)
    diagnostics.class_balance(train_df)

    model = LGBMMatcher(feature_columns=FEATURE_COLUMNS)
    model.train(
        X_train=train_df,
        y_train=train_df["label"].values,
        X_val=val_df if len(val_df) else None,
        y_val=val_df["label"].values if len(val_df) else None,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        num_threads=args.threads,
    )

    LOGGER.info("Top features by gain:\n%s", model.feature_importance().head(15).to_string(index=False))

    val_df = val_df.copy()
    val_df["prob"] = model.predict_proba(val_df) if len(val_df) else pd.Series(dtype=float)

    true_matches = ground_truth_to_true_matches(gt_df)
    val_s1_ids = list(val_ids)
    threshold_config = select_thresholds(val_df, true_matches, val_s1_ids)
    LOGGER.info("Selected thresholds: %s", threshold_config)

    model_dir = ensure_dir(args.model_dir)
    model.save(model_dir)
    with open(model_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=2)
    with open(model_dir / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)

    LOGGER.info("Training complete. Validation macro F0.5: %.4f", threshold_config["macro_f05"])
    LOGGER.info("Model + thresholds saved to %s", model_dir)


if __name__ == "__main__":
    main()
