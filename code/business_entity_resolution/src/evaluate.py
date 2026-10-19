"""Evaluate a trained matcher on a held-out validation split of the training data.

Example:
    python src/evaluate.py --data-root "C:/path/to/Amazon ML Hackthon/dataset"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .dataset import ground_truth_to_positive_pairs, ground_truth_to_true_matches, group_train_val_split
from .diagnostics import candidate_report, dataset_stats
from .features import build_meta_lookup, compute_features_batch, compute_rare_tokens
from .hard_negatives import label_pairs
from .inference import PipelineConfig, generate_candidates, prepare_normalized, retrieve_and_trim
from .io import read_ground_truth, read_source
from .metric import macro_f05_from_predictions
from .model import LGBMMatcher
from .threshold import apply_thresholds
from .utils import get_logger, set_seed, timer

LOGGER = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the trained matcher on a validation split.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=_PROJECT_ROOT / "artifacts")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)

    model = LGBMMatcher.load(args.model_dir)
    with open(args.model_dir / "thresholds.json", "r", encoding="utf-8") as f:
        threshold_config = json.load(f)
    with open(args.model_dir / "pipeline_config.json", "r", encoding="utf-8") as f:
        pipeline_config = PipelineConfig(**json.load(f))

    train_dir = args.data_root / "train"
    s1_df = read_source(train_dir / "train_source1.tsv")
    s2_df = read_source(train_dir / "train_source2.tsv")
    s3_df = read_source(train_dir / "train_source3.tsv")
    gt_df = read_ground_truth(train_dir / "train_ground_truth.tsv")
    dataset_stats(s1_df, s2_df, s3_df, gt_df)

    n1, n2, n3 = prepare_normalized(s1_df, s2_df, s3_df)
    n23 = pd.concat([n2, n3], ignore_index=True)

    candidates = generate_candidates(n1, n2, n3, pipeline_config)
    trimmed = retrieve_and_trim(n1, n23, candidates, pipeline_config)
    candidate_report(trimmed, gt_df)

    s1_meta = build_meta_lookup(n1)
    s23_meta = build_meta_lookup(n23)
    rare_tokens = compute_rare_tokens(n1, n23)

    with timer("evaluation feature extraction", LOGGER):
        feature_df = compute_features_batch(trimmed, s1_meta, s23_meta, rare_tokens)

    positive_pairs = ground_truth_to_positive_pairs(gt_df)
    labeled_df = label_pairs(feature_df, positive_pairs)

    _, val_ids = group_train_val_split(labeled_df["source1_entity_id"].values, val_fraction=args.val_fraction, random_state=args.seed)
    val_df = labeled_df[labeled_df["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
    val_df["prob"] = model.predict_proba(val_df) if len(val_df) else pd.Series(dtype=float)

    true_matches = ground_truth_to_true_matches(gt_df)
    predicted = apply_thresholds(val_df, threshold_config)
    result = macro_f05_from_predictions(true_matches, predicted, list(val_ids))

    LOGGER.info(
        "Validation macro F0.5: %.4f (n_entities=%d, singletons=%d/%d correct)",
        result["macro_f05"], result["n_entities"], result["n_singletons_correct"], result["n_singletons"],
    )
    LOGGER.info("TP=%d FP=%d FN=%d", result["total_tp"], result["total_fp"], result["total_fn"])

    if args.report_path:
        summary = {k: v for k, v in result.items() if k != "per_entity"}
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        LOGGER.info("Wrote evaluation report to %s", args.report_path)


if __name__ == "__main__":
    main()
