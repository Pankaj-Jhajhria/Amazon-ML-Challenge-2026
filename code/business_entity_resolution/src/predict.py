"""Generate final predictions for the challenge test set.

Example:
    python src/predict.py --data-root "C:/path/to/Amazon ML Hackthon/dataset"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inference import PipelineConfig, run_full_pipeline
from .io import read_source
from .model import LGBMMatcher
from .output import validate_outputs, write_candidate_pairs, write_matching_results
from .utils import get_logger, set_seed, timer

LOGGER = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference and produce the submission TSVs.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=_PROJECT_ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "output")
    parser.add_argument("--seed", type=int, default=42)
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

    test_dir = args.data_root / "test"
    s1_df = read_source(test_dir / "test_source1.tsv")
    s2_df = read_source(test_dir / "test_source2.tsv")
    s3_df = read_source(test_dir / "test_source3.tsv")

    with timer("full inference pipeline", LOGGER):
        result = run_full_pipeline(s1_df, s2_df, s3_df, model, threshold_config, pipeline_config)

    all_s1_ids = s1_df["entity_id"].tolist()
    valid_target_ids = set(s2_df["entity_id"]) | set(s3_df["entity_id"])

    matching_path = args.output_dir / "matching_results.tsv"
    candidate_path = args.output_dir / "candidate_pairs.tsv"

    write_matching_results(result["final_matches"], all_s1_ids, matching_path)
    write_candidate_pairs(result["candidates"], all_s1_ids, candidate_path)

    issues = validate_outputs(matching_path, candidate_path, valid_target_ids, all_s1_ids)
    if issues:
        LOGGER.warning("Output validation found %d issue(s); see log above.", len(issues))
    else:
        LOGGER.info("Outputs written and validated successfully: %s, %s", matching_path, candidate_path)


if __name__ == "__main__":
    main()
