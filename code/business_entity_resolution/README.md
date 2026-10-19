# Business Entity Resolution -- Pipeline README

Independent implementation for the Amazon ML Challenge 2026 (Business Entity
Resolution). Given three noisy business-record sources, this pipeline finds
every Source-2/Source-3 record that matches each Source-1 (deduplicated
reference) entity, optimizing directly for the official **macro F0.5**
metric.

This folder is fully self-contained: it does not import from, depend on, or
modify any other project. It only needs to be pointed at a dataset root via
`--data-root`.

---

## 1. Environment setup

- Python 3.13 (also runs on 3.10+)
- Works on Windows / macOS / Linux. Tuned for: Windows, 16 GB RAM, RTX 3050
  Laptop GPU (4 GB VRAM) -- **no GPU is required or used**; LightGBM trains
  on CPU with a small, laptop-friendly configuration.

Create a virtual environment (recommended):

```
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

## 2. Installation

From `code/business_entity_resolution/`:

```
pip install -r requirements.txt
```

Every dependency in `requirements.txt` is pinned and actually imported by the
code (see the comment at the top of that file for the one package -- scipy --
that was deliberately left out and why).

## 3. Dataset format

Point `--data-root` at the folder that contains `train/` and `test/`
(this is **not** copied into this project; the dataset stays wherever it
already lives on disk):

```
<data-root>/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

Each source file has columns `entity_id, business_name, business_address,
country`. `train_ground_truth.tsv` has `source1_entity_id,
matched_entity_ids` (comma-separated, empty for singletons). Training covers
US and India; the test set additionally has France. `country` is treated as
an open-set string everywhere in this code -- it is used as a *signal*, never
as a hard filter, so an unseen country never causes an entity to be silently
dropped.

## 4. Exact training command

```
python src/train.py --data-root "PATH_TO_DATASET"
```

Useful flags (all optional, shown with defaults):

```
python src/train.py --data-root "PATH_TO_DATASET" ^
  --max-candidates 150 ^
  --retrieval-top-k 40 ^
  --negative-ratio 5.0 ^
  --val-fraction 0.15 ^
  --num-boost-round 500 ^
  --early-stopping-rounds 30 ^
  --threads 4 ^
  --duckdb-memory-limit 4GB ^
  --seed 42
```

This normalizes both sources, runs multi-view blocking + retrieval, builds
labeled pairs (positives from ground truth, hard + random negatives),
trains LightGBM with a grouped train/validation split (no Source-1 entity
appears in both splits), sweeps thresholds to maximize validation macro F0.5,
and saves everything needed for `evaluate.py` / `predict.py` to
`../../artifacts/` (i.e. `Amazon_ML_Hackthon2/artifacts/`): `model.txt`,
`feature_columns.json`, `params.json`, `thresholds.json`,
`pipeline_config.json`.

## 5. Exact evaluation command

```
python src/evaluate.py --data-root "PATH_TO_DATASET"
```

Re-derives the same grouped validation split (same seed/fraction as
training by default), scores it with the saved model + thresholds, and
prints the macro F0.5, TP/FP/FN, and singleton accuracy. Optionally add
`--report-path artifacts/eval_report.json` to also write a JSON summary.

## 6. Exact prediction command

```
python src/predict.py --data-root "PATH_TO_DATASET"
```

Runs the full pipeline on `test/`, writes
`Amazon_ML_Hackthon2/output/matching_results.tsv` and
`Amazon_ML_Hackthon2/output/candidate_pairs.tsv`, and runs the built-in
validator against the official output rules (one row per Source-1 entity, no
duplicate IDs, no self-matches, matches are a subset of candidates, etc.).

## 7. `main.py` convenience wrapper

```
python src/main.py --data-root "PATH_TO_DATASET" --mode train
python src/main.py --data-root "PATH_TO_DATASET" --mode evaluate
python src/main.py --data-root "PATH_TO_DATASET" --mode predict
```

Every flag accepted by `train.py` / `evaluate.py` / `predict.py` also works
here -- `main.py` just forwards whatever follows `--mode` to the matching
module. The three scripts also run standalone, exactly as in sections 4-6.

## 8. Expected outputs

- `Amazon_ML_Hackthon2/artifacts/` -- trained model + thresholds (created by
  `train.py`; not part of the official submission directory list, but
  required at runtime so `predict.py`/`evaluate.py` have something to load).
- `Amazon_ML_Hackthon2/output/matching_results.tsv` and
  `candidate_pairs.tsv` -- the two files required by the challenge, written
  by `predict.py`.

## 9. Architecture

```
RAW DATA -> LOAD/VALIDATE (io.py) -> NORMALIZATION (normalize.py, aliases.py)
  -> MULTI-VIEW BLOCKING (blocking.py, DuckDB joins)
  -> RETRIEVAL / TRIM (retrieval.py, RapidFuzz re-ranking)
  -> FEATURES (features.py) -> HARD NEGATIVE MINING (hard_negatives.py)
  -> DATASET ASSEMBLY + GROUPED SPLIT (dataset.py)
  -> LIGHTGBM MODEL (model.py) -> THRESHOLD OPTIMIZATION (threshold.py, metric.py)
  -> CONSISTENCY POST-PROCESSING (consistency.py) -> OUTPUT (output.py)
```

`inference.py` wires these stages together for both training-time validation
and test-time prediction; `diagnostics.py` reports recall/class-balance/memory
at each stage.

## 10. Candidate generation

Blocking (`blocking.py`) builds several independent "key tables" per record
(rare name tokens, rare address tokens, name/address character 4-5-grams,
postal code, street/building numbers, sorted-core-name exact key) and joins
Source-1 against Source-2+Source-3 **per view** using DuckDB SQL rather than
Python nested loops, so it scales past the 10M-record scale without dense
matrices or giant Python lists. Views are UNIONed and ranked by how many
independent views agree; each Source-1 entity keeps at most
`--max-candidates` rows. `retrieval.py` then re-scores that (already small)
pool with RapidFuzz `WRatio` and trims to `--retrieval-top-k`, which is what
actually ends up in `candidate_pairs.tsv` -- exactly the pool the model
scores, per the challenge's definition of that file.

Run `diagnostics.candidate_report(...)` (printed automatically by
`train.py`/`evaluate.py`) to see candidate recall against ground truth at
each stage, plus average/max candidates per Source-1 entity.

## 11. Features

`features.py` computes ~30 pairwise features per candidate: exact/fuzzy name
similarity (WRatio, ratio, token-sort/set/partial ratio, token overlap and
Jaccard, character n-gram Jaccard, prefix/suffix match, rare-token
agreement), the same family for addresses plus postal-code and
street-number agreement, and cross-field signals (country agreement,
missing-country flag, combined number+postal agreement, number of blocking
views that surfaced the pair).

## 12. Model

`model.py` wraps LightGBM (MIT-licensed, far under the 8B-parameter model
constraint) as a binary classifier with early stopping on a grouped
validation split and automatic `scale_pos_weight` for class imbalance. It
exposes `save()`/`load()` (model + feature list + params as plain files) and
`feature_importance()` for inspection.

## 13. Threshold optimization

`threshold.py` sweeps thresholds from 0.05 to 0.95 directly against the
**macro F0.5** metric (`metric.py`, computed per the exact challenge
definition including singleton handling), first for a single global
threshold, then trying independent thresholds per candidate source
(`S2-*` / `S3-*`) -- the per-source scheme is only kept if it beats the
global threshold on the validation split.

## 14. Memory considerations

- DuckDB is given an explicit memory limit (`--duckdb-memory-limit`,
  default 4GB) and thread count (`--threads`), and does the blocking joins
  out-of-core rather than in Python.
- Feature extraction and scoring run in chunks of Source-1 entities
  (`--chunk-size`, default 2000) so peak RAM stays bounded regardless of test
  set size.
- No dense similarity matrix is ever built over the full dataset; retrieval
  re-ranking only touches each Source-1 entity's already-blocked candidate
  list.
- GPU is intentionally unused -- LightGBM's CPU training on these bounded
  candidate/feature tables is fast enough on a laptop, and the RTX 3050's 4GB
  VRAM would not meaningfully help a tabular GBDT model.

## 15. Reproducibility

All randomized steps (`--seed`, default 42) -- negative sampling, the
grouped train/val split, LightGBM's own seed -- are seeded. `pipeline_config.json`
and `thresholds.json` saved by `train.py` are re-loaded by `evaluate.py` and
`predict.py` so a prediction run always uses exactly the configuration that
produced the saved model.

## 16. Generating the final submission package

```
<team_name>_submission.zip
  output/matching_results.tsv        (from predict.py)
  output/candidate_pairs.tsv         (from predict.py)
  code/business_entity_resolution/   (this folder: src/, README.md, requirements.txt)
  Documentation_template.md          (filled in, at the project root)
```

From the `Amazon_ML_Hackthon2/` project root, after running `train.py` and
`predict.py`:

```
cd Amazon_ML_Hackthon2
zip -r <team_name>_submission.zip output code/business_entity_resolution Documentation_template.md
```

(On Windows, use Explorer's "Compress to ZIP" or `Compress-Archive` on the
same three paths.)
