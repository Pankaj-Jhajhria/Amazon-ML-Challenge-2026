# Business Entity Resolution -- Methodology Document

**Team:** _fill in_
**Submission:** independent implementation, `code/business_entity_resolution/`

> All numbers in this document must come from actually running
> `train.py` / `evaluate.py` / `predict.py` on your machine against the real
> dataset. Every metric below is left as `TBD` -- fill it in from your own
> run's logs (`train.py` prints candidate recall, class balance, selected
> thresholds and validation macro F0.5; `evaluate.py` prints the same macro
> F0.5 plus TP/FP/FN and singleton accuracy). Do not invent numbers.

## 1. Approach overview

The pipeline follows: normalize -> multi-view blocking (high recall) ->
RapidFuzz retrieval trim -> rich pairwise feature engineering -> hard
negative mining -> LightGBM binary classifier -> macro-F0.5-optimized
thresholding -> conservative consistency post-processing -> output. The
guiding principle is that candidate generation sets a hard ceiling on
achievable recall, so it is optimized for recall first; everything after
that (features, model, threshold) is optimized for precision, since the
official metric (macro F0.5) weights precision twice as heavily as recall.

## 2. Preprocessing / normalization

For each record we keep several parallel representations rather than
collapsing to one string (see `normalize.py`, `aliases.py`):

- **Name:** Unicode-normalized + transliterated + lowercased
  character-normalized form; a token-normalized form with business
  abbreviations expanded (`mfg` -> `manufacturing`, `&` -> `and`, ...); a
  "core" form with legal suffixes (`Corp`, `Ltd`, `Pvt`, `LLC`, ...) stripped
  after canonicalization, used for suffix-insensitive matching.
- **Address:** character-normalized form; a token-normalized form with
  common abbreviations expanded (`St`->`Street`, `Rd`->`Road`, directional
  terms, apartment/suite markers, ...); extracted street/building numbers;
  a regex-extracted postal/PIN/ZIP code; an extracted unit/suite/floor
  identifier.

`country` is lowercased/trimmed but never filtered on -- it is treated as an
open string set everywhere downstream (blocking signal, feature, nothing
more), so the France-only test country is handled the same way as US/India
without any special-casing.

## 3. Candidate generation / blocking strategy

`blocking.py` builds several independent key tables per source (rare name
tokens, rare address tokens, name/address character n-grams, postal code,
street/building numbers, sorted-core-name exact key) and joins Source-1
against Source-2+Source-3 per view using DuckDB SQL (out-of-core, no dense
matrices, no giant Python candidate lists). Views are UNIONed; a candidate
kept by more independent views ranks higher, and each Source-1 entity is
capped at `--max-candidates` (default 150) raw blocking candidates.

`retrieval.py` then re-scores that pool with RapidFuzz `WRatio` on the
combined core-name + address text and trims to `--retrieval-top-k` (default
40) -- this trimmed set is exactly what is written to `candidate_pairs.tsv`
and what the model actually scores.

**Candidate recall (from your `train.py` / `evaluate.py` log):**

| Stage | Candidate pairs | Pair recall vs. ground truth |
|---|---|---|
| After blocking (pre-retrieval-trim) | TBD | TBD |
| After retrieval trim (final candidate pool) | TBD | TBD |

Average / max candidates per Source-1 entity: TBD.

## 4. Retrieval

See section 3 -- retrieval here is a re-ranking/trimming step (RapidFuzz
`WRatio`) rather than a separate dense-vector stage, since the blocking
stage already reduces the pool to a size where per-pair fuzzy scoring is
cheap; this avoids ever constructing a TF-IDF or embedding matrix over the
full 10M+ record dataset.

## 5. Feature engineering

`features.py` computes (see that file for exact definitions): name
features (exact/core-exact match, WRatio, ratio, token-sort/set/partial
ratio, token overlap and Jaccard, character n-gram Jaccard, prefix/suffix
match, length ratio, rare-token agreement count), the analogous address
features plus postal-code and street-number agreement, and cross-field
features (country exact/missing, combined number+postal agreement, number
of blocking views that surfaced the pair).

Top features by LightGBM gain (from your `train.py` log): TBD.

## 6. Hard negative mining

`hard_negatives.py` down-samples the negative class to
`--negative-ratio` x positives (default 5x), with 70% of negatives chosen as
the *hardest* available in the candidate pool (highest combined
name/address similarity among true negatives) and the remainder random, so
the classifier is forced to learn a sharp boundary between true matches and
merely-similar-looking distinct businesses rather than an easy
similar-vs-dissimilar split.

Class balance used for training (from your log): TBD positives / TBD
negatives (ratio TBD).

## 7. Model

LightGBM binary classifier (MIT license, well under the 8B-parameter
constraint), trained with early stopping on a grouped train/validation split
(no Source-1 entity appears in both splits) and automatic
`scale_pos_weight` for class imbalance. Default hyperparameters are in
`model.py::DEFAULT_PARAMS` (31 leaves, 0.05 learning rate, 0.85
feature/bagging fraction, min 20 samples per leaf).

## 8. Threshold selection

`threshold.py` sweeps thresholds 0.05-0.95 directly against macro F0.5 on
the validation split: first a single global threshold, then independent
per-source (`S2-*` / `S3-*`) thresholds, keeping the per-source scheme only
if it beats the global one.

Selected threshold(s) (from your log): TBD (global) / TBD (per-source, if
used).

## 9. Post-processing / consistency

`consistency.py` resolves the one situation that can hurt precision without
violating the challenge's explicit "one Source-1 may match many Source-2/3
records" rule: if the *same* Source-2/3 record is claimed by more than one
Source-1 entity (which should not happen, since a Source-2/3 record
describes exactly one real business), only the claim(s) within a small
margin of the highest-confidence claim for that target are kept. This never
forces one-to-one matching and never removes a legitimate one-to-many match.

## 10. Validation methodology

Evaluation uses a **grouped** split of the training data by Source-1 entity
id (`dataset.py::group_train_val_split`), so no Source-1 entity's pairs
appear in both the training and validation sets -- this avoids leakage that
would otherwise inflate validation macro F0.5. `metric.py` implements the
official per-entity precision/recall/F0.5 definition exactly, including the
singleton rule (empty-true + empty-pred = 1.0; empty-true + any-pred = 0.0).

**Validation results (from `evaluate.py`):**

- Macro F0.5: TBD
- Entities evaluated: TBD (of which TBD singletons, TBD predicted correctly)
- TP / FP / FN: TBD / TBD / TBD

## 11. Resource usage

- Peak RSS during training / prediction (from `diagnostics.report_memory`
  logs): TBD
- Wall-clock time for `train.py` / `predict.py` on the full dataset: TBD
- DuckDB memory limit / thread count used: TBD (flags: `--duckdb-memory-limit`,
  `--threads`)

## 12. Limitations

- Blocking recall is the hard ceiling on achievable macro F0.5; any
  ground-truth pair that survives neither blocking nor the retrieval trim
  can never be recovered by the model. (See the candidate-recall numbers in
  section 3 for how close to 100% this pipeline gets on this dataset.)
- The "common name/token" downweighting in blocking (`RARE_DF_THRESHOLD`)
  can drop legitimate candidates when a business name is both very common
  and correct -- this is a recall/scalability trade-off.
- Country is used only as a soft signal, so it does not, by itself, resolve
  ambiguity between two identically-named businesses in different
  countries; that is left to the name/address features and the model.

## 13. Reproducibility

All randomized steps are seeded via `--seed` (default 42): negative
sampling, the grouped train/val split, and LightGBM's internal seed.
`train.py` saves `pipeline_config.json` and `thresholds.json` alongside the
model so that `evaluate.py` and `predict.py` always reuse the exact
configuration that produced the saved model, rather than re-deriving it from
current CLI defaults.

## 14. Fair play statement

No external data, APIs, geocoding services, or business-registration
lookups are used anywhere in this pipeline -- all normalization, blocking,
features, and matching are derived solely from the three provided TSV
sources.
