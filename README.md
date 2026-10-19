# Amazon_ML_Hackthon2 -- Business Entity Resolution (independent implementation)

This is a **second, fully independent** implementation for the Amazon ML
Challenge 2026 Business Entity Resolution track. It does not import,
modify, or otherwise depend on any other project -- it only reads the
official dataset TSVs from a path you provide.

Full details (setup, every command, architecture, features, memory
considerations, reproducibility, packaging) live in
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).
The filled-in methodology write-up is
[`Documentation_template.md`](Documentation_template.md).

## Quick start

```
cd code/business_entity_resolution
pip install -r requirements.txt

python src/train.py    --data-root "PATH_TO_DATASET"
python src/evaluate.py --data-root "PATH_TO_DATASET"
python src/predict.py  --data-root "PATH_TO_DATASET"
```

`PATH_TO_DATASET` is the folder containing `train/` and `test/` (see the
dataset layout in the pipeline README) -- it is read in place and never
copied into this project.

## Layout

```
Amazon_ML_Hackthon2/
  output/                                 # matching_results.tsv, candidate_pairs.tsv (written by predict.py)
  artifacts/                              # trained model + thresholds (written by train.py)
  code/business_entity_resolution/
    README.md                             # full instructions
    requirements.txt                      # pinned dependencies
    src/                                  # all pipeline source code
  Documentation_template.md               # methodology write-up
```
