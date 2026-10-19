"""Convenience entry point.

    python src/main.py --data-root "PATH_TO_DATASET" --mode train
    python src/main.py --data-root "PATH_TO_DATASET" --mode evaluate
    python src/main.py --data-root "PATH_TO_DATASET" --mode predict

Any arguments after --mode are forwarded verbatim to the corresponding
module (train.py / evaluate.py / predict.py), which also work as standalone
scripts.
"""
from __future__ import annotations

import argparse
import sys

from . import evaluate as evaluate_module
from . import predict as predict_module
from . import train as train_module

MODES = {"train": train_module, "evaluate": evaluate_module, "predict": predict_module}


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline entry point.", add_help=False)
    parser.add_argument("--mode", required=True, choices=list(MODES.keys()))
    known_args, remaining = parser.parse_known_args(argv)
    module = MODES[known_args.mode]
    module.main(remaining)


if __name__ == "__main__":
    main()
