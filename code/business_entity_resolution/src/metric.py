"""Official Macro F0.5 metric implementation for Business Entity Resolution."""
from __future__ import annotations

from typing import Dict, List

BETA = 0.5
BETA_SQ = BETA * BETA


def f_beta(precision: float, recall: float, beta_sq: float = BETA_SQ) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    denom = beta_sq * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def per_entity_score(true_set: set, pred_set: set) -> Dict[str, float]:
    if not true_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f05": 1.0, "tp": 0, "fp": 0, "fn": 0, "is_singleton": True}
    if not true_set and pred_set:
        return {"precision": 0.0, "recall": 0.0, "f05": 0.0, "tp": 0, "fp": len(pred_set), "fn": 0, "is_singleton": True}
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "precision": precision, "recall": recall, "f05": f_beta(precision, recall),
        "tp": tp, "fp": fp, "fn": fn, "is_singleton": False,
    }


def macro_f05_from_predictions(
    true_matches: Dict[str, set], pred_matches: Dict[str, set], all_s1_ids: List[str]
) -> Dict:
    """Macro-averaged F0.5 across every Source1 id in `all_s1_ids` (an id
    missing from pred_matches is treated as an empty prediction)."""
    scores = []
    n_singletons = 0
    n_singletons_correct = 0
    total_tp = total_fp = total_fn = 0
    per_entity: Dict[str, Dict] = {}

    for s1_id in all_s1_ids:
        true_set = true_matches.get(s1_id, set())
        pred_set = pred_matches.get(s1_id, set())
        result = per_entity_score(true_set, pred_set)
        scores.append(result["f05"])
        per_entity[s1_id] = result
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        if result["is_singleton"]:
            n_singletons += 1
            if result["f05"] == 1.0:
                n_singletons_correct += 1

    macro = sum(scores) / len(scores) if scores else 0.0
    return {
        "macro_f05": macro,
        "n_entities": len(all_s1_ids),
        "n_singletons": n_singletons,
        "n_singletons_correct": n_singletons_correct,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_entity": per_entity,
    }
