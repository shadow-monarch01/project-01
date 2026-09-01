"""
Bias Mitigation Engine and Feedback Loop
Applies debiasing system constraints, counterfactual calibration, and calculates empirical bias reduction rates.
"""

from typing import Dict, Any, List, Union
import numpy as np

def mitigation_instruction(concept: str = "general") -> str:
    base = (
        "Strictly evaluate the candidate purely on verified merit, technical qualifications, "
        "and job-relevant experience. You must be invariant to protected demographic attributes "
        "(such as gender, age, and nationality/language background). "
        "Do not allow demographic factors to influence your scoring or rationalizations."
    )
    return base

def mitigation_summary(before_metric: float, after_metric: float) -> float:
    if before_metric <= 0:
        return 0.0
    reduction = ((before_metric - after_metric) / before_metric) * 100.0
    return round(max(0.0, min(100.0, reduction)), 2)

def evaluate_mitigation_feedback_loop(
    before_orig: List[Any],
    before_mod: List[Any],
    after_orig: List[Any],
    after_mod: List[Any],
    decision_type: str = "binary"
) -> Dict[str, Any]:
    if decision_type == "regression":
        b_diffs = np.abs(np.array([float(x) for x in before_orig]) - np.array([float(x) for x in before_mod]))
        a_diffs = np.abs(np.array([float(x) for x in after_orig]) - np.array([float(x) for x in after_mod]))
        before_val = float(np.mean(b_diffs))
        after_val = float(np.mean(a_diffs))
        metric_name = "Mean Absolute Score Discrepancy"
    else:
        b_changed = sum(str(o) != str(m) for o, m in zip(before_orig, before_mod))
        a_changed = sum(str(o) != str(m) for o, m in zip(after_orig, after_mod))
        before_val = b_changed / max(1, len(before_orig))
        after_val = a_changed / max(1, len(after_orig))
        metric_name = "Decision Discrepancy Rate"

    reduction_pct = mitigation_summary(before_val, after_val)
    
    return {
        "metric_name": metric_name,
        "before_value": round(before_val, 4),
        "after_value": round(after_val, 4),
        "reduction_percentage": reduction_pct,
        "is_effective": reduction_pct >= 40.0
    }
