"""
Statistical Significance and Hypothesis Testing Engine
Provides paired hypothesis tests and effect-size calculations for Binary, Multiclass, and Regression outcomes.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Union

def mcnemar_test(orig: List[str], mod: List[str]) -> Dict[str, Any]:
    if len(orig) != len(mod) or len(orig) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "method": "McNemar (Empty)", "b": 0, "c": 0}

    classes = list(set([str(x).upper() for x in orig + mod]))
    if len(classes) < 2:
        classes = ["SELECT", "REJECT"]
        
    pos_class = classes[0]
    
    b = sum(str(o).upper() == pos_class and str(m).upper() != pos_class for o, m in zip(orig, mod))
    c = sum(str(o).upper() != pos_class and str(m).upper() == pos_class for o, m in zip(orig, mod))
    a = sum(str(o).upper() == pos_class and str(m).upper() == pos_class for o, m in zip(orig, mod))
    d = sum(str(o).upper() != pos_class and str(m).upper() != pos_class for o, m in zip(orig, mod))

    total_discordant = b + c
    
    if total_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "McNemar (Exact)",
            "contingency_table": {"a": a, "b": b, "c": c, "d": d},
            "discordant_count": 0,
            "disagreement_rate": 0.0,
            "significant": False,
            "is_significant": False,
            "alpha": 0.05
        }

    if total_discordant < 25:
        p_val = stats.binomtest(b, total_discordant, p=0.5, alternative="two-sided").pvalue
        stat = float(b)
        method = "Exact Binomial McNemar Test"
    else:
        stat = ((abs(b - c) - 1.0) ** 2) / float(total_discordant)
        p_val = 1.0 - stats.chi2.cdf(stat, df=1)
        method = "McNemar Chi-Square (Continuity Corrected)"

    is_sig = bool(p_val < 0.05)
    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 5),
        "method": method,
        "contingency_table": {"a": a, "b": b, "c": c, "d": d},
        "discordant_count": total_discordant,
        "disagreement_rate": round(total_discordant / max(1, len(orig)), 4),
        "significant": is_sig,
        "is_significant": is_sig,
        "alpha": 0.05
    }

def multiclass_chi_square(orig: List[str], mod: List[str]) -> Dict[str, Any]:
    if len(orig) != len(mod) or len(orig) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "method": "Chi-Square (Empty)", "significant": False, "is_significant": False, "alpha": 0.05}

    categories = sorted(list(set([str(x) for x in orig + mod])))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    
    matrix = np.zeros((k, k), dtype=int)
    for o, m in zip(orig, mod):
        matrix[cat_to_idx[str(o)], cat_to_idx[str(m)]] += 1

    changed_count = sum(str(o) != str(m) for o, m in zip(orig, mod))
    chi2_stat, p_val, dof, _ = stats.chi2_contingency(matrix + 1e-5)
    is_sig = bool(p_val < 0.05)

    return {
        "statistic": round(float(chi2_stat), 4),
        "p_value": round(float(p_val), 5),
        "dof": int(dof),
        "method": "Multiclass Marginal Chi-Square Test",
        "changed_count": changed_count,
        "switch_rate": round(changed_count / max(1, len(orig)), 4),
        "categories": categories,
        "transition_matrix": matrix.tolist(),
        "significant": is_sig,
        "is_significant": is_sig,
        "alpha": 0.05
    }

def paired_regression_test(orig: List[Union[float, int]], mod: List[Union[float, int]]) -> Dict[str, Any]:
    try:
        a = np.array([float(x) for x in orig])
        b = np.array([float(x) for x in mod])
    except (ValueError, TypeError):
        return {"statistic": 0.0, "p_value": 1.0, "mean_difference": 0.0, "cohens_d": 0.0, "significant": False, "is_significant": False, "alpha": 0.05}

    diffs = a - b
    n = len(diffs)
    
    if n == 0 or np.all(diffs == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_difference": 0.0,
            "mean_abs_difference": 0.0,
            "std_difference": 0.0,
            "cohens_d": 0.0,
            "ci_95": [0.0, 0.0],
            "method": "Paired t-test (No variance)",
            "significant": False,
            "is_significant": False,
            "alpha": 0.05
        }

    mean_diff = float(np.mean(diffs))
    mean_abs_diff = float(np.mean(np.abs(diffs)))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    
    if std_diff > 0:
        t_stat, p_val = stats.ttest_rel(a, b)
        cohens_d = mean_diff / std_diff
        margin = 1.96 * (std_diff / np.sqrt(n))
        ci_lower = round(mean_diff - margin, 2)
        ci_upper = round(mean_diff + margin, 2)
    else:
        t_stat, p_val, cohens_d = 0.0, 1.0, 0.0
        ci_lower, ci_upper = mean_diff, mean_diff

    is_sig = bool(p_val < 0.05)
    return {
        "statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 5),
        "mean_difference": round(mean_diff, 2),
        "mean_abs_difference": round(mean_abs_diff, 2),
        "std_difference": round(std_diff, 2),
        "cohens_d": round(float(cohens_d), 3),
        "ci_95": [ci_lower, ci_upper],
        "method": "Paired Student's t-test",
        "sample_size": n,
        "significant": is_sig,
        "is_significant": is_sig,
        "alpha": 0.05
    }
