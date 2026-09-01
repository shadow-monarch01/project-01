"""
Counterfactual Perturbation Module
Creates controlled paired candidate inputs isolating sensitive demographic attributes across any user dataset.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

CONCEPT_COLUMN_MAP = {
    "gender": ["gender", "sex"],
    "language": ["language", "english", "language proficiency", "english proficiency", "accent"],
    "age": ["age_group", "age", "generation", "age_bracket"],
    "education": ["education", "degree", "university_tier", "education_level"],
    "location": ["location", "city", "region", "country", "state"]
}

DEFAULT_PAIRS = {
    "gender": ("Male", "Female"),
    "language": ("Fluent", "Basic"),
    "age": ("25-34", "35-44"),
    "education": ("B.S. Computer Science", "Ph.D. Computer Science"),
    "location": ("Metropolitan", "Rural")
}

def get_all_dataset_concepts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Intelligently discovers all candidate demographic / categorical columns in any user dataset.
    """
    concepts = []
    seen_cols = set()

    # 1. Match known standard concepts first if present
    df_cols_lower = {col.lower(): col for col in df.columns}
    for concept_id, candidate_names in CONCEPT_COLUMN_MAP.items():
        for cand in candidate_names:
            if cand in df_cols_lower:
                actual_col = df_cols_lower[cand]
                vals = get_available_values(df, actual_col)
                if len(vals) >= 2 and actual_col not in seen_cols:
                    concepts.append({
                        "id": concept_id,
                        "display_name": f"{actual_col.replace('_', ' ').title()} ({len(vals)} groups)",
                        "column": actual_col,
                        "values": vals[:15]
                    })
                    seen_cols.add(actual_col)
                break

    # 2. Add any other categorical/string columns in user's dataset with 2 to 20 unique values
    for col in df.columns:
        if col in seen_cols:
            continue
        # Skip purely numeric continuous columns, IDs, skills, and excluded demographics (ethnicity/race/religion)
        c_lower = col.lower()
        if c_lower in ["candidate_id", "id", "name", "candidate_name", "interview_score", "score", "experience_years", "salary", "cluster", "ethnicity", "race", "ethnic_group", "religion", "faith", "creed", "belief", "technical_skills", "skills"]:
            continue
        
        vals = get_available_values(df, col)
        if 2 <= len(vals) <= 25:
            concepts.append({
                "id": col.lower().replace(" ", "_"),
                "display_name": f"{col.replace('_', ' ').title()} (Custom: {len(vals)} groups)",
                "column": col,
                "values": vals
            })
            seen_cols.add(col)

    # Fallback to gender if empty
    if not concepts:
        concepts.append({
            "id": "gender",
            "display_name": "Gender (Male vs Female)",
            "column": "gender" if "gender" in df.columns else df.columns[0],
            "values": ["Male", "Female"]
        })

    return concepts

def resolve_column_for_concept(df: pd.DataFrame, concept: str) -> Optional[str]:
    c_lower = concept.strip().lower()
    df_cols_lower = {col.lower(): col for col in df.columns}
    
    # Direct column match
    if c_lower in df_cols_lower:
        return df_cols_lower[c_lower]

    # Map match
    candidate_cols = CONCEPT_COLUMN_MAP.get(c_lower, [c_lower])
    for cand in candidate_cols:
        if cand in df_cols_lower:
            return df_cols_lower[cand]

    # Fuzzy match
    for col in df.columns:
        if c_lower in col.lower() or col.lower() in c_lower:
            return col

    return None

def get_available_values(df: pd.DataFrame, column: str) -> List[str]:
    if column not in df.columns:
        return []
    vals = df[column].dropna().astype(str).unique().tolist()
    clean = [v.strip() for v in vals if v.strip() and v.strip().lower() != "nan"]
    return sorted(list(set(clean)))

def default_pair(df: pd.DataFrame, concept: str) -> Optional[Tuple[str, str, str]]:
    column = resolve_column_for_concept(df, concept)
    if not column:
        return None
    
    vals = get_available_values(df, column)
    if len(vals) >= 2:
        return column, vals[0], vals[1]
    
    c_lower = concept.strip().lower()
    if c_lower in DEFAULT_PAIRS:
        val_a, val_b = DEFAULT_PAIRS[c_lower]
        return column, val_a, val_b
    return None

def make_variation(row: pd.Series, concept: str, value: str, column: Optional[str] = None) -> Tuple[Optional[pd.Series], Optional[str]]:
    new_row = row.copy()
    col = column or resolve_column_for_concept(pd.DataFrame([row]), concept)
    if not col or col not in new_row.index:
        return None, f"Target column for concept '{concept}' not found in candidate record."
    new_row[col] = value
    return new_row, None

def generate_counterfactual_dataset(
    df: pd.DataFrame,
    concept: str,
    value_a: str,
    value_b: str,
    column: Optional[str] = None
) -> Tuple[List[pd.Series], List[pd.Series]]:
    col = column or resolve_column_for_concept(df, concept)
    if not col or col not in df.columns:
        raise ValueError(f"Could not resolve valid dataset column for concept '{concept}' in dataset.")

    rows_a, rows_b = [], []
    for _, row in df.iterrows():
        a, err_a = make_variation(row, concept, value_a, col)
        b, err_b = make_variation(row, concept, value_b, col)
        if a is not None and b is not None:
            rows_a.append(a)
            rows_b.append(b)

    return rows_a, rows_b
