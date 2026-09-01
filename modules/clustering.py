"""
Input Text Clustering Module
Applies TF-IDF vectorization and K-Means clustering to partition candidate profiles into coherent semantic subgroups.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Optional

def build_profile_text(row: pd.Series, text_columns: List[str]) -> str:
    parts = []
    for col in text_columns:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(f"{col}: {val}")
    return " | ".join(parts)

def cluster_dataframe(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    n_clusters: int = 3
) -> pd.DataFrame:
    data = df.copy()
    if text_columns is None:
        exclude_cols = ["candidate_id", "decision", "cluster", "name"]
        text_columns = [c for c in data.columns if c not in exclude_cols]

    texts = data.apply(lambda r: build_profile_text(r, text_columns), axis=1)
    
    k = max(1, min(n_clusters, len(data)))
    if len(data) < 2 or k == 1:
        data["cluster"] = 0
        return data

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    X = vectorizer.fit_transform(texts)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    data["cluster"] = kmeans.fit_predict(X)
    return data
