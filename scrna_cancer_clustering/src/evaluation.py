"""
Cluster evaluation module for scRNA-seq data.

Computes silhouette score, Davies-Bouldin index, and Calinski-Harabasz index
for all clustering results. Generates a summary table and recommends the best
clustering configuration based on a composite score.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


def evaluate_clustering(adata, cluster_key, embedding_key="X_pca"):
    """Evaluate a single clustering result.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with clustering labels and embeddings.
    cluster_key : str
        Column name in ``adata.obs`` containing cluster labels.
    embedding_key : str
        Key in ``adata.obsm`` for the embedding to evaluate on.

    Returns
    -------
    dict
        Dictionary with metric names as keys and scores as values.
        Returns None if clustering has fewer than 2 clusters.
    """
    labels = adata.obs[cluster_key].astype(int).values if adata.obs[cluster_key].dtype.name == "category" else adata.obs[cluster_key].astype(int).values
    X = adata.obsm[embedding_key]

    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        logger.warning(
            "Skipping evaluation for '%s': only %d cluster(s) found.",
            cluster_key,
            n_clusters,
        )
        return None

    try:
        sil = silhouette_score(X, labels, random_state=42)
    except Exception as e:
        logger.warning("Silhouette score failed for '%s': %s", cluster_key, e)
        sil = np.nan

    try:
        dbi = davies_bouldin_score(X, labels)
    except Exception as e:
        logger.warning("Davies-Bouldin index failed for '%s': %s", cluster_key, e)
        dbi = np.nan

    try:
        chi = calinski_harabasz_score(X, labels)
    except Exception as e:
        logger.warning("Calinski-Harabasz index failed for '%s': %s", cluster_key, e)
        chi = np.nan

    return {
        "cluster_key": cluster_key,
        "n_clusters": n_clusters,
        "silhouette_score": sil,
        "davies_bouldin_index": dbi,
        "calinski_harabasz_index": chi,
    }


def evaluate_all_clusterings(adata, clustering_results):
    """Evaluate all clustering configurations and build a summary table.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with clustering labels and PCA embedding.
    clustering_results : dict
        Dictionary mapping method names to lists of obs column names.

    Returns
    -------
    pandas.DataFrame
        Summary table with evaluation metrics for every configuration.
    """
    logger.info("Evaluating all clustering configurations...")
    records = []

    for method, keys in clustering_results.items():
        for key in keys:
            result = evaluate_clustering(adata, key)
            if result is not None:
                result["method"] = method
                records.append(result)

    if not records:
        logger.warning("No clustering configurations could be evaluated.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Reorder columns
    cols = [
        "method",
        "cluster_key",
        "n_clusters",
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_index",
    ]
    df = df[cols]
    logger.info("Evaluation complete for %d configurations.", len(df))
    return df


def compute_composite_score(df):
    """Compute a composite score to rank clustering configurations.

    The composite score is the mean of three normalized (0-1) metrics:
    - Silhouette score (higher is better) -> normalized directly
    - Davies-Bouldin index (lower is better) -> inverted
    - Calinski-Harabasz index (higher is better) -> normalized directly

    Parameters
    ----------
    df : pandas.DataFrame
        Evaluation summary table.

    Returns
    -------
    pandas.DataFrame
        Input table with an added ``composite_score`` column, sorted
        descending by composite score.
    """
    if df.empty:
        return df

    df = df.copy()

    def _normalize(series, invert=False):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series(0.5, index=series.index)
        normed = (series - s_min) / (s_max - s_min)
        if invert:
            normed = 1.0 - normed
        return normed

    df["norm_silhouette"] = _normalize(df["silhouette_score"])
    df["norm_dbi"] = _normalize(df["davies_bouldin_index"], invert=True)
    df["norm_chi"] = _normalize(df["calinski_harabasz_index"])

    df["composite_score"] = (
        df["norm_silhouette"] + df["norm_dbi"] + df["norm_chi"]
    ) / 3.0

    df = df.drop(columns=["norm_silhouette", "norm_dbi", "norm_chi"])
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    return df


def recommend_best(df):
    """Return the best clustering configuration.

    Parameters
    ----------
    df : pandas.DataFrame
        Evaluation table with composite scores (sorted).

    Returns
    -------
    pandas.Series or None
        The row for the best configuration, or None if table is empty.
    """
    if df.empty:
        logger.warning("Cannot recommend: evaluation table is empty.")
        return None

    best = df.iloc[0]
    logger.info(
        "Recommended clustering: %s (n_clusters=%d, composite_score=%.4f)",
        best["cluster_key"],
        best["n_clusters"],
        best["composite_score"],
    )
    return best


def run_evaluation(adata, clustering_results, results_dir):
    """Run full evaluation pipeline: compute metrics, rank, save.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with clustering results and PCA embedding.
    clustering_results : dict
        Dictionary mapping method names to obs column name lists.
    results_dir : str
        Directory to save the evaluation CSV.

    Returns
    -------
    pandas.DataFrame
        Ranked evaluation table.
    pandas.Series or None
        Best configuration row.
    """
    import os

    os.makedirs(results_dir, exist_ok=True)

    df = evaluate_all_clusterings(adata, clustering_results)
    df = compute_composite_score(df)

    csv_path = os.path.join(results_dir, "clustering_evaluation.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Evaluation summary saved to %s", csv_path)

    best = recommend_best(df)
    return df, best
