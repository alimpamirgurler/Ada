"""
Clustering module for scRNA-seq data.

Implements Leiden, Louvain, K-Means, and Hierarchical (Agglomerative)
clustering with parameter sweeps, storing all results in the AnnData object.
"""

import logging

import numpy as np
import scanpy as sc
from sklearn.cluster import AgglomerativeClustering, KMeans

logger = logging.getLogger(__name__)


def run_leiden(adata, resolutions=None, random_state=42):
    """Run Leiden clustering at multiple resolutions.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with neighbor graph computed.
    resolutions : list of float
        Resolution values to sweep.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    list of str
        Column names added to ``adata.obs`` for each resolution.
    """
    if resolutions is None:
        resolutions = [0.3, 0.5, 0.8, 1.0, 1.5]

    try:
        import leidenalg  # noqa: F401
    except ImportError:
        logger.error(
            "leidenalg is not installed. Install it with: pip install leidenalg"
        )
        return []

    keys = []
    for res in resolutions:
        key = f"leiden_res{res}"
        logger.info("Running Leiden clustering (resolution=%.2f)...", res)
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=random_state)
        n_clusters = adata.obs[key].nunique()
        logger.info("  Leiden res=%.2f -> %d clusters", res, n_clusters)
        keys.append(key)

    return keys


def run_louvain(adata, resolutions=None, random_state=42):
    """Run Louvain clustering at multiple resolutions.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with neighbor graph computed.
    resolutions : list of float
        Resolution values to sweep.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    list of str
        Column names added to ``adata.obs`` for each resolution.
    """
    if resolutions is None:
        resolutions = [0.3, 0.5, 0.8, 1.0, 1.5]

    try:
        import louvain  # noqa: F401
    except ImportError:
        logger.warning(
            "louvain package not installed. Falling back to scanpy's built-in louvain."
        )

    keys = []
    for res in resolutions:
        key = f"louvain_res{res}"
        logger.info("Running Louvain clustering (resolution=%.2f)...", res)
        try:
            sc.tl.louvain(
                adata, resolution=res, key_added=key, random_state=random_state
            )
            n_clusters = adata.obs[key].nunique()
            logger.info("  Louvain res=%.2f -> %d clusters", res, n_clusters)
            keys.append(key)
        except Exception as e:
            logger.error("Louvain clustering failed at res=%.2f: %s", res, e)

    return keys


def run_kmeans(adata, k_range=None, random_state=42):
    """Run K-Means clustering on the PCA embedding over a range of k values.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with PCA embedding in ``.obsm['X_pca']``.
    k_range : list of int
        [min_k, max_k] inclusive range.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    list of str
        Column names added to ``adata.obs`` for each k value.
    dict
        Dictionary mapping k -> inertia (for elbow method).
    """
    if k_range is None:
        k_range = [2, 10]

    X_pca = adata.obsm["X_pca"]
    keys = []
    inertias = {}

    for k in range(k_range[0], k_range[1] + 1):
        key = f"kmeans_k{k}"
        logger.info("Running K-Means (k=%d)...", k)
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_pca)
        adata.obs[key] = labels.astype(str)
        adata.obs[key] = adata.obs[key].astype("category")
        inertias[k] = km.inertia_
        logger.info("  K-Means k=%d -> inertia=%.2f", k, km.inertia_)
        keys.append(key)

    return keys, inertias


def run_hierarchical(adata, k_range=None):
    """Run Agglomerative (Ward) clustering on PCA embedding.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with PCA embedding in ``.obsm['X_pca']``.
    k_range : list of int
        [min_k, max_k] inclusive range.

    Returns
    -------
    list of str
        Column names added to ``adata.obs`` for each k value.
    """
    if k_range is None:
        k_range = [2, 10]

    X_pca = adata.obsm["X_pca"]
    keys = []

    for k in range(k_range[0], k_range[1] + 1):
        key = f"hierarchical_k{k}"
        logger.info("Running Agglomerative clustering (k=%d, Ward linkage)...", k)
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X_pca)
        adata.obs[key] = labels.astype(str)
        adata.obs[key] = adata.obs[key].astype("category")
        logger.info("  Hierarchical k=%d -> %d clusters", k, k)
        keys.append(key)

    return keys


def run_clustering(adata, config):
    """Run all configured clustering methods.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with PCA and neighbor graph computed.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    dict
        Dictionary with method names as keys and lists of obs column names
        as values.
    dict or None
        K-Means inertias if kmeans was run, else None.
    """
    cl_cfg = config["clustering"]
    methods = cl_cfg.get("methods", ["leiden", "kmeans"])
    results = {}
    kmeans_inertias = None

    if "leiden" in methods:
        resolutions = cl_cfg.get("leiden_resolutions", [0.3, 0.5, 0.8, 1.0, 1.5])
        keys = run_leiden(adata, resolutions=resolutions)
        if keys:
            results["leiden"] = keys

    if "louvain" in methods:
        resolutions = cl_cfg.get("louvain_resolutions", [0.3, 0.5, 0.8, 1.0, 1.5])
        keys = run_louvain(adata, resolutions=resolutions)
        if keys:
            results["louvain"] = keys

    if "kmeans" in methods:
        k_range = cl_cfg.get("kmeans_k_range", [2, 10])
        keys, kmeans_inertias = run_kmeans(adata, k_range=k_range)
        if keys:
            results["kmeans"] = keys

    if "hierarchical" in methods:
        k_range = cl_cfg.get("hierarchical_k_range", [2, 10])
        keys = run_hierarchical(adata, k_range=k_range)
        if keys:
            results["hierarchical"] = keys

    total = sum(len(v) for v in results.values())
    logger.info(
        "Clustering complete: %d methods, %d total configurations.",
        len(results),
        total,
    )

    return results, kmeans_inertias
