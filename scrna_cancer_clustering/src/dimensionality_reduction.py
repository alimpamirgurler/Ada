"""
Dimensionality reduction module for scRNA-seq data.

Provides PCA, UMAP, and t-SNE via scanpy, with configurable parameters.
"""

import logging

import scanpy as sc

logger = logging.getLogger(__name__)


def run_pca(adata, n_comps=50, random_state=42):
    """Run PCA on the preprocessed data.

    Parameters
    ----------
    adata : anndata.AnnData
        Preprocessed and scaled data.
    n_comps : int
        Number of principal components to compute.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    anndata.AnnData
        AnnData with PCA results stored in ``.obsm['X_pca']``.
    """
    logger.info("Running PCA with %d components...", n_comps)
    sc.tl.pca(adata, n_comps=n_comps, random_state=random_state)
    variance_explained = adata.uns["pca"]["variance_ratio"].sum() * 100
    logger.info(
        "PCA complete. Total variance explained by %d PCs: %.1f%%",
        n_comps,
        variance_explained,
    )
    return adata


def compute_neighbors(adata, n_pcs=30, n_neighbors=15, random_state=42):
    """Compute the k-nearest neighbor graph on the PCA embedding.

    Required before running UMAP, t-SNE, Leiden, or Louvain.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with PCA computed.
    n_pcs : int
        Number of PCs to use for the neighbor graph.
    n_neighbors : int
        Number of neighbors for the kNN graph.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    anndata.AnnData
        AnnData with neighbor graph stored.
    """
    logger.info(
        "Computing neighbor graph (n_pcs=%d, n_neighbors=%d)...", n_pcs, n_neighbors
    )
    sc.pp.neighbors(
        adata, n_pcs=n_pcs, n_neighbors=n_neighbors, random_state=random_state
    )
    logger.info("Neighbor graph computed.")
    return adata


def run_umap(adata, min_dist=0.1, random_state=42):
    """Run UMAP on the neighbor graph.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with neighbor graph computed.
    min_dist : float
        Minimum distance parameter for UMAP.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    anndata.AnnData
        AnnData with UMAP embedding in ``.obsm['X_umap']``.
    """
    logger.info("Running UMAP (min_dist=%.2f)...", min_dist)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    logger.info("UMAP complete.")
    return adata


def run_tsne(adata, n_pcs=30, random_state=42):
    """Run t-SNE on the PCA embedding.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with PCA computed.
    n_pcs : int
        Number of PCs to use.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    anndata.AnnData
        AnnData with t-SNE embedding in ``.obsm['X_tsne']``.
    """
    logger.info("Running t-SNE (n_pcs=%d)...", n_pcs)
    sc.tl.tsne(adata, n_pcs=n_pcs, random_state=random_state)
    logger.info("t-SNE complete.")
    return adata


def run_dimensionality_reduction(adata, config):
    """Run the full dimensionality reduction pipeline.

    Parameters
    ----------
    adata : anndata.AnnData
        Preprocessed data.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    anndata.AnnData
        AnnData with PCA, neighbors, UMAP, and t-SNE computed.
    """
    dr_cfg = config["dimensionality_reduction"]
    n_pcs = dr_cfg.get("n_pcs", 50)
    n_pcs_use = dr_cfg.get("n_pcs_use", 30)
    n_neighbors = dr_cfg.get("umap_n_neighbors", 15)
    min_dist = dr_cfg.get("umap_min_dist", 0.1)

    adata = run_pca(adata, n_comps=n_pcs)
    adata = compute_neighbors(adata, n_pcs=n_pcs_use, n_neighbors=n_neighbors)
    adata = run_umap(adata, min_dist=min_dist)
    adata = run_tsne(adata, n_pcs=n_pcs_use)

    return adata
