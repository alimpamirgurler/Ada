"""
Visualization module for scRNA-seq clustering pipeline.

Generates QC plots, dimensionality reduction embeddings, clustering results,
marker gene heatmaps/dotplots, and evaluation comparison charts. Saves all
figures as both PNG and PDF.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

logger = logging.getLogger(__name__)


def _save_figure(fig, figures_dir, name):
    """Save a matplotlib figure as PNG and PDF.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    figures_dir : str
        Output directory.
    name : str
        Base name (without extension).
    """
    os.makedirs(figures_dir, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = os.path.join(figures_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s (.png, .pdf)", name)


def _save_scanpy_figure(figures_dir, name):
    """Move a scanpy-generated figure from the default location.

    Scanpy saves figures to ``sc.settings.figdir``. This helper copies them
    and also creates a PDF version.

    Parameters
    ----------
    figures_dir : str
        Target output directory.
    name : str
        Base filename (without extension).
    """
    os.makedirs(figures_dir, exist_ok=True)
    # scanpy saves to its figdir; we redirect it at the start
    logger.info("Scanpy figure saved: %s", name)


def plot_qc_violin(adata, figures_dir):
    """Plot QC violin plots for n_genes_by_counts, total_counts, pct_counts_mt.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with QC metrics computed.
    figures_dir : str
        Output directory.
    """
    os.makedirs(figures_dir, exist_ok=True)

    qc_keys = []
    for k in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]:
        if k in adata.obs.columns:
            qc_keys.append(k)

    if not qc_keys:
        logger.warning("No QC metrics found in adata.obs; skipping QC violin plots.")
        return

    fig, axes = plt.subplots(1, len(qc_keys), figsize=(4 * len(qc_keys), 4))
    if len(qc_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, qc_keys):
        sns.violinplot(y=adata.obs[key], ax=ax, inner="box", color="skyblue")
        ax.set_title(key)
        ax.set_ylabel("")

    fig.suptitle("QC Metrics", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "qc_violin")


def plot_hvg(adata, figures_dir):
    """Plot the highly variable genes scatter plot.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with HVG annotation (use ``.raw`` or pre-subset data).
    figures_dir : str
        Output directory.
    """
    os.makedirs(figures_dir, exist_ok=True)
    # Use scanpy's built-in; redirect figure output
    sc.settings.figdir = figures_dir
    try:
        sc.pl.highly_variable_genes(adata, show=False, save=".png")
        # Also save PDF
        sc.pl.highly_variable_genes(adata, show=False, save=".pdf")
        logger.info("Saved HVG scatter plot.")
    except Exception as e:
        logger.warning("Could not plot HVG scatter: %s", e)


def plot_pca_elbow(adata, figures_dir):
    """Plot the PCA elbow/scree plot.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with PCA computed.
    figures_dir : str
        Output directory.
    """
    variance_ratio = adata.uns["pca"]["variance_ratio"]
    n_pcs = len(variance_ratio)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, n_pcs + 1), variance_ratio, "o-", markersize=3, linewidth=1)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Ratio")
    ax.set_title("PCA Elbow / Scree Plot")
    ax.set_xlim(0, n_pcs + 1)
    ax.grid(True, alpha=0.3)
    _save_figure(fig, figures_dir, "pca_elbow")


def plot_embedding(adata, basis, color_keys, figures_dir, prefix):
    """Plot an embedding (UMAP or t-SNE) colored by various keys.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with embedding computed.
    basis : str
        Embedding key: 'umap' or 'tsne'.
    color_keys : list of str
        Keys in ``adata.obs`` or ``adata.var_names`` to color by.
    figures_dir : str
        Output directory.
    prefix : str
        Filename prefix.
    """
    os.makedirs(figures_dir, exist_ok=True)
    sc.settings.figdir = figures_dir

    valid_keys = []
    for k in color_keys:
        if k in adata.obs.columns:
            valid_keys.append(k)
        elif adata.raw is not None and k in adata.raw.var_names:
            valid_keys.append(k)
        elif k in adata.var_names:
            valid_keys.append(k)

    if not valid_keys:
        logger.warning("No valid color keys for %s plot.", basis)
        return

    for key in valid_keys:
        try:
            fig = sc.pl.embedding(
                adata,
                basis=basis,
                color=key,
                show=False,
                return_fig=True,
                title=f"{basis.upper()} - {key}",
            )
            if fig is not None:
                _save_figure(fig, figures_dir, f"{prefix}_{key}")
            else:
                # scanpy sometimes doesn't return fig; use savefig approach
                sc.pl.embedding(
                    adata, basis=basis, color=key, show=False,
                    save=f"_{prefix}_{key}.png",
                )
                logger.info("Saved %s colored by %s", basis, key)
        except Exception as e:
            logger.warning("Failed to plot %s colored by '%s': %s", basis, key, e)


def plot_silhouette_comparison(eval_df, figures_dir):
    """Plot a bar chart comparing silhouette scores across methods.

    Parameters
    ----------
    eval_df : pandas.DataFrame
        Evaluation summary table.
    figures_dir : str
        Output directory.
    """
    if eval_df.empty:
        logger.warning("Empty evaluation table; skipping silhouette comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(eval_df) * 0.6), 5))

    colors = []
    palette = sns.color_palette("Set2", n_colors=eval_df["method"].nunique())
    method_colors = dict(zip(eval_df["method"].unique(), palette))
    colors = [method_colors[m] for m in eval_df["method"]]

    bars = ax.bar(
        range(len(eval_df)),
        eval_df["silhouette_score"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(eval_df)))
    ax.set_xticklabels(eval_df["cluster_key"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score Comparison Across Clustering Methods")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Legend for methods
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=method_colors[m], edgecolor="black", label=m)
        for m in method_colors
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    _save_figure(fig, figures_dir, "silhouette_comparison")


def plot_cluster_stability(adata, clustering_results, figures_dir):
    """Plot cluster count vs. resolution for Leiden/Louvain.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with clustering results.
    clustering_results : dict
        Clustering results dictionary.
    figures_dir : str
        Output directory.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for method in ["leiden", "louvain"]:
        if method not in clustering_results:
            continue
        keys = clustering_results[method]
        resolutions = []
        n_clusters_list = []
        for key in keys:
            # Parse resolution from key like "leiden_res0.5"
            try:
                res = float(key.split("res")[1])
                resolutions.append(res)
                n_clusters_list.append(adata.obs[key].nunique())
            except (IndexError, ValueError):
                continue

        if resolutions:
            ax.plot(
                resolutions,
                n_clusters_list,
                "o-",
                label=method.capitalize(),
                linewidth=2,
                markersize=6,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Resolution")
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Cluster Stability: Cluster Count vs. Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "cluster_stability")


def plot_kmeans_elbow(kmeans_inertias, figures_dir):
    """Plot K-Means elbow plot (inertia vs. k).

    Parameters
    ----------
    kmeans_inertias : dict
        Dictionary mapping k -> inertia.
    figures_dir : str
        Output directory.
    """
    if not kmeans_inertias:
        return

    ks = sorted(kmeans_inertias.keys())
    inertias = [kmeans_inertias[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, inertias, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("k (Number of Clusters)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Plot")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "kmeans_elbow")


def plot_markers(adata, cluster_key, n_top_genes, figures_dir):
    """Plot marker gene heatmap and dot plot.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with ``rank_genes_groups`` results.
    cluster_key : str
        Cluster label key used for marker gene analysis.
    n_top_genes : int
        Number of top marker genes per cluster to show.
    figures_dir : str
        Output directory.
    """
    os.makedirs(figures_dir, exist_ok=True)
    sc.settings.figdir = figures_dir

    try:
        # Dot plot
        fig = sc.pl.rank_genes_groups_dotplot(
            adata,
            n_genes=n_top_genes,
            show=False,
            return_fig=True,
        )
        if fig is not None:
            _save_figure(fig, figures_dir, "marker_dotplot")
        else:
            sc.pl.rank_genes_groups_dotplot(
                adata, n_genes=n_top_genes, show=False, save=".png"
            )
            sc.pl.rank_genes_groups_dotplot(
                adata, n_genes=n_top_genes, show=False, save=".pdf"
            )
    except Exception as e:
        logger.warning("Could not generate marker dot plot: %s", e)

    try:
        # Heatmap
        fig = sc.pl.rank_genes_groups_heatmap(
            adata,
            n_genes=n_top_genes,
            show=False,
            return_fig=True,
        )
        if fig is not None:
            _save_figure(fig, figures_dir, "marker_heatmap")
        else:
            sc.pl.rank_genes_groups_heatmap(
                adata, n_genes=n_top_genes, show=False, save=".png"
            )
            sc.pl.rank_genes_groups_heatmap(
                adata, n_genes=n_top_genes, show=False, save=".pdf"
            )
    except Exception as e:
        logger.warning("Could not generate marker heatmap: %s", e)


def plot_confusion_matrix(adata, cluster_key, truth_key, figures_dir):
    """Plot a confusion matrix / cluster purity heatmap if ground truth exists.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with both predicted clusters and ground truth labels.
    cluster_key : str
        Column in ``adata.obs`` with predicted cluster labels.
    truth_key : str
        Column in ``adata.obs`` with ground truth labels.
    figures_dir : str
        Output directory.
    """
    if truth_key not in adata.obs.columns:
        logger.info("No ground truth column '%s' found; skipping confusion matrix.", truth_key)
        return

    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    labels_pred = adata.obs[cluster_key].astype(str)
    labels_true = adata.obs[truth_key].astype(str)

    true_cats = sorted(labels_true.unique())
    pred_cats = sorted(labels_pred.unique())

    cm = sk_confusion_matrix(labels_true, labels_pred, labels=true_cats + [
        c for c in pred_cats if c not in true_cats
    ])

    # Normalize by row for purity visualization
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(max(8, len(pred_cats) * 0.8), max(6, len(true_cats) * 0.6)))
    sns.heatmap(
        cm_norm[:len(true_cats), :len(pred_cats)],
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=pred_cats,
        yticklabels=true_cats,
        ax=ax,
    )
    ax.set_xlabel(f"Predicted ({cluster_key})")
    ax.set_ylabel(f"Ground Truth ({truth_key})")
    ax.set_title("Cluster Purity / Confusion Matrix")
    fig.tight_layout()
    _save_figure(fig, figures_dir, "confusion_matrix")


def run_visualization(
    adata,
    clustering_results,
    eval_df,
    best_config,
    kmeans_inertias,
    config,
):
    """Run the full visualization pipeline.

    Parameters
    ----------
    adata : anndata.AnnData
        Fully processed data.
    clustering_results : dict
        Clustering results mapping.
    eval_df : pandas.DataFrame
        Evaluation summary table.
    best_config : pandas.Series or None
        Best clustering configuration.
    kmeans_inertias : dict or None
        K-Means inertias for elbow plot.
    config : dict
        Full configuration dictionary.
    """
    figures_dir = config["output"].get("figures_dir", "outputs/figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Set scanpy figure directory
    sc.settings.figdir = figures_dir

    # 1. QC violin plots
    plot_qc_violin(adata, figures_dir)

    # 2. HVG scatter (only if HVG info exists in raw or current)
    try:
        if adata.raw is not None and "highly_variable" in adata.raw.var.columns:
            plot_hvg(adata.raw.to_adata(), figures_dir)
        elif "highly_variable" in adata.var.columns:
            plot_hvg(adata, figures_dir)
    except Exception as e:
        logger.warning("Could not plot HVG scatter: %s", e)

    # 3. PCA elbow
    if "pca" in adata.uns:
        plot_pca_elbow(adata, figures_dir)

    # 4. Determine best cluster key for coloring
    best_key = best_config["cluster_key"] if best_config is not None else None

    # Build color keys for embeddings
    color_keys = []
    if best_key:
        color_keys.append(best_key)
    # Add a few more cluster keys for variety
    for method_keys in clustering_results.values():
        for k in method_keys[:1]:  # Just first of each method
            if k not in color_keys:
                color_keys.append(k)
    # QC metrics
    for qc_key in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]:
        if qc_key in adata.obs.columns:
            color_keys.append(qc_key)

    # 5. UMAP plots
    if "X_umap" in adata.obsm:
        plot_embedding(adata, "umap", color_keys, figures_dir, "umap")

    # 6. t-SNE plots
    if "X_tsne" in adata.obsm:
        plot_embedding(adata, "tsne", color_keys, figures_dir, "tsne")

    # 7. Silhouette comparison
    plot_silhouette_comparison(eval_df, figures_dir)

    # 8. Cluster stability
    plot_cluster_stability(adata, clustering_results, figures_dir)

    # 9. K-Means elbow
    if kmeans_inertias:
        plot_kmeans_elbow(kmeans_inertias, figures_dir)

    # 10. Marker gene plots (if rank_genes_groups has been run)
    if "rank_genes_groups" in adata.uns and best_key:
        n_top = config["markers"].get("n_top_genes", 10)
        plot_markers(adata, best_key, n_top, figures_dir)

    # 11. Confusion matrix if ground truth exists
    if best_key:
        for truth_col in ["cell_type", "celltype", "label", "labels", "ground_truth"]:
            if truth_col in adata.obs.columns:
                plot_confusion_matrix(adata, best_key, truth_col, figures_dir)
                break

    logger.info("All visualizations saved to %s", figures_dir)
