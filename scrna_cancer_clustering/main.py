#!/usr/bin/env python3
"""
scRNA-seq Cancer Subtype Clustering Pipeline

End-to-end unsupervised clustering pipeline for single-cell RNA sequencing
data. Supports multiple clustering algorithms, evaluation metrics, marker
gene identification, and comprehensive visualization.

Usage:
    python main.py                       # Use defaults (config.yaml, demo dataset)
    python main.py --config my_config.yaml
    python main.py --input data/my_data.h5ad
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import yaml

from src.clustering import run_clustering
from src.dimensionality_reduction import run_dimensionality_reduction
from src.evaluation import run_evaluation
from src.preprocessing import load_data, run_preprocessing
from src.visualization import run_visualization

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging():
    """Configure logging to stdout with INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load the YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Marker gene identification
# ---------------------------------------------------------------------------

def identify_markers(adata, best_key, config):
    """Run marker gene identification on the best clustering.

    Parameters
    ----------
    adata : anndata.AnnData
        Fully processed data with clustering results.
    best_key : str
        Column in ``adata.obs`` with the best cluster labels.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    pandas.DataFrame or None
        Marker gene results table, or None on failure.
    """
    logger = logging.getLogger(__name__)
    marker_cfg = config.get("markers", {})
    method = marker_cfg.get("method", "wilcoxon")
    n_top = marker_cfg.get("n_top_genes", 10)
    results_dir = config["output"].get("results_dir", "outputs/results")
    os.makedirs(results_dir, exist_ok=True)

    logger.info(
        "Identifying marker genes for '%s' using %s test...", best_key, method
    )

    try:
        # Use raw data if available for marker gene analysis
        use_raw = adata.raw is not None
        sc.tl.rank_genes_groups(
            adata, groupby=best_key, method=method, use_raw=use_raw
        )
        logger.info("Marker gene identification complete.")

        # Extract results to a DataFrame
        result = adata.uns["rank_genes_groups"]
        groups = result["names"].dtype.names
        records = []
        for group in groups:
            for i in range(n_top):
                try:
                    records.append({
                        "cluster": group,
                        "rank": i + 1,
                        "gene": result["names"][group][i],
                        "score": result["scores"][group][i],
                        "logfoldchange": result["logfoldchanges"][group][i],
                        "pval": result["pvals"][group][i],
                        "pval_adj": result["pvals_adj"][group][i],
                    })
                except (IndexError, KeyError):
                    break

        marker_df = pd.DataFrame(records)
        csv_path = os.path.join(results_dir, "marker_genes.csv")
        marker_df.to_csv(csv_path, index=False)
        logger.info("Marker genes saved to %s", csv_path)
        return marker_df

    except Exception as e:
        logger.error("Marker gene identification failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(adata, clustering_results, eval_df, best_config, config):
    """Print a final summary of the pipeline run.

    Parameters
    ----------
    adata : anndata.AnnData
        Processed data.
    clustering_results : dict
        Clustering results mapping.
    eval_df : pandas.DataFrame
        Evaluation summary.
    best_config : pandas.Series or None
        Best clustering configuration.
    config : dict
        Full configuration dictionary.
    """
    logger = logging.getLogger(__name__)
    figures_dir = config["output"].get("figures_dir", "outputs/figures")
    results_dir = config["output"].get("results_dir", "outputs/results")

    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Cells:  {adata.n_obs}")
    print(f"  Genes:  {adata.n_vars}")
    print()

    print("  Clustering Results:")
    for method, keys in clustering_results.items():
        for key in keys:
            n_clust = adata.obs[key].nunique()
            print(f"    {key}: {n_clust} clusters")
    print()

    if best_config is not None:
        print("  Best Configuration:")
        print(f"    Method:           {best_config['cluster_key']}")
        print(f"    Clusters:         {int(best_config['n_clusters'])}")
        print(f"    Silhouette:       {best_config['silhouette_score']:.4f}")
        print(f"    Davies-Bouldin:   {best_config['davies_bouldin_index']:.4f}")
        print(f"    Calinski-Harabasz:{best_config['calinski_harabasz_index']:.1f}")
        print(f"    Composite Score:  {best_config['composite_score']:.4f}")
    print()

    print("  Output Files:")
    print(f"    Figures:    {os.path.abspath(figures_dir)}/")
    print(f"    Results:    {os.path.abspath(results_dir)}/")

    # List generated files
    for d in [figures_dir, results_dir]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                print(f"      - {f}")

    h5ad_path = os.path.join(results_dir, "processed.h5ad")
    if os.path.exists(h5ad_path):
        print(f"    AnnData:    {os.path.abspath(h5ad_path)}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the full scRNA-seq clustering pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="scRNA-seq Cancer Subtype Clustering Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input data file (overrides config)",
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        # Look relative to this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, config_path)
        if os.path.exists(candidate):
            config_path = candidate

    logger.info("Loading configuration from %s", config_path)
    config = load_config(config_path)

    # Override input path if provided via CLI
    input_path = args.input or config.get("data", {}).get("input_path")
    fmt = config.get("data", {}).get("format", "auto")

    figures_dir = config["output"].get("figures_dir", "outputs/figures")
    results_dir = config["output"].get("results_dir", "outputs/results")

    # Make output dirs relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(figures_dir):
        figures_dir = os.path.join(script_dir, figures_dir)
        config["output"]["figures_dir"] = figures_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(script_dir, results_dir)
        config["output"]["results_dir"] = results_dir

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Set scanpy settings
    sc.settings.verbosity = 1
    sc.settings.figdir = figures_dir
    sc.settings.set_figure_params(dpi=150, frameon=False)

    start_time = time.time()

    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)
    adata = load_data(input_path, fmt=fmt)

    # Store a copy of raw counts for QC plotting later
    adata_raw_for_qc = adata.copy()

    # -----------------------------------------------------------------------
    # Step 2: Preprocessing
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing")
    logger.info("=" * 60)

    # Compute QC metrics on unfiltered data for "before" plots
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # Plot QC before filtering
    from src.visualization import plot_qc_violin
    plot_qc_violin(adata, os.path.join(figures_dir, "before_filtering"))

    adata = run_preprocessing(adata, config)

    # -----------------------------------------------------------------------
    # Step 3: Dimensionality reduction
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Dimensionality Reduction")
    logger.info("=" * 60)
    adata = run_dimensionality_reduction(adata, config)

    # -----------------------------------------------------------------------
    # Step 4: Clustering
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Clustering")
    logger.info("=" * 60)
    clustering_results, kmeans_inertias = run_clustering(adata, config)

    # -----------------------------------------------------------------------
    # Step 5: Evaluation
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Cluster Evaluation")
    logger.info("=" * 60)
    eval_df, best_config = run_evaluation(adata, clustering_results, results_dir)

    # -----------------------------------------------------------------------
    # Step 6: Marker gene identification
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6: Marker Gene Identification")
    logger.info("=" * 60)
    best_key = best_config["cluster_key"] if best_config is not None else None
    marker_df = None
    if best_key:
        marker_df = identify_markers(adata, best_key, config)
    else:
        logger.warning("No best clustering found; skipping marker gene analysis.")

    # -----------------------------------------------------------------------
    # Step 7: Visualization
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 7: Visualization")
    logger.info("=" * 60)
    run_visualization(
        adata,
        clustering_results,
        eval_df,
        best_config,
        kmeans_inertias,
        config,
    )

    # -----------------------------------------------------------------------
    # Step 8: Save processed AnnData
    # -----------------------------------------------------------------------
    if config["output"].get("save_h5ad", True):
        h5ad_path = os.path.join(results_dir, "processed.h5ad")
        logger.info("Saving processed AnnData to %s", h5ad_path)
        adata.write_h5ad(h5ad_path)
        logger.info("AnnData saved.")

    elapsed = time.time() - start_time
    logger.info("Pipeline completed in %.1f seconds.", elapsed)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_summary(adata, clustering_results, eval_df, best_config, config)


if __name__ == "__main__":
    main()
