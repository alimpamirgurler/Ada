"""
Preprocessing module for scRNA-seq data.

Handles data ingestion (h5ad, csv, mtx formats), quality control filtering,
normalization, highly variable gene selection, scaling, and optional cell
cycle regression.
"""

import logging
import os
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)


def load_data(input_path, fmt="auto"):
    """Load scRNA-seq data from file or download a demo dataset.

    Parameters
    ----------
    input_path : str or None
        Path to the input file. If None, downloads the PBMC3k demo dataset.
    fmt : str
        File format: 'auto', 'h5ad', 'csv', or 'mtx'.

    Returns
    -------
    anndata.AnnData
        Loaded data as an AnnData object.
    """
    if input_path is None:
        logger.info("No input file provided. Downloading PBMC3k demo dataset...")
        adata = sc.datasets.pbmc3k()
        logger.info(
            "Loaded PBMC3k dataset: %d cells x %d genes", adata.n_obs, adata.n_vars
        )
        return adata

    input_path = str(input_path)
    if fmt == "auto":
        if input_path.endswith(".h5ad"):
            fmt = "h5ad"
        elif input_path.endswith(".csv"):
            fmt = "csv"
        elif input_path.endswith(".mtx") or input_path.endswith(".mtx.gz"):
            fmt = "mtx"
        else:
            raise ValueError(
                f"Cannot auto-detect format for '{input_path}'. "
                "Supported extensions: .h5ad, .csv, .mtx, .mtx.gz"
            )

    if fmt == "h5ad":
        logger.info("Loading h5ad file: %s", input_path)
        adata = sc.read_h5ad(input_path)
    elif fmt == "csv":
        logger.info("Loading CSV file: %s", input_path)
        adata = sc.read_csv(input_path)
        # If genes > cells, assume genes x cells layout and transpose
        if adata.n_vars < adata.n_obs:
            logger.info(
                "Detected genes x cells layout (n_obs=%d < n_vars=%d). Transposing.",
                adata.n_obs,
                adata.n_vars,
            )
            adata = adata.T
    elif fmt == "mtx":
        logger.info("Loading 10x MTX directory: %s", input_path)
        mtx_dir = (
            os.path.dirname(input_path) if os.path.isfile(input_path) else input_path
        )
        adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)
    else:
        raise ValueError(f"Unsupported format: '{fmt}'")

    logger.info("Loaded data: %d cells x %d genes", adata.n_obs, adata.n_vars)
    return adata


def quality_control(adata, min_genes=200, max_genes=5000, max_pct_mito=20):
    """Run quality control filtering on the AnnData object.

    Computes QC metrics, stores them in ``.obs``, and filters cells based on
    the provided thresholds.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data (modified in place).
    min_genes : int
        Minimum number of genes expressed per cell.
    max_genes : int
        Maximum number of genes expressed per cell (doublet filter).
    max_pct_mito : float
        Maximum percentage of mitochondrial reads per cell.

    Returns
    -------
    anndata.AnnData
        Filtered AnnData object.
    """
    logger.info("Running quality control...")

    # Ensure gene names are strings for the mitochondrial check
    adata.var_names_make_unique()

    # Identify mitochondrial genes (prefix MT- or mt-)
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    n_before = adata.n_obs
    logger.info(
        "QC thresholds: min_genes=%d, max_genes=%d, max_pct_mito=%.1f%%",
        min_genes,
        max_genes,
        max_pct_mito,
    )

    adata = adata[adata.obs["n_genes_by_counts"] >= min_genes, :].copy()
    adata = adata[adata.obs["n_genes_by_counts"] <= max_genes, :].copy()
    adata = adata[adata.obs["pct_counts_mt"] <= max_pct_mito, :].copy()

    n_after = adata.n_obs
    logger.info(
        "QC filtering: %d -> %d cells (%d removed)",
        n_before,
        n_after,
        n_before - n_after,
    )

    return adata


def normalize_data(adata, target_sum=1e4):
    """Normalize counts: library-size normalization followed by log1p.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data (modified in place).
    target_sum : float
        Target total counts per cell after normalization.

    Returns
    -------
    anndata.AnnData
        Normalized AnnData object.
    """
    logger.info("Normalizing data (target_sum=%d)...", int(target_sum))
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_highly_variable_genes(adata, n_top_genes=2000):
    """Select highly variable genes and subset the data.

    Parameters
    ----------
    adata : anndata.AnnData
        Normalized data.
    n_top_genes : int
        Number of top highly variable genes to select.

    Returns
    -------
    anndata.AnnData
        AnnData with HVGs annotated; raw counts stored in ``.raw``.
    """
    logger.info("Selecting top %d highly variable genes...", n_top_genes)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3" if "counts" in adata.layers else "seurat")
    n_hvg = adata.var["highly_variable"].sum()
    logger.info("Found %d highly variable genes", n_hvg)

    # Store full data in .raw before subsetting
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info("Subset to HVGs: %d cells x %d genes", adata.n_obs, adata.n_vars)
    return adata


def scale_data(adata, max_value=10):
    """Scale data to zero mean and unit variance, clipping at max_value.

    Parameters
    ----------
    adata : anndata.AnnData
        HVG-subset data.
    max_value : float
        Maximum value after scaling (clips outliers).

    Returns
    -------
    anndata.AnnData
        Scaled AnnData object.
    """
    logger.info("Scaling data (max_value=%d)...", max_value)
    sc.pp.scale(adata, max_value=max_value)
    return adata


def score_cell_cycle(adata):
    """Score and regress out cell cycle effects.

    Uses the Tirosh et al. 2015 cell cycle gene sets bundled with scanpy.

    Parameters
    ----------
    adata : anndata.AnnData
        Scaled data.

    Returns
    -------
    anndata.AnnData
        Data with cell cycle effects regressed out.
    """
    logger.info("Scoring cell cycle genes and regressing out effects...")

    # Cell cycle genes from Tirosh et al. 2015
    cell_cycle_genes = [
        x.strip()
        for x in open(
            os.path.join(
                os.path.dirname(sc.__file__), "datasets", "_data", "cell_cycle_genes.txt"
            )
        ).readlines()
    ] if os.path.exists(
        os.path.join(
            os.path.dirname(sc.__file__), "datasets", "_data", "cell_cycle_genes.txt"
        )
    ) else None

    if cell_cycle_genes is None:
        # Fallback: standard S and G2M gene lists
        s_genes = [
            "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
            "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
            "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
            "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
            "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
            "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
        ]
        g2m_genes = [
            "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
            "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
            "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
            "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
            "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
            "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
            "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
            "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
        ]
    else:
        s_genes = cell_cycle_genes[:43]
        g2m_genes = cell_cycle_genes[43:]

    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    sc.pp.regress_out(adata, ["S_score", "G2M_score"])
    sc.pp.scale(adata, max_value=10)

    logger.info("Cell cycle regression complete.")
    return adata


def run_preprocessing(adata, config):
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    adata : anndata.AnnData
        Raw loaded data.
    config : dict
        Preprocessing section of the configuration.

    Returns
    -------
    anndata.AnnData
        Fully preprocessed AnnData object.
    """
    pp_cfg = config["preprocessing"]

    adata = quality_control(
        adata,
        min_genes=pp_cfg.get("min_genes", 200),
        max_genes=pp_cfg.get("max_genes", 5000),
        max_pct_mito=pp_cfg.get("max_pct_mito", 20),
    )

    adata = normalize_data(adata)
    adata = select_highly_variable_genes(
        adata, n_top_genes=pp_cfg.get("n_top_genes", 2000)
    )
    adata = scale_data(adata)

    if pp_cfg.get("regress_cell_cycle", False):
        adata = score_cell_cycle(adata)

    return adata
