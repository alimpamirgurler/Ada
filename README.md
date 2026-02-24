# The Project was named after my dog, Ada
![Ada the dog](C:\Users\alimp\Downloads\Ada in the Snow_Square.png)

# scRNA-seq Cancer Subtype Clustering Pipeline

An end-to-end unsupervised clustering pipeline for single-cell RNA sequencing (scRNA-seq) data, designed for cancer subtype discovery. The pipeline integrates preprocessing, dimensionality reduction, multiple clustering algorithms, quantitative evaluation, marker gene identification, and comprehensive visualization.

## Project Structure

```
scrna_cancer_clustering/
├── data/                              # Raw and processed data
├── outputs/
│   ├── figures/                       # Generated plots (PNG + PDF)
│   └── results/                       # CSVs, processed h5ad
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Data loading, QC, normalization, HVG, scaling
│   ├── dimensionality_reduction.py    # PCA, UMAP, t-SNE
│   ├── clustering.py                  # Leiden, Louvain, K-Means, Hierarchical
│   ├── visualization.py               # All plotting functions
│   └── evaluation.py                  # Silhouette, DBI, CHI, composite scoring
├── main.py                            # CLI entry point
├── config.yaml                        # Pipeline configuration
├── requirements.txt                   # Python dependencies
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
# Uses the default config.yaml and downloads the PBMC3k demo dataset
python main.py
```

The pipeline will automatically download the PBMC 3k dataset from scanpy's built-in datasets if no input file is specified.

### 3. Run with your own data

```bash
# Override input via CLI
python main.py --input path/to/your/data.h5ad

# Or use a custom config
python main.py --config my_config.yaml
```

## Supported Input Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| AnnData | `.h5ad` | Native format, recommended |
| CSV | `.csv` | Cells × genes or genes × cells (auto-detected) |
| 10x MTX | `.mtx` / `.mtx.gz` | Expects `barcodes.tsv` and `features.tsv` in same directory |

The format is auto-detected from the file extension, or can be set explicitly in `config.yaml`.

## Pipeline Steps

### 1. Data Ingestion
- Loads data from h5ad, CSV, or 10x MTX formats
- Downloads PBMC3k demo dataset if no input is provided

### 2. Preprocessing
- **Quality Control**: Filters cells by min/max genes expressed and max mitochondrial read percentage
- **Normalization**: Library-size normalization (target 10,000 counts/cell) + log1p transform
- **HVG Selection**: Selects top N highly variable genes (default: 2000)
- **Scaling**: Zero-mean, unit-variance scaling (clipped at 10)
- **Cell Cycle Regression** (optional): Scores and regresses out cell cycle effects

### 3. Dimensionality Reduction
- **PCA**: 50 components with elbow/scree plot
- **UMAP**: On PCA embedding with configurable neighbors and min_dist
- **t-SNE**: On PCA embedding as alternative visualization

### 4. Clustering
- **Leiden**: Resolution sweep (e.g., 0.3, 0.5, 0.8, 1.0, 1.5)
- **Louvain**: Same resolution sweep
- **K-Means**: k from 2 to 10 on PCA embedding
- **Hierarchical**: Ward linkage, k from 2 to 10 on PCA embedding

### 5. Evaluation
- **Silhouette Score**: Cohesion vs. separation (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Cluster compactness vs. separation (lower is better)
- **Calinski-Harabasz Index**: Between-cluster / within-cluster variance (higher is better)
- **Composite Score**: Normalized combination of all three metrics
- Automatic recommendation of the best clustering configuration

### 6. Marker Gene Identification
- Wilcoxon rank-sum test via `scanpy.tl.rank_genes_groups`
- Top N marker genes per cluster exported to CSV
- Heatmap and dot plot visualization

### 7. Visualization
All figures are saved as both PNG and PDF:

- QC violin plots (before and after filtering)
- Highly variable genes scatter plot
- PCA elbow/scree plot
- UMAP colored by cluster labels, QC metrics, and top marker genes
- t-SNE with same coloring scheme
- Silhouette score bar chart comparing all methods
- Cluster stability plot (cluster count vs. resolution)
- K-Means elbow plot
- Marker gene heatmap and dot plot
- Confusion matrix / cluster purity (if ground truth labels exist)

## Configuration

All parameters are controlled via `config.yaml`:

```yaml
data:
  input_path: null   # null = download demo dataset
  format: auto       # auto, h5ad, csv, mtx

preprocessing:
  min_genes: 200
  max_genes: 5000
  max_pct_mito: 20
  n_top_genes: 2000
  regress_cell_cycle: false

dimensionality_reduction:
  n_pcs: 50
  n_pcs_use: 30
  umap_n_neighbors: 15
  umap_min_dist: 0.1

clustering:
  methods: [leiden, kmeans]
  leiden_resolutions: [0.3, 0.5, 0.8, 1.0, 1.5]
  louvain_resolutions: [0.3, 0.5, 0.8, 1.0, 1.5]
  kmeans_k_range: [2, 10]
  hierarchical_k_range: [2, 10]

markers:
  n_top_genes: 10
  method: wilcoxon

output:
  figures_dir: outputs/figures
  results_dir: outputs/results
  save_h5ad: true
```

## CLI Options

```
python main.py [--config CONFIG] [--input INPUT]

Options:
  --config CONFIG   Path to YAML config file (default: config.yaml)
  --input INPUT     Path to input data file (overrides config)
```

## Output

After a successful run, the `outputs/` directory will contain:

```
outputs/
├── figures/
│   ├── before_filtering/          # Pre-QC violin plots
│   ├── qc_violin.png/pdf         # Post-QC violin plots
│   ├── pca_elbow.png/pdf         # PCA scree plot
│   ├── umap_*.png/pdf            # UMAP embeddings
│   ├── tsne_*.png/pdf            # t-SNE embeddings
│   ├── silhouette_comparison.png/pdf
│   ├── cluster_stability.png/pdf
│   ├── kmeans_elbow.png/pdf
│   ├── marker_dotplot.png/pdf
│   └── marker_heatmap.png/pdf
└── results/
    ├── clustering_evaluation.csv  # All metrics for all configurations
    ├── marker_genes.csv           # Top marker genes per cluster
    └── processed.h5ad             # Full processed AnnData object
```

## Reproducibility

All stochastic operations use `random_state=42` for full reproducibility.

## Dependencies

- scanpy >= 1.9.0
- anndata >= 0.9.0
- numpy >= 1.23.0
- pandas >= 1.5.0
- scikit-learn >= 1.2.0
- umap-learn >= 0.5.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- leidenalg >= 0.9.0
- python-igraph >= 0.10.0
- pyyaml >= 6.0
- scipy >= 1.10.0
