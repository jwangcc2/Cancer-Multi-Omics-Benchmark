# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLOmics is a comprehensive cancer multi-omics database and benchmarking framework for machine learning research. It contains 8,314 patient samples across 32 cancer types with four omics data types (mRNA, miRNA, CNV, Methylation) and provides standardized datasets, baseline implementations, and evaluation metrics for three main tasks: classification, clustering, and imputation.

## Environment Setup

```bash
# Create and activate conda environment (Python 3.9 recommended for TensorFlow 2.10 compatibility)
conda create -n mlomics python=3.9
conda activate mlomics

# Install dependencies
pip install -r requirements.txt

# Download datasets from HuggingFace (requires git-lfs)
./download.sh
```

**Important**: This codebase requires TensorFlow 2.10.0, which is compatible with Python 3.7-3.10. Python 3.9 is recommended for best compatibility.

Alternative: Download datasets manually from Figshare: https://figshare.com/articles/dataset/MLOmics_Cancer_Multi-Omics_Database_for_Machine_Learning/28729127

## Key Commands

### Running Baseline Models

All baselines follow a standardized interface via shell scripts in the `scripts/` directory:

```bash
# General pattern
cd scripts/<TaskType>
./<model>.sh <dataset> <version> [options]
```

**Clustering Tasks:**
```bash
cd scripts/Clustering
./SubtypeGAN.sh ACC Top
```

This executes SubtypeGAN in two phases:
1. Feature extraction and initial clustering with dummy K=4
2. Consensus clustering to determine optimal K (tests K=2 to K=8)

**Imputation Tasks:**
```bash
cd scripts/Imputation
./GAIN.sh BRCA CNV 0.3  # 30% missing rate
./GRAPE.sh COAD mRNA 0.5  # 50% missing rate
```

**Classification Tasks:**
Classification baselines are run directly from their respective directories in `Baseline_and_Metric/Classification/Baselines/Python/`.

### Direct Model Execution (Advanced)

For more control, you can run models directly:

```bash
cd Baseline_and_Metric/Clustering/Baselines/Python/Subtype-GAN/

# Phase 1: Feature extraction + clustering with specified K
python SubtypeGAN.py -m SubtypeGAN -n 4 -t ACC_Top

# Phase 2: Consensus clustering to find optimal K
python SubtypeGAN.py -m cc -t ACC_Top
```

Arguments:
- `-m`: Run mode (SubtypeGAN or cc)
- `-n`: Number of clusters (required for SubtypeGAN mode)
- `-t`: Dataset type (format: `{cancer}_{version}`)
- `-e`: Number of training epochs (default: 200)
- `-w`: Discriminator weight (default: 1e-4)

Output directories created:
- `./fea/`: Extracted features (TSV format)
- `./model/`: Trained model weights (.h5 files)
- `./results/`: Clustering results and timing information

### Evaluation Metrics

Metrics are implemented in task-specific Python modules:

- Classification: `Baseline_and_Metric/Classification/Metrics/Classification_metrics.py`
- Clustering: `Baseline_and_Metric/Clustering/Metrics/Clustering_metrics.py`
- Imputation: Located within each imputation baseline implementation

## Architecture Overview

### Multi-Scale Feature Design

MLOmics provides three feature scale versions for each dataset to address sample-feature imbalance:

- **Original**: Complete unfiltered feature set from raw data
- **Top**: ANOVA-selected significant features (typically 5000 for mRNA, 200 for miRNA, 5000 for CNV/Methy)
- **Aligned**: Feature intersection across all sub-datasets (e.g., 10452 mRNA, 254 miRNA, 10347 Methy, 10154 CNV for clustering datasets)

This multi-scale approach is critical for model compatibility and computational feasibility.

### Dataset Organization

```
Main_Dataset/
├── Classification_datasets/
│   ├── Pan-cancer/          # 32 cancer types, labeled
│   └── Golden_Standard/     # GS-BRCA, GS-COAD, GS-GBM, GS-LGG, GS-OV (5 datasets)
├── Clustering_datasets/     # ACC, KIRP, KIRC, LIHC, LUAD, LUSC, PRAD, THCA, THYM (9 datasets)
└── Imputation_datasets/     # Imp-BRCA, Imp-COAD, Imp-GBM, Imp-LGG, Imp-OV (5 datasets)
```

Each dataset contains separate CSV files for each omics type (mRNA.csv, miRNA.csv, CNV.csv, Methy.csv) and labels where applicable (Label.csv).

### Baseline Implementation Structure

```
Baseline_and_Metric/
├── Classification/
│   ├── Baselines/Python/    # Deep learning models (CustOmics, DCAP, MAUI, XOmiVAE, etc.)
│   └── Metrics/
├── Clustering/
│   ├── Baselines/
│   │   ├── Python/          # Deep learning (Subtype-GAN, MCluster-VAEs)
│   │   └── R/               # Traditional ML (SNF, NEMO, CIMLR, iClusterBayes, moCluster)
│   └── Metrics/
└── Imputation/
    ├── GAIN/                # Generative Adversarial Imputation Nets
    ├── GRAPE/               # Graph neural network for tabular data
    └── [Traditional methods: Mean, KNN, MICE, SVD, Spectral]
```

**Important**: Most baselines are implemented in Python (deep learning), but several key clustering methods (SNF, NEMO, CIMLR, iClusterBayes, moCluster) are R-based. Ensure R and required packages are installed when working with these methods.

### Downstream Analysis Integration

The `Downstream_Analysis_Tools_and_Resources/` directory provides:

- **Analysis_tools/Analysis_Tools.py**: Functions for differential expression analysis, volcano plots, and KEGG pathway enrichment analysis
- **Knowledge_bases/**: STRING and KEGG database mappings for biological interpretation
- **Clinical_annotation/clinical_record.csv**: Patient clinical data for survival analysis and clinical correlation

Key analysis workflow:
1. Run clustering/classification baseline → generates log/results
2. Use Analysis_Tools.py to identify differentially expressed genes between clusters
3. Map genes to biological pathways using KEGG_mapping.csv or STRING_mapping.csv
4. Visualize with volcano plots and pathway enrichment plots

## Task-Specific Guidance

### Classification Tasks

- **Pan-cancer**: Classify patient samples into one of 32 cancer types
- **Golden-standard subtypes**: Classify into validated cancer subtypes (e.g., BRCA has PAM50 subtypes)
- **Metrics**: Precision (weighted), NMI, ARI
- **Key models**: XOmiVAE and MCluster-VAEs achieve best performance (PREC > 0.88)

### Clustering Tasks

- **Objective**: Discover cancer subtypes without ground truth labels
- **Metrics**: Silhouette coefficient (SIL), log-rank test p-value on survival (LPS)
- **Validation**: Results should correlate with survival outcomes and clinical characteristics
- **Note**: Use R-based methods (SNF, NEMO) for traditional statistical approaches

### Imputation Tasks

- **Formats**: Work with pre-masked omics data at 30%, 50%, 70% missing rates
- **Metrics**: RMSE, MAE
- **Best performers**: SVD and Spectral methods generally outperform deep learning approaches (GAIN, GRAPE) at lower missing rates
- **Dataset naming**: Use format `{cancer_type}_{omics_type}` (e.g., BRCA_CNV, COAD_mRNA)

## Cancer Type Abbreviations

Common abbreviations used throughout the codebase:
- BRCA: Breast Invasive Carcinoma
- COAD: Colon Adenocarcinoma
- GBM: Glioblastoma Multiforme
- LGG: Brain Lower Grade Glioma
- LUAD: Lung Adenocarcinoma
- LUSC: Lung Squamous Cell Carcinoma
- KIRC: Kidney Clear Cell Carcinoma
- KIRP: Kidney Papillary Cell Carcinoma

See README.md for complete list of all 32 cancer types.

## Development Notes

### Adding New Baselines

1. Implement model in `Baseline_and_Metric/<Task>/Baselines/Python/` or `R/`
2. Create wrapper script in `scripts/<Task>/<ModelName>.sh`
3. Follow naming convention: `<dataset>_<version>` for inputs
4. Output results to a standardized format compatible with metrics modules

### Data Loading Patterns

- Omics data files use sample IDs as index (rows) and gene/feature IDs as columns
- Label files contain: sample ID index and 'label' or 'subtype' column
- All data is pre-normalized and log-transformed where appropriate
- Missing values in imputation datasets are represented as NaN

### Common Integration Patterns

Most deep learning baselines follow this architecture:
1. **Encoder per omics type**: Separate neural networks encode each omics modality
2. **Latent space fusion**: Concatenate or integrate encoded representations
3. **Task-specific decoder**: Classification head, clustering loss, or reconstruction decoder
4. **Common losses**: Cross-entropy (classification), MMD/consensus loss (clustering), MSE (imputation)

VAE-based models (XOmiVAE, MCluster-VAEs, MAUI) add KL-divergence loss for regularization.

#### SubtypeGAN Architecture Details

SubtypeGAN uses a specific fusion strategy with weighted latent dimensions:
- **Weight allocation**: [0.3, 0.1, 0.1, 0.5] for [mRNA, miRNA, CNV, Methy]
- **Latent dimension**: Default 100, split proportionally among omics types
- **Activation**: Custom GeLU activation function
- **Training**: Adversarial training with discriminator on latent space
- **Epochs**: Adaptive based on sample size (typically 30 * batch_size)

#### GAIN Architecture Details

GAIN (Generative Adversarial Imputation Nets) uses:
- **Generator**: Takes concatenated [data, mask] → produces imputed values
- **Discriminator**: Takes concatenated [imputed_data, hint] → predicts which values are real
- **Hint mechanism**: Randomly reveals some true values to discriminator (hint_rate=0.9)
- **Loss**: Combines adversarial loss with MSE reconstruction loss (alpha=100)
- **Data normalization**: Min-max normalization before training, reversed after imputation

### Working with Shell Scripts

Shell scripts in `scripts/` directory:
- Save current directory before navigation
- Navigate to baseline implementation folder (paths use relative navigation like `../../../`)
- Execute Python/R code with appropriate arguments
- Log execution time and results
- Return to original directory

When modifying scripts, maintain this pattern for consistency across the benchmark.

### Troubleshooting Common Issues

**TensorFlow compatibility errors:**
- Ensure Python 3.7-3.10 (3.9 recommended)
- TensorFlow 2.10.0 is pinned in requirements.txt
- Some models use `tensorflow.compat.v1` API

**Path issues in SubtypeGAN:**
- The data path uses unusual pattern: `'../../../....//Main_Dataset/Clustering_datasets/'`
- This navigates from the model directory to the project root
- Ensure you're running from the correct working directory

**R-based models:**
- Install R (version 3.6+) and required packages
- Common R dependencies: SNFtool, iClusterPlus, NEMO, mogsa
- R scripts may need manual package installation

**Missing output directories:**
- Models automatically create `./fea/`, `./model/`, `./results/` directories
- Ensure write permissions in the baseline model directory
