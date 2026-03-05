## Environment

- OS: macOS (Apple Silicon)
- Python: 3.x
- Framework: PyTorch (CPU)
- Reproducibility: Fixed seeds, documented dependencies

# #################################################### 
# Machine Unlearning: Privacy–Efficiency Trade-offs

This repository contains experimental implementations and evaluations of **machine unlearning (MU)** methods, with a focus on understanding the **privacy–efficiency trade-off**. The current stage of the project explores **centralized machine unlearning**, serving as a foundation before extending the work to **federated unlearning**.

This work is part of an undergraduate final-year individual research project in **Artificial Intelligence & Data Science**.

## Project Motivation

Modern machine learning systems are increasingly subject to privacy regulations (e.g., GDPR’s *Right to be Forgotten*), which require the removal of specific data points from trained models.

**Machine Unlearning (MU)** aims to:

* Remove the influence of specific training samples
* Avoid full retraining where possible
* Balance **privacy guarantees** with **computational efficiency**

This project systematically compares multiple unlearning strategies under a unified evaluation framework.

## Current Scope (Centralized Setting)

At this stage, all experiments are conducted in a **centralized (non-federated)** learning setup using the **MNIST dataset** and a simple CNN model.

### Implemented Unlearning Methods
          
**Exact Retraining (Gold Standard)** - Fully retrains the model from scratch after removing the target data. Used as a privacy-optimal baseline.
**Gradient Ascent Unlearning** - Approximate unlearning by performing gradient ascent steps on the deleted data to reduce its influence.
**First-Epoch Reversal** - Checkpoint-based unlearning that restores early model weights and retrains only on retained data.

## Evaluation Metrics

Each unlearning method is evaluated across multiple dimensions:

### Privacy Metrics
* **Delete Set Loss/Accuracy** - How well the model "forgets" the deleted data
* **Per-Class Forgetting Score** - Fine-grained forgetting analysis for each digit class
* **Membership Inference Attack (MIA)** - Privacy evaluation using attack accuracy, TPR, FPR, and AUC-ROC

### Utility Metrics
* **Retain Set Accuracy/Loss** - How well the model maintains performance on retained data
* **Per-Class Utility Loss** - Class-level impact analysis on retained performance
* **Confusion Matrices** - Detailed classification behavior visualization

### Efficiency Metrics
* **Unlearning Time** - Computational cost compared to exact retraining baseline
* **Speedup Factor** - How many times faster than full retraining

### Model Distance Metrics
* **L2 Distance (Absolute & Relative)** - Parameter-level similarity to exact retrained model (gold standard)

All metrics are captured in structured JSON format and visualized through comprehensive plots including confusion matrices, per-class accuracy comparisons, and privacy-utility-efficiency trade-off analyses.

##  Repository Structure

```text
.
├── mia.py                  # Membership Inference Attack evaluation module
├── requirements.txt        # Python dependencies
├── notebooks/
│   ├── gradient_ascent_and_gold_model.ipynb  # Exact retraining & gradient ascent
│   ├── first_epoch_reversal.ipynb            # Checkpoint-based unlearning
│   ├── technique_comparison.ipynb            # Cross-method comparison & visualization
│   ├── data/MNIST/         # (ignored) MNIST dataset files
│   └── models/             # (ignored) saved model checkpoints (.pt files)
├── results/
│   ├── metadata/           # JSON experiment results (technique_seed_XXXX_YpYY.json)
│   ├── csv/                # CSV comparison exports
│   └── plots_*/            # Organized plot outputs per experiment
├── data_splits/            # (ignored) reproducible train/delete/retain splits (.npz)
└── README.md
```

> ⚠️ Dataset files, trained models, and experimental results are intentionally excluded from version control.

##  Setup Instructions

### Create a virtual environment

```bash
python -m venv muvenv
source muvenv/bin/activate   # Linux / macOS
muvenv\Scripts\activate      # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```


##  Running Experiments

Experiments are conducted through three main Jupyter notebooks:

### 1. **Exact Retraining & Gradient Ascent**
`notebooks/gradient_ascent_and_gold_model.ipynb`

Implements baseline training, exact retraining (gold standard), and gradient ascent unlearning. Includes per-class accuracy analysis and MIA privacy evaluation.

### 2. **First-Epoch Reversal (Checkpoint-based)**
`notebooks/first_epoch_reversal.ipynb`

Implements checkpoint-based unlearning by reverting to early training checkpoints and retraining on retained data only.

### 3. **Cross-Method Comparison**
`notebooks/technique_comparison.ipynb`

Loads results from both experiment notebooks and generates comparative visualizations across all three methods (exact, gradient ascent, first-epoch reversal).

### Experiment Configuration

Each notebook is parameterized with two key variables:

```python
SEED = 42              # Random seed for reproducibility
DELETE_RATIO = 0.01    # Fraction of data to unlearn (e.g., 0.01 = 1%)
```

To run experiments with different configurations:
1. Modify `SEED` and `DELETE_RATIO` values in the configuration cell
2. Execute all cells in sequence
3. Results automatically save to structured folders with consistent naming

All file paths (JSON outputs, plots, model checkpoints) **automatically adjust** based on these parameters, ensuring organized and reproducible experiments without manual path updates.

##  Results Format

### Structured Output Organization

All experimental results are organized in a consistent, reproducible structure:

**JSON Metadata** (`results/metadata/`):
```
gradient_ascent_and_gold_seed_0042_0p01_results.json
first_epoch_reversal_seed_0042_0p01_results.json
```

**CSV Exports** (`results/csv/`):
```
technique_comparison_seed_0042_0p01.csv
```

**Plots** (`results/plots_[technique]_seed_[XXXX]_delete_[YpYY]/`):
```
plots_gradient_ascent_and_gold_seed_0042_delete_0p01/
plots_first_epoch_reversal_seed_0042_delete_0p01/
plots_technique_comparison_seed_0042_delete_0p01/
```

### JSON Schema Example

All experiments follow a unified JSON structure:

```json
{
  "experiment_info": {
    "dataset": "MNIST",
    "model": "SimpleCNN",
    "deletion_ratio": 0.01,
    "seed": 42,
    "timestamp": "2026-03-06T10:30:00"
  },
  "[method_name]": {
    "method": "gradient_ascent | first_epoch_reversal",
    "metrics": {
      "deleted_loss": 0.0048,
      "deleted_accuracy": 0.999,
      "retained_loss": 0.0324,
      "retained_accuracy": 0.999
    },
    "per_class_accuracy": {
      "retain_set": {"0": 0.998, "1": 0.997, ...},
      "delete_set": {"0": 0.996, "1": 0.995, ...}
    },
    "time_seconds": 2.04,
    "distance_metrics": {
      "l2_absolute": 5.23,
      "l2_relative": 0.0234
    }
  },
  "privacy_evaluation": {
    "mia_baseline": {"mia_accuracy": 0.62, ...},
    "mia_unlearned": {"mia_accuracy": 0.51, ...},
    "privacy_improvement": 0.11
  }
}
```

This structured format enables:
- **Reproducible experiments** across different seeds and delete ratios
- **Automated comparison** in the technique_comparison notebook
- **Version-controlled metadata** without storing large model files

### File Naming Convention

All output files follow a consistent naming pattern:

- **Seed Format**: `XXXX` (4-digit zero-padded, e.g., `0042` for SEED=42)
- **Delete Ratio Format**: `YpYY` (decimal point replaced with 'p', e.g., `0p01` for 0.01, `0p05` for 0.05)

**Examples:**
- `gradient_ascent_and_gold_seed_0042_0p01_results.json`
- `plots_first_epoch_reversal_seed_0042_delete_0p01/`
- `technique_comparison_seed_0100_0p05.csv`

This convention ensures lexicographic sorting matches numerical ordering and eliminates ambiguity across different delete ratios.
##  Results Format

All experiments follow a unified JSON schema, for example:

```json
{
  "experiment_info": {
    "dataset": "MNIST",
    "model": "SimpleCNN",
    "deletion_ratio": 0.05
  },
  "method_results": {
    "deleted_loss": 0.07,
    "retained_accuracy": 0.99,
    "time_seconds": 37.2
  }
}
```

This enables fair and consistent cross-method comparisons.

##  Roadmap

- [x] **Centralized machine unlearning baselines** (Completed)
  - [x] Exact retraining (gold standard)
  - [x] Gradient ascent unlearning
  - [x] First-epoch reversal (checkpoint-based)
  - [x] Comprehensive evaluation framework
  - [x] Structured results organization
- [ ] **Federated learning setup (FL)**
- [ ] **Federated unlearning methods**
- [ ] **Privacy–efficiency trade-off analysis**
- [ ] **Thesis-level evaluation and reporting**

## References

Key inspiration and background from:

* Cao & Yang, *Towards Making Systems Forget with Machine Unlearning*
* Ginart et al., *Making AI Forget You*
* Bourtoule et al., *Machine Unlearning*
* Recent work on **Federated Unlearning**

## References

Key inspiration and background from:

* Pillutla et al., *Descent-to-Delete: Gradient-Based Methods for Machine Unlearning* (NeurIPS 2021)
* Cao & Yang, *Towards Making Systems Forget with Machine Unlearning* (IEEE S&P 2015)
* Ginart et al., *Making AI Forget You: Data Deletion in Machine Learning* (NeurIPS 2019)
* Bourtoule et al., *Machine Unlearning* (IEEE S&P 2021)
* Recent work on **Federated Unlearning**

## Authors

**Nina Abeyratne**
BSc (Hons) Artificial Intelligence & Data Science
Individual Research Project

**Nipuna Senanayake**
MSc(Computer Science, USA)
B.Sc. (Hons) (Kelaniya)
Senior Lecturer - Grade II
Informatics Institute of Technology (IIT).