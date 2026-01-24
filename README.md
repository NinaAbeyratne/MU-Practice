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

Each unlearning method is evaluated using:

* **Deleted data loss** (privacy effectiveness)
* **Retained data loss / accuracy** (utility preservation)
* **Unlearning time** (efficiency)
* **Confidence metrics** (where applicable)

All experiment results are saved in **structured JSON format** to enable reproducible comparison across methods.

##  Repository Structure

```text
.
├── train.py          # Model training utilities
├── unlearn.py        # Unlearning method implementations
├── evaluate.py       # Evaluation and metrics
├── utils.py          # Shared helper functions
├── requirements.txt  # Python dependencies
├── notebooks/
│   ├── gradient_ascent_and_gold_model.ipynb
│   ├── first_epoch_reversal.ipynb
│   ├── comparison_analysis.ipynb
│   ├── data/         # (ignored) datasets
│   ├── models/       # (ignored) saved checkpoints
│   └── results/      # experiment JSON outputs
└── README.md
```

> ⚠️ Dataset files and trained models results are intentionally excluded from version control.

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

Experiments are primarily conducted through Jupyter notebooks:

* **Exact retraining & gradient ascent:**
  `notebooks/gradient_ascent_and_gold_model.ipynb`

* **Checkpoint-based unlearning:**
  `notebooks/first_epoch_reversal.ipynb`

* **Method comparison & visualization:**
  `notebooks/comparison_analysis.ipynb`

Each notebook produces a JSON file capturing experiment metadata and evaluation metrics.

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

* Centralized machine unlearning baselines
* - Federated learning setup (FL)
* - Federated unlearning methods
* - Privacy–efficiency trade-off analysis
* - Thesis-level evaluation and reporting

## References

Key inspiration and background from:

* Cao & Yang, *Towards Making Systems Forget with Machine Unlearning*
* Ginart et al., *Making AI Forget You*
* Bourtoule et al., *Machine Unlearning*
* Recent work on **Federated Unlearning**

## Author

**Nina Abeyratne**
BSc (Hons) Artificial Intelligence & Data Science
Individual Research Project

## Supervisor

**Nipuna Senanayake**
MSc(Computer Science, USA)
B.Sc. (Hons) (Kelaniya)
Senior Lecturer - Grade II
Informatics Institute of Technology (IIT).