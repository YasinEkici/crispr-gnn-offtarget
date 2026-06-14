# Updated Project Folder Structure (uv-first, no Makefile)

Project: **Context-Aware GNN for CRISPR-Cas9 Off-Target Prediction**

This document defines repository layout, dependency management, and tooling decisions:

> **Coding agents work in the GitHub repository. Colab is only a runner for training and experiments. Google Drive stores large datasets and run artifacts. Markdown literature files are agent-readable context, not training data. Dependencies are managed with `uv`, using `pyproject.toml` + `uv.lock` as the source of truth. Commands are documented as plain `uv run ...` examples instead of using a Makefile.**

The goal is not to build a heavy ML platform. The goal is to keep a capstone/research project reproducible, navigable, and easy for both humans and coding agents to work on.

> **Companion document:** Scientific decisions, sprint plan, and methodology live in `CRISPR_GNN_PROJECT_PLAN.md`. This document focuses on *how* to organize code; the plan focuses on *what* to build and *why*.

---

## 0. High-Level Workflow

```text
GitHub repository
    code + configs + docs + literature notes + tests

Google Drive
    large datasets + checkpoints + large run outputs

Colab
    temporary GPU runtime that clones repo and runs commands

Markdown literature layer
    agent-readable research context; not part of model training
```

### Main Rule

Do not write core project code inside Colab notebooks. Colab should execute commands from the repo:

```python
!pip install uv

!git clone https://github.com/<user>/crispr-gnn-offtarget.git
%cd crispr-gnn-offtarget
!uv sync

!uv run python scripts/prepare_data.py --config configs/data/mak2022.yaml
!uv run python scripts/train.py --config configs/experiments/gcn_minimal.yaml
!uv run python scripts/evaluate.py --run outputs/sprint4/graph_a/runs/gcn_minimal
```

---

## 1. Design Principles

### 1. Small `AGENTS.md`, deeper docs elsewhere

`AGENTS.md` should be a short map, not a giant manual. It should tell agents where to look next.

### 2. Repository-local knowledge is the source of truth

Project decisions should live in versioned repo files, not only in chats, Google Docs, or notebook comments.

Important docs:

```text
docs/PROJECT_CONTEXT.md
docs/EXPERIMENT_PLAN.md
docs/EVALUATION_PROTOCOL.md
docs/DATASET_AUDIT.md
docs/LABEL_SCHEMES.md
docs/FEATURE_PARSING.md
docs/DECISIONS.md
docs/COMMANDS.md
```

### 3. Colab is a runner, not the development environment

Colab notebooks should:

- install dependencies,
- clone/pull the repo,
- copy data from Drive to local Colab disk,
- run scripts,
- copy outputs back to Drive.

They should not contain core model logic.

### 4. Literature Markdown is context, not code

Paper notes should help agents understand the project. They should not be parsed during model training.

### 5. Reproducible experiments over ad-hoc notebooks

Every experiment should have:

- a YAML config,
- saved split files,
- saved metrics,
- saved predictions,
- a documented label scheme,
- a documented graph schema.

### 6. Use `uv` for dependency management

Use `uv` rather than a manually maintained `requirements.txt`. The project dependency source of truth is:

```text
pyproject.toml
uv.lock
```

Rules:

- Commit `pyproject.toml` and `uv.lock`.
- Do not manually edit `uv.lock`.
- Add dependencies with `uv add ...`.
- Add dev dependencies with `uv add --dev ...`.
- Use `uv run ...` for scripts and tests.
- Generate `requirements.txt` only if a specific external environment requires it.
- Treat PyTorch/PyG GPU installation as a documented special-case setup note when a GPU runner needs it.
- Do not require a Makefile; use documented `uv run ...` commands so the workflow stays OS-friendly.

### 7. Keep the project lightweight

No heavy CI/CD, no production deployment, no large infrastructure. Use simple checks:

```bash
uv run pytest -q
uv run python scripts/train.py --config configs/experiments/gcn_minimal.yaml --debug --max-epochs 1
```

---

## 2. Recommended Repository Layout

```text
crispr-gnn-offtarget/
├── AGENTS.md
├── README.md
├── CRISPR_GNN_PROJECT_PLAN.md      # scientific plan (companion to this file)
├── PROJECT_FOLDER_STRUCTURE.md     # this file
├── pyproject.toml                  # dependency source of truth
├── uv.lock                         # committed lockfile
├── .python-version                 # optional but recommended
├── .gitignore
├── .env.example
│
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── PROJECT_CONTEXT.md
│   ├── RESOURCES.md
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENT_PLAN.md
│   ├── DATASET_AUDIT.md            # created in Sprint 1
│   ├── DATA_SCHEMA.md
│   ├── LABEL_SCHEMES.md            # created in Sprint 1
│   ├── FEATURE_PARSING.md          # created in Sprint 1
│   ├── EVALUATION_PROTOCOL.md
│   ├── MODEL_NOTES.md
│   ├── MODEL_CARDS.md
│   ├── DECISIONS.md
│   ├── COMMANDS.md
│   │
│   ├── literature/
│   │   ├── literature_index.md
│   │   ├── paper_registry.yaml
│   │   └── papers/
│   │       ├── 2021_vinodkumar_gcn_crispr/
│   │       │   ├── paper.md
│   │       │   ├── metadata.yaml
│   │       │   ├── original.pdf
│   │       │   └── assets/
│   │       │       ├── figures/
│   │       │       ├── tables/
│   │       │       └── equations/
│   │       │
│   │       ├── 2022_mak_epigenetic_descriptors/
│   │       │   ├── paper.md
│   │       │   ├── metadata.yaml
│   │       │   ├── original.pdf
│   │       │   └── assets/
│   │       │
│   │       ├── 2021_stortz_crisprsql/
│   │       │   ├── paper.md
│   │       │   ├── metadata.yaml
│   │       │   ├── original.pdf
│   │       │   └── assets/
│   │       │
│   │       ├── 2025_crisprofft/
│   │       │   ├── paper.md
│   │       │   ├── metadata.yaml
│   │       │   ├── original.pdf
│   │       │   └── assets/
│   │       │
│   │       └── 2025_graph_crispr/
│   │           ├── paper.md
│   │           ├── metadata.yaml
│   │           ├── original.pdf
│   │           └── assets/
│   │
│   ├── thesis/
│   │   ├── README.md
│   │   ├── notes/
│   │   │   ├── main_narrative_framing.md
│   │   │   ├── btu_template_verification.md
│   │   │   └── tez_yazim_meta_kurallari.md
│   │   ├── templates/
│   │   │   └── btu_docx/
│   │   │       └── BTU_BM_Tez_Yazım_Sablonu_2019.docx
│   │   └── latex/
│   │       └── btu_template/
│   │           ├── main.tex
│   │           ├── metadata.tex
│   │           ├── btu-thesis.cls
│   │           ├── chapters/
│   │           ├── appendices/
│   │           └── figures/
│   │
│   └── exec-plans/
│       ├── active/
│       ├── completed/
│       └── tech-debt.md
│
├── configs/
│   ├── data/
│   │   ├── mak2022.yaml
│   │   ├── crisprsql.yaml
│   │   └── crisprofft.yaml
│   │
│   ├── experiments/
│   │   ├── baseline_xgboost.yaml
│   │   ├── baseline_mlp.yaml
│   │   ├── sequence_cnn_bilstm.yaml
│   │   ├── gcn_minimal.yaml
│   │   ├── gcn_enriched.yaml
│   │   ├── gat_enriched.yaml
│   │   ├── gatv2_enriched.yaml
│   │   └── sage_ablation.yaml
│   │
│   └── sweeps/
│       ├── epigenetic_ablation.yaml
│       ├── graph_schema_ablation.yaml
│       └── loss_ablation.yaml
│
├── data/
│   ├── raw/                  # not committed
│   ├── interim/              # not committed
│   ├── processed/            # not committed
│   ├── splits/               # can commit small split manifests if safe
│   ├── sample/               # tiny sample data only
│   └── README.md
│
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_dataset_audit.ipynb
│   ├── 02_label_threshold_sensitivity.ipynb
│   ├── 03_feature_missingness.ipynb
│   ├── 04_train_baseline_runner.ipynb
│   ├── 05_train_gcn_runner.ipynb
│   ├── 06_train_gat_runner.ipynb
│   └── 07_result_analysis.ipynb
│
├── scripts/
│   ├── prepare_data.py
│   ├── audit_dataset.py
│   ├── build_splits.py
│   ├── build_graph.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_ablation.py
│   ├── convert_papers.py
│   └── export_report_tables.py
│
├── src/
│   └── crispr_gnn/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── schemas.py
│       │   ├── load_mak2022.py
│       │   ├── load_crisprsql.py
│       │   ├── load_crisprofft.py
│       │   ├── parsers.py
│       │   ├── preprocessing.py
│       │   ├── labels.py
│       │   └── splits.py
│       │
│       ├── features/
│       │   ├── sequence_encoding.py
│       │   ├── mismatch_features.py
│       │   ├── epigenetic_features.py
│       │   ├── nucleosome_features.py
│       │   └── similarity_edges.py
│       │
│       ├── graph/
│       │   ├── graph_builder.py
│       │   ├── graph_schemas.py
│       │   ├── minimal_bipartite.py
│       │   ├── enriched_graph.py
│       │   ├── hetero_graph.py
│       │   └── pyg_dataset.py
│       │
│       ├── models/
│       │   ├── tabular_baselines.py
│       │   ├── sequence_baseline.py
│       │   ├── gcn.py
│       │   ├── gat.py
│       │   ├── gatv2.py
│       │   ├── graphsage.py
│       │   ├── hetero_gnn.py
│       │   └── losses.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── samplers.py
│       │   ├── callbacks.py
│       │   └── seed.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── thresholding.py
│       │   ├── guide_level_eval.py
│       │   ├── leakage_checks.py
│       │   └── reports.py
│       │
│       └── utils/
│           ├── logging.py
│           ├── paths.py
│           └── io.py
│
├── tests/
│   ├── test_labels.py
│   ├── test_splits.py
│   ├── test_no_leakage.py
│   ├── test_graph_builder.py
│   ├── test_metrics.py
│   ├── test_config_loads.py
│   └── test_smoke_train.py
│
├── outputs/
│   ├── sprint1/
│   ├── sprint2/
│   │   ├── diagnostics/
│   │   └── figures/
│   ├── sprint3/
│   ├── sprint4/
│   │   └── graph_a/
│   │       ├── diagnostics/
│   │       ├── figures/
│   │       └── runs/         # not committed
│   ├── splits/
│   └── features/
│
└── colab/
    ├── README.md
    ├── 00_setup_and_mount_drive.ipynb
    ├── 01_run_dataset_audit.ipynb
    ├── 02_run_training.ipynb
    └── 03_copy_outputs_to_drive.ipynb
```

---

## 3. Folder Responsibilities

### Root Files

#### `README.md`

Human-facing project introduction.

Should include:

- one-paragraph project explanation,
- setup instructions,
- common commands,
- dataset acquisition note,
- current sprint status.

#### `CRISPR_GNN_PROJECT_PLAN.md`

Scientific plan: research question, contribution, dataset strategy, label schemes, sprint plan, evaluation protocol, do-not-overclaim list. Companion to this structure document.

#### `PROJECT_FOLDER_STRUCTURE.md` (this file)

Repository layout, dependency management, Colab workflow, configs, scripts, and tests. Companion to the project plan.

#### `PROJECT_CONTEXT.md`

Short, stable project context for agents.

Should explain:

- what the project is,
- main dataset choice,
- main novelty,
- current model sequence,
- what is must-have vs stretch.

#### `AGENTS.md`

Short coding-agent map.

Should not contain all literature details. It should point to the right docs.

Example:

```md
# AGENTS.md

This repository implements a context-aware GNN for CRISPR-Cas9 off-target prediction.

## Read first

1. PROJECT_CONTEXT.md
2. CRISPR_GNN_PROJECT_PLAN.md
3. docs/EXPERIMENT_PLAN.md
4. docs/DATASET_AUDIT.md
5. docs/EVALUATION_PROTOCOL.md
6. docs/literature/literature_index.md

## Ground rules

- Colab is not the development environment. Core logic must go under src/.
- Notebooks are runners or exploratory analysis only.
- Do not change label definitions without updating docs/DATASET_AUDIT.md and docs/DECISIONS.md.
- Do not report random-split results as final results.
- Final evaluation must include guide-level split and AUPRC.
- Test set must contain only measured=1 rows (see docs/DATASET_AUDIT.md).
- Every experiment must have a config under configs/.
- Every run must save config.yaml, metrics.json, and predictions.csv.
- Use `uv sync` for setup and `uv run ...` for commands.
- Do not manually edit `uv.lock`.
- Do not use `pip install` inside the repo except for documented Colab/GPU workarounds.
- Run `uv run pytest -q` and one smoke train before claiming completion.
- Do not assume `make` is available; prefer explicit `uv run ...` commands.

## Dataset reference

Primary dataset path: `data/raw/260520_putative_nucleosomal.parquet`
For verified dimensions, schema, and access methodology, see `docs/DATASET_AUDIT.md`.
```

---

## 4. Dependency Management with `uv`

This project should use `uv` instead of a hand-maintained `requirements.txt`.

### Source of truth

```text
pyproject.toml
uv.lock
```

`requirements.txt` is not part of the normal workflow. If a hosted environment requires it, generate it from the lockfile and document why it exists.

### Initial setup

```bash
uv sync
```

### Add dependencies

```bash
uv add pandas pyarrow numpy scikit-learn pyyaml tqdm rich matplotlib duckdb
uv add --dev pytest ruff ipykernel
```

### Run commands

```bash
uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml
uv run pytest -q
```

### Initial dependency split

Sprint 1 kept dependencies lightweight. Sprint 2 later added `xgboost` for boosted-tree baselines and `torch` for non-graph sequence baselines. PyTorch Geometric remains deferred until graph-model implementation.

```toml
[project]
name = "crispr-gnn-offtarget"
version = "0.1.0"
description = "Context-aware GNN for CRISPR-Cas9 off-target prediction"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "pyyaml",
    "tqdm",
    "rich",
    "matplotlib",
    "duckdb",
]

[dependency-groups]
dev = [
    "pytest",
    "ruff",
    "ipykernel",
]
```

### Torch / PyG note

PyTorch is allowed for Sprint 2 non-graph sequence baselines as documented in `docs/DECISIONS.md`. Add PyTorch Geometric later, when graph-model training starts. CUDA/Colab may need a documented GPU-specific install workaround. Keep that workaround in `docs/DECISIONS.md` or `colab/README.md`; do not make ad-hoc `pip install` commands the default repo setup.

---

## 5. Docs Structure

### `docs/PROJECT_OVERVIEW.md`

High-level explanation of the research project.

### `docs/RESOURCES.md`

Paper and dataset links.

### `docs/EXPERIMENT_PLAN.md`

Model and experiment order:

```text
Mak et al. 2022 / crisprSQL-derived dataset
    ↓
Dataset + label audit
    ↓
Tabular baseline + sequence DL baseline
    ↓
Minimal bipartite graph
    ↓
GCN baseline
    ↓
Graph enrichment ablation
    ↓
Epigenetic feature ablation
    ↓
Imbalance-aware training
    ↓
GAT / GATv2
    ↓
Optional CRISPRoffT external validation
```

### `docs/DATA_SCHEMA.md`

Document fields such as:

- guide sequence,
- target sequence,
- PAM,
- mismatch count,
- cleavage activity,
- binary label,
- epigenetic features (experimental scalars + computed string arrays),
- cell line,
- measured flag,
- split group.

### `docs/DATASET_AUDIT.md`

Created in Sprint 1.

Must include:

- row count,
- unique guide count,
- unique target count,
- positive/negative ratio per scheme,
- chosen label scheme,
- threshold sensitivity table,
- missingness of epigenetic features,
- measured=1 vs measured=0 distribution,
- final chosen dataset version,
- dataset access methodology,
- citation strategy.

### `docs/LABEL_SCHEMES.md`

Created in Sprint 1.

Must include:

- detailed definitions of label schemes A, B, C, D,
- outlier handling rules for NaN, negative, and >1 `cleavage_freq` values,
- Box-Cox reproduction details for the paper comparison track.

### `docs/FEATURE_PARSING.md`

Created in Sprint 1.

Must include:

- parser code and strategy for the string-formatted 23-element computed feature arrays,
- the three dimensionality strategies (position-resolved, aggregated, PAM-focused) for Sprint 5 ablation.

### `docs/EVALUATION_PROTOCOL.md`

Must define:

- random split is debug only,
- guide-level split is final,
- no guide leakage,
- test set contains only measured=1 rows,
- AUPRC is primary metric,
- AUROC is secondary,
- same split required for fair model comparison,
- two-track strategy: paper reproduction (regression, Spearman/Pearson) + binary classification (AUPRC).

### `docs/DECISIONS.md`

Dated decision log.

Example:

```md
## 2026-04-29 — Use Mak et al. 2022 as Phase 1 dataset

Reason: more controlled than CRISPRoffT, includes epigenetic/nucleosome features,
suitable for first GCN/GAT pipeline.

Alternatives considered:
- raw crisprSQL
- CRISPRoffT full database

Decision: use Mak 2022 first, CRISPRoffT as stretch external validation.
```

---

## 6. Literature Markdown Layer

Literature files should be organized per paper, not as one giant Markdown file.

### Recommended Structure

```text
docs/literature/
├── literature_index.md
├── paper_registry.yaml
└── papers/
    ├── 2021_vinodkumar_gcn_crispr/
    │   ├── paper.md
    │   ├── metadata.yaml
    │   ├── original.pdf
    │   └── assets/
    ├── 2022_mak_epigenetic_descriptors/
    │   ├── paper.md
    │   ├── metadata.yaml
    │   ├── original.pdf
    │   └── assets/
    └── 2025_crisprofft/
        ├── paper.md
        ├── metadata.yaml
        ├── original.pdf
        └── assets/
```

### `literature_index.md`

Agent-readable map of the literature.

Example:

```md
# Literature Index

## Vinodkumar et al. 2021 — GCN-CRISPR

Role: Main GCN/link prediction baseline.

Project usage:
- graph construction
- GCN baseline
- edge prediction setup

Related modules:
- src/crispr_gnn/graph/
- src/crispr_gnn/models/gcn.py
- configs/experiments/gcn_minimal.yaml

## Mak et al. 2022 — Epigenetic descriptors

Role: Main epigenetic feature source AND primary dataset.

Project usage:
- feature selection
- epigenetic ablation
- data schema
- label scheme reference

Related modules:
- src/crispr_gnn/features/epigenetic_features.py
- src/crispr_gnn/data/load_mak2022.py
- src/crispr_gnn/data/parsers.py
- docs/DATA_SCHEMA.md
- docs/DATASET_AUDIT.md
- docs/LABEL_SCHEMES.md
```

### `paper_registry.yaml`

Machine-readable paper list.

Example:

```yaml
papers:
  - id: vinodkumar_2021_gcn_crispr
    title: Prediction of sgRNA Off-Target Activity using GCN
    year: 2021
    role: gcn_baseline
    priority: must_read
    local_md: docs/literature/papers/2021_vinodkumar_gcn_crispr/paper.md
    local_pdf: docs/literature/papers/2021_vinodkumar_gcn_crispr/original.pdf

  - id: mak_2022_epigenetic
    title: Comprehensive computational analysis of epigenetic descriptors affecting CRISPR-Cas9 off-target activity
    year: 2022
    role: epigenetic_features_and_primary_dataset
    priority: must_read
    local_md: docs/literature/papers/2022_mak_epigenetic_descriptors/paper.md
    local_pdf: docs/literature/papers/2022_mak_epigenetic_descriptors/original.pdf
```

### Per-Paper `paper.md` Template

```md
---
title: ""
year:
authors: ""
source_url: ""
doi: ""
project_role: ""
conversion_status: "pending"
---

# Abstract

# Key Contributions

# Dataset

# Features

# Methods

# Metrics

# Useful for Our Project

# Figures and Tables

# Notes / Risks
```

### Important Rule

`paper.md` is not the original source. It is an agent-readable representation.

Always keep:

```text
original.pdf
metadata.yaml
assets/
```

If formulas, tables, or figures cannot be converted reliably, keep them as assets and reference them in `paper.md`.

Example:

```md
Equation 3 could not be reliably converted to LaTeX.
Original location: original.pdf, page 7.
Image: assets/equations/equation_3.png
```

---

## 7. Data Layout

```text
data/
├── raw/
├── interim/
├── processed/
├── splits/
├── sample/
└── README.md
```

### `data/raw/`

Original downloaded files (e.g., `260520_putative_nucleosomal.parquet`).

Do not commit.

### `data/interim/`

Partially cleaned intermediate files.

Do not commit unless tiny.

### `data/processed/`

Model-ready tables or graph objects.

Do not commit large files.

### `data/splits/`

Train/validation/test split manifests.

Small split manifest files may be committed if they do not contain sensitive or huge data.

### `data/sample/`

Tiny sample data for tests and smoke runs.

This can be committed.

### `data/README.md`

Should explain:

- how to download the primary dataset (note any access workarounds),
- where to place it (`data/raw/`),
- expected file name and size,
- how to run preprocessing,
- how to reproduce processed files,
- backup mirror locations if any.

---

## 8. Colab Workflow

Colab should clone the repo, copy data locally, run scripts, and copy outputs back to Drive.

### Recommended Colab Command Pattern

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install uv

!git clone https://github.com/<user>/crispr-gnn-offtarget.git
%cd crispr-gnn-offtarget
!uv sync

# Copy processed dataset from Drive to fast local Colab disk
!mkdir -p data/processed
!cp /content/drive/MyDrive/crispr_project/data/processed/mak2022.parquet data/processed/

# Train
!uv run python scripts/train.py --config configs/experiments/gat_enriched.yaml

# Save outputs back to Drive
!mkdir -p /content/drive/MyDrive/crispr_project/outputs/sprint7/runs
!cp -r outputs/sprint7/runs/gat_enriched /content/drive/MyDrive/crispr_project/outputs/sprint7/runs/
```

### Colab Rules

Do:

```text
✅ use Colab for GPU training
✅ run scripts from repo with `uv run ...`
✅ keep outputs backed up to Drive
✅ use notebooks as runners
```

Do not:

```text
❌ write model classes inside Colab notebooks
❌ keep final preprocessing logic inside notebooks
❌ train by manually editing cells for each experiment
❌ use `pip install -r requirements.txt` as the default setup
❌ parse literature Markdown during training
```

---

## 9. Configs

Every experiment should be config-driven.

Example:

```yaml
experiment_name: gcn_minimal
seed: 42

data:
  dataset: mak2022
  dataset_path: data/processed/mak2022.parquet
  label_scheme: <chosen_in_sprint_1>      # final scheme set after audit
  split_path: data/splits/guide_level_seed42.json

features:
  sequence: true
  mismatch: true
  experimental_epigenetic: false
  computed_nucleosome: false
  binding_energy: false

graph:
  schema: minimal_bipartite
  similarity_edges: false

model:
  name: gcn
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2

training:
  loss: weighted_bce
  batch_size: 1024
  max_epochs: 100
  patience: 10

split_rules:
  test_measured_only: true
  val_measured_preferred: true

metrics:
  primary: auprc
  secondary:
    - auroc
    - f1
    - mcc
```

---

## 10. Core Scripts

### `scripts/prepare_data.py`

Loads raw data and creates processed data.

### `scripts/audit_dataset.py`

Creates dataset audit tables.

Outputs:

```text
outputs/sprint1/dataset_audit.md
outputs/sprint1/label_threshold_sensitivity.md
outputs/sprint1/feature_missingness.md
```

### `scripts/build_splits.py`

Builds random and guide-level splits. Enforces measured=1 test rule.

### `scripts/build_graph.py`

Builds graph objects (Graph A/B/C from project plan Section 8).

### `scripts/train.py`

Runs one experiment from a config.

Every run should save:

```text
outputs/<sprint>/<schema_label>/runs/<run_id>/config.yaml
outputs/<sprint>/<schema_label>/runs/<run_id>/metrics.json
outputs/<sprint>/<schema_label>/runs/<run_id>/predictions.csv
outputs/<sprint>/<schema_label>/runs/<run_id>/model.pt
```

### `scripts/evaluate.py`

Evaluates one run or compares multiple runs.

### `scripts/convert_papers.py`

Optional helper for converting papers to Markdown notes.

This should not be part of model training.

---

## 11. Tests

Keep tests lightweight but targeted at project-breaking risks.

```text
tests/test_labels.py
tests/test_splits.py
tests/test_no_leakage.py
tests/test_graph_builder.py
tests/test_metrics.py
tests/test_config_loads.py
tests/test_smoke_train.py
```

### Most Important Tests

#### `test_labels.py`

Checks that label schemes behave as expected.

#### `test_no_leakage.py`

Checks that train/test guide IDs do not overlap in guide-level split, AND that test set contains only measured=1 rows.

#### `test_metrics.py`

Checks AUPRC/AUROC calculation.

#### `test_smoke_train.py`

Runs one tiny training job on `data/sample/`.

---

## 12. Command Reference Instead of Makefile

We do **not** require a Makefile. A Makefile can be convenient on Linux/macOS, but it can add friction on Windows and is unnecessary because `uv` already gives us a consistent command runner.

Use plain commands in `docs/COMMANDS.md`, `README.md`, and agent execution plans. This keeps the workflow cross-platform and easy to copy into local terminals, Colab, or coding-agent instructions.

### Recommended commands

```bash
# Install / sync environment
uv sync

# Dataset audit
uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml

# Build splits
uv run python scripts/build_splits.py --config configs/data/mak2022.yaml

# Build graph
uv run python scripts/build_graph.py --config configs/experiments/gcn_enriched.yaml

# Train baselines
uv run python scripts/train.py --config configs/experiments/baseline_xgboost.yaml
uv run python scripts/train.py --config configs/experiments/sequence_cnn_bilstm.yaml

# Train graph models
uv run python scripts/train.py --config configs/experiments/gcn_minimal.yaml
uv run python scripts/train.py --config configs/experiments/gat_enriched.yaml

# Evaluate
uv run python scripts/evaluate.py --latest

# Tests
uv run pytest -q

# Smoke run
uv run python scripts/train.py --config configs/experiments/gcn_minimal.yaml --debug --max-epochs 1
```

### Optional aliases

If someone personally wants shortcuts, they can define shell aliases locally. Do not make them required for the project.

Examples:

```bash
alias crispr-test="uv run pytest -q"
alias crispr-audit="uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml"
```

### Rule

The canonical command format is:

```bash
uv run python scripts/<script>.py --config configs/<config>.yaml
```

---

## 13. Agent-Oriented Execution Plans

For complex tasks, create an execution plan in:

```text
docs/exec-plans/active/
```

Example:

```text
docs/exec-plans/active/001-dataset-audit.md
docs/exec-plans/active/002-guide-level-split.md
docs/exec-plans/active/003-gcn-minimal.md
```

When finished, move it to:

```text
docs/exec-plans/completed/
```

### Execution Plan Template

```md
# Exec Plan: <task name>

## Goal

<one-sentence goal>

## Inputs

- <required configs / docs / data files>

## Steps

1. <step>
2. <step>
3. <step>

## Acceptance Criteria

- <expected output file 1> exists.
- <expected output file 2> exists.
- relevant docs are updated.
- `uv run pytest -q` passes.
```

---

## 14. Implementation Order

```text
1. Create repo scaffold
2. Add AGENTS.md and PROJECT_CONTEXT.md
3. Add docs/ and literature index skeleton
4. Add configs/data/mak2022.yaml
5. Add dataset audit script
6. Implement label schemes
7. Implement feature parsers
8. Implement guide-level split
9. Add leakage tests
10. Add tabular baseline
11. Add sequence DL baseline
12. Build minimal bipartite graph
13. Train GCN minimal
14. Add enriched graph edges
15. Run epigenetic ablation
16. Run imbalance-loss comparison
17. Train GAT/GATv2
18. Optional: CRISPRoffT external validation
19. Optional: GraphSAGE / hetero-GNN ablation
```

---

## 15. Risk-to-Structure Mapping

```text
Risk: label ambiguity
→ src/crispr_gnn/data/labels.py
→ docs/DATASET_AUDIT.md
→ docs/LABEL_SCHEMES.md
→ tests/test_labels.py

Risk: guide leakage or putative-row leakage into test set
→ src/crispr_gnn/data/splits.py
→ tests/test_no_leakage.py
→ docs/EVALUATION_PROTOCOL.md

Risk: GNN value unclear
→ src/crispr_gnn/graph/minimal_bipartite.py
→ src/crispr_gnn/graph/enriched_graph.py
→ configs/sweeps/graph_schema_ablation.yaml

Risk: weak baseline
→ src/crispr_gnn/models/sequence_baseline.py
→ configs/experiments/sequence_cnn_bilstm.yaml

Risk: feature parsing errors
→ src/crispr_gnn/data/parsers.py
→ docs/FEATURE_PARSING.md

Risk: experiment confusion
→ configs/experiments/*.yaml
→ outputs/<sprint>/<schema_label>/runs/<run_id>/config.yaml

Risk: agent/human gets lost
→ AGENTS.md
→ CRISPR_GNN_PROJECT_PLAN.md
→ PROJECT_CONTEXT.md
→ docs/literature/literature_index.md
→ docs/DECISIONS.md
```

---

## 16. What Not To Do

```text
❌ Do not put core model logic in notebooks.
❌ Do not commit large datasets.
❌ Do not use random split as final result.
❌ Do not change label threshold silently.
❌ Do not include measured=0 rows in test set.
❌ Do not compare GCN and GAT on different splits.
❌ Do not claim epigenetic features help without an ablation.
❌ Do not train on CRISPRoffT before the Mak/crisprSQL pipeline is stable.
❌ Do not maintain `requirements.txt` by hand; use `pyproject.toml` + `uv.lock`.
❌ Do not require Makefile-based commands; use explicit `uv run ...` commands.
❌ Do not manually edit `uv.lock`.
❌ Do not turn AGENTS.md into a giant research document.
❌ Do not convert all papers into one huge Markdown file.
❌ Do not contradict CRISPR_GNN_PROJECT_PLAN.md on scientific decisions; resolve in favor of the plan.
```

---

## 17. Final Recommended Setup

```text
GitHub repo
    code, configs, docs, paper notes, tests, pyproject.toml, uv.lock,
    CRISPR_GNN_PROJECT_PLAN.md, PROJECT_FOLDER_STRUCTURE.md

uv
    dependency manager and command runner for local/Colab workflows

Google Drive
    large data, model checkpoints, run artifacts

Colab
    runner notebooks only

AGENTS.md
    short navigation map

docs/
    source of truth (operational protocols, decisions, schemas)

docs/literature/
    agent-readable literature layer

src/crispr_gnn/
    core project code

configs/
    reproducible experiments

tests/
    lightweight guardrails

CRISPR_GNN_PROJECT_PLAN.md
    scientific plan (research question, methodology, sprints)

PROJECT_FOLDER_STRUCTURE.md
    this file (organizational layout, tooling, workflow)
```

This structure is intentionally simple but robust enough for a capstone that uses coding agents, Colab training, and literature-heavy bioinformatics context.
