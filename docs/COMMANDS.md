# Commands

## Setup

```bash
uv sync
```

### macOS XGBoost runtime

XGBoost requires the OpenMP runtime on macOS. If `import xgboost` fails with a missing `libomp.dylib`, install:

```bash
brew install libomp
```

## Tests

```bash
uv run pytest -q
```

## Sprint 0 audit smoke command

```bash
uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml --sample
```

The former placeholder `gcn_minimal.yaml --debug` commands are not a current
training workflow. As of Sprint 4 Slice 1, `gcn_minimal.yaml` declares the
locked headline guide-level protocol and must not be used for random-edge or
debug performance reporting.

## Sprint 2 split artifacts

```bash
uv run python scripts/build_splits.py --config configs/data/mak2022.yaml
```

## Sprint 2 feature catalog

```bash
uv run python scripts/build_features.py --config configs/data/mak2022.yaml
```

## Sprint 2 dummy and Logistic Regression baselines

```bash
uv run python scripts/train.py --config configs/experiments/baseline_logistic_regression.yaml
```

This command also writes Logistic Regression diagnostics under:

```text
outputs/sprint2/diagnostics/
```

## Sprint 2 XGBoost baselines

```bash
uv run python scripts/train.py --config configs/experiments/baseline_xgboost.yaml
```

This command also writes:

```text
outputs/sprint2/diagnostics/xgboost_feature_column_audit.csv
outputs/sprint2/diagnostics/xgboost_feature_importance.csv
```

## Sprint 2 tabular MLP baselines

```bash
uv run python scripts/train.py --config configs/experiments/baseline_mlp.yaml
```

This command also writes:

```text
outputs/sprint2/diagnostics/tabular_mlp_training_summary.csv
outputs/sprint2/diagnostics/tabular_mlp_feature_column_audit.csv
```

## Sprint 2 sequence-only CNN/BiLSTM baselines

```bash
uv run python scripts/train.py --config configs/experiments/sequence_cnn_bilstm.yaml
```

This command also writes:

```text
outputs/sprint2/diagnostics/sequence_input_audit.csv
outputs/sprint2/diagnostics/sequence_training_summary.csv
```

## Sprint 2 CNN + F3/F4 late-fusion baselines

```bash
uv run python scripts/train.py --config configs/experiments/sequence_cnn_late_fusion.yaml
```

This command also writes:

```text
outputs/sprint2/diagnostics/sequence_late_fusion_input_audit.csv
outputs/sprint2/diagnostics/sequence_late_fusion_training_summary.csv
```

## Sprint 3 graph construction and leakage-control artifacts

```bash
uv run python scripts/build_graph.py --config configs/data/mak2022.yaml --schema-config configs/sweeps/graph_schema_ablation.yaml
```

This constructs dependency-light Graph A/B/C tables and manifests under:

```text
data/processed/graphs/sprint3/
```

The canonical tracked handoff report is:

```text
outputs/sprint3/graph_schema_report.md
```

This command does not train graph models and does not require PyTorch Geometric.

## Sprint 4 graph materialization foundation validation

Sprint 4 Slice 1 adds the minimum PyTorch Geometric dependency required to
materialize validated Sprint 3 typed artifacts as strict-inductive
`HeteroData` views. It does not train a graph model.

```bash
uv sync
uv run pytest -q tests/test_graph_loader.py tests/test_graph_leakage.py tests/test_config_loads.py
uv run ruff check scripts src tests
uv run pytest -q
```

The materialization path consumes existing typed tables and manifests under:

```text
data/processed/graphs/sprint3/
```

It must not rebuild graph topology or rewrite Sprint 3 artifacts.

## Sprint 4 Colab runner infrastructure validation

Slice 4A prepares the Colab runner workflow only. It does not execute a full
Graph A GPU run and does not produce final Sprint 4 performance claims.

Validate a copied Sprint 3 graph artifact set before any headline Colab
training:

```bash
uv run python scripts/validate_graph_artifacts.py --artifact-dir data/processed/graphs/sprint3 --approved-source drive_sprint3_handoff --output outputs/sprint4/graph_a/<run_id>/graph_artifact_provenance.json
```

This command consumes serialized Sprint 3 graph tables and manifests through
the Sprint 4 loader, validates the frozen graph contract, and records SHA256
checksums for the artifact files. It must pass before a Graph A Colab result is
eligible for headline reporting.

Colab setup pattern:

```bash
pip install uv
git clone <repository-url> crispr-gnn-offtarget
cd crispr-gnn-offtarget
git checkout <approved-sprint4-branch-or-commit>
uv sync
uv run python -c "import torch, torch_geometric; print('torch', torch.__version__); print('pyg', torch_geometric.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda', torch.version.cuda)"
```

Colab Pro may be used for GPU capacity, but reported runs must still record the
commit SHA, resolved config, random seed, device/runtime, graph schema, feature
bundle, split ID, visibility policy, output paths, and artifact provenance.

The local canonical Graph A training command is:

```bash
uv run python scripts/train.py --config configs/experiments/gcn_minimal.yaml
```

Run this command only after local Slice 1-3 validation and the artifact
provenance gate pass. It writes canonical Sprint 4 outputs. Use separate
reviewed debug output paths for smoke checks; do not use canonical headline
paths for throwaway Colab experiments.

For the Colab full run, keep the base config unchanged and run against the
run-specific resolved config produced by the runner:

```bash
uv run python scripts/train.py --config outputs/sprint4/graph_a/<run_id>/resolved_config.yaml
```

The resolved config records runtime-only choices such as `run_id` and
`training.device` without mutating `configs/experiments/gcn_minimal.yaml`.

After validated Graph A and Graph C returned artifacts exist locally, generate
the consolidated same-contract Sprint 4 comparison with:

```bash
uv run python scripts/compare_sprint4_gcn.py --output-root outputs/sprint4
```

This command validates required Graph A/C result, diagnostic, figure, and
provenance artifacts before writing:

```text
outputs/sprint4/gcn_sprint4_comparison_results.csv
outputs/sprint4/gcn_sprint4_comparison_report.md
outputs/sprint4/figures/gcn_sprint4_schema_auprc_comparison.png
outputs/sprint4/figures/gcn_sprint4_pr_curves.png
```

## Later sprint command pattern

```bash
uv run python scripts/<script>.py --config configs/<config>.yaml
```
