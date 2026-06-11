# Context-Aware GNN for CRISPR-Cas9 Off-Target Prediction — BTU CENG Spring '26 Senior Project

Repository for a two-person capstone project on epigenetic-context-aware GNNs for CRISPR-Cas9 off-target prediction.

The primary dataset is the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset. Large datasets, checkpoints, and generated run artifacts are not committed to git.

## Project Status

| Sprint | Description | Status |
|---|---|---|
| Sprint 1 | Dataset audit, label policy, feature parsing policy | ✅ Complete |
| Sprint 2 | Non-graph and sequence DL baselines, locked guide-level split | ✅ Complete |
| Sprint 3 | Graph A/B/C artifact construction and leakage controls | ✅ Complete |
| Sprint 4 | GCN baseline training — Graph A, B (control), C | ✅ Complete |
| Sprint 5 | Graph A feature-family ablation + Graph C energy sensitivity | ✅ Complete |
| Sprint 6 | Imbalance, threshold, and loss comparison | ✅ Complete (Slices 0–4; Slice 5 optional) |
| Sprint 7 | GAT/GATv2 architecture, Graph C mechanism ablations, target-context encoder | ✅ Complete |
| Sprint 8 | Model improvement — target-context + context-edge interaction (8A), CRISPR-Net-adapted sequence context (8B) | 🟡 8A complete; 8B planned |
| Sprint 9 | Robustness — guide-level bootstrap CIs, paired-difference, multi-seed variance | ⏳ Optional / stretch |

## Current Results

All reported model runs use the frozen Sprint 2/3 contract (`scheme_a`, `sprint2_main_seed42`, measured-only headline rows, `experiment_id=18` excluded, `strict_inductive_primary`). Primary metric is AUPRC (threshold-free, appropriate for ~90% positive prevalence). No GNN result beats the XGBoost F4 reference on primary AUPRC, but Sprint 7F/8A narrowed the gap and improved rare-negative operating-point metrics with Graph C GATv2 target-context modelling.

| Model | Setting | Test AUPRC | Test AUROC | Test Macro F1 | Test MCC |
|---|---|---:|---:|---:|---|
| `xgboost_unweighted` | F4 tabular baseline | **0.9925** | 0.9384 | 0.6427 | 0.3452 |
| `gatv2_graph_c_sprint7f_exp_emphasis` | Graph C + `S5F2_energy` + family-aware target-context encoder, experimental emphasis | 0.9849 | 0.9266 | 0.7772 | 0.5681 |
| `S8A_R2_context_edge_film` | Sprint 8A validation-selected candidate: Graph C GATv2 + head-only FiLM context-edge interaction | 0.9828 | 0.9106 | 0.7780 | 0.5637 |
| `gatv2_graph_c_sprint7f_family_aware` | Graph C + `S5F2_energy` + family-aware target-context encoder | 0.9821 | 0.9066 | **0.8017** | **0.6035** |
| `gcn_graph_a_sprint6_wbce` | Graph A + `S5F2_energy` + weighted BCE | 0.9769 | 0.8200 | 0.6989 | 0.4837 |
| `gcn_graph_c_sprint5b_energy` | Graph C + `S5F2_energy` | 0.9725 | 0.8362 | 0.5524 | 0.2743 |
| `gcn_graph_a` | Sprint 4 Graph A — minimal physical target | 0.9663 | 0.7451 | 0.6021 | 0.3008 |
| `gcn_graph_b` *(secondary control)* | Sprint 4 Graph B — guide-similarity topology | 0.9666 | 0.7436 | 0.4918 | 0.1266† |
| `gcn_graph_c` | Sprint 4 Graph C — observation-level context | 0.9616 | 0.7599 | 0.6776 | 0.4537 |

†Threshold-dependent metrics are interpretation-only. Several GCN runs have high AUPRC but weak negative-class recognition under the validation-selected threshold; Sprint 5B Graph C + `S5F2_energy` has TN/FP/FN/TP `14/155/0/1533`. Sprint 6 established weighted BCE as the fixed loss policy, and Sprint 7 showed that Graph C GATv2 target-context modeling can improve rare-negative recognition while keeping the same frozen evaluation contract.

Sprint 5 takeaway: binding-energy features (`energy_1`-`energy_5`) are the strongest GCN feature-family signal. Raw experimental epigenetic scalars and computed context features do not improve the current Graph A GCN when appended as candidate-edge feature tables. Graph C must not be interpreted as a topology-only experiment — it changes both topology and target semantics relative to Graph A.

Sprint 7 takeaway: edge-aware GAT/GATv2 on Graph A did not beat the weighted-BCE Graph A GCN reference, but Graph C GATv2 exposed a useful target-context signal. Sprint 7D/7E isolated direct `target_observation` features, especially experimental epigenetic features, as the critical mechanism; Sprint 7F's family-aware target-context encoder produced the strongest same-contract GNN results so far.

Sprint 8A takeaway: the predeclared 5-run target-context/context-edge interaction sprint selected `S8A_R2_context_edge_film` by validation AUPRC (`0.9875`). R2 improved the Sprint 8A operating point, but no Sprint 8A run beat the XGBoost F4 AUPRC bar; Slice 7 HP refinement was skipped to avoid post-result overtuning, and superiority/variance claims are deferred to Sprint 9 robustness.

Key reports:

- Sprint 4 comparison: `outputs/sprint4/gcn_sprint4_comparison_report.md`
- Sprint 5 Graph A feature ablation: `outputs/sprint5/graph_a_feature_ablation/sprint5_graph_a_feature_ablation_report.md`
- Sprint 5B Graph C energy sensitivity: `outputs/sprint5b/graph_c/gcn_graph_c_report.md`
- Sprint 6 loss/imbalance comparison: `outputs/sprint6/loss_comparison/sprint6_loss_comparison_report.md`
- Sprint 7F target-context encoder: `outputs/sprint7f/target_context_encoder_report.md`
- Sprint 8A target-context interaction: `outputs/sprint8a/target_context_interaction_report.md`

## Setup

```bash
uv sync
```

This project uses `pyproject.toml` and `uv.lock`. Do not maintain a manual `requirements.txt`.

## Key Commands

```bash
# Tests
uv run pytest -q

# Sprint 2 baselines
uv run python scripts/train.py --config configs/experiments/baseline_xgboost.yaml

# Sprint 3 graph construction
uv run python scripts/build_graph.py --config configs/data/mak2022.yaml --schema-config configs/sweeps/graph_schema_ablation.yaml

# Sprint 4 GCN training (Graph A — minimal baseline)
uv run python scripts/train.py --config configs/experiments/gcn_minimal.yaml

# Sprint 4 GCN training (Graph C — context-enriched)
uv run python scripts/train.py --config configs/experiments/gcn_graph_c.yaml

# Sprint 4 GCN training (Graph B — bounded secondary control)
uv run python scripts/train.py --config configs/experiments/gcn_graph_b.yaml

# Sprint 4 consolidated comparison (regenerate after new runs)
uv run python scripts/compare_sprint4_gcn.py --output-root outputs/sprint4

# Sprint 5 Graph A feature artifact build
uv run python scripts/build_sprint5_graph_a_features.py \
  --data-config configs/data/mak2022.yaml \
  --schema-config configs/sweeps/graph_schema_ablation.yaml \
  --artifact-dir data/processed/graphs/sprint5 \
  --report-path outputs/sprint5/graph_a_feature_ablation_artifact_report.md

# Sprint 5 Graph A feature ablation sweep
uv run python scripts/run_sprint5_feature_ablation.py \
  --config configs/sweeps/sprint5_graph_a_feature_ablation.yaml \
  --run-id sprint5_graph_a_feature_ablation_seed42_<timestamp>

# Sprint 5B Graph C energy sensitivity
uv run python scripts/build_sprint5b_graph_c_energy_features.py \
  --data-config configs/data/mak2022.yaml \
  --schema-config configs/sweeps/graph_schema_ablation.yaml \
  --source-artifact-dir data/processed/graphs/sprint3 \
  --artifact-dir data/processed/graphs/sprint5b \
  --report-path outputs/sprint5b/graph_c_energy_sensitivity_artifact_report.md
uv run python scripts/train.py --config configs/sweeps/sprint5b_graph_c_energy_sensitivity.yaml
```

For full GPU training, use the Colab runner notebooks under `colab/`. See `docs/COMMANDS.md` for the complete command reference.

## Data

Place the primary raw dataset at:

```text
data/raw/260520_putative_nucleosomal.parquet
```

Raw, interim, and processed data are gitignored. Use `data/sample/` for tiny test fixtures only.

## Key Artifact Locations

### Sprint 2
- Results: `outputs/sprint2/baseline_results.csv`
- Report: `outputs/sprint2/baseline_report.md`
- Split manifest: `outputs/splits/sprint2_guides.json`
- Feature catalog: `outputs/features/sprint2_feature_catalog.md`

### Sprint 3
- Graph schema report: `outputs/sprint3/graph_schema_report.md`
- Typed graph tables and manifests: `data/processed/graphs/sprint3/` *(gitignored)*
- Schema config: `configs/sweeps/graph_schema_ablation.yaml`

### Sprint 4
- Graph A results: `outputs/sprint4/graph_a/gcn_graph_a_results.csv`
- Graph C results: `outputs/sprint4/graph_c/gcn_graph_c_results.csv`
- Graph B results: `outputs/sprint4/graph_b/gcn_graph_b_results.csv`
- Consolidated comparison: `outputs/sprint4/gcn_sprint4_comparison_results.csv`
- Comparison figures: `outputs/sprint4/figures/`
- Run provenance (per schema): `outputs/sprint4/<schema>/<run_id>/graph_artifact_provenance.json`
- Model checkpoints: `outputs/sprint4/<schema>/<run_id>/model.pt` *(gitignored)*

### Sprint 5
- Graph A feature-ablation report: `outputs/sprint5/graph_a_feature_ablation/sprint5_graph_a_feature_ablation_report.md`
- Graph A feature-ablation results: `outputs/sprint5/graph_a_feature_ablation/sprint5_graph_a_feature_ablation_results.csv`
- Graph A diagnostics and figures: `outputs/sprint5/graph_a_feature_ablation/diagnostics_sprint5_graph_a/`, `outputs/sprint5/graph_a_feature_ablation/figures_sprint5_graph_a/`
- Sprint 5B Graph C report: `outputs/sprint5b/graph_c/gcn_graph_c_report.md`
- Sprint 5B Graph C results: `outputs/sprint5b/graph_c/gcn_graph_c_results.csv`
- Sprint 5B diagnostics and figures: `outputs/sprint5b/graph_c/diagnostics/`, `outputs/sprint5b/graph_c/figures/`
- Model checkpoints: Drive returned-output folders only; do not commit `model.pt`

### Sprint 8A
- Target-context interaction comparison: `outputs/sprint8a/target_context_interaction_comparison.csv`
- Target-context interaction report: `outputs/sprint8a/target_context_interaction_report.md`
- Diagnostics and figures: `outputs/sprint8a/diagnostics/`, `outputs/sprint8a/figures/`
- Run manifest/provenance: `outputs/sprint8a/target_context_interaction_run_manifest.json`, `outputs/sprint8a/graph_artifact_provenance.json`

## Documentation

- Evaluation protocol and metric rationale: `docs/EVALUATION_PROTOCOL.md`
- All major decisions with reasoning: `docs/DECISIONS.md`
- Reproducible commands for each sprint: `docs/COMMANDS.md`
- Project context and sprint direction: `docs/PROJECT_CONTEXT.md`
- Completed execution plans: `docs/exec-plans/completed/`
