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
| Sprint 5 | Epigenetic feature ablation (main novelty) | 🔜 Next |
| Sprint 6 | Imbalance-method comparison | ⏳ Planned |
| Sprint 7 | GAT/GATv2 architecture | ⏳ Planned |

## Sprint 4 Results

All three GCN schemas were trained on Google Colab GPU under the frozen Sprint 2/3 contract (`scheme_a`, `sprint2_main_seed42`, `strict_inductive_primary`). Primary metric is AUPRC (threshold-free, appropriate for ~90% positive prevalence). None beats the XGBoost F4 reference on primary AUPRC.

| Model | Schema | Test AUPRC | Test AUROC | Test MCC | vs F4 XGBoost |
|---|---|---:|---:|---:|---|
| `xgboost_unweighted` | F4 tabular baseline | **0.9925** | 0.9384 | 0.345 | — |
| `gcn_graph_a` | Graph A — minimal physical-target | 0.9663 | 0.7451 | 0.301 | does not beat |
| `gcn_graph_b` *(secondary control)* | Graph B — + guide-similarity topology | 0.9667 | 0.7436 | 0.127† | does not beat |
| `gcn_graph_c` | Graph C — observation-level context | 0.9616 | 0.7599 | 0.454 | does not beat |

†Graph B MCC is low due to threshold sensitivity at high positive prevalence; AUPRC is the primary metric. Graph C must not be interpreted as a topology-only experiment — it changes both topology and target semantics relative to Graph A.

Consolidated comparison: `outputs/sprint4/gcn_sprint4_comparison_results.csv`

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

## Documentation

- Evaluation protocol and metric rationale: `docs/EVALUATION_PROTOCOL.md`
- All major decisions with reasoning: `docs/DECISIONS.md`
- Reproducible commands for each sprint: `docs/COMMANDS.md`
- Project context and sprint direction: `docs/PROJECT_CONTEXT.md`
- Completed execution plans: `docs/exec-plans/completed/`
