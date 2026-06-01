# Colab Runner Notes

Colab is a GPU runner only. Final preprocessing, graph loading, model,
training, evaluation, plotting, and reporting logic must live in this
repository under `src/`, `scripts/`, and `configs/`.

This project may use Colab Pro for longer GPU sessions, but Colab Pro is a
runtime capacity choice, not a scientific workflow change. Reported results
must still record the exact git commit, resolved config, runtime, graph
artifact provenance, and output locations.

## Runner Boundary

Notebook cells may:

- clone or update the repository;
- mount Google Drive;
- install/sync dependencies using the documented `uv` workflow;
- copy raw data or Sprint 3 graph artifacts from Drive to Colab local disk;
- run repository commands with `uv run ...`;
- copy generated outputs back to Google Drive.

Notebook cells must not:

- define model classes;
- rebuild graph topology or scientific preprocessing;
- implement final evaluation, metrics, plotting, or report generation;
- edit labels, split manifests, graph visibility, or experiment configs;
- hide dependency workarounds that are not documented in the repository.

## Suggested Drive Layout

Use a project-specific Drive folder and copy artifacts to Colab local disk for
training I/O.

```text
/content/drive/MyDrive/crispr_gnn_offtarget/
  data/processed/graphs/sprint3/
  data/raw/
  returned_outputs/
```

The exact Drive root can differ, but it must be recorded in the run notes. Do
not commit Drive-specific paths, copied raw data, processed graph parquet
files, checkpoints, or Colab caches.

## Setup Pattern

Run these commands from Colab cells. Replace repository URL, branch, and Drive
paths with the approved values for the run.

```bash
pip install uv
git clone <repository-url> crispr-gnn-offtarget
cd crispr-gnn-offtarget
git checkout <approved-sprint4-branch-or-commit>
uv sync
```

Confirm the runtime and dependency versions before any headline run:

```bash
uv run python -c "import torch, torch_geometric; print('torch', torch.__version__); print('pyg', torch_geometric.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda', torch.version.cuda)"
```

The approved dependency source is `pyproject.toml` plus `uv.lock`. Do not rely
on an undocumented notebook-only `pip install torch-geometric` workaround. If
Colab requires a CUDA/PyG-specific workaround, document it in
`docs/DECISIONS.md` and `docs/COMMANDS.md` before using the run as a headline
result.

## Copy Inputs To Local Disk

Copy the required Sprint 3 artifacts from Drive to the repository-local path
expected by `configs/experiments/gcn_minimal.yaml`.

```bash
mkdir -p data/processed/graphs
cp -r /content/drive/MyDrive/crispr_gnn_offtarget/data/processed/graphs/sprint3 data/processed/graphs/
```

If raw data is needed for a separate documented command, copy it in the same
explicit way. The current Graph A training path consumes the Sprint 3 typed
artifacts rather than rebuilding graph topology from raw records.

## Pre-Training Graph Artifact Provenance Gate

Before any headline Graph A training, validate the copied Sprint 3 artifacts
and write a checksum/provenance record:

```bash
mkdir -p outputs/sprint4/graph_a/<run_id>
uv run python scripts/validate_graph_artifacts.py \
  --artifact-dir data/processed/graphs/sprint3 \
  --approved-source drive_sprint3_handoff \
  --output outputs/sprint4/graph_a/<run_id>/graph_artifact_provenance.json
```

This command must pass before a Graph A run can be treated as a valid headline
run. It validates the frozen loader contract and records SHA256 checksums for
the copied artifact files. A result produced without this record is provisional
or debug-only.

## Graph A Training Command

Only run this after local Slice 1-3 tests have passed and the provenance gate
above has succeeded:

```bash
uv run python scripts/train.py --config outputs/sprint4/graph_a/<run_id>/resolved_config.yaml
```

The Colab runner should leave `configs/experiments/gcn_minimal.yaml`
unchanged. It creates a run-specific `resolved_config.yaml` under
`outputs/sprint4/graph_a/<run_id>/`, applying only runtime fields such as
`run_id` and `training.device`. This keeps the repository base config stable
while preserving the exact config used by the reported run. Do not use the
canonical command as an informal smoke command unless output paths are
explicitly redirected to non-canonical debug locations in a reviewed config.

## Graph C Runner Workflow

Graph C is the primary context-enriched comparison after the validated Graph A
run. Use `colab/sprint4_graph_c_runner.ipynb` for the manual Colab run. The
notebook is runner-only and must not contain model, preprocessing, evaluation,
or plotting logic.

Graph C uses the same frozen Sprint 2/Sprint 3 contract as Graph A:

- label scheme: `scheme_a`;
- split ID: `sprint2_main_seed42`;
- visibility policy: `strict_inductive_primary`;
- loss: `weighted_bce`;
- headline protocol: `headline_guide_level`;
- schema: `graph_c_context_observation`.

Graph C changes both topology and target semantics/context representation
relative to Graph A. It must not be reported as a topology-only experiment.

The Graph C runner creates a run-specific resolved config from
`configs/experiments/gcn_graph_c.yaml` under:

```text
outputs/sprint4/graph_c/<run_id>/resolved_config.yaml
```

The full Graph C training command must be:

```bash
uv run python scripts/train.py --config outputs/sprint4/graph_c/<run_id>/resolved_config.yaml
```

Run this command only after the provenance gate below passes. Do not mutate the
base config in place.

## Graph C Pre-Training Provenance Gate

Before headline Graph C training, validate the copied Sprint 3 artifacts and
write a checksum/provenance record:

```bash
mkdir -p outputs/sprint4/graph_c/<run_id>
uv run python scripts/validate_graph_artifacts.py \
  --artifact-dir data/processed/graphs/sprint3 \
  --approved-source drive_sprint3_handoff \
  --output outputs/sprint4/graph_c/<run_id>/graph_artifact_provenance.json
```

The runner must then assert the Graph C schema counts from the provenance JSON:

```text
graph_c_context_observation:
  sgRNA = 150
  target_observation = 11446
  candidate_pair = 11446
  context_similar_to = 91754
  split_id = sprint2_main_seed42
  label_scheme = scheme_a
  visibility_policy = strict_inductive_primary
```

A Graph C run without this provenance record and count validation is
provisional/debug-only and must not enter headline reporting.

## Required Returned Artifacts

After a real Colab run, copy these outputs back to durable Drive storage:

```text
outputs/sprint4/graph_a/gcn_graph_a_results.csv
outputs/sprint4/graph_a/gcn_graph_a_report.md
outputs/sprint4/graph_a/diagnostics/
outputs/sprint4/graph_a/figures/
outputs/sprint4/graph_a/<run_id>/graph_artifact_provenance.json
outputs/sprint4/graph_a/<run_id>/resolved_config.yaml
outputs/sprint4/graph_a/<run_id>/runtime.json
outputs/sprint4/graph_a/<run_id>/training_history.csv
outputs/sprint4/graph_a/<run_id>/model.pt
```

After a real Graph C Colab run, copy these outputs back to durable Drive
storage:

```text
outputs/sprint4/graph_c/gcn_graph_c_results.csv
outputs/sprint4/graph_c/gcn_graph_c_report.md
outputs/sprint4/graph_c/diagnostics/
outputs/sprint4/graph_c/figures/
outputs/sprint4/graph_c/<run_id>/graph_artifact_provenance.json
outputs/sprint4/graph_c/<run_id>/resolved_config.yaml
outputs/sprint4/graph_c/<run_id>/runtime.json
outputs/sprint4/graph_c/<run_id>/training_history.csv
outputs/sprint4/graph_c/<run_id>/model.pt
```

The Graph C `figures/` directory must include:

```text
gcn_graph_c_graph_schema_auprc_comparison.png
gcn_graph_c_pr_curves.png
gcn_graph_c_roc_curves.png
gcn_graph_c_training_curves.png
gcn_graph_c_score_distributions.png
gcn_graph_c_confusion_matrices.png
gcn_graph_c_decile_lift.png
gcn_graph_c_per_genome_metrics.png
gcn_graph_c_view_sanity_example.png
```

The Graph C `diagnostics/` directory must include:

```text
gcn_graph_c_predictions.csv
gcn_graph_c_training_history.csv
gcn_graph_c_score_direction.csv
gcn_graph_c_fixed_threshold_metrics.csv
gcn_graph_c_score_deciles.csv
gcn_graph_c_per_genome_metrics.csv
gcn_graph_c_test_per_guide_metrics.csv
```

Large run artifacts and checkpoints should normally remain untracked and live
in Drive. Final report tables and figures may be tracked according to the
repository artifact policy after Slice 4C validates that they came from a real
run and preserve the Sprint 2/Sprint 3 contracts. The same policy applies to
Graph C after its returned-artifact validation slice.

## Graph B Runner Workflow

Graph B is the bounded secondary control after validated Graph A and Graph C runs. Use
`colab/sprint4_graph_b_runner.ipynb` for the manual Colab run. The notebook is runner-only
and must not contain model, preprocessing, evaluation, or plotting logic.

Graph B inherits Graph A's frozen contract:

- label scheme: `scheme_a`;
- split ID: `sprint2_main_seed42`;
- visibility policy: `strict_inductive_primary`;
- loss: `weighted_bce`;
- headline protocol: `headline_guide_level`;
- schema: `graph_b_guide_similarity_control`;
- target node representation: `zero_type_feature` (featureless physical targets).

The only structural addition relative to Graph A is deterministic, label-free
guide-sequence similarity edges (`sequence_similar_to`, 1208 edges). Graph B must not
be treated as a primary result or described as equivalent to Graph C's topology+semantics
change.

The Graph B runner creates a run-specific resolved config from
`configs/experiments/gcn_graph_b.yaml` under:

```text
outputs/sprint4/graph_b/<run_id>/resolved_config.yaml
```

The full Graph B training command must be:

```bash
uv run python scripts/train.py --config outputs/sprint4/graph_b/<run_id>/resolved_config.yaml
```

Run this command only after the provenance gate below passes. Do not mutate the base config.

## Graph B Pre-Training Provenance Gate

Before headline Graph B training, validate the copied Sprint 3 artifacts and write a
checksum/provenance record:

```bash
mkdir -p outputs/sprint4/graph_b/<run_id>
uv run python scripts/validate_graph_artifacts.py \
  --artifact-dir data/processed/graphs/sprint3 \
  --approved-source drive_sprint3_handoff \
  --output outputs/sprint4/graph_b/<run_id>/graph_artifact_provenance.json
```

The runner must then assert the Graph B schema counts from the provenance JSON:

```text
graph_b_guide_similarity_control:
  sgRNA = 150
  physical_target_site = 9880
  candidate_pair = 11446
  sequence_similar_to = 1208
  split_id = sprint2_main_seed42
  label_scheme = scheme_a
  visibility_policy = strict_inductive_primary
```

A Graph B run without this provenance record and count validation is provisional/debug-only
and must not enter headline reporting.

## Required Returned Graph B Artifacts

After a real Graph B Colab run, copy these outputs back to durable Drive storage:

```text
outputs/sprint4/graph_b/gcn_graph_b_results.csv
outputs/sprint4/graph_b/gcn_graph_b_report.md
outputs/sprint4/graph_b/diagnostics/
outputs/sprint4/graph_b/figures/
outputs/sprint4/graph_b/<run_id>/graph_artifact_provenance.json
outputs/sprint4/graph_b/<run_id>/resolved_config.yaml
outputs/sprint4/graph_b/<run_id>/runtime.json
outputs/sprint4/graph_b/<run_id>/training_history.csv
outputs/sprint4/graph_b/<run_id>/model.pt
```

The Graph B `figures/` directory must include:

```text
gcn_graph_b_graph_schema_auprc_comparison.png
gcn_graph_b_pr_curves.png
gcn_graph_b_roc_curves.png
gcn_graph_b_training_curves.png
gcn_graph_b_score_distributions.png
gcn_graph_b_confusion_matrices.png
gcn_graph_b_decile_lift.png
gcn_graph_b_per_genome_metrics.png
gcn_graph_b_view_sanity_example.png
```

The Graph B `diagnostics/` directory must include:

```text
gcn_graph_b_predictions.csv
gcn_graph_b_training_history.csv
gcn_graph_b_score_direction.csv
gcn_graph_b_fixed_threshold_metrics.csv
gcn_graph_b_score_deciles.csv
gcn_graph_b_per_genome_metrics.csv
gcn_graph_b_test_per_guide_metrics.csv
```

## Returned Artifact Validation

Slice 4C must inspect returned Graph A artifacts before any final claim. At
minimum it must verify:

- `graph_artifact_provenance.json` exists and was generated before training;
- result rows use `sprint2_main_seed42`, Scheme A, strict-inductive visibility,
  and Graph A;
- required diagnostics and all Sprint 4 figure files exist under
  `outputs/sprint4/graph_a/figures/`;
- no Graph C or Graph B run has been mixed into the Graph A report;
- test diagnostics were not used to revise training choices.

For Graph C, returned-artifact validation must additionally verify:

- `graph_artifact_provenance.json` exists and was generated before training;
- provenance records `graph_c_context_observation` with the expected node and
  relation counts;
- result rows use `sprint2_main_seed42`, Scheme A, strict-inductive visibility,
  and Graph C;
- required diagnostics and all Graph C figure files exist under
  `outputs/sprint4/graph_c/`;
- Graph C is described as changing both topology and target semantics/context
  representation;
- test diagnostics were not used to revise training choices.
