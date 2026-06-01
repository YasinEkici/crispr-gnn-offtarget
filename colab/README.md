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

Large run artifacts and checkpoints should normally remain untracked and live
in Drive. Final report tables and figures may be tracked according to the
repository artifact policy after Slice 4C validates that they came from a real
run and preserve the Sprint 2/Sprint 3 contracts.

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
