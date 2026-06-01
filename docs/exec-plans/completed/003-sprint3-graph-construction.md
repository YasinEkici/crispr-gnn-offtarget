# Execution Plan: Sprint 3 Graph Construction + Leakage Control

## 1. Goal

Construct deterministic, leakage-controlled graph datasets for later Sprint 4 graph-model training. Sprint 3 produces graph artifacts, validation tests, and a schema report only; it does not train or evaluate a GNN.

The primary schemas are:

- `Graph A`: minimal sgRNA-to-physical-target graph with row-varying context on candidate edges.
- `Graph C`: context-observation graph with label-free context-similarity relations.
- `Graph B`: guide-sequence similarity enrichment retained as a bounded secondary control.

## 2. Inputs

- Dataset: Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome data.
- Dataset config: `configs/data/mak2022.yaml`.
- Label: Scheme A, `cleavage_freq > 1e-5`.
- Locked split manifest: `outputs/splits/sprint2_guides.json` (`sprint2_main_seed42`).
- Baseline reference: `xgboost_unweighted / F4` from `outputs/sprint2/baseline_report.md`.
- Feature and preprocessing contract: `src/crispr_gnn/features/tabular.py`, `src/crispr_gnn/features/sequence.py`, and `outputs/features/sprint2_feature_catalog.md`.
- Policy sources: `docs/EVALUATION_PROTOCOL.md`, `docs/LABEL_SCHEMES.md`, `docs/FEATURE_PARSING.md`, and `docs/DECISIONS.md`.

## 3. Frozen Sprint 2 Contract

- Exclude NaN `cleavage_freq` from supervised graph edges.
- Keep negative `cleavage_freq` as negative Scheme A labels; keep values greater than 1 positive without clipping.
- Reuse the locked guide-level split without rebalancing.
- Keep main train, validation, and test graph edges measured-only and exclude `experiment_id=18`.
- Do not use held-out labels, scores, or test diagnostics for construction or schema selection.
- Fit imputers, scalers, and any learned similarity transformations from training data only.
- Sprint 4 must report AUPRC and positive prevalence and compare to `xgboost_unweighted / F4`.

## 4. Scope

- Define typed dependency-light graph tables and manifests.
- Build Graph A, B, and C artifacts from the locked supervised universe.
- Map Sprint 2 sequence and F1-F4 features into graph feature tables without changing the feature policy.
- Implement strict-inductive topology visibility and leakage audits.
- Generate `outputs/sprint3/graph_schema_report.md`.
- Document the graph schema and evaluation decisions.

## 5. Out Of Scope

- GCN, GAT, GraphSAGE, heterogeneous GNN, or any graph-model training.
- Adding PyTorch Geometric or selecting a Sprint 4 architecture.
- Tuning schema parameters against test performance.
- Rebuilding the locked split, changing Scheme A, or adding putative-negative rows to main artifacts.
- Full 299-dimensional position-resolved computed feature inputs.

## 6. Schema Definitions

### Graph A: Minimal Physical-Target Graph

- `sgRNA` nodes keyed by `grna_target_id`, with guide-only sequence features.
- `physical_target_site` nodes keyed by `genome`, `target_chr`, `target_start`, `target_end`, and `target_strand`.
- `candidate_pair` relations keyed by source row `id`, labelled by Scheme A and assigned the locked split.
- Row-varying target sequence, mismatch, binding energy, and F3/F4 context remain edge feature bundles.

### Graph B: Guide-Similarity Control

- Inherit Graph A candidate relations and features.
- Add `sgRNA --sequence_similar_to--> sgRNA` from guide sequence Hamming distance only.
- Use deterministic `top_k=5`; remove self-links, symmetrize, and deduplicate.
- Train topology uses train guides only; validation/test queries may connect only to training guides.

### Graph C: Context-Observation Graph

- `sgRNA` nodes follow Graph A.
- `target_observation` nodes are keyed by source row `id`, retaining target-side sequence and context features.
- `candidate_pair` relations retain pairwise mismatch and binding energy features and Scheme A labels.
- Add `target_observation --context_similar_to--> target_observation` from F3/F4 context only.
- Context imputation and standardization are fit on training observations only; deterministic `top_k=5` similarity uses Euclidean distance.
- Validation/test observations may connect only to training observations.

Graph C changes target semantics from physical loci to assay/context observations; it must be interpreted separately from Graph A.

## 7. Feature And Representation Mapping

- `S1` supplies pair/sequence information without silently adding identifiers or labels.
- `F1` and `F2` remain pairwise edge inputs.
- `F3` and `F4` remain edge-level on Graph A because repeated physical target coordinates have row-varying context.
- Graph C places context on observation nodes and pairwise mismatch/energy on candidate edges.
- F4 continues to use aggregated computed nucleosome features and missingness indicators, with train-only imputation.
- Full position-resolved computed features remain deferred.

## 8. Expected Files

New implementation and artifact files:

- `src/crispr_gnn/graph/graph_schemas.py`
- `src/crispr_gnn/graph/graph_builder.py`
- `scripts/build_graph.py`
- `configs/sweeps/graph_schema_ablation.yaml`
- `tests/test_graph_builder.py`
- `tests/test_graph_leakage.py`
- `outputs/sprint3/graph_schema_report.md`

Updated supporting files:

- `src/crispr_gnn/graph/__init__.py`
- `docs/DECISIONS.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/COMMANDS.md`
- `README.md` or `docs/PROJECT_CONTEXT.md` after validated artifact generation.

Graph tables are generated under ignored `data/processed/graphs/sprint3/`. Dependency files do not change in Sprint 3.

## 9. Implementation Steps

1. Validate that the input rows reproduce the locked, measured-only Sprint 2 main universe.
2. Define typed table schemas, deterministic serialization, manifests, and graph configuration.
3. Build Graph A and validate physical-node/edge-level context placement.
4. Preserve Sprint 2 F4 train-only preprocessing state in graph feature preparation.
5. Build Graph B guide-sequence similarity edges under strict-inductive visibility.
6. Build Graph C observation nodes and train-fitted, label-free context-similarity edges.
7. Run leakage tests for labels, splits, predictive columns, preprocessing, and topology visibility.
8. Emit graph schema manifests and the canonical Sprint 3 report.
9. Hand off typed artifacts for later `HeteroData` loading in Sprint 4.

## 10. Risks

- Shared physical targets can leak row-varying context if context is attached to Graph A physical nodes.
- Similarity edges can expose held-out information if topology is created across validation/test observations or selected from test behavior.
- Global preprocessing would leak validation/test distributions.
- Graph C differs semantically from Graph A and cannot be described as topology-only enrichment.
- Graph benefit may be overclaimed if it is not compared against the strong F4 XGBoost baseline.
- Guide-size, genome composition, and high test prevalence remain interpretation risks.

## 11. Acceptance Criteria

- Graph A, B, and C are built deterministically from the locked main universe.
- Labels, splits, measured-only membership, and experiment exclusion are asserted automatically.
- Feature tables exclude identifiers and outcome fields as predictive inputs.
- Graph A leaves row-varying context off physical target nodes.
- Graph B and C similarity relations are label-free and strict-inductive.
- F4 preprocessing is train-fit only and recorded in manifests.
- The schema report documents counts, caveats, visibility rules, and Sprint 4 handoff.
- No graph model is trained and no graph dependency is added.

## 12. Commands

```bash
uv sync
uv run ruff check scripts src tests
uv run pytest -q
uv run python scripts/build_graph.py --config configs/data/mak2022.yaml --schema-config configs/sweeps/graph_schema_ablation.yaml
uv run pytest -q
```

The existing split and feature builders may be rerun for contract reproduction checks, but must not redefine the frozen evaluation policy.

## 13. Documentation Updates

- Record physical-target identity, context placement, strict-inductive visibility, Graph B role, and deferred PyG loading in `docs/DECISIONS.md`.
- Add graph visibility and comparison rules to `docs/EVALUATION_PROTOCOL.md`.
- Add the reproducible graph command and output locations to `docs/COMMANDS.md`.
- Update project status after the validated graph artifacts and report are generated.
