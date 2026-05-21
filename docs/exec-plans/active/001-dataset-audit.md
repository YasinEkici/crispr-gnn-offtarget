# Exec Plan: Sprint 1 Dataset + Label Audit

## Goal

Independently audit the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset from the local raw file, compare all computed audit statistics against the expected reference values from earlier exploratory inspection, finalize label and outlier policy, finalize computed-feature parser behavior, and define split/reporting constraints before any real model training begins.

Dataset source and access are already resolved. Sprint 1 is not a source-discovery task. The dataset is the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset originally published at:

```text
https://crisprsql.com/downloads/260520_putative_nucleosomal.parquet.gz
```

That original URL is permanently unreachable with 404 while the rest of `crisprsql.com` remains available. The project dataset was retrieved from the Internet Archive Wayback Machine snapshot of that exact URL, and that Wayback snapshot is the single working source for this project. Citation must cite Mak et al. 2022, BMC Genomics, DOI `10.1186/s12864-022-09012-7`, CC-BY 4.0, plus the Wayback snapshot URL.

Sprint 1 should answer:

- Does the local raw file independently reproduce the expected reference snapshot statistics, or are there discrepancies?
- Which observed rows, labels, and outliers are safe to use for binary off-target prediction?
- Is `cleavage_freq` the available raw label column, and is a transformed `CA` column empirically absent or present?
- How are the 19 epigenetic/nucleosome-context features actually represented in the file?
- Can the 13 computed nucleosome features be parsed as string-formatted 23-element arrays, and how should missing/malformed cases be handled?
- What split and reporting constraints must later model sprints obey, especially the `measured=1` test rule?

## Inputs

- `CRISPR_GNN_PROJECT_PLAN.md`
- `PROJECT_FOLDER_STRUCTURE.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/DATASET_AUDIT.md`
- `docs/LABEL_SCHEMES.md`
- `docs/FEATURE_PARSING.md`
- `docs/DECISIONS.md`
- `docs/literature/literature_index.md`
- `docs/literature/paper_registry.yaml`
- Mak et al. 2022 paper notes under `docs/literature/axes/axis_3_epigenetics_crispr_activity/3A_epigenetic_ml_crispr/2022_mak_crispr_cas9_off_target/`
- crisprSQL paper notes under `docs/literature/axes/axis_3_epigenetics_crispr_activity/3A_epigenetic_ml_crispr/2021_stortz_crisprsql_database_platform_off_target/`
- Dataset config: `configs/data/mak2022.yaml`
- Expected local raw dataset path: `data/raw/260520_putative_nucleosomal.parquet`
- Settled source: Internet Archive Wayback snapshot of `https://crisprsql.com/downloads/260520_putative_nucleosomal.parquet.gz`

Reference values to reproduce and confirm from the raw file:

- 310,142 rows x 45 columns.
- 25,632 rows with `measured=1`; 284,510 rows with `measured=0`.
- 154 unique sgRNAs.
- 138,747 unique target locations, using proposed key `target_chr + target_start + target_end + target_strand`; audit must state the exact key used.
- 7 cell lines; approximately 14,108 rows missing `cell_line`, concentrated in `experiment_id=18`.
- 5 genomes: `hg19` approximately 244K and dominant, `rn5` approximately 51K, `hg38` approximately 7K, plus `mm10` and `mm9`.
- Approximately 15,153 rows missing computed nucleosome features, approximately 4.9%.
- 19 epigenetic features expected present:
  - 6 experimental scalar features: `epigen_ctcf`, `epigen_dnase`, `epigen_rrbs`, `epigen_h3k4me3`, `epigen_drip`, `MNase`.
  - 13 computed nucleosome features expected as string-formatted 23-element arrays: `GCContent`, `WSScore`, `YRScore`, `NucleotideBDM`, `StrongWeakBDM`, `NuPoP_Occup_147_human`, `NuPoP_Viterbi_147_human`, `NuPoP_Affinity_147_human`, `nuCpos_Occup_147_yeast`, `nuCpos_Viterbi_147_yeast`, `nuCpos_Affinity_147_yeast`, `VanDerHeijden`, `LeNupH3Q85C`.
  - 5 binding-energy scalar features: `energy_1` through `energy_5`.
- Label column expected to be `cleavage_freq`, raw frequency, not paper-transformed `CA`.
- A transformed `CA` column is expected to be absent; the audit must empirically confirm or deny this.
- `cleavage_freq` expected range approximately `[-0.0015, 4.53]`, with approximately 78 NaN and 685 negative values.
- Expected approximate threshold sensitivity on the whole dataset before measured filtering:
  - `cleavage_freq > 1e-5`: approximately 21,365 positives, approximately 1:14 imbalance.
  - `cleavage_freq > 1e-3`: approximately 8,280 positives, approximately 1:36 imbalance.
  - `cleavage_freq > 0.1`: approximately 1,184 positives, approximately 1:261 imbalance.

These values are reference values to reproduce and confirm, not trusted facts. The audit must compute each value independently from the raw file and compare actual vs expected with explicit PASS / DISCREPANCY markers.

## Scope

Sprint 1 covers dataset and label audit only.

In scope:

- Verify local Mak 2022 dataset file availability and independently compute schema/statistics from the raw file.
- Treat dataset access/source as settled: Mak et al. 2022 plus Wayback snapshot of the original `crisprsql.com` dataset URL.
- Document dataset access methodology as a resolved fact, including original source URL, 404 status, Wayback snapshot usage, local filename, license/citation notes, and version caveats.
- Implement first-principles computation of every audit statistic and compare each computed value against expected reference values.
- Emit explicit PASS / DISCREPANCY markers in console output and reports for row/column counts, measured split, guide count, target count, genome/cell-line composition, missingness, feature presence, label columns, `cleavage_freq` quality, and threshold distributions.
- Audit row count, column count, guide count, target count, measured flag distribution, genome/cell-line composition, and target-count distribution per sgRNA.
- Reconcile actual columns enumerated from the file against the expected 45-column schema and feature groups.
- Compute label distributions for Schemes A and C directly from `cleavage_freq`.
- Empirically confirm or deny the presence of a transformed `CA` column.
- Document Scheme B as a Box-Cox / CA reproduction requirement because no `CA` column is expected.
- Decide and document treatment of NaN, negative, and extreme `cleavage_freq` values based on actual observed cases.
- Empirically confirm the 13 computed nucleosome features are string-formatted 23-element arrays before implementing parser behavior.
- Plan and implement parser requirements for the 13 computed features.
- Analyze missingness of the 6 experimental epigenetic scalar features and 13 computed nucleosome features.
- Confirm the `measured=1` test rule and document its impact on future split generation.
- Prepare report artifacts under `outputs/reports/`.
- Update canonical docs under `docs/`.
- Add focused tests for any implemented parsing or label behavior.

## Out Of Scope

Do not implement real ML models in Sprint 1.

Out of scope:

- PyTorch or PyTorch Geometric installation.
- GCN, GAT, GraphSAGE, heterogeneous GNN, or sequence-model implementation.
- Final graph construction.
- Full guide-level split implementation beyond documenting requirements and optionally computing distributions needed to design it.
- Model training, model evaluation, or performance claims.
- CRISPRoffT external validation.
- Alternative dataset source discovery, mirror investigation, or repository-source comparison. Dataset access is closed: use the Wayback snapshot of the original Mak 2022 URL.
- Large generated data commits.
- Putting core logic in notebooks.
- Adding large raw, interim, processed, checkpoint, PDF, or extracted asset files to Git.

## Current Status

This plan remains active. Steps 1-12 are complete; Step 13 remains.

Completed implementation state:

- Steps 1-10 completed the raw dataset audit, label sensitivity audit, computed-feature parser validation, missingness analysis, and split-policy implications.
- Step 11 generated the required report artifacts:
  - `outputs/reports/dataset_audit.md`
  - `outputs/reports/label_threshold_sensitivity.md`
  - `outputs/reports/feature_missingness.md`
- Step 11.5 completed a behavior-preserving refactor of `scripts/audit_dataset.py` before Step 12 documentation updates.
- Step 12 updated the canonical docs:
  - `docs/DATASET_AUDIT.md`
  - `docs/LABEL_SCHEMES.md`
  - `docs/FEATURE_PARSING.md`
  - `docs/DECISIONS.md`
  - `docs/EVALUATION_PROTOCOL.md`

Step 11.5 refactor details:

- `scripts/audit_dataset.py` is now a small CLI runner.
- Audit constants, expected references, and feature-group definitions live in `src/crispr_gnn/data/schemas.py`.
- Console audit logic lives in `src/crispr_gnn/data/audit_console.py`.
- Markdown report generation lives in `src/crispr_gnn/data/audit_reports.py`.
- Feature parsing remains in `src/crispr_gnn/data/parsers.py`.
- The refactor did not change label policy, split policy, report semantics, or the project decision to keep Mak CA / Box-Cox reproduction as a later paper-comparison track only.

Latest verification after Step 11.5:

- `uv run ruff check scripts/audit_dataset.py src tests` passed.
- `uv run pytest -q` passed with 15 tests.
- `uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml --sample` passed.
- `uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml` passed and regenerated the three Step 11 reports.
- `git status --short` could not be checked in this environment because `git` was not available on PATH.

Latest verification after Step 12:

- `uv run ruff check scripts/audit_dataset.py src tests` passed.
- `uv run pytest -q` passed with 15 tests.
- `uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml --sample` passed.

Next action:

- Continue with Step 13 validation and review.
- Do not redo Steps 1-12 unless Step 13 review exposes a concrete inconsistency.

## Files Expected To Be Modified Later

Code and tests:

- `scripts/audit_dataset.py`
- `src/crispr_gnn/data/labels.py`
- `src/crispr_gnn/data/parsers.py`
- `src/crispr_gnn/data/load_mak2022.py`
- `src/crispr_gnn/data/schemas.py`
- `tests/test_labels.py`
- `tests/test_config_loads.py`
- New focused tests as needed, likely:
  - `tests/test_feature_parsers.py`
  - `tests/test_dataset_audit_smoke.py`

Configs and docs:

- `configs/data/mak2022.yaml`
- `docs/DATASET_AUDIT.md`
- `docs/LABEL_SCHEMES.md`
- `docs/FEATURE_PARSING.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/DECISIONS.md`
- `docs/COMMANDS.md`
- `README.md` if setup or dataset placement instructions change.

Generated report artifacts:

- `outputs/reports/dataset_audit.md`
- `outputs/reports/label_threshold_sensitivity.md`
- `outputs/reports/feature_missingness.md`
- Optional small tables if intentionally tracked:
  - `outputs/tables/label_distribution.csv`
  - `outputs/tables/feature_missingness.csv`

Do not commit large raw data, processed parquet files, graph objects, or full run artifacts.

## Step-By-Step Implementation Plan

1. Verify environment, local data path, and reference baseline.

   - Run `uv sync`.
   - Confirm `configs/data/mak2022.yaml` points to `data/raw/260520_putative_nucleosomal.parquet`.
   - Check whether the raw dataset exists locally.
   - If missing, fail gracefully with a clear message explaining where to place the dataset.
   - Do not plan or perform alternative source discovery. Dataset source is resolved as the Wayback snapshot of the original Mak 2022 URL.
   - Define the expected reference baseline in code/config/docs for comparison:
     - 310,142 x 45 shape,
     - measured split,
     - unique sgRNA count,
     - unique target count and key,
     - cell-line/genome composition,
     - feature presence,
     - computed-feature missingness,
     - `cleavage_freq` range/NaN/negative counts,
     - threshold sensitivity.
   - The audit must compute actuals independently from the raw file and compare actual vs expected. Reference values are not to be used as computed results.

2. Enumerate actual columns and reconcile against expected schema.

   - Load the raw parquet file and enumerate actual columns from the file itself.
   - Compare the actual column list against the expected 45-column schema.
   - Explicitly check for:
     - 6 experimental scalar epigenetic features,
     - 13 computed nucleosome features,
     - 5 binding-energy scalar features,
     - `measured`,
     - `experiment_id`,
     - `cell_line`,
     - genome field,
     - `cleavage_freq`,
     - transformed `CA` or any CA-like column.
   - Report extra, missing, renamed, or ambiguous columns explicitly.
   - Print and report each schema check with PASS / DISCREPANCY.

3. Implement dataset audit loading path.

   - Load the raw parquet file using pandas/pyarrow.
   - Keep the initial implementation simple and deterministic.
   - Avoid modifying or writing processed data during the audit unless explicitly needed for small report tables.
   - Add or preserve a `--sample` mode that can run without the full dataset for smoke testing.
   - Ensure full-dataset mode emits console output that summarizes PASS / DISCREPANCY checks.

4. Compute high-level dataset audit metrics and compare against references.

   - Compute row count and column count from the file.
   - Compare against expected 310,142 rows x 45 columns and print/report PASS / DISCREPANCY.
   - Compute `measured=1` and `measured=0` counts.
   - Compare against expected 25,632 measured rows and 284,510 putative rows.
   - Compute unique sgRNA count using the selected guide key; document the exact key used.
   - Compare against expected 154 unique sgRNAs.
   - Compute unique target/off-target location count using the proposed key `target_chr + target_start + target_end + target_strand` unless the actual schema requires a documented adjustment.
   - Compare against expected 138,747 unique target locations.
   - Count rows by genome and compare against expected 5-genome composition: `hg19` dominant around 244K, `rn5` around 51K, `hg38` around 7K, plus `mm10` and `mm9`.
   - Count rows by cell line and compare against expected 7 cell lines.
   - Count missing `cell_line` rows and compare against expected approximately 14,108, concentrated in `experiment_id=18`.
   - Count rows by `experiment_id`, with special attention to `experiment_id=18`.
   - Compute sgRNA target-count distribution, including min, median, max, high-count guides, and whether mega-guides may dominate splits.
   - Every metric must show computed value, expected reference value, and PASS / DISCREPANCY status in reports.

5. Audit `cleavage_freq` quality and compare against references.

   - Empirically confirm `cleavage_freq` exists and is the raw frequency label column.
   - Empirically confirm or deny whether transformed `CA` or CA-like columns exist.
   - Compute `cleavage_freq` min and max.
   - Compare against expected approximate range `[-0.0015, 4.53]`.
   - Count NaN values and compare against expected approximately 78.
   - Count negative values and compare against expected approximately 685.
   - Count zero values.
   - Count values above 1 and report as extreme raw-frequency cases.
   - Count values in `(0, 1e-5]`, `(1e-5, 1e-3]`, `(1e-3, 1]`, and `>1`.
   - Print/report computed value, expected reference value where available, and PASS / DISCREPANCY for each reference-backed metric.
   - After recomputing actual cases, decide and document handling policy for NaN, negative, and extreme `cleavage_freq` values.
   - Do not silently clip, drop, or transform values without documenting the policy.

6. Compute label distributions for Schemes A, B, and C.

   - Scheme A: compute directly as `cleavage_freq > 1e-5`.
   - Compare whole-dataset positive count and imbalance against expected approximately 21,365 positives and approximately 1:14 imbalance.
   - Scheme C: compute directly as `cleavage_freq > 1e-3`.
   - Compare whole-dataset positive count and imbalance against expected approximately 8,280 positives and approximately 1:36 imbalance.
   - Also compute the exploratory high threshold `cleavage_freq > 0.1` and compare against expected approximately 1,184 positives and approximately 1:261 imbalance.
   - For each threshold report:
     - total positive and negative counts,
     - positive rate,
     - imbalance ratio,
     - measured=1 positive/negative counts,
     - measured=0 positive/negative counts.
   - Scheme B: do not treat as directly computable from a stored column unless the audit empirically finds such a column.
   - The audit must empirically confirm or deny the expected absence of transformed `CA`.
   - Document Scheme B as a Sprint 1 reproduction requirement requiring recomputing per-study Box-Cox ourselves:
     - paper applies per-study Box-Cox transformation,
     - standardized to Gaussian mean 0 and std 2,
     - clipped to `[-4, 4]`,
     - values below assay accuracy `1e-5` set to `CA = -4`.
   - State that Scheme A is paper-aligned for the negative boundary because values below `1e-5` are set to `CA = -4` in the paper's Methods.

7. Decide outlier handling policy.

   - Use actual audit results to propose explicit handling for:
     - NaN `cleavage_freq`,
     - negative `cleavage_freq`,
     - `cleavage_freq > 1`,
     - `experiment_id=18`,
     - non-human genomes.
   - Record final decisions in `docs/DECISIONS.md`.
   - Reflect these decisions in `docs/LABEL_SCHEMES.md`.
   - If actual counts differ from reference values, record discrepancy before deciding policy; do not hide the mismatch by changing the reference.

8. Empirically validate and implement parser for computed nucleosome features.

   - Identify the 13 computed nucleosome features described in the Mak 2022 paper/project plan.
   - Empirically verify each feature exists in the actual column list.
   - Empirically verify each is string-formatted and expected to contain 23 numeric positions.
   - Recompute rows missing computed nucleosome features and compare against expected approximately 15,153 rows and approximately 4.9%.
   - Report computed missingness, expected missingness, and PASS / DISCREPANCY.
   - Implement a parser in `src/crispr_gnn/data/parsers.py`.
   - Parser requirements:
     - parse valid 23-element arrays,
     - preserve or explicitly handle missing values,
     - reject malformed lengths unless policy says otherwise,
     - support later feature strategies: position-resolved, aggregated, and PAM-focused.
   - Add focused parser tests with valid, missing, and malformed examples.

9. Analyze epigenetic and binding-energy feature missingness.

   - For 6 experimental scalar epigenetic features:
     - confirm presence,
     - count missing values,
     - compute missingness percentage,
     - summarize by `measured`, genome, cell line, and experiment ID where useful.
   - For 13 computed nucleosome features:
     - count missing raw string fields,
     - count malformed arrays,
     - count arrays with incorrect length,
     - count arrays with non-numeric entries.
   - For 5 binding-energy scalar features:
     - confirm `energy_1` through `energy_5` presence,
     - compute missingness,
     - note whether they are safe for later feature-set ablations.
   - Report which features appear safe for Sprint 2/3 baseline work and which require imputation or exclusion policy.
   - Every reference-backed missingness or presence expectation must be reported with computed value, expected value, and PASS / DISCREPANCY.

10. Document split implications and measured=1 rule.

   - Do not build final splits in this sprint unless explicitly chosen later.
   - Document that final test sets must contain only `measured=1` rows.
   - Document that validation should prefer `measured=1` rows.
   - Document training may include `measured=0` putative negatives only with label-noise caveat.
   - Include per-scheme label distribution for measured-only rows to support future split design.
   - Use sgRNA target-count distribution to document stratification risks for guide-level splitting.

11. Generate report artifacts.

   - Create `outputs/reports/dataset_audit.md`.
   - Create `outputs/reports/label_threshold_sensitivity.md`.
   - Create `outputs/reports/feature_missingness.md`.
   - Reports must include actual computed values, expected reference values, and PASS / DISCREPANCY markers.
   - Any deviation from expected reference values must be explicitly reported as a DISCREPANCY in both console output and the relevant report.
   - Do not silently reconcile, overwrite, or hide mismatches.
   - Keep reports concise but complete enough to support later modeling decisions.
   - Avoid committing large generated tables unless they are small, reviewable, and intentionally tracked.

12. Update canonical docs.

   - Update `docs/DATASET_AUDIT.md` with final audit results, settled access methodology, reference comparisons, and discrepancy table.
   - Update `docs/LABEL_SCHEMES.md` with final Scheme A/B/C definitions, CA-column audit result, Scheme B reproduction requirement, and outlier policy.
   - Update `docs/FEATURE_PARSING.md` with empirical feature representation results, parser behavior, and feature dimensionality strategies.
   - Update `docs/DECISIONS.md` with dated decisions about dataset source, label scheme, outliers, `measured`, and feature parsing.
   - Update `docs/EVALUATION_PROTOCOL.md` only if the measured-test or split policy is refined.

13. Validate and review.

   - Run tests.
   - Run the audit in sample mode.
   - Run the audit on the full dataset if the file exists locally.
   - Check that no large data files are staged.
   - Review generated docs and reports for claims that exceed what the audit proves.
   - Review all PASS / DISCREPANCY output before accepting the sprint.

## Risks

- The local raw dataset file may be missing from `data/raw/`, may have a different filename, or may be in compressed form.
- The local file may differ from the expected reference snapshot, including shape, measured distribution, guide/target counts, feature missingness, or threshold distributions; the audit must detect and report any such discrepancy rather than assume a match.
- Column names may differ from assumptions in the project plan or paper notes.
- Some guide or target identifier fields may be ambiguous; unique counts must document the chosen key.
- `cleavage_freq` includes NaN, negative, and values above 1; unrecorded handling would invalidate later labels.
- Scheme B may not be exactly reproducible without implementing per-study Box-Cox transformation and standardization ourselves.
- Putative `measured=0` rows are not ground-truth test labels and can introduce label noise if used in training.
- Feature parsing can silently corrupt data if malformed string arrays are coerced too aggressively.
- Epigenetic missingness may be cell-line, genome, or experiment dependent, affecting later ablations.
- Large generated outputs can accidentally enter Git if `.gitignore` is bypassed.
- Markdown literature notes are useful context but should not be treated as authoritative replacements for source papers.

## Acceptance Criteria

- `uv sync` succeeds.
- `uv run pytest -q` succeeds.
- `uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml --sample` succeeds.
- Full audit command succeeds when `data/raw/260520_putative_nucleosomal.parquet` exists.
- Audit independently computes row/column counts, measured split, unique sgRNA count, unique target count, genome and cell-line composition, missingness, feature presence, `cleavage_freq` quality, and threshold distributions.
- Audit prints each computed value alongside its expected reference value with a PASS / DISCREPANCY marker.
- Any deviation from expected reference values is explicitly reported in `outputs/reports/dataset_audit.md` or the relevant report and is not silently reconciled.
- Dataset schema report documents:
  - row count,
  - column count,
  - missing/extra columns,
  - guide count and exact guide key,
  - target count and exact target key,
  - measured distribution,
  - feature group presence,
  - CA-column presence/absence check.
- Audit empirically confirms or denies presence of a transformed `CA` column rather than assuming it.
- Label sensitivity report documents Scheme A and Scheme C direct distributions, the `>0.1` sensitivity threshold, and explains Scheme B as a Box-Cox reproduction requirement.
- Outlier policy for NaN, negative, and extreme `cleavage_freq` values is documented.
- Feature missingness report documents all 19 epigenetic/nucleosome-context features and 5 binding-energy features.
- Parser behavior for 13 computed 23-position features is documented and tested.
- `measured=1` test rule is restated in audit and evaluation docs.
- Required report artifacts exist:
  - `outputs/reports/dataset_audit.md`
  - `outputs/reports/label_threshold_sensitivity.md`
  - `outputs/reports/feature_missingness.md`
- Required docs are updated:
  - `docs/DATASET_AUDIT.md`
  - `docs/LABEL_SCHEMES.md`
  - `docs/FEATURE_PARSING.md`
  - `docs/DECISIONS.md`
- No large raw, interim, processed, checkpoint, PDF, extracted asset, or run files are staged for commit.

## Commands To Run

Setup:

```bash
uv sync
```

Smoke tests:

```bash
uv run pytest -q
uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml --sample
```

Full Sprint 1 audit, when the raw dataset exists locally:

```bash
uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml
```

Optional focused test commands after parser/label tests are added:

```bash
uv run pytest tests/test_labels.py -q
uv run pytest tests/test_feature_parsers.py -q
```

Git hygiene checks before commit:

```bash
git status --short
git status --short | Select-String "data/raw|data/interim|data/processed|outputs/runs|.parquet|.pt|.ckpt|original.pdf|assets"
```

## Documentation Updates Required

Update `docs/DATASET_AUDIT.md` with:

- settled dataset source and access methodology,
- original URL and Wayback snapshot citation,
- local path and file format,
- reference values vs independently computed values,
- PASS / DISCREPANCY summary table,
- row and column counts,
- guide and target counts with exact keys,
- measured distribution,
- genome, cell-line, and experiment summaries,
- schema reconciliation against Mak 2022 and crisprSQL context,
- target-count distribution per guide,
- dataset limitations and caveats.

Update `docs/LABEL_SCHEMES.md` with:

- final Scheme A/B/C definitions,
- exact threshold logic,
- Scheme A paper-aligned rationale from Mak Methods,
- Scheme B reproducibility status and Box-Cox reproduction requirement,
- empirical CA-column presence/absence result,
- NaN, negative, zero, and >1 handling policy,
- measured=0 caveat,
- threshold sensitivity summary with reference comparisons.

Update `docs/FEATURE_PARSING.md` with:

- the list of 13 computed nucleosome features,
- empirical confirmation of string-formatted 23-element representation,
- exact parser assumptions,
- missing/malformed handling,
- 23-position validation rule,
- position-resolved, aggregated, and PAM-focused strategy notes,
- examples of valid and invalid parsed values.

Update `docs/DECISIONS.md` with dated entries for:

- Mak 2022 dataset access and Wayback source method,
- final primary label scheme,
- outlier policy,
- treatment of `measured=0` rows,
- treatment of `experiment_id=18`,
- treatment of non-human genomes,
- parser strategy for computed features.

Update `docs/EVALUATION_PROTOCOL.md` if Sprint 1 changes or sharpens:

- final split constraints,
- measured-only test rule,
- validation set preference,
- allowed use of putative negatives in training,
- leakage warnings for future graph construction.
