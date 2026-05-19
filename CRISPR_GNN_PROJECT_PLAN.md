# Updated Project Guide: Context-Aware GNN for CRISPR-Cas9 Off-Target Prediction

> **Purpose:** This document is the scope-controlled project plan after reviewing six technical critiques and after the dataset audit on 2026-04-29.
> **Use case:** Share with coding agents, collaborators, or the thesis advisor as the current execution guide.
> **Companion documents:** Repository structure, paths, and tooling decisions live in `PROJECT_FOLDER_STRUCTURE.md`. Detailed reference material lives under `docs/`.

---

## 0. One-Sentence Project Definition

We will build an **epigenetic-context-aware GNN framework** for **CRISPR-Cas9 off-target prediction**, starting from a reproducible **GCN edge/link-prediction baseline**, then testing whether **graph enrichment**, **GAT/GATv2 attention**, **epigenetic features**, and **imbalance-aware training** improve performance over strong **sequence-based deep learning baselines** under **guide-level evaluation**.

---

## 1. Updated Main Contribution

### English contribution statement

> We develop an epigenetic-context-aware GNN framework for CRISPR-Cas9 off-target prediction and systematically test whether graph enrichment, attention-based message passing, and imbalance-aware training improve guide-level generalization over sequence-based deep learning baselines.

### Turkish contribution statement

> Bu projede CRISPR-Cas9 off-target tahminini sgRNA-target graph üzerinde ele alıyor; yalnızca minimal bipartite graph değil, sequence/context similarity ile zenginleştirilmiş graph yapılarının, GAT attention mekanizmasının, epigenetik özelliklerin ve imbalance-aware training stratejilerinin guide-level genelleme üzerindeki etkisini sequence-based deep learning baseline'larına karşı test ediyoruz.

---

## 2. Final Strategic Decision

### Primary starting dataset

Use the **Mak et al. 2022 crisprSQL-derived epigenetic dataset** as the first working dataset.

Why:

- More controlled than CRISPRoffT.
- Built on crisprSQL.
- Includes 19 epigenetic and nucleosome-context features (verified 2026-04-29).
- Better starting point for graph construction, label audit, GCN/GAT comparison, and epigenetic ablation.

### Fallback dataset

Use **raw crisprSQL** (`260520.zip` from crisprsql.com) if Mak et al. 2022 dataset access, schema, or labels become problematic.

### Stretch / external validation dataset

Use **CRISPRoffT** only after the core pipeline works.

Why CRISPRoffT is not the first pipeline dataset:

- It is not a single clean benchmark; it is a large integrated database/resource.
- It combines many studies, technologies, cell types, Cas/gRNA combinations, and experimental conditions.
- It is excellent for external validation or larger-scale experiments, but it can increase early overhead through filtering, leakage control, and heterogeneity management.

---

## 3. Updated Verdict on the Six Critiques

| Critique | Verdict | Plan Change |
|---|---:|---|
| 1. Graph structure may not add enough value | **Correct** | Add graph schema ablation: minimal bipartite vs similarity/context-enriched graph. |
| 2. Heterogeneous GNNs were skipped | **Partly correct** | Keep HeteroConv/R-GCN/HGT as stretch, not core scope. |
| 3. Sequence-based DL baselines are missing | **Correct** | Add at least one same-split sequence DL baseline. |
| 4. Mak et al. 2022 label threshold is unclear | **Resolved** | Audit done 2026-04-29; primary scheme = `cleavage_freq > 1e-5` (paper-aligned). |
| 5. Eight full sprints are too ambitious | **Correct** | Move CRISPRoffT to stretch; move epigenetic ablation before full imbalance study. |
| 6. GraphSAGE removal needs justification | **Partly correct** | Keep GraphSAGE as quick optional ablation, not a must-have. |

---

## 4. Core Literature Anchors

### 4.1 Direct graph-based off-target baseline

**Vinodkumar et al. 2021**
*Prediction of sgRNA Off-Target Activity in CRISPR/Cas9 Gene Editing Using Graph Convolution Network*
Link: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8156774/>

Role in this project:

- Closest direct GNN/off-target prediction paper.
- Frames CRISPR off-target prediction as a graph/link-prediction problem.
- Provides the rationale for starting with a GCN baseline.

Important limitation:

- The graph is close to an sgRNA-target relation graph.
- If used as-is, the graph can become a set of weakly connected or isolated clusters.
- This motivates our graph-enrichment ablation.

---

### 4.2 Primary dataset and epigenetic feature basis

**Störtz & Minary 2021** — *crisprSQL: a novel database platform for CRISPR/Cas off-target cleavage assays*
Link: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7778913/>
Role: Source database for guide-target pairs and epigenetic marker references.

**Mak, Störtz & Minary 2022** — *Comprehensive computational analysis of epigenetic descriptors affecting CRISPR-Cas9 off-target activity*
Article: <https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-022-09012-7>
DOI: 10.1186/s12864-022-09012-7
Code: <https://github.com/jeffmak/crispr-cas9-epigenetics>
Original dataset URL (now unreachable): <https://crisprsql.com/downloads/260520_putative_nucleosomal.parquet.gz>
Backup mirror (alternate format): <https://github.com/florianst/picrispr/blob/main/offtarget_260520_nuc.csv.zip>

Role:

- Preferred initial dataset.
- Provides 6 experimental and 13 computed nucleosome-related epigenetic features.
- Supports the main novelty: **epigenetic-context-aware GNN**.

Dataset access status (2026-04-29):

- Original URL returns 404.
- Dataset retrieved from Internet Archive Wayback Machine snapshot of the same URL.
- Verified contents: 310,142 rows × 45 columns, all 19 epigenetic features present.
- License: CC-BY 4.0 (BMC Genomics open access).
- Access methodology and citation strategy: see `docs/DATASET_AUDIT.md` and `docs/DECISIONS.md`.

For dataset schema, label distributions, and outlier handling decisions, see `docs/DATASET_AUDIT.md`.
For paper-specific reference material (quotes, parameters, reproduction details), see `docs/literature/papers/2022_mak_epigenetic_descriptors/paper.md`.

---

### 4.3 Larger external validation resource

**Wang et al. 2025**
*CRISPRoffT: comprehensive database of CRISPR/Cas off-targets*
Paper: <https://academic.oup.com/nar/article/53/D1/D914/7889256>
Download: <https://ccsm.uth.edu/CRISPRoffT/download.html>

Role:

- Stretch goal for external validation or larger heterogeneous testing.
- Not recommended as the first dataset unless the team has enough time for database filtering and attrition analysis.

---

### 4.4 Sequence-based DL baselines

Potential baselines:

- **CnnCrispr** — Paper: <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-3395-z>, Code: <https://github.com/LQYoLH/CnnCrispr>
- **CRISPR-Net** — Code: <https://github.com/JasonLinjc/CRISPR-Net>
- **DeepCRISPR** — Code: <https://github.com/bm2-lab/DeepCRISPR>
- **CRISPR-IP** — Paper: <https://www.sciencedirect.com/science/article/pii/S2001037022000137>

Plan decision:

- Add **at least one sequence-based DL baseline** on the same dataset, split, and metrics.
- Recommended starting choice: CnnCrispr (open code, PyTorch-friendly, simplest to reproduce).
- Do not rely only on paper-reported metrics, because different datasets/splits/negative-sampling strategies are not directly comparable.

---

### 4.5 Optional secondary-structure inspiration

**Graph-CRISPR 2025**
*Graph-CRISPR: a gene editing efficiency prediction model based on graph neural network with integrated sequence and secondary structure feature extraction*
PubMed: <https://pubmed.ncbi.nlm.nih.gov/40814228/>
GitHub: <https://github.com/MoonLBH/Graph-CRISPR>

Role: Optional future-work inspiration for adding sgRNA secondary-structure features. Not a direct off-target prediction paper.

---

## 5. Updated Dataset Strategy

### 5.1 Phase 1 dataset

Preferred:

```text
Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset
File: 260520_putative_nucleosomal.parquet (~120 MB)
Source: Wayback Machine snapshot
```

Verified at audit (2026-04-29):

- 310,142 rows × 45 columns
- 154 unique sgRNAs, 138,747 unique target locations
- measured=1 (real experimental): 25,632 rows
- measured=0 (putative off-targets, paper assigns CA = -4): 284,510 rows
- 19 epigenetic features confirmed (6 experimental scalars + 13 computed 23-element string arrays)
- 7 cell lines, 5 genomes (mostly hg19)

Note on the paper's reported 251,854 datapoints: this corresponds to the original publication snapshot. Our 310,142 includes ~58K additional post-publication updates with identical structure.

Fallback:

```text
raw crisprSQL CSV/database export (260520.zip from crisprsql.com)
```

### 5.2 Phase 2 dataset

```text
CRISPRoffT filtered subset for external validation
```

### 5.3 Dataset audit requirements

Sprint 1 must produce a complete audit at `docs/DATASET_AUDIT.md` and `outputs/reports/dataset_audit.md`. The full task list lives in Section 12 (Sprint 1).

---

## 6. Label Strategy and Threshold Sensitivity

### 6.1 Why this is critical

Mak et al. 2022 stores raw `cleavage_freq` (frequency, observed range [-0.0015, 4.53] with 78 NaN and 685 negative values). The paper itself uses a Box-Cox transformed `CA` value clipped to [-4, 4]. Binary labels must be defined carefully because AUPRC changes significantly with the threshold choice.

### 6.2 Candidate label schemes (summary)

| Scheme | Definition | Role |
|---|---|---|
| **A** | `cleavage_freq > 1e-5` | **PRIMARY** — paper-aligned negative threshold |
| **B** | Box-Cox transformed `CA > -4` | Reproduces paper's exact target (used for paper comparison track) |
| **C** | `cleavage_freq > 1e-3` (excluding mid range) | High-confidence ablation in Sprint 6 |
| **D** | continuous regression on `cleavage_freq` or transformed `CA` | Reserved for regression head if used |

Paper Methods quote: *"cleavage activity values below the lowest reported assay accuracy of 10⁻⁵ set to -4"* — this anchors Scheme A to the paper's negative boundary.

For full scheme definitions, outlier handling rules (NaN, negative, > 1 values), and Box-Cox reproduction details, see `docs/LABEL_SCHEMES.md`.

### 6.3 Required Sprint 1 outputs

Sprint 1 must produce three artifacts:

1. **Threshold sensitivity table** on the full dataset (310K rows).
2. **Per-split label distribution table** after applying the measured=1 test rule (Section 6.5).
3. **Outlier handling decision log** documenting the chosen treatment for NaN/negative/extreme `cleavage_freq` rows.

All three artifacts go to `outputs/reports/label_threshold_sensitivity.md`.

Decision rule:

- Adopt **Scheme A** as the primary label for Sprints 2–7.
- Reproduce **Scheme B** only in the paper comparison track (Section 11.5).
- Run **Scheme C** as a robustness ablation in Sprint 6.
- Reserve **Scheme D** for the regression head if Sprint 5 epigenetic ablation warrants continuous targets.

### 6.4 The `measured` flag — critical split rule

The dataset contains a `measured` column distinguishing two row types:

- **measured=1** (25,632 rows): real experimental crisprSQL data.
- **measured=0** (284,510 rows): putative off-targets generated by Mak et al. via batmis sequence alignment with < 7 mismatches; paper assigns CA = -4.

**Hard rules for splits:**

1. Test set MUST contain only `measured=1` rows. Putative rows cannot be used as test ground truth.
2. Validation set SHOULD prefer `measured=1` rows.
3. Training set MAY include `measured=0` rows as negative examples, but this introduces label noise.
4. Report per-set composition explicitly: `train: X measured=1 + Y measured=0, val: Z measured=1, test: W measured=1`.

Biological justification: Paper Discussion shows that putative sites with low Nucleotide BDM cluster near nucleosome dyads, suggesting biologically plausible negatives — not random synthetic noise. Some `measured=0` rows may still be true positives that were never tested (label noise risk).

---

## 7. Graph Construction Plan

### 7.1 Node types

#### sgRNA node

One node per unique sgRNA (keyed by `grna_target_id` or `grna_target_chr + grna_target_start`).

Potential features:

- One-hot sequence encoding.
- k-mer counts.
- GC content.
- Positional encoding.
- Optional secondary-structure features.

#### target/off-target site node

One node per unique candidate target/off-target site (keyed by `target_chr + target_start + target_end + target_strand`).

Potential features:

- One-hot target sequence.
- PAM feature.
- Mismatch summary features.
- Epigenetic features (6 experimental + 13 computed; see Section 7.3).
- Genomic coordinate features if reliable.

### 7.2 Edge types

#### Required edge type

```text
sgRNA --candidate_pair--> target_site
```

Edge label:

```text
1 = off-target cleavage/activity positive
0 = negative / putative non-cleaved
```

Edge features:

- Mismatch count.
- Mismatch positions.
- Mismatch types.
- PAM compatibility.
- Pairwise sequence encoding.
- Optional binding-energy features (`energy_1` to `energy_5` are present in the dataset).

### 7.3 Computed feature parsing

The 13 computed nucleosome features are stored as **string-formatted 23-element arrays** in the Parquet file (one value per target sequence position). Sprint 1 must implement a parser at `src/crispr_gnn/data/parsers.py`.

Three feature dimensionality strategies will be compared in Sprint 5 (epigenetic ablation):

- Position-resolved (full 23 dims per feature → 299 + 6 = 305 features).
- Aggregated scalars (mean/max/sum/std → ~58 features).
- PAM-region focused (seed region or PAM-proximal subset).

Full parsing code, examples, and strategy details are in `docs/FEATURE_PARSING.md`.

---

## 8. Graph Schema Ablation

This is a required update after Critique 1.

### Graph A: Minimal bipartite graph

Baseline graph closest to GCN-CRISPR.

```text
sgRNA_i  --pair_edge-->  target_j
```

Purpose: reproduce the simplest graph formulation; establish the GCN baseline.
Risk: may behave like isolated star clusters; message passing may add limited value over sequence-pair models.

### Graph B: sgRNA-similarity-enriched graph

Add similarity edges among sgRNAs.

```text
sgRNA_i --similar_to--> sgRNA_j
sgRNA_i --pair_edge--> target_j
```

Candidate similarity metrics: Hamming distance, edit distance, k-mer Jaccard, sequence embedding cosine similarity.
Recommended first implementation: top-k nearest neighbors by edit distance or k-mer similarity (k = 5 or 10).

### Graph C: target/context-enriched graph

Add similarity or context edges among target sites.

```text
target_i --similar_to--> target_j
sgRNA_i --pair_edge--> target_j
```

Candidate similarity metrics: target sequence Hamming/edit distance, k-mer similarity, mismatch-profile similarity, epigenetic-feature cosine similarity, genomic proximity.
Recommended first implementation: top-k target-target edges by sequence/context similarity (k = 5 or 10).

### Graph schema comparison

Run at least:

```text
Graph A + GCN
Graph B + GCN
Graph C + GCN
```

If time allows: Graph A/B/C + GAT or GATv2.

Main question:

> Does graph enrichment actually improve AUPRC over a minimal bipartite graph and sequence DL baselines?

---

## 9. Model Plan

### 9.1 Model 0: Non-GNN tabular baselines

Purpose: check that the dataset and labels contain learnable signal; provide simple sanity-check baselines.

Models: Logistic Regression, Random Forest or XGBoost, MLP.
Input: sequence-derived features, mismatch features, epigenetic/nucleosome features.

### 9.2 Model 1: Sequence-based deep learning baseline

This is now a **must-have**, not optional.

Recommended starting choice: **CnnCrispr-inspired CNN/BiLSTM baseline** (open code, simplest to reproduce on PyTorch).

Alternative options: CRISPR-IP-inspired CNN+BiLSTM+attention; CRISPR-Net reproduction.

If published code is difficult to run due to old dependencies, implement a **published-architecture-inspired baseline** and clearly state it.

Use the same dataset, split, label scheme, and AUPRC/AUROC metrics as the GNN models.

### 9.3 Model 2: GCN baseline

Purpose: literature-aligned graph baseline; first real GNN model.

Architecture: PyTorch Geometric `GCNConv`; edge classifier / link predictor on sgRNA-target pairs; weighted BCE as default starting loss.

Run on Graph A first, then Graph B/C for graph-enrichment ablation.

**Architectural difference vs Mak et al. 2022's models:** Mak's XGBoost and CNN intentionally exclude sgRNA-DNA sequence input to isolate epigenetic feature importance via SHAP. Our GNN approach combines three information sources: (1) sequence, (2) epigenetic features, (3) graph topology. This combination is the project's main novelty.

### 9.4 Model 3: GAT / GATv2

Purpose: test whether attention-based message passing improves over GCN; analyze whether attention weights reveal useful guide/target/context patterns.

Architecture: PyG `GATConv` or `GATv2Conv`; same graph, split, features, and label scheme as GCN.

Main comparison: GCN vs GAT vs GATv2.

Do not make GAT the first model. It is the main improvement candidate after the pipeline works.

### 9.5 Optional Model 4: GraphSAGE quick ablation

GraphSAGE is not core, but can be tested quickly using `SAGEConv`.

Better justification if not included:

> GraphSAGE is most useful for large, evolving, inductive graph settings with neighbor sampling needs. Our core research question is not neighbor-sampling scalability; it is whether epigenetic context, graph enrichment, and attention improve guide-level off-target prediction. Therefore GraphSAGE is a stretch ablation, not a must-have.

If added: GCN vs GAT vs GraphSAGE under same graph/split/features.

### 9.6 Optional Model 5: Heterogeneous GNN

Keep as stretch.

Possible implementations: PyG `HeteroData`, `HeteroConv`, R-GCN-style relation-specific convolution, HGT.

Use only if graph enrichment introduces multiple useful relation types:

```text
sgRNA --pair_edge--> target
sgRNA --similar_to--> sgRNA
target --similar_to--> target
target --has_context--> cell_line / assay / gene
```

Do not make HGT a core requirement unless the team has enough time.

---

## 10. Training and Class Imbalance Strategy

### 10.1 Actual imbalance ratios

Audit (2026-04-29) shows imbalance depends strongly on threshold choice:

- Scheme A (`cleavage_freq > 1e-5`, paper-aligned): 1:14 — manageable.
- Scheme C (`> 1e-3`, high-confidence): 1:36.
- Threshold 0.1+: 1:261.

Earlier project notes assumed 1:1000+ imbalance based on a different dataset; the Mak augmented dataset is more balanced because augmentation added putative negatives at a controlled ratio.

### 10.2 Default training setup

For Scheme A: weighted BCE is sufficient as primary; focal loss as ablation.
For Scheme C: focal loss recommended.
For Scheme D / regression: MSE or Huber loss.

### 10.3 Full imbalance comparison

Sprint 6 (after epigenetic ablation) compares:

- Weighted BCE.
- Focal Loss.
- Dice Loss.
- Balanced batch sampling.
- Optional hard negative mining.

Main question:

> Which imbalance-aware strategy improves AUPRC under guide-level split?

---

## 11. Evaluation Plan

### 11.1 Primary metric

```text
AUPRC
```

Reason:

- The task is highly imbalanced.
- ROC-AUC may look strong even when positive-class retrieval is weak.

### 11.2 Secondary metrics

- AUROC, F1, MCC, Precision@K, Recall at fixed FPR, confusion matrix.
- For paper comparison track: Spearman, Pearson.

### 11.3 Split strategy

#### Debug split

```text
random edge split — quick sanity checks only
```

#### Main split

```text
guide-level split (leave-guide-out style)
```

Stratify by sgRNA size: some sgRNAs have 19 targets, others 50K — without stratification, mega-sgRNAs can dominate the test set.

#### Special handling

- Test set MUST contain only `measured=1` rows (Section 6.4 rule).
- `experiment_id=18` (~14K rows with missing cell_line): exclude entirely OR hold out as a separate "noisy" evaluation set.
- Non-human genomes (rn5, mm10, mm9): include in training; report per-genome performance to detect species effects.

#### Cross-genome generalization (stretch)

Train on hg19 only, test on hg38 + rodent genomes.

### 11.4 Leakage rules

- Do not allow the same guide to appear in both train and test for guide-level evaluation.
- Similarity edges must be built without using labels.
- Feature normalization/statistics must be fit only on train data.
- Do not generate target-target or sgRNA-sgRNA similarity using test labels.
- If graph transductive access is used, clearly document what information is available during training.

### 11.5 Comparison strategy with Mak et al. 2022

Two parallel evaluation tracks position our GNN against the paper:

- **Track 1 — Paper reproduction (regression).** Target = Box-Cox CA (Scheme B); reproduce paper's XGBoost (Spearman 0.42, Pearson 0.62) and CNN (Spearman 0.42, Pearson 0.59) using their GitHub code; report our GCN/GAT on same metrics for direct comparison.
- **Track 2 — Our binary classification (AUPRC primary).** Target = Scheme A; compare against sequence-only DL baseline, GCN-CRISPR style baseline, and our context-aware GNN.

Detailed reporting templates and protocol are in `docs/EVALUATION_PROTOCOL.md`.

---

## 12. Updated Sprint Plan

### Sprint 1: Dataset + Label Audit

Goal: confirm dataset usability for binary classification and define the label scheme.

Tasks:

- Document Wayback Machine access methodology.
- Verify column schema; reconcile with paper Methods.
- Implement parser for 13 string-array computed features.
- Compute label distribution under Schemes A, B, C; pick primary.
- Decide handling of NaN, negative, and extreme `cleavage_freq` outliers.
- Decide handling of `experiment_id=18` and non-human genomes.
- Compute sgRNA target-count distribution; design stratification strategy.
- Send notification email to Mak about broken URL.

Deliverables:

```text
outputs/reports/dataset_audit.md
outputs/reports/label_threshold_sensitivity.md
outputs/reports/feature_missingness.md
src/crispr_gnn/data/parsers.py
docs/DATASET_AUDIT.md (updated)
docs/DECISIONS.md (Wayback access decision logged)
```

---

### Sprint 2: Fair Non-Graph and Sequence Baselines

Goal: establish strong baselines before claiming GNN novelty.

Tasks:

- Build feature table from sequences, mismatch features, and epigenetic features.
- Train Logistic Regression, XGBoost or Random Forest, MLP.
- Add at least one sequence-based DL baseline (CnnCrispr-inspired recommended).
- Evaluate all baselines on same splits and metrics.

Deliverables:

```text
outputs/results/baseline_results.csv
outputs/reports/baseline_report.md
```

---

### Sprint 3: Graph Construction + Leakage Control

Goal: build trainable graph datasets and avoid leakage.

Tasks:

- Create sgRNA nodes, target nodes, sgRNA-target edges and labels.
- Implement Graphs A, B, C from Section 8.
- Implement guide-level split and random edge split for debug.
- Build PyG `Data` or `HeteroData` objects.

Deliverables:

```text
src/crispr_gnn/graph/graph_builder.py
src/crispr_gnn/data/splits.py
outputs/reports/graph_schema_report.md
```

---

### Sprint 4: GCN Baseline

Goal: build the first literature-aligned GNN baseline.

Tasks:

- Implement GCN edge classifier.
- Train on Graph A.
- Evaluate under random split and guide-level split.
- Train on Graphs B and C for graph-enrichment ablation.
- Compare against sequence DL baseline from Sprint 2.

Deliverables:

```text
src/crispr_gnn/models/gcn.py
outputs/results/gcn_results.csv
outputs/reports/gcn_report.md
```

---

### Sprint 5: Epigenetic Feature Ablation

Goal: test the main biological novelty early.

Feature sets to compare (same model, same split):

1. Sequence only.
2. Sequence + mismatch features.
3. Sequence + mismatch + 5 binding energy scores.
4. Sequence + mismatch + 6 experimental epigenetic features.
5. (4) + 13 computed features as aggregated scalars.
6. (4) + 13 computed features position-resolved (full 305-dim).

Tasks:

- Run feature ablations on best stable baseline model.
- Prefer GCN first; repeat on GAT later if time allows.
- Per-cell-line breakdown: does epigenetic context help more in some cell lines?
- Optional: per-feature SHAP analysis on best model (replicates paper methodology).

Deliverables:

```text
outputs/results/epigenetic_ablation.csv
outputs/reports/epigenetic_ablation_report.md
outputs/figures/sprint5/feature_set_auprc_comparison.png
```

This sprint is the project's main novelty experiment. Prioritize over Sprint 6.

---

### Sprint 6: Imbalance-Aware Training Comparison

Goal: systematically test class-imbalance strategies.

Tasks:

- Compare Weighted BCE, Focal Loss, Dice Loss, balanced sampler.
- Optional: hard negative mining.
- Evaluate all under guide-level split with AUPRC as primary metric.

Deliverables:

```text
outputs/results/imbalance_comparison.csv
outputs/reports/imbalance_report.md
```

---

### Sprint 7: GAT / GATv2 Attention Model

Goal: test whether attention improves over GCN.

Tasks:

- Implement GATConv model; GATv2Conv if feasible.
- Use same splits, same features, same graph schemas.
- Compare GCN vs GAT vs GATv2.
- Analyze attention weights if available.

Deliverables:

```text
src/crispr_gnn/models/gat.py
outputs/results/gat_comparison.csv
outputs/reports/gat_report.md
```

---

### Stretch Sprint A: CRISPRoffT External Validation

Goal: test whether the model generalizes beyond crisprSQL/Mak-derived data.

Tasks: download CRISPRoffT data; perform attrition analysis; create controlled subset; align features and labels; evaluate trained model or retrain under comparable setup.

Deliverables:

```text
outputs/reports/crisprofft_attrition.md
outputs/results/crisprofft_external_validation.csv
```

---

### Stretch Sprint B: Heterogeneous GNN

Goal: test whether explicit node/edge types improve performance.

Tasks: convert graph to PyG `HeteroData`; add relation-specific edges; try HeteroConv or R-GCN-style model; optional HGT.

Deliverables:

```text
src/crispr_gnn/models/hetero_gnn.py
outputs/results/hetero_gnn_results.csv
```

---

### Stretch Sprint C: GraphSAGE Quick Ablation

Goal: answer the possible committee question "Why not GraphSAGE?"

Tasks: implement SAGEConv model; run same split/features/graph as GCN/GAT; add one comparison table.

Deliverables:

```text
outputs/results/graphsage_ablation.csv
```

---

## 13. Must-Have / Should-Have / Stretch Scope

### Must-have

- Dataset + label audit.
- Threshold sensitivity analysis.
- Non-GNN baseline.
- Same-split sequence DL baseline.
- Graph construction (Graph A minimum).
- GCN baseline.
- Epigenetic feature ablation.
- GAT/GATv2 comparison.
- Guide-level split.
- AUPRC-first evaluation.

### Should-have

- Weighted BCE vs focal loss comparison.
- Graph enrichment ablation (Graphs B, C).
- Attention-weight or feature-importance interpretation.

### Stretch

- CRISPRoffT external validation.
- Heterogeneous GNN / R-GCN / HGT.
- GraphSAGE quick ablation.
- Full systematic imbalance study.
- sgRNA secondary-structure features inspired by Graph-CRISPR.

---

## 14. One-Person vs Two-Person Execution

### If one person

Core plan:

```text
Sprint 1: Dataset + label audit
Sprint 2: non-GNN + sequence DL baseline
Sprint 3: graph construction
Sprint 4: GCN baseline
Sprint 5: epigenetic ablation (main novelty)
Sprint 6: minimal imbalance comparison
Sprint 7: GAT/GATv2
Stretch: CRISPRoffT / HeteroGNN / GraphSAGE
```

Do not make CRISPRoffT, HGT, or GraphSAGE core requirements.

### If two people

Suggested split:

#### Person A: data/model pipeline

- Dataset audit.
- Label thresholding.
- Baselines.
- GCN/GAT training.
- Evaluation metrics.

#### Person B: graph/context pipeline

- Similarity edge construction.
- Epigenetic feature ablation.
- Heterogeneous graph stretch.
- CRISPRoffT attrition analysis.
- GraphSAGE quick ablation.

---

## 15. Repository Structure

For full repository structure, configuration files, and tooling decisions (uv-based environment, Colab workflow, naming conventions), see:

→ **`PROJECT_FOLDER_STRUCTURE.md`**

This separation keeps the project plan focused on "what to build and why" while the structure document focuses on "how to organize code." Any contradictions between this document and `PROJECT_FOLDER_STRUCTURE.md` should be resolved in favor of the structure document for paths and naming, and in favor of this document for scientific decisions.

Quick reference:

- Sprint deliverables: `outputs/reports/<sprint_name>.md`
- Generated figures: `outputs/figures/<sprint_name>/`
- Trained model checkpoints: `outputs/runs/<run_id>/`
- Decisions log: `docs/DECISIONS.md`
- Dataset audit: `docs/DATASET_AUDIT.md` (canonical) + `outputs/reports/dataset_audit.md` (raw audit data)

---

## 16. Final Decision Tree

```text
1. Can the Mak et al. 2022 dataset be downloaded and parsed?
   ├── Yes → Use it as primary dataset.
   │         (Audit verified 2026-04-29: 310,142 × 45, all 19 features present.
   │          Final adoption pending Sprint 1 outlier handling decisions.)
   └── No  → Use raw crisprSQL as fallback.

2. Can labels be clearly defined?
   ├── Yes → Default candidate: Scheme A (cleavage_freq > 1e-5, paper-aligned).
   │         Final choice confirmed after Sprint 1 threshold sensitivity analysis.
   └── No  → Use regression or high-confidence binary subset.

3. Do non-GNN and sequence DL baselines learn useful signal?
   ├── Yes → Proceed to graph models.
   └── No  → Debug labels, features, and splits.

4. Does minimal GCN work?
   ├── Yes → Try graph enrichment and GAT.
   └── No  → Debug graph construction and leakage.

5. Does graph enrichment improve over minimal graph?
   ├── Yes → Use enriched graph as main graph schema.
   └── No  → Keep minimal graph and report ablation.

6. Do epigenetic features improve AUPRC?
   ├── Yes → Main novelty = epigenetic-context-aware GNN.
   └── No  → Main story shifts to graph enrichment / attention / imbalance.

7. Does GAT improve over GCN?
   ├── Yes → Main model = GAT/GATv2.
   └── No  → Main model = GCN with better features/training; report GAT ablation.

8. Is there time left?
   ├── Yes → CRISPRoffT external validation, HeteroGNN, GraphSAGE.
   └── No  → Finalize crisprSQL/Mak results with strong evaluation.
```

---

## 17. Do-Not-Overclaim List

Avoid these claims unless directly supported by experiments:

- "GAT is always better than GCN."
- "GNN is superior to sequence-pair models."
- "Epigenetic features always improve off-target prediction."
- "CRISPRoffT is a clean benchmark dataset."
- "Mak et al. 2022 is already a ready-made binary classification dataset."
- "GraphSAGE is unnecessary because guide-level split is inductive."
- "Mak et al. used a small dataset because they removed non-human data." — FALSE; they kept human + rodent.
- "Our 310K rows give us 12x more data than baseline papers." — MISLEADING; 25,632 measured + 284,510 putative.
- "All 19 epigenetic features are clean and complete." — FALSE; ~5% missing.
- "Class imbalance is extreme (1:1000+) in this dataset." — FALSE; ~1:14 at paper-aligned threshold.
- "Mak et al.'s CNN is a weak baseline because it ignores sequence." — UNFAIR; intentional design for SHAP isolation.
- "Our GNN beats Mak et al." — Only valid if same metric + same split + same target.
- "Putative off-targets are synthetic noise." — NUANCED; biologically plausible negatives, but with label noise.

Safer claims:

- "We evaluate whether GAT improves over GCN under the same dataset and split."
- "We test whether graph enrichment improves over a minimal bipartite graph."
- "We test whether epigenetic/nucleosome context improves guide-level AUPRC."
- "CRISPRoffT is reserved as a larger heterogeneous external validation resource."
- "We extend Mak et al.'s feature analysis by combining sequence + features + graph topology in a unified GNN."

---

## 18. Final Recommended Route

```text
Mak et al. 2022 crisprSQL-derived epigenetic dataset (verified)
        ↓
Dataset + label audit
        ↓
Non-GNN + sequence DL baselines
        ↓
Minimal and enriched graph construction
        ↓
GCN baseline
        ↓
Epigenetic feature ablation (main novelty)
        ↓
Imbalance-aware training comparison
        ↓
GAT / GATv2 attention model
        ↓
Optional: CRISPRoffT external validation
        ↓
Optional: Heterogeneous GNN / GraphSAGE / secondary structure
```

---

## 19. Short Advisor-Facing Summary

We start with the Mak et al. 2022 crisprSQL-derived epigenetic dataset (310K rows, 19 epigenetic features, verified 2026-04-29) because it is more controlled than CRISPRoffT and already contains epigenetic/nucleosome-context features. First, we audit labels and thresholds; the paper-aligned scheme (`cleavage_freq > 1e-5`) gives a manageable 1:14 imbalance. Then, we compare tabular and sequence-based deep learning baselines against a GCN edge-prediction model. To address the concern that a simple bipartite graph may not add much value, we test both minimal and similarity/context-enriched graph schemas. The main novelty is evaluating whether epigenetic-context features and GAT/GATv2 attention improve guide-level AUPRC under class imbalance. We also reproduce Mak et al.'s metrics for direct paper comparison. CRISPRoffT, heterogeneous GNNs, GraphSAGE, and secondary-structure features remain stretch goals.
