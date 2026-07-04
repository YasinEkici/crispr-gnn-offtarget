# Off-target predictor positioning (for the thesis literature comparison)

Purpose: a single, grounded source for a **qualitative** positioning of this thesis against CRISPR-Cas9 off-target *predictor* papers. It feeds the thesis positioning table (S1). It is **not** a raw-score leaderboard: reported scores come from different data universes, negative-sampling schemes, splits, and class prevalences, so numeric values are not directly comparable.

Every field below is extracted only from each paper's own text (its `paper.md` or `original.pdf`). Fields not stated in a paper are marked `not stated`. Scope: off-target predictors only (imbalance / biology / architecture-component papers are excluded).

## Summary table

| Model / year | Data source(s) / assay | Test split | Negatives | Class distribution | Primary metric(s) | Indel |
| --- | --- | --- | --- | --- | --- | --- |
| DeepCRISPR (2018) | Genome-wide detection (GUIDE-seq, Digenome-seq, BLESS, HTGTS, IDLV); 13 cell types | Cell-type / multi-scenario; guide-disjoint `not stated` | Genome-wide mismatch loci; batch bootstrapping for imbalance | Negative-heavy | ROC-AUC + PR-AUC | mismatch-focused |
| CRISPR-Net (2020) | CIRCLE-seq, GUIDE-seq (+2 mismatch-only sets) | Train on CIRCLE-seq, test on independent sets; guide-disjoint `not stated` | gRNA-target pairs; cleaved-read pairs = positive, rest = negative | Negative-heavy | PR-AUC + ROC-AUC | yes (mismatch + indel) |
| CRISPR-IP (2022) | CIRCLE-seq, SITE-seq | **LOGOCV (leave-one-gRNA-out)** | Genome-wide gRNA-DNA pairs; severe imbalance | Negative-heavy | PR-AUC, ROC-AUC (+F1/Prec/Rec/Acc) | mismatch + bulge-aware coding |
| GCN-Vinodkumar (2021) | HEK293 (18 sgRNA) + K562 (12 sgRNA); genome-wide detection; ≤6 mismatches (Bowtie) | Link-prediction; cluster/bootstrap sampling; guide-disjoint `not stated` | Non-off-target pairs = negative links (0); bootstrap balancing | Negative-heavy | auROC | mismatch-only (≤6) |
| CRISPR-M (2024) | CIRCLE (indel+mismatch), GUIDE_I, +5 mismatch-only sets | **LOGOCV** on CIRCLE + cross-dataset (CIRCLE→GUIDE) | **Cas-OFFinder-generated** inactive loci (synthetic candidates) | Negative-heavy (e.g. 340 pos vs 252,539 neg) | AUROC, AUPRC (+Acc/Prec/Rec/F1/F2) | yes |
| CRISPR-BERT (2024) | 7 sets: 5 mismatch-only (K562, HEK293t, II4–II6) + 2 indel (I1, I2) | **5-fold CV + leave-one-sgRNA-out + independent test** | Candidate negatives; adaptive batch class-balancing | Negative-heavy (imbalance ratios vary, up to IR≈831) | AUROC, PR-AUC | yes |
| CCLMoff (2025) | Comprehensive set from **13 genome-wide detection technologies** (CIRCLE/GUIDE/SITE/CHANGE/DISCOVER/BLESS/Digenome/DIG/IDLV/HTGTS/SURRO/Extru-seq) | **Leave-one-sgRNA-out + cross-dataset "unseen sgRNA" generalization** (RNA LM pretrained) | **Cas-OFFinder-constructed** (≤6 mismatches +1 bulge) | Negative-heavy | AUROC, AUPRC | yes (mismatch + bulge) |
| **This thesis** | crisprSQL / Mak measured-only (epigenetic-nucleosome) | **Guide-disjoint single fixed split** + guide-cluster bootstrap + multi-seed | **Measured** negatives (Scheme A, `cleavage_freq ≤ 1e-5`); no synthetic; putative `measured=0` not ground truth | **Positive-heavy**, prevalence **0.900705** | **AUPRC** | mismatch |

## Per-paper detail

### 1. DeepCRISPR — Chuai et al. 2018
- Data source(s): genome-wide off-target detection (GUIDE-seq, Digenome-seq, BLESS, HTGTS, IDLV); ~160,000 off-target samples; on-target Haeussler benchmark (HCT116/HEK293T/HeLa/HL60); epigenetic features from 13 cell types.
- Test split: cell-type generalization / "eight testing scenarios"; guide-level disjointness `not stated`.
- Negative generation: mismatch loci across the genome; bootstrapping integrated into batch training to alleviate imbalance.
- Class distribution: negative-heavy (true off-target sites small among all mismatch loci).
- Primary metric(s): ROC-AUC and PR-AUC.
- Indel: mismatch-focused (sgRNA + off-target locus pairs); indels `not stated`.
- Source: abstract; "Off-target data sources"; results.

### 2. CRISPR-Net — Lin et al. 2020
- Data source(s): CIRCLE-seq and GUIDE-seq (indels + mismatches); two independent mismatch-only datasets.
- Test split: trained on CIRCLE-seq, evaluated on independent datasets; guide-level disjointness `not stated`.
- Negative generation: gRNA-target pairs; pairs with nuclease-cleaved reads = positive, remaining candidate pairs = negative.
- Class distribution: negative-heavy.
- Primary metric(s): PR-AUC, ROC-AUC.
- Indel: yes (mismatches and indels — core contribution).
- Source: abstract; methods.

### 3. CRISPR-IP — Zhang et al. 2022
- Data source(s): CIRCLE-seq and SITE-seq.
- Test split: LOGOCV (leave-one-gRNA-out cross-validation).
- Negative generation: genome-wide gRNA-DNA candidate pairs; severe class imbalance.
- Class distribution: negative-heavy (paper notes accuracy 0.989 is unable to evaluate objectively because of imbalance).
- Primary metric(s): PR-AUC, ROC-AUC (also F1, Precision, Recall, Accuracy).
- Indel: mismatch with bulge-aware coding scheme.
- Source: abstract; abbreviations (LOGOCV, PR-AUC, ROC-AUC); results.

### 4. GCN-Vinodkumar — Vinodkumar et al. 2021
- Data source(s): HEK293 (18 sgRNA) + K562 (12 sgRNA) = 30 sgRNA; whole-genome off-target via GUIDE-seq/Digenome-seq/BLESS/HTGTS/IDLV; ≤6 mismatches (Bowtie).
- Test split: link-prediction with cluster data sampling; bootstrapping to balance; guide-level disjointness `not stated`.
- Negative generation: sgRNA-target pairs producing no off-target = negative links (label 0); bootstrap from minor class.
- Class distribution: negative-heavy (heavily imbalanced).
- Primary metric(s): auROC (headline auROC 0.987 — different universe, not directly comparable to this project).
- Indel: mismatch-only (≤6).
- Source: abstract; dataset section.

### 5. CRISPR-M — Sun et al. 2024
- Data source(s): CIRCLE (indel + mismatch), GUIDE_I, plus five mismatch-only datasets (Table 1). CIRCLE from 10 gRNAs.
- Test split: LOGOCV (CIRCLE split into 10 parts by sgRNA) + cross-dataset (train CIRCLE, validate GUIDE_I).
- Negative generation: **Cas-OFFinder-generated** inactive off-target loci (synthetic candidates), e.g. 252,539 inactive indel and 325,039 inactive mismatch-only vs 340/7031 active.
- Class distribution: negative-heavy.
- Primary metric(s): AUROC, AUPRC (also Accuracy, Precision, Recall, F1, F2).
- Indel: yes.
- Source: "Datasets"; "Comparisons on the target sites containing both mismatches and indels" (LOGOCV).

### 6. CRISPR-BERT — Luo et al. 2024
- Data source(s): seven public datasets — five mismatch-only (K562, HEK293t, II4, II5, II6) and two mismatch+indel (I1, I2). K562/HEK293t integrated by Chuai et al.
- Test split: two-round five-fold cross-validation + leave-one-sgRNA-out procedure (K562/HEK293t) + an independent test set (imbalance ratio IR≈831).
- Negative generation: candidate non-off-target pairs (label 0); adaptive batch-wise class-balancing to combat imbalance noise.
- Class distribution: negative-heavy (datasets have varying imbalance ratios).
- Primary metric(s): AUROC, PR-AUC.
- Indel: yes.
- Source: abstract; "2.1 Dataset"; results (five-fold CV, leave-one-sgRNA-out).

### 7. CCLMoff — Du et al. 2025
- Data source(s): a curated comprehensive off-target dataset spanning 13 genome-wide detection technologies (Extru-seq, SITE-seq, CIRCLE-seq, DISCOVER-seq(+), CHANGE-seq, BLESS, GUIDE-seq, Digenome-seq, DIG-seq, IDLV, HTGTS, SURRO-seq).
- Test split: leave-one-sgRNA-out + cross-dataset generalization to unseen sgRNAs/datasets; RNA language model pretrained on RNAcentral.
- Negative generation: **Cas-OFFinder-constructed** negatives (up to 6 mismatches + 1 bulge for bulge-positives; mismatch otherwise) — synthetic candidates.
- Class distribution: negative-heavy.
- Primary metric(s): AUROC, AUPRC.
- Indel: yes (mismatch + bulge).
- Source: abstract; methods (dataset curation and negative construction); results (leave-one-sgRNA-out).

### ★ This thesis (reference row)
- Data source(s): crisprSQL / Mak-derived measured-only epigenetic-nucleosome universe.
- Test split: guide-disjoint single fixed split (`sprint2_main_seed42`); uncertainty via guide-cluster bootstrap, paired-difference, multi-seed.
- Negative generation: **measured** negatives only (Scheme A, `cleavage_freq ≤ 1e-5`); no synthetic candidates; putative `measured=0` rows are not validation/test ground truth.
- Class distribution: **positive-heavy**, positive prevalence 0.900705 (test: 1533 positives / 169 negatives).
- Primary metric(s): AUPRC (non-GNN bar: XGBoost F4).
- Indel: mismatch (measured universe).
- Source: `docs/thesis/notes/main_narrative_framing.md`.

## Cross-cutting observations (for the thesis framing)

1. **Opposite class direction.** Every literature comparator evaluates on a **negative-heavy** genome-wide assay universe; this thesis evaluates on a **positive-heavy** measured-only universe (prevalence 0.900705). PR-AUC/AUPRC is prevalence-sensitive, so raw AUPRC values are not comparable across these universes.
2. **Synthetic vs measured negatives.** The comparators construct negatives computationally (Cas-OFFinder candidate loci, or "all remaining pairs"); this thesis uses only **measured** negatives and never treats putative `measured=0` rows as ground truth. This is a label-integrity difference, not a tuning difference.
3. **Guide-level evaluation is common in recent work.** LOGOCV / leave-one-sgRNA-out appears in CRISPR-IP, CRISPR-M, CRISPR-BERT, and CCLMoff. Guide-level generalization is therefore not unique to this thesis; the thesis contribution is the **combined contract** (measured-only + guide-disjoint + train-only preprocessing + validation-only selection + no-test-tuning), not the split type alone.
4. **Why not cross-run models.** Because label definition, negative generation, and class prevalence differ, running these models on the measured-only universe (or this model on their universes) would not yield a valid numeric comparison; positioning is qualitative by necessity.

## `not stated` fields
- DeepCRISPR, CRISPR-Net, GCN-Vinodkumar: guide-level disjointness of the test split is `not stated` (not claimed as guide-disjoint; not extractable as such).
- DeepCRISPR: indel handling `not stated` (mismatch-focused).
- Headline numeric scores are omitted except where clearly stated (e.g. GCN-Vinodkumar auROC 0.987); all such values carry the "different universe — not directly comparable" caveat.
