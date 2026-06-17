# Thesis Progress Handoff

## 1. Purpose

This file is the durable handoff for thesis-writing progress, final scientific claim boundaries, and next editing work. It exists so future thesis-writing sessions do not depend on chat history.

## 2. Current Thesis State

The main thesis chapters `01_giris.tex`, `02_materyal_yontem.tex`, `03_deneysel_calismalar.tex`, and `04_sonuc_oneriler.tex` are structured first drafts. They contain the main narrative, citations, tables, figure references, and Sprint 9 robustness framing, but they are not final thesis text.

The front matter has been mostly cleaned. `metadata.tex` now contains the real thesis titles, thesis keywords, submission month/year, and advisor name (`Doç. Dr. Mustafa Özgür CİNGİZ`). Student name/number, jury, chair, and defense date remain explicit placeholders until the real administrative information is available. `main.tex` now contains a real ÖNSÖZ, a thesis-specific KISALTMALAR list, a thesis-specific SEMBOLLER list, and completed Turkish ÖZET and English SUMMARY drafts. The CV/ÖZGEÇMİŞ area is intentionally left as a placeholder for now. `appendices/ek-a.tex` still needs to be replaced or removed if no real appendix is needed.

Recent completed front-matter work:

- `metadata.tex`: real Turkish/English thesis titles, approval title, advisor name, keywords, and June 2026 month-year fields were added. Student, jury, chair, and defense-date placeholders were intentionally kept.
- `main.tex` ÖNSÖZ: placeholder instructions were replaced with a one-page Turkish acknowledgements text thanking the advisor, department faculty, family, and friends.
- `main.tex` KISALTMALAR: the sparse template list was replaced with thesis-specific abbreviations such as AUPRC, AUROC, BCE, GATv2, GCN, GNN, MCC, sgRNA, SHAP, and XGBoost.
- `main.tex` SEMBOLLER: irrelevant template symbols were replaced with thesis notation such as `\mathcal{D}_0`, `y`, `f_c`, `g_i`, `o_{ij}`, `x_{ij}^{obs}`, `h_o`, `\hat{p}_{ij}`, and confusion-matrix counts.
- `main.tex` ÖZET/SUMMARY: placeholder abstract text was replaced with 300-500 word thesis-specific Turkish and English abstracts. Both avoid citations, figures, and tables, and keep the Sprint 9-bounded claim boundary.
- PDF spot checks were performed for metadata/title pages, KISALTMALAR, SEMBOLLER, ÖNSÖZ, ÖZET, and SUMMARY. These pages fit visually after the updates.

## 3. Binding Thesis Notes

- `btu_template_verification.md` controls BTU template and formatting expectations: A4 page, left 4 cm margin, other margins 2.5 cm, Times New Roman-style body, Arial-style outer cover, 12 pt body, 18 pt leading, justified text, heading spacing, caption sizing, and front-matter page-number behavior. A rendered PDF must be checked before final formatting judgment.
- `main_narrative_framing.md` controls the scientific story and claim boundaries. The thesis must not become a "GNN beats all" narrative. The correct story is context-aware Graph C/GATv2 variants improving rare-negative operating behavior in some validation-locked settings, while Sprint 9 does not support robust AUPRC superiority over XGBoost F4.
- `tez_yazim_meta_kurallari.md` controls thesis-writing style: academic Turkish, passive voice, no first person, APA author-date citations, no numbered citations, 300-500 word ÖZET/SUMMARY, figure/table references before placement, figure captions below, table captions above, and final references sorted alphabetically by author surname.

## 4. Final Scientific Claim Boundary

- Label scheme: Scheme A, `cleavage_freq > 1e-5`.
- Main split: `sprint2_main_seed42`, guide-disjoint.
- Headline evaluation: measured-only validation/test rows; `experiment_id = 18` excluded.
- Primary metric: AUPRC.
- Secondary threshold metrics: MCC, specificity, macro-F1, AUROC, and TN/FP/FN/TP at validation-selected thresholds.
- No test-driven tuning: no test-set threshold, hyperparameter, architecture, feature, or model selection.
- XGBoost F4 is the strongest non-GNN reference and remains the highest-AUPRC bar.
- Sprint 8A and Sprint 8B candidates are validation-selected model-improvement candidates, not superiority claims.
- Sprint 9 is completed and controls final thesis claim strength.
- Attention, gates, FiLM weights, feature masks, and embeddings may support model-behavior interpretation only; they do not prove biological causality.
- No literature method may be called a reproduction unless dataset, split, target, metrics, and architecture match the cited paper's setup.

## 5. Sprint 9 Robustness Takeaway

Exact Sprint 9-supported claim: no Sprint 8 GNN candidate robustly beats XGBoost F4 on AUPRC. All predeclared paired AUPRC difference intervals include zero. GNN seed variance exceeds the small Sprint 8 candidate gains. Threshold-dependent rare-negative behavior exists, but it does not override AUPRC.

Key Sprint 9 AUPRC values:

- XGB_F4 regenerated AUPRC `0.992338`, CI `[0.950179, 0.999336]`, multiseed `0.990649 ± 0.001944`.
- S8B_R2 AUPRC `0.986020`, CI `[0.929981, 0.998966]`, multiseed `0.978963 ± 0.011322`.
- S8A_R2 AUPRC `0.982757`, CI `[0.910478, 0.999892]`, multiseed `0.975538 ± 0.012187`.
- S8B_R2 - S8A_R2 paired AUPRC delta `+0.003263`, CI `[-0.01484, +0.03706]`.
- S8B_R2 - XGB_F4 paired AUPRC delta about `-0.00632`, CI `[-0.03124, +0.00117]`.

Key limitation: the test set has `1702` rows but only `29` guides, `169` negatives, negatives in only `9` guides, and guide `9251` contains `80/169` negatives. Uncertainty is therefore governed by guide composition, not by the row count alone.

## 6. Supported Thesis Narrative

The project developed a controlled context-aware GNN workflow for CRISPR-Cas9 off-target prediction under a measured-only, guide-disjoint, no-test-tuning evaluation protocol. Context-aware Graph C/GATv2 variants improved rare-negative operating behavior in some validation-locked settings, especially through target-observation context and family-aware/context interaction designs. Sprint 9 does not support robust AUPRC superiority over XGBoost F4, so the final thesis contribution is a bounded representation-and-evaluation result rather than a state-of-the-art predictor claim.

## 7. Current Draft Risks

- Do not claim GNN superiority over XGBoost F4 on AUPRC.
- Avoid statistical significance language unless a paired interval directly justifies it, and even then keep the finite-sample guide-cluster caveat.
- Avoid equivalence or non-inferiority language; no prespecified margin exists.
- Avoid biological causal interpretation from attention, gates, FiLM, feature masks, or embeddings.
- Avoid global claims that sequence models fail; only the local S1/Conv-BiLSTM-style paths under this frozen contract failed.
- Avoid saying the AUPRC prevalence baseline is a floor. It is a no-skill PR baseline.
- Avoid implying that 1702 test rows make uncertainty small; the effective limitation is 29 guides and 9 negative-bearing guides.

## 8. Chapter-Level Next Work

- `main.tex` / front matter: ÖNSÖZ, KISALTMALAR, SEMBOLLER, Turkish ÖZET, and English SUMMARY have been updated. CV/ÖZGEÇMİŞ should remain placeholder until real personal CV details are provided.
- `metadata.tex`: title, English title, approval title, advisor, keywords, and submission/foreword month-year have been updated. Student name/number, jury, chair, and defense date still need real administrative values.
- `01_giris.tex`: split dense paragraphs, standardize Turkish/English terminology, and keep the introduction centered on context-aware representation plus bounded AUPRC claims.
- `02_materyal_yontem.tex`: keep methods separate from interpretation, verify robustness wording as finite-sample guide-cluster compatibility, and ensure no method implies test-driven tuning.
- `03_deneysel_calismalar.tex`: tighten Sprint 9 as the controlling claim-boundary section; separate AUPRC ranking from threshold operating behavior; revise any wording that sounds like unqualified significance or causal mechanism.
- `04_sonuc_oneriler.tex`: sharpen the final conclusion around no robust AUPRC superiority, rare-negative operating-point contribution, guide/seed fragility, and need for external guide-diverse validation.
- `appendices/ek-a.tex`: replace template content with a real appendix or remove appendix inclusion if no appendix is needed.

## 9. Figure/Table/PDF Work

All thesis-referenced result figures appear present under `docs/thesis/latex/btu_template/figures/results/`. A local `main.pdf` has been rendered during front-matter work for visual QA, but the PDF is treated as a generated artifact and is not part of the durable source handoff unless explicitly committed later.

Recent PDF checks verified that the updated title/metadata pages fit, the KISALTMALAR page fits on one page, the SEMBOLLER page fits on one page, the ÖNSÖZ page fits on one page with date/name alignment intact, and the ÖZET/SUMMARY pages each fit on one page with keywords visible. The remaining PDF QA pass must still check captions, table width, figure placement, cross-references, table/figure lists, page breaks, front-matter numbering, and whether result figures look thesis-ready rather than diagnostic-only. Sprint 9 figures and tables must remain visible enough to support the final claim boundary.

## 10. Literature/References Work

References are currently manual in `main.tex`; no `.bib` file was found. Final APA consistency, author-date formatting, and alphabetical ordering must be checked manually unless a bibliography workflow is introduced later.

Do not invent references. Any literature claim should be verified against `docs/literature/literature_index.md`, `docs/literature/paper_registry.yaml`, and the relevant local literature notes. CRISPR-Net, CRISPR-IP, DeepCRISPR, GCN/GATv2, FiLM/SENet-style modules, leakage/overtuning, AUPRC, and bootstrap/uncertainty references must be framed as local adaptations or methodological anchors, not reproductions unless the full setup matches.

## 11. ÖZET/SUMMARY Status

ÖZET and SUMMARY have been drafted in `main.tex`. The current word counts are approximately 322 words for the Turkish ÖZET and 394 words for the English SUMMARY. Both are within the 300-500 word requirement and contain no citations, figures, tables, or raw reference markers.

Core abstract ingredients:

- Contribution: a controlled, context-aware GNN workflow for CRISPR-Cas9 off-target prediction under guide-disjoint measured-only evaluation.
- Final result: XGBoost F4 remains the strongest robust AUPRC reference.
- Robustness statement: Sprint 9 found no paired AUPRC difference excluding zero, and GNN seed variability exceeded the small Sprint 8 candidate gains.
- Limitation statement: conclusions are limited by 29 test guides, 169 negatives concentrated in 9 guides, guide `9251` dominance, one fixed split, no external validation, and no causal biological interpretation.

## 12. Prioritized Next Editing Plan

1. Replace remaining administrative placeholders when real student, jury, chair, and defense-date information is available.
2. Leave CV/ÖZGEÇMİŞ placeholder until real CV details are provided.
3. Tighten Sprint 9 claim wording throughout chapters 03 and 04.
4. Split dense chapter paragraphs and improve subsection structure where needed.
5. Standardize Turkish/English terminology beyond the already-updated KISALTMALAR and SEMBOLLER pages.
6. Clean citation and reference formatting against APA-style author-date rules.
7. Render the thesis PDF and perform full figure/table/layout QA.
8. Run final render and validation checks, then fix any cross-reference, overflow, caption, or bibliography issues.
