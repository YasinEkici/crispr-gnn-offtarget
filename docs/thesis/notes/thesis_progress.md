# Thesis Progress Handoff

## 1. Purpose

This file is the durable handoff for thesis-writing progress, final scientific claim boundaries, and next editing work. It exists so future thesis-writing sessions do not depend on chat history.

## 2. Current Thesis State

The main thesis chapters `01_giris.tex`, `02_materyal_yontem.tex`, `03_deneysel_calismalar.tex`, and `04_sonuc_oneriler.tex` are structured first drafts. They contain the main narrative, citations, tables, figure references, and Sprint 9 robustness framing, but they are not final thesis text.

The latest chapter-text pass strengthened the literature-to-results bridge without changing the bounded scientific claim. `01_giris.tex` now contains a broader literature synthesis across sequence models, epigenetic/chromatin context, graph representations, attention/family-aware/FiLM-style modules, and evaluation robustness. `02_materyal_yontem.tex` now includes a compact model-family settings table. `03_deneysel_calismalar.tex` now explains why the experiments are ordered as a controlled evidence chain. `04_sonuc_oneriler.tex` now includes a literature-linked conclusion section that ties the final results back to sequence, context, graph, and evaluation literature while preserving the no-robust-AUPRC-superiority claim.

The front matter has been mostly cleaned. `metadata.tex` now contains the real thesis titles, thesis keywords, submission month/year, advisor name (`Doç. Dr. Mustafa Özgür CİNGİZ`), and student information for Kasım DELİACI (`21360859021`) and Yasin EKİCİ (`21360859029`). The final defense date remains an explicit placeholder until the real administrative information is available. Jury/chair placeholders remain in metadata only for the optional approval-page macro, but that page is not printed in the default 2026 DOCX-aligned flow. `main.tex` now contains a real ÖNSÖZ, a thesis-specific KISALTMALAR list, a thesis-specific SEMBOLLER list, and completed Turkish ÖZET and English SUMMARY drafts. The CV/ÖZGEÇMİŞ area is intentionally left as a placeholder for now. `appendices/ek-a.tex` still needs to be replaced or removed if no real appendix is needed.

The first two cover pages have been visually aligned to `BTU_BM_Tez_Yazım_Sablonu_2026_updated.docx`. The outer cover now uses the DOCX background artwork and page-centered blue title blocks, while the inner cover follows the DOCX's plain, logosuz, text-area-centered layout. The DOCX instructional note boxes were intentionally not reproduced in LaTeX output because they are template comments, not final thesis cover content.

Recent completed front-matter work:

- `metadata.tex`: real Turkish/English thesis titles, approval title, advisor name, student names/numbers, keywords, and June 2026 month-year fields were added. Defense-date placeholder was intentionally kept; jury/chair placeholders remain only for the optional approval-page macro.
- `main.tex` ÖNSÖZ: placeholder instructions were replaced with a one-page Turkish acknowledgements text thanking the advisor, department faculty, family, and friends.
- `main.tex` KISALTMALAR: the sparse template list was replaced with thesis-specific abbreviations such as AUPRC, AUROC, BCE, GATv2, GCN, GNN, MCC, sgRNA, SHAP, and XGBoost.
- `main.tex` SEMBOLLER: irrelevant template symbols were replaced with thesis notation such as `\mathcal{D}_0`, `y`, `f_c`, `g_i`, `o_{ij}`, `x_{ij}^{obs}`, `h_o`, `\hat{p}_{ij}`, and confusion-matrix counts.
- `main.tex` ÖZET/SUMMARY: placeholder abstract text was replaced with 2026 DOCX-compliant 250-750 word thesis-specific Turkish and English abstracts. Both avoid citations, figures, and tables, and keep the Sprint 9-bounded claim boundary.
- PDF spot checks were performed for metadata/title pages, KISALTMALAR, SEMBOLLER, ÖNSÖZ, ÖZET, and SUMMARY. These pages fit visually after the updates.
- Cover-page visual QA was performed by rendering the DOCX and LaTeX PDFs side by side. The outer-cover text centers now follow the page center; the inner-cover text centers follow the 4 cm / 2.5 cm text-area center.
- Two-student front-matter QA was performed by rendering the outer cover, inner cover, and plagiarism declaration. The cover pages show both students, the inner cover pairs each name with its student number, and the plagiarism declaration uses plural wording with two signature rows.
- Final front-heading QA compared the 2019 DOCX, 2026 DOCX, and current LaTeX render. The 2019 and 2026 DOCX files both place `ÖNSÖZ` at approximately `y=126.2`, while the current LaTeX render places it at `y=126.3`; therefore the separate foreword heading position is intentional rather than a 2026-only drift. The 2026 DOCX and current LaTeX list headings also match closely: `İÇİNDEKİLER`, `KISALTMALAR`, `SEMBOLLER`, `ÇİZELGE LİSTESİ`, and `ŞEKİL LİSTESİ` render at approximately `y=144.2`.

Recent completed chapter-text work:

- `01_giris.tex`: the literature review was expanded into five explicit axes: sequence/language-model predictors, epigenetic and chromatin context, graph representation and the Graph C gap, attention/family-aware/FiLM-style mechanisms, and evaluation robustness.
- `references.bib`: four additional literature entries were added and all citations were kept under the active BibLaTeX workflow.
- `02_materyal_yontem.tex`: a model-family architecture and training-settings table was added so XGBoost, sequence baselines, GCN/GAT/GATv2, family-aware encoders, FiLM, and fusion variants are easier to compare.
- `03_deneysel_calismalar.tex`: a `Deney Akışının Gerekçesi` section was added to explain why experiments move from non-graph baselines to graph schemas, feature/loss ablations, attention/mechanism analyses, family-aware/context interaction models, and robustness checks.
- `04_sonuc_oneriler.tex`: a `Literatürle İlişkilendirilen Sonuç` section was added to connect the final claims back to sequence, context, graph, and evaluation literature without turning the thesis into an unbounded model-superiority claim.
- After this pass, the Tectonic render produced a 65-page `main.pdf`. Static citation validation showed 44 `.bib` entries, 44 unique cited keys, 163 citation-key uses, no missing keys, and no uncited bibliography entries. The only remaining compile warning is the known `main.lof` overfull from the long paired-difference figure-list caption.

## 3. Binding Thesis Notes

- `btu_template_verification.md` controls BTU template and formatting expectations: A4 page, left 4 cm margin, other margins 2.5 cm, Times New Roman-style body, Arial-style outer cover, 12 pt body, 18 pt leading, justified text, heading spacing, caption sizing, and front-matter page-number behavior. A rendered PDF must be checked before final formatting judgment.
- `main_narrative_framing.md` controls the scientific story and claim boundaries. The thesis must not become a "GNN beats all" narrative. The correct story is context-aware Graph C/GATv2 variants improving rare-negative operating behavior in some validation-locked settings, while Sprint 9 does not support robust AUPRC superiority over XGBoost F4.
- `tez_yazim_meta_kurallari.md` controls thesis-writing style: academic Turkish, passive voice, no first person, APA author-date citations, no numbered citations, 250-750 word ÖZET/SUMMARY, figure/table references before placement, figure captions below, table captions above, and final references sorted alphabetically by author surname.

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
- `metadata.tex`: title, English title, approval title, advisor, student names/numbers, keywords, and submission/foreword month-year have been updated. The defense date still needs the real administrative value. Jury/chair fields are currently relevant only if the optional approval-page macro is re-enabled.
- `01_giris.tex`: perform a final copyedit for paragraph density and Turkish/English terminology consistency; keep the introduction centered on context-aware representation plus bounded AUPRC claims.
- `02_materyal_yontem.tex`: perform final consistency checks on method/table numbering, keep methods separate from interpretation, and ensure no method implies test-driven tuning.
- `03_deneysel_calismalar.tex`: perform final copyedit after the new experiment-flow rationale; keep AUPRC ranking separate from threshold operating behavior.
- `04_sonuc_oneriler.tex`: perform final copyedit after the new literature-linked conclusion; keep the no-robust-AUPRC-superiority claim, rare-negative operating-point contribution, guide/seed fragility, and external-validation need explicit.
- `appendices/ek-a.tex`: replace template content with a real appendix or remove appendix inclusion if no appendix is needed.

## 9. Figure/Table/PDF Work

All thesis-referenced result figures appear present under `docs/thesis/latex/btu_template/figures/results/`. A local `main.pdf` has been rendered during front-matter work for visual QA, but the PDF is treated as a generated artifact and is not part of the durable source handoff unless explicitly committed later.

Recent PDF checks verified that the updated title/metadata pages fit, the KISALTMALAR page fits on one page, the SEMBOLLER page fits on one page, the ÖNSÖZ page fits on one page with date/name alignment intact, and the ÖZET/SUMMARY pages each fit on one page with keywords visible. The 2026 template transition pass also verified the front-matter sequence, table/figure list rendering, bold list labels, caption punctuation, and the KAYNAKLAR hanging-indent layout. The remaining PDF QA pass must still check all figure placements, table width, page breaks, and whether result figures look thesis-ready rather than diagnostic-only. Sprint 9 figures and tables must remain visible enough to support the final claim boundary.

## 10. Literature/References Work

References now use an active `.bib` workflow. The thesis bibliography metadata lives in `docs/thesis/latex/btu_template/references.bib`; `main.tex` prints it through `btu-apa-biblatex.tex` with BibLaTeX `backend=bibtex`, which works under the bundled Tectonic runtime. In-text citations were converted from free-form author-year text to `\btucitep{...}` and `\btutextcite{...}` commands so citation keys can be checked statically against the `.bib` file. The custom bibliography layer preserves the BTU `KAYNAKLAR` heading, single spacing, 2.5 cm hanging indent, Turkish in-text `ve` / `ve diğ.` behavior, APA-style author ordering, italic source titles and volumes, `Article ...` numbers, DOI URL form, and consistent `In ...` proceedings entries.

The current static validation state after the latest literature pass is: 44 `.bib` entries, 44 unique cited keys, 163 citation-key uses, no missing citation keys, no uncited bibliography entries, and no remaining free-form author-year citation candidates in the chapter sources. The bibliography currently contains 39 DOI fields and 4 URL fields. Earlier DOI/URL validation checked 35 DOI fields and 4 URL fields before the latest four literature additions; a final DOI/URL validation should be repeated before thesis submission. A Tectonic PDF render and raster check of the rendered `KAYNAKLAR` pages verified that the bibliography is generated from `main.bbl`, remains visually readable, and keeps the expected hanging-indent layout.

Do not invent references. Any literature claim should be verified against `docs/literature/literature_index.md`, `docs/literature/paper_registry.yaml`, and the relevant local literature notes. CRISPR-Net, CRISPR-IP, DeepCRISPR, GCN/GATv2, FiLM/SENet-style modules, leakage/overtuning, AUPRC, and bootstrap/uncertainty references must be framed as local adaptations or methodological anchors, not reproductions unless the full setup matches.

## 11. ÖZET/SUMMARY Status

ÖZET and SUMMARY have been drafted in `main.tex`. The current word counts are approximately 322 words for the Turkish ÖZET and 394 words for the English SUMMARY. Both are within the updated 2026 DOCX 250-750 word requirement and contain no citations, figures, tables, or raw reference markers.

Core abstract ingredients:

- Contribution: a controlled, context-aware GNN workflow for CRISPR-Cas9 off-target prediction under guide-disjoint measured-only evaluation.
- Final result: XGBoost F4 remains the strongest robust AUPRC reference.
- Robustness statement: Sprint 9 found no paired AUPRC difference excluding zero, and GNN seed variability exceeded the small Sprint 8 candidate gains.
- Limitation statement: conclusions are limited by 29 test guides, 169 negatives concentrated in 9 guides, guide `9251` dominance, one fixed split, no external validation, and no causal biological interpretation.

## 12. Prioritized Next Editing Plan

1. Replace the remaining defense-date and optional jury/chair placeholders when the real administrative information is available.
2. Leave CV/ÖZGEÇMİŞ placeholder until real CV details are provided.
3. Perform a full chapter-level copyedit for paragraph density, terminology consistency, and repeated claim wording.
4. Render the thesis PDF and perform full figure/table/layout QA.
5. Repeat final citation, DOI/URL, cross-reference, overflow, caption, and bibliography validation checks.
6. Fix the known long figure-list caption warning if the final formatting pass requires a warning-clean build.

## 13. 2026 Template Transition Status

`BTU_BM_Tez_Yazım_Sablonu_2026_updated.docx` was rendered with the Documents workflow and its embedded comments were inspected. The current LaTeX template was not replaced wholesale with the shared graduate LaTeX template because that template is graduate-oriented and uses an older LaTeX stack. Instead, transferable rules were applied to the current XeLaTeX/Tectonic-compatible undergraduate template.

Completed in this pass:

- `btu-thesis.cls`: KAYNAKLAR now renders with 2.5 cm hanging indent, single spacing, justification, and bold figure/table list labels.
- `btu-thesis.cls`: the 2026 DOCX outer-cover background artwork is used on the LaTeX outer cover; cover text placement, font sizes, and color were adjusted against rendered DOCX page measurements.
- `btu-thesis.cls`: the inner cover was changed to the 2026 DOCX plain text layout, with an optional second-advisor line controlled by `\BTUSecondAdvisor`.
- `main.tex`: front matter now follows the 2026 DOCX render sequence: outer cover, inner cover, blank page, plagiarism statement, foreword.
- `02_materyal_yontem.tex` and `03_deneysel_calismalar.tex`: figure captions now end with periods.
- `tez_yazim_meta_kurallari.md`: updated abstract range, caption punctuation, and reference-list formatting to match 2026 DOCX notes.
- `template_2026_migration_audit.md`: added as a durable migration note with remaining front-matter decisions.
- `template_2026_migration_audit.md`: records the final 2019/2026/LaTeX `ÖNSÖZ` and list-heading coordinate check.
