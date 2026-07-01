# Poster Export Handoff

This is the Slice 9 export handoff for the thesis-poster planning track. It packages the current state for the human/team member who will produce the final poster in Canva, LaTeX, slides, or another tool.

This folder does not contain a final poster export yet. No print-approved PDF/image has been produced in this planning track. There are now two LaTeX working artifacts: the earlier low-fidelity fit-check and the newer BTU school-template transfer draft. The BTU template draft is the current practical working direction, but it is still not a final print-approved export.

The Turkish poster copy source of truth is `docs/poster/plan/poster_copy_deck.md`. Future design/layout passes should pull from that deck, not from the older `latex_fit_check` draft.

## 1. Current status

Status: `handoff-ready with required pre-export fixes`

The planning and draft-preparation package is ready to hand to a designer/teammate, but it is not ready for final printing.

Completed:

- poster constitution and narrative planning;
- Turkish writing rules;
- section-by-section content plan;
- slice-based execution plan;
- section shortlist;
- figure-production plan;
- low-fidelity layout brief;
- Turkish microcopy draft;
- editable SVG working assets;
- LaTeX 70x100 cm fit-check draft;
- BTU school-template LaTeX transfer draft;
- claim/number audit;
- print/accessibility source review.

Not completed:

- final title selection;
- final production-tool decision;
- final high-fidelity design;
- real exported proof;
- physical or PDF print proof;
- grayscale/colorblind check on an actual export;
- final reference formatting;
- final PDF/image handoff.
- final student number/email fields.

## 2. Required pre-export fixes

These must be handled before any final print/export.

1. **Keep the honesty caveat single.**
   The legacy fit-check duplicated the exact honesty caveat between `fig03_two_axis_results.svg` and `docs/poster/drafts/latex_fit_check/poster_fit_check.tex`. The active BTÜ draft keeps it once inside the results band, currently as the layout-owned `Okuma.` line:

   > `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

2. **Create/review a real exported proof.**
   The active BTÜ draft compiles to a 70x100 cm proof PDF. Review a final full-size proof again after every layout/figure-redraw pass before print decisions.

3. **Verify Turkish rendering.**
   Check Turkish characters and names in the exported proof, especially:
   `"Kasım Deliacı"`, `"Doç. Dr. Mustafa Özgür Cingiz"`, `"Bursa Teknik Üniversitesi"`.

4. **Run print/accessibility checks on the exported proof.**
   Check caveat readability, grayscale distinguishability, colorblind resilience, and text density.

5. **Keep title status explicit.**
   The current title is a candidate used for fit-checking. The final title remains deferred until design review.

6. **Fill footer placeholders.**
   The BTU school-template draft currently keeps student number and email fields as placeholders because the repository does not contain final values.

## 3. Package inventory

### Governing notes

- `docs/poster/notes/poster_design_decisions.md` - poster constitution and non-negotiable design/narrative decisions.
- `docs/poster/notes/poster_narrative_framing.md` - narrative spine, two-axis story, claim boundary, numeric anchors.
- `docs/poster/notes/poster_yazim_kurallari.md` - Turkish writing, visual-language, number-formatting, and caption rules.
- `docs/poster/notes/poster_progress.md` - lightweight progress tracker.

### Planning docs

- `docs/poster/plan/poster_copy_deck.md` - canonical Turkish text source, with full and short variants per section.
- `docs/poster/plan/poster_content_plan.md` - section-by-section content and figure plan.
- `docs/poster/plan/poster_execution_plan.md` - slice tracker.
- `docs/poster/plan/poster_section_shortlist.md` - keep/compress/cut decisions.
- `docs/poster/plan/figure_production_plan.md` - must-have and optional figure specs.
- `docs/poster/plan/layout_draft_brief.md` - 70x100 cm layout-zone strategy.
- `docs/poster/plan/poster_microcopy_draft.md` - Turkish microcopy candidate bank.
- `docs/poster/plan/poster_claim_number_audit.md` - claim and number audit.
- `docs/poster/plan/poster_print_accessibility_review.md` - source-based print/accessibility review.

### Working assets

- `docs/poster/assets/fig01_measured_only_funnel.svg`
- `docs/poster/assets/fig02_graph_abc_semantic_comparison.svg`
- `docs/poster/assets/fig03_two_axis_results.svg`
- `docs/poster/assets/fig04_tn169_rare_negative_recognition.svg`
- `docs/poster/assets/fig05_literature_ab_positioning.svg`
- `docs/poster/assets/figure_sources.md`

### Low-fidelity draft

- `docs/poster/drafts/latex_fit_check/poster_fit_check.tex`
- `docs/poster/drafts/latex_fit_check/README.md`

This LaTeX draft is useful for content fit and spatial hierarchy. It is not a final source file unless the team later chooses LaTeX as the production tool and revises it accordingly.

### BTU school-template draft

- `docs/poster/templates/btu_poster_latex_template/` - untouched copy of the school-provided template package.
- `docs/poster/drafts/btu_school_template/poster.tex` - first content transfer into the BTU header/footer and section-box layout.
- `docs/poster/drafts/btu_school_template/poster.pdf` - generated proof when compiled.
- `docs/poster/drafts/btu_school_template/README.md` - notes for compiling and reviewing the school-template draft.

This is the current LaTeX working draft for reviewing the poster inside the school format. It preserves the BTU header/footer and main section order, but box sizes, figure sizing, text density, final title, and footer metadata still need review.

The older `docs/poster/drafts/latex_fit_check/poster_fit_check.tex` should be treated only as a historical content-fit preview.

## 4. Claim guardrails for final production

Do not change these boundaries while moving into Canva, LaTeX, slides, or another tool.

- No robust AUPRC superiority over XGBoost F4.
- AUPRC remains the primary ranking metric.
- MCC and specificity are operating-point evidence, not replacement primary metrics.
- Threshold and rare-negative gains are validation-locked and seed/guide sensitive.
- `0.900705` is the no-skill PR baseline, not a performance floor.
- `measured=0` rows are not validation/test ground truth.
- Graph arrows, model internals, attention, gates, FiLM, masking, embeddings, and feature-family visuals are not biological-causality evidence.
- Literature comparison stays qualitative; no raw-score leaderboard against other papers.
- Where intervals include zero, use compatibility language, not equivalence language.

## 5. Approved scientific number pool

Use only these poster numbers unless a later value is verified directly against the thesis and approved.

- `310142 -> 25632 -> 1702`
- test: 29 guides, 1533 positives, 169 negatives
- negatives in 9 guides; 80 in guide 9251
- prevalence `0.900705` as no-skill PR baseline
- XGBoost F4 AUPRC `0.992338 [0.950179, 0.999336]`
- XGBoost F4 multi-seed `0.990649 ± 0.001944`
- historical XGBoost F4 `0.992522` only as version-drift note
- S8B_R2 AUPRC `0.986020 [0.929981, 0.998966]`
- S8B_R2 multi-seed `0.978963 ± 0.011322`
- true negatives over 169: XGBoost F4 = 40, Graph C GCN = 14, Graph C GATv2 = 63, family-aware encoder = 110
- specificities `0.236`, `0.083`, `0.373`, `0.651`
- family-aware encoder MCC `0.603489`
- feature families: 6 experimental-epigenetic + 13 computed-nucleosome + 5 binding-energy
- 154 sgRNA, 138747 target locations

## 6. Final export checklist

Before handing a file to print:

- [ ] Final production tool is documented.
- [ ] Final title is chosen and reviewed.
- [ ] Exact honesty caveat appears once, inside the results band.
- [ ] Every scientific number is traceable to the approved pool or thesis.
- [ ] AUPRC and operating-point results are visually separated.
- [ ] TN/169 visual uses the shared denominator clearly.
- [ ] `0.900705` is labeled as no-skill PR baseline, not floor.
- [ ] `measured=0` is not treated as test ground truth.
- [ ] Seed/guide fragility is readable near operating-point evidence.
- [ ] No-causality caveat is readable near Graph/model visuals.
- [ ] Literature panel remains qualitative and has no raw-score leaderboard.
- [ ] Turkish characters render correctly.
- [ ] Grayscale proof remains understandable.
- [ ] Colorblind check passes or labels/shapes fully preserve meaning.
- [ ] Main title, headings, metric labels, captions, and caveats are readable at 70x100 cm print scale.
- [ ] Final source file is saved.
- [ ] Exported print PDF/image is saved.
- [ ] Source assets are collected.
- [ ] References/citations are included in compact final form.

## 7. Tool-specific handoff notes

### If Canva is used

- Import the SVG assets directly or redraw them with the same labels and caveats.
- Preserve the Apple-like / keynote-inspired discipline: low text density, large visual anchors, restrained palette, and generous negative space.
- Do not hide caveats as tiny footer text.
- Keep the exact honesty caveat once inside the result area.

### If LaTeX is used

- Start from `docs/poster/drafts/btu_school_template/poster.tex` if the school-provided BTU format is required.
- Use `docs/poster/drafts/latex_fit_check/poster_fit_check.tex` only as the earlier content-fit reference.
- Treat both as working drafts until a print proof is reviewed.
- Compile with XeLaTeX and SVG support if possible.
- Check SVG text rendering carefully; direct SVG conversion may change fonts or line breaks.
- Remove the duplicate honesty caveat before final export.

### If slides are used

- Use 70x100 cm portrait page setup or equivalent custom dimensions.
- Place SVGs as editable/vector assets where possible.
- Keep the result band and TN/169 visual large enough for print.
- Apply the same claim and number audit to the slide export.

## 8. Handoff recommendation

Recommended next human/team action:

1. Review `docs/poster/drafts/btu_school_template/poster.pdf`.
2. Fill student number/email placeholders.
3. Reconcile any text edits against `docs/poster/plan/poster_copy_deck.md`.
4. Select the final title.
5. Tune school-template box sizes, figure sizing, and text density.
6. Produce one full-size exported proof.
7. Run the checklist in `docs/poster/plan/poster_print_accessibility_review.md`.
8. Run the claim/number checklist in `docs/poster/plan/poster_claim_number_audit.md`.
9. Collect final source, export, assets, and references in this `docs/poster/final/` folder or another explicitly chosen final-delivery location.

Do not edit the thesis for poster production. Do not add new poster numbers without tracing them to the thesis.
