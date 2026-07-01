# Poster Claim and Number Audit

This document is the Slice 7 claim-boundary and number audit for the current poster working draft. It audits the low-fidelity LaTeX fit-check and SVG assets, not a final production poster.

Status values:

- `pass` - acceptable for the current working draft.
- `watch` - acceptable for fit-checking, but should be revisited before final production.
- `fix-needed` - should be changed before any higher-fidelity poster draft.

## 1. Scope

Audited files:

- `docs/poster/drafts/latex_fit_check/poster_fit_check.tex`
- `docs/poster/drafts/latex_fit_check/README.md`
- `docs/poster/assets/fig01_measured_only_funnel.svg`
- `docs/poster/assets/fig02_graph_abc_semantic_comparison.svg`
- `docs/poster/assets/fig03_two_axis_results.svg`
- `docs/poster/assets/fig04_tn169_rare_negative_recognition.svg`
- `docs/poster/assets/fig05_literature_ab_positioning.svg`
- `docs/poster/assets/figure_sources.md`

Governed by:

- `docs/poster/notes/poster_design_decisions.md`
- `docs/poster/notes/poster_narrative_framing.md`
- `docs/poster/notes/poster_yazim_kurallari.md`
- `docs/poster/plan/poster_content_plan.md`
- `docs/poster/plan/poster_microcopy_draft.md`

Out of scope:

- final visual polish;
- final title selection;
- print legibility measurement;
- LaTeX compilation;
- thesis edits;
- new figure production.

## 2. Overall audit result

Overall status: `watch`

The current fit-check is scientifically usable as a working draft: the main claim boundary is preserved, AUPRC and operating-point evidence are separated, numbers come from the approved anchor pool, and literature positioning avoids a raw-score leaderboard.

The main issue is not a scientific claim violation but a production-readiness issue: the exact honesty caveat appears both inside `fig03_two_axis_results.svg` and again as a separate LaTeX caveat in `poster_fit_check.tex`. Before a higher-fidelity draft, keep the caveat visually present once inside the results band.

## 3. Claim-boundary checklist

| Check | Status | Audit note |
| --- | --- | --- |
| No robust AUPRC superiority over XGBoost F4 | `pass` | The draft says XGBoost F4 remains the AUPRC bar and includes `"AUPRC üstünlüğü iddia edilmedi."` |
| Exact honesty caveat inside results | `watch` | Present in the result area, but duplicated between `fig03_two_axis_results.svg` and `poster_fit_check.tex`. Keep one visual occurrence later. |
| AUPRC remains primary ranking metric | `pass` | `fig03` labels `"Sıralama (AUPRC)"` as the primary ranking metric. |
| MCC/specificity framed as operating-point evidence | `pass` | `poster_fit_check.tex`, `fig03`, `fig04`, and `figure_sources.md` frame MCC/specificity as decision-threshold / operating-point evidence. |
| Operating-point gains marked seed/guide-fragile | `pass` | `fig04` and the LaTeX limits block state validation-locked and seed/guide sensitivity. |
| `0.900705` labeled no-skill PR baseline, not floor | `pass` | `fig03`, `poster_fit_check.tex`, README, and `figure_sources.md` use no-skill baseline wording and explicitly avoid floor wording. |
| `measured=0` not treated as validation/test ground truth | `pass` | `fig01` states `"Doğrulanmamış adaylar (measured=0) test etiketi yapılmadı."` |
| No biological-causality claim from model internals or graph arrows | `pass` | `fig02` and the limits block state representation/model behavior only, not biological causality. |
| No "sequence models fail" generalization | `pass` | No such phrasing appears in the audited draft/assets. |
| No raw-score leaderboard against other papers | `pass` | `fig05` uses qualitative literature positioning and explicitly says no raw AUPRC leaderboard. |
| No "equivalent" claim where intervals overlap | `pass` | The audited fit-check does not claim equivalence. |

## 4. Numeric anchor audit

The following scientific poster numbers appear in the audited files and are in the approved anchor pool.

| Number / expression | Where used | Approved source category | Status |
| --- | --- | --- | --- |
| `310142 -> 25632 -> 1702` | `fig01`, `figure_sources.md` | Universe -> measured-only -> test | `pass` |
| `29` guides | `fig01`, `figure_sources.md` | Test composition | `pass` |
| `1533` positives | `fig01`, `poster_fit_check.tex`, `figure_sources.md` | Test composition | `pass` |
| `169` negatives | `fig01`, `fig03`, `fig04`, `poster_fit_check.tex`, `figure_sources.md` | Test composition / TN denominator | `pass` |
| `9` guides with negatives | `poster_fit_check.tex` | Negative concentration | `pass` |
| `80` in guide `9251` | `poster_fit_check.tex` | Negative concentration | `pass` |
| `0.900705` | `fig03`, `poster_fit_check.tex`, README, `figure_sources.md` | No-skill PR baseline | `pass` |
| `0.992338` | `fig03`, `figure_sources.md` | XGBoost F4 regenerated AUPRC | `pass` |
| `[0.950179, 0.999336]` | `fig03`, `figure_sources.md` | XGBoost F4 guide-cluster interval | `pass` |
| `0.986020` | `fig03`, `figure_sources.md` | S8B_R2 AUPRC | `pass` |
| `[0.929981, 0.998966]` | `fig03`, `figure_sources.md` | S8B_R2 guide-cluster interval | `pass` |
| `40/169`, `14/169`, `63/169`, `110/169` | `fig03`, `fig04`, `figure_sources.md` | True negatives over 169 | `pass` |
| `0.236`, `0.083`, `0.373`, `0.651` | `fig04`, `figure_sources.md` | Specificities | `pass` |
| `0.603489` | `fig03`, `fig04`, `figure_sources.md` | Family-aware encoder MCC | `pass` |

Numbers that appear only as layout, SVG, LaTeX, typography, dimensions, colors, coordinates, or file names are not scientific poster anchors. Examples: `70x100 cm`, SVG canvas sizes, font sizes, RGB/hex colors, path coordinates, line widths, file names such as `fig03`, and LaTeX spacing values.

## 5. File-by-file audit

### `poster_fit_check.tex`

Status: `watch`

Passes:

- Uses 70x100 cm portrait as a fit-check canvas.
- Keeps production tool non-binding in the footer.
- Places Graph C, measured-only funnel, two-axis results, TN/169 visual, literature panel, contribution chips, and limits in one integrated surface.
- Separates AUPRC from MCC/specificity.
- Labels `0.900705` as no-skill PR baseline, not a floor.
- Includes no-AUPRC-superiority, seed/guide sensitivity, and no-causality limits.

Watch items:

- The exact honesty caveat is repeated because `fig03_two_axis_results.svg` already contains it. Keep only one occurrence in the final integrated design.
- The title currently uses a candidate title as if selected. This is acceptable for fit-checking, but the final title remains deferred.
- LaTeX was not compiled in the current environment; visual overlap and print legibility remain for Slice 8.

### `fig01_measured_only_funnel.svg`

Status: `pass`

Passes:

- Uses only approved data-contract anchors.
- States `measured=0` rows were not made test labels.
- Avoids treating unverified candidates as negatives.

Watch items:

- None for claim/number audit.

### `fig02_graph_abc_semantic_comparison.svg`

Status: `pass`

Passes:

- Frames Graph C as target-observation representation.
- Says Graph C changes node semantics, not just topology.
- Explicitly says arrows do not show biological causality.

Watch items:

- None for claim/number audit.

### `fig03_two_axis_results.svg`

Status: `watch`

Passes:

- Separates `"Sıralama (AUPRC)"` and `"Karar Eşiği"`.
- Uses approved AUPRC, interval, no-skill baseline, MCC, and TN anchors.
- Frames MCC/specificity as operating-point evidence.
- Contains the exact honesty caveat inside the results panel.

Watch items:

- In the integrated LaTeX fit-check, this caveat is duplicated outside the SVG. Keep one occurrence later.
- The no-skill baseline line is schematic rather than a full PR plot. This is acceptable for fit-checking, but final visual styling should avoid implying a precise plotted PR curve unless a real curve is drawn.

### `fig04_tn169_rare_negative_recognition.svg`

Status: `pass`

Passes:

- Uses the same denominator, `169`, for every model.
- Shows the non-monotone Graph C GCN result (`14/169`) rather than hiding it.
- Includes validation-locked and seed/guide-sensitive caveat.
- Uses approved specificity and MCC anchors.

Watch items:

- None for claim/number audit. Print readability of the caveat belongs to Slice 8.

### `fig05_literature_ab_positioning.svg`

Status: `pass`

Passes:

- Uses qualitative axes only.
- States why direct comparison is not valid.
- Explicitly avoids raw AUPRC leaderboard framing.

Watch items:

- No named-paper detail appears yet. This is acceptable for fit-checking; final references can be added later without turning the panel into a score table.

### `figure_sources.md`

Status: `pass`

Passes:

- Records source categories, allowed numbers, and caveats for each figure.
- Keeps final tool non-binding.
- Repeats the global claim rules correctly.

Watch items:

- None for claim/number audit.

## 6. Required fixes before higher-fidelity draft

Fix-needed:

1. Keep the exact honesty caveat visually present once inside the results band. Current fit-check repeats it in `fig03_two_axis_results.svg` and `poster_fit_check.tex`.

No scientific-number fixes are required based on this audit.

## 7. Watch list for Slice 8

These are not claim failures, but they should be checked in the print/accessibility pass:

- Whether the exact honesty caveat remains readable after SVG scaling.
- Whether `fig03` and `fig04` together make the operating-point story look like a global model win.
- Whether `0.900705` is visually read as a reference baseline rather than a performance floor.
- Whether Graph A/B/C arrows are visually calm enough to avoid causal overread.
- Whether the candidate title looks final; title is still deferred.
- Whether LaTeX/SVG rendering preserves Turkish characters and line breaks.
- Whether the 70x100 cm draft has enough whitespace after actual compilation.

## 8. Final audit checklist

- [x] Every scientific number in the audited draft maps to the approved anchor pool.
- [x] AUPRC and operating-point evidence are separated.
- [x] MCC/specificity are not framed as replacement primary metrics.
- [x] No AUPRC superiority over XGBoost F4 is claimed.
- [x] `0.900705` is no-skill PR baseline, not floor.
- [x] `measured=0` is not validation/test ground truth.
- [x] Literature positioning is qualitative only.
- [x] No biological-causality claim appears.
- [x] Seed/guide fragility appears near operating-point evidence.
- [ ] Exact honesty caveat appears only once visually in the integrated draft.

Current recommendation: proceed to Slice 8 only after deciding whether the caveat should live inside `fig03` or as a LaTeX-level results-band callout. Do not keep both in the next integrated draft.
