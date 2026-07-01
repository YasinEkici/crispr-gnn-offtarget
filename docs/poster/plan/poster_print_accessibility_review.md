# Poster Print and Accessibility Review

This document is the Slice 8 print and accessibility pass for the current thesis-poster working draft. It reviews the LaTeX fit-check source and SVG assets for print legibility, text density, contrast, grayscale/colorblind resilience, Turkish rendering risk, and caveat visibility.

This is not a final design approval. The LaTeX fit-check has not been compiled in the current environment, so this review is source-based rather than a measured PDF/print inspection.

Status values:

- `pass` - acceptable for the current working draft.
- `watch` - acceptable for fit-checking, but must be checked after compilation/export.
- `fix-needed` - should be changed before final export or a higher-fidelity draft.

## 1. Scope

Reviewed files:

- `docs/poster/drafts/latex_fit_check/poster_fit_check.tex`
- `docs/poster/drafts/latex_fit_check/README.md`
- `docs/poster/assets/fig01_measured_only_funnel.svg`
- `docs/poster/assets/fig02_graph_abc_semantic_comparison.svg`
- `docs/poster/assets/fig03_two_axis_results.svg`
- `docs/poster/assets/fig04_tn169_rare_negative_recognition.svg`
- `docs/poster/assets/fig05_literature_ab_positioning.svg`
- `docs/poster/plan/poster_claim_number_audit.md`

Out of scope:

- editing the LaTeX draft;
- editing SVG assets;
- producing a final export;
- choosing the final production tool;
- testing an actual printed proof;
- thesis edits.

## 2. Overall result

Overall status: `watch`

The current draft follows the intended Apple-like / keynote-inspired discipline: clean off-white surface, restrained palette, large visual anchors, low body-text density, and visible section hierarchy. It is suitable as a planning fit-check.

Before a higher-fidelity poster or final export, three items need attention:

1. The exact honesty caveat is duplicated between `fig03_two_axis_results.svg` and `poster_fit_check.tex`; keep one visible occurrence inside the results band.
2. The SVG assets must be checked after actual import/compilation because text inside SVGs may shrink or render differently.
3. Turkish characters must be verified in the final production environment, especially if the draft is compiled through LaTeX/SVG conversion.

## 3. Print legibility

| Item | Status | Review note |
| --- | --- | --- |
| 70x100 cm portrait canvas | `pass` | `poster_fit_check.tex` uses `paperwidth=70cm,paperheight=100cm`. |
| Main title hierarchy | `watch` | The title is large in source, but it is long and may become a three-line block after real compilation. Final title remains deferred. |
| Section headings | `pass` | Source uses large section commands and short headings. |
| Body text density | `pass` | Body text is mostly short and visual-led; no thesis paragraphs dominate the draft. |
| Figure labels | `watch` | SVG labels are large in standalone SVGs, but final readability depends on scale inside LaTeX or another tool. |
| Caveat readability | `watch` | Caveats are visible in source, but some are embedded inside scaled SVGs. Verify after export. |
| Footer readability | `watch` | Footer text is compact and may be too small if printed or exported with scaling. |

## 4. Text density

Status: `pass`

The source draft keeps a low-density structure:

- title and one-line thesis at top;
- Graph C and measured-only visuals as the first major layer;
- two-axis results as the central layer;
- interpretation, literature, contribution, and limits as compact lower blocks;
- no dense traditional academic-poster paragraph grid.

Watch items:

- `fig03_two_axis_results.svg` and `fig04_tn169_rare_negative_recognition.svg` may feel visually dense when placed side by side.
- The lower interpretation/literature/contribution row should be checked after compilation for crowding.
- The footer should stay informational and not become unreadable microtext.

## 5. Contrast and grayscale resilience

Status: `watch`

Passes in source:

- Text is primarily dark slate on off-white or white backgrounds.
- Caveats use a pale orange background with dark text.
- Color is generally paired with labels, not used alone.
- Metric values are written as text, not only encoded by color.

Watch items:

- Blue/teal/orange/purple bars in `fig04` should be checked in grayscale; the numeric labels preserve meaning, but color contrast may compress.
- Light gray lines and muted labels may be too faint after printing.
- Pale backgrounds such as `SoftBlue`, `SoftTeal`, and `SoftOrange` should be checked on the actual printer/export profile.

Recommendation for final design:

- Keep every metric label and caveat readable without relying on color.
- If grayscale proof weakens the bars, add stronger shape/position encoding or darken bar outlines.

## 6. Colorblind safety

Status: `watch`

The current SVGs do not rely on color alone for scientific meaning: bars and panels have text labels and numeric callouts. This is acceptable for a working draft.

Watch items:

- In `fig04`, the four model bars use different colors. Because each bar has text and a fixed position, the meaning survives colorblind viewing, but the visual distinction should be checked after export.
- In `fig02`, Graph C emphasis uses blue stroke and placement. The label and caption carry the meaning, so the claim is not color-only.

Final check:

- Review a grayscale export and a colorblind simulation before final print.

## 7. Caveat visibility

Status: `fix-needed`

The caveats are visible, but the exact honesty caveat currently appears twice:

- inside `fig03_two_axis_results.svg`;
- as a LaTeX-level `\Caveat{...}` block under the results band in `poster_fit_check.tex`.

Required fix before a higher-fidelity integrated draft:

- Keep the exact honesty caveat once, inside the results band.
- Prefer one of these approaches:
  - keep it in `fig03` and remove the LaTeX-level duplicate;
  - remove it from `fig03` and keep the LaTeX-level callout directly under the result visuals.

Other caveats:

- `0.900705` no-skill baseline caveat is visible in the interpretation area and in `fig03`.
- `measured=0` caveat is visible in `fig01`.
- seed/guide caveat is visible in `fig04` and the limits block.
- no-causality caveat is visible in `fig02` and the limits block.
- no raw-score leaderboard caveat is visible in `fig05`.

These pass conceptually, but their print readability remains `watch` until a compiled/exported proof exists.

## 8. Turkish rendering and font risk

Status: `watch`

The source uses Turkish poster text and `fontspec` with `Arial`. This is reasonable for a local XeLaTeX fit-check, but it depends on local font availability and SVG conversion behavior.

Risks:

- Turkish characters may render incorrectly if the production tool changes encoding.
- The `svg` package may convert text through Inkscape or a fallback path that changes fonts or line breaks.
- Terminal output may show mojibake even when source files are valid UTF-8; judge final rendering from the exported file, not terminal output.

Final check:

- Verify these Turkish characters in the exported proof: `ı`, `İ`, `ğ`, `Ğ`, `ş`, `Ş`, `ö`, `Ö`, `ü`, `Ü`, `ç`, `Ç`.
- Confirm names render correctly: `"Kasım Deliacı"`, `"Doç. Dr. Mustafa Özgür Cingiz"`, `"Bursa Teknik Üniversitesi"`.

## 9. Figure-by-figure print/accessibility notes

| Figure | Status | Print/accessibility note |
| --- | --- | --- |
| `fig01_measured_only_funnel.svg` | `watch` | Large numbers and labels are strong. Check whether the bottom measured-only caveat remains readable after scaling. |
| `fig02_graph_abc_semantic_comparison.svg` | `watch` | Good visual hierarchy. Check Graph A/B/C captions after scaling; they carry important semantic and no-causality boundaries. |
| `fig03_two_axis_results.svg` | `fix-needed` | Strong axis split, but exact honesty caveat duplication must be resolved at the integrated-draft level. Check no-skill baseline line does not read as a real PR curve unless intentionally styled that way. |
| `fig04_tn169_rare_negative_recognition.svg` | `watch` | Strong shared-denominator visual. Check bar labels and seed/guide caveat at final scale. |
| `fig05_literature_ab_positioning.svg` | `watch` | Qualitative panel is claim-safe. Check lower caveat and small axis labels after scaling. |

## 10. Apple-like / keynote-inspired discipline

Status: `pass`

The draft follows the intended design discipline:

- spacious 70x100 portrait composition;
- large hero and result visuals;
- restrained palette;
- low text density;
- no many-box academic grid;
- caveats shown as calm scientific boundaries rather than alarm boxes;
- figures carry most of the explanation.

Watch item:

- If later edits add more text, the design can quickly slide back into a dense academic poster. Cut text before shrinking caveats.

## 11. BTU section visibility

Status: `pass`

The draft preserves BTU poster functions:

- overview / aim: title, one-line thesis, Graph C hero;
- method: measured-only funnel and Graph A/B/C;
- results: two-axis result band and TN/169 visual;
- discussion / conclusion: axis divergence, contribution chips, limits;
- references / author info: footer and identity block.

Watch item:

- References are intentionally placeholder-level. Final reference count and formatting remain deferred.

## 12. Final export risks to carry forward

Carry these risks into Slice 9:

- No compiled PDF was available in this environment.
- No physical print proof was inspected.
- SVG-to-LaTeX rendering may alter fonts, line breaks, and caveat size.
- The exact honesty caveat is duplicated and must be reduced to one visual occurrence.
- Candidate title is used for fit-checking but remains deferred.
- Grayscale and colorblind checks still require an actual export.
- Footer/reference readability is not verified.

## 13. Pre-export checklist

Before final export or handoff:

- [ ] Compile or export one full 70x100 cm proof.
- [ ] Verify Turkish character rendering.
- [ ] Ensure the exact honesty caveat appears once, inside results.
- [ ] Confirm `0.900705` reads as no-skill PR baseline, not floor.
- [ ] Confirm TN/169 bars remain readable and do not imply a global model win.
- [ ] Confirm seed/guide caveat is readable near the operating-point visual.
- [ ] Confirm no-causality caveat is readable near Graph/model visuals.
- [ ] Confirm literature panel stays qualitative and no raw-score leaderboard appears.
- [ ] Check grayscale distinguishability.
- [ ] Check colorblind distinguishability.
- [ ] Check title, section heads, figure labels, metric labels, captions, and caveats at print scale.
- [ ] Cut text before shrinking caveats.
- [ ] Keep production-tool decision documented separately from this fit-check.

## 14. Recommended next action

Before Slice 9 final export handoff, fix the duplicated honesty caveat in the integrated draft and produce a real exported proof from whichever tool is used next. If LaTeX remains only a fit-check, the same review checklist should be applied to the Canva or slides version.
