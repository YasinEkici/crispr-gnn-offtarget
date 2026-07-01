# Poster Execution Plan

This document is the active execution plan for moving from poster planning notes to a first thesis-poster draft. It is slice-based and reviewable: each slice should produce one small artifact or one bounded decision before the next slice starts.

This is not the poster itself, not final poster copy, not a figure file, and not a production implementation. It does not choose the final production tool and does not replace `docs/poster/plan/poster_content_plan.md`. Any later LaTeX draft in this track is only a temporary low-fidelity fit-check / working draft, not a final production decision.

Planning documents are written in English. Poster-bound Turkish phrases stay quoted.

## 1. Purpose and scope

This plan tracks the remaining poster work from stable planning documents toward a first integrated draft. Its job is to keep the work small, auditable, and claim-safe.

Scope:

- track which poster-production slices are pending, in progress, complete, or blocked;
- state each slice's goal, inputs, likely output, risks, and acceptance criteria;
- preserve the non-binding production-tool decision;
- keep the later draft direction aligned with the poster constitution and content plan.

Out of scope:

- final poster production;
- final Turkish copy;
- figure creation;
- LaTeX / Canva / slides implementation;
- thesis edits.

## 2. Inputs and dependencies

Required upstream planning files:

- `docs/poster/notes/poster_design_decisions.md`
- `docs/poster/notes/poster_narrative_framing.md`
- `docs/poster/notes/poster_yazim_kurallari.md`
- `docs/poster/plan/poster_content_plan.md`
- `docs/poster/notes/poster_progress.md`

Thesis and project sources used only for verification:

- thesis chapters under `docs/thesis/latex/btu_template/chapters/`
- `docs/PROJECT_CONTEXT.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/DECISIONS.md`

Do not edit the thesis while executing this plan.

## 3. Global constraints

- Planning documents stay in English; the poster stays in Turkish.
- No thesis edits.
- No invented numbers.
- No raw-score literature leaderboard.
- No AUPRC superiority claim over XGBoost F4.
- MCC and specificity are operating-point evidence, not replacement primary metrics.
- Figures must not imply biological causality from attention, gate, FiLM, masking, embedding, feature-family effects, or graph arrows.
- The production tool remains non-binding until a later slice.
- A later low-fidelity draft should follow an Apple-like / keynote-inspired visual discipline: clean, spacious, visual-first, premium-feeling, calm, and low-density.
- The aesthetic is subordinate to scientific accuracy: it cannot hide caveats, distort metrics, or make unsupported claims.
- Avoid a crowded academic-poster grid; use hierarchy, whitespace, and strong visual anchors.

## 4. Slice status table

Status values: `pending`, `in progress`, `complete`, `blocked`.

| Slice | Name | Goal | Primary output | Status |
| --- | --- | --- | --- | --- |
| 0 | Planning baseline freeze | Confirm existing planning docs and locked decisions | Stable planning baseline | complete |
| 1 | Content inventory and section shortlist | Decide which blocks survive the first draft | Section shortlist / draft brief | pending |
| 2 | Figure production plan | Specify must-have visuals before layout | `docs/poster/plan/figure_production_plan.md` | pending |
| 3 | Low-fidelity layout draft brief | Define layout zones without choosing final tool | `docs/poster/plan/layout_draft_brief.md` | pending |
| 4 | Turkish microcopy draft | Draft short poster text blocks | `docs/poster/plan/poster_microcopy_draft.md` | pending |
| 5 | Figure production / adaptation | Create or adapt approved figure assets | `docs/poster/assets/` or approved path | pending |
| 6 | Integrated first poster draft | Combine layout, microcopy, and figures | Tool TBD | pending |
| 7 | Claim-boundary and number audit | Audit the integrated draft | Audit notes or checklist pass | pending |
| 8 | Print and accessibility pass | Check legibility, contrast, and density | Print/accessibility review notes | pending |
| 9 | Final export handoff | Prepare print/export materials | Final source, export, assets, references | pending |

## 5. Slice 0 - Planning baseline freeze

Goal: confirm existing planning docs and locked decisions before moving into figure, layout, or draft work.

Inputs:

- current poster notes;
- `poster_content_plan.md`;
- `poster_progress.md`.

Allowed file changes:

- `docs/poster/notes/poster_progress.md` only if the baseline changes or an important status update is missing.

Output:

- stable planning baseline.

Acceptance criteria:

- all existing planning docs are listed;
- locked narrative decisions are visible;
- locked numeric anchors are visible;
- production tool remains non-binding;
- thesis remains untouched.

Current status: complete, because the constitution, narrative framing, writing rules, content plan, and progress tracker already exist.

## 6. Slice 1 - Content inventory and section shortlist

Goal: decide which content blocks survive the first poster draft.

Inputs:

- `docs/poster/plan/poster_content_plan.md`.

Allowed file changes:

- a future content-shortlist document, or a focused update to this execution plan;
- do not write that artifact as part of this slice unless it is explicitly requested.

Output:

- section shortlist / draft brief.

Risks:

- too much text;
- overexplaining methods;
- losing the visitor layer;
- cutting the measured-only contract or Graph C contribution too aggressively.

Acceptance criteria:

- primary, secondary, and tertiary content are separated;
- each poster section has a keep / compress / cut decision;
- the title zone, hero representation, measured-only funnel, Graph A/B/C, and two-axis result story remain visible;
- the poster still reads contribution-first.

## 7. Slice 2 - Figure production plan

Goal: specify must-have visuals before drafting layout.

Inputs:

- `docs/poster/plan/poster_content_plan.md` figure plan;
- `docs/poster/notes/poster_yazim_kurallari.md` caption and visual-language rules.

Primary output suggestion:

- `docs/poster/plan/figure_production_plan.md`.

Must cover:

- measured-only funnel;
- Graph A/B/C semantic comparison;
- two-axis result panel;
- TN/169 rare-negative recognition visual;
- literature A+B panel.

Acceptance criteria:

- each figure has a purpose;
- each figure has source data or thesis source;
- each figure has a likely visual form;
- each figure has a caption rule;
- each figure has a claim risk;
- no figure implies biological causality or AUPRC superiority.

## 8. Slice 3 - Low-fidelity layout draft brief

Goal: define layout zones without choosing the final production tool.

Output suggestion:

- `docs/poster/plan/layout_draft_brief.md`.

Design direction:

- Apple-like / keynote-inspired visual discipline;
- minimal but premium;
- visual-first;
- generous negative space;
- strong typographic hierarchy;
- calm confidence;
- restrained palette with vivid accents;
- one dominant hero / central representation;
- low text density;
- no crowded academic-poster box grid.

Tool note:

- LaTeX may be used later as a fit-check working draft, not as the final production decision.
- The friend/team may still choose Canva, slides, LaTeX, or another final tool.

Acceptance criteria:

- reading path is visible;
- BTU section functions are preserved;
- text density is controlled;
- contribution-first message is visible at first glance;
- claim caveats are not hidden by the aesthetic;
- layout can be implemented in more than one production tool.

## 9. Slice 4 - Turkish microcopy draft

Goal: draft short Turkish poster text blocks from the content plan.

Output suggestion:

- `docs/poster/plan/poster_microcopy_draft.md`.

Rules:

- obey `poster_yazim_kurallari.md`;
- keep candidate poster text short;
- do not turn the file into final poster copy;
- keep the single honesty caveat inside the results area only:

> `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

Acceptance criteria:

- no AI-filler;
- all claims are safe;
- Turkish text is short and visual-first;
- operating-point gains are tagged as validation-locked and seed/guide-fragile;
- no biological-causality wording appears.

## 10. Slice 5 - Figure production / adaptation

Goal: create or adapt figure assets after figure specs are approved.

Output location suggestion:

- `docs/poster/assets/` or another approved path.

Do not execute this slice in the current task.

Acceptance criteria:

- each figure has claim-safe labels and caption text;
- every number in a figure is traceable;
- visual encodings remain readable in print;
- color does not carry meaning alone;
- source data and thesis references are recorded.

## 11. Slice 6 - Integrated first poster draft

Goal: combine layout, microcopy, and figures into a first full draft.

Output tool:

- TBD later.

Possible tools:

- Canva;
- LaTeX/tikzposter;
- slides;
- another team choice.

Tool note:

- if LaTeX is used here, it is a low-fidelity working draft / fit-check only, not the final production decision.

Acceptance criteria:

- draft fits 70x100 cm portrait;
- visitor and jury reading layers both work;
- Apple-like / keynote-inspired discipline is visible;
- contribution-first framing is clear;
- BTU section functions remain identifiable;
- claim boundaries and caveats remain visible.

## 12. Slice 7 - Claim-boundary and number audit

Goal: audit the integrated draft before any design polish.

Checks:

- every number is traceable;
- no AUPRC superiority is claimed;
- no biological causality is inferred from model internals;
- `0.900705` is labeled as the no-skill PR baseline;
- MCC and specificity are framed as operating-point evidence;
- true-negative counts use the shared denominator of 169;
- operating-point gains are seed/guide-fragile;
- literature panel is qualitative only;
- `measured=0` rows are not treated as validation/test ground truth.

Acceptance criteria:

- all checklist items pass;
- any failed item is fixed before visual polish;
- caveats are visible, not hidden in tiny footnotes.

## 13. Slice 8 - Print and accessibility pass

Goal: check legibility, contrast, colorblind/grayscale distinguishability, and text density.

Acceptance criteria:

- main title is readable from several meters;
- section heads are readable from normal walking distance;
- captions and metric labels are readable from close inspection;
- numbers and caveats have sufficient contrast;
- the design remains understandable in grayscale;
- no text block looks like a dense thesis paragraph.

## 14. Slice 9 - Final export handoff

Goal: prepare final print/export handoff after review.

Acceptance criteria:

- final source file is collected;
- exported PDF/image is collected;
- source assets are collected;
- citation/reference information is collected;
- final production tool is documented;
- no unapproved thesis, data, or claim changes are introduced.

Final production tool may be Canva, LaTeX, slides, or another team choice.

## 15. Current recommendation

Recommended next action after this execution plan:

1. Write `docs/poster/plan/figure_production_plan.md` before any full poster draft.
2. Then write `docs/poster/plan/layout_draft_brief.md` with the Apple-like / keynote-inspired visual direction.
3. Then optionally create a LaTeX low-fidelity working draft only for fit-checking.

Reason:

- figures are the main content bottleneck;
- layout should be designed around visual anchors;
- production tool should remain non-binding until the content fit is clear;
- LaTeX can be useful as a temporary working draft, but it should not decide the final tool.

## 16. Update rules

Update this file when:

- a slice starts;
- a slice completes;
- a slice blocks;
- a slice changes scope;
- the production-tool decision becomes fixed.

Do not update this file:

- for tiny wording edits;
- to duplicate the full content plan;
- for unapproved experiments;
- for thesis edits;
- to store final poster copy.
