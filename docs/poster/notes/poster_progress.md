# Poster Progress

This document is a lightweight progress tracker for the thesis-poster planning effort. It does not replace the poster constitution, narrative framing, writing rules, or content plan. Update it only when it helps the team resume work cleanly, not after every small wording edit.

## 1. Purpose

Use this file to see:

- which poster-planning files exist,
- which narrative and claim decisions are locked,
- which numeric anchors are approved for poster use,
- which decisions remain open,
- what the next recommended planning step is.

This is not poster copy, a layout plan, or a figure-production file.

## 2. Current status snapshot

- Poster planning documents now exist under `docs/poster/`.
- A BTU school-template LaTeX working draft now exists at `docs/poster/drafts/btu_school_template/poster.tex`.
- The production tool is still not finally fixed; the school-template LaTeX draft is the current working direction, not a print-approved export.
- The thesis is finished and must not be edited for poster planning.
- Planning documents are written in English.
- The poster itself will be written in Turkish.
- Poster-bound Turkish phrases stay quoted inside planning documents.
- Turkish poster copy source of truth: `docs/poster/plan/poster_copy_deck.md`.

## 3. Planning files written

- `docs/poster/notes/poster_design_decisions.md`
  The poster constitution. It fixes scope, audience, size, language, contribution-first reframe, claim boundary, literature-positioning decision, visual strategy, and non-binding production format.

- `docs/poster/notes/poster_narrative_framing.md`
  The narrative spine. It defines the one-sentence thesis, title candidates, gap-to-contribution arc, two-axis result story, axis-divergence explanation, contributions, literature A+B positioning, numeric anchors, and forbidden phrasings.

- `docs/poster/notes/poster_yazim_kurallari.md`
  The writing and visual-language rulebook. It fixes register, banned AI-filler, terminology, claim-safe wording, text-density budgets, number formatting, figure-caption rules, accessibility guidance, and a block-level checklist.

- `docs/poster/plan/poster_content_plan.md`
  The section-by-section content and figure plan. It maps the narrative into poster sections, figure priorities, numeric-anchor placement, claim-boundary placement, literature panel structure, text budgets, open decisions, and pre-production checklist.

- `docs/poster/plan/poster_copy_deck.md`
  The canonical Turkish copy reservoir. Future draft edits should pull from its full/short variants instead of rewriting claims inside the LaTeX file.

- `docs/poster/plan/poster_execution_plan.md`
  The slice-by-slice execution tracker. It turns the planning baseline into reviewable next steps: content shortlist, figure production plan, layout brief, Turkish microcopy, figure adaptation, first draft, audit, print pass, and final handoff.

- `docs/poster/drafts/btu_school_template/poster.tex`
  The first transfer of the poster content into the BTU school-provided LaTeX poster template. It is now the active school-template draft path. It preserves the BTU header/footer and main poster sections while leaving box sizes and figure placement adjustable.

## 4. Locked narrative decisions

- The poster is contribution-first.
- The headline is not `"XGBoost'u geçemedik"`.
- Graph C target-observation representation is a central contribution.
- Scheme A and measured-only discipline are central trust signals.
- Results are told on two axes:
  - ranking axis: AUPRC;
  - operating-point axis: MCC, specificity, and true-negative recognition.
- XGBoost F4 remaining the AUPRC bar is a finding inside results, not the poster's top message.
- The single honesty caveat lives inside the results area only:

> `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

- Literature comparison uses qualitative A+B positioning:
  - A: contextual positioning, `"neden doğrudan kıyaslanamaz?"`;
  - B: spiritual comparison, ranking/retrieval question vs measured-only binary question.
- No raw-score leaderboard against other papers is allowed.

## 5. Locked claim boundary

Non-negotiable limits:

- No robust AUPRC superiority over XGBoost F4 is claimed.
- Attention, gate, FiLM, embedding, masking, and feature-ablation outputs are not biological-causality evidence.
- Threshold and rare-negative gains are seed/guide-fragile.
- `0.900705` is the no-skill PR baseline, not a performance floor.
- No general claim that sequence models fail.
- No raw-score performance comparison against other papers.
- Where paired intervals include zero, use compatibility language, not equivalence language.
- `measured=0` rows are not validation/test ground truth.

## 6. Locked numeric anchor pool

Approved numeric anchors for poster use:

- Universe / split: 310142 rows -> measured-only 25632 -> test 1702.
- Test: 29 guides, 1533 positives, 169 negatives.
- Negatives: concentrated in 9 guides; 80 of 169 in guide 9251.
- Prevalence: 0.900705, as the no-skill PR baseline.
- XGBoost F4 regenerated AUPRC: 0.992338 [0.950179, 0.999336].
- XGBoost F4 multi-seed: 0.990649 ± 0.001944.
- XGBoost F4 historical AUPRC: 0.992522, only as a version-drift note.
- S8B_R2 AUPRC: 0.986020 [0.929981, 0.998966].
- S8B_R2 multi-seed: 0.978963 ± 0.011322.
- True negatives over 169:
  - XGBoost F4 = 40;
  - Graph C GCN = 14;
  - Graph C GATv2 = 63;
  - family-aware encoder = 110.
- Specificities: 0.236, 0.083, 0.373, 0.651.
- Family-aware encoder MCC: 0.603489.
- Feature families: 6 experimental-epigenetic + 13 computed-nucleosome + 5 binding-energy.
- Dataset scale anchors: 154 sgRNA, 138747 target locations.

Do not add numbers here unless they are verified against the thesis and intentionally approved for poster use.

## 7. Open decisions

Still unresolved:

- final title,
- exact hero visual,
- palette and accent colors,
- final production tool / final confirmation that the BTU LaTeX template will be used,
- final figure styling,
- whether thesis figures are recolored,
- exact placement of the literature A+B panel,
- whether the axis-divergence explanation is its own panel or nested inside results,
- exact reference count.

## 8. Recommended next step

Recommended next step: review the BTU school-template LaTeX draft visually, then tune box sizes, figure sizes, and text density.

Pragmatic sequence:

1. Fill missing student number and email placeholders if available.
2. Decide whether to keep the school-template LaTeX draft as the production direction.
3. Tune section box heights and figure placements while preserving the BTU header/footer.
4. Re-run compile and visual proof checks after every layout pass.
5. Run claim/number and print/accessibility checks before final export.

## 9. Update rules

Update this file when:

- a new poster planning file is created,
- a locked narrative decision changes,
- a locked numeric anchor changes,
- the production tool is chosen,
- the first poster draft exists.

Do not update this file:

- after every small wording tweak,
- for unapproved experiments or speculative ideas,
- for thesis edits,
- to duplicate the full content plan.

## 10. Quick handoff checklist

Anyone resuming the poster work should:

- [ ] read `poster_design_decisions.md`,
- [ ] read `poster_narrative_framing.md`,
- [ ] read `poster_yazim_kurallari.md`,
- [ ] read `poster_content_plan.md`,
- [ ] preserve the contribution-first reframe,
- [ ] preserve the claim boundary,
- [ ] use only approved numeric anchors,
- [ ] keep poster-bound Turkish phrases quoted in planning docs,
- [ ] avoid editing the thesis,
- [ ] ask before changing locked narrative decisions or numeric anchors.
