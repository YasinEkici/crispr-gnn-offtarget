# Poster Section Shortlist

This document is the Slice 1 content inventory and section shortlist for the CRISPR-Cas9 off-target prediction thesis poster. It reduces `poster_content_plan.md` into first-draft decisions: what must stay, what can be compressed, and what should be cut unless space remains.

This is not final poster copy, not a layout file, not a figure-production plan, and not a production-tool decision. It exists to keep the first draft visual-first, contribution-first, and claim-safe.

Planning language is English. Poster-bound Turkish phrases are quoted.

## 1. Purpose and scope

The first poster draft should not try to carry the entire thesis or the full content plan. Its job is to prove that the central poster story fits on a 70x100 cm portrait surface:

- strict measured-only data contract;
- Graph C target-observation representation;
- two-axis result story;
- rare-negative operating-point behavior;
- honest caveats that remain visible.

This shortlist decides the content stack before figure specs and layout work begin.

Out of scope:

- final Turkish copy;
- final title choice;
- figure creation;
- LaTeX / Canva / slides implementation;
- thesis edits;
- raw-score literature comparison.

## 2. Inputs and governing rules

Governing files:

- `docs/poster/notes/poster_design_decisions.md`
- `docs/poster/notes/poster_narrative_framing.md`
- `docs/poster/notes/poster_yazim_kurallari.md`
- `docs/poster/plan/poster_content_plan.md`
- `docs/poster/notes/poster_progress.md`
- `docs/poster/plan/poster_execution_plan.md`

Working rules:

- Use only approved numeric anchors.
- Keep the poster contribution-first.
- Keep AUPRC and operating-point results visually separated.
- Treat MCC and specificity as operating-point evidence, not replacement primary metrics.
- Keep claim caveats visible; do not bury them as tiny footnotes.
- Keep the production tool non-binding.
- Keep the later draft Apple-like / keynote-inspired in discipline: low text density, strong hierarchy, one central visual idea, and generous whitespace.

## 3. Shortlist principle

Use this priority order when space becomes tight:

1. **Keep the visual argument.** The viewer must understand the measured-only funnel, Graph C semantics, and the two-axis result story without reading long text.
2. **Keep the claim boundary.** Caveats are not decorative; they are part of the scientific claim.
3. **Compress explanations before cutting evidence.** Long method prose becomes a figure label, chip, callout, or caption.
4. **Cut tertiary detail before core visuals.** Version drift, detailed interval notes, and literature nuance can shrink before the funnel, Graph A/B/C, or rare-negative result panel.
5. **Do not solve layout here.** This file decides content survival, not exact placement.

## 4. First-draft content stack

The first draft should carry this stack in order:

1. Title zone and one-line thesis.
2. Hero representation: Graph C / target-observation idea.
3. Data contract: measured-only funnel and Scheme A.
4. Graph A/B/C semantic comparison.
5. Two-axis results: `"Sıralama (AUPRC)"` and `"Karar Eşiği"`.
6. Rare-negative recognition: true negatives out of 169.
7. Why the axes diverge: positives saturate ranking; scarce negatives drive operating-point behavior and fragility.
8. Literature A+B qualitative positioning.
9. Contribution / takeaway.
10. Limits / future work.
11. References and author information.

If this stack feels crowded, compress the mechanism chain and literature panel first; do not remove the measured-only funnel, Graph A/B/C, two-axis result split, or honesty caveat.

## 5. Keep / compress / cut table

| Content block | First-draft decision | Reason |
| --- | --- | --- |
| Title / hook | Keep as major block | Visitor entry point and first contribution signal. |
| One-line thesis | Keep but compact | Gives the whole poster a claim-safe frame. |
| CRISPR for-dummies explanation | Compress | Needed for visitors, but should be a small visual or one sentence. |
| Problem / aim | Keep but compact | Establishes why the evaluation question matters. |
| Scheme A label rule | Keep as callout | Label discipline is a trust signal. |
| Measured-only funnel | Keep as major visual | Core evidence discipline; must survive compression. |
| Guide-disjoint / no-test-tuning note | Keep but compact | Jury-layer credibility; can live in caption or small method strip. |
| Graph A/B/C comparison | Keep as major visual | Core representation contribution. |
| Graph C metaphor | Keep if visually paired with technical term | Helps visitor layer without changing the claim. |
| Model / mechanism chain | Compress | Useful connective tissue, but not the main poster burden. |
| Feature-family counts | Compress or defer | Use only if they support the mechanism chain or feature-family strip. |
| Ranking axis / AUPRC | Keep as result panel | AUPRC remains the primary ranking metric. |
| Operating-point axis / MCC-specificity | Keep as result panel | Central rare-negative behavior; must not be secondary afterthought. |
| TN/169 rare-negative recognition | Keep as major visual | Most direct way to show why operating-point metrics matter here. |
| Axis-divergence explanation | Keep but compact | Prevents AUPRC and MCC stories from looking contradictory. |
| Honesty caveat | Keep exactly once in results | Required claim boundary; never headline. |
| Literature A+B | Keep but compact | Required positioning; qualitative only. |
| Raw-score comparison to papers | Cut | Violates claim boundary. |
| Full bootstrap / multi-seed details | Compress or defer | Important, but too dense for first draft unless shown as compact uncertainty. |
| Historical XGBoost F4 version-drift note | Cut unless space remains | Tertiary detail; avoid distracting from main result. |
| Contribution bullets | Keep but compact | Needed for strong close. |
| Limits / future work | Keep but compact | Scientific boundary must be visible. |
| References | Compress | Academic requirement; not a visual anchor. |
| Author / advisor / BTU info | Keep as footer | Required poster identity. |

## 6. Section decisions

### Problem / aim

Decision: keep but compact.

Role in first draft:

- Give non-specialists the CRISPR off-target entry point.
- Frame the question as context representation under a strict evaluation contract.

Keep:

- one short problem sentence;
- one short research-question sentence;
- optional small CRISPR for-dummies sketch.

Compress:

- background biology;
- long literature motivation;
- generic "importance" framing.

Cut from first draft:

- extended CRISPR mechanism explanation;
- broad clinical claims.

Poster-bound candidate direction:

- `"Soru: bağlamı çizge içinde doğru temsil etmek, model davranışını nerede değiştirir?"`

### Data contract / label discipline

Decision: keep as major evidence block.

Role in first draft:

- Show why the evaluation is trustworthy before showing results.
- Make measured-only discipline visible.

Keep:

- `310142 -> 25632 -> 1702`;
- Scheme A: `cleavage_freq > 1e-5`;
- test composition: 29 guides, 1533 positives, 169 negatives;
- `measured=0` rows are not validation/test ground truth.

Compress:

- exact procedural details of preprocessing;
- long split explanation.

Cut from first draft:

- nonessential dataset audit detail.

Poster-bound candidate direction:

- `"Doğrulanmamış adaylar (measured=0) test etiketi yapılmadı."`

### Graph representation: Graph A/B/C

Decision: keep as major visual.

Role in first draft:

- Carry the representation contribution.
- Make Graph C understandable as target-observation semantics, not just "more graph."

Keep:

- three-panel Graph A/B/C comparison;
- Graph C target-observation label;
- `"aynı adres, farklı ziyaretler"` metaphor only if paired with technical wording.

Compress:

- graph construction details;
- edge-count or implementation details unless needed by the visual.

Cut from first draft:

- detailed graph schema tables.

Poster-bound candidate direction:

- `"Graph C katkısı salt topoloji değil, düğüm semantiği değişimidir."`

### Model / mechanism chain

Decision: compress.

Role in first draft:

- Connect Graph C and context features to GATv2 / family-aware encoder without becoming an architecture poster.

Keep:

- one compact chain: data contract -> Graph C -> model -> two-axis results;
- no-causality caveat if attention, masking, FiLM, or feature-family visuals appear.

Compress:

- architecture internals;
- feature-family detail;
- mechanism-isolation wording.

Cut from first draft:

- detailed encoder diagrams unless space remains after result panels.

Poster-bound candidate direction:

- `"Model içi açıklamalar biyolojik nedensellik kanıtı değildir."`

### Results: ranking axis

Decision: keep as result panel.

Role in first draft:

- State AUPRC honestly and non-defensively.
- Show that XGBoost F4 remains the ranking bar.

Keep:

- XGBoost F4 AUPRC 0.992338 [0.950179, 0.999336];
- S8B_R2 AUPRC 0.986020 [0.929981, 0.998966];
- `0.900705` as no-skill PR baseline;
- compatibility language, not equivalence language.

Compress:

- multi-seed values if the panel becomes too dense;
- full PR-curve detail.

Cut from first draft:

- historical XGBoost F4 0.992522 unless needed as a tiny version-drift note.

Poster-bound candidate direction:

- `"XGBoost F4, AUPRC bakımından en sağlam referans olarak kaldı."`

### Results: operating-point axis

Decision: keep as result panel with strong visual weight.

Role in first draft:

- Show why MCC, specificity, and true negatives matter in this imbalanced test universe.
- Make the rare-negative behavior visible without claiming AUPRC superiority.

Keep:

- true negatives over 169: XGBoost F4 = 40, Graph C GCN = 14, Graph C GATv2 = 63, family-aware encoder = 110;
- specificities 0.236, 0.083, 0.373, 0.651 if space allows;
- family-aware encoder MCC 0.603489 if MCC is highlighted;
- validation-locked and seed/guide-fragile caveat.

Compress:

- detailed threshold explanation;
- secondary metrics beyond MCC/specificity.

Cut from first draft:

- any framing that says MCC is globally "better" than AUPRC.

Poster-bound candidate direction:

- `"MCC/specificity bu nadir negatif davranışını görünür kılar."`

### Why the axes diverge

Decision: keep but compact.

Role in first draft:

- Explain why AUPRC looks close while operating-point metrics differ.
- Tie the finding and fragility to the same scarcity structure.

Keep:

- prevalence 0.900705;
- 1533 positives vs 169 negatives;
- negatives in 9 guides, 80 in guide 9251;
- one sentence explaining saturation vs negative sensitivity.

Compress:

- mathematical explanation of PR curves or MCC;
- full robustness discussion.

Cut from first draft:

- any claim that operating-point gains are robust across all guides or seeds.

Poster-bound candidate direction:

- `"Sıralama bol sayıda pozitifle doyuma ulaşır; karar eşiği az sayıdaki negatife bakar."`

### Literature positioning A+B

Decision: keep but compact.

Role in first draft:

- Satisfy the literature comparison need without misleading score comparisons.

Keep:

- qualitative A+B structure;
- question `"Neden doğrudan kıyaslanamaz?"`;
- axes: leakage control, guide-disjoint evaluation, prevalence awareness, measured-only universe.

Compress:

- named-paper detail;
- long prose about each literature axis.

Cut from first draft:

- raw AUPRC or accuracy comparison against other papers;
- "state of the art" language.

Poster-bound candidate direction:

- `"Kıyas, skor yarışı değil; soru ve sözleşme karşılaştırmasıdır."`

### Contribution / takeaway

Decision: keep but compact.

Role in first draft:

- Close the poster with what the thesis contributes.

Keep:

- measured-only / leakage-aware evaluation;
- Graph C target-observation representation;
- rare-negative operating-point behavior;
- honest uncertainty boundary.

Compress:

- contribution list to 3-4 short bullets or chips.

Cut from first draft:

- broad claims about best model or clinical utility.

Poster-bound candidate direction:

- `"Bağlam katkısını AUPRC zaferi olarak değil, karar eşiği davranışı olarak konumlandırdık."`

### Limits / future work

Decision: keep but compact.

Role in first draft:

- Keep the scientific boundary visible without turning the poster into an apology.

Keep:

- no AUPRC superiority claim;
- seed/guide fragility;
- more measured negatives and external validation as future direction;
- no biological-causality claim from model internals.

Compress:

- future work to 2-3 bullets.

Cut from first draft:

- detailed experimental roadmap.

Poster-bound candidate direction:

- `"Nadir negatif kazanım seed/guide duyarlıdır."`

### References and author info

Decision: keep as compact footer.

Role in first draft:

- Satisfy academic poster requirements and identify the thesis team.

Keep:

- thesis title;
- authors;
- advisor;
- department / university;
- essential references.

Compress:

- references to the minimum that supports the poster.

Cut from first draft:

- long bibliography.

## 7. Must-survive visuals

These visuals should survive the first draft unless a later approved design decision replaces them with an equivalent claim-safe form:

1. **Measured-only funnel:** `310142 -> 25632 -> 1702`.
2. **Graph A/B/C semantic comparison:** Graph C as target-observation representation.
3. **Two-axis result panel:** AUPRC ranking vs operating-point MCC/specificity.
4. **Rare-negative recognition visual:** true negatives out of 169.
5. **Literature A+B panel:** qualitative positioning, no raw-score leaderboard.

Should-have if space allows:

- CRISPR for-dummies mini sketch;
- compact model / mechanism chain;
- feature-family chip strip.

Cut first if crowded:

- detailed bootstrap inset;
- historical XGBoost F4 version-drift note;
- detailed feature-family count block if already implied by the mechanism chain.

## 8. Text compression rules for first draft

Use these compression rules before cutting core content:

- Convert method paragraphs into callouts or figure labels.
- Convert contribution prose into 3-4 chips.
- Convert caveats into visible one-line captions, not tiny footnotes.
- Keep numeric explanations close to the figure they explain.
- Use one Turkish sentence for the visitor layer, then a compact jury caption for detail.
- Avoid repeating the same caveat in multiple long blocks; place it once where it controls interpretation.

Maximum first-draft text density:

- title zone: maximum 2 title lines and 2 subtitle lines;
- body blocks: usually 1-2 sentences;
- figure captions: 12-25 words;
- contribution and limits: short bullets or chips;
- literature panel: qualitative labels, not a paragraph.

## 9. Deferred / cut-if-needed content

Defer unless a later layout shows clear space:

- historical XGBoost F4 0.992522 version-drift note;
- full multi-seed values for both ranking models;
- detailed bootstrap interval explanation;
- exact feature-family count block if not visually useful;
- named-paper details in the literature panel;
- detailed thesis section references;
- extended CRISPR background;
- detailed encoder internals.

Do not defer:

- honesty caveat inside results;
- measured-only caveat for `measured=0`;
- no-skill PR baseline label for `0.900705`;
- seed/guide fragility for operating-point gains;
- no-causality caveat if model-internal explanations are shown.

## 10. Claim-boundary placement in the shortlist

The claim boundary must remain attached to the content it controls.

Required placements:

- **Results band:** the exact honesty caveat appears once, inside results:

> `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

- **Operating-point panel:** validation-locked and seed/guide-fragile caveat.
- **Ranking / PR panel:** `0.900705` labeled as no-skill PR baseline, not performance floor.
- **Data contract panel:** `measured=0` rows are not validation/test ground truth.
- **Model / mechanism visuals:** model-internal outputs are not biological-causality evidence.
- **Literature panel:** qualitative comparison only, no raw-score leaderboard.

These caveats are keep items, not cut-if-needed items.

## 11. Acceptance checklist for moving to Slice 2

Before starting the figure-production plan:

- [ ] Is the first-draft content stack clear?
- [ ] Are keep / compress / cut decisions stated for every planned section?
- [ ] Are the measured-only funnel and Graph A/B/C comparison protected?
- [ ] Are AUPRC and operating-point results both protected and separated?
- [ ] Is MCC/specificity framed only as operating-point evidence?
- [ ] Is the TN/169 rare-negative visual protected?
- [ ] Is the honesty caveat protected inside results?
- [ ] Are claim caveats visible and attached to the right panels?
- [ ] Is literature positioning qualitative only?
- [ ] Are deferred items truly tertiary?
- [ ] Are no new numbers introduced?
- [ ] Is the document short enough to drive the next slice without duplicating the full content plan?
