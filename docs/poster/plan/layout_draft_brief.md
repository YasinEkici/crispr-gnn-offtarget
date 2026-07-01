# Layout Draft Brief

This document is the Slice 3 low-fidelity layout draft brief for the CRISPR-Cas9 off-target prediction thesis poster. It defines layout zones and reading-path priorities before any poster production begins.

This is not a final design, not final Turkish copy, not a figure file, and not a LaTeX / Canva / slides implementation. It does not choose the final production tool. If a LaTeX draft is created later, it remains a low-fidelity fit-check / working draft, not a final production decision.

Planning language is English. Poster-bound Turkish phrases are quoted.

## 1. Purpose and scope

The layout brief turns the content shortlist and figure-production plan into a spatial strategy for a 70x100 cm portrait poster. Its job is to answer:

- what the viewer sees first;
- where the major visual anchors live;
- how the visitor and jury reading layers are preserved;
- where claim caveats stay visible;
- how the poster avoids a crowded academic grid.

Out of scope:

- final title selection;
- exact palette and typography;
- final figure styling;
- figure production;
- final poster copy;
- final tool choice;
- thesis edits.

## 2. Inputs and constraints

Governing files:

- `docs/poster/notes/poster_design_decisions.md`
- `docs/poster/notes/poster_narrative_framing.md`
- `docs/poster/notes/poster_yazim_kurallari.md`
- `docs/poster/plan/poster_content_plan.md`
- `docs/poster/plan/poster_section_shortlist.md`
- `docs/poster/plan/figure_production_plan.md`
- `docs/poster/plan/poster_execution_plan.md`
- `docs/poster/notes/poster_progress.md`

Constraints:

- 70x100 cm portrait, color print.
- Turkish poster, English planning docs.
- Contribution-first framing.
- No AUPRC superiority claim over XGBoost F4.
- MCC and specificity are operating-point evidence, not replacement primary metrics.
- Caveats must remain visible in the relevant layout zones.
- Production tool remains non-binding.
- Design direction is Apple-like / keynote-inspired, but still academic and claim-safe.

## 3. Design direction

The working draft should feel modern, calm, and visually led:

- minimal but premium;
- strong typographic hierarchy;
- generous negative space;
- low text density;
- restrained palette with vivid accents;
- one dominant hero / central representation;
- figures carrying explanation instead of long paragraphs;
- clean zones rather than many bordered cards;
- visual confidence without unsupported claims.

The aesthetic never overrides scientific accuracy. It cannot hide caveats, soften uncertainty into superiority, or make operating-point behavior look like a ranking win.

Avoid:

- dense academic-poster box grids;
- text walls;
- many equally weighted panels;
- decorative clutter;
- overuse of bordered cards;
- tiny caveat footnotes;
- trophy / winner / leaderboard styling.

## 4. Poster canvas assumptions

Canvas:

- 70x100 cm portrait.
- Designed for color print.
- Must work from two distances:
  - first-glance visitor reading;
  - close-inspection jury reading.

Reading layers:

- **Visitor layer:** title, hero Graph C idea, measured-only trust signal, and two-axis takeaway.
- **Jury layer:** data contract, graph semantics, metric distinction, numeric anchors, uncertainty, and claim boundary.

The poster should be understandable as a guided vertical path, not as a collection of equal boxes.

## 5. Recommended reading path

Recommended path:

1. **Top:** title candidate area, one-line thesis, authors/advisor signal.
2. **Upper hero:** Graph C / target-observation representation as the dominant visual idea.
3. **Upper-middle method:** measured-only funnel and Scheme A data contract.
4. **Center:** Graph A/B/C semantic comparison, if not already fused with the hero.
5. **Central results band:** two-axis results: `"Sıralama (AUPRC)"` vs `"Karar Eşiği"`.
6. **Results support:** TN/169 rare-negative recognition visual.
7. **Lower-middle:** why the axes diverge and why fragility follows.
8. **Lower side or lower band:** literature A+B qualitative positioning.
9. **Bottom:** contribution, limits / future work, references, author / BTU footer.

The reading path should be visually numbered or guided through alignment, scale, and spacing. It should not depend on the viewer reading long paragraphs in order.

## 6. Low-fidelity zone map

### Zone A - Title / identity

Function:

- First-glance title, thesis line, project identity.

Contains:

- final or candidate title placeholder;
- one-line thesis or subtitle;
- authors, advisor, BTU identity in compact form.

Layout guidance:

- Use the largest typographic hierarchy here.
- Keep title to maximum two lines.
- Avoid placing the honesty caveat here.

### Zone B - Hero Graph C representation

Function:

- Show the central contribution visually.

Contains:

- Graph C target-observation hero or near-hero;
- optional Graph C metaphor: `"aynı adres, farklı ziyaretler"`;
- short technical label: `"Graph C: hedef-gözlem (target-observation) şeması"`.

Layout guidance:

- This is the dominant visual anchor.
- It can span the upper center or occupy a large upper-middle block.
- If Graph A/B/C comparison is separate, Graph C still receives the strongest emphasis.

Claim guard:

- Arrows show representation, not biological causality.

### Zone C - Data contract / measured-only method

Function:

- Establish trust before results.

Contains:

- measured-only funnel: `310142 -> 25632 -> 1702`;
- Scheme A callout: `cleavage_freq > 1e-5`;
- test composition: 29 guides, 1533 positives, 169 negatives;
- `measured=0` caveat.

Layout guidance:

- Place early, near the hero or before the result band.
- Make the three funnel numbers large and simple.
- Keep procedural text as caption/callout.

Claim guard:

- Include `"Doğrulanmamış adaylar (measured=0) test etiketi yapılmadı."`

### Zone D - Graph A/B/C comparison

Function:

- Explain the representation progression.

Contains:

- Graph A mini-panel;
- Graph B mini-panel;
- Graph C mini-panel;
- one caption explaining semantic shift.

Layout guidance:

- Three aligned panels or a clean step strip.
- Avoid dense schema detail.
- If the hero already covers Graph C, this zone can be smaller but must still clarify A/B/C semantics.

Claim guard:

- Caption must say Graph C changes representation semantics, not biological mechanism.

### Zone E - Central results band

Function:

- Carry the main empirical finding.

Contains:

- `"Sıralama (AUPRC)"` panel;
- `"Karar Eşiği"` panel;
- exact honesty caveat inside the results area.

Layout guidance:

- Make this a visually unified band across the central width.
- The two axes must be clearly separated.
- AUPRC and operating-point visuals should have distinct labels and visual grammar.
- Do not use leaderboard styling.

Claim guard:

Use this exact sentence inside the band:

> `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

### Zone F - Rare-negative recognition

Function:

- Make the operating-point behavior concrete.

Contains:

- TN/169 visual:
  - XGBoost F4 = 40;
  - Graph C GCN = 14;
  - Graph C GATv2 = 63;
  - family-aware encoder = 110.
- optional specificities 0.236, 0.083, 0.373, 0.651.

Layout guidance:

- Place directly beside or below the operating-point side of the result band.
- Use same denominator visually.
- Make the non-monotone Graph C GCN value visible.

Claim guard:

- Add validation-locked and seed/guide-fragile caveat.

### Zone G - Axis-divergence explanation

Function:

- Explain why AUPRC and MCC/specificity tell different parts of the story.

Contains:

- prevalence 0.900705 as no-skill PR baseline;
- 1533 positives vs 169 negatives;
- negatives in 9 guides, 80 in guide 9251;
- one compact sentence on saturation and fragility.

Layout guidance:

- Use a small explanatory visual or split mini-panel.
- Keep the text compact.
- Place close enough to results that it reads as interpretation, not a separate claim.

Poster-bound direction:

- `"Sıralama bol sayıda pozitifle doyuma ulaşır; karar eşiği az sayıdaki negatife bakar."`

### Zone H - Literature A+B positioning

Function:

- Show literature positioning without score comparison.

Contains:

- Panel A: `"Neden doğrudan kıyaslanamaz?"`;
- qualitative axes: leakage control, guide-disjoint evaluation, prevalence awareness, measured-only universe;
- Panel B: ranking/retrieval question vs measured-only binary question.

Layout guidance:

- Keep lower or side placement.
- Use compact qualitative marks or axis labels.
- Avoid making this look like a performance table.

Claim guard:

- Include `"Kıyas, skor yarışı değil; soru ve sözleşme karşılaştırmasıdır."`

### Zone I - Contribution / limits close

Function:

- Close with what the thesis contributes and what it does not claim.

Contains:

- 3-4 contribution chips;
- 2-3 limits/future-work bullets.

Layout guidance:

- Put near lower end before footer.
- Keep confident but bounded.
- Avoid apology framing.

Claim guard:

- Limits must include no AUPRC superiority, seed/guide fragility, and no biological-causality claim if model internals were shown.

### Zone J - References / author footer

Function:

- Satisfy academic poster requirements and identify the project.

Contains:

- compact references;
- thesis title if not already in title/subtitle;
- authors;
- advisor;
- department / university.

Layout guidance:

- Footer treatment.
- Keep compact and readable.
- Do not let references compete with result visuals.

## 7. Must-have visual placement

| Visual | Preferred zone | Layout role |
| --- | --- | --- |
| Measured-only funnel | Zone C | Early trust signal and method anchor. |
| Graph A/B/C semantic comparison | Zone B or D | Core representation contribution. |
| Two-axis result panel | Zone E | Main empirical finding. |
| TN/169 rare-negative visual | Zone F | Operating-point evidence. |
| Literature A+B panel | Zone H | Claim-safe literature positioning. |

If space is tight:

- Merge Graph C hero and Graph A/B/C comparison.
- Fold axis divergence into the result band as a compact mini-panel.
- Keep literature A+B compact.
- Cut detailed bootstrap and version-drift notes first.

Do not cut:

- measured-only funnel;
- Graph C semantics;
- two-axis split;
- TN/169 denominator visual;
- honesty caveat inside results.

## 8. Text density and hierarchy rules

Text hierarchy:

- Title: largest, maximum two lines.
- Subtitle / thesis line: second level, maximum two lines.
- Section labels: 1-4 words.
- Figure labels: short Turkish terms.
- Captions: 12-25 words.
- Caveats: visible one-liners, not footnote clutter.

Layout rules:

- Prefer figure labels and numeric callouts over paragraphs.
- Use chips for contributions and limits.
- Use whitespace between major zones.
- Do not put cards inside cards.
- Avoid equal-weight boxes; create a clear visual priority.
- A block that needs more than three short lines should become a figure or caption.

## 9. Claim-boundary placement in layout

Required placements:

- **Results band:** exact honesty caveat, once, not in title.
- **Operating-point / TN area:** validation-locked and seed/guide-fragile caveat.
- **AUPRC / PR area:** `0.900705` labeled as no-skill PR baseline, not performance floor.
- **Data contract area:** `measured=0` rows are not validation/test ground truth.
- **Graph / model areas:** arrows and internals are model/representation behavior, not biological causality.
- **Literature area:** qualitative positioning only; no raw-score leaderboard.

Caveats should be visible but visually calm. They should read as scientific boundaries, not alarm boxes.

## 10. BTU section preservation

The layout can be free-form, but BTU poster functions must remain identifiable:

| BTU function | Layout coverage |
| --- | --- |
| Abstract / overview | Zone A title/thesis and Zone B hero. |
| Introduction + aim | Zone A/B plus optional CRISPR mini sketch. |
| Method | Zone C measured-only funnel and Zone D Graph A/B/C. |
| Results | Zone E two-axis result band and Zone F TN/169 visual. |
| Discussion / conclusion | Zone G axis divergence and Zone I contribution/limits. |
| References | Zone J footer references. |
| Author information | Zone A identity or Zone J footer. |

The section headings do not need to copy thesis headings, but a jury member must be able to locate each function quickly.

## 11. Low-fidelity draft options

### Option A - Hero-centered

Best when:

- Graph C target-observation representation is the dominant first impression.

Structure:

- Large title and Graph C hero at top.
- Data contract and Graph A/B/C immediately below.
- Results band in center.
- Literature and contribution/limits lower.

Strength:

- Strong contribution-first signal.

Risk:

- Results may feel secondary if the central band is too small.

### Option B - Results-centered

Best when:

- The two-axis finding should be the strongest visual moment.

Structure:

- Compact title and hero at top.
- Results band occupies central width.
- TN/169 visual receives strong emphasis.
- Method visuals frame the results above/left.

Strength:

- Makes MCC/specificity and rare-negative behavior visible.

Risk:

- Must avoid making operating-point behavior look like AUPRC superiority.

### Option C - Evidence-ladder / guided path

Best when:

- The poster needs a clear two-minute spoken pitch path.

Structure:

- Numbered or visually guided sequence:
  1. problem;
  2. data contract;
  3. Graph C;
  4. two-axis results;
  5. interpretation and boundary.

Strength:

- Good for jury reading and presentation.

Risk:

- Can become too linear and boxy if over-numbered.

Recommended starting point:

- Start from Option A or B, then borrow the guided clarity of Option C without turning the poster into a dense grid.

## 12. What not to do

Do not:

- choose the final production tool in this brief;
- build a LaTeX / Canva / slides draft here;
- use a many-box academic grid;
- fill empty space with paragraphs;
- make caveats tiny;
- present a raw-score literature leaderboard;
- use causal arrows for graph/model diagrams;
- use winner/trophy styling for metrics;
- merge AUPRC and operating-point results into one ambiguous score story;
- hide BTU poster functions behind a purely editorial layout;
- use decorative backgrounds that compete with figures.

## 13. Acceptance checklist for Slice 4 / later draft

Before moving to Turkish microcopy or a low-fidelity draft:

- [ ] Is there one clear first-glance visual idea?
- [ ] Is Graph C / target-observation visible early?
- [ ] Is the measured-only funnel visible before results?
- [ ] Are AUPRC and operating-point results separated?
- [ ] Is TN/169 rare-negative recognition visible?
- [ ] Is the honesty caveat inside results and not in the title?
- [ ] Are seed/guide fragility and no-causality caveats visible where needed?
- [ ] Is the literature panel qualitative only?
- [ ] Are BTU section functions identifiable?
- [ ] Is text density low enough for 70x100 cm print?
- [ ] Does the layout remain tool-agnostic?
- [ ] Does the Apple-like / keynote-inspired aesthetic support, not hide, the scientific boundary?
