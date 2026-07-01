# Figure Production Plan

This document is the Slice 2 figure-production plan for the CRISPR-Cas9 off-target prediction thesis poster. It specifies the figures that should be produced or adapted later, but it does not create figures, assets, layouts, or final poster copy.

The plan is production-tool agnostic. The same figure specs should work for Canva, LaTeX/tikzposter, slides, or another later production tool. Any later LaTeX artifact remains a low-fidelity fit-check / working draft, not a final production decision.

Planning language is English. Poster-bound Turkish labels and captions are quoted.

## 1. Purpose and scope

The goal is to define the visual evidence stack before layout work begins. Each figure spec states:

- what question the figure answers;
- which thesis or approved anchor source supports it;
- which numbers may appear;
- what visual form is likely to work;
- what Turkish labels or captions can be used;
- what claim caveat must travel with the figure;
- what claim risk the designer must avoid.

Out of scope:

- creating figures;
- choosing final colors or typography;
- finalizing Turkish copy;
- building a LaTeX / Canva / slides draft;
- editing the thesis.

## 2. Inputs and constraints

Governing files:

- `docs/poster/notes/poster_design_decisions.md`
- `docs/poster/notes/poster_narrative_framing.md`
- `docs/poster/notes/poster_yazim_kurallari.md`
- `docs/poster/plan/poster_content_plan.md`
- `docs/poster/plan/poster_section_shortlist.md`
- `docs/poster/plan/poster_execution_plan.md`
- `docs/poster/notes/poster_progress.md`

Constraints:

- Use only approved numeric anchors.
- Keep figures contribution-first.
- Do not claim robust AUPRC superiority over XGBoost F4.
- Do not imply biological causality from attention, gate, FiLM, masking, embedding, feature-family effects, or graph arrows.
- Treat MCC and specificity as operating-point evidence, not replacement primary metrics.
- Keep the literature panel qualitative; no raw-score leaderboard.
- Keep caveats visible in or near the figure they control.
- Use Turkish labels inside poster figures.
- Keep visual density low enough for an Apple-like / keynote-inspired poster draft: clean, spacious, visual-first, and calm.

## 3. Figure priority map

### Must-have

These figures form the core visual argument:

1. Measured-only funnel.
2. Graph A/B/C semantic comparison.
3. Two-axis result panel.
4. TN/169 rare-negative recognition visual.
5. Literature A+B qualitative positioning panel.

### Should-have

These figures improve accessibility or method continuity if space allows:

6. CRISPR for-dummies mini sketch.
7. Model / mechanism chain.
8. Feature-family chip strip.

### Cut-if-needed

These figures are useful only after the must-have visuals fit cleanly:

9. Detailed bootstrap / interval inset.
10. Historical XGBoost F4 version-drift note.
11. Detailed encoder or feature-family internals.

## 4. Shared visual-language rules

- Use Turkish labels and short captions.
- Keep schema names as `Graph A`, `Graph B`, and `Graph C`.
- Use arrows only for data flow, representation flow, or reading order; never for biological causality.
- Pair color with labels, shape, or position; do not encode meaning by color alone.
- Put numbers in callouts close to the visual object they explain.
- Keep caveats near the figure, not in a remote footnote.
- Use one dominant visual idea per figure.
- Prefer uncluttered visual forms over dense tables.
- Use whitespace and hierarchy instead of many boxed panels.
- If a visual cannot carry its caveat legibly, the visual is too dense.

## 5. Must-have figure specs

### Figure 1 - Measured-only funnel

Purpose:

- Make the evaluation contract and label discipline visible before the results.

Poster question answered:

- Which rows are allowed to become test evidence?

Source data / thesis source:

- Thesis data contract and approved poster anchor pool.

Required numbers:

- `310142 -> 25632 -> 1702`
- test: 29 guides, 1533 positives, 169 negatives

Likely visual form:

- A clean horizontal or vertical funnel with three large numeric stages.
- Use a small side callout for test composition.
- Avoid a dense data-processing flowchart.

Turkish labels / caption candidates:

- `"Tüm adaylar"`
- `"Ölçülmüş veri evreni"`
- `"Test evreni"`
- `"Doğrulanmamış adaylar (measured=0) test etiketi yapılmadı."`

Caption rule:

- State that `measured=0` rows are unverified candidates, not validation/test ground truth.

Claim caveat:

- `measured=0` must not be treated as false negatives or safe negatives.

Claim risk:

- A funnel can visually imply discarded rows are false. The caption must prevent that reading.

Production notes:

- This is a trust-building method visual and should appear early.
- It should be visually simple enough to scan from a distance.

### Figure 2 - Graph A/B/C semantic comparison

Purpose:

- Show the central representation contribution.

Poster question answered:

- What changes from Graph A and Graph B to Graph C?

Source data / thesis source:

- Thesis Graph A/B/C methodology sections and poster narrative framing.

Required numbers:

- None required.

Likely visual form:

- Three small aligned panels using consistent visual grammar.
- Graph A: physical target node.
- Graph B: guide-similarity control.
- Graph C: target-observation node that carries context.

Turkish labels / caption candidates:

- `"Graph A: fiziksel hedef şeması"`
- `"Graph B: rehber benzerliği kontrolü"`
- `"Graph C: hedef-gözlem (target-observation) şeması"`
- `"Aynı adres, farklı ziyaretler"`
- `"Graph C katkısı salt topoloji değil, düğüm semantiği değişimidir."`

Caption rule:

- Say Graph C changes target semantics, not just topology.

Claim caveat:

- Graph arrows show representation structure, not biological cause-effect.

Claim risk:

- A designer may draw Graph C as if context causes cleavage biologically. Avoid causal arrow language and causal arrow styling.

Production notes:

- This is the likely hero or near-hero visual.
- Keep node labels short; put technical explanation in a caption.

### Figure 3 - Two-axis result panel

Purpose:

- Separate ranking behavior from operating-point behavior.

Poster question answered:

- What happens on AUPRC ranking, and what happens at the decision threshold?

Source data / thesis source:

- Thesis results and robustness sections; approved poster anchor pool.

Required numbers:

- XGBoost F4 AUPRC 0.992338 [0.950179, 0.999336]
- S8B_R2 AUPRC 0.986020 [0.929981, 0.998966]
- prevalence 0.900705 as no-skill PR baseline
- family-aware encoder MCC 0.603489 if MCC is highlighted
- specificities 0.236, 0.083, 0.373, 0.651 if included beside the operating-point panel

Likely visual form:

- Two adjacent panels or two stacked rows:
  - `"Sıralama (AUPRC)"`
  - `"Karar Eşiği (MCC / specificity)"`
- AUPRC side can be a compact interval / point comparison rather than a crowded leaderboard.
- Operating-point side should point toward the TN/169 visual rather than duplicate all detail.

Turkish labels / caption candidates:

- `"Sıralama (AUPRC)"`
- `"Karar Eşiği"`
- `"AUPRC sıralama kalitesini ölçer."`
- `"MCC/specificity karar eşiğindeki negatif tanıma davranışını görünür kılar."`

Caption rule:

- State AUPRC remains the primary ranking metric and MCC/specificity are operating-point evidence.

Claim caveat:

- Include the exact honesty caveat inside or immediately below the results band:

> `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

Claim risk:

- The viewer may read the operating-point improvement as an AUPRC win. The axis split must make that impossible.

Production notes:

- This panel should visually connect to the rare-negative recognition visual.
- Do not use trophy, winner, or leaderboard styling.

### Figure 4 - TN/169 rare-negative recognition visual

Purpose:

- Make the operating-point result concrete.

Poster question answered:

- How many of the 169 measured negatives are recognized at the decision threshold?

Source data / thesis source:

- Approved poster anchor pool and thesis operating-point results.

Required numbers:

- true negatives over 169:
  - XGBoost F4 = 40
  - Graph C GCN = 14
  - Graph C GATv2 = 63
  - family-aware encoder = 110
- specificities 0.236, 0.083, 0.373, 0.651 if space allows

Likely visual form:

- Four equal-denominator bars, dot strips, or filled counters all labeled out of 169.
- The equal denominator must be visually obvious.
- Prefer "TN / 169" over percentages.

Turkish labels / caption candidates:

- `"Doğru negatif / 169"`
- `"XGBoost F4: 40/169"`
- `"Graph C GCN: 14/169"`
- `"Graph C GATv2: 63/169"`
- `"Aile-duyarlı encoder: 110/169"`
- `"Karar eşiği sonuçları validation-kilitli ve seed/guide duyarlıdır."`

Caption rule:

- Say all values are true negatives over the same 169 measured negatives at validation-selected thresholds.

Claim caveat:

- Operating-point gains are validation-locked, seed/guide-fragile, and limited by the 169-negative test universe.

Claim risk:

- The 110/169 value can look like a robust global win if the caveat is absent.
- The 14/169 Graph C GCN value must remain visible enough to avoid a false monotone "all graph context improves" story.

Production notes:

- This visual should be one of the clearest metric visuals on the poster.
- Use consistent ordering and labels across all result figures.

### Figure 5 - Literature A+B qualitative positioning panel

Purpose:

- Position the study against literature without invalid raw-score comparison.

Poster question answered:

- Why is this not a direct score leaderboard against other papers?

Source data / thesis source:

- Poster narrative framing and literature index.

Required numbers:

- None.

Likely visual form:

- A two-part qualitative panel:
  - A: contextual positioning
  - B: different-question / spiritual comparison
- A can use qualitative axes or check-style criteria:
  - leakage control
  - guide-disjoint evaluation
  - prevalence awareness
  - measured-only universe
- B can contrast ranking/retrieval question vs measured-only binary question.

Turkish labels / caption candidates:

- `"Neden doğrudan kıyaslanamaz?"`
- `"Veri, split, negatif üretimi ve prevalans farklı."`
- `"Kıyas, skor yarışı değil; soru ve sözleşme karşılaştırmasıdır."`

Caption rule:

- State that raw scores from other papers are not directly comparable because data, split, negatives, and prevalence differ.

Claim caveat:

- No raw AUPRC, accuracy, or leaderboard score against other papers.

Claim risk:

- Checkmarks can imply "better than literature" if not worded carefully. Use "stricter on this evaluation contract" rather than "best."

Production notes:

- Keep compact; this panel supports the jury layer and advisor request but should not dominate the poster.

## 6. Should-have figure specs

### Figure 6 - CRISPR for-dummies mini sketch

Purpose:

- Give non-specialist visitors a fast entry point.

Poster question answered:

- What is the off-target problem in one visual?

Source data / thesis source:

- Thesis introduction.

Required numbers:

- None.

Likely visual form:

- Simple guide RNA -> target DNA -> similar off-target region sketch.
- Use restrained icon-like forms, not a detailed molecular biology diagram.

Turkish labels / caption candidates:

- `"Rehber RNA"`
- `"Hedef DNA"`
- `"Benzer hedef dışı bölge"`
- `"Benzer bölgelerde hedef dışı kesim riski doğar."`

Caption rule:

- Keep it conceptual and accessible.

Claim caveat:

- Do not imply clinical safety assessment or biological validation.

Claim risk:

- Oversimplification may suggest sequence similarity is the whole study. Tie the sketch to context-aware representation.

### Figure 7 - Model / mechanism chain

Purpose:

- Connect data contract, Graph C, model, and results without overloading method text.

Poster question answered:

- How does the evidence chain flow?

Source data / thesis source:

- Thesis method and model sections; poster narrative framing.

Required numbers:

- Optional feature-family counts only if the chain includes context input families:
  - 6 experimental-epigenetic
  - 13 computed-nucleosome
  - 5 binding-energy

Likely visual form:

- A compact strip:
  - data contract -> Graph C -> GATv2 / family-aware encoder -> two-axis results

Turkish labels / caption candidates:

- `"Veri sözleşmesi"`
- `"Graph C"`
- `"Aile-duyarlı encoder"`
- `"İki eksenli sonuç"`
- `"Model içi açıklamalar biyolojik nedensellik kanıtı değildir."`

Caption rule:

- State this is an experiment/model chain, not a causal biological pathway.

Claim caveat:

- No-causality caveat required if model internals, attention, masking, FiLM, or feature families are shown.

Claim risk:

- A left-to-right chain can look causal. Use neutral process wording and avoid biological mechanism labels.

### Figure 8 - Feature-family chip strip

Purpose:

- Show what "context" contains without a dense table.

Poster question answered:

- Which input families carry context?

Source data / thesis source:

- Approved feature-family anchor pool and thesis feature sections.

Required numbers:

- 6 experimental-epigenetic + 13 computed-nucleosome + 5 binding-energy

Likely visual form:

- Small chips or grouped labels near the model / mechanism chain.

Turkish labels / caption candidates:

- `"6 deneysel epigenetik"`
- `"13 hesaplanmış nükleozom"`
- `"5 bağlanma enerjisi"`
- `"Özellik aileleri model girdisidir; biyolojik nedensellik kanıtı değildir."`

Caption rule:

- State these are model inputs and not causal claims.

Claim caveat:

- Avoid claiming a feature family biologically determines off-target activity.

Claim risk:

- A viewer may read feature families as mechanistic proof. Caption must frame them as model information under the evaluation contract.

## 7. Cut-if-needed figure specs

### Figure 9 - Detailed bootstrap / interval inset

Purpose:

- Show uncertainty rigor if the result panel has enough room.

Source data / thesis source:

- Thesis robustness section and approved AUPRC intervals.

Required numbers:

- XGBoost F4 AUPRC interval [0.950179, 0.999336]
- S8B_R2 AUPRC interval [0.929981, 0.998966]
- optional multi-seed:
  - XGBoost F4 0.990649 ± 0.001944
  - S8B_R2 0.978963 ± 0.011322

Likely visual form:

- Tiny interval strip or uncertainty footnote beside the AUPRC panel.

Caption rule:

- Use compatibility language, not equivalence language.

Claim risk:

- Too much interval detail can clutter the first draft. Cut this inset if the main two-axis result panel already carries uncertainty.

### Figure 10 - Historical XGBoost F4 version-drift note

Purpose:

- Explain regenerated vs historical XGBoost F4 only if someone needs the detail.

Source data / thesis source:

- Approved numeric anchor pool.

Required numbers:

- historical XGBoost F4 0.992522
- regenerated XGBoost F4 0.992338

Likely visual form:

- A tiny note, not a standalone graphic.

Caption rule:

- Historical value is a version-drift note; regenerated value is the robustness bar.

Claim risk:

- Can distract from the main story or look like cherry-picking. Omit unless necessary.

### Figure 11 - Detailed encoder or feature-family internals

Purpose:

- Support a technical jury question if there is extra space.

Source data / thesis source:

- Thesis model / encoder section.

Required numbers:

- None required beyond feature-family counts already specified.

Likely visual form:

- Small inset or schematic, not a main panel.

Caption rule:

- Model-internal outputs are model-behavior evidence only.

Claim risk:

- High risk of biological-causality overread. Prefer cutting this figure unless the layout is unusually spacious.

## 8. Figure-to-section mapping

| Poster section function | Figure support |
| --- | --- |
| Abstract / overview | Title zone plus Graph C hero visual |
| Introduction / aim | CRISPR for-dummies sketch, if space allows |
| Method | Measured-only funnel; Graph A/B/C comparison; model / mechanism chain |
| Results | Two-axis result panel; TN/169 rare-negative recognition visual |
| Discussion / conclusion | Axis-divergence explanation; contribution chips; limits strip |
| Literature positioning | Literature A+B qualitative panel |
| References / author info | Compact footer, no major figure |

The first layout should be built around the must-have method and result visuals, not around text boxes.

## 9. Numeric anchor placement inside figures

| Numeric anchor | Figure placement |
| --- | --- |
| `310142 -> 25632 -> 1702` | Measured-only funnel; must appear. |
| 29 guides, 1533 positives, 169 negatives | Funnel side callout or axis-divergence mini-panel; must appear somewhere. |
| Negatives in 9 guides; 80 in guide 9251 | Axis-divergence mini-panel or limits strip; should appear. |
| Prevalence 0.900705 | Ranking / PR panel as no-skill PR baseline; must appear if PR/AUPRC visual includes baseline. |
| XGBoost F4 AUPRC 0.992338 [0.950179, 0.999336] | AUPRC ranking panel; must appear if intervals fit. |
| XGBoost F4 multi-seed 0.990649 ± 0.001944 | Optional uncertainty inset. |
| Historical XGBoost F4 0.992522 | Optional version-drift note only. |
| S8B_R2 AUPRC 0.986020 [0.929981, 0.998966] | AUPRC ranking panel; must appear as best single GNN. |
| S8B_R2 multi-seed 0.978963 ± 0.011322 | Optional uncertainty inset. |
| TN over 169: 40, 14, 63, 110 | TN/169 rare-negative visual; must appear. |
| Specificities 0.236, 0.083, 0.373, 0.651 | Under TN/169 visual or operating-point panel; should appear if legible. |
| Family-aware encoder MCC 0.603489 | Operating-point panel if MCC is singled out. |
| 6 experimental-epigenetic + 13 computed-nucleosome + 5 binding-energy | Feature-family chip strip or model chain; should appear only if useful. |
| 154 sgRNA, 138747 target locations | Optional data-scale note; omit unless the data panel needs more scale context. |

No additional numbers should enter figure labels unless verified against the thesis and approved for poster use.

## 10. Caption and caveat placement

Required figure-level caveats:

- **Measured-only funnel:** `"Doğrulanmamış adaylar (measured=0) test etiketi yapılmadı."`
- **Graph A/B/C:** Graph C changes representation semantics; arrows are not biological causality.
- **AUPRC panel:** `0.900705` is `"no-skill PR baseline"`, not a performance floor.
- **Results band:** exact honesty caveat:

> `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

- **Operating-point / TN panel:** `"Karar eşiği sonuçları validation-kilitlidir ve seed/guide duyarlıdır."`
- **Model / mechanism visuals:** `"Model içi açıklamalar biyolojik nedensellik kanıtı değildir."`
- **Literature panel:** `"Kıyas, skor yarışı değil; soru ve sözleşme karşılaştırmasıdır."`

Do not place these caveats only in an endnote. They must travel with the figure that needs them.

## 11. Production handoff checklist

Before moving to Slice 3 layout draft brief:

- [ ] Does every must-have figure have a clear purpose?
- [ ] Does every must-have figure have a source data / thesis source?
- [ ] Are all numbers from the approved anchor pool?
- [ ] Are AUPRC and operating-point visuals visibly separated?
- [ ] Is MCC/specificity framed as operating-point evidence only?
- [ ] Does the TN/169 visual use the same denominator for all models?
- [ ] Is Graph C shown as target-observation semantics, not biological causality?
- [ ] Is the measured-only funnel clear about `measured=0`?
- [ ] Is the literature panel qualitative only?
- [ ] Are claim caveats attached to the relevant figures?
- [ ] Is each figure simple enough for a low-density 70x100 cm poster?
- [ ] Can the figure set support an Apple-like / keynote-inspired layout without hiding scientific limits?
