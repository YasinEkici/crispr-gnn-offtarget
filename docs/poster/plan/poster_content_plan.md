# Poster Content Plan

This document turns the poster constitution, narrative spine, and writing rules into a section-by-section content and figure plan for the CRISPR-Cas9 off-target prediction thesis poster. It is tool-agnostic and is not a final design, final copy sheet, or production layout. It gives a designer enough structure to draft the poster in Canva, LaTeX/tikzposter, slides, or another production tool without changing the scientific claims.

Poster language is Turkish. This planning document is English. Candidate text that may appear on the poster is quoted in Turkish.

## 1. Purpose and dependencies

This plan specifies what content belongs on the poster, where each claim should live, which numbers are allowed, and which figures are needed. It inherits the contribution-first reframe: the poster foregrounds Graph C target-observation representation, measured-only discipline, Scheme A, and rare-negative operating-point behavior. The fact that XGBoost F4 remains the AUPRC bar is reported as one result caveat, not as the headline.

Upstream dependencies:

- `docs/poster/notes/poster_design_decisions.md`
- `docs/poster/notes/poster_narrative_framing.md`
- `docs/poster/notes/poster_yazim_kurallari.md`
- `docs/thesis/notes/main_narrative_framing.md`
- thesis chapters under `docs/thesis/latex/btu_template/chapters/`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/DECISIONS.md`
- `docs/literature/literature_index.md`

This plan does not create figures and does not choose the final title.

## 2. Poster reading path

Recommended path for a 70x100 cm portrait poster:

1. **Top / first glance:** title, one-line thesis, hero visual showing context-aware graph representation.
2. **Upper-left or first numbered block:** problem and aim in for-dummies language.
3. **Upper-middle:** data contract and measured-only funnel.
4. **Center:** Graph A/B/C semantic comparison, making Graph C the main visual contribution.
5. **Center-right or middle band:** model/mechanism chain from schema to family-aware encoder.
6. **Results band:** two visibly separated axes: `"Sıralama (AUPRC)"` and `"Karar eşiği"`.
7. **Lower-middle:** why the axes diverge, tied to prevalence and 169 negatives.
8. **Lower-left or side panel:** literature positioning A+B.
9. **Bottom:** contribution, limits/future work, references, author/advisor info.

Visitor layer:

- Reads title, hero, Graph C metaphor, and main takeaway.
- Should leave with: context-aware graph representation did not claim AUPRC superiority, but located a rare-negative decision behavior.

Jury layer:

- Inspects Scheme A, measured-only, guide-disjoint split, no test tuning, Graph A/B/C semantics, metric distinction, uncertainty, and claim limits.

BTÜ section functions must remain identifiable even if the layout is free-form:

- abstract / overview,
- introduction and aim,
- method,
- results,
- discussion / conclusion,
- references,
- author information.

## 3. Content hierarchy

**Primary content** must survive any compression:

- title / hook,
- hero representation of context-aware Graph C,
- main two-axis takeaway,
- measured-only discipline as a trust signal,
- one-line honesty caveat inside results.

**Secondary content** supports jury evaluation:

- Scheme A label rule,
- 310142 -> 25632 -> 1702 funnel,
- Graph A/B/C semantic comparison,
- family-aware encoder and mechanism chain,
- ranking vs operating-point result panels,
- claim boundary on seed/guide fragility and non-causality.

**Tertiary content** can shrink first:

- exact bootstrap intervals if space is tight,
- detailed version-drift note,
- detailed feature-family counts,
- literature A+B nuance,
- references.

If space becomes tight, cut tertiary detail before cutting the measured-only funnel, Graph A/B/C semantic comparison, or two-axis result panel.

## 4. Candidate poster title zone

Final title is deferred. The title must be contribution-first, not a failure frame.

Candidate A:

> `"Aynı dizi, farklı bağlam: CRISPR-Cas9 hedef dışı tahmininde bağlam-duyarlı çizge temsili"`

Use when the hero visual foregrounds Graph C and target-observation semantics.

Candidate B:

> `"Sıralama mı, karar eşiği mi? Hedef dışı tahminde iki ayrı soru"`

Use when the design foregrounds the two-axis result story.

Candidate C:

> `"Doğrulanmış veri, sızıntısız değerlendirme: bağlam-duyarlı GNN'lerin nadir negatif katkısı"`

Use when the design foregrounds evaluation rigor and rare-negative behavior.

Candidate D:

> `"Aynı adres, farklı ziyaretler: Graph C ile hedef-gözlem bağlamı"`

Use only if the metaphor is visually explained and does not replace the technical term.

Candidate E:

> `"Bağlam nerede işe yarıyor? CRISPR-Cas9 hedef dışı tahmininde iki eksenli değerlendirme"`

Use when the poster wants a direct research-question headline.

Do not use:

- `"XGBoost'u geçemedik"`
- `"GNN XGBoost'u geçti"`
- `"En iyi hedef dışı tahmin modeli"`

## 5. Section-by-section content plan

### Problem / aim

Section function: give the visitor an accessible entry point and define the research question without overexplaining CRISPR.

Candidate Turkish heading:

> `"Problem ve Amaç"`

Key messages:

- CRISPR-Cas9 uses guide-target matching, but similar unintended regions can be cut.
- Sequence similarity alone is not the whole poster story; target context can matter.
- The aim is to test context-aware graph representations under a strict evaluation contract.

Poster-bound microcopy candidates:

- `"CRISPR-Cas9, hedef DNA'yı rehber RNA ile bulur; benzer bölgelerde hedef dışı kesim riski doğar."`
- `"Soru: bağlamı çizge içinde doğru temsil etmek, model davranışını nerede değiştirir?"`
- `"Bu poster en iyi model iddiası değil, bağlamın hangi eksende sinyal verdiğini gösterir."`

Required numbers: none in the main sentence.

Required visual: small CRISPR for-dummies sketch or guide-target-off-target icon chain.

Claim caveat: do not imply clinical safety or biological validation.

### Data contract / label discipline

Section function: make the evaluation trustworthiness visible early.

Candidate Turkish heading:

> `"Veri Sözleşmesi"`

Key messages:

- Scheme A is the primary binary label: `cleavage_freq > 1e-5`.
- Test/validation headline universe is measured-only.
- `measured=0` rows are not validation/test ground truth.
- Split discipline is guide-disjoint and no-test-tuning.

Poster-bound microcopy candidates:

- `"Scheme A: cleavage_freq > 1e-5"`
- `"310142 satırdan ölçülmüş veri evrenine, oradan 1702 satırlık test evrenine."`
- `"Doğrulanmamış adaylar (measured=0) test etiketi yapılmadı."`
- `"Model ve eşik seçimi validation ile kilitlendi; test üzerinde ayar yapılmadı."`

Required numbers:

- 310142 -> 25632 -> 1702
- 29 guides, 1533 positives, 169 negatives

Required visual: measured-only funnel.

Claim caveat: `measured=0` can be described as `"doğrulanmamış ipuçları"` only with the explicit no-test-ground-truth caveat.

### Graph representation: Graph A/B/C

Section function: explain the core representation contribution.

Candidate Turkish heading:

> `"Graph A/B/C"`

Key messages:

- Graph A is the minimal physical target graph.
- Graph B is a bounded guide-similarity control.
- Graph C changes target semantics by representing each row as a target-observation node.
- Graph C is not just "more edges"; it changes where context lives.

Poster-bound microcopy candidates:

- `"Graph A: fiziksel hedef düğümü."`
- `"Graph B: rehber benzerliği kontrolü."`
- `"Graph C: hedef-gözlem bağlamı."`
- `"Aynı adres, farklı ziyaretler: aynı fiziksel hedef farklı bağlamlarda farklı gözlemler taşır."`
- `"Graph C katkısı salt topoloji değil, düğüm semantiği değişimidir."`

Required numbers: none required, optional feature-family counts if space allows.

Required visual: Graph A/B/C semantic comparison with three small panels.

Claim caveat: avoid drawing Graph C as a causal biological pathway.

### Model / mechanism chain

Section function: connect representation to the model components without turning the poster into an architecture paper.

Candidate Turkish heading:

> `"Mekanizma Zinciri"`

Key messages:

- The evidence chain moves from schema to context features to family-aware encoder.
- Feature families are treated structurally rather than as one flat vector.
- Attention, FiLM, masking, and encoder outputs are model-behavior evidence only.

Poster-bound microcopy candidates:

- `"Bağlam sinyali hedef-gözlem düğümünde taşındı."`
- `"Aile-duyarlı encoder, özellik ailelerini ayrı işler."`
- `"Masking ve attention çıktıları model davranışını gösterir; biyolojik nedensellik kanıtı değildir."`

Required numbers:

- 6 experimental-epigenetic + 13 computed-nucleosome + 5 binding-energy features, if used.

Required visual: simplified evidence ladder or compact model-chain strip: data contract -> Graph C -> GATv2 -> family-aware encoder -> two-axis results.

Claim caveat: no causal interpretation from model internals.

### Results: ranking axis

Section function: state the AUPRC finding clearly and non-defensively.

Candidate Turkish heading:

> `"Sıralama (AUPRC)"`

Key messages:

- AUPRC remains the primary ranking metric.
- XGBoost F4 is the strongest AUPRC bar.
- Best single GNN comes close, but robustness intervals do not support superiority.
- This is a finding, not an apology.

Poster-bound microcopy candidates:

- `"AUPRC sıralama kalitesini ölçer."`
- `"XGBoost F4, AUPRC bakımından en sağlam referans olarak kaldı."`
- `"S8B_R2 yaklaştı; sağlam üstünlük iddiası kurulmadı."`
- `"Aralıklar fark yokluğu ile uyumlu; bu eşdeğerlik iddiası değildir."`

Required numbers:

- XGBoost F4 AUPRC 0.992338 [0.950179, 0.999336]
- XGBoost F4 multi-seed 0.990649 ± 0.001944
- S8B_R2 AUPRC 0.986020 [0.929981, 0.998966]
- S8B_R2 multi-seed 0.978963 ± 0.011322
- prevalence 0.900705 as no-skill PR baseline

Required visual: compact AUPRC comparison or PR-curve-inspired panel with prevalence line.

Claim caveat: do not imply equivalence; do not use a victory/leaderboard tone.

### Results: operating-point axis

Section function: give MCC/specificity and true-negative recognition enough visible weight.

Candidate Turkish heading:

> `"Karar Eşiği"`

Key messages:

- MCC and specificity describe behavior at a validation-selected threshold.
- Here the rare class is negative, so true negatives matter.
- Graph C GATv2 and family-aware encoder recover more measured negatives at the operating point.
- This does not override AUPRC ranking and is seed/guide fragile.

Poster-bound microcopy candidates:

- `"Karar eşiği, modelin hangi örneğe pozitif/negatif dediğini gösterir."`
- `"Bu test evreninde nadir sınıf negatiftir: 169 negatif."`
- `"Doğru negatif sayısı: 40 -> 63 -> 110."`
- `"MCC/specificity bu nadir negatif davranışını görünür kılar."`

Required numbers:

- true negatives over 169: XGBoost F4 = 40, Graph C GCN = 14, Graph C GATv2 = 63, family-aware encoder = 110
- specificities 0.236, 0.083, 0.373, 0.651
- family-aware encoder MCC 0.603489

Required visual: rare-negative recognition visual, preferably "true negatives out of 169" with equal denominator.

Claim caveat: must say validation-kilitli and seed/guide-fragile.

### Why the axes diverge

Section function: prevent the viewer from seeing the AUPRC and MCC stories as contradictory.

Candidate Turkish heading:

> `"Neden İki Eksen?"`

Key messages:

- AUPRC is threshold-free and dominated here by abundant positives.
- The test prevalence is 0.900705, so strong models cluster near high AUPRC.
- MCC/specificity are sensitive to the scarce negatives.
- The same scarcity makes operating-point gains fragile.

Poster-bound microcopy candidates:

- `"Sıralama bol sayıda pozitifle doyuma ulaşır; karar eşiği az sayıdaki negatife bakar."`
- `"Fark da, kırılganlık da buradan gelir."`
- `"169 negatif, 9 guide içinde yoğunlaşır; 80'i guide 9251 üzerindedir."`

Required numbers:

- prevalence 0.900705
- 1533 positives, 169 negatives
- negatives in 9 guides, 80 in guide 9251

Required visual: split explanatory mini-panel: positives drive ranking saturation; negatives drive operating-point swings.

Claim caveat: do not say MCC is "better" globally. Say it answers a different question here.

### Literature positioning A+B

Section function: answer the advisor's literature-comparison request without invalid raw-score comparisons.

Candidate Turkish heading:

> `"Literatürde Nereye Oturuyor?"`

Key messages:

- Direct raw-score comparison is not valid across different data, splits, negatives, and prevalence.
- Contextual positioning shows this study is stricter on leakage control, guide-disjoint evaluation, prevalence awareness, and measured-only universe.
- Spiritual comparison: much literature asks ranking/retrieval questions; this poster asks a measured-only binary decision question.

Poster-bound microcopy candidates:

- `"Neden doğrudan kıyaslanamaz?"`
- `"Veri, split, negatif üretimi ve prevalans farklı."`
- `"Bizim soru: ölçülmüş evrende bağlam, karar davranışını nerede değiştirir?"`

Required numbers: none.

Required visual: qualitative A+B panel with axis ticks or checkmarks, not scores.

Claim caveat: no raw-score leaderboard against other papers.

### Contribution / takeaway

Section function: give the confident "what we add" message.

Candidate Turkish heading:

> `"Ne Katıyoruz?"`

Key messages:

- Controlled measured-only and leakage-aware evaluation.
- Graph C target-observation representation.
- Mechanism-isolated rare-negative recognition behavior.
- Honest uncertainty boundary.

Poster-bound microcopy candidates:

- `"Bağlamı hedef-gözlem düzeyinde temsil ettik."`
- `"Putative adayları test ground-truth yapmadık."`
- `"Bağlam katkısını AUPRC zaferi olarak değil, karar eşiği davranışı olarak konumlandırdık."`
- `"Sonuç: güçlü, sınırlı ve dürüst bir değerlendirme çerçevesi."`

Required numbers: optional; avoid overloading this section.

Required visual: four contribution chips or compact numbered list.

Claim caveat: do not imply state-of-the-art predictor.

### Limits / future work

Section function: make the scientific boundary visible without making it the main story.

Candidate Turkish heading:

> `"Sınırlar ve Sonraki Adım"`

Key messages:

- AUPRC superiority over XGBoost F4 is not claimed.
- Rare-negative operating-point effects are threshold-, seed-, and guide-fragile.
- More measured negatives and more negative-bearing guides are needed.
- Biological validation is future work, not this poster's claim.

Poster-bound microcopy candidates:

- `"AUPRC üstünlüğü iddia edilmedi."`
- `"Nadir negatif kazanım seed/guide duyarlıdır."`
- `"Daha fazla ölçülmüş negatif ve dış doğrulama gerekir."`
- `"Model içi açıklamalar biyolojik nedensellik kanıtı değildir."`

Required numbers:

- 169 negatives and/or 9 guides if not already visible elsewhere.

Required visual: compact caveat strip, not a large warning panel.

Claim caveat: this section is the caveat.

### References and author info

Section function: satisfy academic poster requirements without consuming the main visual space.

Candidate Turkish heading:

> `"Kaynaklar ve Bilgi"`

Key messages:

- Include essential literature anchors only.
- Include thesis title, authors, advisor, department, university.
- Keep references compact.

Poster-bound microcopy candidates:

- `"Kasım Deliacı & Yasin Ekici"`
- `"Danışman: Doç. Dr. Mustafa Özgür Cingiz"`
- `"Bursa Teknik Üniversitesi, Bilgisayar Mühendisliği"`

Required numbers: none.

Required visual: author/info footer with BTÜ identity.

Claim caveat: none, except avoid suggesting direct reproduction of cited papers.

## 6. Figure plan

### Must-have visuals

**1. Measured-only funnel**

- Purpose: make label integrity and evaluation discipline visible.
- Source data / thesis source: thesis data contract and poster anchor pool.
- Likely visual form: horizontal or vertical funnel: `310142 -> 25632 -> 1702`.
- Caption rule: state measured-only and no test use of `measured=0`.
- Claim risk: do not imply `measured=0` rows are false; they are unverified candidates.

**2. Graph A/B/C semantic comparison**

- Purpose: show the central representation contribution.
- Source data / thesis source: thesis Graph A/B/C methodology sections.
- Likely visual form: three mini-panels with consistent node/edge symbols.
- Caption rule: Graph C changes target semantics, not just topology.
- Claim risk: arrows must not imply biological causality.

**3. Two-axis result panel**

- Purpose: split ranking and decision-threshold findings.
- Source data / thesis source: thesis results and robustness sections.
- Likely visual form: two adjacent panels: AUPRC / operating point.
- Caption rule: AUPRC primary; MCC/specificity secondary but central for rare negatives.
- Claim risk: viewer must not read operating-point gain as AUPRC win.

**4. Rare-negative recognition visual**

- Purpose: show why MCC/specificity matter here.
- Source data / thesis source: true-negative counts in the approved anchor pool.
- Likely visual form: four bars or 169-dot denominator strips.
- Caption rule: all values are true negatives out of 169 at validation-selected thresholds.
- Claim risk: must carry seed/guide-fragile caveat.

**5. Literature positioning A+B**

- Purpose: satisfy literature comparison without invalid score ranking.
- Source data / thesis source: literature index and poster narrative framing.
- Likely visual form: qualitative axis/check matrix plus "different question" note.
- Caption rule: `"neden doğrudan kıyaslanamaz?"`
- Claim risk: no raw AUPRC/scores from other papers.

### Should-have visuals

**6. CRISPR for-dummies mini sketch**

- Purpose: accessible entry point for non-specialists.
- Source data / thesis source: thesis introduction.
- Likely visual form: guide RNA -> target DNA -> off-target lookalike.
- Caption rule: simple risk framing, no clinical promise.
- Claim risk: avoid oversimplifying off-target biology.

**7. Model / mechanism chain**

- Purpose: connect Graph C, context features, GATv2, and family-aware encoder.
- Source data / thesis source: thesis model and mechanism sections.
- Likely visual form: evidence ladder or compact flow.
- Caption rule: model-behavior evidence only.
- Claim risk: avoid causal biological arrows.

**8. Feature-family chip strip**

- Purpose: show what "context" includes.
- Source data / thesis source: feature-family counts.
- Likely visual form: chips for sequence, energy, experimental epigenetic, computed nucleosome, missingness.
- Caption rule: features are model inputs, not causal claims.
- Claim risk: do not overstate computed-nucleosome contribution.

### Cut-if-needed visuals

**9. Detailed bootstrap interval inset**

- Purpose: show uncertainty rigor.
- Source data / thesis source: Sprint 9 robustness.
- Likely visual form: small interval strip.
- Caption rule: intervals compatible with no difference.
- Claim risk: may be too dense; cut first if the two-axis result panel already carries uncertainty.

**10. Version-drift note graphic**

- Purpose: explain historical vs regenerated F4.
- Source data / thesis source: Sprint 9 decision.
- Likely visual form: tiny footnote only if needed.
- Caption rule: regenerated F4 is the robustness bar; historical value is version drift note.
- Claim risk: overexplaining may distract from the main narrative.

## 7. Numeric anchor placement

| Anchor | Placement decision |
| --- | --- |
| 310142 rows -> measured-only 25632 -> test 1702 | Must appear in data contract funnel. |
| Test: 29 guides, 1533 positives, 169 negatives | Must appear near the funnel or axis-divergence panel. |
| Negatives in 9 guides, 80 in guide 9251 | Should appear in axis-divergence or limits; use to explain fragility. |
| Prevalence 0.900705 | Must appear in ranking/PR context as no-skill PR baseline, not a floor. |
| XGBoost F4 AUPRC 0.992338 [0.950179, 0.999336] | Must appear in ranking axis if space permits intervals; otherwise point value with interval in caption. |
| XGBoost F4 multi-seed 0.990649 ± 0.001944 | Should appear in ranking axis or small uncertainty note. |
| Historical XGBoost F4 0.992522 | Optional footnote only; do not mix with regenerated value as if identical. |
| S8B_R2 AUPRC 0.986020 [0.929981, 0.998966] | Must appear as best single GNN in ranking axis. |
| S8B_R2 multi-seed 0.978963 ± 0.011322 | Should appear in uncertainty note if multi-seed is shown. |
| TN over 169: XGBoost F4 = 40, Graph C GCN = 14, Graph C GATv2 = 63, family-aware encoder = 110 | Must appear in rare-negative recognition visual. |
| Specificities 0.236, 0.083, 0.373, 0.651 | Should appear under the TN visual or in a compact tooltip/caption. |
| Family-aware encoder MCC 0.603489 | Must appear in operating-point panel if MCC is singled out. |
| 6 experimental-epigenetic + 13 computed-nucleosome + 5 binding-energy | Should appear in feature-family strip or model/mechanism chain. |
| 154 sgRNA, 138747 target locations | Optional; include only if the data panel needs scale beyond row counts. |

No additional numeric values should be added unless read directly from the thesis and approved for poster use.

## 8. Claim boundary placement

The single honesty caveat belongs inside the results band, between or under the two result axes. It must not appear in the title zone.

Use this sentence exactly:

> `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

Seed/guide fragility appears in:

- operating-point panel caption,
- axis-divergence mini-panel,
- limits/future-work strip.

Suggested Turkish caveat:

> `"Karar eşiği kazanımı validation-kilitli, seed/guide duyarlı ve 169 negatifle sınırlıdır."`

No-causality caveat appears in:

- model/mechanism chain,
- figure captions for attention, FiLM, masking, encoder, or feature-family visuals.

Suggested Turkish caveat:

> `"Model içi açıklamalar biyolojik nedensellik kanıtı değildir."`

No-skill baseline caveat appears in:

- ranking axis / PR visual.

Suggested Turkish caveat:

> `"0.900705 no-skill PR baseline'dır; performans alt sınırı değildir."`

## 9. Literature positioning panel

The panel has two parts.

**A. Contextual positioning**

Panel question:

> `"Neden doğrudan kıyaslanamaz?"`

Use qualitative axes only:

- leakage control,
- guide-disjoint evaluation,
- prevalence awareness,
- measured-only universe.

The design can use checkmarks, axis ticks, or short labels. It must not show raw AUPRC or leaderboard scores from other papers.

**B. Spiritual comparison**

Message:

- much of the literature answers ranking/retrieval questions;
- this poster answers a measured-only binary decision question under a strict contract;
- therefore the study is positioned by problem framing and evaluation rigor, not by raw score.

Poster-bound microcopy candidates:

- `"Literatürdeki skorlar farklı evrenlerden gelir."`
- `"Bizim soru: ölçülmüş ve rehber ayrık evrende bağlam neyi değiştirir?"`
- `"Kıyas, skor yarışı değil; soru ve sözleşme karşılaştırmasıdır."`

## 10. Text budget

Apply the writing-rules budgets. The content plan should keep the poster visual-first.

| Section | Rough budget | Mostly visual? |
| --- | ---: | --- |
| Title zone | title max 2 lines; subtitle max 2 lines | Yes |
| Problem / aim | 25-45 words | Partly |
| Data contract | 20-35 words plus funnel labels | Yes |
| Graph A/B/C | 30-50 words plus schema labels | Yes |
| Model / mechanism chain | 25-45 words | Yes |
| Ranking axis | 20-35 words plus metrics | Yes |
| Operating-point axis | 20-40 words plus TN/MCC labels | Yes |
| Why axes diverge | 35-55 words | Partly |
| Literature A+B | 35-55 words | Partly |
| Contribution / takeaway | 4-6 short bullets | No, but compact |
| Limits / future work | 3-4 short bullets | No, but compact |
| References / author info | compact footer | No |

Mostly visual sections:

- measured-only funnel,
- Graph A/B/C semantic comparison,
- two-axis result panel,
- rare-negative recognition visual,
- literature positioning A+B.

Avoid body paragraphs longer than three short lines.

## 11. Open decisions

These are intentionally deferred:

- final poster title,
- exact hero visual,
- exact palette and accent colors,
- exact typography,
- final production tool,
- final figure styling and whether thesis figures are recolored,
- whether axis-divergence is its own panel or nested under results,
- exact placement of the literature A+B panel,
- exact reference count.

Do not resolve these inside this content plan unless a later design phase fixes them.

## 12. Pre-production checklist

Before turning this plan into a poster draft:

- [ ] Is the top message contribution-first?
- [ ] Are BTÜ section functions identifiable?
- [ ] Is the poster readable in visitor and jury layers?
- [ ] Are Graph C and target-observation semantics visible?
- [ ] Is the measured-only funnel visible?
- [ ] Is every number traceable to the approved anchor pool or thesis?
- [ ] Is `0.900705` labeled as no-skill PR baseline, not a floor?
- [ ] Are AUPRC and operating-point results separated?
- [ ] Is the honesty caveat present exactly once inside results?
- [ ] Are MCC/specificity and true negatives treated as central operating-point evidence?
- [ ] Are operating-point gains marked validation-kilitli and seed/guide duyarlı?
- [ ] Is no biological causality claimed from model internals?
- [ ] Is the literature panel qualitative, with no raw-score leaderboard?
- [ ] Is there no AI-filler wording?
- [ ] Is text density acceptable for 70x100 cm portrait print?
