# Poster Writing Rules

This document defines the writing and visual-language rules for the CRISPR-Cas9 off-target prediction thesis poster. It is written in English because it is a planning note. Any phrase that may appear directly on the poster is kept in Turkish inside quotes.

Governing documents:

- `docs/poster/notes/poster_design_decisions.md`
- `docs/poster/notes/poster_narrative_framing.md`
- `docs/thesis/notes/tez_yazim_meta_kurallari.md`
- the current thesis chapters under `docs/thesis/latex/btu_template/chapters/`

## 1. Purpose and scope

This rules document governs poster text, headings, figure captions, callouts, metric labels, caveats, and the wording embedded inside visuals. It does not design the poster, choose the final title, build figures, or fix the production tool.

All poster writing inherits the poster constitution and narrative spine:

- contribution-first, not `"XGBoost'u geçemedik"` as the story;
- Turkish poster text, English planning documents;
- accessible wording without changing any claim;
- no invented numbers;
- no AUPRC superiority claim over XGBoost F4;
- no biological-causality claim from attention, gate, FiLM, embedding, masking, or feature-ablation outputs.

Use this document as a checklist while drafting every poster block.

## 2. Register and tone

The poster uses concise, glanceable, active Turkish. It should feel confident and concrete, not defensive or inflated.

Write for two reading layers:

- **Visitor layer:** title, hero visual, main takeaway, and the central metaphor must be understandable in a few seconds.
- **Jury layer:** data contract, graph semantics, metrics, uncertainty, and claim boundary must be inspectable without a spoken explanation.

Rules:

- Prefer short sentences with one job each.
- Prefer active, direct phrasing over thesis-style heavy passive voice.
- Use Turkish first; keep technical English terms only where they are canonical or clearer.
- Define a technical term once, then reuse the same form.
- Use a "for-dummies" metaphor only if it changes no claim.
- Put the contribution before the caveat, then state the caveat plainly.

Good poster register:

- `"Aynı fiziksel hedef, farklı ölçüm bağlamlarında farklı gözlemler taşır."`
- `"Graph C bu bağlamı hedef-gözlem düğümünde taşır."`
- `"AUPRC sıralamayı, karar eşiği ise negatif tanıma davranışını gösterir."`

Avoid:

- long thesis paragraphs;
- decorative scientific-sounding claims;
- apology framing;
- casual slogans that hide the uncertainty.

## 3. AI-filler and banned patterns

The poster must not sound like generic AI-generated academic text. Remove empty frames and decorative connectors.

| Banned pattern | Do not write | Instead, write |
| --- | --- | --- |
| Empty time framing | `"Günümüzde CRISPR önemli bir yer tutmaktadır."` / "In today's world..." | Start with the concrete problem: `"CRISPR-Cas9 hedef dışı kesimler güvenilirlik sorunudur."` |
| Technology filler | `"Gelişen teknolojiyle birlikte..."` / "With developing technology..." | Name the actual object: `"Rehber RNA ile hedef DNA eşleşmesi hesaplanabilir bir tahmin problemidir."` |
| Inflated importance | `"Literatürde önemli bir yer tutmaktadır."` / "plays an important role" | State the role: `"Bu çalışma XGBoost F4'ü aynı sözleşmede güçlü referans olarak kullanır."` |
| Contrast padding | `"yalnızca X değil, aynı zamanda Y"` / "not only X but also Y" | Use a direct sentence: `"Graph C hem düğüm semantiğini hem context yerleşimini değiştirir."` |
| Chained connectors | `"Ayrıca / Bununla birlikte / Dahası"` at every block start | Use visual grouping or a concrete transition: `"İkinci eksen: karar eşiği."` |
| Decorative em-dashes | `"katkı - dikkat çekici biçimde - burada görülür"` | Use commas or a new sentence: `"Katkı burada görülür. Bu katkı seed/guide duyarlıdır."` |
| Empty value words | `"değer katmak"`, `"fark yaratmak"`, "adds value" | Name the measured behavior: `"63/169 negatifi doğru negatif sınıflandırır."` |
| Fake triples | `"hızlı, güvenilir ve etkili"` / "robust, scalable, effective" | Use one supported property: `"guide-disjoint"` or `"measured-only"` |

Quick scan before keeping any text: search for `"Günümüzde"`, `"yalnızca"`, `"önemli bir yer"`, `"Ayrıca"`, `"Bununla birlikte"`, `"Dahası"`, and decorative dashes.

## 4. Terminology and consistency

At first use, give the Turkish term plus a short gloss if needed. After that, keep the same form throughout the poster.

Canonical forms:

| Concept | Poster form |
| --- | --- |
| CRISPR-Cas9 off-target prediction | `"CRISPR-Cas9 hedef dışı tahmini"` |
| off-target site | `"hedef dışı bölge (off-target site)"` at first use; then `"hedef dışı bölge"` |
| sgRNA | `"sgRNA"` or `"rehber RNA (sgRNA)"` at first use |
| guide | `"rehber"` in Turkish prose; `guide` only inside fixed terms |
| guide-disjoint | `"rehber ayrık (guide-disjoint)"` at first use; then `"rehber ayrık"` |
| measured-only | `"ölçülmüş veri evreni (measured-only)"` at first use; then `"ölçülmüş veri evreni"` |
| measured=0 | `"doğrulanmamış adaylar (measured=0)"`; metaphor allowed: `"doğrulanmamış ipuçları"` |
| Scheme A | `"Scheme A: cleavage_freq > 1e-5"` |
| Graph A | `"Graph A: fiziksel hedef şeması"` |
| Graph B | `"Graph B: rehber benzerliği kontrolü"` |
| Graph C | `"Graph C: hedef-gözlem (target-observation) şeması"` |
| target-observation | `"hedef-gözlem (target-observation)"` at first use; then `"hedef-gözlem"` |
| candidate edge | `"aday kenar"` |
| target context | `"hedef bağlamı"` or `"target context"` only in model-name contexts |
| family-aware encoder | `"aile-duyarlı encoder"` |
| GATv2 | `"GATv2"` |
| XGBoost F4 | `"XGBoost F4"` |
| AUPRC | `"AUPRC"`; first use may add `"precision-recall eğrisi altındaki alan"` |
| MCC | `"MCC"`; first use may add `"Matthews korelasyon katsayısı"` |
| specificity | `"specificity (özgüllük)"` at first use; then pick one form per panel |
| operating point | `"karar eşiği (operating point)"` at first use; then `"karar eşiği"` |
| no-skill PR baseline | `"no-skill PR baseline"` with the caveat that it is not a floor |
| validation-selected threshold | `"validation-kilitli karar eşiği"` |
| context-aware | `"bağlam duyarlı"` |
| feature family | `"özellik ailesi"` |
| computed nucleosome | `"hesaplanmış nükleozom"` |
| experimental epigenetic | `"deneysel epigenetik"` |
| binding energy | `"bağlanma enerjisi"` |

Do not switch among `"graf"`, `"graph"`, and `"çizge"` randomly. The poster may use `"çizge"` in explanatory Turkish, but schema names stay as `Graph A/B/C`.

## 5. Claim-safe phrasing rules

Every text block and figure must respect the same claim boundary.

### AUPRC and superiority

Forbidden:

- `"XGBoost'u geçtik."`
- `"GNN en iyi model oldu."`
- `"Graph C XGBoost'tan üstündür."`

Allowed:

- `"XGBoost F4, AUPRC bakımından en sağlam referans olarak kaldı."`
- `"En iyi tekil GNN sonucu AUPRC'de yaklaştı, fakat sağlam üstünlük göstermedi."`
- `"AUPRC farkları guide ve tohum belirsizliğiyle uyumludur."`

Use the single honesty caveat only inside the results area, never as the title:

> `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

### Compatibility, not equivalence

Forbidden:

- `"XGBoost ile eşdeğer."`
- `"GNN ve XGBoost aynı performansta."`

Allowed:

- `"Aralık sıfırı içerdiği için fark, fark yokluğu ile uyumludur."`
- `"Sonuç eşdeğerlik iddiası değildir."`

If writing in Turkish, use:

- `"fark yokluğu ile uyumlu"`
- not `"eşdeğer"`

### Model behavior, not biological causality

Forbidden:

- `"Attention biyolojik mekanizmayı açıkladı."`
- `"Epigenetik özellikler negatifleri nedensel olarak belirledi."`
- `"FiLM bağlamın biyolojik etkisini kanıtladı."`

Allowed:

- `"Bu çıktı model davranışını açıklar; biyolojik nedensellik kanıtı değildir."`
- `"Deneysel epigenetik aile, bu ölçüm sözleşmesi altında ayırt edici model bilgisi taşır."`
- `"Masking sonucu, modelin hangi girdiye duyarlı olduğunu gösterir."`

### Fragility

Any operating-point gain must carry a caveat:

- `"seed/guide duyarlı"`
- `"validation-kilitli karar eşiği altında"`
- `"169 negatif ve 29 guide ile sınırlı test evreninde"`

Allowed:

- `"GATv2 ve aile-duyarlı encoder daha fazla nadir negatifi tanıdı; bu davranış seed/guide duyarlıdır."`

Forbidden:

- `"Nadir negatif problemi çözüldü."`

### Sequence models

Forbidden:

- `"Sequence modeller başarısızdır."`
- `"Dizi bilgisi işe yaramaz."`

Allowed:

- `"Bu ölçüm sözleşmesi ve bu mimari adaylar altında sequence-only yol sınırlı kaldı."`

## 6. Text-density and length budgets

The poster is 70x100 cm portrait and visual-first. Use short blocks, not paragraphs.

Recommended caps:

| Element | Budget |
| --- | --- |
| Main title | 8-12 Turkish words if possible; maximum 2 lines |
| Subtitle / thesis line | 14-22 words; maximum 2 lines |
| Section heading | 1-4 words |
| Panel question | 4-8 words, e.g. `"Sıralama mı, karar eşiği mi?"` |
| Body chunk | 20-35 words; maximum 3 short lines |
| Contribution bullet | 8-14 words |
| Figure caption | 12-25 words |
| Numeric callout | 1 number + 3-7 words |
| Honesty caveat | exactly one line if layout allows; do not expand into a paragraph |
| Main takeaway | 18-28 words; maximum 2 lines |
| References | compact; only essential citations |

Rules:

- One visual block should answer one question.
- Put numbers in callouts, not buried in prose.
- Use bullets for lists of contributions or constraints.
- Do not use long explanatory paragraphs except in a tiny methods/contract block, and even there keep the block under 45 words.
- If a block needs more than three sentences, split it into a figure plus caption.

## 7. Number formatting rules

Use the thesis numeric style for consistency.

### Decimal and count conventions

- Decimal separator: period, as in `0.992338`, `0.900705`, `0.603489`.
- Do not use comma decimals on the poster, even in Turkish prose.
- Large anchor counts use the thesis/plain style without thousands separators: `310142`, `25632`, `1702`.
- If visual readability requires grouping in a figure, group with spaces only after review, e.g. `310 142`; do not silently mix styles in the same panel.

### Metrics

- AUPRC, MCC, specificity, AUROC, macro-F1: show as decimals.
- Do not convert metric values to percentages unless the figure axis is explicitly percent-based.
- Specificity may be shown with three decimals in compact callouts only if the exact thesis value is preserved elsewhere in the figure or caption. Preferred for metric tables: six decimals.

Examples:

- `"AUPRC 0.992338"`
- `"MCC 0.603489"`
- `"specificity 0.651"` only as a compact callout for `0.650888`; avoid if exactness matters.
- `"110/169 doğru negatif"`

### Intervals and uncertainty

- Intervals: `[0.950179, 0.999336]`.
- Paired differences may use signs: `+0.003263`, `-0.00632`.
- Use `±` for mean and standard deviation: `0.990649 ± 0.001944`.
- Do not write `"equal"` or `"equivalent"` when an interval includes zero.

### Numeric source rule

Every number must trace to the thesis or the approved poster anchor pool. Do not invent:

- new rounded prevalence values;
- new percentages;
- new headline deltas;
- simplified row counts;
- cross-paper score comparisons.

The approved anchor pool includes universe/split counts, prevalence `0.900705`, XGBoost F4 regenerated and historical AUPRC values, S8B_R2 AUPRC values, operating-point true-negative counts, specificities, family-aware encoder MCC, and feature-family counts from `poster_narrative_framing.md` section 10.

## 8. Figure / visual-language wording

Captions carry the claim boundary. A figure is not allowed to imply a stronger claim than the text.

Rules:

- Use Turkish labels inside figures.
- Use schema names `Graph A`, `Graph B`, `Graph C` unchanged.
- Label the two result axes visibly: `"Sıralama (AUPRC)"` and `"Karar eşiği (MCC / specificity)"`.
- Any PR curve or AUPRC comparison that includes prevalence must label `0.900705` as `"no-skill PR baseline"`, not `"taban performans"`.
- Arrows show data flow, representation flow, or experiment sequence; they must not imply biological causality.
- Encoder boxes should be labeled as model components, not biological mechanisms.
- Attention/gate/FiLM visuals need an interpretation-only caption.
- Literature positioning must be qualitative. Do not draw a raw-score leaderboard against other papers.

Allowed metaphors:

- `"aynı adres, farklı ziyaretler"` for Graph C, if paired with target-observation wording.
- `"doğrulanmamış ipuçları"` for `measured=0`, if the caption says they are not validation/test ground truth.
- `"iki ayrı soru"` for AUPRC vs operating point.

Caption examples:

- `"Graph C, aynı fiziksel hedefi farklı hedef-gözlem bağlamlarıyla temsil eder."`
- `"Attention çıktıları model davranışını gösterir; biyolojik nedensellik kanıtı değildir."`
- `"0.900705 no-skill PR baseline'dır; performans alt sınırı değildir."`
- `"Karar eşiği sonuçları validation-kilitlidir ve seed/guide duyarlıdır."`

Do not write:

- `"Graph C bağlamın biyolojik etkisini kanıtlar."`
- `"Bu ok epigenetik etkinin nedeni olduğunu gösterir."`
- `"Literatüre göre en iyi sonuç."`

## 9. Accessibility and print legibility rules

These are wording-level rules for the draft. Exact palette and typography are fixed later.

Rules:

- Do not encode meaning by color alone. Pair color with labels, icons, shape, or position.
- Every colored metric or class label must remain distinguishable in grayscale.
- Use high-contrast text for all numbers and caveats.
- Avoid pale text on saturated backgrounds in captions or metric labels.
- Keep dense metric tables minimal; move detail to figure labels or callouts.
- Avoid all-caps Turkish for long text; use it only for short headings if the design requires it.
- Keep figure labels short enough to read from poster distance.
- Do not place critical caveats in tiny footnotes.
- Use the same term shape across all panels so the reader does not need to relearn labels.

Minimum practical legibility targets for 70x100 cm print:

- title readable from several meters;
- section heads readable from normal walking distance;
- body/caption text readable from close inspection;
- metric labels and axis labels readable without zooming or asking for the PDF.

If a claim boundary needs tiny text to fit, the block is too dense.

## 10. Heading / section preservation

The poster may use a free visual layout, but the BTÜ poster sections must remain identifiable.

Preserve these section functions:

- **Abstract / overview:** the one-glance problem and contribution.
- **Introduction + aim:** why off-target prediction and why context-aware graph representation.
- **Method:** data contract, Scheme A, measured-only, guide-disjoint split, Graph A/B/C.
- **Results:** two-axis findings, with the honesty caveat inside the results area.
- **Discussion / conclusion:** what the finding means and what it does not mean.
- **References:** compact literature anchors.
- **Author information:** thesis title, authors, advisor, department, university.

The headings do not need to copy the thesis exactly, but a jury member must be able to identify these functions quickly.

Recommended poster-facing section labels:

- `"Problem ve Amaç"`
- `"Veri Sözleşmesi"`
- `"Graph A/B/C"`
- `"Bulgular: İki Eksen"`
- `"Ne Katıyoruz?"`
- `"Sınırlar"`
- `"Kaynaklar"`

## 11. Quick checklist

Run this checklist over every poster block before keeping it:

- [ ] Does the block support the contribution-first reframe?
- [ ] Is the claim limited to the measured-only, guide-disjoint evaluation contract?
- [ ] Is every number traceable to the thesis or approved anchor pool?
- [ ] Are decimals, intervals, and `±` formatted consistently?
- [ ] Does the text avoid AUPRC superiority over XGBoost F4?
- [ ] If an interval includes zero, does it say `"fark yokluğu ile uyumlu"` rather than `"eşdeğer"`?
- [ ] Are operating-point gains tagged as validation-kilitli and seed/guide duyarlı?
- [ ] Are attention, gate, FiLM, masking, and encoder outputs framed as model behavior, not biological causality?
- [ ] Does `0.900705` appear only as the no-skill PR baseline, not a performance floor?
- [ ] Is `measured=0` never treated as validation/test ground truth?
- [ ] Are terms consistent with the canonical list?
- [ ] Are banned AI-filler patterns absent?
- [ ] Is the block within the text-density budget?
- [ ] Does the figure/caption imply no stronger claim than the text?
- [ ] Can the block be read quickly on a 70x100 cm printed poster?
