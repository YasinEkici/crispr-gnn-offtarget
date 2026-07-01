# Poster Microcopy Draft

This document is the Slice 4 Turkish microcopy draft for the CRISPR-Cas9 off-target prediction thesis poster. It collects short candidate headings, labels, captions, caveats, and section-level text fragments that can be used in a later low-fidelity poster draft.

This is not final poster copy, not a layout file, not a figure file, and not a production-tool decision. The final title, final wording, final palette, and final production tool remain deferred.

Planning language is English. Any phrase that may appear on the poster is quoted in Turkish.

## 1. Purpose and scope

The purpose is to give the designer a controlled Turkish phrase bank for:

- title-zone candidates;
- short section headings;
- figure labels;
- figure captions;
- result caveats;
- contribution and limits chips;
- visitor-layer one-liners;
- jury-layer precision text.

The microcopy must stay compatible with:

- `docs/poster/notes/poster_design_decisions.md`
- `docs/poster/notes/poster_narrative_framing.md`
- `docs/poster/notes/poster_yazim_kurallari.md`
- `docs/poster/plan/poster_content_plan.md`
- `docs/poster/plan/poster_section_shortlist.md`
- `docs/poster/plan/figure_production_plan.md`
- `docs/poster/plan/layout_draft_brief.md`

Use this file as a candidate pool, not as a script to paste wholesale into the poster.

## 2. Microcopy status legend

Use these labels when selecting text for the actual draft:

- **Exact required:** must be used exactly if that claim appears.
- **Candidate:** usable wording, still reviewable.
- **Deferred:** do not choose yet; keep as an option.
- **Avoid:** forbidden or risky wording.

The poster should use short, active Turkish. Avoid thesis-style paragraphs, generic importance claims, and apology framing.

## 3. Title zone candidates

Final title is deferred. The title must be contribution-first and must not frame the poster as a failure to beat XGBoost.

Candidate titles:

- `"Aynı dizi, farklı bağlam: CRISPR-Cas9 hedef dışı tahmininde bağlam-duyarlı çizge temsili"`
- `"Sıralama mı, karar eşiği mi? Hedef dışı tahminde iki ayrı soru"`
- `"Doğrulanmış veri, sızıntısız değerlendirme: bağlam-duyarlı GNN'lerin nadir negatif katkısı"`
- `"Aynı adres, farklı ziyaretler: Graph C ile hedef-gözlem bağlamı"`
- `"Bağlam nerede işe yarıyor? CRISPR-Cas9 hedef dışı tahmininde iki eksenli değerlendirme"`

Avoid:

- `"XGBoost'u geçemedik"`
- `"XGBoost'u geçtik"`
- `"En iyi hedef dışı tahmin modeli"`
- `"GNN her metrikte daha iyi"`

## 4. One-line thesis candidates

These are subtitle or takeaway candidates. Use one, not all.

- `"Graph C, aynı fiziksel hedefi farklı hedef-gözlem bağlamlarıyla temsil eder."`
- `"Bu çalışma bağlam katkısını AUPRC zaferi olarak değil, karar eşiği davranışı olarak konumlandırır."`
- `"Ölçülmüş veri evreninde bağlam, sıralamadan çok nadir negatif karar davranışını değiştirir."`
- `"AUPRC sıralamayı, MCC/specificity karar eşiğindeki negatif tanıma davranışını gösterir."`

Keep these short. If the subtitle needs a caveat to be safe, move the caveat into the results or limits area instead of overloading the title zone.

## 5. Section heading candidates

Preferred short headings:

- `"Problem ve Amaç"`
- `"Veri Sözleşmesi"`
- `"Graph A/B/C"`
- `"Mekanizma Zinciri"`
- `"Bulgular: İki Eksen"`
- `"Sıralama (AUPRC)"`
- `"Karar Eşiği"`
- `"Neden İki Eksen?"`
- `"Literatürde Nereye Oturuyor?"`
- `"Ne Katıyoruz?"`
- `"Sınırlar"`
- `"Kaynaklar ve Bilgi"`

Alternative compact headings:

- `"Hedef Dışı Risk"`
- `"Ölçülmüş Evren"`
- `"Hedef-Gözlem"`
- `"Nadir Negatifler"`
- `"Kıyas Değil, Konumlandırma"`
- `"Sonuç ve Sınır"`

## 6. Figure labels and captions

### Measured-only funnel

Labels:

- `"Tüm adaylar"`
- `"Ölçülmüş veri evreni"`
- `"Test evreni"`
- `"29 rehber"`
- `"1533 pozitif"`
- `"169 negatif"`

Caption candidates:

- `"Doğrulanmamış adaylar (measured=0) test etiketi yapılmadı."`
- `"310142 satırdan ölçülmüş veri evrenine, oradan 1702 satırlık test evrenine geçildi."`
- `"Test evreni rehber ayrık ve ölçülmüş veriyle sınırlıdır."`

Claim guard:

- Do not write that `measured=0` rows are false, safe, or negative.

### Graph A/B/C semantic comparison

Labels:

- `"Graph A: fiziksel hedef şeması"`
- `"Graph B: rehber benzerliği kontrolü"`
- `"Graph C: hedef-gözlem (target-observation) şeması"`
- `"Aynı adres, farklı ziyaretler"`

Caption candidates:

- `"Graph C, aynı fiziksel hedefi farklı hedef-gözlem bağlamlarıyla temsil eder."`
- `"Graph C katkısı salt topoloji değil, düğüm semantiği değişimidir."`
- `"Oklar temsil akışını gösterir; biyolojik nedensellik göstermez."`

Claim guard:

- Do not make graph arrows look like biological cause-effect.

### Two-axis result panel

Labels:

- `"Sıralama (AUPRC)"`
- `"Karar Eşiği"`
- `"AUPRC"`
- `"MCC"`
- `"specificity"`
- `"no-skill PR baseline: 0.900705"`

Caption candidates:

- `"AUPRC sıralama kalitesini ölçer."`
- `"MCC/specificity karar eşiğindeki negatif tanıma davranışını görünür kılar."`
- `"0.900705 no-skill PR baseline'dır; performans alt sınırı değildir."`

Exact required caveat, if the result band is shown:

> `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

Claim guard:

- The operating-point side must not read as an AUPRC win.

### TN/169 rare-negative recognition visual

Labels:

- `"Doğru negatif / 169"`
- `"XGBoost F4: 40/169"`
- `"Graph C GCN: 14/169"`
- `"Graph C GATv2: 63/169"`
- `"Aile-duyarlı encoder: 110/169"`
- `"specificity: 0.236, 0.083, 0.373, 0.651"`
- `"MCC: 0.603489"`

Caption candidates:

- `"Tüm değerler validation-kilitli karar eşiğinde 169 ölçülmüş negatif üzerinden verilir."`
- `"GATv2 ve aile-duyarlı encoder daha fazla nadir negatifi tanıdı; bu davranış seed/guide duyarlıdır."`
- `"Graph C GCN değeri düşük kalır; sonuç monoton bir graph kazanımı değildir."`

Claim guard:

- Keep the shared denominator visible.
- Keep seed/guide fragility visible.

### Literature A+B positioning panel

Labels:

- `"A: Neden doğrudan kıyaslanamaz?"`
- `"B: Hangi soru soruluyor?"`
- `"sızıntı kontrolü"`
- `"rehber ayrık değerlendirme"`
- `"prevalans farkındalığı"`
- `"ölçülmüş veri evreni"`
- `"sıralama / retrieval"`
- `"ölçülmüş ikili karar"`

Caption candidates:

- `"Veri, split, negatif üretimi ve prevalans farklı olduğu için ham skor kıyası yapılmaz."`
- `"Kıyas, skor yarışı değil; soru ve sözleşme karşılaştırmasıdır."`
- `"Bu poster literatürü nitel eksenlerde konumlandırır."`

Claim guard:

- Do not add raw scores from other papers.

### CRISPR for-dummies mini sketch

Labels:

- `"Rehber RNA"`
- `"Hedef DNA"`
- `"Benzer hedef dışı bölge"`
- `"kesim riski"`

Caption candidates:

- `"CRISPR-Cas9, hedef DNA'yı rehber RNA ile bulur; benzer bölgelerde hedef dışı kesim riski doğar."`
- `"Bu poster klinik güvenlik iddiası değil, ölçülmüş veri altında tahmin davranışı analizidir."`

Claim guard:

- Do not imply clinical validation.

### Model / mechanism chain

Labels:

- `"Veri sözleşmesi"`
- `"Graph C"`
- `"GATv2"`
- `"Aile-duyarlı encoder"`
- `"İki eksenli sonuç"`

Caption candidates:

- `"Bağlam sinyali hedef-gözlem düğümünde taşındı."`
- `"Aile-duyarlı encoder, özellik ailelerini ayrı işler."`
- `"Model içi açıklamalar biyolojik nedensellik kanıtı değildir."`

Claim guard:

- The chain is an experiment/model chain, not a biological pathway.

### Feature-family chip strip

Labels:

- `"6 deneysel epigenetik"`
- `"13 hesaplanmış nükleozom"`
- `"5 bağlanma enerjisi"`
- `"özellik ailesi"`

Caption candidates:

- `"Özellik aileleri model girdisidir; biyolojik nedensellik kanıtı değildir."`
- `"Bağlam, deneysel epigenetik, hesaplanmış nükleozom ve bağlanma enerjisi aileleriyle temsil edildi."`

Claim guard:

- Do not say a feature family biologically determines cleavage.

## 7. Section-by-section microcopy candidates

### Problem / aim

Candidate heading:

- `"Problem ve Amaç"`

Candidate body fragments:

- `"CRISPR-Cas9 hedef DNA'yı rehber RNA ile bulur; benzer bölgelerde hedef dışı kesim riski doğar."`
- `"Soru: bağlamı çizge içinde doğru temsil etmek, model davranışını nerede değiştirir?"`
- `"Bu poster en iyi model iddiası değil, bağlamın hangi eksende sinyal verdiğini gösterir."`

Use at most one problem sentence and one research-question sentence.

### Data contract / label discipline

Candidate heading:

- `"Veri Sözleşmesi"`

Candidate body fragments:

- `"Scheme A: cleavage_freq > 1e-5"`
- `"Ölçülmüş veri evreni (measured-only) test iddiasının sınırını belirler."`
- `"Doğrulanmamış adaylar (measured=0) test etiketi yapılmadı."`
- `"Model ve eşik seçimi validation ile kilitlendi; test üzerinde ayar yapılmadı."`

Use numbers only through the approved funnel and test-composition anchors.

### Graph representation

Candidate heading:

- `"Graph A/B/C"`

Candidate body fragments:

- `"Graph A fiziksel hedefi, Graph B rehber benzerliği kontrolünü, Graph C hedef-gözlem bağlamını taşır."`
- `"Aynı adres, farklı ziyaretler: aynı fiziksel hedef farklı bağlamlarda farklı gözlemler taşır."`
- `"Graph C katkısı salt topoloji değil, düğüm semantiği değişimidir."`

Pair the metaphor with the technical term `hedef-gözlem`.

### Model / mechanism chain

Candidate heading:

- `"Mekanizma Zinciri"`

Candidate body fragments:

- `"Bağlam sinyali hedef-gözlem düğümünde taşındı."`
- `"Aile-duyarlı encoder, özellik ailelerini ayrı işler."`
- `"Model içi açıklamalar biyolojik nedensellik kanıtı değildir."`

Keep this section compact. It supports the evidence chain, but the poster should not become an architecture diagram.

### Ranking axis

Candidate heading:

- `"Sıralama (AUPRC)"`

Candidate body fragments:

- `"AUPRC sıralama kalitesini ölçer."`
- `"XGBoost F4, AUPRC bakımından en sağlam referans olarak kaldı."`
- `"S8B_R2 yaklaştı; sağlam üstünlük iddiası kurulmadı."`
- `"Aralıklar fark yokluğu ile uyumlu; bu eşdeğerlik iddiası değildir."`
- `"0.900705 no-skill PR baseline'dır; performans alt sınırı değildir."`

Use compatibility wording, not equivalence wording.

### Operating-point axis

Candidate heading:

- `"Karar Eşiği"`

Candidate body fragments:

- `"Karar eşiği, modelin hangi örneğe pozitif veya negatif dediğini gösterir."`
- `"Bu test evreninde nadir sınıf negatiftir: 169 negatif."`
- `"Doğru negatif sayısı aynı paydada okunur: 40, 14, 63, 110 / 169."`
- `"MCC/specificity bu nadir negatif davranışını görünür kılar."`
- `"Karar eşiği sonuçları validation-kilitlidir ve seed/guide duyarlıdır."`

Do not say MCC is globally better than AUPRC. Say it answers the operating-point question.

### Why the axes diverge

Candidate heading:

- `"Neden İki Eksen?"`

Candidate body fragments:

- `"Sıralama bol sayıda pozitifle doyuma ulaşır; karar eşiği az sayıdaki negatife bakar."`
- `"Fark da, kırılganlık da buradan gelir."`
- `"169 negatif, 9 rehber içinde yoğunlaşır; 80'i guide 9251 üzerindedir."`
- `"AUPRC ve MCC aynı sonucu tekrarlamaz; farklı soruları görünür kılar."`

This explanation should sit close to the result band.

### Literature positioning A+B

Candidate heading:

- `"Literatürde Nereye Oturuyor?"`

Candidate body fragments:

- `"Neden doğrudan kıyaslanamaz?"`
- `"Veri, split, negatif üretimi ve prevalans farklı."`
- `"Bizim soru: ölçülmüş evrende bağlam, karar davranışını nerede değiştirir?"`
- `"Kıyas, skor yarışı değil; soru ve sözleşme karşılaştırmasıdır."`

Keep the literature panel qualitative.

### Contribution / takeaway

Candidate heading:

- `"Ne Katıyoruz?"`

Candidate chips:

- `"Hedef-gözlem düzeyinde Graph C temsili"`
- `"Scheme A ve ölçülmüş veri disiplini"`
- `"Rehber ayrık değerlendirme"`
- `"Nadir negatif karar davranışı"`
- `"Sınırı açık, dürüst sonuç çerçevesi"`

Candidate closing line:

- `"Bağlam katkısını AUPRC zaferi olarak değil, karar eşiği davranışı olarak konumlandırdık."`

### Limits / future work

Candidate heading:

- `"Sınırlar"`

Candidate bullets:

- `"AUPRC üstünlüğü iddia edilmedi."`
- `"Nadir negatif kazanım seed/guide duyarlıdır."`
- `"Daha fazla ölçülmüş negatif ve dış doğrulama gerekir."`
- `"Model içi açıklamalar biyolojik nedensellik kanıtı değildir."`

Keep this visible but compact.

### References and author info

Candidate heading:

- `"Kaynaklar ve Bilgi"`

Candidate footer text:

- `"Kasım Deliacı & Yasin Ekici"`
- `"Danışman: Doç. Dr. Mustafa Özgür Cingiz"`
- `"Bursa Teknik Üniversitesi, Bilgisayar Mühendisliği"`

References should be compact and selected later.

## 8. Required caveat bank

Use these as controlled one-liners. Do not expand them into long paragraphs.

Exact required result caveat:

> `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

Seed/guide fragility:

- `"Karar eşiği sonuçları validation-kilitlidir ve seed/guide duyarlıdır."`
- `"Nadir negatif kazanım 169 negatif ve rehber dağılımı ile sınırlıdır."`

No-causality:

- `"Model içi açıklamalar biyolojik nedensellik kanıtı değildir."`
- `"Oklar temsil akışını gösterir; biyolojik neden-sonuç göstermez."`

No-skill PR baseline:

- `"0.900705 no-skill PR baseline'dır; performans alt sınırı değildir."`

Measured-only:

- `"Doğrulanmamış adaylar (measured=0) test etiketi yapılmadı."`

Literature:

- `"Kıyas, skor yarışı değil; soru ve sözleşme karşılaştırmasıdır."`

Compatibility, not equivalence:

- `"Aralıklar fark yokluğu ile uyumlu; bu eşdeğerlik iddiası değildir."`

## 9. Avoid phrase bank

Do not use these poster phrasings:

- `"Günümüzde CRISPR önemli bir yer tutmaktadır."`
- `"Gelişen teknolojiyle birlikte"`
- `"yalnızca ... değil, aynı zamanda ..."`
- `"Ayrıca"`
- `"Bununla birlikte"`
- `"Dahası"`
- `"XGBoost'u geçtik."`
- `"XGBoost'u geçemedik."`
- `"GNN en iyi model oldu."`
- `"Graph C biyolojik mekanizmayı kanıtladı."`
- `"Attention biyolojik açıklama verdi."`
- `"Nadir negatif problemi çözüldü."`
- `"Sequence modeller başarısızdır."`
- `"MCC, AUPRC'den daha iyi metriktir."`
- `"Literatüre göre en iyi sonuç."`

Safer replacements:

- `"Bu ölçüm sözleşmesi altında ..."`
- `"Sıralama ekseninde ..."`
- `"Karar eşiğinde ..."`
- `"fark yokluğu ile uyumlu"`
- `"model davranışı"`
- `"validation-kilitli"`
- `"seed/guide duyarlı"`

## 10. Microcopy-to-layout mapping

| Layout zone | Use these microcopy groups |
| --- | --- |
| Zone A - Title / identity | Title candidates, one-line thesis, author footer text |
| Zone B - Hero Graph C | Graph C labels, Graph C metaphor, target-observation caption |
| Zone C - Data contract | Funnel labels, Scheme A, measured-only caveat |
| Zone D - Graph A/B/C | Graph A/B/C labels and semantic caption |
| Zone E - Results band | Two-axis labels, AUPRC text, exact honesty caveat |
| Zone F - Rare-negative recognition | TN/169 labels, operating-point caveat |
| Zone G - Axis divergence | Prevalence/no-skill baseline, positive-negative imbalance, fragility line |
| Zone H - Literature A+B | Qualitative comparison labels and no-leaderboard caveat |
| Zone I - Contribution / limits | Contribution chips and limits bullets |
| Zone J - Footer | References heading, authors, advisor, BTU identity |

If a zone feels crowded, keep labels and caveats first, then cut explanatory prose.

## 11. Numeric callout candidates

These callouts use only the approved anchor pool.

Data contract:

- `"310142 -> 25632 -> 1702"`
- `"29 rehber"`
- `"1533 pozitif"`
- `"169 negatif"`

Ranking:

- `"XGBoost F4 AUPRC 0.992338 [0.950179, 0.999336]"`
- `"S8B_R2 AUPRC 0.986020 [0.929981, 0.998966]"`
- `"no-skill PR baseline 0.900705"`

Operating point:

- `"40/169 doğru negatif"`
- `"14/169 doğru negatif"`
- `"63/169 doğru negatif"`
- `"110/169 doğru negatif"`
- `"specificity 0.236 / 0.083 / 0.373 / 0.651"`
- `"MCC 0.603489"`

Axis divergence:

- `"169 negatif, 9 rehber"`
- `"80 negatif guide 9251 üzerinde"`

Context features:

- `"6 deneysel epigenetik"`
- `"13 hesaplanmış nükleozom"`
- `"5 bağlanma enerjisi"`

Do not create new rounded deltas or percentages from these values unless a later audit approves them.

## 12. Review checklist before integration

Before using any phrase in a poster draft:

- [ ] Is the phrase short enough for a 70x100 cm visual-first poster?
- [ ] Is the phrase Turkish if it appears on the poster?
- [ ] Is any poster-bound Turkish text quoted in this planning file?
- [ ] Does it avoid AI-filler and decorative academic phrasing?
- [ ] Does it keep AUPRC and operating-point evidence separate?
- [ ] Does it avoid AUPRC superiority over XGBoost F4?
- [ ] Does MCC/specificity stay framed as operating-point evidence?
- [ ] Is every number from the approved anchor pool?
- [ ] Is `0.900705` labeled as no-skill PR baseline, not a floor?
- [ ] Is `measured=0` never treated as validation/test ground truth?
- [ ] Are operating-point gains tagged validation-locked and seed/guide sensitive?
- [ ] Are model-internal outputs framed as model behavior, not biological causality?
- [ ] Is the literature panel qualitative only?
- [ ] Is the exact honesty caveat present only inside results if used?
