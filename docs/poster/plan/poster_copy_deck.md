# Poster Copy Deck (Turkish source text)

This document is a reservoir of ready-to-use **Turkish** poster copy for the CRISPR-Cas9 off-target prediction thesis poster. The deliverable is the Turkish text itself; only the scaffolding, notes, and status tags are in English. A designer copies blocks from here and trims them to fit whatever layout is chosen later.

## 1. Purpose and how to use

- **Reservoir, not layout.** This is a source pool of full paragraphs. Any final poster is *trimmed* from these blocks. It is deliberately fuller than the low-density LaTeX fit-check.
- **No template binding.** The final poster format and tool are undecided and probably will not be the official BTÜ template. That template was seen only as a reference for expected text volume and to make sure the canonical academic section set is not forgotten.
- **Canonical headings as a checklist:** Özet, Giriş ve Amaç, Metot, Sonuçlar, Tartışma, Referanslar (plus figure/table captions and author identity). These are memory-aid section functions, not a fixed layout.
- **Reconciliation with the low-density plan.** `poster_content_plan.md` §10 and `poster_section_shortlist.md` §8 still govern the *final* poster's low text density. This deck does not override them; it feeds them. Write full here, cut later.
- **Independent of the fit-check.** This file does not reference or modify `drafts/latex_fit_check/`.

### Two governing corrections (baked into every block below)

1. **Impersonal-active register.** No first person ("biz yaptık/önerdik"). Use impersonal forms ("önerilir", "görülür", "konumlandırır"). This follows `tez_yazim_meta_kurallari.md` §1 while staying direct per `poster_yazim_kurallari.md` §2.
2. **True-negative lens, not "catching negatives with MCC".** The operating-point contribution is a *specificity / true-negative* behavior (more of the 169 measured negatives classified as true negatives). MCC and specificity are the metrics that *make this visible* — they are not the mechanism. Never phrase it as "MCC ile negatif yakalama".

## 2. Usage legend

Each copy block is tagged and given an approximate Turkish word count so it can be sized to any box.

- **hazır** — usable as-is if that claim appears.
- **kırpılabilir** — a shorter variant is provided; trim toward it.
- **opsiyonel-kes** — cut first if space is tight.

All numbers trace to the thesis anchor pool (§11). No number here is invented.

## 3. Başlık bölgesi (title / identity)

**Title candidates** (A is the selected working title). *(hazır)*

- A — `"CRISPR-CAS9 HEDEF DIŞI TAHMİNİNDE BAĞLAM DUYARLI ÇİZGE SİNİR AĞLARININ DEĞERLENDİRİLMESİ"`
- B — `"Sıralama mı, karar eşiği mi? Hedef dışı tahminde iki ayrı soru"`
- C — `"Doğrulanmış veri, sızıntısız değerlendirme: bağlam-duyarlı GNN'lerin nadir negatif katkısı"`
- D — `"Aynı adres, farklı ziyaretler: Graph C ile hedef-gözlem bağlamı"`
- E — `"Bağlam nerede işe yarıyor? CRISPR-Cas9 hedef dışı tahmininde iki eksenli değerlendirme"`

**Subtitle / one-line thesis candidates** (use one). *(hazır)*

- `"Bağlam katkısı bir AUPRC zaferi değil, karar eşiği davranışıdır."`
- `"AUPRC sıralamayı, karar eşiği ise nadir negatif tanıma davranışını gösterir."`

**Author / advisor identity** (fill placeholders). *(hazır)*

> Kasım Deliacı & Yasin Ekici
> Öğrenci No: [xxxxx] · E-posta: [xxx@yyy.com]
> Danışman: Doç. Dr. Mustafa Özgür Cingiz · E-posta: [xxx@yyy.com]
> Bursa Teknik Üniversitesi, Bilgisayar Mühendisliği Bölümü
> BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ BİTİRME PROJESİ SERGİSİ – Temmuz 2026

## 4. Özet

**Full (~145 kelime).** *(hazır)*

> CRISPR-Cas9 gen düzenlemesinde hedef dışı kesim, güvenli rehber tasarımını doğrudan kısıtlar. Bu çalışma, hedef dışı aktiviteyi tahmin etmek için bağlam duyarlı çizge sinir ağlarını; sızıntıya karşı denetlenmiş, rehber ayrık (guide-disjoint) ve yalnızca ölçülmüş veri evreni (measured-only) üzerine kurulmuş bir değerlendirme sözleşmesi altında inceler. İkili etiket Scheme A ile tanımlanır (cleavage_freq > 1e-5); doğrulanmamış adaylar (measured=0) test doğrusu yapılmaz. Aynı fiziksel hedefi farklı hedef-gözlem (target-observation) bağlamlarıyla temsil eden özgün bir çizge şeması (Graph C) ve özellik ailelerini yapısal olarak işleyen aile-duyarlı encoder önerilir. Sıralama ekseninde (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kalır. Bağlamın katkısı ise karar eşiğinde (operating point), nadir ölçülmüş negatifleri daha yüksek oranda doğru negatif olarak tanımada görülür; MCC ve specificity bu davranışı görünür kılan metriklerdir. Bu kazanım seed/guide duyarlıdır ve bir üstünlük iddiası olarak sunulmaz. Katkı, bağlamın hangi eksende sinyal verdiğini dürüst ve sınırları açık bir çerçevede konumlandırmaktır.

**Kısa (~80 kelime).** *(kırpılabilir)*

> CRISPR-Cas9 hedef dışı tahmininde bağlam duyarlı çizge sinir ağları; sızıntıya karşı denetlenmiş, rehber ayrık ve yalnızca ölçülmüş veri evreni üzerine kurulmuş bir sözleşme altında incelenir. Hedef-gözlem bağlamını taşıyan özgün bir çizge şeması (Graph C) ve aile-duyarlı encoder önerilir. Sıralamada (AUPRC) XGBoost F4 en sağlam çıta olarak kalır; bağlamın katkısı karar eşiğinde nadir ölçülmüş negatifleri daha çok doğru negatif olarak tanımada görülür. MCC ve specificity bu davranışı görünür kılar; kazanım seed/guide duyarlıdır, üstünlük iddiası değildir.

**Anahtar kelimeler:** CRISPR-Cas9, hedef dışı tahmin, çizge sinir ağları, hedef-gözlem bağlamı, ölçülmüş veri, rehber ayrık değerlendirme.

## 5. Giriş ve Amaç

**Full (3 paragraf, ~120 kelime).** *(hazır)*

> CRISPR-Cas9, hedef DNA'yı rehber RNA (sgRNA) ile eşleştirerek keser; ancak diziye benzeyen istem dışı bölgelerde de kesim, yani hedef dışı bölge (off-target site) aktivitesi gerçekleşebilir. Bu riski önceden kestirmek, güvenli rehber tasarımının önkoşuludur.
>
> Literatürdeki birçok yaklaşım dizi benzerliğine odaklanır. Oysa aynı fiziksel hedef, farklı deneysel ve epigenetik bağlamlarda farklı gözlemler taşıyabilir. Bu bağlamın çizge içinde nasıl temsil edildiği, sızıntıya karşı denetlenmiş ve ölçülmüş bir değerlendirme altında yeterince incelenmemiştir.
>
> Amaç, bağlamın çizge temsilinin karar davranışını hangi eksende değiştirdiğini rehber ayrık (guide-disjoint) ve yalnızca ölçülmüş veri evreni (measured-only) üzerine kurulmuş bir sözleşme altında sınamaktır. Bu çalışma bir "en iyi model" iddiası değil; bağlamın nerede sinyal verdiğinin analizidir.

**Kısa (~55 kelime).** *(kırpılabilir)*

> CRISPR-Cas9 hedef dışı kesim riski, güvenli rehber tasarımının önündeki temel sorundur. Dizi benzerliğine dayalı yaklaşımlar, aynı hedefin farklı bağlamlarda farklı gözlemler taşıyabildiğini çoğu kez göz ardı eder. Bu çalışma, bağlamın çizge temsilinin karar davranışını hangi eksende değiştirdiğini sızıntıya karşı denetlenmiş ve ölçülmüş bir değerlendirme altında sınar.

## 6. Metot

**Full (4 alt paragraf, ~200 kelime).** *(hazır)*

> **Veri sözleşmesi.** İkili etiket Scheme A ile tanımlanır: cleavage_freq > 1e-5. Test ve doğrulama iddiaları yalnızca ölçülmüş veri evreni (measured-only) üzerine kurulur; doğrulanmamış adaylar (measured=0) test doğrusu olarak kullanılmaz. 310142 aday satırdan ölçülmüş 25632 satıra, oradan rehber ayrık 1702 satırlık test evrenine inilir. Test evreni 29 rehber, 1533 pozitif ve 169 negatiften oluşur. Model ve karar eşiği seçimi yalnızca doğrulama kümesiyle kilitlenir; test üzerinde ayar yapılmaz.
>
> **Çizge temsili.** Üç şema karşılaştırılır. Graph A fiziksel hedefi tek düğümle temsil eder. Graph B rehber benzerliğini ayrı bir kontrol sinyali olarak taşır. Graph C ise her satırı bir hedef-gözlem (target-observation) düğümü olarak ele alır: aynı fiziksel hedef, farklı bağlamlarda farklı gözlemler taşır. Graph C'nin katkısı salt topoloji değil, düğüm semantiğinin değişmesidir.
>
> **Model.** Bağlam sinyali hedef-gözlem düğümünde taşınır ve GATv2 tabanlı dikkat mekanizmasıyla işlenir. Aile-duyarlı encoder, özellik ailelerini düz bir vektör yerine yapısal olarak ayrı işler: 6 deneysel epigenetik, 13 hesaplanmış nükleozom ve 5 bağlanma enerjisi özelliği. Kazanım deneysel-epigenetik bağlama yerelleştirilerek mekanizma izole edilir.
>
> **Değerlendirme.** Sıralama kalitesi AUPRC ile, karar eşiği (operating point) davranışı MCC ve specificity (özgüllük) ile ölçülür. Referans olarak güçlü tablo temelli XGBoost F4 aynı sözleşme altında kullanılır. Belirsizlik, bootstrap ve çok-tohum (multi-seed) çözümlemesiyle sınırlanır. Model içi çıktılar (dikkat, maskeleme, FiLM) model davranışını açıklar; biyolojik nedensellik kanıtı değildir.

**Kısa (~70 kelime).** *(kırpılabilir)*

> İkili etiket Scheme A ile tanımlanır (cleavage_freq > 1e-5). Değerlendirme yalnızca ölçülmüş veri evreninde, rehber ayrık bölünmeyle ve test üzerinde ayar yapılmadan yürütülür (310142 → 25632 → 1702; 29 rehber, 1533 pozitif, 169 negatif). Graph A/B/C şemaları karşılaştırılır; Graph C bağlamı hedef-gözlem düğümünde taşır. GATv2 ve aile-duyarlı encoder ile modellenen bağlam, AUPRC ve karar eşiği (MCC/specificity) üzerinden değerlendirilir; referans XGBoost F4'tür.

## 7. Sonuçlar

**Full (~160 kelime).** *(hazır)*

> Sonuçlar iki ayrı soruyu yanıtlar. Sıralama ekseninde AUPRC ölçülür: güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kalır (AUPRC 0.992338; %95 aralık [0.950179, 0.999336]). En iyi tekil GNN olan S8B_R2 yaklaşır (0.986020; [0.929981, 0.998966]); aralıklar fark yokluğu ile uyumludur ve bu bir eşdeğerlik iddiası değildir.
>
> Karar eşiği ekseninde nadir sınıf negatiftir (169 negatif). Doğru negatif sayısı aynı payda üzerinden okunur: XGBoost F4 40, Graph C GCN 14, Graph C GATv2 63, aile-duyarlı encoder 110 / 169 (specificity sırasıyla 0.236, 0.083, 0.373, 0.651; aile-duyarlı encoder MCC 0.603489). Bağlam duyarlı GATv2 ve aile-duyarlı encoder, nadir ölçülmüş negatifleri daha yüksek oranda doğru negatif olarak tanır; MCC ve specificity bu davranışı görünür kılan metriklerdir. Graph C GCN düşük kalır, dolayısıyla sonuç monoton bir çizge kazanımı değildir ve seed/guide duyarlıdır.

**Honesty caveat** (exactly once, inside results). *(hazır — zorunlu)*

> Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor.

**Kısa (~65 kelime).** *(kırpılabilir)*

> Sıralamada (AUPRC) XGBoost F4 en sağlam çıta olarak kalır (0.992338); en iyi tekil GNN S8B_R2 yaklaşır (0.986020) ama üstünlük göstermez. Karar eşiğinde doğru negatifler 40, 14, 63, 110 / 169 olarak okunur; bağlam duyarlı GATv2 ve aile-duyarlı encoder nadir negatifleri daha çok doğru negatif olarak tanır (MCC/specificity ile görünür). Kazanım seed/guide duyarlıdır.

## 8. Tartışma

**Full (4 alt paragraf, ~200 kelime).** *(hazır)*

> **Neden iki eksen?** AUPRC ve MCC/specificity farklı soruları yanıtlar. Sıralama, bol sayıda pozitifle (prevalans 0.900705) doyuma ulaşır ve güçlü modeller yüksek AUPRC'de kümelenir; bu nedenle sıralama ekseninde ayrışma sınırlıdır. Karar eşiği ise az sayıdaki negatife bakar. 169 negatif 9 rehberde yoğunlaşır, 80'i tek bir rehberdedir (guide 9251). Farkın da, kırılganlığın da kaynağı bu kıtlıktır. 0.900705 no-skill PR baseline'dır; performans alt sınırı değildir.
>
> **Literatürde konumlandırma.** Diğer çalışmalarla ham skor kıyası yapılmaz; veri, bölünme, negatif üretimi ve prevalans farklıdır. Konumlandırma niteldir: bu çalışma sızıntı kontrolü, rehber ayrık değerlendirme, prevalans farkındalığı ve ölçülmüş veri evreni eksenlerinde daha katıdır. Literatürdeki birçok çalışma sıralama/retrieval sorusunu ele alır; bu çalışma ise ölçülmüş evrende ikili karar sorusunu ele alır. Kıyas, skor yarışı değil; soru ve sözleşme karşılaştırmasıdır.
>
> **Katkı.** Katkı, bağlamı hedef-gözlem düzeyinde temsil eden ve putatif adayları test doğrusu yapmayan, sızıntıya karşı denetlenmiş bir değerlendirme çerçevesidir. Bağlamın katkısı bir AUPRC zaferi olarak değil, karar eşiğinde nadir negatif davranışı olarak konumlandırılır.
>
> **Sınırlar ve sonraki adım.** AUPRC üstünlüğü iddia edilmez. Karar eşiği kazanımı validation-kilitli, seed/guide duyarlı ve 169 negatifle sınırlıdır. Daha fazla ölçülmüş negatif ve dış doğrulama gerekir. Model içi açıklamalar biyolojik nedensellik kanıtı değildir.

**Kısa (~75 kelime).** *(kırpılabilir)*

> Sıralama bol sayıda pozitifle (prevalans 0.900705) doyuma ulaşır; karar eşiği ise az sayıdaki negatife bakar (169 negatif, 9 rehber; 80'i guide 9251). Farkın da, kırılganlığın da kaynağı bu kıtlıktır. Diğer çalışmalarla ham skor kıyası yapılmaz; konumlandırma nitel ve sözleşme temellidir. Bağlamın katkısı, karar eşiğinde nadir negatif davranışı olarak konumlandırılır; kazanım seed/guide duyarlıdır ve biyolojik nedensellik iddiası taşımaz.

**Sınırlar (madde madde).** *(hazır)*

> - AUPRC üstünlüğü iddia edilmedi.
> - Nadir negatif kazanım validation-kilitli ve seed/guide duyarlıdır.
> - Daha fazla ölçülmüş negatif ve dış doğrulama gerekir.
> - Model içi açıklamalar biyolojik nedensellik kanıtı değildir.

**Katkı çipleri (kısa etiketler).** *(hazır)*

> - Graph C hedef-gözlem temsili
> - Scheme A · ölçülmüş veri disiplini
> - Rehber ayrık değerlendirme
> - Nadir negatif karar davranışı
> - Sınırı açık, dürüst değerlendirme çerçevesi

## 9. Referanslar

Poster uses a numbered list (matching the official poster format). Note: the *thesis body* uses APA author-date (`tez_yazim_meta_kurallari.md` §3); the poster's numbered style is a poster-only convention. Keep 6-8 essential anchors. *(kırpılabilir — trim count if tight)*

> [1] Kipf, T. N., Welling, M. Semi-Supervised Classification with Graph Convolutional Networks. 2017.
> [2] Vinodkumar vd. Prediction of sgRNA Off-Target Activity in CRISPR/Cas9 Gene Editing Using Graph Neural Networks. 2021.
> [3] Liu vd. Graph-CRISPR: A Gene Editing Efficiency Prediction Model Based on Graph Neural Networks. 2025.
> [4] Chuai vd. DeepCRISPR: Optimized CRISPR Guide RNA Design by Deep Learning. 2018.
> [5] Lin vd. CRISPR-Net: A Recurrent Convolutional Network Quantifies CRISPR Off-Target Activities. 2020.
> [6] Gao vd. Data Imbalance in CRISPR Off-Target Prediction. 2020.
> [7] Guan vd. A Systematic Method for Solving Data Imbalance in CRISPR Off-Target Prediction. 2024.

## 10. Şekil ve tablo yazıları

Turkish captions for the existing poster assets. Captions end with a period (`tez_yazim_meta_kurallari.md` §6). *(hazır)*

> **Şekil 1.** Ölçülmüş veri evreni hunisi: 310142 aday satırdan ölçülmüş 25632 satıra ve rehber ayrık 1702 satırlık test evrenine. Doğrulanmamış adaylar (measured=0) test doğrusu yapılmadı.
>
> **Şekil 2.** Graph A/B/C semantik karşılaştırması. Graph C, aynı fiziksel hedefi farklı hedef-gözlem bağlamlarıyla temsil eder. Oklar temsil akışını gösterir; biyolojik nedensellik göstermez.
>
> **Şekil 3.** İki eksenli bulgular: sıralama (AUPRC) ve karar eşiği (MCC/specificity). AUPRC birincil sıralama metriğidir; 0.900705 no-skill PR baseline'dır.
>
> **Şekil 4.** Karar eşiğinde nadir negatif tanıma: 169 ölçülmüş negatif üzerinden doğru negatif sayıları. Sonuçlar validation-kilitli ve seed/guide duyarlıdır.
>
> **Şekil 5.** Literatürde nitel konumlandırma (A+B). Ham skor leaderboard kurulmaz; kıyas soru ve sözleşme karşılaştırmasıdır.

**Optional table** *(opsiyonel-kes)*

> **Tablo 1.** Karar eşiğinde doğru negatif ve specificity değerleri (169 ölçülmüş negatif üzerinden): XGBoost F4 40 (0.236), Graph C GCN 14 (0.083), Graph C GATv2 63 (0.373), aile-duyarlı encoder 110 (0.651).

## 11. Sayısal çıpa eki (approved anchor pool)

Every number in this deck traces to one of these thesis-verified values. Do not add others without reading them from the thesis.

- Universe / split: 310142 → 25632 → 1702; test 29 rehber, 1533 pozitif, 169 negatif; negatifler 9 rehberde, 80'i guide 9251'de.
- Prevalence: 0.900705 (no-skill PR baseline, not a floor).
- Ranking: XGBoost F4 AUPRC 0.992338, aralık [0.950179, 0.999336], çok-tohum 0.990649 ± 0.001944; tarihsel 0.992522 (version-drift dipnotu, opsiyonel).
- Best single GNN: S8B_R2 AUPRC 0.986020, aralık [0.929981, 0.998966], çok-tohum 0.978963 ± 0.011322.
- Operating point (169 negatif üzerinden doğru negatif): XGBoost F4 40, Graph C GCN 14, Graph C GATv2 63, aile-duyarlı encoder 110; specificity 0.236, 0.083, 0.373, 0.651; aile-duyarlı encoder MCC 0.603489.
- Feature families: 6 deneysel epigenetik + 13 hesaplanmış nükleozom + 5 bağlanma enerjisi.
- Scale (opsiyonel): 154 sgRNA, 138747 hedef lokasyonu.

## 12. Kırpma sırası (cut order if space is tight)

Trim from the bottom of this list first; protect the top.

**Kesilmez (protect):**
- Ölçülmüş veri evreni huni sayıları (310142 → 25632 → 1702).
- İki eksen ayrımı (sıralama vs karar eşiği).
- Honesty caveat (Sonuçlar içinde, bir kez).
- Claim caveat'ları: 0.900705 no-skill baseline; measured=0 test doğrusu değil; seed/guide duyarlı; biyolojik nedensellik yok.

**Önce kırp (trim first):**
1. Tarihsel XGBoost F4 0.992522 version-drift notu.
2. Çok-tohum ± değerleri.
3. Feature-family sayı dökümü (6/13/5), gerekmiyorsa.
4. Literatür A+B nitel eksen ayrıntısı.
5. Referans sayısı (8 → 6).
6. Metot ve Tartışma paragraflarını "kısa" varyanta indir.
