# Ana Tez Anlatısı ve Konumlandırma

Bu dosya tez yazımı sırasında ana hikayeyi, iddia sınırlarını, literatür konumlandırmasını ve Sprint 1-9 deney zincirini aynı yerde tutmak için hazırlanmıştır. Tez Türkçe yazılacaktır; başlık önerileri İngilizce bırakılabilir, fakat ana metinde kavramlar ilk geçtiği yerde Türkçe açıklanmalıdır.

## Çalışma Başlığı

Önerilen başlık:

> Context-Aware Graph Neural Networks for CRISPR-Cas9 Off-Target Prediction: Ranking Performance and Rare Negative Recognition

Türkçe tez içi karşılığı:

> CRISPR-Cas9 off-target tahmininde bağlam duyarlı çizge sinir ağları: sıralama başarımı ve nadir negatif sınıf tanıma

Bu başlık iki ana ekseni birlikte taşır: literatürle uyumlu sıralama metriği olan AUPRC ve bu çalışmanın en güçlü özgün bulgularından biri olan nadir negatif sınıfı validation-kilitli karar eşiğinde ayırabilme.

## Tek Cümlelik Tez

Bu çalışmanın en savunulabilir ana tezi şudur:

> CRISPR-Cas9 off-target tahmininde sequence/energy bilgisi tek başına yeterli değildir; hedef bölgenin epigenetik ve kromatin bağlamı Graph C benzeri target-observation temsili ve aile-duyarlı GATv2 mimarisi içinde doğru temsil edildiğinde nadir ölçülmüş negatifleri ayırma davranışı belirgin biçimde güçlenir, ancak bu kazanım XGBoost F4'e karşı robust bir AUPRC üstünlüğü olarak sunulmamalıdır.

Bu nedenle tez "GNN bütün baseline'ları geçti" hikayesi değildir. Daha güçlü hikaye, context-aware GNN çizgisinin nerede işe yaradığını, nerede yaramadığını ve neden AUPRC ile operating-point metriklerinin ayrı okunması gerektiğini göstermesidir.

## Çalışmanın Ne Olmadığı

- Bu çalışma Mak et al. 2022'nin birebir reprodüksiyonu değildir. Mak çalışması CA dönüşümü, korelasyon/SHAP analizi ve sürekli aktivite hedefi etrafında durur; bu proje Scheme A ikili sınıflandırma, guide-disjoint split ve measured-only test evreni kullanır.
- Bu çalışma "state-of-the-art off-target predictor" iddiası kurmamalıdır. XGBoost F4 primary AUPRC bar olarak kalır.
- Attention, gate, FiLM veya feature importance çıktıları biyolojik nedensellik kanıtı değildir. Bunlar model-interpretation ve hipotez üretme sinyalidir.
- Sprint 9'da paired AUPRC aralıkları sıfırı kestiği için GNN modellerin XGBoost F4'e veya birbirlerine karşı robust AUPRC üstünlüğü iddia edilmemelidir.

## Dondurulmuş Değerlendirme Sözleşmesi

Tez boyunca her ana sonuç bu kontratla birlikte raporlanmalıdır:

- Veri: Mak et al. 2022 / crisprSQL türevi epigenetik-nükleozom veri seti.
- Ana etiket: Scheme A, `cleavage_freq > 1e-5`.
- NaN `cleavage_freq` satırları supervised label üretiminden çıkarılır.
- Ana split: `sprint2_main_seed42`, guide-disjoint.
- Ana test evreni: measured-only, `experiment_id = 18` dışarıda.
- Test seti: 1702 satır, 29 guide, 1533 pozitif, 169 negatif, pozitif prevalans 0.900705.
- Preprocessing: train-only imputation/scaling.
- Model seçimi: validation-only checkpoint, validation-only threshold.
- Test tuning yoktur.
- Primary metric: AUPRC.
- Secondary/core operating-point metrics: AUROC, MCC, macro-F1, specificity, TN/FP/FN/TP.
- Non-GNN authoritative bar: `xgboost_unweighted / F4`.

AUPRC için 0.900705 değeri "floor" değil, measured-only test evrenindeki no-skill PR baseline'dır.

## Ana Bulguların Kısa Özeti

Ranking ekseni:

- XGBoost F4 en güçlü ve en stabil primary-AUPRC bar olarak kaldı.
- En iyi single-seed GNN AUPRC sonucu Sprint 8B `S8B_R2_sequence_plus_context` ile 0.986020 oldu; bu güçlü ve rekabetçi bir sonuçtur, fakat F4'ün 0.992522 historical AUPRC değerini geçmedi.
- Sprint 9'da paired AUPRC farklarının hiçbiri sıfırı dışlamadı; bu yüzden AUPRC üstünlüğü iddiası kurulamaz.

Operating-point / rare-negative ekseni:

- Context-aware Graph C GATv2 çizgisi, özellikle Sprint 7F/8A/8B modelleri, validation-kilitli threshold altında F4'e göre daha fazla nadir ölçülmüş negatifi yakaladı.
- Sprint 7F `S7F_R2_family_aware_context_encoder`: AUPRC 0.982062, MCC 0.603489, macro-F1 0.801716, specificity 0.650888, TN 110.
- Sprint 8B `S8B_R2_sequence_plus_context`: AUPRC 0.986020, MCC 0.567309, macro-F1 0.760058, specificity 0.863905, TN 146.
- Sprint 9 paired operating-point analizinde F4'e karşı bazı specificity/MCC/macro-F1 farkları sıfırı dışladı; bu, threshold'a bağlı rare-negative recognition katkısını destekler.
- Bu operating-point katkı seed-fragile ve guide-composition-fragile olarak raporlanmalıdır; AUPRC sıralamasının yerine geçmez.

## Sprint 1-9 Kanıt Zinciri

### Sprint 1: Veri Denetimi ve Etiket Politikası

Sprint 1 veri kaynağını, label politikasını ve leakage risklerini sabitledi. Kullanılan Parquet dosyasında 310142 satır, 45 kolon, 25632 measured=1 satırı, 284510 measured=0 satırı, 154 sgRNA ve 138747 target location doğrulandı. Veri 6 experimental epigenetic scalar, 13 computed nucleosome array feature ve 5 binding-energy feature içeriyor.

En önemli karar Scheme A'nın ana ikili etiket olmasıdır: `cleavage_freq > 1e-5`. Bu eşik Mak et al. 2022'deki assay-accuracy boundary ile uyumludur. CA dönüşümü veri içinde hazır olmadığı için Mak-style regression/paper-comparison track ertelendi. Measured=0 satırları validation/test ground truth olamaz; yalnızca ayrıca caveat verilmiş training-only noisy negative senaryolarında kullanılabilir.

### Sprint 2: Tabular ve Sequence Baselines

Sprint 2 guide-disjoint `sprint2_main_seed42` split'ini ve baseline evrenini kilitledi. Test seti measured-only ve pozitif ağırlıklıdır: 1533 pozitif, 169 negatif, prevalans 0.900705. Bu yapı AUPRC'nin yüksek görünmesini kolaylaştırır fakat negatif sınıfı ayırmayı zorlaştırır.

XGBoost F4 authoritative non-GNN bar oldu: test AUPRC 0.992522, AUROC 0.938416, MCC 0.345198, TN/FP/FN/TP = 38/131/21/1512. F4 feature seti sequence/mismatch, binding energy, 6 experimental epigenetic scalar ve aggregated computed nucleosome + missingness feature'larını içerir. Pure sequence CNN/BiLSTM baselineları daha zayıf kaldı; bu durum güçlü engineered/context tabular feature'ların önemini gösterdi.

### Sprint 3: Graph Artifact ve Leakage Kontrolü

Sprint 3 model eğitmedi; graph artifact'larını ve strict-inductive visibility politikasını üretti. Graph A minimal physical target graph, Graph B guide-similarity bounded control, Graph C ise target-observation context graph olarak tanımlandı.

Graph C kritik karardır: aynı fiziksel target koordinatı farklı observation/context değerleri taşıyabildiği için context'i shared physical node'a zorlamak yerine row-level `target_observation` node'ları kullanıldı. Bu, epigenetic/chromatin context'in hedef gözlem seviyesinde temsil edilmesini sağlar.

### Sprint 4: GCN Baseline

Sprint 4 frozen Sprint 2/3 kontratı altında ilk GCN karşılaştırmasını yaptı. Graph A GCN AUPRC 0.966287, MCC 0.300781; Graph B GCN AUPRC 0.966570, MCC 0.126559; Graph C GCN AUPRC 0.961586, MCC 0.453738 verdi.

Bu sprintin sonucu iki yönlüdür. Birincisi, hiçbir GCN XGBoost F4 AUPRC barını geçmedi. İkincisi, Graph C'nin MCC tarafında daha iyi davranması context/target-observation temsilinin operating-point tarafında potansiyel taşıdığını gösterdi, fakat Graph C tek başına topology-only veya AUPRC kazanımı olarak yorumlanamaz.

### Sprint 5 ve Sprint 5B: Feature-Family Ablation

Sprint 5, Graph A üzerinde feature-family ablation yaptı. En güçlü Graph A GCN sinyali `S5F2_energy` oldu: AUPRC 0.976585, AUROC 0.817765, macro-F1 0.695284, MCC 0.477933, specificity 0.284024, TN/FP/FN/TP = 48/121/6/1527.

Bu sonuç binding-energy feature'larının GCN için güçlü ve stabil bir candidate-edge sinyali taşıdığını gösterdi. Buna karşılık Graph A edge-only kullanımında experimental epigenetic scalars ve computed nucleosome grupları beklenen kadar güçlü çalışmadı; bazı computed feature setleri threshold collapse gösterdi. Bu, epigenetik sinyal yok anlamına gelmez; Graph A formulation içinde doğru yerde tüketilmediğini düşündürür.

Sprint 5B, Graph C + S5F2_energy duyarlılığını kontrol etti: AUPRC 0.972481, MCC 0.274287, specificity 0.082840. Ranking Graph C Sprint 4'e göre iyileşti, fakat operating point zayıftı. Bu bulgu Graph C context temsilinin daha iyi mimariyle test edilmesi gerektiğini gösterdi.

### Sprint 6: Imbalance / Loss Comparison

Sprint 6, fixed Graph A + S5F2_energy üzerinde loss/sampling karşılaştırmasını tamamladı. En iyi headline run weighted BCE oldu: AUPRC 0.976935, AUROC 0.819972, macro-F1 0.698939, MCC 0.483719, specificity 0.289941, TN/FP/FN/TP = 49/120/6/1527.

Unweighted BCE, focal varyantları, generalized Dice, Tversky ve measured-only balanced sampling weighted BCE'yi primary AUPRC'de geçmedi. Generalized Dice no-skill baseline'ın altına indi ve TN=0 threshold collapse gösterdi. Bu yüzden kalan sorun yalnızca loss seçimi değildir. Graph A GCN'de S5F2 edge feature'ları message passing içinde değil, edge-classifier head içinde concatenate edilmektedir; bu da Sprint 7'nin edge-aware attention/message passing motivasyonunu doğurdu.

### Sprint 7: Graph A GAT/GATv2

Sprint 7, aynı Graph A + S5F2_energy + weighted BCE kontratı altında GCN, GAT ve GATv2'yi karşılaştırdı. GAT AUPRC 0.950763 ve MCC -0.016850 ile geriledi. GATv2 AUPRC 0.965449, MCC 0.291367 verdi. Graph A reference GCN ise 0.976935 AUPRC ve 0.483719 MCC ile daha iyi kaldı.

Bu sonuç çok önemlidir: sadece GCNConv'u GAT/GATv2 ile değiştirmek problemi çözmedi. Edge-aware attention doğru implemente edilmiş olsa bile Graph A'nın target/context temsili sınırlı kaldı. Sprint 7'nin negatif sonucu Sprint 7B/7D/7E yönünü bilimsel olarak güçlendirdi.

### Sprint 7B: Graph B/C Topology ve GATv2

Sprint 7B, GATv2'nin Graph B ve Graph C üzerinde ne yaptığını test etti. Graph B GATv2 AUPRC 0.979139 ile Graph B GCN'den daha yüksek ranking verdi, fakat MCC 0.272970 ve specificity 0.094675 ile negatif tanıma zayıf kaldı.

Graph C GATv2 ise AUPRC 0.969078 ile AUPRC lideri olmadı ama MCC 0.531774, macro-F1 0.739526, specificity 0.372781 ve TN=63 ile o ana kadarki en güçlü GNN negative-class profile'ını verdi. Bu, asıl sinyalin Graph B guide-similarity topology'sinden değil Graph C target-observation context temsilinden gelebileceğini gösterdi.

### Sprint 7C: Explanation Diagnostics

Sprint 7C training yapmadan Graph C GATv2 davranışını açıklamaya dönük diagnostikler üretti. Bu sprintin rolü model geliştirme değil, attention/score/guide-level patternlerin nasıl yorumlanması gerektiğini sınırlandırmaktı. Sonuçlar attention ve benzeri çıktıları causal biology yerine interpretation-only artifact olarak ele alma kararını güçlendirdi.

### Sprint 7D: Graph C Mechanism Ablation

Sprint 7D Graph C GATv2 kazancının hangi bileşenden geldiğini izole etti. Full Graph C GATv2 AUPRC 0.969078, MCC 0.531774, specificity 0.372781 idi.

- `no_context_edges`: AUPRC 0.965598, MCC 0.517970, specificity 0.366864. Full modele çok yakın kaldı.
- `edge_blind_attention`: AUPRC 0.945691, MCC 0.182915. Candidate-edge S5F2_energy'nin attention/message passing içinde tüketilmesi önemlidir.
- `mask_target_context_features`: AUPRC 0.893657, MCC -0.013952, specificity 0.0. Target-observation context feature'ları kaldırılınca model çöktü.

Bu sprint ana mekanizmayı netleştirdi: Graph C'nin gücü büyük ölçüde explicit `context_similar_to` edge'lerinden değil, target-observation context feature'larının GATv2 mimarisi içinde kullanılması ve edge-aware attention ile birleşmesinden geliyor.

### Sprint 7E: Target-Context Subgroup Ablation

Sprint 7E, Graph C target-observation context'in hangi alt ailelerden beslendiğini araştırdı. Masking sonuçları:

- Target sequence one-hot mask: AUPRC 0.970151, MCC 0.489597. Sequence context tek başına ana taşıyıcı değil.
- Experimental epigenetic mask: AUPRC 0.885321, MCC -0.011388, specificity 0.0. Model çöküyor.
- Computed nucleosome aggregate mask: AUPRC 0.955915, MCC 0.511808. Katkı var fakat ana taşıyıcı değil.
- Computed missingness mask: AUPRC 0.949947, MCC 0.447562. Missingness/context coverage da davranışı etkiliyor.
- All nonsequence context mask: AUPRC 0.890660, MCC -0.011388, specificity 0.0.

En güçlü mekanizma bulgusu budur: Graph C GATv2'nin rare-negative behavior'ını özellikle experimental epigenetic target-observation feature'ları taşır. Bu biyolojik olarak plausible bir sinyaldir, fakat cell-line/source/coverage confound olasılığı nedeniyle causal claim'e çevrilmemelidir.

### Sprint 7F: Family-Aware Target Context Encoder

Sprint 7F, Sprint 7D/7E mekanizma bulgusunu model iyileştirmeye çevirdi. Unified deep encoder daha fazla kapasiteye rağmen güçlü çıkmadı: AUPRC 0.969201, MCC 0.500460.

Family-aware encoder büyük sıçrama verdi:

- `S7F_R2_family_aware_context_encoder`: AUPRC 0.982062, AUROC 0.906557, macro-F1 0.801716, MCC 0.603489, specificity 0.650888, TN=110.
- `S7F_R3_family_aware_experimental_emphasis`: AUPRC 0.984945, AUROC 0.926551, macro-F1 0.777185, MCC 0.568108, specificity 0.497041, TN=84.

Bu sprintin ana yorumu: daha büyük tek parça MLP değil, feature-family structure'a saygı duyan context encoder kazandırdı. Bu bulgu tezde model tasarım katkısı olarak anlatılmalıdır.

### Sprint 8A: Target-Context + Edge Interaction

Sprint 8A, context embedding ile candidate-edge S5F2_energy arasındaki interaction'ı test etti. Seçilen `S8A_R2_context_edge_film` sonucu: AUPRC 0.982757, AUROC 0.910575, macro-F1 0.777992, MCC 0.563656, specificity 0.520710, TN=88.

Bu, Sprint 7F çizgisine mekanizma-driven interaction eklenmesinin competitive kalabildiğini gösterdi. Ancak single-seed ve validation-selected olduğu için Sprint 9 öncesinde superiority claim kurulamaz.

### Sprint 8B: Sequence + Context Encoder

Sprint 8B, sequence/context fusion'ın Sprint 8A üzerine ek değer sağlayıp sağlamadığını test etti. Sequence-only path çöktü: AUPRC 0.856666, MCC -0.019749, specificity 0.0. Bu, local split altında context'siz sequence encoder'ın yeterli olmadığını gösterdi.

`S8B_R2_sequence_plus_context` en yüksek single-seed GNN AUPRC'yi verdi: AUPRC 0.986020, AUROC 0.903480, macro-F1 0.760058, MCC 0.567309, specificity 0.863905, TN=146. Bu model rare-negative recovery açısından çok güçlüdür; fakat FN=180 ile daha agresif negatif ayırma trade-off'u üretir. XGBoost F4 AUPRC barını geçmedi.

### Sprint 9: Robustness, Bootstrap ve Multi-Seed

Sprint 9 model geliştirmedi; saved predictions, guide-cluster bootstrap, paired-difference bootstrap ve multi-seed retraining ile iddia sınırlarını belirledi.

Ana AUPRC sonucu:

- XGB_F4 regenerated AUPRC 0.992338, 95% guide-cluster interval [0.950179, 0.999336].
- S8B_R2 AUPRC 0.986020, interval [0.929981, 0.998966].
- S7F_R3 AUPRC 0.984945, interval [0.923532, 0.999913].
- S8A_R2 AUPRC 0.982757, interval [0.910478, 0.999892].
- S7F_R2 AUPRC 0.982062, interval [0.914690, 0.999242].

Hiçbir paired AUPRC farkı sıfırı dışlamadı. Multi-seed AUPRC mean/std değerlerinde F4 0.990649 +/- 0.001944 ile en yüksek ve en stabil modeldir; GNN adaylarında std 0.004-0.012 aralığındadır ve aday kazanımlarından büyüktür.

Operating-point sonucu:

- GNN modeller single-seed threshold'ta F4'e göre daha fazla negatif yakalar.
- S7F_R2 MCC 0.6035 [0.426, 0.883], specificity 0.6509 [0.496, 0.900], macro-F1 0.8017 [0.712, 0.940].
- S8B_R2 specificity 0.8639 [0.575, 0.972].
- F4 specificity 0.2367 [0.047, 0.678], MCC 0.3511 [0.079, 0.699].

Fakat bu operating-point avantaj seed-fragile'dır ve negatifler yalnızca 9 guide içinde, özellikle guide 9251'de yoğunlaştığı için guide-composition-fragile olarak raporlanmalıdır.

## Literatürde Konumlandırma

### Sequence-Based CRISPR Off-Target Modeller

DeepCRISPR, CRISPR-Net, CnnCrispr, AttnToMismatch_CNN / AttnToOff, R-CRISPR, CRISPR-IP, CRISPR-M, Crispr-SGRU, CRISPR-BERT ve benzeri çalışmalar CRISPR off-target alanında sequence encoder, mismatch/indel representation, CNN/RNN/Transformer ve attention tabanlı yaklaşımların ana gövdesini oluşturur.

Bu literatür tezde üç amaçla kullanılmalıdır:

1. Off-target prediction'da guide-target sequence representation'ın merkezi önemini göstermek.
2. AUPRC/PR-AUC/AUROC metriklerinin neden literatür standardı olduğunu açıklamak.
3. Bizim çalışmanın doğrudan sequence-only benchmark olmadığını, context-aware graph/model-design sorusuna odaklandığını belirtmek.

Raw skorlar doğrudan karşılaştırılmamalıdır. Bu paperların çoğu genome-wide candidate pool, farklı negative sampling, farklı guide split, farklı label ve farklı prevalence kullanır. Bizim measured-only test setimiz pozitif ağırlıklıdır; bu yüzden PR-AUC sayıları literatürdeki negative-heavy retrieval evrenleriyle birebir kıyaslanamaz.

Sprint 8B bu literatüre kontrollü bir cevap verir: local sequence-only path zayıf kaldı, fakat sequence + context late fusion en iyi single-seed GNN AUPRC'yi verdi. Tezde bu, "sequence'i bırakmadık; context ile birlikte test ettik" şeklinde anlatılmalıdır.

### Graph-Based CRISPR ve GNN Literatürü

Kipf & Welling GCN, Vinodkumar et al. GCN-CRISPR ve GraphCRISPR gibi çalışmalar GNN yaklaşımının graph-structured biyolojik prediction problemlerine uygulanabilirliğini destekler. Vinodkumar et al. CRISPR off-target için sgRNA-target link prediction fikrini literatürde konumlandırır.

Bizim farkımız:

- Guide-disjoint, measured-only, no-test-tuning kontratı açıkça korunur.
- Graph A/B/C schema ablation yapıldı.
- Graph C target-observation context semantics ayrı bir katkı olarak incelendi.
- Edge-aware GATv2'de S5F2_energy message passing/attention hattına sokuldu.
- Robustness Sprint 9 ile AUPRC superiority sınırı açıkça çizildi.

Bu nedenle "GCN-CRISPR'i reproduce ettik" denmemelidir. Daha doğru ifade: Graph link-prediction fikrini CRISPR off-target için guide-disjoint epigenetic-context-aware bir workflow'a uyarladık.

### GAT/GATv2 ve Attention

GAT attention-weighted neighbor aggregation sağlar; GATv2, Brody et al.'ın static vs dynamic attention ayrımı nedeniyle bu proje için daha anlamlı varyanttır. Sprint 7 sonucu, GATv2'nin teorik olarak daha expressive olmasının tek başına başarı garantisi olmadığını gösterdi. Graph A üzerinde GAT/GATv2, aynı Graph A GCN referansını geçmedi.

Tezin doğru yorumu:

- Attention ancak doğru feature ve graph semantics ile birleştiğinde anlamlıdır.
- Edge-aware GATv2 + Graph C target-observation context başarılı operating-point davranışı üretti.
- Attention weights causal biological evidence değildir.

### Epigenetik ve Kromatin Context Literatürü

crisprSQL ve Mak et al. 2022 bu projenin veri ve feature-lineage merkezidir. Mak et al. 19 epigenetic descriptor'ı inceler: 6 experimental epigenetic ve 13 computed nucleosome feature. Mak çalışması computed nucleosome/BDM/NuPoP family'lerinin korelasyon/SHAP tarafında güçlü olduğunu, experimental epigenetic feature'ların ise kendi kurduğu continuous-target analysis içinde daha sınırlı katkı verdiğini raporlar.

Bizim bulgumuz Mak ile birebir aynı değildir ve bu iyi açıklanmalıdır. Bizim Scheme A, measured-only, guide-disjoint Graph C GATv2 setting'imizde direct target-observation experimental epigenetic feature'ların maskelenmesi modeli çökertti. Bu, Mak'in continuous CA ve SHAP analizinden farklı bir problemde, farklı bir model architecture'ında, experimental epigenetic context'in rare-negative recognition için çok kritik olduğunu gösterir. Bu çelişki gibi değil, problem-formulation farkı olarak yazılmalıdır.

Wu dCas9 ChIP-seq, Horlbeck, Isaac, Daer, Yarrington, DIG-seq, CHANGE-seq ve Verkuijl & Rots review gibi biyoloji çalışmaları chromatin accessibility, nucleosome positioning/occupancy ve epigenetic state'in Cas9 binding/cleavage davranışını etkileyebileceğini destekler. Bu literatür Sprint 7E/7F bulgusunun biyolojik olarak plausible olduğunu gösterir; ancak local model feature'larının causal effect olduğunu kanıtlamaz.

### Imbalance ve Metric Literatürü

Gao et al. ve Guan et al. CRISPR off-target prediction'da class imbalance'ın değerlendirmeyi nasıl yanıltabileceğini vurgular. Focal loss, Dice/Tversky, class-balanced loss, LDAM, SMOTE ve genel imbalance survey literatürü Sprint 6'nın neden predeclared loss/sampling comparison yaptığını açıklar.

Bu projede önemli terslik şudur: genome-wide off-target retrieval literatüründe çoğu kez pozitif off-target rare class'tır; bizim measured-only headline test evrenimizde pozitifler %90.07, negatifler rare class'tır. Bu yüzden imbalance yöntemleri körlemesine aktarılmadı. Sprint 6'da weighted BCE en iyi çıktı; SMOTE gibi synthetic data yöntemleri biyolojik graph/sequence geçerliliği belirsiz olduğu için ana workflow'a alınmadı.

Davis & Goadrich, Saito & Rehmsmeier, Williams ve Boyd et al. AUPRC/PR eğrisi ve imbalance altında metric interpretation için kullanılmalıdır. Sprint 9'da Field & Welsh, Efron, Schenker & Gentleman, Bengio & Grandvalet ve Bethard gibi kaynaklar guide-cluster bootstrap, paired-difference ve seed sensitivity sınırlarını destekler.

### Feature Fusion ve Context Encoder Literatürü

SE blocks, FiLM, GNN-FiLM, FiBiNET, dropout/shortcut learning ve overtuning/leakage literatürü Sprint 7F/8A/8B mimari kararlarını çerçeveler. Bu paperlar birebir reproduce edilmedi. Tezde "inspired by" dili kullanılmalı: family-aware context encoder, FiLM-style context-edge interaction ve sequence-context fusion, bu fikirlerin local CRISPR Graph C kontratına uyarlanmış halleridir.

Sprint 7F'nin en önemli mimari mesajı şudur: raw capacity değil, feature-family structure işe yaradı. Sprint 8A/8B ise bu structured context representation'ın edge interaction ve sequence fusion ile daha da rekabetçi hale gelebileceğini gösterdi.

## Tezin Ana Katkıları

1. Guide-disjoint, measured-only, no-test-tuning bir CRISPR off-target GNN evaluation workflow'u kuruldu.
2. Graph A/B/C ile graph schema ve target-context representation ayrıştırıldı.
3. Binding-energy feature'larının Graph A GCN için en güçlü edge sinyali olduğu gösterildi.
4. Loss/sampling değişikliklerinin tek başına remaining limitation'ı çözmediği gösterildi.
5. GAT/GATv2 attention'ın ancak edge-aware ve doğru target-observation context temsiliyle anlamlı hale geldiği gösterildi.
6. Graph C GATv2 kazanımı Sprint 7D/7E ile target-observation context, özellikle experimental epigenetic feature'lara lokalize edildi.
7. Family-aware target-context encoder rare-negative recognition ve macro-F1/MCC tarafında güçlü iyileştirme sağladı.
8. Sprint 9 ile AUPRC superiority sınırı dürüstçe çizildi; operating-point rare-negative katkı ayrı bir eksen olarak raporlandı.

## Önerilen Tez Bölüm Kurgusu

### Giriş

Girişte problem ikiye ayrılmalı: CRISPR-Cas9 off-target prediction için hem ranking hem de karar eşiğinde güvenilir negatif tanıma gereklidir. Literatür çoğunlukla genome-wide retrieval/ranking tarafına odaklanır; bu çalışma measured-only guide-disjoint benchmark'ta context-aware GNN'in nerede değer kattığını inceler.

Ana problem cümlesi:

> Bu çalışmada CRISPR-Cas9 off-target tahmininde epigenetik/kromatin bağlamının graph-based modeller içinde nasıl temsil edilmesi gerektiğini ve bu temsilin sıralama başarımı ile nadir negatif sınıf tanıma davranışına etkisini inceliyoruz.

### Literatür

Literatür liste halinde paper özeti olmamalı. Şu eksenlerle yazılmalı:

- Sequence-based CRISPR off-target prediction.
- Graph/GNN-based CRISPR modeling.
- Epigenetic/chromatin context and Cas9 activity.
- Class imbalance and PR-metric interpretation.
- Context/fusion/attention architecture components.
- Robust evaluation, guide-disjoint splitting, bootstrap and seed sensitivity.

### Materyal ve Yöntem

Burada veri audit'i, label scheme, split, feature groups, graph schemas, model families ve validation-only selection policy net anlatılmalı. Mak CA reproduction yapılmadığı açıkça belirtilmeli.

### Bulgular

Bulgular sprint sırasıyla değil, bilimsel soru sırasıyla verilebilir:

1. Baseline ve evaluation contract.
2. Graph schema baseline.
3. Feature-family ve loss/sampling kontrollü deneyleri.
4. Attention ve topology/context ayrımı.
5. Graph C mechanism ablation.
6. Context encoder ve sequence/context model improvement.
7. Robustness ve claim boundaries.

### Tartışma

Tartışma iki eksenli olmalı:

- AUPRC/ranking: GNN'ler rekabetçi fakat F4 robust bar.
- Rare-negative operating point: context-aware GNN'ler pratik ve biyolojik olarak anlamlı negatif recovery katkısı sağlıyor.

Bu iki eksen çatışmıyor; farklı bilimsel soruları yanıtlıyor.

## Kullanılacak İddia Dili

Güçlü ve doğru:

> Context-aware Graph C GATv2 modelleri, özellikle family-aware target-context encoder ve sequence/context fusion varyantları, XGBoost F4'e karşı robust AUPRC üstünlüğü göstermese de validation-kilitli operating point altında rare measured negatives için daha güçlü tanıma davranışı sergiledi.

Güçlü ve doğru:

> Sprint 7D/7E ablationları Graph C kazanımını explicit context edges yerine direct target-observation context feature'larına, özellikle experimental epigenetic alt aileye lokalize etti.

Güçlü ve doğru:

> Sprint 9 sonuçları, single-seed AUPRC point gain'lerinin guide-cluster ve seed uncertainty içinde kaldığını; buna karşılık threshold-dependent negative recovery etkisinin raporlanmaya değer fakat seed-fragile olduğunu gösterdi.

Kaçınılacak:

> Modelimiz state-of-the-art'tır.

Kaçınılacak:

> GNN XGBoost'u geçti.

Kaçınılacak:

> Attention weights biyolojik mekanizmayı açıkladı.

Kaçınılacak:

> Epigenetic feature'lar causal olarak off-target negatiflerini belirliyor.

## Metrik Raporlama Kuralları

- AUPRC her zaman primary metric olarak önce verilmeli.
- AUPRC yanında positive prevalence/no-skill PR baseline mutlaka belirtilmeli.
- AUROC secondary ranking metric olarak verilmeli, fakat imbalance altında AUPRC'nin yerine geçirilmemeli.
- MCC, macro-F1, specificity ve confusion matrix "ek tablo" gibi küçümsenmemeli; bu çalışmanın rare-negative recognition katkısının ana kanıtlarıdır.
- Threshold metrics validation-selected threshold ile hesaplanır; test threshold tuning yapılmadığı açıkça söylenmelidir.
- Sprint 9 sonuçlarıyla uyumlu olarak AUPRC farkları için "compatible with no difference" dili kullanılmalı; "equivalent" denmemelidir.

## Tezde Öne Çıkarılacak Sayısal Anchor'lar

| Karşılaştırma | AUPRC | MCC | Macro-F1 | Specificity | TN/FP/FN/TP | Not |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| XGBoost F4 | 0.992522 | 0.345198 | - | - | 38/131/21/1512 | Primary AUPRC bar |
| Graph A GCN WBCE | 0.976935 | 0.483719 | 0.698939 | 0.289941 | 49/120/6/1527 | Sprint 6 best loss |
| Graph C GATv2 | 0.969078 | 0.531774 | 0.739526 | 0.372781 | 63/106/12/1521 | Sprint 7B context signal |
| S7F R2 family-aware | 0.982062 | 0.603489 | 0.801716 | 0.650888 | 110/59/63/1470 | Best MCC/macro-F1 family |
| S7F R3 exp-emphasis | 0.984945 | 0.568108 | 0.777185 | 0.497041 | 84/85/31/1502 | Strong GNN AUPRC/AUROC |
| S8A R2 FiLM | 0.982757 | 0.563656 | 0.777992 | 0.520710 | 88/81/39/1494 | Context-edge interaction |
| S8B R2 seq+context | 0.986020 | 0.567309 | 0.760058 | 0.863905 | 146/23/180/1353 | Highest single-seed GNN AUPRC and TN recovery |

Bu tablo tezde "final leaderboard" olarak değil, anlatı anchor'ı olarak kullanılmalıdır. Sprint 9 robustness tablosu ile birlikte verildiğinde iddia dengesi korunur.

## Sonuç Anlatısı

Sonuç bölümü şu dengeyle bitmelidir:

> Bu çalışma, CRISPR-Cas9 off-target tahmininde context-aware GNN yaklaşımının değerini primary AUPRC üstünlüğüyle değil, mekanizma kontrollü ve validation-kilitli rare-negative recognition iyileşmesiyle göstermektedir. XGBoost F4 yüksek ve stabil AUPRC bar olarak kalırken, Graph C target-observation context ve family-aware GATv2 encoder çizgisi negatif sınıfı ayırma davranışında anlamlı bir katkı sağlamıştır. Bu katkı biyolojik olarak chromatin/epigenetic literatürüyle uyumludur, fakat causal interpretation gerektirmez; daha geniş guide population, daha fazla measured negative ve bağımsız external benchmark ile doğrulanması gereken predictive evidence olarak sunulmalıdır.
