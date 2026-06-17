# 2026 Şablon Geçiş Denetimi

Bu not, `BTU_BM_Tez_Yazım_Sablonu_2026_updated.docx` ve paylaşılan `btu-lisansustu-tez-yazim-sablonu-latex` klasörü incelenerek çıkarılan geçiş kararlarını özetler.

## Kullanılan referanslar

- `docs/thesis/templates/btu_docx/BTU_BM_Tez_Yazım_Sablonu_2026_updated.docx`
- `docs/thesis/templates/btu_docx/BTU_BM_Tez_Yazım_Sablonu_2019.docx`
- `docs/thesis/btu-lisansustu-tez-yazim-sablonu-latex/BTU Şablon/thesis.tex`
- `docs/thesis/btu-lisansustu-tez-yazim-sablonu-latex/BTU Şablon/styles/tez.sty`
- Mevcut LaTeX şablonu: `docs/thesis/latex/btu_template/`

DOCX, `documents` aracıyla sayfa görsellerine ve PDF'e render edilmiştir. Render 40 sayfa üretmiştir. DOCX içinde 125 açıklama/not bulunduğu doğrulanmıştır.

## Güvenli taşınan kurallar

- Özet ve Summary aralığı güncel DOCX'e göre 250-750 kelimedir.
- Özet/Summary içinde kaynak, şekil ve çizelge verilmez.
- KISALTMALAR ve SEMBOLLER listelerinde terim koyu, iki nokta ve açıklama normal biçimde verilir.
- Şekil açıklaması altta, çizelge açıklaması üsttedir.
- Şekil ve çizelge açıklamaları nokta ile bitirilir.
- Tek satırlı açıklamalar ortalı, çok satırlı açıklamalar iki yana yaslı/asılı düzende olmalıdır.
- KAYNAKLAR tek satır aralıklı, iki yana yaslı ve 2,5 cm asılı girintili olmalıdır.
- Bölüm başlıkları yeni sayfadan başlar; birinci derece başlıklarda 72 pt önce ve 18 pt sonra boşluk kuralı korunur.
- Birinci derece başlıklar tamamen büyük, ikinci derece başlıklar kelime baş harfleri büyük, üçüncü/dördüncü derece başlıklar yalnız ilk harf büyük olacak şekilde izlenmelidir.
- Beşinci derece başlık numaralandırılmamalı ve içindekiler listesine alınmamalıdır.

## Dikkat gerektiren farklar

- Güncel DOCX render akışı dış kapak, iç kapak, boş sayfa, intihal beyanı biçiminde görünmektedir. Mevcut LaTeX varsayılan akışı bu sıraya çekilmiştir. Onay/jüri sayfası makrosu class içinde korunur, ancak varsayılan `main.tex` akışında basılmaz.
- Paylaşılan lisansüstü LaTeX şablonu yüksek lisans/doktora diline ve eski pdfLaTeX paketlerine dayanmaktadır. Bu nedenle doğrudan kopyalanmamalı; kural ve biçim davranışı mevcut XeLaTeX/Tectonic uyumlu lisans şablonuna seçici biçimde taşınmalıdır.
- DOCX kapak sayfasında not kutuları ve arka plan görseli bulunur. Bunlar çıktı tasarımının parçası olarak değil, şablon hazırlama notu olarak değerlendirilmelidir.
- Kaynakça için DOCX yazar-tarih ve APA örnekleri vermektedir. Tezde tek sistem olarak APA/yazar-tarih kullanılmaya devam edilmelidir.

## Bu geçişte yapılanlar

- `btu-thesis.cls` kaynakça ortamı 2,5 cm asılı girintiyle güncellendi.
- `main.tex` ön sayfa akışı 2026 DOCX render sırasına çekildi: dış kapak, iç kapak, boş sayfa, intihal beyanı.
- Dış kapakta DOCX içinden çıkarılan arka plan görseli kullanılmaya başlandı; kapak metinleri DOCX'teki sayfa merkezli yerleşim, renk ve punto düzenine yaklaştırıldı.
- İç kapak DOCX'teki logosuz, 4 cm / 2,5 cm metin alanı merkezli düzene çekildi; ikinci danışman satırı yalnız `\BTUSecondAdvisor` doluysa basılacak şekilde koşullu bırakıldı.
- Final render cross-check sırasında ÖNSÖZ ve ön liste başlık koordinatları DOCX render'ına göre tekrar hizalandı. Ölçülen 2026 DOCX/LaTeX başlık üst koordinatları: ÖNSÖZ `y=126.2/126.3`; İÇİNDEKİLER `144.2/144.2`; KISALTMALAR `144.2/144.1`; SEMBOLLER `144.2/144.1`; ÇİZELGE LİSTESİ `144.2/144.2`; ŞEKİL LİSTESİ `144.2/144.2`.
- 2019 DOCX ayrıca render edilerek ÖNSÖZ konumunun 2026 dosyasındaki gibi `y=126.2` olduğu doğrulandı. Bu nedenle ÖNSÖZ başlığı, ortak ön bölüm başlığı ritmine alınmadı; şablondaki ayrı konumlandırma korundu.
- DOCX'te 125 açıklama/yorum bulunduğu doğrulandı. Bunlar şablon kılavuz notları olarak değerlendirildi; final LaTeX çıktısına yorum kutusu veya kapaktaki mavi açıklama kutuları taşınmadı.
- Noktasız kalan şekil başlıkları 2026 kuralına göre nokta ile bitirildi.
- `tez_yazim_meta_kurallari.md` özet aralığı, caption noktalaması ve kaynakça biçimi açısından güncellendi.
- `btu_template/README.md` 2019 referansını tarihsel bilgiye çekip 2026 DOCX ve lisansüstü LaTeX şablonunu güncel referans olarak belirtti.

## Sonraki karar noktaları

1. CV/Özgeçmiş sayfası gerçek bilgiyle doldurulacak mı, yoksa şimdilik placeholder kalmaya devam edecek mi?
2. Mevcut bölüm başlıkları 2026 başlık hiyerarşisine göre ayrıca taranmalı; özellikle ikinci/üçüncü derece başlıkların büyük-küçük harf düzeni kontrol edilmelidir.
3. Kaynakça APA içeriği biçimsel olarak ayrıca son kez denetlenmeli; bu geçiş yalnız LaTeX dizgisi ve şablon uyumunu kapsar.
