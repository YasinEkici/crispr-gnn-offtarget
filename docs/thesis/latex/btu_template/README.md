# BTÜ Bilgisayar Mühendisliği Lisans Bitirme Çalışması LaTeX Şablonu

Bu klasör, BTÜ Bilgisayar Mühendisliği lisans bitirme çalışması için hazırlanmış XeLaTeX/Tectonic uyumlu LaTeX şablonudur. İlk sürüm `BTU_BM_Tez_Yazım_Sablonu_2019.docx` esas alınarak hazırlanmış; biçim kuralları `BTU_BM_Tez_Yazım_Sablonu_2026_updated.docx` içindeki güncel notlar ve paylaşılan lisansüstü LaTeX şablonu referans alınarak güncellenmektedir.

## Derleme

Tectonic ile:

```bash
tectonic main.tex
```

Yerel TeX Live ile:

```bash
xelatex main.tex
xelatex main.tex
xelatex main.tex
```

## Nereleri değiştireceksin?

Önce `metadata.tex` dosyasını doldur:

- Türkçe ve İngilizce tez başlığı
- öğrenci adı soyadı ve öğrenci numarası
- danışman bilgisi ve idari tarih alanları
- savunma tarihi, ay/yıl bilgisi
- anahtar kelimeler

Sonra `chapters/` ve `appendices/` klasörlerindeki dosyaları kendi tez içeriğinle değiştir.

## Şablonun uyguladığı ana kurallar

- A4 sayfa.
- Sol/iç kenar boşluğu 4 cm; sağ/dış, üst ve alt kenar boşlukları 2.5 cm.
- Ana metin 12 punto; DOCX varsayılanına uygun olarak Times New Roman kullanılır. Times New Roman sistemde yoksa Tinos/TeX Gyre Termes fallback kullanılır. Dış kapak Arial-benzeri sans fontla, iç kapak ve tez gövdesi Times-benzeri serif fontla ayarlanmıştır.
- Ana metin 1.5 satır aralığı ve iki yana yaslı.
- Önsöz, içindekiler, kısaltmalar, semboller, şekil/çizelge listeleri, özetler ve kaynaklar tek satır aralığına yakın düzenlenmiştir.
- 2026 DOCX akışıyla uyumlu olarak dış kapak, iç kapak, boş sayfa ve İNTİHAL BEYANI sayılır ama sayfa numarası gösterilmez; ÖNSÖZ sayfası roman `v` ile görünür.
- Ana metin `1` numarasından başlar.
- 1., 2., 3. ve 4. derece başlıklar numaralandırılır; 5. derece başlık numarasızdır ve içindekiler listesinde verilmez.
- Şekil başlıkları altta, çizelge başlıkları üstte; başlıklar DOCX `ResimYazs` stiline uygun olarak 10 punto basılır ve etiketler `Şekil 2.1 :` / `Çizelge 2.1 :` formundadır. Güncel 2026 şablon notlarına göre şekil ve çizelge açıklamaları nokta ile bitirilir.
- Kaynakça tek satır aralıklı, iki yana yaslı ve 2.5 cm asılı girintili basılır.

## Not

Varsayılan ön sayfa akışı 2026 DOCX render akışıyla eşleştirilmiştir: dış kapak, iç kapak, boş sayfa, İNTİHAL BEYANI, ÖNSÖZ. Onay/jüri sayfası makrosu class içinde korunur, ancak varsayılan `main.tex` akışında basılmaz.


## Font notu

Word şablonunun DOCX iç varsayılanı Times New Roman’dır. Şablon, XeLaTeX/Tectonic ile derlenirken sistemde Times New Roman varsa doğrudan onu kullanır. Arial dış kapak için tercih edilir; Arial yoksa Arimo/Liberation Sans kullanılır. Önizleme PDF’i Linux ortamında derlendiği için Times New Roman yerine Tinos ve Arial yerine Arimo görebilirsin; kendi makinenizde Times New Roman/Arial varsa çıktı Microsoft fontlarıyla oluşur.

Bu repodaki lokal doğrulamada derlenen PDF’in font kaynakları kontrol edildi ve gövdenin gerçek Times New Roman ile üretildiği doğrulandı. Ayrıntılar için `docs/thesis/notes/btu_template_verification.md` dosyasına bak.
