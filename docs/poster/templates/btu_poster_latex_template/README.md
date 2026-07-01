# BTÜ Bilgisayar Mühendisliği 70x100 Poster LaTeX Şablonu

Bu klasör, `Poster Sergisi Sunum Formatı.pptx` dosyasındaki poster düzeninin LaTeX/Tectonic ile kullanılabilecek yeniden kurulmuş sürümüdür.

## Derleme

```bash
tectonic poster.tex
```

veya XeLaTeX ile:

```bash
xelatex poster.tex
```

Çıktı PDF sayfa boyutu doğrudan **70 cm x 100 cm** olacak şekilde ayarlanmıştır.

## Nereleri değiştireceğim?

`poster.tex` içinde en üstteki alanları düzenle:

```tex
\newcommand{\ProjectTitle}{...}
\newcommand{\ProjectSubtitle}{...}
\newcommand{\StudentName}{...}
\newcommand{\StudentNumber}{...}
\newcommand{\StudentEmail}{...}
\newcommand{\AdvisorName}{...}
\newcommand{\AdvisorEmail}{...}
```

Ana kutular şunlar:

```tex
\PosterBlock{3.83}{14.33}{19.54}{22.18}{Özet}{ ... }
```

Sırasıyla: `x`, `y`, `genişlik`, `gövde yüksekliği`, `başlık`, `içerik`.
Koordinatlar santimetredir ve sol üst köşeden başlar.

Metni sola hizalamak için:

```tex
\PosterBlock[\raggedright]{...}{Başlık}{...}
```

Şekil açıklaması için:

```tex
\ImageWithCaption{x}{y}{genişlik}{yükseklik}{assets/gorsel.png}{1}{Şekil yazısı.}
```

Sadece yer tutucu için:

```tex
\FigureSlot{x}{y}{genişlik}{yükseklik}{1}{Şekil yazısı.}
```

Tablo açıklaması:

```tex
\TableCaption{x}{y}{genişlik}{1}{Tablo adı.}
```

## Font notu

PPTX içinde üst başlıkta Arimo Bold, içerik/alt bilgi tarafında Aptos/Calibri ailesi görünüyor. LaTeX dosyası sisteminde varsa bu fontları kullanır. Yoksa otomatik olarak benzer açık kaynak fontlara düşer: Calibri yerine Carlito, Aptos yerine Noto Sans/TeX Gyre Heros.

Font dosyaları bu pakete dahil edilmemiştir.

## Ölçü notu

Kaynak PowerPoint dosyasının kendi slayt ölçüsü yaklaşık 59.37 cm x 84.10 cm olarak okunuyor; fakat şablon metninde ve istenen teslim formatında 70 cm x 100 cm belirtiliyor. Bu LaTeX sürümü gerçek PDF sayfasını 70 cm x 100 cm üretir ve PPTX'teki yerleşimi bu ölçüye ölçekler.
