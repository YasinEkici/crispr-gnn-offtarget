# BTU Thesis Template Verification

This note records the local verification performed after importing the Bursa Technical University Computer Engineering thesis DOCX template into the repository as a XeLaTeX/Tectonic template. It documents the original 2019 mapping; for the 2026 transition decisions, see `template_2026_migration_audit.md`.

## Source Files

- Original DOCX reference: `docs/thesis/templates/btu_docx/BTU_BM_Tez_Yazım_Sablonu_2019.docx`
- LaTeX template: `docs/thesis/latex/btu_template/`
- Main class file: `docs/thesis/latex/btu_template/btu-thesis.cls`

## DOCX Reference Findings

The DOCX was inspected through its OOXML contents (`word/styles.xml` and `word/document.xml`).

- Document default font: `Times New Roman` for `ascii`, `hAnsi`, `eastAsia`, and `cs` font slots.
- Page size: A4 (`11907 x 16840` twips in the inspected file).
- Margins: left `4.0 cm`; top, right, and bottom approximately `2.5 cm`.
- Body style (`GOVDE`): before `120` twips, after `120` twips, line `360` twips, justified.
- Heading styles:
  - `BASLIK1`: before `1440`, after `360`, line `360`, bold.
  - `BASLIK2`: before `360`, after `240`, line `360`, bold.
  - `BASLIK3` / `BASLIK4`: before `240`, after `120`, line `360`, bold.
- Caption style (`ResimYazs`): 10 pt (`w:sz=20`) with bold label styling.

The DOCX outer cover uses Arial-like direct formatting. The inner cover and thesis body inherit the Times New Roman document default.

## LaTeX Mapping

The LaTeX template maps these requirements as follows:

- `geometry`: A4 with left `4cm`, right/top/bottom `2.5cm`.
- `fontspec`: uses real `Times New Roman` when available; falls back to `Tinos` or `TeX Gyre Termes` only if Times New Roman is unavailable.
- Sans font for outer cover: uses real `Arial` when available; falls back to compatible sans fonts.
- Body: 12 pt with 18 pt leading, one-and-a-half line spacing, justified text, no paragraph indent, and 6 pt paragraph skip.
- Headings: spacing matched to the DOCX heading styles above.
- Captions: 10 pt caption font with bold label and `Şekil 2.1 :` / `Çizelge 2.1 :` separators.
- Front matter: roman page numbering starts at the front matter and visible numbering starts from `ÖNSÖZ` as `v`; main matter restarts at Arabic `1`.

## Fixes Applied During Verification

- The approval-page opening paragraph originally used aggressive no-hyphenation and `\mbox{...}` wrappers, which produced visibly bad inter-word spacing after conversion. The paragraph was simplified while preserving the text and fixed page placement.
- Caption font size was corrected from normal body size to 10 pt to match the DOCX `ResimYazs` style.

## Compile And Font Verification

The template was compiled with bundled Tectonic:

```bash
python3 /Users/arcustin2/.codex/plugins/cache/openai-bundled/latex/0.2.2/scripts/compile_latex.py \
  /Users/arcustin2/kasim/crispr-gnn-offtarget/docs/thesis/latex/btu_template/main.tex \
  --compiler tectonic --json
```

The compile succeeded and produced a 20-page PDF.

PDF font resources were inspected with `pypdf`. The generated PDF used:

- `Arial-BoldMT` on the outer cover.
- `TimesNewRomanPSMT`, `TimesNewRomanPS-BoldMT`, and `TimesNewRomanPS-ItalicMT` on inner/front/body pages.
- `XITSMath-Regular` for math.

This confirms that the local build used real Times New Roman rather than the fallback font.

## Visual QA

Representative PDF pages were rendered through macOS Quick Look thumbnails and visually inspected:

- Page 1: outer cover.
- Legacy 2019 mapping page 3: approval page after spacing fix. In the current 2026 DOCX-aligned default flow this page is not printed; page 3 is a blank page after the inner cover.
- Page 6: table of contents.
- Page 13: main text and heading hierarchy.
- Page 14: figure caption, table caption, and equation layout.
- Page 20: CV page.

No clipping, overlapping text, missing glyphs, or obvious page-boundary issues were observed in the inspected pages.

## Limitations

LibreOffice / `soffice` was not available in this local environment, so the original DOCX was not rendered to page PNGs through the DOCX rendering workflow. The verification used direct OOXML inspection for the DOCX and compiled-PDF visual QA for the LaTeX output.
