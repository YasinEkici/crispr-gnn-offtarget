# Thesis / Report Workspace

This folder stores thesis/report writing material after the experimental sprint phase.

## Layout

- `notes/`: narrative framing, outline notes, claim boundaries, and thesis-writing context.
- `latex/`: LaTeX or Overleaf-compatible thesis/report sources. Keep institution-specific templates here when available.
- `latex/btu_template/`: BTU Computer Engineering thesis template converted from the official DOCX format.
- `templates/btu_docx/`: original DOCX thesis template kept as a formatting reference.

## BTU Template

The current BTU template source is under:

```text
docs/thesis/latex/btu_template/
```

The original Word template reference is under:

```text
docs/thesis/templates/btu_docx/BTU_BM_Tez_Yazım_Sablonu_2019.docx
```

The DOCX-to-LaTeX verification note is under:

```text
docs/thesis/notes/btu_template_verification.md
```

To compile the LaTeX template locally:

```bash
cd docs/thesis/latex/btu_template
tectonic main.tex
```

or, with a local TeX Live / MacTeX setup:

```bash
cd docs/thesis/latex/btu_template
xelatex main.tex
xelatex main.tex
xelatex main.tex
```

## Ground Rules

- Treat experiment outputs under `outputs/` as evidence; do not duplicate large artifacts here.
- Keep source LaTeX and small figures/tables under version control only when they are lightweight.
- Do not move model checkpoints, large prediction files, or raw data into this folder.
- Keep thesis claims aligned with the frozen evaluation contract and Sprint 9 robustness boundaries.
- Do not commit generated thesis PDFs or LaTeX build artifacts; they are ignored by `.gitignore`.
