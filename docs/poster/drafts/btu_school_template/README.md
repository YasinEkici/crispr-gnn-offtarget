# BTU School-Template Poster Draft

This folder contains the first transfer of the earlier thesis-poster fit-check content into the school-provided BTU poster template.

It is a working draft, not a final print export.

## Files

- `poster.tex` - 70x100 cm portrait LaTeX poster using the BTU header, footer, blue section bars, and coordinate-based block layout.
- `poster.pdf` - generated proof after compilation, when present.
- `assets/btu-symbol.png` - BTU symbol copied from the school template.

The untouched source template is kept separately at:

- `docs/poster/templates/btu_poster_latex_template/`

## Text Source of Truth

Use this file as the canonical Turkish text source for future edits:

- `docs/poster/plan/poster_copy_deck.md`

It contains full and short variants for each poster section. Future layout passes should pull from the short variants first, then expand only where the BTU template boxes have room.

Do not invent new wording, claims, numbers, or percentages inside `poster.tex`. If a change seems scientifically necessary, flag it before changing the poster.

## Current Draft Source

The content was carried over from:

- `docs/poster/plan/poster_copy_deck.md`
- `docs/poster/drafts/latex_fit_check/poster_fit_check.tex`
- `docs/poster/assets/rendered/*.pdf`
- poster planning files under `docs/poster/notes/` and `docs/poster/plan/`

The earlier `latex_fit_check` draft is now only a content-fit reference. The active school-template draft is:

- `docs/poster/drafts/btu_school_template/poster.tex`

## Current Intent

The draft preserves the required school structure:

- BTU header;
- project title and subtitle;
- main poster sections in a readable order;
- footer with student/advisor information;
- references.

Box sizes and figure placements are intentionally adjustable. The current draft is meant to test whether the previous content can live inside the school format.

The design phase should still address the carry-forward issues from the legacy fit-check diagnosis:

- `docs/poster/plan/poster_overhaul_diagnosis.md`

That diagnosis is not a literal diagnosis of this active BTU draft; it mainly carries forward the figure redraw, shared visual system, and text-density concerns.

## Known Placeholders

The project repository does not currently contain final student numbers or emails. These are left as `[eklenecek]` in `poster.tex` and must be filled before final export.

## Compile

From this folder:

```bash
tectonic poster.tex
```

or with the bundled Codex LaTeX compiler:

```bash
python3 /Users/arcustin2/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py /Users/arcustin2/kasim/crispr-gnn-offtarget/docs/poster/drafts/btu_school_template/poster.tex
```

## Before Final Export

- Choose the final title.
- Fill student number and email fields.
- Check Turkish character rendering.
- Check grayscale and colorblind readability on the exported proof.
- Confirm that the honesty caveat appears once and remains readable.
- Verify that no raw-score literature leaderboard is introduced.
