# LaTeX Fit-Check Draft

This folder contains a low-fidelity integrated poster draft for fit-checking the thesis-poster plan. It is not the final poster, not final Turkish copy, and not a final production-tool decision.

The team may still produce the final poster in Canva, LaTeX, slides, or another tool. This draft exists only to see how the current content, microcopy, and vector figure assets fit on a 70x100 cm portrait surface.

## Files

- `poster_fit_check.tex` - 70x100 cm portrait LaTeX working draft.

The draft references rendered vector PDF assets from:

- `../../assets/rendered/fig01_measured_only_funnel.pdf`
- `../../assets/rendered/fig02_graph_abc_semantic_comparison.pdf`
- `../../assets/rendered/fig03_two_axis_results.pdf`
- `../../assets/rendered/fig04_tn169_rare_negative_recognition.pdf`
- `../../assets/rendered/fig05_literature_ab_positioning.pdf`

The SVG master files remain in `../../assets/` and should stay as editable source assets. The rendered PDFs are the LaTeX/print-facing versions for this fit-check.

## Compile Note

Compile from this folder with Tectonic:

```powershell
tectonic poster_fit_check.tex
```

The LaTeX draft does not call Inkscape during compilation. If the SVG sources change, regenerate the vector PDFs in `../../assets/rendered/` before rebuilding.

## Design Direction

The draft follows the current Apple-like / keynote-inspired direction:

- clean white / off-white surface;
- restrained slate text;
- blue, teal, and orange accents;
- low text density;
- large visual anchors;
- visible caveats;
- no dense academic-poster box grid.

Scientific boundaries override visual polish. Caveats must stay visible in any later redesign.

## Claim Rules Preserved

- No robust AUPRC superiority over XGBoost F4 is claimed.
- AUPRC remains the primary ranking metric.
- MCC and specificity are operating-point evidence.
- Operating-point and rare-negative gains are validation-locked and seed/guide sensitive.
- `0.900705` is the no-skill PR baseline, not a performance floor.
- Graph/model arrows do not imply biological causality.
- Literature positioning is qualitative; no raw-score leaderboard is used.

## Next Review Questions

- Does the title/hero area make Graph C and target-observation visible fast enough?
- Are the measured-only funnel and two-axis result band large enough?
- Is the TN/169 visual strong enough without looking like a global model win?
- Is the literature panel compact enough?
- Do the caveats remain readable at 70x100 cm print scale?
- Which microcopy should be cut before moving to any higher-fidelity draft?
