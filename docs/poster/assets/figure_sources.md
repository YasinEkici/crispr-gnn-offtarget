# Poster Figure Sources

This file records the source, allowed numbers, claim caveat, and claim risk for the Slice 5 poster figure assets. The SVGs are low-fidelity / medium-fidelity working assets for planning and fit-checking. They are not final poster production files and do not choose Canva, LaTeX, slides, or any other final tool.

The visual direction follows the current Apple-like / keynote-inspired discipline: clean background, restrained palette, high typographic hierarchy, low text density, and visible caveats. Scientific accuracy overrides aesthetic polish.

## Assets

| File | Figure role | Source / allowed anchors | Required caveat |
| --- | --- | --- | --- |
| `fig01_measured_only_funnel.svg` | Data contract and measured-only funnel | `310142 -> 25632 -> 1702`; test: 29 guides, 1533 positives, 169 negatives; Scheme A | `"Doğrulanmamış adaylar (measured=0) test etiketi yapılmadı."` |
| `fig02_graph_abc_semantic_comparison.svg` | Graph A/B/C semantic comparison | Thesis Graph A/B/C methodology and approved terminology | Graph arrows show representation flow, not biological causality. |
| `fig03_two_axis_results.svg` | AUPRC ranking vs operating-point result split | XGBoost F4 AUPRC 0.992338 [0.950179, 0.999336]; dizi+bağlam Graph C GATv2 AUPRC 0.986020 [0.929981, 0.998966]; no-skill PR baseline 0.900705; feature-group encoder MCC 0.603489 | Exact honesty caveat inside the result band. |
| `fig04_tn169_rare_negative_recognition.svg` | Rare-negative recognition at decision threshold | TN over 169: 40, 14, 63, 110; specificities 0.236, 0.083, 0.373, 0.651; MCC 0.603489 | Operating-point results are validation-locked and seed/guide sensitive. |
| `fig05_literature_ab_positioning.svg` | Qualitative literature positioning A+B | Literature index and poster narrative framing; no numeric literature scores | No raw-score leaderboard against other papers. |

## Shared Claim Rules

- Do not claim robust AUPRC superiority over XGBoost F4.
- Keep AUPRC as the primary ranking metric.
- Frame MCC and specificity as operating-point evidence.
- Treat threshold and rare-negative gains as seed/guide-fragile.
- Label `0.900705` as the no-skill PR baseline, not a performance floor.
- Do not infer biological causality from graph arrows, attention, gates, FiLM, masking, embeddings, or feature-group effects.
- Do not compare raw performance scores against other papers.

## Design Notes

- These SVGs use editable text and simple shapes.
- The palette is intentionally restrained: slate text, blue/teal accents, and orange caveat highlights.
- Color is paired with text labels; meaning should not depend on color alone.
- Rounded containers are used sparingly as figure structure, not as a dense academic-poster grid.
- The files may be restyled later for Canva, LaTeX, slides, or another production tool.

## Remaining TODOs

- Review SVG rendering in the chosen production environment.
- Adjust exact sizing after the integrated 70x100 cm poster draft exists.
- Decide whether to merge `fig03_two_axis_results.svg` and `fig04_tn169_rare_negative_recognition.svg` in the final layout.
- Replace or refine these working assets if a later designer produces higher-fidelity equivalents with the same claim boundaries.
