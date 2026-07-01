# Poster Overhaul Diagnosis

Read-only diagnostic pass on the current CRISPR-Cas9 thesis-poster fit-check. It names concrete formatting and figure problems and diagnoses their causes, judged against (a) the official BTÜ poster template's rigidly gridded formatting and (b) standard academic-poster composition. It proposes no full solutions; findings feed a later overhaul phase.

This is an English planning note; poster-bound phrases are quoted in Turkish. No poster, figure, SVG, rendered PDF, or `.tex` was modified.

## 1. Method

- Compiled `docs/poster/drafts/latex_fit_check/poster_fit_check.tex` with Tectonic and rendered the single-page PDF to PNG at ~1.1x (pypdfium2) for full-poster inspection.
- Rendered each master SVG (`docs/poster/assets/fig01..fig05`) to PNG at ~1500 px width (cairosvg) and inspected each at full size.
- Cross-read `poster_design_decisions.md`, `poster_narrative_framing.md`, `poster_yazim_kurallari.md`, `layout_draft_brief.md`, `poster_content_plan.md`, and `poster_copy_deck.md` for intended zones, aesthetic, terminology, and claim boundary.
- Each finding below is stated as observation → diagnosis → severity, from what was actually seen in the renders.

## 2. Global layout and grid diagnosis

**Observation.** Section tops do not align across the two body columns; figure titles begin at a different x/y than the numbered kicker rules above them; the left column (sections 1-3) ends noticeably higher than the right column (sections 4-6), leaving a bottom gap; gutters between blocks are not uniform.

**Diagnosis.** There is no shared layout grid. Each figure is a self-contained SVG carrying its own internal margins, title baseline, and scale, dropped into LaTeX `minipage`s that are sized independently. Nothing snaps to common horizontal/vertical gridlines, so the eye reads "placed by hand," not "formatted." This is the root cause of the user's "randomly placed / unformatted" complaint. Severity: **Blocker**.

**Observation.** Most content sits either inside a figure's own soft-panel background or as bare text directly on the poster background; there is no consistent section container.

**Diagnosis.** Unlike the BTÜ template's uniform bordered section boxes with colored header bars, the current poster mixes framed (figure) and unframed (text) blocks, so sections have uneven visual weight and blur into each other. Severity: **Major**.

## 3. Typography and text-placement diagnosis

**Observation.** Every figure repeats a large title ("Veri Sözleşmesi", "Bulgular: iki ayrı eksen", "Karar eşiğinde nadir negatif tanıma", "Graph A/B/C: temsil neyi değiştiriyor?", "Literatürde nereye oturuyor?") directly under a numbered kicker that says nearly the same thing ("③ YÖNTEM · VERİ SÖZLEŞMESİ", "⑤ BULGULAR · İKİ AYRI EKSEN", etc.).

**Diagnosis.** Two heading systems compete: the LaTeX kicker rail and the figure-embedded titles. They use different fonts and sizes and duplicate meaning, which both wastes vertical space and reads as unstructured. A poster needs one heading system, owned by the layout, with figures as untitled visuals. Severity: **Blocker**.

**Observation.** Body paragraphs are justified and read well; the kicker rules are consistent. But heading-to-figure spacing is inherited from each figure's internal top padding, so the gap between a kicker and its figure varies section to section.

**Diagnosis.** Because figures carry their own top whitespace, the text-to-figure rhythm is set inside the assets rather than by the layout, producing inconsistent vertical spacing. Severity: **Major**.

## 4. Per-figure diagnosis

### fig01 — measured-only funnel (reference-quality)

**Observation.** Three rounded number cards (310142 → 25632 → 1702) with clear arrows, a legend row (29 rehber / 1533 pozitif / 169 negatif), and a two-line caption. Clean, legible, well-spaced.

**Diagnosis.** This is the strongest asset and the quality bar the others should meet. Minor issues only: the three cards are equal-sized rather than actually narrowing (a "funnel" that does not funnel), and the legend colors (blue/teal/orange) get reused with different meaning elsewhere (see §5). Severity: **Minor**.

### fig02 — Graph A/B/C semantic comparison (priority failure)

**Observation.** In the Graph C panel, "gözlem 1" and "gözlem 2" labels sit on top of the circle strokes. Nodes are large relative to the 360 px panels, leaving little breathing room. All three panels look like generic 2-3 circle clusters; the only differences are node count and one blue tint.

**Diagnosis.** The hero method figure fails to communicate its single most important idea. (1) Label/node collisions come from placing text at circle centers without reserving space or moving labels outside. (2) The A→B→C semantic progression — single physical-target node → guide-similarity control → per-observation "target-observation" split — has no visual grammar to carry it; nothing shows "aynı adres, farklı ziyaretler" (same address, different visits). A viewer cannot see why Graph C is different in kind, only that it has more circles. This figure needs a genuine redraw, not a tweak. Severity: **Blocker**.

### fig03 — two-axis results (confusing artifacts + clipping)

**Observation.** The "0.900705 no-skill PR baseline" is drawn as a horizontal line with a filled dot near its right end. Under "0.992338" and "0.986020" there are colored underline bars. The right "Karar Eşiği" panel's line "Doğru negatifler aynı paydada okunur: 40, 14, 63, 110 / 169." runs into the right edge and clips. The two inner panels are unequal height.

**Diagnosis.** (1) The baseline line-plus-dot reads like an interactive slider/toggle, implying a control that does not exist. (2) The underline bars imply to-scale value bars but are decorative and not proportional, risking misreading of a near-tie as a large gap. (3) The right-panel text overflows its container, the same class of overflow already fixed in fig02/fig05. (4) Unequal panel heights break the two-axis symmetry the figure is supposed to convey. Severity: **Major**.

### fig04 — TN/169 rare-negative recognition (strong, minor color issue)

**Observation.** Four horizontal bars over a shared 169 denominator, each with a value (40/14/63/110) and a specificity label, plus a validation-locked / seed-guide caveat and the non-monotone note about Graph C GCN. Clear and honest.

**Diagnosis.** Communicates well and respects the claim boundary. Only issue: it introduces purple for "aile-duyarlı encoder" (a color not in the poster/BTÜ palette) and uses orange for "Graph C GCN" while orange means "169 negatif" in fig01 — inconsistent color semantics across figures. Severity: **Minor**.

### fig05 — literature A+B positioning (acceptable)

**Observation.** Two panels (A: "Neden doğrudan kıyaslanamaz?" with four qualitative axes; B: "Hangi soru soruluyor?" with a ranking→binary flow) and a warning strip forbidding a raw-score leaderboard. Text wrapping was previously fixed.

**Diagnosis.** Functionally sound and claim-safe. It shares the double-title problem (§3) and the standalone-panel styling that differs from the other figures' card styles (§5). Severity: **Minor**.

## 5. Visual-consistency diagnosis

**Observation.** Across figures: orange = "169 negatif" (fig01) but also = "Graph C GCN" (fig04); fig04 adds purple; blue and teal recur with shifting referents; panel corner radii and stroke weights differ (fig02 soft panels vs fig03/fig05 bordered white panels); figure background tints vary (`#fbfdff`, `#f8fafc`).

**Diagnosis.** There is no shared visual system (palette-with-meaning, one card style, one radius/stroke scale, one legend). Each figure was styled in isolation, so the set does not read as one poster. Severity: **Major**.

## 6. Claim-boundary visual check

**Observation and diagnosis.**
- Arrows in fig01/fig02 already carry "Oklar temsil akışını gösterir; biyolojik nedensellik göstermez." — compliant.
- No raw-score leaderboard anywhere; fig05 explicitly forbids one — compliant.
- 0.900705 is labeled "no-skill PR baseline", not a floor — compliant.
- fig03 right panel says "MCC/specificity operating-point evidence olarak yorumlanır." Not a violation, but per the true-negative-lens correction it should foreground the true-negative / specificity behavior and treat MCC/specificity as the lens; wording like "nadir ölçülmüş negatifleri doğru negatif olarak tanıma" is preferable.
- fig03 caption uses first-person-free wording; fine. No first-person violations were seen in the figures.
Severity: **Minor** (wording alignment only; no hard breach).

## 7. Comparison table — BTÜ template vs current poster

| Formatting quality | BTÜ template | Current poster | Verdict |
| --- | --- | --- | --- |
| Shared multi-column grid | rigid, everything snaps | figures self-place; columns unaligned | **fail** |
| Section containers | uniform boxes + header bars | mixed framed/unframed | **weak** |
| Uniform gutters/margins | consistent | varies per figure padding | **weak** |
| Single heading system | one per box | kicker + figure title compete | **fail** |
| Figure clarity | n/a (placeholders) | fig01/fig04 good; fig02 fails, fig03 confusing | **weak** |
| Numbered captions (Şekil/Tablo) | present | absent (titles baked into figures) | **fail** |
| Header band | centered, formal | left title + right identity; ok | **pass** |
| Footer identity | clean | present, compact | **pass** |
| Color system | uniform | inconsistent semantics across figures | **weak** |

## 8. Prioritized findings

**Blocker** (causes the unformatted / unreadable impression):
1. No shared layout grid; figures self-place (§2).
2. Competing double heading system (kicker + figure titles) (§3).
3. fig02 Graph A/B/C fails to communicate; node/label collisions (§4).

**Major:**
4. No consistent section-container system vs the BTÜ boxed grid (§2).
5. fig03 slider-like baseline artifact, non-proportional underline bars, edge-clipped text, unequal panels (§4).
6. No shared visual system across figures (color meaning, card style, radii) (§5).
7. Figure-owned top padding makes text-to-figure spacing inconsistent (§3).

**Minor:**
8. fig04 palette drift (purple) and cross-figure color-meaning clash (§4/§5).
9. Left-column bottom gap / uneven vertical rhythm (§2).
10. fig03 true-negative-lens wording alignment (§6).
11. fig01 cards do not actually narrow (§4).

## 9. Overhaul direction (diagnosis-level only)

The overhaul must hit these target qualities (not designed here):

- **One explicit grid.** A fixed column/row grid that all sections and figures snap to, with uniform gutters and margins, so nothing floats.
- **One heading system.** Layout-owned section headers (or numbered cards); figures become untitled visuals with numbered "Şekil n" captions, removing the double-title competition.
- **Consistent section cards.** A single card style (border/header-bar, radius, padding) applied to every section, giving uniform visual weight like the BTÜ template.
- **Redrawn Graph A/B/C (fig02).** A figure with real visual grammar for the A→B→C semantic progression and no label/node collisions, so "hedef-gözlem" and "aynı adres, farklı ziyaretler" actually read.
- **Reworked fig03.** Remove the slider-like baseline and decorative underline bars; use one honest, proportional (or clearly non-scaled) representation; fix the overflow; equalize the two axis panels.
- **A shared visual system.** One palette with fixed meaning, one card/radius/stroke scale, one legend convention across all figures; keep it within the BTÜ/poster palette (no stray purple).
- **Preserve throughout:** the claim boundary, the honesty caveat once inside results, impersonal-active register, and the true-negative lens.

The redraw specs, grid dimensions, and new figure designs belong to the separate overhaul phase and are intentionally not fixed here.
