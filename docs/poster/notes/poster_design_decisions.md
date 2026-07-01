# Poster Design Decisions

This document is the **poster constitution** for the CRISPR-Cas9 off-target prediction thesis. It fixes the principles the poster is built on and the rationale behind each decision. It contains no layout, no section copy, and no final design; those belong to later phases (content plan, figure production, writing rules). The sibling documents are not written yet, but this document provides their frame.

Note on language: these planning documents are written in **English**, while the **poster itself is in Turkish** (see §4). Where a phrase that will appear on the poster is quoted, it is kept in Turkish.

Upstream sources: the current thesis (`docs/thesis/latex/btu_template/chapters/`), `docs/thesis/notes/main_narrative_framing.md`, the literature axes (`docs/literature/literature_index.md`), `AGENTS.md`, and the project docs.

## 1. Purpose and Scope

The poster will be presented at the graduation project exhibition to both the jury and non-specialist visitors. It therefore serves two functions at once: conveying the core message at a single glance, and supporting an approximately two-minute spoken pitch. Because the topic required more research and development than a typical graduation project, the poster balances simplification against scientific rigor. The goal is to stay accessible in a "crispr-cas9 for dummies" spirit while changing no claim.

This document records decisions and their rationale, not the poster production itself.

## 2. Audience and Reading Path

The poster is read in two layers:

- **Visitor layer:** the title, the hero visual, and the main takeaway must be understandable at a glance.
- **Jury layer:** the methodological discipline, the experiment chain, and the claim boundary must be inspectable on closer reading.

The reading path is numbered and guided; a visitor should see where to start and where to go via visual cues. The structural backbone is the thesis's controlled evidence chain (Şekil 3.1, evidence ladder): problem and reference, schema separation, mechanism, encoder, robustness. This order makes the experiments read as an evidence flow rather than a list of single results.

## 3. Size and Physical Properties

- **Size:** 70×100 cm, portrait.
- **Color:** color; legible at viewing distance in print.
- **Approach:** visual-first. The 70×100 area is chosen because the content is dense; however, the space is used to visualize decisions and the data flow, not to fill the poster with text.

## 4. Language

The poster is written in **Turkish**. Technical terms are given with their Turkish equivalent at first use and then used consistently. Terminology is kept aligned with the current thesis: Graph A/B/C, target-observation, measured-only, Scheme A, AUPRC, operating point, and similar terms carry the same meaning as in the thesis.

## 5. Narrative Framing (Reframe)

The poster story is built on a **gap → contribution** arc. The title and main takeaway foreground the study's **contribution and methodological rigor**:

- the novel Graph C target-observation context representation,
- the Scheme A `cleavage_freq > 1e-5` binary-label decision and its rationale,
- the discipline of reducing 310142 rows to a measured-only universe of 25632 (label integrity + leakage control),
- the mechanism-isolated rare-negative recognition finding,
- the honestly bounded uncertainty from bootstrap and multi-seed analysis.

The study's headline is **not** `"XGBoost'u geçemedik"`. A strong tabular baseline remaining the AUPRC bar is a finding, and it appears only as a single honesty caveat inside the results — never as the poster's top message. This is not a marketing choice; it is the correct reflection of the two-axis narrative defined in the thesis `main_narrative_framing.md` (the ranking axis and the operating-point axis are different questions). The detailed story arc will be built in `poster_narrative_framing.md`.

## 6. Claim Boundary (Text and Figures)

The following limits are the poster's version of the claim boundary preserved throughout the thesis, and they apply to **both text and figures**:

- No robust AUPRC superiority over XGBoost F4 is claimed.
- Attention, gate, FiLM, embedding, and masking outputs are not evidence of biological causality.
- Threshold and rare-negative gains are presented as seed/guide-fragile.
- The 0.900705 positive prevalence is the no-skill PR baseline; it is not a performance floor.
- No "sequence models fail" generalization is made; the limit holds only under this evaluation contract and these architectural conditions.
- No misleading leaderboard against other papers' raw scores is constructed.

Figure note: arrows, encoder boxes, and transitions must not read as causal effects; the visual language also carries the boundary above.

## 7. Literature Positioning Decision

The literature comparison requested by the advisor is handled, without breaking the claim boundary, in an **A+B** form:

- **A — Contextual positioning:** state `"neden doğrudan kıyaslanamaz?"` explicitly (different data, split, negative sampling, and prevalence). Then present a qualitative positioning showing that the study stands stricter along the axes of leakage control, guide-disjoint evaluation, prevalence awareness, and the measured-only universe.
- **B — "Spiritual" comparison:** emphasize that the literature mostly targets the ranking/retrieval question while this study targets the binary measured-only question, and that these are different questions.

No side-by-side bar or table against other papers' AUPRC/scores is built; that would be misleading because of the different universes.

## 8. Simplification ("For Dummies") Principle

Simplification applies to both text and figures. Visual metaphors are encouraged; for example, `"aynı adres, farklı ziyaretler"` for Graph C, or `"doğrulanmamış ipuçları"` for `measured=0` rows. The only condition: a metaphor or simplification must **change no claim** and must not break the boundary in §6. Accessibility never overrides scientific accuracy; both are preserved together.

## 9. Visual / Figure Strategy

Priority is on the **core decision visuals**: visualizing the study's key decisions sits at the center of the poster.

- the Scheme A frequency threshold (`cleavage_freq > 1e-5`) and what it means,
- the 310142 → measured-only 25632 → test 1702 universe selection,
- the Graph A → Graph B → Graph C transition and the semantic difference between them,
- the feature families (sequence, energy, experimental epigenetic, computed nucleosome, missingness) and an introduction to the data produced over the graph.

Existing thesis figures (schema, encoder, pipeline, and result figures) are reused where useful, or recolored to the poster palette. Curation is **soft**: the must/should/cut distinction is guidance, not a rigid cap; the figure set may shift as content and copy evolve. The core decision visuals are the priority category intended to be preserved in all cases.

## 10. Aesthetic and Visual Language

Even in the draft stage, the poster targets a **modern, contemporary** visual language:

- strong typographic hierarchy (few, clear levels),
- generous negative space and low text density,
- a restrained but vivid accent palette,
- subtle gradients and soft/rounded forms,
- a single hero visual,
- a clean, contemporary product-communication (modern advertising) feel.

This approach is balanced with print and accessibility: sufficient contrast, colorblind-aware choices, and grayscale distinguishability where it matters.

The accent-color direction is left as "restrained-but-vivid" for now; the exact color(s) will be fixed in the figure phase. Relationship to the thesis BTÜ palette: the poster may evolve toward a brighter, softer modern palette, and existing thesis figures may be **recolored later** to match. This is an intentional and reversible design decision; the aesthetic never permits visual choices that distort the data (§6).

## 11. Production Format (Intentionally Non-Binding)

The production tool is **not** fixed as a design decision. What will be produced next is a **draft (working artifact)** to make sense of our content and layout thinking and to check fit; it is not the finished design. The final production tool (Canva, presentation software, LaTeX, etc.) is the team's decision. This document commits to **design principles** that carry to any tool, not to a toolchain.

## 12. Preserving the Headings

Even with a free layout, the section/heading set must remain identifiable. Per the department guidance, the poster's main sections (such as abstract / introduction and aim, method, results, discussion, references, and author information) must stay distinguishable even if the layout changes.

## 13. Out of Scope and Later Phases

This document does not cover the following; they are left to later phases:

- final design and printing,
- per-section poster copy (→ content plan),
- figure production and recoloring,
- writing and style rules (→ rules document).

## 14. Dependent Documents

Sibling poster documents to be written:

- `poster_narrative_framing.md` — story arc and claim boundary.
- `poster_yazim_kurallari.md` — writing/style and visual-style rules.
- `plan/poster_content_plan.md` — sliced, section-by-section content and figure plan.

Upstream sources:

- the current thesis (`docs/thesis/latex/btu_template/chapters/`),
- `docs/thesis/notes/main_narrative_framing.md`,
- `docs/literature/literature_index.md` and the literature axes,
- `AGENTS.md` and the project docs (`docs/PROJECT_CONTEXT.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/DECISIONS.md`).
