# Poster Narrative Framing

This document is the **narrative spine** of the CRISPR-Cas9 off-target prediction poster: the story arc, the claim guardrails, and the literature positioning. It is the poster-scale sibling of the thesis `docs/thesis/notes/main_narrative_framing.md`. It does not specify layout or full per-section copy (layout and copy belong to `plan/poster_content_plan.md`); it fixes *what* the poster says and *within which limits*, so every later phase inherits one consistent story.

Note on language: this planning document is written in **English**, while the **poster itself is in Turkish**. Where a phrase is intended to appear on the poster, it is quoted in Turkish.

Governing document: `docs/poster/notes/poster_design_decisions.md` (the constitution). This narrative obeys its reframe (§5), claim boundary (§6), literature decision (§7), and simplification principle (§8).

Upstream sources for every claim and number below: the current thesis chapters (`docs/thesis/latex/btu_template/chapters/`), `docs/thesis/notes/main_narrative_framing.md`, and `docs/literature/literature_index.md`. No number on the poster may originate here; each must trace to the thesis.

## 1. Purpose

This document gives the poster a single, confident, claim-safe narrative so that the title, the hero visual, the result panels, the literature panel, and the figures all tell the *same* story. It translates the thesis's controlled evidence chain into a five-beat arc a visitor can follow in about two minutes, and it records the guardrails (claim boundary, forbidden phrasings) that every beat and every figure must respect. It is the reference the content plan and the writing-rules document build on.

## 2. One-sentence poster thesis

The poster's core proposition, contribution-first and explicitly not a superiority claim (poster wording, Turkish):

> `"Sızıntı-kontrollü ve ölçülmüş (measured-only) veri sözleşmesi altında, hedef-gözlem bağlamını taşıyan özgün bir çizge temsili (Graph C) ve aile-duyarlı GATv2 kodlayıcı, nadir ölçülmüş negatifleri tanıma davranışını -mekanizması izole edilmiş ve dürüstçe sınırlanmış biçimde- güçlendirir."`

This sentence makes a contribution claim (a representation and an encoder, evaluated under a strict contract, that change behavior on the negative-sensitive axis). It does not claim ranking superiority over the tabular baseline.

## 3. Headline / hook candidates (TITLE DEFERRED)

> **`[BAŞLIK: TBD — placeholder]`** The final title is intentionally not chosen yet; it will be decided as the design progresses. The candidates below are placeholders, each claim-safe and contribution-first.

- Candidate A — `"CRISPR-CAS9 HEDEF DIŞI TAHMİNİNDE BAĞLAM DUYARLI ÇİZGE SİNİR AĞLARININ DEĞERLENDİRİLMESİ"`
  Rationale: matches the official thesis title from `docs/thesis/latex/btu_template/metadata.tex`.
- Candidate B — `"Sıralama mı, karar eşiği mi? Hedef dışı tahminde iki ayrı soru"`
  Rationale: foregrounds the two-axis insight, which is the poster's most defensible and most interesting claim.
- Candidate C — `"Doğrulanmış veri, sızıntısız değerlendirme: bağlam-duyarlı GNN'lerin nadir negatif katkısı"`
  Rationale: foregrounds methodological rigor plus the located finding.

**Single honesty caveat** (lives *inside* the results, never as the headline; poster wording, Turkish):

> `"Sıralamada (AUPRC) güçlü tablo temelli referans XGBoost F4 en sağlam çıta olarak kaldı; bağlamın katkısı sıralamada değil, karar eşiğinde ortaya çıkıyor."`

## 4. Story arc (gap → contribution)

Five beats, mapped onto the thesis's controlled evidence chain (Şekil 3.1, evidence ladder). Each beat gives the one-line poster message and the thesis source that backs it.

1. **Problem / hook.** Message (Turkish, for-dummies spirit): off-target (hedef dışı) cleavage is the safety bottleneck of CRISPR-Cas9; the same guide can behave differently depending on context. Source: thesis 1.1.
2. **The gap.** Message: prior work often mixes leakage-prone splits, inconsistent negatives, and prevalence-blind metrics; what is missing is a controlled, leakage-aware, measured-only evaluation of context-aware graph models. Source: thesis 1.2.6 (gap synthesis), literature axes.
3. **Approach / novelty.** Message: a strict evaluation contract (measured-only universe + guide-disjoint split + no test-tuning) plus three representations (Graph A → B → C) and a family-aware GATv2 encoder. Source: thesis 2.x (data contract, Scheme A label, Graph schemas), 2.9 (encoder).
4. **Findings on two axes.** Message: on the ranking axis the strong tabular baseline holds; on the operating-point axis context-aware models recognize more of the rare measured negatives — and we say why this split is expected (§6). Source: thesis 3.x result tables/figures, Şekil 3.6 (PR).
5. **Contribution / takeaway + future work.** Message: the contribution is a controlled, honest framework that *locates* where context helps and bounds its fragility; future work targets the negative scarcity directly. Source: thesis 3.7 (robustness), 4 (sonuç ve öneriler).

## 5. The two-axis result story

The results live on two distinct axes, presented so that neither is overclaimed:

- **Ranking axis (AUPRC).** XGBoost F4 remains the strongest AUPRC bar. This is stated confidently as a *finding*, not an apology. The best single GNN (S8B_R2, sequence+context fusion) comes very close, and paired intervals are compatible with no difference — which is reported as *compatibility*, not equivalence.
- **Operating-point axis (MCC / specificity at a fixed threshold).** Context-aware architectures (Graph C GATv2, the family-aware encoder) recognize more of the scarce measured negatives than the GCN baseline and than the tabular baseline at the operating point. This is presented *with* its seed/guide fragility, never as robust superiority.

Presentation rule: the two axes appear as two clearly separated panels (or two clearly separated rows of one panel), each labeled with the question it answers (`"Sıralama"` vs `"Karar eşiği"`), so a viewer never reads an operating-point gain as a ranking win.

## 6. Why the axes diverge (the nature of the problem)

This section explains, claim-safely, *why* the gap nearly closes on AUPRC yet appears larger on MCC/specificity. The divergence is structural and expected given the dataset, not a contradiction.

- **AUPRC saturates and is insensitive here.** The test universe is highly imbalanced toward positives (prevalence 0.900705; 1533 positives vs 169 negatives). Ranking quality is dominated by ordering the abundant positives, which strong models do near-perfectly, so AUPRC sits near a ceiling (~0.99) for many models at once. The remaining headroom is tiny, and differences fall inside bootstrap intervals. A strong tabular model like XGBoost F4 occupies this ceiling, leaving little room to separate models *on this axis*.
- **MCC and specificity are governed by the scarce negatives, so they have high leverage.** At a fixed decision threshold, specificity is exactly `TN / (TN + FP)`, computed over only the 169 negatives, which are concentrated in 9 guides (80 of them in guide 9251). MCC is a balanced coefficient that the minority class dominates. With so few negatives, correctly flagging a handful more of them moves these metrics substantially. The negative-recognition counts make this concrete: true-negatives at the operating point run XGBoost F4 = 40, Graph C GCN = 14, Graph C GATv2 = 63, family-aware encoder = 110 — large relative swings over a pool of 169 that barely move AUPRC.
- **Context lands precisely on the negative-sensitive axis.** The contribution of context-aware representations shows up where the metric is sensitive to rare negatives (operating point), not where it is saturated and majority-dominated (ranking). This is the central, defensible insight, and it is *why* the same models look "tied" on AUPRC and "ahead" on MCC/specificity.
- **The honest twin.** The very scarcity that makes MCC/specificity sensitive also makes the gains high-variance: across seeds and guide clusters the operating-point deltas carry wide intervals, because they hinge on ~169 negatives bunched into a few guides. The poster reports the divergence and its fragility together; one explains the other.

Poster framing (Turkish, one line): `"Sıralama bol sayıda pozitifle doyuma ulaşır; karar eşiği ise az sayıdaki negatife bakar -işte fark da, kırılganlık da buradan gelir."`

## 7. Hypotheses, poster-sized

Compact, claim-safe restatement of the thesis hypotheses (thesis 1.3):

- **H1 — Schema.** Carrying target-observation context in the graph (Graph C) changes model behavior versus context-free schemas (Graph A/B).
- **H2 — Encoder.** A family-aware encoder that respects feature families (sequence, energy, experimental-epigenetic, computed-nucleosome, missingness) affects how rare measured negatives are recognized.
- **H3 — Axis separation.** Under a strict measured-only, guide-disjoint contract, ranking (AUPRC) and operating-point (MCC/specificity) behave as different questions; an effect on one need not appear on the other.

Each is stated as a question the experiments examine, not a settled superiority result.

## 8. Contributions ("ne katıyoruz")

Confident, poster-sized contribution bullets:

- **Controlled, leakage-aware evaluation.** A guide-disjoint split with no test-tuning, so reported behavior is not inflated by leakage.
- **Label-integrity / measured-only discipline.** Reducing 310142 rows to a 25632-row measured-only universe, keeping only observations with measured outcomes (`measured=1`) rather than treating unverified rows as negatives.
- **Three graph representations with explicit semantics.** Graph A → B → C, where Graph C introduces target-observation nodes (the same target carries distinct observation contexts).
- **Mechanism isolation.** Locating the operating-point gain in the experimental-epigenetic context rather than asserting it globally.
- **Family-aware encoder.** An encoder that treats feature families structurally rather than as a flat vector.
- **Honest boundary.** A bootstrap and multi-seed robustness analysis (Sprint 9) that bounds the fragility of the gains instead of hiding it.

## 9. Literature positioning panels (A+B)

Handled per constitution §7, without breaking the claim boundary. No side-by-side raw-score leaderboard against other papers is built — different universes (data, split, negative sampling, prevalence) make raw AUPRC/score comparison misleading.

- **Panel A — Contextual positioning.** State the question explicitly (Turkish): `"neden doğrudan kıyaslanamaz?"` Then position the study *qualitatively* along four axes where it stands stricter: leakage control, guide-disjoint evaluation, prevalence awareness, and the measured-only universe. The output is a qualitative positioning (e.g., a small axis/criteria diagram), not a performance ranking.
- **Panel B — "Spiritual" comparison.** Emphasize that much of the literature targets the ranking/retrieval question, whereas this study targets the binary measured-only question; these are different questions. Turkish framing: the comparison is `"hem ruhsal hem metriksel"` in spirit but never a numeric leaderboard.

## 10. Claim-safe numeric anchors

The pool of numbers permitted on the poster. Every value must be verified against the current thesis before use; none may be invented or rounded in a way that changes meaning. Which subset actually appears is deferred to the content plan (§13).

- **Universe / split.** 310142 rows → measured-only 25632 → test 1702. Test: 29 guides, 1533 positives, 169 negatives. Negatives concentrated in 9 guides; guide 9251 carries 80 of 169.
- **Prevalence.** 0.900705 — the no-skill PR baseline (a reference line, not a performance floor).
- **Ranking bar.** XGBoost F4 AUPRC regenerated 0.992338 (guide-cluster interval [0.950179, 0.999336]; multi-seed 0.990649 ± 0.001944); historical 0.992522, with a version-drift note explaining the small difference.
- **Best single GNN.** S8B_R2 (sequence+context fusion) AUPRC 0.986020 (interval [0.929981, 0.998966]; multi-seed 0.978963 ± 0.011322).
- **Operating-point (always with the seed/guide-fragile caveat).** Negative recognition at the operating point — true negatives over 169: XGBoost F4 = 40, Graph C GCN = 14, Graph C GATv2 = 63, family-aware encoder = 110; corresponding specificities 0.236, 0.083, 0.373, 0.651. Family-aware encoder MCC 0.603489.
- **Feature families.** 6 experimental-epigenetic + 13 computed-nucleosome + 5 binding-energy features; 154 sgRNA, 138747 target locations.

## 11. Claim boundary and forbidden phrasings

The boundary applies to **text and figures** alike (constitution §6).

Boundary:
- No robust AUPRC superiority over XGBoost F4 is claimed.
- Attention, gate, FiLM, embedding, and masking outputs are not biological-causality evidence.
- Threshold and rare-negative gains are seed/guide-fragile.
- 0.900705 prevalence is the no-skill PR baseline, not a floor.
- No "sequence models fail" generalization; the limit holds only under this contract and these conditions.
- No misleading raw-score leaderboard against other papers.

Do **not** say / do **not** draw:
- "We beat XGBoost" / `"XGBoost'u geçtik"` as a headline or a robust result.
- "Equivalent to XGBoost" — say *compatible with no difference* where intervals include zero.
- "Context causes / proves biologically …" — say *changes behavior under this contract*.
- A bar chart or table ranking this study's score against other papers' scores.
- Arrows or encoder boxes drawn so they read as causal effects.
- Any number not traceable to the current thesis.

## 12. Mapping to thesis sources

| Narrative beat | Thesis source |
| --- | --- |
| Problem / hook | 1.1 |
| Gap | 1.2.6; literature axes (`literature_index.md`) |
| Hypotheses | 1.3 |
| Data contract, Scheme A label, measured-only | 2.x (data/label sections) |
| Graph A/B/C schemas | 2.x (schema sections) |
| Family-aware encoder | 2.9 |
| Evidence-chain backbone | Şekil 3.1 |
| Ranking axis (AUPRC) | 3.x AUPRC tables; XGBoost F4 reference |
| Operating-point axis (MCC/specificity) | 3.5 (Graph C), 3.8 (family-aware encoder) |
| Axis divergence rationale | prevalence/no-skill discussion; 3.x operating-point tables |
| PR / no-skill baseline | Şekil 3.6 |
| Robustness / fragility | 3.7 (bootstrap, multi-seed) |
| Contribution / future work | 4 (sonuç ve öneriler) |

(Exact subsection and float numbers are read from the current chapters when the content plan is written.)

## 13. Open questions (deferred)

- Final title (placeholder in §3) — to be chosen as design progresses.
- The hero visual — which single image leads the poster.
- Which numeric anchors from §10 actually make the cut (→ content plan / figure phase).
- Exact form of Panel A's qualitative positioning diagram.
- Whether the axis-divergence explanation (§6) is its own panel or folded into the results panel.
