"""Reproducible thesis result-figure generator.

This is an ADDITIVE, standalone tool. It reads the canonical result CSVs that
already back the chapter-3 tables and re-draws the eight thesis result figures
in a clean, publication style (no embedded titles, Turkish labels, no run-id /
path strings, colorblind-safe + grayscale-distinguishable, >=300 dpi). Output
filenames match exactly what ``chapters/03_deneysel_calismalar.tex`` references,
so no LaTeX change is required.

It does NOT re-run models, read checkpoints, or invent values. PR curves are
computed from stored per-row label/score (plotting, not metric invention). All
plotted numbers come from repo CSVs and are cross-checked against the known
chapter-3 anchors.

Usage::

    uv run python scripts/build_thesis_figures.py --check   # dry-run report
    uv run python scripts/build_thesis_figures.py           # overwrite figures
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
DEFAULT_OUT = REPO / "docs" / "thesis" / "latex" / "btu_template" / "figures" / "results"

PREVALENCE = 0.900705
F4_HISTORICAL_AUPRC = 0.992522  # Sprint 2 historical XGBoost F4 (matches non-graph table)

# Consistent BTU thesis palette (identical to the TikZ schematic figures in
# main.tex) so every figure shares one visual language.
BTU = {
    "ink": "#1F2933",
    "blue": "#1F4E79",
    "teal": "#1B7F79",
    "orange": "#B7651B",
    "green": "#4B7F2A",
    "gray": "#6B7280",
}
GRID = "#D9DCE1"
PRIMARY = BTU["blue"]          # single-series AUPRC bars
NOSKILL_COLOR = BTU["gray"]    # no-skill PR baseline line
F4_COLOR = BTU["orange"]       # XGBoost F4 reference line
ZERO_COLOR = BTU["orange"]     # paired-difference zero line
FOREST_POINT = BTU["blue"]
FOREST_ERR = BTU["gray"]

# Three threshold metrics share ONE consistent color set across 3.4/3.5/3.7.
# Solid fills, thin white separators, no hatch.
METRIC_STYLE = [
    ("test_mcc", "MCC", BTU["blue"]),
    ("test_specificity", "Özgüllük", BTU["orange"]),
    ("test_macro_f1", "Makro-F1", BTU["green"]),
]

# PR curves: distinct color + linestyle (still separable in grayscale).
CURVE_STYLE = {
    "S8B_R1_sequence_only": (BTU["orange"], "--"),
    "S8B_R2_sequence_plus_context": (BTU["blue"], "-"),
}

# --- id -> Turkish display label maps (must match chapter-3 table wording) ---
LABELS_GCN_SCHEMA = {
    "gcn_graph_a": "Graph A GCN",
    "gcn_graph_b": "Graph B GCN",
    "gcn_graph_c": "Graph C GCN",
    "gcn_graph_c_sprint5b_energy": "Graph C GCN, enerji",
}
LABELS_MECHANISM = {
    "S7D_REF_FULL_GRAPH_C_GATV2": "Tam Graph C GATv2",
    "S7D_R1_no_context_edges": "Context-similarity kenarları kaldırılmış",
    "S7D_R2_edge_blind_attention": "Edge-blind attention",
    "S7D_R3_mask_target_context_features": "Target context feature'ları maskelenmiş",
}
LABELS_SUBGROUP = {
    "S7E_REF_NO_CONTEXT_EDGE_GATV2": "Context-edge kaldırılmış referans",
    "S7E_R1_mask_target_sequence": "Hedef dizisi maskelenmiş",
    "S7E_R2_mask_experimental_epigenetic": "Deneysel epigenetik özellikler maskelenmiş",
    "S7E_R3_mask_computed_nucleosome_aggregates": "Hesaplanmış nükleozom özetleri maskelenmiş",
    "S7E_R4_mask_computed_nucleosome_missingness": "Missingness göstergeleri maskelenmiş",
    "S7E_R5_mask_all_nonsequence_context": "Tüm non-sequence context maskelenmiş",
}
LABELS_SEQCTX = {
    "S8B_R0_reference": "Context-edge FiLM (referans)",
    "S8B_R1_sequence_only": "Dizi-only",
    "S8B_R2_sequence_plus_context": "Dizi + bağlam birleşimi",
}
LABELS_FOREST = {
    "XGB_F4": "XGBoost F4 (yeniden üretim)",
    "S8B_R2": "Dizi + bağlam birleşimi",
    "S7F_R3": "Family-aware deneysel vurgu",
    "S8A_R2": "Context-edge FiLM",
    "S7F_R2": "Family-aware encoder",
}
LABELS_PAIR_SHORT = {
    "S8B_R2": "Dizi+bağlam",
    "S8A_R2": "Context FiLM",
    "S7F_R3": "Family deneysel",
    "S7F_R2": "Family encoder",
    "XGB_F4": "XGBoost F4",
}

NO_SKILL_LABEL = f"no-skill PR taban (prevalans={PREVALENCE:.3f})"

_warnings: list[str] = []


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Tinos", "TeX Gyre Termes", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": BTU["gray"],
            "axes.linewidth": 0.8,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _load(rel: str, required: list[str]) -> pd.DataFrame:
    path = OUTPUTS / rel
    if not path.exists():
        raise FileNotFoundError(f"source CSV missing: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns {missing}; found {list(df.columns)[:40]}")
    return df


def _check(value: float, expected: float, what: str, tol: float = 2e-3) -> None:
    if abs(float(value) - expected) > tol:
        _warnings.append(f"value drift: {what} plotted={value:.6f} expected≈{expected:.6f}")


def _wrap(text: str, width: int = 18) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _grouped_barh(ax, variant_labels, values_by_metric) -> None:
    """Horizontal grouped bars with value labels; legend outside (no overlap)."""
    n = len(variant_labels)
    y = range(n)
    bar_h = 0.26
    offsets = [bar_h, 0.0, -bar_h]
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    all_vals: list[float] = []
    for (col, mlabel, color), off in zip(METRIC_STYLE, offsets):
        vals = [float(v) for v in values_by_metric[col]]
        all_vals += vals
        ypos = [yy + off for yy in y]
        ax.barh(ypos, vals, height=bar_h, color=color, edgecolor="white",
                linewidth=0.6, label=mlabel)
        for yy, v in zip(ypos, vals):
            ax.annotate(f"{v:.3f}", (v, yy),
                        xytext=(3 if v >= 0 else -3, 0), textcoords="offset points",
                        va="center", ha="left" if v >= 0 else "right",
                        fontsize=7, color=BTU["ink"])
    ax.set_yticks(list(y))
    ax.set_yticklabels([_wrap(v, 22) for v in variant_labels])
    ax.invert_yaxis()
    ax.set_xlabel("Metrik değeri")
    lo = min(0.0, min(all_vals))
    ax.set_xlim(lo - 0.03, 1.12)
    if lo < 0:
        ax.axvline(0.0, color=BTU["gray"], linewidth=0.6)
    # Legend outside the plot area so it never overlaps the bars.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)


def _grouped_barv(ax, variant_labels, values_by_metric) -> None:
    """Vertical grouped bars for short label sets."""
    n = len(variant_labels)
    x = range(n)
    bar_w = 0.26
    offsets = [-bar_w, 0.0, bar_w]
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for (col, mlabel, color), off in zip(METRIC_STYLE, offsets):
        vals = values_by_metric[col]
        ax.bar(
            [xx + off for xx in x], vals, width=bar_w, color=color,
            edgecolor="white", linewidth=0.6, label=mlabel,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([_wrap(v, 16) for v in variant_labels])
    ax.set_ylabel("Metrik değeri")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor=GRID)


def _forest(ax, labels, points, los, his, *, xlabel, zero_line=False) -> None:
    y = range(len(labels))
    lo_err = [p - lo for p, lo in zip(points, los)]
    hi_err = [hi - p for p, hi in zip(points, his)]
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.errorbar(
        points, list(y), xerr=[lo_err, hi_err], fmt="o", color=FOREST_POINT,
        ecolor=FOREST_ERR, elinewidth=1.4, capsize=3, markersize=5,
        markeredgecolor="white", markeredgewidth=0.6,
    )
    if zero_line:
        ax.axvline(0.0, color=ZERO_COLOR, linewidth=1.1, linestyle="--")
    for yy, p in zip(y, points):
        ax.annotate(f"{p:+.3f}" if zero_line else f"{p:.3f}", (p, yy),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels([_wrap(v, 26) for v in labels])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)


# --------------------------------------------------------------------------- #
# Per-figure builders. Each returns (filename, [source_rel_paths], labels_used).
# --------------------------------------------------------------------------- #
def fig_32(out: Path):
    df = _load("sprint2/baseline_results.csv",
               ["model_name", "feature_set", "test_auprc", "test_positive_rate"])
    rows = df[df["model_name"] == "xgboost_unweighted"].copy()
    order = ["F1", "F2", "F3", "F4"]
    rows = rows[rows["feature_set"].isin(order)].set_index("feature_set").loc[order]
    vals = rows["test_auprc"].astype(float).tolist()
    _check(vals[-1], F4_HISTORICAL_AUPRC, "3.2 XGBoost F4")
    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.bar(order, vals, color=PRIMARY, edgecolor="white", linewidth=0.6, width=0.6)
    ax.axhline(PREVALENCE, color=NOSKILL_COLOR, linestyle="--", linewidth=1.0, label=NO_SKILL_LABEL)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8)
    ax.set_ylabel("AUPRC")
    ax.set_xlabel("Özellik seti")
    ax.set_ylim(0.80, 1.01)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.savefig(out / "03_non_graph_xgboost_feature_auprc.png")
    plt.close(fig)
    return "03_non_graph_xgboost_feature_auprc.png", ["sprint2/baseline_results.csv"], order


def fig_33(out: Path):
    a = _load("sprint4/gcn_sprint4_comparison_results.csv", ["model_name", "test_auprc"])
    b = _load("sprint5b/graph_c_context_observation/gcn_graph_c_results.csv",
              ["model_name", "test_auprc"])
    order = ["gcn_graph_a", "gcn_graph_b", "gcn_graph_c", "gcn_graph_c_sprint5b_energy"]
    auprc = {}
    for mid in order[:3]:
        auprc[mid] = float(a.loc[a["model_name"] == mid, "test_auprc"].iloc[0])
    auprc["gcn_graph_c_sprint5b_energy"] = float(
        b.loc[b["model_name"] == "gcn_graph_c_sprint5b_energy", "test_auprc"].iloc[0])
    _check(auprc["gcn_graph_c_sprint5b_energy"], 0.972481, "3.3 Graph C enerji")
    labels = [LABELS_GCN_SCHEMA[m] for m in order]
    vals = [auprc[m] for m in order]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    xpos = list(range(len(labels)))
    ax.bar(xpos, vals, color=PRIMARY, edgecolor="white", linewidth=0.6, width=0.6)
    ax.axhline(F4_HISTORICAL_AUPRC, color=F4_COLOR, linestyle="--",
               linewidth=1.1, label="XGBoost F4")
    ax.axhline(PREVALENCE, color=NOSKILL_COLOR, linestyle=":", linewidth=1.1, label=NO_SKILL_LABEL)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([_wrap(v, 14) for v in labels])
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0.90, 1.005)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.savefig(out / "03_gcn_graph_schema_auprc.png")
    plt.close(fig)
    return ("03_gcn_graph_schema_auprc.png",
            ["sprint4/gcn_sprint4_comparison_results.csv",
             "sprint5b/graph_c_context_observation/gcn_graph_c_results.csv"], labels)


def _threshold_fig(out: Path, *, rel, ids, label_map, fname, horizontal, figsize, required_id_col):
    df = _load(rel, [required_id_col, "test_mcc", "test_specificity", "test_macro_f1"])
    df = df.set_index(required_id_col)
    labels = [label_map[i] for i in ids]
    values_by_metric = {
        col: [float(df.loc[i, col]) for i in ids] for col, *_ in METRIC_STYLE
    }
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        _grouped_barh(ax, labels, values_by_metric)
    else:
        _grouped_barv(ax, labels, values_by_metric)
    fig.savefig(out / fname)
    plt.close(fig)
    return fname, [rel], labels


def fig_34(out: Path):
    return _threshold_fig(
        out, rel="sprint7d/graphc_gatv2_mechanism_ablation.csv",
        ids=list(LABELS_MECHANISM), label_map=LABELS_MECHANISM,
        fname="03_graphc_mechanism_threshold_metrics.png",
        horizontal=True, figsize=(5.6, 3.6), required_id_col="predeclared_run_id")


def fig_35(out: Path):
    return _threshold_fig(
        out, rel="sprint7e/target_context_subgroup_ablation.csv",
        ids=list(LABELS_SUBGROUP), label_map=LABELS_SUBGROUP,
        fname="03_target_context_subgroup_threshold_metrics.png",
        horizontal=True, figsize=(5.8, 4.4), required_id_col="predeclared_run_id")


def fig_37(out: Path):
    return _threshold_fig(
        out, rel="sprint8b/sequence_context_comparison.csv",
        ids=list(LABELS_SEQCTX), label_map=LABELS_SEQCTX,
        fname="03_sequence_context_threshold_metrics.png",
        horizontal=True, figsize=(5.6, 3.2), required_id_col="predeclared_run_id")


def fig_36(out: Path):
    preds = _load("sprint8b/diagnostics/sequence_context_predictions.csv",
                  ["predeclared_run_id", "split", "label", "score"])
    comp = _load("sprint8b/sequence_context_comparison.csv",
                 ["predeclared_run_id", "test_auprc"]).set_index("predeclared_run_id")
    preds = preds[preds["split"] == "test"]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for rid in ("S8B_R1_sequence_only", "S8B_R2_sequence_plus_context"):
        g = preds[preds["predeclared_run_id"] == rid]
        precision, recall, _ = precision_recall_curve(g["label"], g["score"])
        ap = float(comp.loc[rid, "test_auprc"])
        color, ls = CURVE_STYLE[rid]
        ax.plot(recall, precision, color=color, linestyle=ls, linewidth=1.8,
                label=f"{LABELS_SEQCTX[rid]} (AUPRC={ap:.3f})")
    _check(float(comp.loc["S8B_R2_sequence_plus_context", "test_auprc"]), 0.986020, "3.6 S8B_R2")
    ax.axhline(PREVALENCE, color=NOSKILL_COLOR, linestyle=":", linewidth=1.1, label=NO_SKILL_LABEL)
    ax.set_xlabel("Duyarlılık (recall)")
    ax.set_ylabel("Kesinlik (precision)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", framealpha=0.9)
    fig.savefig(out / "03_sequence_context_pr_curves.png")
    plt.close(fig)
    return ("03_sequence_context_pr_curves.png",
            ["sprint8b/diagnostics/sequence_context_predictions.csv",
             "sprint8b/sequence_context_comparison.csv"],
            [LABELS_SEQCTX["S8B_R1_sequence_only"], LABELS_SEQCTX["S8B_R2_sequence_plus_context"]])


def fig_38(out: Path):
    df = _load("sprint9/robustness_bootstrap_cis.csv",
               ["registry_id", "metric", "point", "percentile_lo", "percentile_hi"])
    df = df[df["metric"] == "auprc"]
    df = df[df["registry_id"].isin(LABELS_FOREST)].copy()
    df = df.sort_values("point", ascending=False)
    labels = [LABELS_FOREST[r] for r in df["registry_id"]]
    pts = df["point"].astype(float).tolist()
    _check(pts[0], 0.992338, "3.8 top (XGBoost F4)")
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    _forest(ax, labels, pts, df["percentile_lo"].astype(float).tolist(),
            df["percentile_hi"].astype(float).tolist(),
            xlabel="AUPRC ve %95 guide-cluster aralığı")
    ax.set_xlim(0.89, 1.005)
    fig.savefig(out / "03_saglamlik_auprc_araliklari.png")
    plt.close(fig)
    return "03_saglamlik_auprc_araliklari.png", ["sprint9/robustness_bootstrap_cis.csv"], labels


def fig_39(out: Path):
    df = _load("sprint9/robustness_paired_differences.csv",
               ["a_id", "b_id", "metric", "point_delta", "percentile_lo", "percentile_hi"])
    df = df[df["metric"] == "auprc"].copy()
    labels = [f"{LABELS_PAIR_SHORT[a]} − {LABELS_PAIR_SHORT[b]}"
              for a, b in zip(df["a_id"], df["b_id"])]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    _forest(ax, labels, df["point_delta"].astype(float).tolist(),
            df["percentile_lo"].astype(float).tolist(),
            df["percentile_hi"].astype(float).tolist(),
            xlabel="AUPRC farkı (model A − model B)", zero_line=True)
    fig.savefig(out / "03_paired_difference_distributions.png")
    plt.close(fig)
    return ("03_paired_difference_distributions.png",
            ["sprint9/robustness_paired_differences.csv"], labels)


BUILDERS = [fig_32, fig_33, fig_34, fig_35, fig_36, fig_37, fig_38, fig_39]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true",
                        help="build into a temp dir and report intended overwrites")
    args = parser.parse_args()

    _setup_style()
    target = args.out
    out = Path(tempfile.mkdtemp(prefix="thesis_fig_")) if args.check else target
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"git_commit": _git_commit(),
                "matplotlib": matplotlib.__version__,
                "figures": {}}
    for build in BUILDERS:
        fname, sources, labels = build(out)
        manifest["figures"][fname] = {"sources": sources, "labels": labels}
        existed = (target / fname).exists()
        print(f"[{'check' if args.check else 'write'}] {fname}  "
              f"({'overwrite' if existed else 'new'})  <- {', '.join(sources)}")

    if args.check:
        print("\n--check: nothing written to the tracked figures dir.")
        print(f"temp dir: {out}")
    else:
        (target / "thesis_figures_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {len(BUILDERS)} figures + manifest to {target}")

    if _warnings:
        print("\nWARNINGS (figure<->table value drift):")
        for w in _warnings:
            print("  -", w)
        return 1
    print("\nfigure<->table value cross-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
