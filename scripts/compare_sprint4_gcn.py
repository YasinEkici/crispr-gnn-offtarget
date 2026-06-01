"""Build consolidated Sprint 4 Graph A/C comparison artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay


GRAPH_A_RUN_ID = "sprint4_graph_a_gcn_seed42_20260601"
GRAPH_C_RUN_ID = "sprint4_graph_c_gcn_seed42_20260601"
F4_BASELINE_AUPRC = 0.992522


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/sprint4", help="Sprint 4 output root.")
    args = parser.parse_args()
    write_sprint4_comparison(Path(args.output_root))


def write_sprint4_comparison(output_root: Path) -> None:
    graph_a_dir = output_root / "graph_a"
    graph_c_dir = output_root / "graph_c"
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    _validate_required_artifacts(graph_a_dir, graph_c_dir)
    results = _load_and_validate_results(graph_a_dir, graph_c_dir)
    _validate_graph_c_provenance(graph_c_dir / GRAPH_C_RUN_ID / "graph_artifact_provenance.json")

    comparison = _comparison_frame(results)
    comparison.to_csv(output_root / "gcn_sprint4_comparison_results.csv", index=False)
    _write_auprc_comparison(comparison, figure_dir / "gcn_sprint4_schema_auprc_comparison.png")
    _write_pr_comparison(graph_a_dir, graph_c_dir, figure_dir / "gcn_sprint4_pr_curves.png")
    _write_report(comparison, output_root / "gcn_sprint4_comparison_report.md")


def _validate_required_artifacts(graph_a_dir: Path, graph_c_dir: Path) -> None:
    required = [
        graph_a_dir / "gcn_graph_a_results.csv",
        graph_a_dir / "gcn_graph_a_report.md",
        graph_a_dir / "diagnostics" / "gcn_graph_a_predictions.csv",
        graph_a_dir / GRAPH_A_RUN_ID / "resolved_config.yaml",
        graph_a_dir / GRAPH_A_RUN_ID / "graph_artifact_provenance.json",
        graph_c_dir / "gcn_graph_c_results.csv",
        graph_c_dir / "gcn_graph_c_report.md",
        graph_c_dir / "diagnostics" / "gcn_graph_c_predictions.csv",
        graph_c_dir / GRAPH_C_RUN_ID / "resolved_config.yaml",
        graph_c_dir / GRAPH_C_RUN_ID / "graph_artifact_provenance.json",
        graph_c_dir / GRAPH_C_RUN_ID / "runtime.json",
    ]
    required.extend(graph_a_dir / "figures" / name for name in _required_figure_names("graph_a"))
    required.extend(graph_c_dir / "figures" / name for name in _required_figure_names("graph_c"))
    required.extend(graph_a_dir / "diagnostics" / name for name in _required_diagnostic_names("graph_a"))
    required.extend(graph_c_dir / "diagnostics" / name for name in _required_diagnostic_names("graph_c"))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Sprint 4 artifacts: {missing}")


def _required_figure_names(schema_label: str) -> list[str]:
    return [
        f"gcn_{schema_label}_graph_schema_auprc_comparison.png",
        f"gcn_{schema_label}_pr_curves.png",
        f"gcn_{schema_label}_roc_curves.png",
        f"gcn_{schema_label}_training_curves.png",
        f"gcn_{schema_label}_score_distributions.png",
        f"gcn_{schema_label}_confusion_matrices.png",
        f"gcn_{schema_label}_decile_lift.png",
        f"gcn_{schema_label}_per_genome_metrics.png",
        f"gcn_{schema_label}_view_sanity_example.png",
    ]


def _required_diagnostic_names(schema_label: str) -> list[str]:
    return [
        f"gcn_{schema_label}_predictions.csv",
        f"gcn_{schema_label}_training_history.csv",
        f"gcn_{schema_label}_score_direction.csv",
        f"gcn_{schema_label}_fixed_threshold_metrics.csv",
        f"gcn_{schema_label}_score_deciles.csv",
        f"gcn_{schema_label}_per_genome_metrics.csv",
        f"gcn_{schema_label}_test_per_guide_metrics.csv",
    ]


def _load_and_validate_results(graph_a_dir: Path, graph_c_dir: Path) -> pd.DataFrame:
    results = pd.concat(
        [
            pd.read_csv(graph_a_dir / "gcn_graph_a_results.csv"),
            pd.read_csv(graph_c_dir / "gcn_graph_c_results.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    expected_values = {
        "label_scheme": "scheme_a",
        "split_id": "sprint2_main_seed42",
        "visibility_policy": "strict_inductive_primary",
        "threshold_selection_split": "validation",
        "checkpoint_selection_split": "validation",
    }
    for column, expected in expected_values.items():
        values = set(results[column].dropna().astype(str))
        if values != {expected}:
            raise ValueError(f"Unexpected {column}: {values}")
    schemas = set(results["graph_schema"].astype(str))
    expected_schemas = {"graph_a_minimal_physical_target", "graph_c_context_observation"}
    if schemas != expected_schemas:
        raise ValueError(f"Unexpected graph schemas: {schemas}")
    baseline_auprc = float(results["baseline_test_auprc"].iloc[0])
    if not math.isclose(baseline_auprc, F4_BASELINE_AUPRC, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Unexpected baseline AUPRC: {baseline_auprc}")
    return results


def _validate_graph_c_provenance(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    graph_c = provenance["schemas"]["graph_c_context_observation"]
    if graph_c["nodes"] != {"sgRNA": 150, "target_observation": 11446}:
        raise ValueError(f"Unexpected Graph C node counts: {graph_c['nodes']}")
    if graph_c["relations"] != {"candidate_pair": 11446, "context_similar_to": 91754}:
        raise ValueError(f"Unexpected Graph C relation counts: {graph_c['relations']}")
    if graph_c["split_id"] != "sprint2_main_seed42" or graph_c["label_scheme"] != "scheme_a":
        raise ValueError("Graph C provenance contract mismatch")


def _comparison_frame(results: pd.DataFrame) -> pd.DataFrame:
    summary_columns = [
        "sprint",
        "label_scheme",
        "split_id",
        "seed",
        "model_name",
        "feature_set",
        "graph_schema",
        "visibility_policy",
        "target_node_representation",
        "target_semantics",
        "loss",
        "checkpoint_selection_split",
        "threshold_selection_split",
        "threshold",
        "best_epoch",
        "epochs_ran",
        "best_val_auprc",
        "baseline_reference",
        "baseline_test_auprc",
        "baseline_test_auroc",
        "baseline_test_mcc",
        "test_rows",
        "test_positive_rate",
        "test_auprc",
        "test_auroc",
        "test_f1",
        "test_mcc",
        "notes",
    ]
    for column in summary_columns:
        if column not in results.columns:
            results[column] = pd.NA
    return results[summary_columns].copy()


def _write_auprc_comparison(comparison: pd.DataFrame, path: Path) -> None:
    rows = comparison.sort_values("graph_schema").copy()
    labels = ["xgboost_unweighted / F4", *rows["model_name"].tolist()]
    baseline_auprc = float(rows["baseline_test_auprc"].iloc[0])
    values = [baseline_auprc, *rows["test_auprc"].astype(float).tolist()]
    prevalence = float(rows["test_positive_rate"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color=["#2b6cb0", "#4a5568", "#805ad5"][: len(values)])
    ax.axhline(prevalence, color="#444444", linestyle="--", linewidth=1.2, label=f"test prevalence={prevalence:.3f}")
    ax.set_ylabel("Test AUPRC")
    ax.set_title("Sprint 4 Graph A/C same-contract AUPRC comparison")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_pr_comparison(graph_a_dir: Path, graph_c_dir: Path, path: Path) -> None:
    predictions = pd.concat(
        [
            pd.read_csv(graph_a_dir / "diagnostics" / "gcn_graph_a_predictions.csv"),
            pd.read_csv(graph_c_dir / "diagnostics" / "gcn_graph_c_predictions.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    required_columns = {"graph_schema", "model_name", "split", "label", "score"}
    missing = sorted(required_columns.difference(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    test = predictions.loc[predictions["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    for (model_name, graph_schema), part in test.groupby(["model_name", "graph_schema"], sort=True):
        PrecisionRecallDisplay.from_predictions(
            part["label"].to_numpy(dtype=int),
            part["score"].to_numpy(dtype=float),
            name=f"{model_name} / {graph_schema}",
            ax=ax,
            plot_chance_level=False,
        )
    prevalence = float(test["label"].mean())
    ax.axhline(prevalence, color="#444444", linestyle="--", linewidth=1, label=f"test prevalence={prevalence:.3f}")
    ax.set_title("Sprint 4 Graph A/C precision-recall curves")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(comparison: pd.DataFrame, path: Path) -> None:
    graph_a = comparison.loc[comparison["graph_schema"] == "graph_a_minimal_physical_target"].iloc[0]
    graph_c = comparison.loc[comparison["graph_schema"] == "graph_c_context_observation"].iloc[0]
    baseline_auprc = float(graph_a["baseline_test_auprc"])
    prevalence = float(graph_a["test_positive_rate"])
    report = f"""# Sprint 4 GCN Graph A/C Comparison Report

## Contract

- Label scheme: `scheme_a`.
- Split ID: `sprint2_main_seed42`.
- Visibility policy: `strict_inductive_primary`.
- Main evaluation remains measured-only with `experiment_id=18` excluded by the frozen Sprint 2 contract.
- Checkpoint and threshold selection use validation only; test diagnostics are interpretation-only.
- Required comparison baseline: `xgboost_unweighted / F4`.
- Test positive prevalence: `{prevalence:.6f}`.

## Result Summary

| Model | Graph schema | Feature set | Test AUPRC | Test AUROC | Test F1 | Test MCC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `xgboost_unweighted` | F4 tabular baseline | F4 | {baseline_auprc:.6f} | {float(graph_a["baseline_test_auroc"]):.6f} | n/a | {float(graph_a["baseline_test_mcc"]):.6f} |
| `{graph_a["model_name"]}` | `{graph_a["graph_schema"]}` | `{graph_a["feature_set"]}` | {float(graph_a["test_auprc"]):.6f} | {float(graph_a["test_auroc"]):.6f} | {float(graph_a["test_f1"]):.6f} | {float(graph_a["test_mcc"]):.6f} |
| `{graph_c["model_name"]}` | `{graph_c["graph_schema"]}` | `{graph_c["feature_set"]}` | {float(graph_c["test_auprc"]):.6f} | {float(graph_c["test_auroc"]):.6f} | {float(graph_c["test_f1"]):.6f} | {float(graph_c["test_mcc"]):.6f} |

## Interpretation

Graph A and Graph C are both validated same-contract GCN runs, but neither beats the frozen `xgboost_unweighted / F4` reference on primary test AUPRC. Graph A test AUPRC is `{float(graph_a["test_auprc"]):.6f}` and Graph C test AUPRC is `{float(graph_c["test_auprc"]):.6f}`, compared with F4 XGBoost test AUPRC `{baseline_auprc:.6f}`.

Graph C must not be interpreted as a topology-only experiment. Relative to Graph A, it changes both topology through `context_similar_to` relations and target semantics through feature-bearing `target_observation` nodes instead of featureless shared `physical_target_site` nodes.

Graph C improves MCC relative to Graph A in this run (`{float(graph_c["test_mcc"]):.6f}` vs `{float(graph_a["test_mcc"]):.6f}`), but this is a secondary metric and does not override the primary AUPRC comparison. Any further changes based on these test diagnostics would violate the no-test-tuning contract.

## Artifact Index

- Consolidated results: `outputs/sprint4/gcn_sprint4_comparison_results.csv`
- AUPRC comparison figure: `outputs/sprint4/figures/gcn_sprint4_schema_auprc_comparison.png`
- PR curve comparison figure: `outputs/sprint4/figures/gcn_sprint4_pr_curves.png`
- Graph A report: `outputs/sprint4/graph_a/gcn_graph_a_report.md`
- Graph C report: `outputs/sprint4/graph_c/gcn_graph_c_report.md`
- Graph C provenance: `outputs/sprint4/graph_c/{GRAPH_C_RUN_ID}/graph_artifact_provenance.json`

## Validation Notes

- Graph C provenance records `sgRNA=150`, `target_observation=11446`, `candidate_pair=11446`, and `context_similar_to=91754`.
- Required Graph A and Graph C diagnostic tables and figures are present in their schema-specific output directories.
- `model.pt` checkpoint files are run artifacts and must remain untracked.
"""
    path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
