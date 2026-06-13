"""Sprint 9 Slice 1 tests — prediction registry and metric replay.

These tests anchor the Sprint 9 uncertainty layer to the frozen Sprint 7F/8A/8B
results: the registry must load every predeclared model from a single authoritative
batch, thresholds must be read (validation-selected) and not recomputed from test,
the guide cluster key must be present (29 test guides), and replayed full-test
metrics must reproduce the source comparison CSVs within tolerance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, confusion_matrix

from crispr_gnn.evaluation.metrics import select_threshold_by_f1
from crispr_gnn.evaluation.robustness import (
    BOOTSTRAP_METRICS,
    GNN_REGISTRY,
    GUIDE_KEY,
    _PRED_F4,
    F4_REGISTRY_ID,
    ModelScores,
    _guide_groups,
    guide_cluster_bootstrap,
    leave_one_guide_influence,
    load_f4_model_scores,
    load_full_registry,
    load_model_scores,
    load_registry,
    replay_check_records,
    replay_split_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_scores() -> ModelScores:
    """2-guide synthetic: guide 1 all-positive, guide 2 holds all negatives."""
    rows = [{"split": "test", GUIDE_KEY: 1, "label": 1, "score": 0.9} for _ in range(10)]
    rows += [{"split": "test", GUIDE_KEY: 2, "label": 1, "score": 0.8} for _ in range(5)]
    rows += [{"split": "test", GUIDE_KEY: 2, "label": 0, "score": 0.2} for _ in range(5)]
    return ModelScores("SYN", "SYN", "run", "synthetic", 0.5, "validation", pd.DataFrame(rows))


def _entry(registry_id: str):
    return next(entry for entry in GNN_REGISTRY if entry.registry_id == registry_id)


# F4 tests require the Slice 2 regeneration to have run (scripts/regenerate_f4_predictions.py).
f4_available = pytest.mark.skipif(
    not _PRED_F4.exists(),
    reason="F4 predictions not regenerated yet (run scripts/regenerate_f4_predictions.py)",
)


def test_registry_complete_and_single_batch():
    registry = load_registry()
    assert len(registry) == 10
    assert {entry.registry_id for entry in GNN_REGISTRY} == set(registry)
    for scores in registry.values():
        # Loader enforces exactly one authoritative batch run_id.
        assert scores.run_id
        assert {"test", "val"} <= set(scores.frame["split"].unique())


def test_threshold_read_from_comparison_and_validation_selected():
    for entry in GNN_REGISTRY:
        scores = load_model_scores(entry)
        comparison = pd.read_csv(entry.comparison_path)
        row = comparison[comparison["predeclared_run_id"] == entry.predeclared_run_id].iloc[0]
        # Read verbatim from the comparison CSV (exact equality, no recomputation).
        assert scores.threshold == pytest.approx(float(row["threshold"]), abs=0.0)
        assert scores.threshold_selection_split == "validation"


def test_threshold_not_recomputed_from_test():
    # The frozen threshold is validation-max-F1; recomputing it on test would give a
    # different (test-optimal) value. Confirm they differ, proving no test tuning.
    scores = load_model_scores(_entry("S8B_R2"))
    test = scores.split("test")
    test_optimal = select_threshold_by_f1(test["label"].to_numpy(), test["score"].to_numpy()).threshold
    assert scores.threshold != pytest.approx(test_optimal, abs=1e-6)


def test_guide_cluster_key_present_with_expected_geometry():
    scores = load_model_scores(_entry("S8B_R2"))
    test = scores.split("test")
    assert GUIDE_KEY in test.columns
    assert len(test) == 1702
    assert test[GUIDE_KEY].nunique() == 29
    assert int((test["label"] == 0).sum()) == 169
    negatives_per_guide = test[test["label"] == 0].groupby(GUIDE_KEY).size()
    assert int((negatives_per_guide > 0).sum()) == 9
    assert int(negatives_per_guide.max()) == 80  # dominant guide 9251 (47.3%)


def test_replay_matches_source_within_tolerance():
    records = replay_check_records(split="test")
    assert records, "no replay records produced"
    mismatches = [record for record in records if not record["within_tol"]]
    assert not mismatches, f"replay mismatches: {mismatches}"
    # All ten GNN registry models are covered.
    assert {record["registry_id"] for record in records} == {
        entry.registry_id for entry in GNN_REGISTRY
    }


@f4_available
def test_f4_reproduces_bar_and_threshold_is_validation_selected():
    # Option C (2026-06-13): F4 is regenerated under the current pinned XGBoost;
    # version drift vs the historical 0.992522 is ~1e-4. Assertions are version-robust
    # (geometry exact; AUPRC within the documented 2e-3 sanity tolerance), not bit-exact.
    f4 = load_f4_model_scores()
    assert f4.registry_id == F4_REGISTRY_ID
    assert f4.threshold_selection_split == "validation"

    test = f4.split("test")
    assert len(test) == 1702
    assert test[GUIDE_KEY].nunique() == 29
    assert int((test["label"] == 0).sum()) == 169

    auprc = average_precision_score(test["label"].to_numpy(), test["score"].to_numpy())
    assert auprc == pytest.approx(0.992522, abs=2e-3)

    # The F4 threshold is validation-selected, never recomputed from test.
    test_optimal = select_threshold_by_f1(test["label"].to_numpy(), test["score"].to_numpy()).threshold
    assert f4.threshold != pytest.approx(test_optimal, abs=1e-6)
    # Confusion is computed at the validation-selected threshold (defined, both classes present).
    predictions = (test["score"].to_numpy() >= f4.threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(test["label"], predictions, labels=[0, 1]).ravel())
    assert tn + fp == 169 and fn + tp == 1533


@f4_available
def test_f4_test_guides_align_with_gnn_for_paired_bootstrap():
    f4_guides = set(load_f4_model_scores().split("test")[GUIDE_KEY].astype(int))
    gnn_guides = set(load_model_scores(_entry("S8B_R2")).split("test")[GUIDE_KEY].astype(int))
    assert f4_guides == gnn_guides


@f4_available
def test_full_registry_includes_f4():
    registry = load_full_registry()
    assert F4_REGISTRY_ID in registry
    assert len(registry) == 11


# --- Slice 3: guide-cluster bootstrap ------------------------------------------


def test_guide_groups_partition_whole_guides():
    scores = load_model_scores(_entry("S8B_R2"))
    frame = scores.split("test")
    guide_ids, labels, values = _guide_groups(frame)
    # One group per guide; concatenation reproduces the full row set (no row split).
    assert len(guide_ids) == 29
    assert sum(len(a) for a in labels) == len(frame)
    assert sum(len(a) for a in values) == len(frame)
    assert sorted(guide_ids) == sorted(frame[GUIDE_KEY].unique())


def test_bootstrap_resamples_guides_not_rows_and_handles_degeneracy():
    # Guide-level resampling: with 2 equal guides, the negative-bearing guide (2) is
    # absent in ~(1/2)^2 = 25% of replicates -> specificity undefined there (counted,
    # not crashed); AUPRC always defined (positives always present).
    result = guide_cluster_bootstrap(_synthetic_scores(), n_boot=4000, seed=12345, dominant_guide=2)
    assert result.n_test_guides == 2
    assert result.undefined_rate["auprc"] == 0.0
    assert 0.18 < result.undefined_rate["specificity"] < 0.32
    assert 1.3 < result.mean_unique_guides < 1.7  # E[unique of 2 with replacement] = 1.5
    assert 0.6 < result.dominant_guide_inclusion_rate < 0.9  # P(guide 2 drawn) = 0.75


def test_bootstrap_point_matches_replay_and_reads_threshold():
    scores = load_model_scores(_entry("S8B_R2"))
    result = guide_cluster_bootstrap(scores, n_boot=100, seed=12345)
    replay = replay_split_metrics(scores, "test")
    # Threshold-free metric: point matches the Slice 1 replay exactly.
    assert result.point["auprc"] == pytest.approx(replay["test_auprc"], abs=1e-9)
    # Thresholded metric matching proves the frozen (read) threshold is applied.
    assert result.point["mcc"] == pytest.approx(replay["test_mcc"], abs=1e-9)
    assert result.n_test_guides == 29
    # Percentile CI brackets the point estimate; BCa carries a trust flag + note.
    lo, hi = result.percentile["auprc"]
    assert lo <= result.point["auprc"] <= hi
    assert isinstance(result.bca_trusted["auprc"], bool)
    assert result.bca_note["auprc"]


def test_leave_one_guide_influence_shape():
    result = guide_cluster_bootstrap(load_model_scores(_entry("S8B_R2")), n_boot=50, seed=12345)
    influence = leave_one_guide_influence(result)
    assert len(influence) == 29 * len(BOOTSTRAP_METRICS)
    assert {row["metric"] for row in influence} == set(BOOTSTRAP_METRICS)


@f4_available
def test_bootstrap_output_contract(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sprint9_robustness as runner

    status = runner.run_bootstrap_stage(tmp_path, n_boot=50, seed=12345, ci=0.95)
    assert status == 0
    for rel in [
        "robustness_bootstrap_cis.csv",
        "diagnostics/bootstrap_replicate_diagnostics.csv",
        "diagnostics/leave_one_guide_influence.csv",
        "figures/robustness_auprc_cis.png",
        "figures/bootstrap_distribution_diagnostics.png",
    ]:
        assert (tmp_path / rel).exists(), f"missing {rel}"
    ci = pd.read_csv(tmp_path / "robustness_bootstrap_cis.csv")
    assert ci["registry_id"].nunique() == 11
    assert set(ci["metric"]) == set(BOOTSTRAP_METRICS)
