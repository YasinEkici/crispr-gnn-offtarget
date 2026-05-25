import pandas as pd
import pytest

from crispr_gnn.graph.graph_builder import validate_main_graph_universe


def base_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "grna_target_id": "train-guide",
                "grna_target_sequence": "A" * 23,
                "target_sequence": "A" * 23,
                "genome": "hg19",
                "target_chr": "chr1",
                "target_start": 1,
                "target_end": 24,
                "target_strand": "+",
                "cleavage_freq": 2e-5,
                "measured": 1,
                "experiment_id": 1,
                "label": 1,
                "split": "train",
            },
            {
                "id": 2,
                "grna_target_id": "val-guide",
                "grna_target_sequence": "C" * 23,
                "target_sequence": "C" * 23,
                "genome": "hg19",
                "target_chr": "chr2",
                "target_start": 1,
                "target_end": 24,
                "target_strand": "+",
                "cleavage_freq": 0.0,
                "measured": 1,
                "experiment_id": 1,
                "label": 0,
                "split": "val",
            },
            {
                "id": 3,
                "grna_target_id": "test-guide",
                "grna_target_sequence": "G" * 23,
                "target_sequence": "G" * 23,
                "genome": "hg19",
                "target_chr": "chr3",
                "target_start": 1,
                "target_end": 24,
                "target_strand": "+",
                "cleavage_freq": 2e-5,
                "measured": 1,
                "experiment_id": 1,
                "label": 1,
                "split": "test",
            },
        ]
    )


def test_graph_universe_rejects_measured_zero_or_experiment_18() -> None:
    putative = base_rows()
    putative.loc[0, "measured"] = 0
    with pytest.raises(ValueError, match="measured=0"):
        validate_main_graph_universe(putative)

    experiment = base_rows()
    experiment.loc[0, "experiment_id"] = 18
    with pytest.raises(ValueError, match="experiment_id=18"):
        validate_main_graph_universe(experiment)


def test_graph_universe_rejects_scheme_a_drift_and_guide_overlap() -> None:
    label_drift = base_rows()
    label_drift.loc[0, "label"] = 0
    with pytest.raises(ValueError, match="Scheme A"):
        validate_main_graph_universe(label_drift)

    overlap = base_rows()
    overlap.loc[1, "grna_target_id"] = "train-guide"
    with pytest.raises(ValueError, match="guide leakage"):
        validate_main_graph_universe(overlap)


def test_graph_universe_rejects_nan_cleavage_frequency() -> None:
    nan_label = base_rows()
    nan_label.loc[0, "cleavage_freq"] = float("nan")
    with pytest.raises(ValueError, match="label-eligible"):
        validate_main_graph_universe(nan_label)


def test_graph_universe_preserves_negative_and_above_one_scheme_a_labels() -> None:
    edge_cases = base_rows()
    edge_cases.loc[0, ["cleavage_freq", "label"]] = [-0.01, 0]
    edge_cases.loc[1, ["cleavage_freq", "label"]] = [1.5, 1]

    validate_main_graph_universe(edge_cases)
