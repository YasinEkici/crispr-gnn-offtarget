import pandas as pd
import torch
from torch_geometric.data import HeteroData

from crispr_gnn.evaluation.diagnostics import write_gcn_diagnostics, write_gcn_report
from crispr_gnn.evaluation.plots import GCN_REQUIRED_FIGURES, write_gcn_plots


def test_gcn_reporting_outputs_are_generated_from_mock_graph_a_artifacts(tmp_path) -> None:
    results = _mock_results()
    predictions = _mock_predictions()
    history = _mock_training_history()
    graph_view = _mock_graph_view()

    diagnostic_tables = write_gcn_diagnostics(results, predictions, tmp_path / "diagnostics")
    figure_paths = write_gcn_plots(results, predictions, history, tmp_path / "figures", graph_view=graph_view)
    report_path = write_gcn_report(
        results,
        diagnostic_tables,
        figure_paths,
        tmp_path / "reports" / "gcn_report.md",
        run_label="mock_reporting_path",
    )

    figure_names = {path.name for path in figure_paths}
    assert set(GCN_REQUIRED_FIGURES).issubset(figure_names)
    assert "gcn_sequence_position_sensitivity.png" not in figure_names
    for path in [*diagnostic_tables, *figure_paths, report_path]:
        assert path.exists()
        assert path.stat().st_size > 0
    report_text = report_path.read_text(encoding="utf-8")
    assert "xgboost_unweighted / F4" in report_text
    assert "Test positive prevalence" in report_text
    assert "Smoke or mocked outputs are not final" in report_text


def test_gcn_sequence_position_sensitivity_is_conditional(tmp_path) -> None:
    sensitivity = pd.DataFrame({"position": [1, 2, 3], "mean_score_delta": [0.01, -0.02, 0.03]})

    figure_paths = write_gcn_plots(
        _mock_results(),
        _mock_predictions(),
        _mock_training_history(),
        tmp_path,
        graph_view=_mock_graph_view(),
        sequence_position_sensitivity=sensitivity,
    )

    assert "gcn_sequence_position_sensitivity.png" in {path.name for path in figure_paths}


def _mock_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sprint": "sprint4",
                "label_scheme": "scheme_a",
                "split_id": "sprint2_main_seed42",
                "seed": 42,
                "training_regime": "measured_only",
                "model_name": "gcn_graph_a",
                "feature_set": "S1_pair+F1",
                "graph_schema": "graph_a_minimal_physical_target",
                "visibility_policy": "strict_inductive_primary",
                "target_node_representation": "zero_type_feature",
                "threshold": 0.5,
                "threshold_selection_split": "validation",
                "baseline_reference": "xgboost_unweighted / F4",
                "baseline_test_auprc": 0.992522,
                "baseline_test_auroc": 0.938416,
                "baseline_test_mcc": 0.345198,
                "test_positive_rate": 0.5,
                "test_auprc": 0.75,
                "test_auroc": 0.75,
                "test_f1": 0.8,
                "test_mcc": 0.5,
            }
        ]
    )


def _mock_predictions() -> pd.DataFrame:
    rows = []
    for split in ["val", "test"]:
        for index, (label, score) in enumerate([(1, 0.9), (0, 0.2), (1, 0.7), (0, 0.4)]):
            rows.append(
                {
                    "model_name": "gcn_graph_a",
                    "graph_schema": "graph_a_minimal_physical_target",
                    "feature_set": "S1_pair+F1",
                    "split": split,
                    "row_index": index,
                    "label": label,
                    "score": score,
                    "genome": "hg19" if index < 2 else "mm10",
                    "grna_target_id": f"guide_{index % 2}",
                }
            )
    return pd.DataFrame(rows)


def _mock_training_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "gcn_graph_a",
                "graph_schema": "graph_a_minimal_physical_target",
                "feature_set": "S1_pair+F1",
                "epoch": epoch,
                "train_loss": 1.0 / epoch,
                "val_auprc": 0.5 + 0.05 * epoch,
                "selection_split": "validation",
            }
            for epoch in [1, 2, 3]
        ]
    )


def _mock_graph_view() -> HeteroData:
    data = HeteroData()
    data.graph_name = "graph_a_minimal_physical_target"
    data.view_name = "test"
    data["sgRNA"].num_nodes = 2
    data["sgRNA"].audit_node_ids = ["guide_0", "guide_1"]
    data["physical_target_site"].num_nodes = 2
    data["physical_target_site"].audit_node_ids = ["target_0", "target_1"]
    edge_store = data["sgRNA", "candidate_pair", "physical_target_site"]
    edge_store.edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    return data
