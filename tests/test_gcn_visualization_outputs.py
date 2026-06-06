import pandas as pd
import torch
from torch_geometric.data import HeteroData

from crispr_gnn.evaluation.diagnostics import (
    write_gcn_diagnostics,
    write_gcn_report,
    write_sprint6_imbalance_diagnostics,
)
from crispr_gnn.evaluation.plots import GCN_REQUIRED_FIGURES, write_gcn_plots, write_sprint6_imbalance_plots


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


def test_gcn_graph_b_reporting_paths_are_schema_specific(tmp_path) -> None:
    results = _mock_results(
        model_name="gcn_graph_b",
        graph_schema="graph_b_guide_similarity_control",
        target_node_representation="zero_type_feature",
    )
    predictions = _mock_predictions(model_name="gcn_graph_b", graph_schema="graph_b_guide_similarity_control")
    history = _mock_training_history(model_name="gcn_graph_b", graph_schema="graph_b_guide_similarity_control")
    graph_view = _mock_graph_b_view()

    diagnostic_tables = write_gcn_diagnostics(
        results,
        predictions,
        tmp_path / "diagnostics",
        schema_label="graph_b",
    )
    figure_paths = write_gcn_plots(
        results,
        predictions,
        history,
        tmp_path / "figures",
        schema_label="graph_b",
        graph_view=graph_view,
    )
    report_path = write_gcn_report(
        results,
        diagnostic_tables,
        figure_paths,
        tmp_path / "reports" / "gcn_graph_b_report.md",
        run_label="mock_graph_b_reporting_path",
    )

    assert {path.name for path in diagnostic_tables} >= {
        "gcn_graph_b_score_direction.csv",
        "gcn_graph_b_fixed_threshold_metrics.csv",
        "gcn_graph_b_score_deciles.csv",
    }
    assert {path.name for path in figure_paths} >= {
        "gcn_graph_b_graph_schema_auprc_comparison.png",
        "gcn_graph_b_view_sanity_example.png",
    }
    report_text = report_path.read_text(encoding="utf-8")
    assert "graph_b_guide_similarity_control" in report_text


def test_gcn_graph_c_reporting_paths_keep_schema_specific_outputs(tmp_path) -> None:
    results = _mock_results(
        model_name="gcn_graph_c",
        graph_schema="graph_c_context_observation",
        target_node_representation="target_observation_context_encoder",
    )
    predictions = _mock_predictions(model_name="gcn_graph_c", graph_schema="graph_c_context_observation")
    history = _mock_training_history(model_name="gcn_graph_c", graph_schema="graph_c_context_observation")
    graph_view = _mock_graph_c_view()

    diagnostic_tables = write_gcn_diagnostics(
        results,
        predictions,
        tmp_path / "diagnostics",
        schema_label="graph_c",
    )
    figure_paths = write_gcn_plots(
        results,
        predictions,
        history,
        tmp_path / "figures",
        schema_label="graph_c",
        graph_view=graph_view,
    )
    report_path = write_gcn_report(
        results,
        diagnostic_tables,
        figure_paths,
        tmp_path / "reports" / "gcn_graph_c_report.md",
        run_label="mock_graph_c_reporting_path",
    )

    assert {path.name for path in diagnostic_tables} >= {
        "gcn_graph_c_score_direction.csv",
        "gcn_graph_c_fixed_threshold_metrics.csv",
        "gcn_graph_c_score_deciles.csv",
    }
    assert {path.name for path in figure_paths} >= {
        "gcn_graph_c_graph_schema_auprc_comparison.png",
        "gcn_graph_c_view_sanity_example.png",
    }
    report_text = report_path.read_text(encoding="utf-8")
    assert "graph_c_context_observation" in report_text
    assert "target_observation_context_encoder" in results["target_node_representation"].iloc[0]


def test_sprint6_imbalance_reporting_outputs_are_run_id_grouped(tmp_path) -> None:
    results = _mock_sprint6_results()
    predictions = _mock_sprint6_predictions()
    history = _mock_sprint6_history()

    diagnostic_tables = write_sprint6_imbalance_diagnostics(results, predictions, tmp_path / "diagnostics")
    figure_paths = write_sprint6_imbalance_plots(results, predictions, history, tmp_path / "figures")

    assert {path.name for path in diagnostic_tables} >= {
        "imbalance_threshold_metrics.csv",
        "imbalance_per_guide_metrics.csv",
        "imbalance_per_guide_metric_distribution.csv",
        "imbalance_positive_retrieval_summary.csv",
        "imbalance_negative_retrieval_summary.csv",
    }
    assert {path.name for path in figure_paths} == {
        "imbalance_auprc_comparison.png",
        "imbalance_pr_curves.png",
        "imbalance_threshold_metrics.png",
        "imbalance_score_distributions.png",
        "imbalance_per_guide_metric_distribution.png",
        "imbalance_positive_retrieval_summary.png",
        "imbalance_negative_retrieval_summary.png",
    }
    threshold_metrics = pd.read_csv(tmp_path / "diagnostics" / "imbalance_threshold_metrics.csv")
    assert set(threshold_metrics["run_id"]) == {"batch_S6R0_wbce", "batch_S6R7_balanced_sampling"}
    assert "tnr" in threshold_metrics.columns
    for path in [*diagnostic_tables, *figure_paths]:
        assert path.exists()
        assert path.stat().st_size > 0


def _mock_results(
    *,
    model_name: str = "gcn_graph_a",
    graph_schema: str = "graph_a_minimal_physical_target",
    target_node_representation: str = "zero_type_feature",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sprint": "sprint4",
                "label_scheme": "scheme_a",
                "split_id": "sprint2_main_seed42",
                "seed": 42,
                "training_regime": "measured_only",
                "model_name": model_name,
                "feature_set": "S1_pair+F1",
                "graph_schema": graph_schema,
                "visibility_policy": "strict_inductive_primary",
                "target_node_representation": target_node_representation,
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


def _mock_predictions(
    *,
    model_name: str = "gcn_graph_a",
    graph_schema: str = "graph_a_minimal_physical_target",
) -> pd.DataFrame:
    rows = []
    for split in ["val", "test"]:
        for index, (label, score) in enumerate([(1, 0.9), (0, 0.2), (1, 0.7), (0, 0.4)]):
            rows.append(
                {
                    "model_name": model_name,
                    "graph_schema": graph_schema,
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


def _mock_training_history(
    *,
    model_name: str = "gcn_graph_a",
    graph_schema: str = "graph_a_minimal_physical_target",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "graph_schema": graph_schema,
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


def _mock_graph_b_view() -> HeteroData:
    data = HeteroData()
    data.graph_name = "graph_b_guide_similarity_control"
    data.view_name = "test"
    data["sgRNA"].num_nodes = 2
    data["sgRNA"].audit_node_ids = ["guide_0", "guide_1"]
    data["physical_target_site"].num_nodes = 2
    data["physical_target_site"].audit_node_ids = ["target_0", "target_1"]
    candidate = data["sgRNA", "candidate_pair", "physical_target_site"]
    candidate.edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    similarity = data["sgRNA", "sequence_similar_to", "sgRNA"]
    similarity.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return data


def _mock_graph_c_view() -> HeteroData:
    data = HeteroData()
    data.graph_name = "graph_c_context_observation"
    data.view_name = "test"
    data["sgRNA"].num_nodes = 2
    data["sgRNA"].audit_node_ids = ["guide_0", "guide_1"]
    data["target_observation"].num_nodes = 3
    data["target_observation"].audit_node_ids = ["obs_train_0", "obs_train_1", "obs_test_0"]
    candidate = data["sgRNA", "candidate_pair", "target_observation"]
    candidate.edge_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    context = data["target_observation", "context_similar_to", "target_observation"]
    context.edge_index = torch.tensor([[2], [0]], dtype=torch.long)
    return data


def _mock_sprint6_results() -> pd.DataFrame:
    base = {
        "sprint": "sprint6",
        "label_scheme": "scheme_a",
        "split_id": "sprint2_main_seed42",
        "seed": 42,
        "training_regime": "measured_only",
        "model_name": "gcn_graph_a_sprint6",
        "feature_set": "S5F2_energy",
        "graph_schema": "graph_a_minimal_physical_target",
        "visibility_policy": "strict_inductive_primary",
        "target_node_representation": "zero_type_feature",
        "threshold": 0.5,
        "threshold_selection_split": "validation",
        "baseline_reference": "xgboost_unweighted / F4",
        "baseline_test_auprc": 0.992522,
        "baseline_test_auroc": 0.938416,
        "baseline_test_mcc": 0.345198,
        "prior_sprint5_s5f2_test_auprc": 0.976585,
        "prior_sprint5_s5f2_test_mcc": 0.477933,
        "prior_test_positive_prevalence": 0.900705,
        "test_positive_rate": 0.900705,
        "test_auroc": 0.8,
        "test_f1": 0.9,
        "test_macro_f1": 0.7,
        "test_sensitivity": 0.95,
        "test_tn": 1,
        "test_fp": 1,
        "test_fn": 1,
        "test_tp": 5,
    }
    return pd.DataFrame(
        [
            {
                **base,
                "run_id": "batch_S6R0_wbce",
                "run_order": 0,
                "predeclared_run_id": "S6R0_wbce",
                "loss": "weighted_bce",
                "loss_params": '{"pos_weight": "auto"}',
                "sampling": "null",
                "role": "baseline",
                "test_auprc": 0.97,
                "test_mcc": 0.4,
                "test_specificity": 0.5,
            },
            {
                **base,
                "run_id": "batch_S6R7_balanced_sampling",
                "run_order": 1,
                "predeclared_run_id": "S6R7_balanced_sampling",
                "loss": "bce_unweighted",
                "loss_params": '{"pos_weight": 1.0}',
                "sampling": '{"strategy": "balanced_subsample"}',
                "role": "sampling",
                "test_auprc": 0.96,
                "test_mcc": 0.5,
                "test_specificity": 0.75,
            },
        ]
    )


def _mock_sprint6_predictions() -> pd.DataFrame:
    rows = []
    for run_id, delta in [("batch_S6R0_wbce", 0.0), ("batch_S6R7_balanced_sampling", -0.05)]:
        for split in ["val", "test"]:
            for index, (label, score) in enumerate(
                [(1, 0.9 + delta), (0, 0.2), (1, 0.7 + delta), (0, 0.6 + delta), (1, 0.8 + delta), (1, 0.4)]
            ):
                rows.append(
                    {
                        "run_id": run_id,
                        "model_name": "gcn_graph_a_sprint6",
                        "graph_schema": "graph_a_minimal_physical_target",
                        "feature_set": "S5F2_energy",
                        "split": split,
                        "row_index": index,
                        "label": label,
                        "score": score,
                        "genome": "hg19" if index < 3 else "mm10",
                        "grna_target_id": f"guide_{index % 3}",
                    }
                )
    return pd.DataFrame(rows)


def _mock_sprint6_history() -> pd.DataFrame:
    rows = []
    for run_id in ["batch_S6R0_wbce", "batch_S6R7_balanced_sampling"]:
        for epoch in [1, 2]:
            rows.append(
                {
                    "run_id": run_id,
                    "model_name": "gcn_graph_a_sprint6",
                    "graph_schema": "graph_a_minimal_physical_target",
                    "feature_set": "S5F2_energy",
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "val_auprc": 0.7 + 0.1 * epoch,
                }
            )
    return pd.DataFrame(rows)
