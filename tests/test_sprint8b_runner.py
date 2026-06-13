"""Sprint 8B Slice 3: runner output-contract test (monkeypatched training)."""

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES, EXPERIMENTAL_EPIGENETIC_FEATURES
from crispr_gnn.models.gcn import GRAPH_C_CONTEXT_EDGE_TYPE, GRAPH_C_EDGE_TYPE
from scripts import run_sprint8b_sequence_context as runner


def test_sprint8b_runner_writes_output_contract(tmp_path, monkeypatch) -> None:
    config_path = _write_runner_config(tmp_path)
    artifact_dir = tmp_path / "graphs" / "sprint5b" / "graph_c_context_observation"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "dummy.parquet").write_bytes(b"hash-only")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "Sprint3HeteroDataLoader", _fake_loader_factory)
    monkeypatch.setattr(runner, "train_graph_c_gcn", _fake_train)
    monkeypatch.setattr(runner, "collect_graph_attention_summary", lambda *a, **k: _fake_attention())
    monkeypatch.setattr(runner, "collect_target_context_encoder_summary", lambda *a, **k: _fake_encoder_summary(a[1]))
    monkeypatch.setattr(
        runner, "collect_context_edge_interaction_summary", lambda *a, **k: _fake_interaction_summary(a[1])
    )
    monkeypatch.setattr(runner, "_runtime_info", lambda device: {"device": device})

    output_dir = runner.run_sprint8b_sequence_context(config_path=config_path, batch_id="batch", max_epochs=1)

    required = [
        output_dir / "sequence_context_comparison.csv",
        output_dir / "sequence_context_report.md",
        output_dir / "sequence_context_run_manifest.json",
        output_dir / "graph_artifact_provenance.json",
        output_dir / "diagnostics" / "sequence_context_threshold_metrics.csv",
        output_dir / "diagnostics" / "sequence_context_deltas.csv",
        output_dir / "diagnostics" / "sequence_context_training_history.csv",
        output_dir / "diagnostics" / "sequence_context_predictions.csv",
        output_dir / "diagnostics" / "sequence_context_score_deciles.csv",
        output_dir / "diagnostics" / "sequence_context_per_guide_score_summary.csv",
        output_dir / "diagnostics" / "sequence_context_sequence_input_audit.csv",
        output_dir / "diagnostics" / "sequence_context_parameter_counts.csv",
        output_dir / "diagnostics" / "sequence_context_attention_summary.csv",
        output_dir / "diagnostics" / "sequence_context_target_context_encoder_summary.csv",
        output_dir / "diagnostics" / "sequence_context_film_summary.csv",
        output_dir / "figures" / "sequence_context_auprc_comparison.png",
        output_dir / "figures" / "sequence_context_threshold_metrics.png",
        output_dir / "figures" / "sequence_context_pr_curves.png",
        output_dir / "figures" / "sequence_context_roc_curves.png",
        output_dir / "figures" / "sequence_context_score_distributions.png",
        output_dir / "figures" / "sequence_context_training_curves.png",
        output_dir / "figures" / "sequence_context_parameter_counts.png",
    ]
    per_run_files = ["resolved_config.yaml", "runtime.json", "training_history.csv", "metrics.csv", "sequence_input_audit.csv", "model.pt"]
    for run_id in runner.HEADLINE_RUN_IDS:
        for filename in per_run_files:
            required.append(output_dir / "runs" / f"batch_{run_id}" / filename)
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    results = pd.read_csv(output_dir / "sequence_context_comparison.csv")
    assert set(results["predeclared_run_id"]) == {*runner.REFERENCE_RUN_IDS, *runner.HEADLINE_RUN_IDS}
    headline = results.loc[results["predeclared_run_id"].isin(runner.HEADLINE_RUN_IDS)]
    assert set(headline["sequence_context_mode"]) == {"sequence_only", "late_fusion"}
    assert headline.loc[
        headline["predeclared_run_id"] == "S8B_R1_sequence_only", "edge_feature_columns"
    ].iloc[0] == 0
    assert headline["external_pretrained_weights"].eq(False).all()
    _assert_canonical_matrix(headline)

    audit = pd.read_csv(output_dir / "diagnostics" / "sequence_context_sequence_input_audit.csv")
    assert set(audit["sequence_context_mode"]) == {"sequence_only", "late_fusion"}
    assert set(audit["representation"]) == {"S1_sequence_pair_from_graph_c_onehot"}
    assert audit["guide_onehot_columns"].eq(115).all()
    assert audit["target_onehot_columns"].eq(115).all()
    assert audit["channels"].eq(11).all()

    params = pd.read_csv(output_dir / "diagnostics" / "sequence_context_parameter_counts.csv")
    assert {"parameter_count", "active_parameter_count", "sequence_context_mode"}.issubset(params.columns)

    manifest = json.loads((output_dir / "sequence_context_run_manifest.json").read_text())
    assert manifest["headline_run_ids"] == list(runner.HEADLINE_RUN_IDS)
    assert manifest["reference_run_ids"] == list(runner.REFERENCE_RUN_IDS)
    assert manifest["selection_metric"] == "validation_auprc"
    manifest_rows = pd.DataFrame([row for row in manifest["runs"] if row["predeclared_id"] in runner.HEADLINE_RUN_IDS])
    manifest_rows = manifest_rows.rename(columns={"predeclared_id": "predeclared_run_id"})
    _assert_canonical_matrix(manifest_rows)


def _assert_canonical_matrix(rows: pd.DataFrame) -> None:
    observed = rows.set_index("predeclared_run_id")
    assert list(observed.index) == list(runner.HEADLINE_RUN_IDS)
    for run_id, expected in runner.CANONICAL_RUN_FLAGS.items():
        row = observed.loc[run_id]
        for column, value in expected.items():
            if value is None:
                assert pd.isna(row[column]), (run_id, column, row[column])
            else:
                assert row[column] == value, (run_id, column)


def _guide_feature_names() -> list[str]:
    return [f"feature__guide_pos_{p:02d}_{b}" for p in range(23) for b in ("A", "C", "G", "T", "N")]


def _target_feature_names() -> list[str]:
    names = [f"feature__target_pos_{p:02d}_{b}" for p in range(23) for b in ("A", "C", "G", "T", "N")]
    names.extend(f"feature__{name}" for name in EXPERIMENTAL_EPIGENETIC_FEATURES)
    for feature in COMPUTED_NUCLEOSOME_FEATURES:
        for suffix in ("mean", "std", "min", "max", "center", "pam_proximal_mean"):
            names.append(f"feature__{feature}_{suffix}")
        names.append(f"feature__{feature}_missing")
    return names


def _onehot_from_sequence(seq: str, length: int = 23) -> torch.Tensor:
    bases = "ACGTN"
    out = torch.zeros(length * 5, dtype=torch.float32)
    for pos in range(length):
        base = seq[pos] if pos < len(seq) else "N"
        base = base if base in bases else "N"
        out[pos * 5 + bases.index(base)] = 1.0
    return out


def _write_runner_config(tmp_path: Path) -> Path:
    source = runner.load_yaml(Path("configs/sweeps/sprint8b_sequence_context.yaml"))
    source["data"]["graph_c_artifact_dir"] = "graphs/sprint5b"
    source["outputs"]["output_dir"] = "outputs/sprint8b"
    source["training"]["device"] = "cpu"
    source["training"]["use_compile"] = False
    source["training"]["use_amp"] = False
    path = tmp_path / "sprint8b_sequence_context.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
    return path


class _FakeMaterialized:
    graph_name = "graph_c_context_observation"
    manifest = {
        "graph_name": "graph_c_context_observation",
        "split_id": "sprint2_main_seed42",
        "label_scheme": "scheme_a",
        "feature_tables": {"S5F2_energy": 268, "target_observation_features": 212},
        "metadata": {"visibility_policy": "strict_inductive_primary"},
    }

    def view(self, _split: str) -> HeteroData:
        return _fake_graph_c_view()


class _FakeLoader:
    def __init__(self, _path: Path) -> None:
        self.path = _path

    def load(self, graph_name: str) -> _FakeMaterialized:
        assert graph_name == "graph_c_context_observation"
        return _FakeMaterialized()


def _fake_loader_factory(path: Path) -> _FakeLoader:
    return _FakeLoader(path)


def _fake_graph_c_view() -> HeteroData:
    data = HeteroData()
    data.graph_name = "graph_c_context_observation"
    data["sgRNA"].x = torch.stack([_onehot_from_sequence("A" * 23), _onehot_from_sequence("C" * 23)])
    data["sgRNA"].feature_names = _guide_feature_names()
    target_rows = []
    for sequence in ("A" * 23, "G" * 23, "T" * 23):
        row = torch.zeros(212)
        row[:115] = _onehot_from_sequence(sequence)
        target_rows.append(row)
    data["target_observation"].x = torch.stack(target_rows)
    data["target_observation"].feature_names = _target_feature_names()
    edge_store = data[GRAPH_C_EDGE_TYPE]
    edge_store.edge_index = torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.long)
    edge_store.edge_label = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    edge_store.supervision_mask = torch.tensor([True, True, True])
    edge_store.edge_attr_s5f2_energy = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
    data[GRAPH_C_CONTEXT_EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return data


def _fake_train(_materialized, config, *, checkpoint_path: Path):
    checkpoint_path.write_bytes(b"fake-checkpoint")
    offset = 0.002 if config.sequence_context_mode == "late_fusion" else 0.0
    params = 210000 if config.sequence_context_mode == "sequence_only" else 450000
    active_params = params if config.sequence_context_mode == "sequence_only" else 350000
    results = pd.DataFrame(
        [
            {
                "sprint": "sprint8b",
                "model_name": config.model_name,
                "architecture": "gatv2",
                "feature_set": "S5F2_energy",
                "graph_schema": "graph_c_context_observation",
                "loss": "weighted_bce",
                "test_positive_rate": 0.900705,
                "test_auprc": 0.981 + offset,
                "test_auroc": 0.91 + offset,
                "test_f1": 0.95,
                "test_macro_f1": 0.77,
                "test_mcc": 0.55 + offset,
                "test_specificity": 0.50,
                "test_sensitivity": 0.97,
                "test_tn": 84,
                "test_fp": 85,
                "test_fn": 31,
                "test_tp": 1502,
                "edge_feature_columns": 0 if config.sequence_context_mode == "sequence_only" else 268,
                "parameter_count": params,
                "active_parameter_count": active_params,
            }
        ]
    )
    predictions = _fake_predictions(config)
    history = pd.DataFrame(
        [
            {
                "model_name": config.model_name,
                "epoch": 1,
                "train_loss": 0.5,
                "val_loss": 0.4,
                "val_auprc": 0.98 + offset,
                "lr": 0.001,
            },
        ]
    )
    return results, predictions, history


def _fake_predictions(config) -> pd.DataFrame:
    rows = []
    for split in ["val", "test"]:
        for index, (label, score) in enumerate([(1, 0.9), (0, 0.2), (1, 0.7), (0, 0.6)]):
            rows.append(
                {
                    "model_name": config.model_name,
                    "architecture": config.architecture,
                    "graph_schema": config.graph_schema,
                    "feature_set": "S5F2_energy",
                    "split": split,
                    "row_index": index,
                    "grna_target_id": f"guide_{index % 2}",
                    "genome": None,
                    "label": label,
                    "score": score,
                }
            )
    return pd.DataFrame(rows)


def _fake_attention() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "gatv2",
                "architecture": "gatv2",
                "split": "test",
                "layer": 0,
                "head": 0,
                "edge_kind": "candidate_forward",
                "edge_count": 3,
                "attention_mean": 0.4,
                "attention_std": 0.1,
                "attention_min": 0.2,
                "attention_max": 0.6,
            }
        ]
    )


def _fake_encoder_summary(config) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": config.model_name,
                "architecture": "gatv2",
                "graph_schema": "graph_c_context_observation",
                "split": "test",
                "target_context_encoder_type": config.target_context_encoder_type,
                "target_context_family": "all_target_observation_features",
                "input_columns": 212,
                "branch_dim": 128,
                "activation_l2_mean": 1.0,
                "target_context_encoder_parameter_count": 123,
            }
        ]
    )


def _fake_interaction_summary(config) -> pd.DataFrame:
    if config.context_edge_interaction == "none":
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "model_name": config.model_name,
                "architecture": "gatv2",
                "graph_schema": "graph_c_context_observation",
                "split": "test",
                "context_edge_interaction": config.context_edge_interaction,
                "interaction_edge_dim": config.interaction_edge_dim,
                "candidate_edges": 3,
                "edge_embed_l2_mean": 0.8,
                "interaction_vector_l2_mean": 0.9,
                "classifier_candidate_edge_attr_abs_sum": 1.2,
                "film_gamma_mean": 0.1,
                "film_gamma_std": 0.05,
                "film_beta_mean": 0.0,
                "film_beta_std": 0.02,
            }
        ]
    )
