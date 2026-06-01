"""Sprint 4 minimal Graph A GCN training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from crispr_gnn.evaluation.metrics import binary_classification_metrics, select_threshold_by_f1
from crispr_gnn.graph.graph_schemas import GRAPH_A
from crispr_gnn.graph.pyg_dataset import (
    LABEL_SCHEME,
    SPLIT_ID,
    VISIBILITY_POLICY,
    MaterializedGraph,
    Sprint3HeteroDataLoader,
    validate_gcn_headline_config,
)
from crispr_gnn.models.gcn import (
    GRAPH_A_EDGE_TYPE,
    TARGET_REPRESENTATION_POLICY,
    GraphAEdgeGCN,
    graph_a_edge_feature_attrs,
    graph_a_feature_dimensions,
)


BASELINE_REFERENCE = "xgboost_unweighted / F4"
BASELINE_TEST_AUPRC = 0.992522
BASELINE_TEST_AUROC = 0.938416
BASELINE_TEST_MCC = 0.345198
CHECKPOINT_POLICY = "validation_auprc"


@dataclass(frozen=True)
class GCNRunConfig:
    sprint: str
    split_id: str
    seed: int
    graph_schema: str = GRAPH_A
    label_scheme: str = LABEL_SCHEME
    visibility_policy: str = VISIBILITY_POLICY
    model_name: str = "gcn_graph_a"
    feature_set: str = "S1_pair+F1"
    edge_feature_sets: tuple[str, ...] = ("s1_pair", "f1")
    target_node_representation: str = TARGET_REPRESENTATION_POLICY
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    loss: str = "weighted_bce"
    clip_grad_norm: float = 1.0
    scheduler: str = "reduce_on_plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-5
    max_epochs: int = 100
    min_epochs: int = 5
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    num_threads: int = 1
    use_compile: bool = False
    use_amp: bool = False


def run_gcn_graph_a_from_config(
    config: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load canonical Sprint 3 Graph A artifacts and train the minimal GCN path."""
    validate_gcn_headline_config(config)
    run_config = gcn_run_config_from_mapping(config)
    graph_dir = root / str(config["data"]["graph_artifact_dir"])
    materialized = Sprint3HeteroDataLoader(graph_dir).load(GRAPH_A)
    return train_graph_a_gcn(materialized, run_config)


def gcn_run_config_from_mapping(config: Mapping[str, Any]) -> GCNRunConfig:
    data = _mapping(config.get("data", {}), "data")
    graph = _mapping(config.get("graph", {}), "graph")
    model = _mapping(config.get("model", {}), "model")
    training = _mapping(config.get("training", {}), "training")
    features = _mapping(config.get("features", {}), "features")
    if graph.get("schema") != GRAPH_A:
        raise ValueError("Slice 2 only supports Graph A training")
    edge_feature_sets = tuple(str(value) for value in features.get("edge_feature_sets", ["s1_pair", "f1"]))
    if not edge_feature_sets:
        raise ValueError("At least one Graph A edge feature set must be configured")
    target_policy = str(model.get("target_node_representation", TARGET_REPRESENTATION_POLICY))
    if target_policy != TARGET_REPRESENTATION_POLICY:
        raise ValueError("Graph A target-node representation must be the approved zero/type policy")
    loss = str(training.get("loss", "weighted_bce"))
    if loss != "weighted_bce":
        raise ValueError("Sprint 4 Slice 2 uses weighted BCE only")
    scheduler = str(training.get("scheduler", "reduce_on_plateau"))
    if scheduler not in {"reduce_on_plateau", "none"}:
        raise ValueError("Sprint 4 scheduler must be 'reduce_on_plateau' or 'none'")
    return GCNRunConfig(
        sprint=str(config.get("sprint", "sprint4")),
        split_id=str(data.get("split_id", SPLIT_ID)),
        seed=int(config.get("seed", 42)),
        graph_schema=str(graph.get("schema", GRAPH_A)),
        label_scheme=str(data.get("label_scheme", LABEL_SCHEME)),
        visibility_policy=str(graph.get("visibility_policy", VISIBILITY_POLICY)),
        model_name=str(model.get("name", "gcn_graph_a")),
        feature_set="+".join(_display_feature_name(value) for value in edge_feature_sets),
        edge_feature_sets=edge_feature_sets,
        target_node_representation=target_policy,
        hidden_dim=int(model.get("hidden_dim", 128)),
        num_layers=int(model.get("num_layers", 2)),
        dropout=float(model.get("dropout", 0.2)),
        loss=loss,
        clip_grad_norm=float(training.get("clip_grad_norm", 1.0)),
        scheduler=scheduler,
        scheduler_factor=float(training.get("scheduler_factor", 0.5)),
        scheduler_patience=int(training.get("scheduler_patience", 5)),
        scheduler_min_lr=float(training.get("scheduler_min_lr", 1e-5)),
        max_epochs=int(training.get("max_epochs", 100)),
        min_epochs=int(training.get("min_epochs", 5)),
        patience=int(training.get("patience", 10)),
        learning_rate=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        device=str(training.get("device", "cpu")),
        num_threads=int(training.get("num_threads", 1)),
        use_compile=bool(training.get("use_compile", False)),
        use_amp=bool(training.get("use_amp", False)),
    )


def train_graph_a_gcn(
    materialized: MaterializedGraph,
    config: GCNRunConfig,
    *,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if materialized.graph_name != GRAPH_A or config.graph_schema != GRAPH_A:
        raise ValueError("Slice 2 training is limited to Graph A")
    if config.split_id != SPLIT_ID or config.label_scheme != LABEL_SCHEME:
        raise ValueError("GCN training config drift from frozen Sprint 2 contract")
    if config.visibility_policy != VISIBILITY_POLICY:
        raise ValueError("GCN training requires strict-inductive visibility")
    _set_determinism(config)
    device = torch.device(config.device)
    edge_feature_attrs = graph_a_edge_feature_attrs(config.edge_feature_sets)
    train_view = materialized.view("train").to(device)
    val_view = materialized.view("val").to(device)
    test_view = materialized.view("test").to(device)
    sgrna_dim, edge_dim = graph_a_feature_dimensions(train_view, edge_feature_attrs)
    model = GraphAEdgeGCN(
        sgrna_input_dim=sgrna_dim,
        edge_input_dim=edge_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    _use_amp = config.use_amp and device.type == "cuda"
    if config.use_compile and device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if config.scheduler == "reduce_on_plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )
    train_labels = _supervised_labels(train_view)
    pos_weight = _weighted_bce_pos_weight(train_labels).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_auprc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    final_epoch = 0

    for epoch in range(1, config.max_epochs + 1):
        final_epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_use_amp):
            logits = model(train_view, edge_feature_attrs=edge_feature_attrs)
        train_mask = train_view[GRAPH_A_EDGE_TYPE].supervision_mask
        loss = loss_fn(logits[train_mask].float(), train_labels)
        loss.backward()
        if config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_grad_norm)
        optimizer.step()

        val_labels, val_scores = _scores_for_view(model, val_view, edge_feature_attrs)
        val_auprc = _safe_average_precision(val_labels, val_scores)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_use_amp):
                val_logits = model(val_view, edge_feature_attrs=edge_feature_attrs)
            val_mask = val_view[GRAPH_A_EDGE_TYPE].supervision_mask
            val_sup_labels = val_view[GRAPH_A_EDGE_TYPE].edge_label[val_mask]
            val_loss_value = float(loss_fn(val_logits[val_mask].float(), val_sup_labels).detach().cpu())
        if lr_scheduler is not None:
            lr_scheduler.step(val_auprc)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(loss.detach().cpu()),
                "val_loss": val_loss_value,
                "val_auprc": float(val_auprc),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "selection_split": "validation",
            }
        )
        if val_auprc > best_val_auprc + 1e-6:
            best_val_auprc = val_auprc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if epoch >= config.min_epochs and epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
    model = model.to(device)
    val_labels, val_scores = _scores_for_view(model, val_view, edge_feature_attrs)
    test_labels, test_scores = _scores_for_view(model, test_view, edge_feature_attrs)
    threshold = select_threshold_by_f1(val_labels, val_scores)
    result = _result_row(
        config=config,
        materialized=materialized,
        val_labels=val_labels,
        val_scores=val_scores,
        test_labels=test_labels,
        test_scores=test_scores,
        threshold=float(threshold.threshold),
        threshold_policy=threshold.policy,
        best_epoch=best_epoch,
        epochs_ran=final_epoch,
        best_val_auprc=float(best_val_auprc),
        edge_dim=edge_dim,
        pos_weight=float(pos_weight.detach().cpu()),
    )
    predictions = _prediction_records(config, val_view, val_labels, val_scores, test_view, test_labels, test_scores)
    training_history = pd.DataFrame(history)
    training_history.insert(0, "model_name", config.model_name)
    training_history.insert(1, "graph_schema", config.graph_schema)
    return pd.DataFrame([result]), predictions, training_history


def _result_row(
    *,
    config: GCNRunConfig,
    materialized: MaterializedGraph,
    val_labels: np.ndarray,
    val_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    threshold_policy: str,
    best_epoch: int,
    epochs_ran: int,
    best_val_auprc: float,
    edge_dim: int,
    pos_weight: float,
) -> dict[str, object]:
    val_metrics = binary_classification_metrics(val_labels, val_scores, threshold, prefix="val_")
    test_metrics = binary_classification_metrics(test_labels, test_scores, threshold, prefix="test_")
    return {
        "sprint": config.sprint,
        "label_scheme": config.label_scheme,
        "split_id": config.split_id,
        "seed": config.seed,
        "training_regime": "measured_only",
        "model_name": config.model_name,
        "feature_set": config.feature_set,
        "graph_schema": config.graph_schema,
        "visibility_policy": config.visibility_policy,
        "target_node_representation": config.target_node_representation,
        "loss": config.loss,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "checkpoint_selection_split": "validation",
        "threshold_policy": threshold_policy,
        "threshold_selection_split": "validation",
        "threshold": float(threshold),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epochs_ran),
        "best_val_auprc": float(best_val_auprc),
        "edge_feature_sets": ",".join(config.edge_feature_sets),
        "edge_feature_columns": int(edge_dim),
        "baseline_reference": BASELINE_REFERENCE,
        "baseline_test_auprc": BASELINE_TEST_AUPRC,
        "baseline_test_auroc": BASELINE_TEST_AUROC,
        "baseline_test_mcc": BASELINE_TEST_MCC,
        "graph_artifact_manifest_schema": materialized.manifest.get("graph_name"),
        "graph_artifact_split_id": materialized.manifest.get("split_id"),
        "weighted_bce_pos_weight": float(pos_weight),
        "use_compile": config.use_compile,
        "use_amp": config.use_amp,
        "notes": "Graph A minimal GCN smoke-capable path; no test-driven selection; no Graph C/B run",
        **val_metrics,
        **test_metrics,
    }


def _prediction_records(
    config: GCNRunConfig,
    val_view: Any,
    val_labels: np.ndarray,
    val_scores: np.ndarray,
    test_view: Any,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for split, view, labels, scores in [
        ("val", val_view, val_labels, val_scores),
        ("test", test_view, test_labels, test_scores),
    ]:
        edge_store = view[GRAPH_A_EDGE_TYPE]
        mask = edge_store.supervision_mask.tolist()
        sgrna_ids = _masked_audit(getattr(edge_store, "audit_sgrna_ids", []), mask)
        genome_vals = _masked_audit(getattr(edge_store, "audit_genome", []), mask)
        for index, (label, score, sgrna_id, genome) in enumerate(
            zip(labels, scores, sgrna_ids, genome_vals, strict=True)
        ):
            rows.append(
                {
                    "model_name": config.model_name,
                    "graph_schema": config.graph_schema,
                    "feature_set": config.feature_set,
                    "split": split,
                    "row_index": int(index),
                    "grna_target_id": str(sgrna_id) if sgrna_id is not None else None,
                    "genome": str(genome) if genome is not None else None,
                    "label": int(label),
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows)


def _masked_audit(items: list[Any], mask: list[bool]) -> list[Any]:
    if not items:
        return [None] * sum(mask)
    return [item for item, m in zip(items, mask, strict=True) if m]


def _scores_for_view(
    model: GraphAEdgeGCN,
    data: Any,
    edge_feature_attrs: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(data, edge_feature_attrs=edge_feature_attrs)
        mask = data[GRAPH_A_EDGE_TYPE].supervision_mask
        scores = torch.sigmoid(logits[mask]).detach().cpu().numpy()
        labels = data[GRAPH_A_EDGE_TYPE].edge_label[mask].detach().cpu().numpy().astype(int)
    return labels, scores


def _supervised_labels(data: Any) -> torch.Tensor:
    edge_store = data[GRAPH_A_EDGE_TYPE]
    return edge_store.edge_label[edge_store.supervision_mask]


def _weighted_bce_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        return torch.tensor(1.0, dtype=torch.float32, device=labels.device)
    return (negatives / positives).float()


def _set_determinism(config: GCNRunConfig) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(config.num_threads)


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if int(np.asarray(labels).sum()) == 0:
        return 0.0
    return float(average_precision_score(labels, scores))


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"GCN config section must be a mapping: {name}")
    return value


def _display_feature_name(value: str) -> str:
    return "S1_pair" if value.lower() == "s1_pair" else value.upper()
