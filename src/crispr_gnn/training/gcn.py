"""Sprint 4 minimal GCN training for validated graph schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from crispr_gnn.evaluation.metrics import binary_classification_metrics, select_threshold_by_f1
from crispr_gnn.graph.graph_schemas import GRAPH_A, GRAPH_B, GRAPH_C
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
    GRAPH_C_EDGE_TYPE,
    GRAPH_C_TARGET_REPRESENTATION_POLICY,
    TARGET_REPRESENTATION_POLICY,
    GraphAEdgeGCN,
    GraphBEdgeGCN,
    GraphCEdgeGCN,
    graph_a_edge_feature_attrs,
    graph_a_feature_dimensions,
    graph_b_edge_feature_attrs,
    graph_b_feature_dimensions,
    graph_c_edge_feature_attrs,
    graph_c_feature_dimensions,
)
from crispr_gnn.models.gat import GraphAEdgeGAT, GraphAEdgeGATv2, GraphBEdgeGATv2, GraphCEdgeGATv2
from crispr_gnn.models.losses import SUPPORTED_LOSSES, build_loss
from crispr_gnn.training.samplers import balanced_subsample_mask


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
    architecture: str = "gcn"
    feature_set: str = "S1_pair+F1"
    edge_feature_sets: tuple[str, ...] = ("s1_pair", "f1")
    target_node_representation: str = TARGET_REPRESENTATION_POLICY
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    attention_heads: int = 4
    attention_concat: bool = True
    attention_dropout: float | None = None
    edge_aware_attention: bool = True
    self_loop_edge_fill: float = 0.0
    gatv2_share_weights: bool = False
    drop_context_similarity_edges: bool = False
    edge_blind_candidate_attention: bool = False
    mask_target_observation_features: bool = False
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
    loss_params: Mapping[str, Any] = field(default_factory=dict)
    sampling: Mapping[str, Any] | None = None


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


def run_gcn_graph_c_from_config(
    config: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load canonical Sprint 3 Graph C artifacts and train the minimal GCN path."""
    validate_gcn_headline_config(config)
    run_config = gcn_run_config_from_mapping(config)
    graph_dir = root / str(config["data"]["graph_artifact_dir"])
    materialized = Sprint3HeteroDataLoader(graph_dir).load(GRAPH_C)
    return train_graph_c_gcn(materialized, run_config)


def gcn_run_config_from_mapping(config: Mapping[str, Any]) -> GCNRunConfig:
    data = _mapping(config.get("data", {}), "data")
    graph = _mapping(config.get("graph", {}), "graph")
    model = _mapping(config.get("model", {}), "model")
    training = _mapping(config.get("training", {}), "training")
    features = _mapping(config.get("features", {}), "features")
    graph_schema = str(graph.get("schema", GRAPH_A))
    if graph_schema not in {GRAPH_A, GRAPH_B, GRAPH_C}:
        raise ValueError("Sprint 4 GCN training supports Graph A, Graph B, and Graph C only")
    default_edge_features = ["candidate_pair_features"] if graph_schema == GRAPH_C else ["s1_pair", "f1"]
    edge_feature_sets = tuple(str(value) for value in features.get("edge_feature_sets", default_edge_features))
    if not edge_feature_sets:
        raise ValueError("At least one GCN edge feature set must be configured")
    expected_target_policy = (
        GRAPH_C_TARGET_REPRESENTATION_POLICY if graph_schema == GRAPH_C else TARGET_REPRESENTATION_POLICY
    )
    target_policy = str(model.get("target_node_representation", expected_target_policy))
    if graph_schema in {GRAPH_A, GRAPH_B} and target_policy != TARGET_REPRESENTATION_POLICY:
        raise ValueError("Graph A/B target-node representation must be the approved zero/type policy")
    if graph_schema == GRAPH_C and target_policy != GRAPH_C_TARGET_REPRESENTATION_POLICY:
        raise ValueError("Graph C target-node representation must use the observation context encoder")
    architecture = str(model.get("architecture", "gcn")).lower()
    if architecture not in {"gcn", "gat", "gatv2"}:
        raise ValueError("model.architecture must be one of: gcn, gat, gatv2")
    if architecture == "gat" and graph_schema != GRAPH_A:
        raise ValueError("GAT architecture is implemented for Graph A only; Sprint 7B uses GATv2 for Graph B/C")
    if architecture in {"gat", "gatv2"}:
        if target_policy != TARGET_REPRESENTATION_POLICY:
            if graph_schema != GRAPH_C:
                raise ValueError("Graph A/B GAT/GATv2 must keep zero_type_feature physical targets")
    attention = _mapping(model.get("attention", {}), "model.attention")
    loss = str(training.get("loss", "weighted_bce"))
    if loss.lower() not in SUPPORTED_LOSSES:
        raise ValueError(
            f"Unsupported GCN loss '{loss}'. Predeclared losses: {sorted(SUPPORTED_LOSSES)}"
        )
    loss = loss.lower()
    loss_params = training.get("loss_params", {})
    if not isinstance(loss_params, Mapping):
        raise ValueError("training.loss_params must be a mapping")
    sampling = training.get("sampling")
    if sampling is not None and not isinstance(sampling, Mapping):
        raise ValueError("training.sampling must be a mapping or null")
    scheduler = str(training.get("scheduler", "reduce_on_plateau"))
    if scheduler not in {"reduce_on_plateau", "none"}:
        raise ValueError("Sprint 4 scheduler must be 'reduce_on_plateau' or 'none'")
    return GCNRunConfig(
        sprint=str(config.get("sprint", "sprint4")),
        split_id=str(data.get("split_id", SPLIT_ID)),
        seed=int(config.get("seed", 42)),
        graph_schema=graph_schema,
        label_scheme=str(data.get("label_scheme", LABEL_SCHEME)),
        visibility_policy=str(graph.get("visibility_policy", VISIBILITY_POLICY)),
        model_name=str(model.get("name", "gcn_graph_a")),
        architecture=architecture,
        feature_set=str(
            features.get("feature_set", "+".join(_display_feature_name(value) for value in edge_feature_sets))
        ),
        edge_feature_sets=edge_feature_sets,
        target_node_representation=target_policy,
        hidden_dim=int(model.get("hidden_dim", 128)),
        num_layers=int(model.get("num_layers", 2)),
        dropout=float(model.get("dropout", 0.2)),
        attention_heads=int(attention.get("heads", model.get("attention_heads", 4))),
        attention_concat=bool(attention.get("concat", model.get("attention_concat", True))),
        attention_dropout=(
            None
            if attention.get("dropout", model.get("attention_dropout")) is None
            else float(attention.get("dropout", model.get("attention_dropout")))
        ),
        edge_aware_attention=bool(attention.get("edge_aware", model.get("edge_aware_attention", True))),
        self_loop_edge_fill=float(attention.get("self_loop_edge_fill", model.get("self_loop_edge_fill", 0.0))),
        gatv2_share_weights=bool(attention.get("gatv2_share_weights", model.get("gatv2_share_weights", False))),
        drop_context_similarity_edges=bool(
            attention.get("drop_context_similarity_edges", model.get("drop_context_similarity_edges", False))
        ),
        edge_blind_candidate_attention=bool(
            attention.get("edge_blind_candidate_attention", model.get("edge_blind_candidate_attention", False))
        ),
        mask_target_observation_features=bool(
            model.get("mask_target_observation_features", graph.get("mask_target_observation_features", False))
        ),
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
        loss_params=dict(loss_params),
        sampling=dict(sampling) if sampling is not None else None,
    )


def train_graph_a_gcn(
    materialized: MaterializedGraph,
    config: GCNRunConfig,
    *,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if materialized.graph_name != GRAPH_A or config.graph_schema != GRAPH_A:
        raise ValueError("Graph A training requires Graph A materialized artifacts and config")
    return _train_gcn(materialized, config, checkpoint_path=checkpoint_path)


def collect_graph_a_attention_summary(
    materialized: MaterializedGraph,
    config: GCNRunConfig,
    *,
    checkpoint_path: Path,
    split: str = "test",
) -> pd.DataFrame:
    """Summarize GAT/GATv2 attention weights for interpretation-only artifacts."""
    if config.graph_schema != GRAPH_A:
        raise ValueError("Graph A attention summary requires Graph A materialized artifacts")
    return collect_graph_attention_summary(
        materialized,
        config,
        checkpoint_path=checkpoint_path,
        split=split,
    )


def collect_graph_attention_summary(
    materialized: MaterializedGraph,
    config: GCNRunConfig,
    *,
    checkpoint_path: Path,
    split: str = "test",
) -> pd.DataFrame:
    """Summarize GAT/GATv2 attention weights for interpretation-only artifacts."""
    if config.architecture not in {"gat", "gatv2"}:
        raise ValueError("Attention summaries are only available for GAT/GATv2 runs")
    if materialized.graph_name != config.graph_schema:
        raise ValueError("Attention summary materialized graph/config schema mismatch")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for attention summary: {checkpoint_path}")
    device = torch.device(config.device)
    edge_feature_attrs = _edge_feature_attrs(config)
    view = materialized.view(split).to(device)
    model, _edge_dim = _build_model(view, config, edge_feature_attrs)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(_normalize_checkpoint_state_dict(state))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(view, edge_feature_attrs=edge_feature_attrs, return_attention=True)
    if not isinstance(output, tuple):
        raise ValueError("Attention model did not return attention weights")
    _logits, attention_records = output
    return _attention_summary_frame(
        attention_records,
        sgrna_nodes=int(view["sgRNA"].num_nodes),
        model_name=config.model_name,
        architecture=config.architecture,
        split=split,
        graph_schema=config.graph_schema,
    )


def train_graph_c_gcn(
    materialized: MaterializedGraph,
    config: GCNRunConfig,
    *,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if materialized.graph_name != GRAPH_C or config.graph_schema != GRAPH_C:
        raise ValueError("Graph C training requires Graph C materialized artifacts and config")
    if config.target_node_representation != GRAPH_C_TARGET_REPRESENTATION_POLICY:
        raise ValueError("Graph C context must enter through the target_observation node encoder")
    return _train_gcn(materialized, config, checkpoint_path=checkpoint_path)


def train_graph_b_gcn(
    materialized: MaterializedGraph,
    config: GCNRunConfig,
    *,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if materialized.graph_name != GRAPH_B or config.graph_schema != GRAPH_B:
        raise ValueError("Graph B training requires Graph B materialized artifacts and config")
    if config.target_node_representation != TARGET_REPRESENTATION_POLICY:
        raise ValueError("Graph B physical target nodes must remain featureless (zero_type_feature)")
    return _train_gcn(materialized, config, checkpoint_path=checkpoint_path)


def _train_gcn(
    materialized: MaterializedGraph,
    config: GCNRunConfig,
    *,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.split_id != SPLIT_ID or config.label_scheme != LABEL_SCHEME:
        raise ValueError("GCN training config drift from frozen Sprint 2 contract")
    if config.visibility_policy != VISIBILITY_POLICY:
        raise ValueError("GCN training requires strict-inductive visibility")
    _set_determinism(config)
    device = torch.device(config.device)
    edge_type = _candidate_edge_type(config.graph_schema)
    edge_feature_attrs = _edge_feature_attrs(config)
    train_view = materialized.view("train").to(device)
    val_view = materialized.view("val").to(device)
    test_view = materialized.view("test").to(device)
    model, edge_dim = _build_model(train_view, config, edge_feature_attrs)
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
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
    # pos_weight is retained for the result-row provenance column and is the value
    # the weighted_bce objective uses (negatives/positives).
    pos_weight = _weighted_bce_pos_weight(train_labels).to(device)
    loss_fn = build_loss(config.loss, config.loss_params, train_labels=train_labels)
    sampler_spec = _balanced_sampler_spec(config.sampling)
    train_labels_np = (
        train_labels.detach().cpu().numpy() if sampler_spec is not None else None
    )

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
        train_mask = train_view[edge_type].supervision_mask
        sup_logits = logits[train_mask].float()
        sup_labels = train_labels
        if sampler_spec is not None:
            keep = balanced_subsample_mask(
                train_labels_np,
                target_ratio=sampler_spec["target_ratio"],
                seed=config.seed,
                epoch=epoch,
            )
            keep_mask = torch.as_tensor(keep, device=sup_logits.device)
            sup_logits = sup_logits[keep_mask]
            sup_labels = sup_labels[keep_mask]
        loss = loss_fn(sup_logits, sup_labels)
        loss.backward()
        if config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_grad_norm)
        optimizer.step()

        val_labels, val_scores = _scores_for_view(model, val_view, edge_feature_attrs, edge_type=edge_type)
        val_auprc = _safe_average_precision(val_labels, val_scores)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_use_amp):
                val_logits = model(val_view, edge_feature_attrs=edge_feature_attrs)
            val_mask = val_view[edge_type].supervision_mask
            val_sup_labels = val_view[edge_type].edge_label[val_mask]
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
    val_labels, val_scores = _scores_for_view(model, val_view, edge_feature_attrs, edge_type=edge_type)
    test_labels, test_scores = _scores_for_view(model, test_view, edge_feature_attrs, edge_type=edge_type)
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
        parameter_count=parameter_count,
        pos_weight=float(pos_weight.detach().cpu()),
    )
    predictions = _prediction_records(
        config,
        val_view,
        val_labels,
        val_scores,
        test_view,
        test_labels,
        test_scores,
        edge_type=edge_type,
    )
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
    parameter_count: int,
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
        "architecture": config.architecture,
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
        "edge_aware_attention": bool(config.edge_aware_attention) if config.architecture in {"gat", "gatv2"} else None,
        "attention_heads": int(config.attention_heads) if config.architecture in {"gat", "gatv2"} else None,
        "attention_concat": bool(config.attention_concat) if config.architecture in {"gat", "gatv2"} else None,
        "attention_dropout": (
            float(config.attention_dropout)
            if config.architecture in {"gat", "gatv2"} and config.attention_dropout is not None
            else None
        ),
        "self_loop_edge_fill": (
            float(config.self_loop_edge_fill) if config.architecture in {"gat", "gatv2"} else None
        ),
        "gatv2_share_weights": (
            bool(config.gatv2_share_weights) if config.architecture == "gatv2" else None
        ),
        "drop_context_similarity_edges": (
            bool(config.drop_context_similarity_edges)
            if config.graph_schema == GRAPH_C and config.architecture == "gatv2"
            else None
        ),
        "edge_blind_candidate_attention": (
            bool(config.edge_blind_candidate_attention)
            if config.graph_schema == GRAPH_C and config.architecture == "gatv2"
            else None
        ),
        "mask_target_observation_features": (
            bool(config.mask_target_observation_features)
            if config.graph_schema == GRAPH_C and config.architecture == "gatv2"
            else None
        ),
        "parameter_count": int(parameter_count),
        "baseline_reference": BASELINE_REFERENCE,
        "baseline_test_auprc": BASELINE_TEST_AUPRC,
        "baseline_test_auroc": BASELINE_TEST_AUROC,
        "baseline_test_mcc": BASELINE_TEST_MCC,
        "graph_artifact_manifest_schema": materialized.manifest.get("graph_name"),
        "graph_artifact_split_id": materialized.manifest.get("split_id"),
        "weighted_bce_pos_weight": float(pos_weight),
        "use_compile": config.use_compile,
        "use_amp": config.use_amp,
        "target_semantics": _target_semantics(config.graph_schema),
        "notes": _run_notes(config),
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
    edge_type: tuple[str, str, str],
) -> pd.DataFrame:
    rows = []
    for split, view, labels, scores in [
        ("val", val_view, val_labels, val_scores),
        ("test", test_view, test_labels, test_scores),
    ]:
        edge_store = view[edge_type]
        mask = edge_store.supervision_mask.tolist()
        sgrna_ids = _masked_audit(getattr(edge_store, "audit_sgrna_ids", []), mask)
        genome_vals = _masked_audit(getattr(edge_store, "audit_genome", []), mask)
        for index, (label, score, sgrna_id, genome) in enumerate(
            zip(labels, scores, sgrna_ids, genome_vals, strict=True)
        ):
            rows.append(
                {
                    "model_name": config.model_name,
                    "architecture": config.architecture,
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
    model: Any,
    data: Any,
    edge_feature_attrs: list[str],
    *,
    edge_type: tuple[str, str, str],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(data, edge_feature_attrs=edge_feature_attrs)
        mask = data[edge_type].supervision_mask
        scores = torch.sigmoid(logits[mask]).detach().cpu().numpy()
        labels = data[edge_type].edge_label[mask].detach().cpu().numpy().astype(int)
    return labels, scores


def _supervised_labels(data: Any) -> torch.Tensor:
    edge_store = data[_candidate_edge_type(data.graph_name)]
    return edge_store.edge_label[edge_store.supervision_mask]


def _candidate_edge_type(graph_schema: str) -> tuple[str, str, str]:
    if graph_schema in {GRAPH_A, GRAPH_B}:
        return GRAPH_A_EDGE_TYPE
    if graph_schema == GRAPH_C:
        return GRAPH_C_EDGE_TYPE
    raise ValueError(f"Unsupported GCN graph schema: {graph_schema}")


def _edge_feature_attrs(config: GCNRunConfig) -> list[str]:
    if config.graph_schema == GRAPH_A:
        return graph_a_edge_feature_attrs(config.edge_feature_sets)
    if config.graph_schema == GRAPH_B:
        return graph_b_edge_feature_attrs(config.edge_feature_sets)
    if config.graph_schema == GRAPH_C:
        return graph_c_edge_feature_attrs(config.edge_feature_sets)
    raise ValueError(f"Unsupported GCN graph schema: {config.graph_schema}")


def _build_model(
    train_view: Any,
    config: GCNRunConfig,
    edge_feature_attrs: list[str],
) -> tuple[
    GraphAEdgeGCN
    | GraphBEdgeGCN
    | GraphCEdgeGCN
    | GraphAEdgeGAT
    | GraphAEdgeGATv2
    | GraphBEdgeGATv2
    | GraphCEdgeGATv2,
    int,
]:
    if config.graph_schema == GRAPH_A:
        sgrna_dim, edge_dim = graph_a_feature_dimensions(train_view, edge_feature_attrs)
        if config.architecture == "gat":
            return (
                GraphAEdgeGAT(
                    sgrna_input_dim=sgrna_dim,
                    edge_input_dim=edge_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    heads=config.attention_heads,
                    concat=config.attention_concat,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    edge_aware_attention=config.edge_aware_attention,
                    self_loop_edge_fill=config.self_loop_edge_fill,
                ),
                edge_dim,
            )
        if config.architecture == "gatv2":
            return (
                GraphAEdgeGATv2(
                    sgrna_input_dim=sgrna_dim,
                    edge_input_dim=edge_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    heads=config.attention_heads,
                    concat=config.attention_concat,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    edge_aware_attention=config.edge_aware_attention,
                    self_loop_edge_fill=config.self_loop_edge_fill,
                    gatv2_share_weights=config.gatv2_share_weights,
                ),
                edge_dim,
            )
        return (
            GraphAEdgeGCN(
                sgrna_input_dim=sgrna_dim,
                edge_input_dim=edge_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            ),
            edge_dim,
        )
    if config.graph_schema == GRAPH_B:
        sgrna_dim, edge_dim = graph_b_feature_dimensions(train_view, edge_feature_attrs)
        if config.architecture == "gatv2":
            return (
                GraphBEdgeGATv2(
                    sgrna_input_dim=sgrna_dim,
                    edge_input_dim=edge_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    heads=config.attention_heads,
                    concat=config.attention_concat,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    edge_aware_attention=config.edge_aware_attention,
                    self_loop_edge_fill=config.self_loop_edge_fill,
                    gatv2_share_weights=config.gatv2_share_weights,
                ),
                edge_dim,
            )
        if config.architecture != "gcn":
            raise ValueError("Graph B supports GCN and Sprint 7B GATv2 only")
        return (
            GraphBEdgeGCN(
                sgrna_input_dim=sgrna_dim,
                edge_input_dim=edge_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            ),
            edge_dim,
        )
    if config.graph_schema == GRAPH_C:
        sgrna_dim, target_dim, edge_dim = graph_c_feature_dimensions(train_view, edge_feature_attrs)
        if config.architecture == "gatv2":
            return (
                GraphCEdgeGATv2(
                    sgrna_input_dim=sgrna_dim,
                    target_observation_input_dim=target_dim,
                    edge_input_dim=edge_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    heads=config.attention_heads,
                    concat=config.attention_concat,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    edge_aware_attention=config.edge_aware_attention,
                    self_loop_edge_fill=config.self_loop_edge_fill,
                    gatv2_share_weights=config.gatv2_share_weights,
                    drop_context_similarity_edges=config.drop_context_similarity_edges,
                    edge_blind_candidate_attention=config.edge_blind_candidate_attention,
                    mask_target_observation_features=config.mask_target_observation_features,
                ),
                edge_dim,
            )
        if config.architecture != "gcn":
            raise ValueError("Graph C supports GCN and Sprint 7B GATv2 only")
        return (
            GraphCEdgeGCN(
                sgrna_input_dim=sgrna_dim,
                target_observation_input_dim=target_dim,
                edge_input_dim=edge_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            ),
            edge_dim,
        )
    raise ValueError(f"Unsupported GCN graph schema: {config.graph_schema}")


def _target_semantics(graph_schema: str) -> str:
    if graph_schema == GRAPH_C:
        return "observation_level_context_target"
    return "minimal_physical_target"


def _run_notes(config: GCNRunConfig) -> str:
    graph_schema = config.graph_schema
    if config.architecture == "gat":
        return (
            "Graph A GATConv path; S5F2 edge features enter attention via edge_attr/edge_dim "
            "when edge_aware_attention=True; self-loop edge features are zero-filled; no test-driven selection"
        )
    if config.architecture == "gatv2":
        ablation_flags = []
        if config.drop_context_similarity_edges:
            ablation_flags.append("context_similar_to edges dropped")
        if config.edge_blind_candidate_attention:
            ablation_flags.append("candidate S5F2 zeroed only inside attention")
        if config.mask_target_observation_features:
            ablation_flags.append("direct target_observation features masked")
        ablation_note = f"; Sprint 7D ablation: {', '.join(ablation_flags)}" if ablation_flags else ""
        return (
            f"{config.graph_schema} GATv2Conv path with dynamic attention; candidate edge features enter attention via edge_attr/edge_dim "
            "when edge_aware_attention=True; self-loop edge features are zero-filled; no test-driven selection"
            f"{ablation_note}"
        )
    if graph_schema == GRAPH_C:
        return (
            "Graph C GCN path uses target_observation context node encoding; "
            "Graph C changes both topology and target semantics; no test-driven selection; no Graph B run"
        )
    if graph_schema == GRAPH_B:
        return (
            "Graph B bounded secondary control; adds label-free guide-similarity topology to Graph A; "
            "featureless physical targets and Graph A candidate features unchanged; no test-driven selection"
        )
    return "Graph A minimal GCN path; no test-driven selection; no Graph C/B run"


def _normalize_checkpoint_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(str(key).startswith("_orig_mod.") for key in state):
        return {str(key).removeprefix("_orig_mod."): value for key, value in state.items()}
    return {str(key): value for key, value in state.items()}


def _attention_summary_frame(
    attention_records: list[dict[str, torch.Tensor]],
    *,
    sgrna_nodes: int,
    model_name: str,
    architecture: str,
    split: str,
    graph_schema: str = GRAPH_A,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in attention_records:
        edge_index = record["edge_index"].detach().cpu()
        alpha = record["alpha"].detach().cpu()
        layer = int(record["layer"].detach().cpu())
        if alpha.ndim == 1:
            alpha = alpha[:, None]
        edge_kind = _attention_edge_kinds(edge_index, sgrna_nodes=sgrna_nodes, graph_schema=graph_schema)
        for head in range(alpha.shape[1]):
            values = alpha[:, head]
            for kind in sorted(set(edge_kind)):
                mask = torch.tensor([value == kind for value in edge_kind], dtype=torch.bool)
                kind_values = values[mask]
                if kind_values.numel() == 0:
                    continue
                rows.append(
                    {
                        "model_name": model_name,
                        "architecture": architecture,
                        "split": split,
                        "layer": layer,
                        "head": int(head),
                        "edge_kind": kind,
                        "edge_count": int(kind_values.numel()),
                        "attention_mean": float(kind_values.mean()),
                        "attention_std": float(kind_values.std(unbiased=False)),
                        "attention_min": float(kind_values.min()),
                        "attention_max": float(kind_values.max()),
                    }
                )
    return pd.DataFrame(rows)


def _attention_edge_kinds(edge_index: torch.Tensor, *, sgrna_nodes: int, graph_schema: str) -> list[str]:
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    kinds = []
    for source, target in zip(src, dst, strict=True):
        if source == target:
            kinds.append("self_loop")
        elif source < sgrna_nodes <= target:
            kinds.append("candidate_forward")
        elif target < sgrna_nodes <= source:
            kinds.append("candidate_reverse")
        elif graph_schema == GRAPH_B and source < sgrna_nodes and target < sgrna_nodes:
            kinds.append("sequence_similar_to")
        elif graph_schema == GRAPH_C and source >= sgrna_nodes and target >= sgrna_nodes:
            kinds.append("context_similar_to")
        else:
            kinds.append("other")
    return kinds


def _weighted_bce_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        return torch.tensor(1.0, dtype=torch.float32, device=labels.device)
    return (negatives / positives).float()


def _balanced_sampler_spec(sampling: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate and normalize the Sprint 6 training-time sampling spec.

    Only the predeclared measured-only ``balanced_subsample`` strategy is wired in
    Slice 1; anything else (e.g. hard-negative mining) is out of scope and rejected.
    Sampling affects training supervision only; validation/test views are untouched.
    """
    if not sampling:
        return None
    strategy = str(sampling.get("strategy", "")).lower()
    if strategy != "balanced_subsample":
        raise ValueError(
            f"Unsupported Sprint 6 sampling strategy '{strategy}'. "
            "Only 'balanced_subsample' is implemented in Slice 1."
        )
    scope = str(sampling.get("scope", "measured_only")).lower()
    if scope != "measured_only":
        raise ValueError("Sprint 6 sampling scope must be 'measured_only'")
    target_ratio = float(sampling.get("target_ratio", 1.0))
    if target_ratio <= 0:
        raise ValueError("Sprint 6 sampling target_ratio must be positive")
    return {"target_ratio": target_ratio}


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
    normalized = value.lower()
    if normalized == "s1_pair":
        return "S1_pair"
    if normalized in {"candidate_pair_features", "candidate_pair"}:
        return "CandidatePair"
    return value.upper()
