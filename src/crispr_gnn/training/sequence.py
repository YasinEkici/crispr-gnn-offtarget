"""PyTorch sequence-only Sprint 2 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

from crispr_gnn.data.splits import LABEL_COLUMN, SPLIT_COLUMN
from crispr_gnn.evaluation.metrics import binary_classification_metrics, select_threshold_by_f1
from crispr_gnn.features.sequence import SEQUENCE_FEATURE_SET, SEQUENCE_REPRESENTATION, build_sequence_pair_encoding
from crispr_gnn.features.tabular import FeatureSetName, TrainOnlyPreprocessor, audit_feature_columns, build_feature_set


@dataclass(frozen=True)
class SequenceRunConfig:
    sprint: str
    split_id: str
    seed: int
    label_scheme: str = "scheme_a"
    training_regime: str = "measured_only"
    max_length: int = 23
    batch_size: int = 512
    max_epochs: int = 120
    min_epochs: int = 15
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    cnn_channels: int = 64
    lstm_hidden_dim: int = 64
    lstm_layers: int = 1
    dropout: float = 0.2
    tabular_projection_dim: int = 32
    device: str = "cpu"
    num_threads: int = 1


def run_sequence_baselines(
    assigned: pd.DataFrame,
    config: SequenceRunConfig,
    *,
    models: list[str],
    include_balanced: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    split_data, audit = _prepared_sequence_split(assigned, config.max_length)
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    training_summary_rows: list[dict[str, object]] = []
    variants = [(model, f"{model}_unweighted") for model in models]
    if include_balanced:
        variants.extend((model, f"{model}_balanced_train_weights") for model in models)

    for model_type, model_name in variants:
        val_scores, test_scores, summary = _fit_sequence_model(model_type, model_name, split_data, config)
        training_summary_rows.append({"model_name": model_name, "feature_set": SEQUENCE_FEATURE_SET, **summary})
        selection = select_threshold_by_f1(split_data["y_val"], val_scores)
        rows.append(
            _result_row(
                model_name=model_name,
                config=config,
                split_data=split_data,
                val_scores=val_scores,
                test_scores=test_scores,
                threshold=selection.threshold,
                threshold_policy=selection.policy,
            )
        )
        predictions.extend(_prediction_records(model_name, SEQUENCE_FEATURE_SET, split_data, val_scores, test_scores))

    return pd.DataFrame(rows), predictions, pd.DataFrame(training_summary_rows), audit


def run_sequence_cnn_late_fusion_baselines(
    assigned: pd.DataFrame,
    config: SequenceRunConfig,
    *,
    tabular_feature_sets: list[FeatureSetName],
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    training_summary_rows: list[dict[str, object]] = []
    audit_frames: list[pd.DataFrame] = []

    for tabular_feature_set in tabular_feature_sets:
        split_data, audit = _prepared_late_fusion_split(assigned, config.max_length, tabular_feature_set)
        audit_frames.append(audit)
        feature_set = f"{SEQUENCE_FEATURE_SET}+{tabular_feature_set}"
        model_name = f"sequence_cnn_plus_{tabular_feature_set}_late_fusion_unweighted"
        val_scores, test_scores, summary = _fit_sequence_cnn_late_fusion_model(model_name, split_data, config)
        training_summary_rows.append({"model_name": model_name, "feature_set": feature_set, **summary})
        selection = select_threshold_by_f1(split_data["y_val"], val_scores)
        rows.append(
            _result_row(
                model_name=model_name,
                feature_set=feature_set,
                feature_columns=int(split_data["X_seq_train"].shape[1] * split_data["X_seq_train"].shape[2] + split_data["X_tab_train"].shape[1]),
                notes=(
                    f"{model_name}; late fusion of {SEQUENCE_REPRESENTATION} and {tabular_feature_set}; "
                    "measured-only main split; experiment_id=18 excluded"
                ),
                config=config,
                split_data=split_data,
                val_scores=val_scores,
                test_scores=test_scores,
                threshold=selection.threshold,
                threshold_policy=selection.policy,
            )
        )
        predictions.extend(_prediction_records(model_name, feature_set, split_data, val_scores, test_scores))

    return pd.DataFrame(rows), predictions, pd.DataFrame(training_summary_rows), pd.concat(audit_frames, ignore_index=True)


def _prepared_sequence_split(assigned: pd.DataFrame, max_length: int) -> tuple[dict[str, Any], pd.DataFrame]:
    encoded = build_sequence_pair_encoding(assigned, max_length=max_length)
    data = pd.DataFrame(
        {
            SPLIT_COLUMN: assigned[SPLIT_COLUMN].to_numpy(),
            LABEL_COLUMN: assigned[LABEL_COLUMN].to_numpy(dtype=int),
            "encoded_position": np.arange(assigned.shape[0]),
        },
        index=assigned.index,
    )
    train = data.loc[data[SPLIT_COLUMN] == "train"]
    val = data.loc[data[SPLIT_COLUMN] == "val"]
    test = data.loc[data[SPLIT_COLUMN] == "test"]
    if train.empty or val.empty or test.empty:
        raise ValueError("Train, validation, and test splits must all be non-empty")
    return {
        "X_train": encoded.encoded[train["encoded_position"].to_numpy(dtype=int)],
        "X_val": encoded.encoded[val["encoded_position"].to_numpy(dtype=int)],
        "X_test": encoded.encoded[test["encoded_position"].to_numpy(dtype=int)],
        "y_train": train[LABEL_COLUMN].to_numpy(dtype=int),
        "y_val": val[LABEL_COLUMN].to_numpy(dtype=int),
        "y_test": test[LABEL_COLUMN].to_numpy(dtype=int),
        "train_index": train.index.to_numpy(),
        "val_index": val.index.to_numpy(),
        "test_index": test.index.to_numpy(),
    }, encoded.audit


def _prepared_late_fusion_split(
    assigned: pd.DataFrame,
    max_length: int,
    tabular_feature_set: FeatureSetName,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sequence_split, sequence_audit = _prepared_sequence_split(assigned, max_length)
    tabular_features = build_feature_set(assigned, tabular_feature_set)
    data = pd.concat([assigned[[SPLIT_COLUMN, LABEL_COLUMN]], tabular_features], axis=1)
    feature_columns = list(tabular_features.columns)
    train = data.loc[data[SPLIT_COLUMN] == "train"]
    val = data.loc[data[SPLIT_COLUMN] == "val"]
    test = data.loc[data[SPLIT_COLUMN] == "test"]
    preprocessor = TrainOnlyPreprocessor(scale=True).fit(train[feature_columns])
    sequence_split.update(
        {
            "X_seq_train": sequence_split.pop("X_train"),
            "X_seq_val": sequence_split.pop("X_val"),
            "X_seq_test": sequence_split.pop("X_test"),
            "X_tab_train": preprocessor.transform(train[feature_columns]).to_numpy(),
            "X_tab_val": preprocessor.transform(val[feature_columns]).to_numpy(),
            "X_tab_test": preprocessor.transform(test[feature_columns]).to_numpy(),
            "tabular_feature_columns": feature_columns,
        }
    )
    sequence_audit = sequence_audit.copy()
    sequence_audit["input_branch"] = "sequence"
    sequence_audit["tabular_feature_set"] = tabular_feature_set
    sequence_audit["tabular_preprocessor_fit_scope"] = "train_only"
    tabular_audit = audit_feature_columns(tabular_feature_set, feature_columns)
    tabular_audit = tabular_audit.rename(columns={"feature": "source_column"})
    tabular_audit["input_representation"] = f"{SEQUENCE_REPRESENTATION}+{tabular_feature_set}"
    tabular_audit["input_branch"] = "tabular"
    tabular_audit["max_length"] = max_length
    tabular_audit["channels"] = np.nan
    tabular_audit["policy"] = f"train-only preprocessed tabular {tabular_feature_set} late-fusion branch"
    tabular_audit["tabular_feature_set"] = tabular_feature_set
    tabular_audit["tabular_preprocessor_fit_scope"] = "train_only"
    return sequence_split, pd.concat([sequence_audit, tabular_audit], ignore_index=True, sort=False)


def _fit_sequence_model(
    model_type: str,
    model_name: str,
    split_data: dict[str, Any],
    config: SequenceRunConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class SequenceCNN(nn.Module):
        def __init__(self, input_channels: int, hidden_channels: int, dropout: float) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 1),
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.network(features.transpose(1, 2)).squeeze(-1)

    class SequenceBiLSTM(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
            super().__init__()
            lstm_dropout = dropout if layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
                dropout=lstm_dropout,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, 1),
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(features)
            pooled = output.mean(dim=1)
            return self.head(pooled).squeeze(-1)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(config.num_threads)
    device = torch.device(config.device)
    input_dim = int(split_data["X_train"].shape[2])
    if model_type == "sequence_cnn":
        model = SequenceCNN(input_channels=input_dim, hidden_channels=config.cnn_channels, dropout=config.dropout)
    elif model_type == "sequence_bilstm":
        model = SequenceBiLSTM(
            input_dim=input_dim,
            hidden_dim=config.lstm_hidden_dim,
            layers=config.lstm_layers,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unsupported sequence model type: {model_type}")
    model = model.to(device)

    train_weights = None
    if model_name.endswith("_balanced_train_weights"):
        train_weights = compute_sample_weight(class_weight="balanced", y=split_data["y_train"]).astype(np.float32)
    train_dataset = TensorDataset(
        torch.as_tensor(np.array(split_data["X_train"], copy=True), dtype=torch.float32),
        torch.as_tensor(np.array(split_data["y_train"], copy=True), dtype=torch.float32),
        torch.as_tensor(
            np.array(train_weights if train_weights is not None else np.ones_like(split_data["y_train"]), copy=True),
            dtype=torch.float32,
        ),
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)

    X_val = torch.as_tensor(np.array(split_data["X_val"], copy=True), dtype=torch.float32, device=device)
    X_test = torch.as_tensor(np.array(split_data["X_test"], copy=True), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best_state: dict[str, torch.Tensor] | None = None
    best_val_auprc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    final_train_loss = float("nan")

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for features, labels, weights in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            batch_loss = loss_fn(model(features), labels)
            loss = (batch_loss * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_train_loss = float(np.mean(losses))

        val_scores = _torch_scores(model, X_val)
        val_auprc = float(average_precision_score(split_data["y_val"], val_scores))
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
    val_scores = _torch_scores(model, X_val)
    test_scores = _torch_scores(model, X_test)
    return val_scores, test_scores, {
        "input_representation": SEQUENCE_REPRESENTATION,
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch),
        "best_val_auprc": float(best_val_auprc),
        "final_train_loss": float(final_train_loss),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "patience": int(config.patience),
        "device": str(device),
        "num_threads": int(config.num_threads),
    }


def _fit_sequence_cnn_late_fusion_model(
    model_name: str,
    split_data: dict[str, Any],
    config: SequenceRunConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class SequenceCNNLateFusion(nn.Module):
        def __init__(self, input_channels: int, tabular_dim: int, hidden_channels: int, tabular_projection_dim: int, dropout: float) -> None:
            super().__init__()
            self.sequence_encoder = nn.Sequential(
                nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
            )
            self.tabular_encoder = nn.Sequential(
                nn.Linear(tabular_dim, tabular_projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_channels + tabular_projection_dim, 1),
            )

        def forward(self, sequence_features: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
            sequence_embedding = self.sequence_encoder(sequence_features.transpose(1, 2))
            tabular_embedding = self.tabular_encoder(tabular_features)
            return self.head(torch.cat([sequence_embedding, tabular_embedding], dim=1)).squeeze(-1)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(config.num_threads)
    device = torch.device(config.device)
    model = SequenceCNNLateFusion(
        input_channels=int(split_data["X_seq_train"].shape[2]),
        tabular_dim=int(split_data["X_tab_train"].shape[1]),
        hidden_channels=config.cnn_channels,
        tabular_projection_dim=config.tabular_projection_dim,
        dropout=config.dropout,
    ).to(device)

    train_dataset = TensorDataset(
        torch.as_tensor(np.array(split_data["X_seq_train"], copy=True), dtype=torch.float32),
        torch.as_tensor(np.array(split_data["X_tab_train"], copy=True), dtype=torch.float32),
        torch.as_tensor(np.array(split_data["y_train"], copy=True), dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    X_seq_val = torch.as_tensor(np.array(split_data["X_seq_val"], copy=True), dtype=torch.float32, device=device)
    X_tab_val = torch.as_tensor(np.array(split_data["X_tab_val"], copy=True), dtype=torch.float32, device=device)
    X_seq_test = torch.as_tensor(np.array(split_data["X_seq_test"], copy=True), dtype=torch.float32, device=device)
    X_tab_test = torch.as_tensor(np.array(split_data["X_tab_test"], copy=True), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_val_auprc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    final_train_loss = float("nan")

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for sequence_features, tabular_features, labels in train_loader:
            sequence_features = sequence_features.to(device)
            tabular_features = tabular_features.to(device)
            labels = labels.to(device)
            loss = loss_fn(model(sequence_features, tabular_features), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_train_loss = float(np.mean(losses))

        val_scores = _torch_late_fusion_scores(model, X_seq_val, X_tab_val)
        val_auprc = float(average_precision_score(split_data["y_val"], val_scores))
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
    val_scores = _torch_late_fusion_scores(model, X_seq_val, X_tab_val)
    test_scores = _torch_late_fusion_scores(model, X_seq_test, X_tab_test)
    return val_scores, test_scores, {
        "input_representation": f"{SEQUENCE_REPRESENTATION}+tabular_late_fusion",
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch),
        "best_val_auprc": float(best_val_auprc),
        "final_train_loss": float(final_train_loss),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "patience": int(config.patience),
        "cnn_channels": int(config.cnn_channels),
        "tabular_projection_dim": int(config.tabular_projection_dim),
        "device": str(device),
        "num_threads": int(config.num_threads),
    }


def _torch_scores(model: Any, features: Any) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(features)).detach().cpu().numpy()


def _torch_late_fusion_scores(model: Any, sequence_features: Any, tabular_features: Any) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(sequence_features, tabular_features)).detach().cpu().numpy()


def _result_row(
    *,
    model_name: str,
    config: SequenceRunConfig,
    split_data: dict[str, Any],
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    threshold_policy: str,
    feature_set: str = SEQUENCE_FEATURE_SET,
    feature_columns: int | None = None,
    notes: str | None = None,
) -> dict[str, object]:
    val_metrics = binary_classification_metrics(split_data["y_val"], val_scores, threshold, prefix="val_")
    test_metrics = binary_classification_metrics(split_data["y_test"], test_scores, threshold, prefix="test_")
    return {
        "sprint": config.sprint,
        "label_scheme": config.label_scheme,
        "split_id": config.split_id,
        "seed": config.seed,
        "training_regime": config.training_regime,
        "model_name": model_name,
        "feature_set": feature_set,
        "train_rows": int(split_data["y_train"].shape[0]),
        "val_rows": int(split_data["y_val"].shape[0]),
        "test_rows": int(split_data["y_test"].shape[0]),
        "feature_columns": feature_columns if feature_columns is not None else int(split_data["X_train"].shape[1] * split_data["X_train"].shape[2]),
        "threshold_policy": threshold_policy,
        "threshold": float(threshold),
        "notes": notes or f"{model_name}; sequence-only {SEQUENCE_REPRESENTATION}; measured-only main split; experiment_id=18 excluded",
        **val_metrics,
        **test_metrics,
    }


def _prediction_records(
    model_name: str,
    feature_set: str,
    split_data: dict[str, Any],
    val_scores: np.ndarray,
    test_scores: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "split": "val",
            "row_index": split_data["val_index"],
            "y_true": split_data["y_val"],
            "y_score": val_scores,
        },
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "split": "test",
            "row_index": split_data["test_index"],
            "y_true": split_data["y_test"],
            "y_score": test_scores,
        },
    ]
