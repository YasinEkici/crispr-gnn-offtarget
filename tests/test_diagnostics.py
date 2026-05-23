import numpy as np
import pandas as pd

from crispr_gnn.data.splits import LABEL_COLUMN, SPLIT_COLUMN
from crispr_gnn.evaluation.diagnostics import write_logistic_regression_diagnostics


def test_write_logistic_regression_diagnostics(tmp_path) -> None:
    assigned = pd.DataFrame(
        {
            SPLIT_COLUMN: ["val", "val", "test", "test"],
            LABEL_COLUMN: [1, 0, 1, 0],
            "grna_target_id": ["g1", "g2", "g3", "g4"],
            "genome": ["hg19", "hg19", "hg38", "hg38"],
            "experiment_id": [1, 1, 1, 1],
            "measured": [1, 1, 1, 1],
        },
        index=[10, 11, 12, 13],
    )
    predictions = [
        {
            "model_name": "logistic_regression",
            "feature_set": "F1",
            "split": "val",
            "row_index": np.array([10, 11]),
            "y_true": np.array([1, 0]),
            "y_score": np.array([0.8, 0.2]),
        },
        {
            "model_name": "logistic_regression",
            "feature_set": "F1",
            "split": "test",
            "row_index": np.array([12, 13]),
            "y_true": np.array([1, 0]),
            "y_score": np.array([0.7, 0.3]),
        },
    ]

    tables, figures = write_logistic_regression_diagnostics(assigned, predictions, tmp_path)

    assert len(tables) == 5
    assert len(figures) == 3
    for path in [*tables, *figures]:
        assert path.exists()
