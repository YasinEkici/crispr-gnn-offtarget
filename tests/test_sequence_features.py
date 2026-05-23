import pandas as pd

from crispr_gnn.features.sequence import SEQUENCE_FEATURE_SET, SEQUENCE_REPRESENTATION, build_sequence_pair_encoding


def test_build_sequence_pair_encoding_uses_only_sequence_columns() -> None:
    df = pd.DataFrame(
        {
            "grna_target_sequence": ["ACGT", "AAAA"],
            "target_sequence": ["ACGA", "AAAT"],
            "label": [1, 0],
            "experiment_id": [1, 2],
        }
    )

    result = build_sequence_pair_encoding(df, max_length=4)

    assert result.encoded.shape == (2, 4, 11)
    assert result.encoded[0, 3, 10] == 1.0
    assert result.audit["feature_set"].eq(SEQUENCE_FEATURE_SET).all()
    assert result.audit["input_representation"].eq(SEQUENCE_REPRESENTATION).all()
    assert not result.audit["is_forbidden"].any()
    assert set(result.audit["source_column"]) == {"grna_target_sequence", "target_sequence"}
