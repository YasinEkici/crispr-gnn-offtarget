import pytest

from crispr_gnn.data.labels import (
    SCHEME_A_THRESHOLD,
    get_label_scheme,
    label_from_cleavage_freq,
    labels_from_cleavage_freq,
)


def test_scheme_a_threshold_is_paper_aligned() -> None:
    assert SCHEME_A_THRESHOLD == pytest.approx(1e-5)


def test_label_from_cleavage_freq_uses_strict_threshold() -> None:
    assert label_from_cleavage_freq(1e-5) == 0
    assert label_from_cleavage_freq(1.1e-5) == 1
    assert label_from_cleavage_freq(None) == 0


def test_labels_from_cleavage_freq() -> None:
    assert labels_from_cleavage_freq([0, 1e-5, 2e-5, None]) == [0, 0, 1, 0]


def test_get_label_scheme_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError):
        get_label_scheme("unknown")
