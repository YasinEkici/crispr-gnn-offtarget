import pytest

from crispr_gnn.data.labels import (
    SCHEME_A_THRESHOLD,
    get_label_scheme,
    is_label_eligible,
    label_from_cleavage_freq,
    labels_from_cleavage_freq,
)


def test_scheme_a_threshold_is_paper_aligned() -> None:
    assert SCHEME_A_THRESHOLD == pytest.approx(1e-5)


def test_label_from_cleavage_freq_uses_strict_threshold() -> None:
    assert label_from_cleavage_freq(1e-5) == 0
    assert label_from_cleavage_freq(1.1e-5) == 1
    assert label_from_cleavage_freq(-0.001) == 0
    assert label_from_cleavage_freq(4.52863) == 1


@pytest.mark.parametrize("value", [None, float("nan")])
def test_label_from_cleavage_freq_rejects_missing_labels(value: float | None) -> None:
    with pytest.raises(ValueError, match="missing or NaN"):
        label_from_cleavage_freq(value)


def test_is_label_eligible_rejects_missing_labels() -> None:
    assert is_label_eligible(None) is False
    assert is_label_eligible(float("nan")) is False
    assert is_label_eligible(0.0) is True


def test_labels_from_cleavage_freq() -> None:
    assert labels_from_cleavage_freq([0, 1e-5, 2e-5, -0.001, 4.52863]) == [0, 0, 1, 0, 1]


def test_labels_from_cleavage_freq_rejects_missing_labels() -> None:
    with pytest.raises(ValueError, match="missing or NaN"):
        labels_from_cleavage_freq([0, None])


def test_get_label_scheme_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError):
        get_label_scheme("unknown")
