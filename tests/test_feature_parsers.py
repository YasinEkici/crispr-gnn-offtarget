import pytest

from crispr_gnn.data.parsers import parse_numeric_array, parse_numeric_array_result


def test_parse_numeric_array_accepts_numpy_style_space_separated_values() -> None:
    values = parse_numeric_array("[0 1 2 3 4 5 6 7 8 9 10 11\n 12 13 14 15 16 17 18 19 20 21 22]")

    assert values == tuple(float(value) for value in range(23))


def test_parse_numeric_array_accepts_comma_separated_values() -> None:
    values = parse_numeric_array(",".join(str(value) for value in range(23)))

    assert values == tuple(float(value) for value in range(23))


@pytest.mark.parametrize("value", [None, float("nan"), "", "nan", "<NA>"])
def test_parse_numeric_array_returns_none_for_missing_values(value: object) -> None:
    assert parse_numeric_array(value) is None
    assert parse_numeric_array_result(value).status == "missing"


def test_parse_numeric_array_rejects_wrong_length() -> None:
    result = parse_numeric_array_result("[1 2 3]")

    assert result.status == "malformed_length"
    with pytest.raises(ValueError, match="expected 23 numeric values"):
        parse_numeric_array("[1 2 3]")


def test_parse_numeric_array_rejects_non_numeric_values() -> None:
    text = " ".join(["1"] * 22 + ["not_a_number"])
    result = parse_numeric_array_result(text)

    assert result.status == "non_numeric"
    with pytest.raises(ValueError):
        parse_numeric_array(text)
