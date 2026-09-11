import pytest

import validation as v


def test_require_finite_accepts_numbers():
    assert v.require_finite(3, "x") == 3.0
    assert v.require_finite(3.5, "x") == 3.5


@pytest.mark.parametrize("bad", [None, "3", float("nan"), float("inf"), True, False])
def test_require_finite_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        v.require_finite(bad, "x")


def test_require_positive():
    assert v.require_positive(1.0, "x") == 1.0
    with pytest.raises(ValueError):
        v.require_positive(0, "x")
    with pytest.raises(ValueError):
        v.require_positive(-1, "x")


def test_require_positive_int():
    assert v.require_positive_int(5, "n") == 5
    with pytest.raises(ValueError):
        v.require_positive_int(0, "n")
    with pytest.raises(ValueError):
        v.require_positive_int(1.5, "n")
    with pytest.raises(ValueError):
        v.require_positive_int(True, "n")


def test_require_range():
    assert v.require_range(1, 2, "x") == (1.0, 2.0)
    with pytest.raises(ValueError):
        v.require_range(2, 1, "x")
    with pytest.raises(ValueError):
        v.require_range(1, 1, "x")


def test_require_nonempty():
    with pytest.raises(ValueError):
        v.require_nonempty([], "x")
    with pytest.raises(ValueError):
        v.require_nonempty(None, "x")
    assert v.require_nonempty([1], "x") == [1]


def test_require_columns():
    rows = [{"a": 1, "b": 2}, {"a": 3}]
    with pytest.raises(ValueError):
        v.require_columns(rows, ["a", "b"], "rows")
    ok_rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert v.require_columns(ok_rows, ["a", "b"], "rows") == ok_rows
