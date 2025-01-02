from camp.model.hyper_moment_rule import (
    _can_partition_2,
    _get_contraction_rules,
    _get_rules,
)


def test_get_rules():
    rules = _get_rules([1, 2], 2)
    assert rules == [[1], [2], [1, 1]]

    rules = _get_rules([1, 2, 3], 3)
    assert rules == [[1], [2], [3], [1, 1], [1, 2], [1, 1, 1]]


def test_can_partition_2():
    assert _can_partition_2([2, 2, 4]) is True
    assert _can_partition_2([1, 2, 3, 4]) is False
    assert _can_partition_2([1, 2, 4], offset=1) is True
    assert _can_partition_2([2], offset=2) is True
    assert _can_partition_2([1, 2, 4], offset=2) is False


def test_get_contraction_rules():
    rule = _get_contraction_rules([2, 4, 1])
    assert rule == "...ab,...abcd,...c->...d"

    rule = _get_contraction_rules([1, 1, 5])
    assert rule == "...a,...b,...abcde->...cde"
