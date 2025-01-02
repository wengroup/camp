from camp.model.atomic_moment_rule import _get_contraction_rules


def test_get_contraction_rule():
    assert _get_contraction_rules(0, 0) == ""

    assert _get_contraction_rules(0, 1) == "...a->...a"
    assert _get_contraction_rules(0, 2) == "...ab->...ab"

    assert _get_contraction_rules(2, 3) == "...ab,...abc->...c"
    assert _get_contraction_rules(2, 2) == "...ab,...ab->..."
