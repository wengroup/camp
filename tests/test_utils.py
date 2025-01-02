from camp.utils import get_unique


def test_get_unique():
    a = [[0, 0, 1], [0, 1, 1], [1, 0, 1]]
    assert get_unique(a) == [[0, 0, 1], [0, 1, 1]]

    a = [[(0, 0), (0, 0), (0, 1)], [(0, 0), (0, 1), (0, 1)], [(0, 1), (0, 1), (0, 0)]]
    assert get_unique(a) == [[(0, 0), (0, 0), (0, 1)], [(0, 0), (0, 1), (0, 1)]]
