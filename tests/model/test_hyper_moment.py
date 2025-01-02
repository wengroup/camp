from camp.utils import check_symmetric
from camp.model.hyper_moment import HyperMoment

from .conftest import atom_feats_or_moment


def test_HyperMoment(config_info):
    _, atom_type, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    max_u = 3
    max_v = 4

    atomic_moments = atom_feats_or_moment(num_atoms, max_u, max_v, min_v=0)

    hm = HyperMoment(max_u=max_u, max_v=max_v)
    h = hm(atomic_moments)

    assert set(h.keys()) == set(range(max_v + 1))

    for v, h in h.items():
        # check shape
        assert h.shape == (max_u + 1, num_atoms, *[3 for _ in range(v)])

        # check symmetry
        assert check_symmetric(h, start_dim=2, atol=1e-7)
