from camp.utils import check_symmetric
from camp.model.atomic_moment import AtomicMoment

from .conftest import atom_feats_or_moment


def test_AtomicMoment(config_info):
    _, atom_type, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    max_u = 3
    max_v = 4

    atom_feats = atom_feats_or_moment(num_atoms, max_u, max_v, min_v=0)

    am = AtomicMoment(
        max_u,
        max_v,
        max_v,
        num_atom_types=num_atom_types,
        num_average_neigh=1.0,
        max_chebyshev_degree=4,
    )
    M = am(edge_vector, edge_idx, atom_type, atom_feats=atom_feats)

    assert set(M.keys()) == set(range(max_v + 1))

    for v, m in M.items():
        # check shape
        assert m.shape == (max_u + 1, num_atoms, *[3 for _ in range(v)])

        # check individual moments are symmetric
        assert check_symmetric(m, start_dim=2, atol=1e-6)
