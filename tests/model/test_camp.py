import torch

from camp.model.camp import CAMP, CAMPLayer
from camp.model.level import get_u, get_v
from camp.model.utils import compute_forces
from camp.utils import check_symmetric

from .conftest import atom_feats_or_moment


def test_CAMPLayer(config_info):
    _, atom_type, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    level_max = 6
    max_u = get_u(level_max)
    max_v = get_v(level_max)

    layer = CAMPLayer(num_atom_types, max_u, max_v, num_average_neigh=1.0)

    atom_feats = atom_feats_or_moment(num_atoms, max_u, max_v, min_v=0)
    out = layer(edge_vector, edge_idx, atom_type, atom_feats)

    assert set(out.keys()) == set(range(max_v + 1))

    for v, h in out.items():
        # check shape
        assert h.shape == (max_u + 1, num_atoms, *[3 for _ in range(v)])

        # check symmetry
        assert check_symmetric(h, start_dim=2, atol=1e-7)


def test_CAMP(batched_config_info):
    (
        coords,
        atom_type,
        edge_vector,
        edge_idx,
        num_atoms,
        num_atom_types,
    ) = batched_config_info

    model = CAMP(num_atom_types, max_u=2, max_v=4, num_average_neigh=1.0, num_layers=2)

    energy = model(edge_vector, edge_idx, atom_type, num_atoms)

    forces = compute_forces(energy, coords)
    assert forces.shape == (num_atoms.sum(), 3)
    assert torch.allclose(forces[:4], forces[4:])
