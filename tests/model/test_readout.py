import torch

from camp.model.level import get_u, get_v
from camp.model.readout import TotalEnergy

from .conftest import atom_feats_or_moment


def test_TotalEnergy(config_info):
    _, atom_type, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    level_max = 6
    max_u = get_u(level_max)
    max_v = get_v(level_max)

    atomic_energy_shift = torch.arange(len(set(atom_type))).float()
    atomic_energy_scale = atomic_energy_shift

    layer = TotalEnergy(
        1, max_u + 1, [max_u + 1], atomic_energy_shift, atomic_energy_scale
    )

    atom_feats = atom_feats_or_moment(num_atoms, max_u, max_v, min_v=0)

    # assuming two configurations
    atom_type = torch.cat([atom_type, atom_type], dim=0)
    atom_feats = {k: torch.cat([v, v], dim=1) for k, v in atom_feats.items()}
    scalar = atom_feats[0]

    out = layer([scalar], atom_type, torch.tensor([num_atoms, num_atoms]))

    assert out.shape == (2,)
    assert torch.allclose(out[0], out[1])
