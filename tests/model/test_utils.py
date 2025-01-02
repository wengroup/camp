import torch

from camp.model.utils import (
    apply_strain,
    apply_strain_single_config,
    compute_forces,
    compute_forces_stress,
)


def test_apply_strain(batched_config_info):
    (
        coords,
        atom_types,
        edge_vector,
        edge_idx,
        num_atoms,
        num_atom_types,
    ) = batched_config_info

    cell = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    strain, strained_pos, strained_cell = apply_strain(coords, cell, batch)

    assert torch.allclose(strain, torch.zeros((2, 3, 3)))
    assert torch.allclose(coords, strained_pos)
    assert torch.allclose(cell, strained_cell)


def test_apply_strain_single_config(config_info):
    coords, atom_types, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    cell = torch.eye(3)
    strain, strained_pos, strained_cell = apply_strain_single_config(coords, cell)

    assert torch.allclose(strain, torch.zeros((3, 3)))
    assert torch.allclose(coords, strained_pos)
    assert torch.allclose(cell, strained_cell)
