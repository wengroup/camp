import pytest
import torch

from camp.data.utils import get_edge_vec
from camp.utils import symmetrize


@pytest.fixture(scope="session")
def B_basis_rules():
    rules = [
        {"moments": [(0, 0)], "einsum_rule": ""},
        {"moments": [(0, 0), (0, 0)], "einsum_rule": ""},
        {"moments": [(0, 0), (1, 1), (0, 1)], "einsum_rule": "j,j"},
        {"moments": [(0, 2), (1, 1), (0, 1)], "einsum_rule": "ij,i,j"},
    ]

    return rules


@pytest.fixture(scope="session")
def uv_pairs(B_basis_rules):
    uv_pairs = set()
    for rule in B_basis_rules:
        uv_pairs.update([tuple(pair) for pair in rule["moments"]])

    return list(uv_pairs)


@pytest.fixture(scope="session")
def config_info():
    # 4 atoms
    atom_types = torch.tensor([0, 1, 1, 1])
    coords = 0.2 * torch.arange(12).reshape(4, 3).to(torch.get_default_dtype())
    coords[0, 0] += 0.1
    coords[1, 1] += 0.2
    coords[2, 2] += 0.3
    coords[3, 1] += 0.1

    coords.requires_grad_(True)

    edge_idx = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
        ]
    )
    edge_vector = torch.stack([coords[j] - coords[i] for i, j in edge_idx.T])

    num_atoms = torch.max(edge_idx) + 1
    num_atom_types = atom_types.max() + 1

    return coords, atom_types, edge_vector, edge_idx, num_atoms, num_atom_types


@pytest.fixture(scope="session")
def batched_config_info(config_info):
    coords, atom_types, edge_vector, edge_idx, num_atoms, num_atom_types = config_info

    coords = torch.vstack([coords, coords])
    atom_types = torch.hstack([atom_types, atom_types])
    edge_idx = torch.hstack([edge_idx, edge_idx + num_atoms])
    num_atoms = torch.tensor([num_atoms, num_atoms])
    num_atom_types = num_atom_types

    shift_vec = torch.zeros_like(edge_vector)
    shift_vec = torch.vstack([shift_vec, shift_vec])
    cell = torch.eye(3)
    cell = torch.vstack([cell, cell])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    edge_vector = get_edge_vec(coords, shift_vec, cell, edge_idx, batch=batch)

    return coords, atom_types, edge_vector, edge_idx, num_atoms, num_atom_types


def atom_feats_or_moment(num_atoms, max_u, max_v, min_v):
    """
    Returns:
        {v: tensor}. v goes from 0 to max_v, and the shape of the tensor is:
        (max_u+1, num_atoms, 3, 3, ...)

    """
    torch.manual_seed(35)

    out = {}
    for v in range(min_v, max_v + 1):
        tmp = []
        for u in range(max_u + 1):
            h = torch.randn(num_atoms, *[3 for _ in range(v)])
            tmp.append(symmetrize(h, start_dim=1))
        out[v] = torch.stack(tmp)

    return out
