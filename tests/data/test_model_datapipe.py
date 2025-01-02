import pytest

from camp.data.utils import get_edge_vec
from camp.model.camp import CAMP
from camp.model.utils import compute_forces


@pytest.fixture
def model():
    m = CAMP(num_atom_types=2, max_v=2, max_u=2, num_average_neigh=5.0)
    return m


def test_datapipe(model, dataloader):
    for batch in dataloader:
        batch.pos.requires_grad_(True)
        edge_vector = get_edge_vec(
            batch.pos, batch.shift_vec, batch.cell, batch.edge_index, batch.batch
        )

        energy = model(edge_vector, batch.edge_index, batch.atom_type, batch.num_atoms)
        assert energy.shape == (batch.num_graphs,)

        forces = compute_forces(energy, batch.pos)
        assert forces.shape == (batch.num_nodes, 3)
