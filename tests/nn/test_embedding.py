import torch

from camp.nn.embedding import AtomEmbedding


def test_atom_embedding():
    atom_type = torch.tensor([1, 0, 1, 0])
    embedding_dim = 16

    se = AtomEmbedding(num_atom_types=len(set(atom_type)), embedding_dim=embedding_dim)
    embedding = se(atom_type)

    assert embedding.shape == (len(atom_type), embedding_dim)
    assert se.linear.weight.shape == (embedding_dim, len(set(atom_type)))
