import torch
from torch import Tensor, nn


class AtomEmbedding(nn.Module):
    """
    Embed atoms as learnable fixed-size vectors.

    Args:
        num_atom_types: number of atom types for embedding.
        embedding_dim: output dim of the species embedding.
    """

    def __init__(self, num_atom_types: int, embedding_dim: int = 16):
        super().__init__()

        self.num_atom_types = num_atom_types
        self.embedding_dim = embedding_dim

        self.linear = nn.Linear(num_atom_types, embedding_dim)

        self.dtype = torch.get_default_dtype()

    def forward(self, atom_type: Tensor) -> Tensor:
        """
        Args:
            atom_type: (n_atoms,) tensor of atom types.

        Returns:
            embedded atom type: (n_atoms, embedding_dim) tensor of embedded atom types.
        """
        one_hot = torch.nn.functional.one_hot(
            atom_type, num_classes=self.num_atom_types
        ).to(self.dtype)

        embed = self.linear(one_hot)

        return embed
