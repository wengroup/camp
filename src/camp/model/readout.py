"""Readout layer."""
from typing import Union

import torch
from torch import Tensor, nn

from camp.nn.mlp import MLP

from .scatter import scatter
from .utils_jit import JITInterface


class TotalEnergy(nn.Module):
    """Get the total energy of the atomic configuration.

    There are multiple layers in the main model. The contribution of the last layer is
    passed through an MLP, while the contribution of the other layers is only multiplied
    a weight matrix. The output is the sum of the contributions of all layers.

    Args:
        num_layers: number of layers in the main model.
        in_features: the number of input features.
        hidden_features: the number of hidden features in each layer for the MLP.
            If a list, it provides the hidden layer sizes of the MLP. If an integer, it
            is interpreted as the number of hidden layers, and the hidden layer sizes
            are set to in_features.
        atomic_energy_shift/scale: the atomic energy shift and scale used to transform
            the output. The output atomic energy is computed as: e = e*scale + shift.
            - If atomic_energy_shift/scale is None, then no scale or shift are applied.
            - If a scalar tensor is provided for atomic_energy_shift/scale, then it is
              then it is used for all atom types.
            - If a tensor of shape (n_atom_types,) is provided for atomic_energy_shift/
              scale, then it is applied to each atom type separately.
    """

    def __init__(
        self,
        num_layers: int,
        in_features: int,
        hidden_features: Union[list[int], int],
        atomic_energy_shift: Tensor = None,
        atomic_energy_scale: Tensor = None,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.register_buffer(
            "atomic_energy_shift",
            atomic_energy_shift if atomic_energy_shift is not None else None,
        )
        self.register_buffer(
            "atomic_energy_scale",
            atomic_energy_scale if atomic_energy_scale is not None else None,
        )

        # early layers
        self.out_layers = nn.ModuleList(
            [nn.Linear(in_features, 1) for _ in range(num_layers - 1)]
        )

        # last layer
        if isinstance(hidden_features, int):
            hidden_features = [in_features for _ in range(hidden_features)]
        self.out_layers.append(
            MLP(in_features, 1, hidden_features, out_activation=False)
        )

    def forward(
        self, atom_feats: list[Tensor], atom_type: Tensor, num_atoms: Tensor
    ) -> Tensor:
        """
        Args:
            atom_feats: list of scalar atomic features, each of shape (n_u, n_atoms),
                where n_u denotes the batch dimension of the radial degree u.
            atom_type: 1D tensor of the atomic type of each atom.
            num_atoms: 1D tensor of the number of atoms in each atomic configuration.

        Returns:
            Total energy of the atomic configuration, 1D tensor of shape (n_config,).
        """
        assert len(atom_feats) == self.num_layers

        V = torch.zeros(1, dtype=atom_type[0].dtype, device=atom_type[0].device)
        for i, feats in enumerate(atom_feats):
            fn: JITInterface = self.out_layers[i]
            V = V + fn.forward(feats.T).reshape(-1)  # shape (n_atoms, )

        # normalization
        if self.atomic_energy_scale is not None:
            if self.atomic_energy_scale.ndim == 0:
                V = V * self.atomic_energy_scale
            else:
                V = V * self.atomic_energy_scale[atom_type]

        if self.atomic_energy_shift is not None:
            if self.atomic_energy_shift.ndim == 0:
                V = V + self.atomic_energy_shift
            else:
                V = V + self.atomic_energy_shift[atom_type]

        # energy of individual configurations
        # shape(num_configurations, 1)
        E = scatter(V, torch.repeat_interleave(num_atoms), reduce="sum", dim=0)

        return E
