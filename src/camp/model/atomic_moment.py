"""Atomic moments.

This is an expansion of the moments defined in MTP. Here we add features of different
rank to the moments.

The atomic moment is defined as:

M_v,p = sum_j f * h_v1 contract D_v2,
where D_v2 = r_ij otimes r_ij otimes ...

M_v = sum_p W_p M_v,p,

We require v1 <= v2.

Multiple contractions between h_v1 and D_v2 can lead to the same v, and this is denoted
by p.

Note, the operations are separate for each radial degree u, and thus u is omitted in
the notation.
"""
from typing import Union

import torch
from torch import Tensor, nn

from camp.nn.linear import LinearCombination, LinearMap
from camp.nn.mlp import MLP
from camp.utils import get_dyadic_tensor

from .atomic_moment_rule import get_atomic_moment_rules
from .radial import RadialPart
from .scatter import scatter
from .utils_jit import JITInterface


class AtomicMoment(nn.Module):
    def __init__(
        self,
        max_u: int,
        max_v1: int,
        max_v2: int,
        num_atom_types: int,
        num_average_neigh: float,
        max_chebyshev_degree: int = 8,
        radial_mlp_hidden_layers: Union[list[int], int] = 2,
        r_cut: float = 5,
        envelope: int = 6,
    ):
        """
        Atomic Moments.

        Args:
            max_u: maximum radial degree u of the moment tensor.
            max_v1: max angular degree v1 of the atomic features. If None, this is set
                to max_v2. This main use of this argument is for the first layer, where
                the atomic features are scalars and thus v1 = 0.
            max_v2: maximum angular degree v2 of the dyadic tensor, that is the maximum
                value of v2 in M_v,p = sum_j f * h_v1 contract D_v2.
            num_atom_types: number of atomic types.
            num_average_neigh:
            max_chebyshev_degree: max degree of the Chebyshev polynomial to use to
                construct the radial basis functions. The total number of chebyshev
                polynomials is `max_chebyshev_degree + 1`; +1 for the zeroth degree.
            radial_mlp_hidden_layers: if list of int, this gives the size of each hidden
                layer in the MLP that is applied to the radial basis functions. If int,
                this gives the number of hidden layers, and the size of each hidden
                layer is set to `max_u + 1`, the number of radial basis functions.
            number of hidden layers in the MLP that is applied to
                the radial basis functions.
            r_cut: cutoff distance.
            envelope: degree of the polynomial envelope function to make the radial
                basis function smooth at r_cut.
        """
        super().__init__()
        self.max_u = max_u
        self.max_v1 = max_v1
        self.max_v2 = max_v2
        self.num_atom_types = num_atom_types
        self.num_average_neigh = num_average_neigh
        self.max_chebyshev_degree = max_chebyshev_degree
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers
        self.r_cut = r_cut
        self.envelope = envelope

        self.radial = RadialPart(
            max_u + 1,
            num_atom_types,
            max_chebyshev_degree=max_chebyshev_degree,
            r_cut=r_cut,
            envelope=envelope,
        )

        atomic_moment_rules = {
            rank: get_atomic_moment_rules(max_in_rank=max_v2, out_rank=rank)
            for rank in range(max_v2 + 1)
        }
        # filter to keep only rules with v1 <= max_v1. By default, the rules consists
        # rules for v1 upto v1<=v2.
        self.atomic_moment_ranks: dict[int, list[list[int]]] = {
            rank: [rule["ranks"] for rule in rules if rule["ranks"][0] <= max_v1]
            for rank, rules in atomic_moment_rules.items()
        }
        self.atomic_moment_einsum_rule: dict[int, list[str]] = {
            rank: [rule["einsum_rule"] for rule in rules if rule["ranks"][0] <= max_v1]
            for rank, rules in atomic_moment_rules.items()
        }

        # MLP on the radial part. This is separate for each combination of v, v1, and v2

        if isinstance(radial_mlp_hidden_layers, int):
            radial_mlp_hidden_layers = [
                max_u + 1 for _ in range(radial_mlp_hidden_layers)
            ]

        self.radial_mlp = nn.ModuleDict()
        for v, rules in self.atomic_moment_ranks.items():
            for rule in rules:
                v1 = rule[0]
                v2 = rule[1]
                self.radial_mlp[f"{v}_{v1}_{v2}"] = MLP(
                    in_features=max_u + 1,
                    out_features=max_u + 1,
                    hidden_features=radial_mlp_hidden_layers,
                    out_activation=False,
                )

        self.linear_path = nn.ModuleDict(
            {
                str(rank): LinearCombination(len(rules), max_u + 1)
                for rank, rules in self.atomic_moment_ranks.items()
                if len(rules) > 1
            }
        )

        self.linear_channel = nn.ModuleDict(
            {
                str(rank): LinearMap(max_u + 1, max_u + 1)
                for rank, _ in self.atomic_moment_ranks.items()
            }
        )

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        atom_feats: dict[int, Tensor],
    ) -> dict[int, Tensor]:
        """

        Args:
            edge_vector:
            edge_idx:
            atom_type:
            atom_feats: atomic features. {v: tensor}, where v is the angular degree,
                and the tensor is of shape (n_u, n_atoms, 3, 3, ...). n_u denotes the
                batch dimension of the radial degree u, and the number of 3s is v.

        Returns:
            Atomic moments: {v: tensor}, where the tensor M_uv has shape
                (n_u, n_atoms, 3, 3, ...).
        """

        i_idx = edge_idx[0]
        j_idx = edge_idx[1]
        i_type = atom_type[i_idx]
        j_type = atom_type[j_idx]

        # radial part, shape (n_edges, n_u)
        fu = self.radial(torch.linalg.vector_norm(edge_vector, dim=-1), i_type, j_type)

        # TODO, check whether normalization is used in the original MTP code
        dyad_tensors = {
            v: get_dyadic_tensor(edge_vector, rank=v, normalize=True)
            for v in range(self.max_v2 + 1)
        }  # (n_edges, 3, 3, ...), number of 3: v

        M: dict[int, Tensor] = {}
        for v, rules in self.atomic_moment_ranks.items():
            # atomic moments of rank v from different paths
            M_uvp = []

            einsum_rules = self.atomic_moment_einsum_rule[v]

            for rule, equation in zip(rules, einsum_rules):
                v1 = rule[0]
                v2 = rule[1]

                # Make indexing ModuleDict work
                # See https://github.com/pytorch/pytorch/issues/68568
                fn: JITInterface = self.radial_mlp[f"{v}_{v1}_{v2}"]
                R = fn.forward(fu)  # shape (n_edges, n_u)
                R = R.T  # shape (n_u, n_edges)

                h = atom_feats[v1]  # (n_u, n_atoms, 3, 3, ...)
                h = h[:, j_idx, ...]  # (n_u, n_edges, 3, 3, ...)

                if v1 == 0 or v2 == 0:
                    t = torch.einsum("ue,ue...,e...->ue...", R, h, dyad_tensors[v2])
                else:
                    # shape (n_u, n_edges, 1, 1, ...,), number of 1: rank
                    shaped_R = R.reshape(R.shape + torch.Size([1] * v))

                    t = shaped_R * torch.einsum(equation, h, dyad_tensors[v2])

                # aggregate atoms j (src) to atom i (dst)
                # shape (n_u, n_atoms, 3, 3, ...), number of 3: rank
                t = (
                    scatter(t, i_idx, reduce="sum", dim=1)
                    / self.num_average_neigh**0.5
                )

                M_uvp.append(t)

            # linear combination of different paths
            if len(M_uvp) > 1:
                M_uvp2 = torch.stack(M_uvp)  # shape (n_rules, n_u, n_atoms, 3, 3, ...)
                fn: JITInterface = self.linear_path[str(v)]
                M_uv = fn.forward(M_uvp2)  # shape (n_u, n_atoms, 3, 3, ...)
            else:
                M_uv = M_uvp[0]  # shape (n_u, n_atoms, 3, 3, ...)

            # linear mix of different channels
            fn: JITInterface = self.linear_channel[str(v)]
            M_uv = fn.forward(M_uv)  # shape (n_u, n_atoms, 3, 3, ...)

            M[v] = M_uv

        return M
