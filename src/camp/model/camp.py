"""CAMP model class."""

from typing import Union

from torch import Tensor, nn

from camp.nn.embedding import AtomEmbedding
from camp.nn.linear import LinearMap

from .atomic_moment import AtomicMoment
from .hyper_moment import HyperMoment
from .readout import TotalEnergy
from .utils_jit import JITInterface


class CAMP(nn.Module):
    """Cartesian Atomic Moment Potential."""

    def __init__(
        self,
        num_atom_types: int,
        max_u: int,
        max_v: int,
        num_average_neigh: float,
        num_layers: int = 2,
        r_cut: float = 5.0,
        # radial
        max_chebyshev_degree: int = 8,
        radial_mlp_hidden_layers: Union[list[int], int] = 2,
        # output module
        output_mlp_hidden_layers: Union[list[int], int] = 2,
        atomic_energy_shift: Tensor = None,
        atomic_energy_scale: Tensor = None,
    ):
        """

        Args:
            num_average_neigh: average number of neighbors of the atoms
            radial_mlp_hidden_layers: if list of int, this gives the size of each hidden
                layer in the MLP that is applied to the radial basis functions. If int,
                this gives the number of hidden layers, and the size of each hidden
                layer is set to `max_u + 1`, the number of radial basis functions.
            output_mlp_hidden_layers: if a list of ints, each element is the number of
                hidden units in the corresponding hidden layer. If an int, this will be
                the number of hidden layers, and the number of hidden units in each
                layer will be the same as the input dimension, that is max_u + 1.
        """
        super().__init__()
        self.num_atom_types = num_atom_types
        self.max_u = max_u
        self.max_v = max_v
        self.num_average_num_neigh = num_average_neigh
        self.num_layers = num_layers
        self.r_cut = r_cut

        self.max_chebyshev_degree = max_chebyshev_degree
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers

        self.output_mlp_hidden_layers = output_mlp_hidden_layers
        self.atomic_energy_shift = atomic_energy_shift
        self.atomic_energy_scale = atomic_energy_scale

        self.atom_embedding = AtomEmbedding(num_atom_types, max_u + 1)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # first layer, the atom feats are scalars, set it explicitly to reduce
            # computational cost
            if i == 0:
                max_atom_feats_rank = 0
                mix_atom_feats_radial_channel = False
            else:
                max_atom_feats_rank = None
                mix_atom_feats_radial_channel = True

            # for the last layer, we are only interested in the scalar output, from
            # which we can compute the total energy
            if i == num_layers - 1:
                max_out_rank = 0
            else:
                max_out_rank = None

            self.layers.append(
                CAMPLayer(
                    num_atom_types=num_atom_types,
                    max_u=max_u,
                    max_v=max_v,
                    num_average_neigh=num_average_neigh,
                    max_chebyshev_degree=max_chebyshev_degree,
                    r_cut=r_cut,
                    radial_mlp_hidden_layers=radial_mlp_hidden_layers,
                    mix_atom_feats_radial_channel=mix_atom_feats_radial_channel,
                    max_atom_feats_rank=max_atom_feats_rank,
                    max_out_rank=max_out_rank,
                )
            )

        self.readout = TotalEnergy(
            num_layers,
            max_u + 1,
            output_mlp_hidden_layers,
            atomic_energy_shift,
            atomic_energy_scale,
        )

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        num_atoms: Tensor,
    ) -> Tensor:
        """
        Args:
            edge_vector:
            edge_idx:
            atom_type:
            num_atoms: 1D tensor of the number of atoms in each atomic configuration.

        Returns:
            1D tensor of the total energy of the configurations.
        """
        # (n_atoms, embedding_dim), where embedding_dim = max_u + 1
        embedding = self.atom_embedding(atom_type)

        # for the first layer, only scalars, so v = 0
        atom_feats = {0: embedding.T}  # {v: (n_u, n_atoms)}

        # TODO, for the first layer, because atom_feats is special (only scalars), we
        #  might be able to use a more constrained version of AtomicMoment in
        #  `CAMPLayer` to save some computation.
        scalar_feats = []
        for layer in self.layers:
            atom_feats = layer(edge_vector, edge_idx, atom_type, atom_feats)
            scalar_feats.append(atom_feats[0])

        energy = self.readout(scalar_feats, atom_type, num_atoms)

        return energy


class CAMPLayer(nn.Module):
    """A CAMP layer consisting of an Atomic Moment Layer and a Hyper Moment Layer."""

    def __init__(
        self,
        num_atom_types: int,
        max_u: int,
        max_v: int,
        num_average_neigh: float,
        max_chebyshev_degree: int = 8,
        r_cut: float = 5.0,
        radial_mlp_hidden_layers: Union[list[int], int] = 2,
        mix_atom_feats_radial_channel: bool = True,
        max_atom_feats_rank: int = None,
        max_out_rank: int = None,
    ):
        """

        Args:
            num_atom_types:
            max_u:
            max_v:
            num_average_neigh:
            max_chebyshev_degree:
            r_cut:
            radial_mlp_hidden_layers: if list of int, this gives the size of each hidden
                layer in the MLP that is applied to the radial basis functions. If int,
                this gives the number of hidden layers, and the size of each hidden
                layer is set to `max_u + 1`, the number of radial basis functions.
            mix_atom_feats_radial_channel: whether to mix the radial channel of the
                input atom feats and then add to the output hyper moment.
            max_atom_feats_rank: the max rank of the input atom feats. If None, set to
                max_v.
            max_out_rank: the max rank of the output hyper moment. If None, set to
                max_v.

        """

        super().__init__()

        self.num_atom_types = num_atom_types
        self.max_u = max_u
        self.max_v = max_v
        self.num_average_neigh = num_average_neigh
        self.max_chebyshev_degree = max_chebyshev_degree
        self.r_cut = r_cut
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers

        self.mix_atom_feats_radial_channel = mix_atom_feats_radial_channel
        self.max_atom_feats_rank = (
            max_v if max_atom_feats_rank is None else max_atom_feats_rank
        )
        self.max_out_rank = max_v if max_out_rank is None else max_out_rank

        self.mlp_mix_atom_feats = nn.ModuleDict(
            {
                str(rank): LinearMap(max_u + 1, max_u + 1)
                for rank in range(self.max_atom_feats_rank + 1)
            }
        )

        self.atom_moment = AtomicMoment(
            max_u=max_u,
            max_v1=self.max_atom_feats_rank,
            max_v2=max_v,
            num_atom_types=num_atom_types,
            num_average_neigh=num_average_neigh,
            max_chebyshev_degree=max_chebyshev_degree,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            r_cut=r_cut,
        )

        self.hyper_moment = HyperMoment(max_u, max_v, self.max_out_rank)

        # number of radial degrees
        n_u = max_u + 1

        # params for mixing radial channel of hyper moment
        self.linear_channel_hyper = nn.ModuleDict(
            {str(rank): LinearMap(n_u, n_u) for rank in range(self.max_out_rank + 1)}
        )

        # params for mixing radial channel of input atom feats
        if mix_atom_feats_radial_channel:
            self.linear_channel_feats = nn.ModuleDict(
                {
                    str(rank): LinearMap(n_u, n_u)
                    for rank in range(self.max_out_rank + 1)
                }
            )
        else:
            self.linear_channel_feats = nn.ModuleDict({})

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        atom_feats_in: dict[int, Tensor],
    ) -> dict[int, Tensor]:
        """

        Args:
            edge_vector:
            edge_idx:
            atom_type:
            atom_feats_in: atomic features. {v: tensor}, where v is the angular degree,
                and the tensor is of shape (n_u, n_atoms, 3, 3, ...). n_u denotes the
                batch dimension of the radial degree u, and the number of 3s is v.

        Returns:
            Update atom feats. {v: tensor}, of the same shape as `atom_feats`.
        """
        # mix atom feats across radial channel
        atom_feats: dict[int, Tensor] = {}
        for v, f in atom_feats_in.items():
            # Make indexing ModuleDict work
            # See https://github.com/pytorch/pytorch/issues/68568
            fn: JITInterface = self.mlp_mix_atom_feats[str(v)]
            atom_feats[v] = fn.forward(f)

        am = self.atom_moment(edge_vector, edge_idx, atom_type, atom_feats)

        hm = self.hyper_moment(am)  # {v: (n_u, n_atoms, 3, 3, ...)}}

        # mix radial channel of hyper moment
        # {v: {n_u, n_atoms, 3, 3, ...)}}
        # hm = {rank: self.linear_channel_hyper[str(rank)](m) for rank, m in hm.items()}

        for rank, m in hm.items():
            fn: JITInterface = self.linear_channel_hyper[str(rank)]
            hm[rank] = fn.forward(m)

        out = hm

        # mix radial channel of input atom feats and add to the output
        if self.mix_atom_feats_radial_channel:
            max_rank = min(self.max_atom_feats_rank, self.max_out_rank)
            for rank in range(max_rank + 1):
                fn: JITInterface = self.linear_channel_feats[str(rank)]
                out[rank] = out[rank] + fn.forward(atom_feats[rank])

        return out
