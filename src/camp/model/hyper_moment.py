"""Generate hyper moments from atomic moments."""
import torch
from torch import Tensor, nn

from camp.nn.linear import LinearCombination

from .hyper_moment_rule import get_hyper_moment_rules
from .utils_jit import JITInterface


class HyperMoment(nn.Module):
    """
    Hyper moment tensor:

    H_uv,p = M_{uv_1} * M_{uv_2} * ...,
    where * denotes tensor product with contraction.

    H_uv = sum_{p=0} W_p H_uv,p,
    where p denote different path that lead to the same v.

    H_uv = sum_{v'=0} W_uv'H_uv'
    """

    def __init__(self, max_u: int, max_v: int, max_out_rank: int = None):
        """

        Args:
            max_u: maximum radial degree u of the moment tensor.
            max_v: maximum angular degree v of the moment tensor.
            max_out_rank: maximum angular degree of the output hyper moment tensor. It
                should be smaller than or equal to max_v. If None, it is set to max_v.
        """
        super().__init__()
        self.max_u = max_u
        self.max_v = max_v
        self.max_out_rank = max_out_rank

        if max_out_rank is None:
            max_out_rank = max_v
        else:
            if max_out_rank > max_v:
                raise AssertionError(
                    f"Expect max_out_rank <= max_v, got {max_out_rank} > {max_v}."
                )

        # see `hyper_moment_rules.yaml` for explicitly written rules
        # min_in_rank starts from 1 to avoid rank 0 that can result in an infinite
        # number of rules
        self.hyper_moment_ranks = {}
        self.hyper_moment_einsum = {}
        for rank in range(max_out_rank + 1):
            o = get_hyper_moment_rules(min_in_rank=1, max_in_rank=max_v, out_rank=rank)
            self.hyper_moment_ranks[rank] = [x["ranks"] for x in o]
            self.hyper_moment_einsum[rank] = [x["einsum_rule"] for x in o]

        self.linear_path = nn.ModuleDict(
            {
                str(rank): LinearCombination(len(rules), max_u + 1)
                for rank, rules in self.hyper_moment_ranks.items()
                if len(rules) > 1
            }
        )

    def forward(self, atomic_moment: dict[int, Tensor]) -> dict[int, Tensor]:
        """

        Args:
            atomic_moment: {v: tensor}, where v is the angular degree, and the tensor
                is of shape (n_u, n_atoms, 3, 3, ...). n_u denotes the batch dimension
                of the radial degree u, and the number of 3s is v.

        Returns:
            Hyper moments: {v: tensor}, where the tensor H_uv has shape
                (n_u, n_atoms, 3, 3, ...).
        """
        # hyper moments
        m_out: dict[int, Tensor] = {}
        for rank, rules in self.hyper_moment_ranks.items():
            H = []

            einsum_rules = self.hyper_moment_einsum[rank]
            for r, equation in zip(rules, einsum_rules):
                # If there is only one value in r, no need to do einsum
                if len(r) == 1:
                    tmp = atomic_moment[r[0]]
                else:
                    # shape of tmp: (n_u, n_atoms, 3, 3, ...) , number of 3: rank
                    tmp = torch.einsum(equation, [atomic_moment[v] for v in r])

                H.append(tmp)

            # linear layer to combine different path
            if len(H) > 1:
                H2 = torch.stack(H)  # shape (n_rules, n_u, n_atoms, 3, 3, ...)

                # Make indexing ModuleDict work
                # See https://github.com/pytorch/pytorch/issues/68568
                fn: JITInterface = self.linear_path[str(rank)]

                m = fn.forward(H2)  # shape (n_u, n_atoms, 3, 3, ...)
            else:
                m = H[0]  # shape (n_u, n_atoms, 3, 3, ...)

            # linear combination of different path
            m_out[rank] = m

        return m_out
