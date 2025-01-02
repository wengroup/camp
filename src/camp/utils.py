import itertools
import string
import warnings
from collections import Counter
from typing import Any

import torch
from torch import Tensor


def get_unique(values: list[list[Any]]) -> list[list[Any]]:
    """Get unique inner lists of an outer list.

    The order of the elements of the inner list does not matter.

    Example:
        >>> get_unique([[0, 1, 1], [1, 1, 0], [0, 0, 1]])
        [[0, 1, 1], [0, 0, 1]]
    """
    seen = set()
    unique = []
    for x in values:
        # using Counter to distinguish the case of x being, e.g. [0, 0, 1] and [0, 1, 1]
        # if we use frozenset(x), then both will be [0, 1], which is not distinguishable
        x_fs = frozenset(Counter(x).items())
        if x_fs not in seen:
            unique.append(x)
            seen.add(x_fs)

    return unique


def letter_index(n: int, start: int = 0) -> str:
    """
    Get a list of letters 'abc...' of length n.

    Args:
        n: the length of the letters
    """
    return string.ascii_lowercase[start : start + n]


def symmetrize(t: Tensor, start_dim: int = 0) -> Tensor:
    """
    Fully symmetrize a tensor.

    Args:
        t: input tensor
        start_dim: the starting dimension from which to symmetrize the tensor.

    Reference:
        Eq 9 of: http://dx.doi.org/10.1080/00018737800101454
    """
    # TODO, benchmarking torch.einsum and torch.permute.
    rank = t.ndim - start_dim

    indices = letter_index(rank)
    perms = itertools.permutations(indices, len(indices))
    rules = [f"...{indices}->...{''.join(p)}" for p in perms]

    # TODO, is there anyway to avoid torch.stack? This creates a large tensor
    #  requiring a lot of memory.
    sym_t = torch.mean(torch.stack([torch.einsum(s, t) for s in rules]), dim=0)

    return sym_t


def check_symmetric(
    T: Tensor, start_dim: int = 0, atol: float = 1e-8, rtol: float = 1e-5
) -> bool:
    """
    Check if a tensor is fully symmetric.

    Args:
        T: input tensor
        start_dim: the starting dimension to check symmetry
    """

    if T.ndim - start_dim <= 1:
        return True

    for p in itertools.permutations(range(start_dim, T.ndim)):
        p = list(range(start_dim)) + list(p)
        permuted = T.permute(*p)
        if not torch.allclose(T, permuted, atol=atol, rtol=rtol):
            e = T - permuted
            error = torch.sum(torch.abs(e))
            return False

    return True


def get_dyadic_tensor(r: Tensor, rank: int = 2, normalize: bool = True) -> Tensor:
    r"""
    Create a generalized dyadic tensor.

    For rank = 0, the dyadic tensor is a scalar, simply equal to 1.
    For rank = 1, the dyadic tensor is a vector, simply equal to r.
    For rank >= 2, the generalized dyadic tensor is the tensor product of the vector r
    with itself, i.e. :math:`r \otimes r \otimes \cdots \otimes r`. The rank is the
    number of vectors in the tensor product.

    Args:
        r: shape (..., 3) the vector to construct the generalized dyadic tensor. Only
            the last dimension is used to construct the tensor. The ellipsis represents
            any number of dimensions that allows batching.
        rank: rank of the generalized dyadic tensor, i.e. the number of times to tensor
            product the vector r with itself. Rank must be greater than or equal to 1.
        normalize: whether to normalize the vector r as a unit vector before
            constructing the generalized dyadic tensor.

    Returns:
        A tensor of shape (..., 3, 3, 3), where the ... represents the batching
        dimensions, and the number of 3's is equal to the rank.
    """
    if rank < 0:
        raise ValueError("Rank must be greater than or equal to 0.")
    elif rank == 0:
        shape = r.shape[:-1]
        return torch.ones(shape).to(r.device)
    else:
        if normalize:
            norm = torch.norm(r, dim=-1, keepdim=True)
            if torch.any(norm < 1e-3):
                warnings.warn("The norm of the vector(s) is smaller than 1e-3.")
            r = r / norm

        indices = letter_index(rank)
        data = [r] * rank
        t = torch.einsum(f"{','.join(['...'+i for i in indices])}->...{indices}", data)

        return t
