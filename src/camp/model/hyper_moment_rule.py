"""Helper functions to generate the hyper moment constructing rules.

Hyper moments is a generalization of the B basis of MTP. In MTP, the B basis is defined
as the scalars obtained by contracting moments M_uv.

Here, the hyper moments are defined as the vectors, higher rank tensors, obtained by
contracting moments.

There are a couple of differences:
- here, we consider each radial degree u separately of the moment M_uv, and the rules
  apply to all moments with the same u.
- here we exclude the scalar moments (i.e. M_uv with v = 0), as they will not change the
  rank of the resulting tensor.
- here, we limit what can be contracted in terms of the number of the moments; we
  require the moment with the largest v to be contracted with all other moments, and the
  indices of all other moments are contracted away.


This module only generates the rules to construct the hyper moments with a rank of 1 and
above. For rank 0 scalars (i.e. the B basis of MTP), see `camp.model.B_basis_rule`.
"""

import itertools
from pathlib import Path
from typing import Any

from monty.serialization import dumpfn

from camp.utils import get_unique, letter_index


def write_hyper_moment_rules(
    max_max_in_rank: int = 5,
    min_in_rank: int = 1,
    min_out_rank: int = 0,
    filename="hyper_moment_rules.yaml",
):
    """Write to a yaml file all the constructing rules of hyper moments.

    Args:
        max_max_in_rank: maximum rank of the moments to be contracted.
        min_in_rank: minimum rank of the moments to be contracted.
        min_out_rank: minimum resulting rank of the hyper moments.
        filename: name of the file to write to.
    """
    if min_in_rank < 1:
        raise ValueError("min_in_rank must be >= 1.")

    rules = {}

    for max_in_rank in range(min_in_rank, max_max_in_rank + 1):
        rules[max_in_rank] = {}
        max_out_rank = max_in_rank

        for out_rank in range(min_out_rank, max_out_rank + 1):
            r = get_hyper_moment_rules(min_in_rank, max_in_rank, out_rank)

            if r:
                rules[max_in_rank][out_rank] = r
                print(
                    f"max_in_rank = {max_in_rank}, out_rank = {out_rank}, "
                    f"num_rules {len(r)}"
                )

    dumpfn(rules, filename)

    file = Path(__file__).name
    with open(filename, "r") as f:
        original = f.read()
    with open(filename, "w") as f:
        f.write(
            f"# This file is automatically generated by {file}.\n"
            f"# Don't edit this file, but edit {file} instead.\n\n"
            f"# This file is not used by the code, but is for reference only.\n\n"
            f"# The out key denote the maximum rank v of the moments M_uv to be used.\n"
            f"# The next key denotes the output rank v of the hyper moment H_uv.\n"
            f"# The `ranks` lists the v's of the moments M_uv, that is used to \n"
            f"# construct H_uv, and the `einsum_rule` gives the corresponding rule \n"
            f"# to contract the moments to get a tensor.\n\n"
        )
        f.write(original)


def get_hyper_moment_rules(
    min_in_rank: int, max_in_rank: int, out_rank: int, copy_scalar: bool = True
) -> list[dict[str, Any]]:
    """Get all the hyper moment constructing rules for a given max level.

    This only deals with the angular part of the moments, i.e. v. of M_uv.

    This finds all combinations of M_uv such that:
    - the largest v is equal to the sum of the rest, plus out_rank.

    Args:
        min_in_rank: minimum rank of the moments to be contracted.
        max_in_rank: maximum rank of the moments to be contracted.
        out_rank: resulting rank of the hyper moments.
        copy_scalar: this function requires min_in_rank >= 1, and thus the rules for the
            scalar moments (when out_rank = 0) only consider the case where higher rank
            (>=1) moments are contracted to a scalar. But for scalar moment,
            input rank = output rank = 0 can also be used to construct the scalar
            moment. So, if `copy_scalar` is True, then the scalar moment is copied to
            the output.

    Returns:
        list[{ranks: list[int], einsum_rule: str}]. `ranks` are the angular part
         (i.e. v) of the moments M_uv, and `einsum_rule` gives the corresponding rule to
         contract them to get a tensor of rank `out_rank`.
    """

    possible_rules = _get_rules(
        list(range(min_in_rank, max_in_rank + 1)), 2 * max_in_rank
    )

    # Select rules that can lead to a tensor of out_rank offset after contraction(s).
    rules = [rule for rule in possible_rules if _can_partition_2(rule, out_rank)]

    contraction_rules = []

    if out_rank == 0 and copy_scalar:
        contraction_rules.append({"ranks": [0], "einsum_rule": ""})

    for rule in rules:
        cr = _get_contraction_rules(rule)
        contraction_rules.append({"ranks": rule, "einsum_rule": cr})

    return contraction_rules


def _get_rules(
    values: list[int], max_v: int, including_single: bool = True
) -> list[list[int]]:
    """
    Find all combinations of values (with repeat) such that the combination is less than
    or equal to max_v.

    Args:
        values: a list of values to be combined.
        max_v: the maximum value of the combination.
        including_single: whether to include combinations of a single value.
    """
    if 0 in values:
        raise ValueError("0 is not allowed in `values`.")

    init_rules = [[v] for v in values if v <= max_v]
    all_rules = [init_rules]

    out_idx = 0
    while out_idx < len(all_rules):
        latest_rule = all_rules[out_idx]

        new_rules = []
        for l_r, o_r in itertools.product(latest_rule, init_rules):
            if sum(l_r + o_r) <= max_v:
                new_rules.append(l_r + o_r)

        if new_rules:
            new_rules = get_unique(new_rules)

            if not including_single and len(new_rules) == 1:
                pass
            else:
                all_rules.append(new_rules)

        out_idx += 1

    all_rules = [r for rules in all_rules for r in rules]

    return all_rules


def _can_partition_2(values: list[int], offset: int = 0) -> bool:
    """
    Where the largest element of a given list of values is equal to the sum of the rest,
    with an offset.

    Examples:
        >>>_can_partition_2([2, 2, 4]) # 4 = 2 + 2
        True
        >>>_can_partition_2([1, 2, 3, 4]) # 4 != 1 + 2 + 3
        False
        >>>_can_partition_2([2], offset=2) # 2 = offset
        True
        >>> _can_partition_2([1, 2, 4], offset=1) # 4 = 1 + 2 + offset
        True
    """
    max_v = max(values)

    total_sum = sum(values) - max_v + offset

    return max_v == total_sum


def _get_contraction_rules(rule: list[int]) -> str:
    """
    Get a contraction rule that leads to a tensor.

    This requires the moments with the highest rank (largest v) to be contracted with
    all other moments. The indices of all other moments are contracted away.


    Args:
        rule: a list of tuples (u, v) that represents M_uv.

    Example:
        >>> _get_contraction_rules([2, 4, 1])
        "...ab,...abcd,...c->...d"
        >>> _get_contraction_rules([1, 1, 5])
        "...a,...b,...abcde->...cde"

    Returns:
        A string of the contraction rules to be used by `torch.einsum` to generate a
        tensor.
    """
    max_v_idx = max(range(len(rule)), key=lambda i: rule[i])
    rank_max = rule[max_v_idx]
    n_contractions = sum(rule) - rule[max_v_idx]

    indices = letter_index(rank_max)

    einsum_rule = ""
    idx = 0
    for i, v in enumerate(rule):
        if v == 0:
            continue
        if i == max_v_idx:
            einsum_rule += "..." + indices + ","
        else:
            einsum_rule += "..." + indices[idx : idx + v] + ","
            idx += v

    einsum_rule = einsum_rule[:-1]  # remove the last comma

    einsum_rule += "->..." + indices[n_contractions:]

    return einsum_rule


if __name__ == "__main__":
    write_hyper_moment_rules(max_max_in_rank=8)
