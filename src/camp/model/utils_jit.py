import torch
from torch import Tensor, nn


@torch.jit.interface
class JITInterface(nn.Module):
    """Note, this should have exactly the same signature (including argument name
    `input` here) as The module it tries to annotate.

    See https://github.com/pytorch/pytorch/issues/68568
    """

    def forward(self, input: Tensor) -> Tensor:
        pass
