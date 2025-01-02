import numpy as np
import scipy
import torch

from camp.model.radial import chebyshev_first


def test_chebyshev_first():
    x = np.arange(-5, 5, 0.1)

    N = 5
    out1 = chebyshev_first(N, torch.tensor(x))
    out2 = np.vstack([scipy.special.eval_chebyt(i, x) for i in range(N + 1)])

    assert torch.allclose(out1, torch.as_tensor(out2))
