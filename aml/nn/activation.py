import torch
from torch.nn import functional as F

from aml.common.registry import registry
from aml.typing import Tensor


@registry.register_activation("shifted_softplus")
class ShiftedSoftplus(torch.nn.Module):
    r"""Shifted version of softplus activation function.
    $$
    \text{Softplus}(x) = \log(1 + \exp(x)) - \log(2)
    $$
    """

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        """Compute activation function.

        Args:
            x (Tensor): An input tensor.

        Returns:
            Tensor: An output tensor.
        """
        return F.softplus(x) - self.shift
