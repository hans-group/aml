"""One-hot encoding of atom types.
This code is based on the Nequip implementation: https://github.com/mir-group/nequip
and slightly modified for integration with neural_iap.

License:
-------------------------------------------------------------------------------
MIT License

Copyright (c) 2021 The President and Fellows of Harvard College

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-------------------------------------------------------------------------------
"""
import torch
import torch.nn.functional
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from aml.data import keys as K
from aml.typing import DataDict

from . import additional_keys as AK
from .graph_mixin import GraphModuleMixin


@compile_mode("script")
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AK.node_attr: Irreps([(self.num_types, (0, 1))])}
        if self.set_features:
            irreps_out[K.node_features] = irreps_out[AK.node_attr]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: DataDict) -> DataDict:
        type_numbers = data[AK.elem_map].squeeze(-1)
        one_hot = torch.nn.functional.one_hot(type_numbers, num_classes=self.num_types).to(
            device=type_numbers.device, dtype=data[K.pos].dtype
        )
        data[AK.node_attr] = one_hot
        if self.set_features:
            data[K.node_features] = one_hot
        return data
