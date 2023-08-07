"""Concat layers.
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
from typing import List

import torch
from e3nn import o3

from aml.typing import DataDict

from .graph_mixin import GraphModuleMixin


class Concat(GraphModuleMixin, torch.nn.Module):
    """Concatenate multiple fields into one."""

    def __init__(self, in_fields: List[str], out_field: str, irreps_in={}):
        super().__init__()
        self.in_fields = list(in_fields)
        self.out_field = out_field
        self._init_irreps(irreps_in=irreps_in, required_irreps_in=self.in_fields)
        self.irreps_out[self.out_field] = sum((self.irreps_in[k] for k in self.in_fields), o3.Irreps())

    def forward(self, data: DataDict) -> DataDict:
        data[self.out_field] = torch.cat([data[k] for k in self.in_fields], dim=-1)
        return data
