"""Utils for nequip.
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
from typing import Any, Dict

from e3nn import o3

_SPECIAL_IRREPS = [None]


def _irreps_compatible(ir1: Dict[str, o3.Irreps], ir2: Dict[str, o3.Irreps]):
    return all(ir1[k] == ir2[k] for k in ir1 if k in ir2)


def _fix_irreps_dict(d: Dict[str, Any]):
    return {k: (i if i in _SPECIAL_IRREPS else o3.Irreps(i)) for k, i in d.items()}


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False
