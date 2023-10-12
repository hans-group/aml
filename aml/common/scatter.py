from functools import partial
from typing import Optional

import torch


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Same as scatter in ``torch_scatter``.
    Use operation upstreamed in pytorch instead of ``torch_scatter``.
    Below is the original docstring from ``torch_scatter``.
    Currently scatter_mul is not implemented.

    Args:
        src: The source tensor.
        index: The indices of elements to scatter.
        dim: The axis along which to index. (default: :obj:`-1`)
        out: The destination tensor.
        dim_size: If :attr:`out` is not given, automatically create output
            with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor
            according to :obj:`index.max() + 1` is returned.
        reduce: The reduce operation (:obj:`"sum"`, :obj:`"mul"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)
    """
    if reduce == "add":
        reduce = "sum"
    if reduce == "min":
        reduce = "amin"
    if reduce == "max":
        reduce = "amax"
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_reduce_(dim, index, src, reduce=reduce)
    else:
        return out.scatter_reduce_(dim, index, src, reduce=reduce)


# Aliases for convinience
scatter_add = partial(scatter, reduce="sum")
scatter_sum = partial(scatter, reduce="sum")
scatter_mean = partial(scatter, reduce="mean")
scatter_min = partial(scatter, reduce="min")
scatter_max = partial(scatter, reduce="max")
