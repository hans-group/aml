import math
from typing import Optional

import torch
from torch_geometric.utils import scatter

from aml.data import keys as K
from aml.typing import DataDict

from ..nequip.graph_mixin import GraphModuleMixin
from ..nequip import additional_keys as AK


class EdgewiseReduce(GraphModuleMixin, torch.nn.Module):
    """Like ``nequip.nn.AtomwiseReduce``, but accumulating per-edge data into per-atom data."""

    _factor: Optional[float]

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        normalize_edge_reduce: bool = True,
        avg_num_neighbors: Optional[float] = None,
        reduce="sum",
        irreps_in={},  # noqa
    ):
        """Sum edges into nodes."""
        super().__init__()
        assert reduce in ("sum", "mean", "min", "max")
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]} if self.field in irreps_in else {},
        )
        self._factor = None
        if normalize_edge_reduce and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

    def forward(self, data: DataDict) -> DataDict:
        # get destination nodes 🚂
        edge_dst = data[K.edge_index][0]

        out = scatter(
            data[self.field],
            edge_dst,
            dim=0,
            dim_size=len(data[K.pos]),
            reduce=self.reduce,
        )

        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            out = out * factor

        data[self.out_field] = out

        return data


class EdgewiseEnergySum(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
    """

    _factor: Optional[float]

    def __init__(
        self,
        num_types: int,
        avg_num_neighbors: Optional[float] = None,
        normalize_edge_energy_sum: bool = True,
        per_edge_species_scale: bool = False,
        irreps_in={},  # noqa
    ):
        """Sum edges into nodes."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={K.edge_energy: "0e"},
            irreps_out={K.atomic_energy: "0e"},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

        self.per_edge_species_scale = per_edge_species_scale
        if self.per_edge_species_scale:
            self.per_edge_scales = torch.nn.Parameter(torch.ones(num_types, num_types))
        else:
            self.register_buffer("per_edge_scales", torch.Tensor())

    def forward(self, data: DataDict) -> DataDict:
        edge_center = data[K.edge_index][0]
        edge_neighbor = data[K.edge_index][1]

        edge_eng = data[K.edge_energy]
        species = data[AK.elem_map].squeeze(-1)
        center_species = species[edge_center]
        neighbor_species = species[edge_neighbor]

        if self.per_edge_species_scale:
            edge_eng = edge_eng * self.per_edge_scales[center_species, neighbor_species].unsqueeze(-1)

        atom_eng = scatter(edge_eng, edge_center, dim=0, dim_size=len(species))
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor

        data[K.atomic_energy] = atom_eng

        return data
