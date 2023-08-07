import itertools
from typing import Any, Dict

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from aml.common.registry import registry
from aml.common.utils import canocialize_species
from aml.data import keys as K
from aml.nn.bpnn import ACSF
from aml.nn.bpnn.acsf import _default_acsf_params
from aml.typing import DataDict, Tensor

from .base import BaseEnergyModel


@registry.register_energy_model("bpnn")
class BPNN(BaseEnergyModel):
    """Behler-Parrinello Neural Network (BPNN) force field model."""

    def __init__(
        self,
        species: list[str],
        acsf_params: Dict[str, Any] = None,
        hidden_layers=(64, 64),
        activation="silu",
        cutoff: float = 5.0,
    ):
        super().__init__(species, cutoff)
        self.register_buffer("atomic_numbers", canocialize_species(species))
        self.acsf_params = acsf_params or _default_acsf_params()
        self.hidden_layers = hidden_layers
        self.activation = activation

        self.acsf = ACSF(species=species, params=self.acsf_params, cutoff=cutoff)
        self.element_nets = torch.nn.ModuleList()
        layer_shape = (self.acsf.n_descriptors, *hidden_layers, 1)
        for _ in species:
            layers = []
            for n_in, n_out in itertools.pairwise(layer_shape[:-1]):
                layers.append(torch.nn.Linear(n_in, n_out))
                layers.append(registry.get_activation_class(activation)())
            n_in_final, n_out_final = layer_shape[-2:]
            layers.append(torch.nn.Linear(n_in_final, n_out_final))
            self.element_nets.append(torch.nn.Sequential(*layers))

        self.reset_parameters()

    def reset_parameters(self):
        for net in self.element_nets:
            for layer in net:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def forward(self, data: DataDict) -> Tensor:
        """Compute ACSF and update the data dictionary."""
        # Compute ACSF
        G = self.acsf(data)
        # Update the data dictionary
        data[K.node_features] = G

        # initialize the output
        energy_i = torch.zeros(size=(G.shape[0], 1), device=G.device, dtype=G.dtype)
        # Apply decoder
        for i, elem in enumerate(self.atomic_numbers):
            mask = data[K.elems] == elem
            energy_i[mask] = self.element_nets[i](G[mask])
        energy_i = energy_i.squeeze(-1)
        energy_i = self.species_energy_scale(data, energy_i)
        energy = scatter(energy_i, data[K.batch], dim=0, reduce="sum")
        return energy

    def initialize(
        self, dataset, stride: int | None = None, use_avg_atomic_energy: bool = True, use_force_rms: bool = True
    ):
        super().initialize(dataset, stride, use_avg_atomic_energy, use_force_rms)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        self.acsf.fit_scales(dataloader)
