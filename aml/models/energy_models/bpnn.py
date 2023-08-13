import itertools
from typing import Any, Dict, Literal

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from aml.common.registry import registry
from aml.common.utils import canocialize_species, compute_neighbor_vecs
from aml.data import keys as K
from aml.nn.bpnn import ACSF
from aml.nn.bpnn.acsf import _default_acsf_params
from aml.typing import DataDict, Tensor

from .base import BaseEnergyModel


@registry.register_energy_model("bpnn")
class BPNN(BaseEnergyModel):
    """Behler-Parrinello Neural Network (BPNN) force field model."""

    embedding_keys = [K.node_features]

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
        for _ in self.atomic_numbers:
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
        compute_neighbor_vecs(data)
        # Compute ACSF
        G = self.acsf(data)
        # Update the data dictionary
        data[K.node_features] = G

        # initialize the output
        energy_i = torch.zeros(size=(G.shape[0], 1), device=G.device, dtype=G.dtype)
        # Apply decoder
        for i, net in enumerate(self.element_nets):
            mask = data[K.elems] == self.atomic_numbers[i]
            energy_i[mask] = net(G[mask])

        energy_i = energy_i.squeeze(-1)
        energy_i = self.species_energy_scale(data, energy_i)
        energy = scatter(energy_i, data[K.batch], dim=0, reduce="sum")
        return energy

    def initialize(
        self,
        energy_shift_mode: Literal["mean", "atomic_energies"] = "atomic_energies",
        energy_scale_mode: Literal["energy_mean", "force_rms"] = "force_rms",
        energy_mean: float | Literal["auto"] | None = None,
        atomic_energies: dict[str, float] | Literal["auto"] | None = "auto",
        energy_scale: float | dict[str, float] | Literal["auto"] | None = "auto",
        trainable_scales: bool = True,
        dataset: InMemoryDataset | None = None,
        dataset_stride: int | None = None,
    ):
        super().initialize(
            energy_shift_mode,
            energy_scale_mode,
            energy_mean,
            atomic_energies,
            energy_scale,
            trainable_scales,
            dataset,
            dataset_stride,
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        self.acsf.fit_scales(dataloader)
