###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

"""Code adapted from MACE. Below is the original license.

MIT License

Copyright (c) 2022 ACEsuit/mace

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""


import warnings
from typing import List, Literal

import ase.data
import numpy as np
import torch
from e3nn import o3
from torch.nn import functional as F

from aml.common.registry import registry
from aml.common.scatter import scatter
from aml.data import keys as K
from aml.data.dataset.base import BaseDataset
from aml.data.utils import compute_neighbor_vecs
from aml.nn.mace.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from aml.nn.scale import GlobalScaleShift
from aml.typing import DataDict, Tensor

from .base import BaseEnergyModel


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def determine_interaction(residual: bool) -> type(torch.nn.Module):
    if residual:
        return RealAgnosticResidualInteractionBlock
    else:
        return RealAgnosticInteractionBlock


@registry.register_energy_model("mace")
class MACE(BaseEnergyModel):
    """A MACE (Multi-ACE) model.
    Uses equivariant message passing to predict the body-ordered energy of the system.
    Provides "node_features" embeddings.

    Args:
        species (List[str]): List of atomic species to consider.
        cutoff (float): Cutoff radius for the ACSF. Default is 5.0.
        num_bessel (int): Number of Bessel RBF functions to use. Default is 8.
        num_polynomial_cutoff (int): Number of polynomial cutoff functions to use. Default is 5.
        num_interactions (int): Number of interaction layers. Default is 2.
        residual_first_interaction (bool): Whether to use residual connections in the first interaction layer.
            Default is False.
        l_max (int): Maximum spherical harmonic degree. Default is 3.
        hidden_irreps (str): Hidden irreps for the interactions. Default is "128x0e + 128x1o".
            See https://e3nn.org/ for more information.
        correlation (int): Maximum body-order to consider. Default is 3.
        MLP_irreps (str): Irreps for the MLP readout. Default is "16x0e".
            See https://e3nn.org/ for more information.
        gate (str): Activation function to use in the MLP readout. Default is "silu".
        atomic_energies (dict[str, float]): Atomic energies to shift. Default is None.
            This is required parameter.
        avg_num_neighbors (float): Average number of neighbors per atom. Default is None.
            This is required parameter.
    """

    embedding_keys = [K.node_features]

    def __init__(
        self,
        species: List[str],
        cutoff: float = 5.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        num_interactions: int = 2,
        residual_first_interaction: bool = True,
        residual_interaction: bool = True,
        l_max: int = 3,
        hidden_irreps: str = "128x0e + 128x1o",
        correlation: int = 3,
        MLP_irreps: str = "16x0e",
        gate="silu",
        atomic_energies: dict[str, float] = None,
        avg_num_neighbors: float = None,
    ):
        super().__init__(species, cutoff)
        del self.species_energy_scale
        self.num_bessel = num_bessel
        self.num_polynomial_cutoff = num_polynomial_cutoff
        self.l_max = l_max
        self.residual_first_interaction = residual_first_interaction
        self.residual_interaction = residual_interaction
        self.correlation = correlation
        self.hidden_irreps = hidden_irreps
        self.MLP_irreps = MLP_irreps
        self.gate = gate
        self.atomic_energies = atomic_energies
        self.avg_num_neighbors = avg_num_neighbors

        if avg_num_neighbors is None:
            raise ValueError("avg_num_neighbors is required")
        gate = registry.get_activation_class(gate)()
        atomic_numbers = [ase.data.atomic_numbers[s] for s in species]
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.long))
        self.register_buffer("num_interactions", torch.tensor(num_interactions, dtype=torch.long))
        self.register_buffer(
            "atomic_numbers_index_map", torch.full((max(self.atomic_numbers) + 1,), -1, dtype=torch.long)
        )
        self.r_max = cutoff
        for i, n in enumerate(self.atomic_numbers):
            self.atomic_numbers_index_map[n] = i

        self.num_elements = len(species)
        # Embedding
        node_attr_irreps = o3.Irreps([(self.num_elements, (0, 1))])
        hidden_irreps = o3.Irreps(hidden_irreps)
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(irreps_in=node_attr_irreps, irreps_out=node_feats_irreps)
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(l_max)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")

        # Interactions and readout
        if atomic_energies is not None:
            self.atomic_energies = atomic_energies
            self.register_buffer("_atomic_energies", torch.tensor(list(atomic_energies.values()), dtype=torch.float))
        else:
            # If it is loaded from a checkpoint, it's okay since it will be overwritten by
            # model.load_state_dict()
            self.register_buffer("_atomic_energies", torch.zeros(len(species), dtype=torch.float))

        self.atomic_energies_fn = AtomicEnergiesBlock(self._atomic_energies)

        # Interactions
        first_interaction_cls = determine_interaction(residual_first_interaction)
        interaction_cls = determine_interaction(residual_interaction)
        inter = first_interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(first_interaction_cls):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=self.num_elements,
            use_sc=use_sc_first,
        )

        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(hidden_irreps[0])  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=self.num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate))
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))
        self.global_energy_scale = GlobalScaleShift()

    def forward(self, data: DataDict) -> Tensor:
        compute_neighbor_vecs(data)
        num_graphs = data[K.batch].max() + 1
        # Atomic energies
        elem_one_hot = F.one_hot(self.atomic_numbers_index_map[data[K.elems]], self.num_elements).to(
            device=data[K.pos].device, dtype=data[K.pos].dtype
        )
        node_e0 = self.atomic_energies_fn(elem_one_hot)
        e0 = scatter(src=node_e0, index=data[K.batch], dim=-1, dim_size=num_graphs)  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(elem_one_hot)
        vectors = data[K.edge_vec]
        lengths = torch.linalg.norm(vectors, dim=-1)

        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths.view(-1, 1))

        # Interactions
        energies = [e0]
        for interaction, product, readout in zip(self.interactions, self.products, self.readouts, strict=False):
            node_feats, sc = interaction(
                node_attrs=elem_one_hot,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data[K.edge_index],
            )
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=elem_one_hot)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter(src=node_energies, index=data[K.batch], dim=-1, dim_size=num_graphs)  # [n_graphs,]
            energy = self.global_energy_scale(data, energy)
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        data[K.node_features] = node_feats  # Only scalar node features

        return total_energy

    def initialize(
        self,
        energy_shift_mode: Literal["mean", "atomic_energies"] = "atomic_energies",
        energy_scale_mode: Literal["energy_std", "force_rms"] = "force_rms",
        energy_mean: Literal["auto"] | float | None = None,  # Must be per atom
        atomic_energies: Literal["auto"] | dict[str, float] | None = "auto",
        energy_scale: Literal["auto"] | float | dict[str, float] | None = "auto",
        trainable_scales: bool = True,
        dataset: BaseDataset | None = None,
        subset_size: int | float | None = None,
    ):
        if subset_size is not None:
            dataset = dataset.subset(subset_size)
        _no_dataset_msg = (
            "Dataset must be provided when initializing the model "
            "if atomic_energies or energy_scale is set to 'auto'."
        )

        if energy_shift_mode != "atomic_energies":
            raise ValueError("For MACE models, energy_shift_mode must be 'atomic_energies'")

        if energy_mean is not None:
            warnings.warn("energy_mean is ignored for MACE models", stacklevel=1)

        # Atomic energies to shift
        if atomic_energies == "auto":
            if dataset is None:
                raise ValueError(_no_dataset_msg)
            atomic_energies = dataset.avg_atomic_property("energy")
        elif atomic_energies is None:
            atomic_energies = {s: 0.0 for s in self.species}
        elif isinstance(atomic_energies, dict):
            if set(self.species) != set(atomic_energies.keys()):
                raise ValueError("Keys of the dictionary must match the species.")
            atomic_energies = atomic_energies
        else:
            raise ValueError("Invalid value is given for atomic_energies. Must be 'auto', dictionart, or None.")

        # Energy scale to multiply
        if energy_scale == "auto":
            if dataset is None:
                raise ValueError(_no_dataset_msg)
            if energy_scale_mode == "energy_std":
                energy_scale = dataset.get_statistics("energy", per_atom=False, reduce="std")
            elif energy_scale_mode == "force_rms":
                energy_scale = dataset.get_statistics("force", per_species=False, reduce="rms")
            else:
                raise ValueError("Invalid value is given for energy_scale_mode. Must be 'energy_std' or 'force_rms'.")
        elif energy_scale is None:
            energy_scale = 1.0
        elif isinstance(energy_scale, (float, int, np.ndarray, torch.Tensor)):
            energy_scale = energy_scale
        else:
            raise ValueError("Invalid value is given for energy_scale. Must be 'auto', float, or None.")

        # Set values
        _atomic_energies = torch.as_tensor([atomic_energies[s] for s in self.species], dtype=torch.float)
        self.atomic_energies_fn.atomic_energies.data = _atomic_energies
        self.global_energy_scale.scale.data = torch.as_tensor(energy_scale, dtype=torch.float)
