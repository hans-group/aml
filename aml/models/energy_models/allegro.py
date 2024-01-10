"""Code adapted from Allegro. Below is the original license.

MIT License

Copyright (c) 2022 The President and Fellows of Harvard College

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
"""
from typing import List, Optional

import torch
from e3nn import o3

from aml.common.registry import registry
from aml.data import keys as K
from aml.data.utils import compute_neighbor_vecs
from aml.models.energy_models.nequip import AtomTypeMapping
from aml.nn.allegro import Allegro_Module, EdgewiseEnergySum, NormalizedBasis, ScalarMLP
from aml.nn.allegro.fc import ScalarMLPFunction
from aml.nn.nequip.atomwise import AtomwiseReduce
from aml.nn.nequip.edge_embedding import RadialBasisEdgeEncoding, SphericalHarmonicEdgeAttrs
from aml.nn.nequip.one_hot_embedding import OneHotAtomEncoding
from aml.nn.nequip.radial_basis import BesselBasis
from aml.typing import DataDict, Tensor

from .base import BaseEnergyModel


@registry.register_energy_model("allegro")
class Allegro(BaseEnergyModel):

    """This class implements E(3)-equivariant ``Allegro`` model
        (original paper: A.Musaelian et al.https://doi.org/10.1038/s41467-023-36329-y)
    This model computes per-edge energy using MLP and then sums them up to get total energy.
    Source code is taken from https://github.com/mir-group/allegro
    This model provides "node_features" as embedding vectors.

    Args:
        species (list[str]): List of species.
        cutoff (float, optional): Cutoff radius. Defaults to 5.0.
        cutoff_p (float, optional): Exponent of cutoff function. Defaults to 6.
        num_layers (int, optional): Number of layers. Defaults to 2.
        l_max (int, optional): Maximum angular momentum of spherical harmonics.
            Increase this to apply higher-order equivariance. Defaults to 1.
        parity (str, optional): Parity of spherical harmonics.
            One of "o3_full", "o3_restricted", "so3". Defaults to "o3_full".
        num_basis (int, optional): Number of radial basis functions. Defaults to 8.
        trainable_basis (bool, optional): Whether to train radial basis functions. Defaults to True.
        normalize_basis (bool, optional): Whether to normalize radial basis functions. Defaults to True.
        two_body_latent_mlp_latent_dimensions (list[int], optional): Latent dimensions of two-body MLP.
            Defaults to [32, 64, 128, 256].
        latent_mlp_latent_dimensions (list[int], optional): Latent dimensions of latent (hidden) MLP.
            Defaults to [256, 256].
        env_embed_multiplicity (int, optional): Multiplicity of local geometry embedding.
            Defaults to 32.
        edge_eng_mlp_latent_dimensions (list[int], optional): Latent dimensions of edge energy MLP.
            Defaults to [128].
        avg_num_neighbors (Optional[float], optional): Average number of neighbors. This is required
            although marked as optional. Defaults to None.
        r_start_cos_ratio (float, optional): Ratio of cutoff radius to start radius of cutoff function.
            Defaults to 0.8.
        per_layer_cutoffs (Optional[List[float]], optional): Cutoff radii for each layer.
            Defaults to None.
        edge_sh_normalize (bool, optional): Whether to normalize spherical harmonics.
            Defaults to True.
        edge_sh_normalization (str, optional): Normalization mode for spherical harmonics.
            One of "integral", "component", "norm". Defaults to "component".
        embed_initial_edge (bool, optional): Whether to embed initial edge. Defaults to True.
        linear_after_env_embed (bool, optional): Whether to use linear layer after local geometry embedding.
            Defaults to False.
        nonscalars_include_parity (bool, optional): Whether to include parity in non-scalar features.
            Defaults to True.
        latent_resnet (bool, optional): Whether to use residual connections in latent MLP.
            Defaults to True.
        latent_resnet_update_ratios (Optional[List[float]], optional): Update ratios for residual connections
            in latent MLP. Defaults to None.
        latent_resnet_update_ratios_learnable (bool, optional): Whether to learn update ratios for residual
            connections in latent MLP. Defaults to False.
        pad_to_alignment (int, optional): Padding to alignment. Defaults to 1.
        sparse_mode (Optional[str], optional): Sparse mode. "coo" or "csr". Defaults to None.

    """

    embedding_keys = [K.node_features]

    def __init__(
        self,
        # Model basics
        species: list[str],
        cutoff: float = 5.0,
        cutoff_p: float = 6,
        num_layers: int = 2,
        l_max: int = 1,
        parity: str = "o3_full",  # o3_full, o3_restricted, so3
        num_basis: int = 8,
        trainable_basis: bool = True,
        normalize_basis: bool = True,
        two_body_latent_mlp_latent_dimensions=[32, 64, 128, 256],  # noqa
        latent_mlp_latent_dimensions=[256, 256],  # noqa
        env_embed_multiplicity: int = 32,
        edge_eng_mlp_latent_dimensions=[128],  # noqa
        avg_num_neighbors: Optional[float] = None,
        # Advanced options
        r_start_cos_ratio: float = 0.8,
        per_layer_cutoffs: Optional[List[float]] = None,
        edge_sh_normalize=True,
        edge_sh_normalization="component",
        embed_initial_edge: bool = True,
        linear_after_env_embed: bool = False,
        nonscalars_include_parity: bool = True,
        latent_resnet: bool = True,
        latent_resnet_update_ratios: Optional[List[float]] = None,
        latent_resnet_update_ratios_learnable: bool = False,
        pad_to_alignment: int = 1,
        sparse_mode: Optional[str] = None,
    ):
        super().__init__(species, cutoff)
        assert parity in ("o3_full", "o3_restricted", "so3"), f"Invalid parity {parity}"
        self.num_types = len(species)
        self.l_max = l_max
        self.parity = parity
        self.irreps_edge_sh = repr(o3.Irreps.spherical_harmonics(l_max, p=(1 if parity == "so3" else -1)))
        self.nonscalars_include_parity = parity == "o3_full"
        self.normalize_basis = normalize_basis
        self.trainable_basis = trainable_basis
        self.num_basis = num_basis
        self.edge_sh_normalization = edge_sh_normalization
        self.edge_sh_normalize = edge_sh_normalize
        self.layers = torch.nn.ModuleDict()
        self.r_max = cutoff
        self.avg_num_neighbors = avg_num_neighbors
        self.r_start_cos_ratio = r_start_cos_ratio
        self.cutoff_p = cutoff_p
        self.per_layer_cutoffs = per_layer_cutoffs

        self.num_layers = num_layers
        self.env_embed_multiplicity = env_embed_multiplicity
        self.embed_initial_edge = embed_initial_edge
        self.linear_after_env_embed = linear_after_env_embed
        self.nonscalars_include_parity = nonscalars_include_parity
        self.latent_resnet = latent_resnet
        self.latent_resnet_update_ratios = latent_resnet_update_ratios
        self.latent_resnet_update_ratios_learnable = latent_resnet_update_ratios_learnable
        self.pad_to_alignment = pad_to_alignment
        self.sparse_mode = sparse_mode
        self.two_body_latent_mlp_latent_dimensions = two_body_latent_mlp_latent_dimensions
        self.latent_mlp_latent_dimensions = latent_mlp_latent_dimensions
        self.edge_eng_mlp_latent_dimensions = edge_eng_mlp_latent_dimensions
        self.atom_type_mapping = AtomTypeMapping(species)

        self.layers["one_hot"] = OneHotAtomEncoding(self.num_types, irreps_in=None)
        if normalize_basis:
            self.layers["radial_basis"] = RadialBasisEdgeEncoding(
                basis=NormalizedBasis,
                basis_kwargs={
                    "r_max": self.r_max,
                    "original_basis_kwargs": {
                        "trainable": self.trainable_basis,
                        "num_basis": self.num_basis,
                        "r_max": self.r_max,
                    },
                },
                cutoff_kwargs={"r_max": self.r_max, "p": self.cutoff_p},
                out_field="edge_embedding",
                irreps_in=self.layers["one_hot"].irreps_out,
            )
        else:
            self.layers["radial_basis"] = RadialBasisEdgeEncoding(
                basis=BesselBasis,
                basis_kwargs={"trainable": self.trainable_basis, "num_basis": self.num_basis, "r_max": self.r_max},
                cutoff_kwargs={"r_max": self.r_max, "p": self.cutoff_p},
                out_field="edge_embedding",
                irreps_in=self.layers["one_hot"].irreps_out,
            )

        self.layers["spharm"] = SphericalHarmonicEdgeAttrs(
            irreps_edge_sh=self.irreps_edge_sh,
            edge_sh_normalization=self.edge_sh_normalization,
            edge_sh_normalize=self.edge_sh_normalize,
            irreps_in=self.layers["radial_basis"].irreps_out,
        )
        self.layers["allegro"] = Allegro_Module(
            # required params
            num_layers=self.num_layers,
            num_types=self.num_types,
            r_max=self.r_max,
            avg_num_neighbors=self.avg_num_neighbors,
            # cutoffs
            r_start_cos_ratio=self.r_start_cos_ratio,
            PolynomialCutoff_p=self.cutoff_p,
            per_layer_cutoffs=self.per_layer_cutoffs,
            # general hyperparameters:
            field="edge_attr",
            edge_invariant_field="edge_embedding",
            node_invariant_field="node_attr",
            env_embed_multiplicity=self.env_embed_multiplicity,
            embed_initial_edge=self.embed_initial_edge,
            linear_after_env_embed=self.linear_after_env_embed,
            nonscalars_include_parity=self.nonscalars_include_parity,
            # MLP parameters:
            two_body_latent=ScalarMLPFunction,
            two_body_latent_kwargs={
                "mlp_latent_dimensions": self.two_body_latent_mlp_latent_dimensions,
                "mlp_initialization": "uniform",
            },
            env_embed=ScalarMLPFunction,
            env_embed_kwargs={
                "mlp_latent_dimensions": [],
                "mlp_initialization": "uniform",
                "mlp_nonlinearity": None,
            },
            latent=ScalarMLPFunction,
            latent_kwargs={"mlp_latent_dimensions": self.latent_mlp_latent_dimensions},
            latent_resnet=self.latent_resnet,
            latent_resnet_update_ratios=self.latent_resnet_update_ratios,
            latent_resnet_update_ratios_learnable=self.latent_resnet_update_ratios_learnable,
            latent_out_field=K.edge_features,
            # Performance parameters:
            pad_to_alignment=self.pad_to_alignment,
            sparse_mode=self.sparse_mode,
            # Other:
            irreps_in=self.layers["spharm"].irreps_out,
        )
        self.layers["edge_eng"] = ScalarMLP(
            field=K.edge_features,
            out_field=K.edge_energy,
            mlp_latent_dimensions=self.edge_eng_mlp_latent_dimensions,
            mlp_initialization="uniform",
            mlp_nonlinearity=None,
            mlp_output_dimension=1,
            irreps_in=self.layers["allegro"].irreps_out,
        )
        self.layers["edge_eng_sum"] = EdgewiseEnergySum(
            self.num_types,
            self.avg_num_neighbors,
            irreps_in=self.layers["edge_eng"].irreps_out,
        )

        self.total_energy_sum = AtomwiseReduce(
            field=K.atomic_energy,
            out_field="energy_pred",
            irreps_in=self.layers["edge_eng_sum"].irreps_out,
        )

    def forward(self, data: DataDict) -> Tensor:
        compute_neighbor_vecs(data)
        data = self.atom_type_mapping(data)
        for layer in self.layers.values():
            data = layer(data)
        energy_i = data[K.atomic_energy].squeeze(-1)
        energy_i = self.species_energy_scale(data, energy_i)
        energy_i = energy_i.unsqueeze(-1)
        data[K.atomic_energy] = energy_i

        data = self.total_energy_sum(data)
        energy = data["energy_pred"].squeeze(-1)
        return energy
