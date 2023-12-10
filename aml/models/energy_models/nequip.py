from collections import OrderedDict

import ase.data
import torch
from e3nn import o3

from aml.common.registry import registry
from aml.common.scatter import scatter
from aml.data.utils import compute_neighbor_vecs
from aml.data import keys as K
from aml.nn.nequip import (
    AtomwiseLinear,
    ConvNetLayer,
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)
from aml.nn.nequip import additional_keys as AK
from aml.nn.nequip.graph_mixin import GraphModuleMixin
from aml.nn.nequip.utils import _irreps_compatible
from aml.nn.scale import PerSpeciesScaleShift
from aml.typing import DataDict, Tensor

from .base import BaseEnergyModel


def SimpleIrrepsConfig(config, prefix: str | None = None):
    """Builder that pre-processes options to allow "simple" configuration of irreps."""

    # We allow some simpler parameters to be provided, but if they are,
    # they have to be correct and not overridden
    simple_irreps_keys = ["l_max", "parity", "num_features"]
    real_irreps_keys = [
        "chemical_embedding_irreps_out",
        "feature_irreps_hidden",
        "irreps_edge_sh",
        "conv_to_output_hidden_irreps_out",
    ]

    prefix = "" if prefix is None else f"{prefix}_"

    has_simple: bool = any((f"{prefix}{k}" in config) or (k in config) for k in simple_irreps_keys)
    has_full: bool = any((f"{prefix}{k}" in config) or (k in config) for k in real_irreps_keys)
    assert has_simple or has_full

    update = {}
    if has_simple:
        # nothing to do if not
        lmax = config.get(f"{prefix}l_max", config["l_max"])
        parity = config.get(f"{prefix}parity", config["parity"])
        num_features = config.get(f"{prefix}num_features", config["num_features"])
        update[f"{prefix}chemical_embedding_irreps_out"] = repr(o3.Irreps([(num_features, (0, 1))]))  # n scalars
        update[f"{prefix}irreps_edge_sh"] = repr(o3.Irreps.spherical_harmonics(lmax=lmax, p=-1 if parity else 1))
        update[f"{prefix}feature_irreps_hidden"] = repr(
            o3.Irreps([(num_features, (l, p)) for p in ((1, -1) if parity else (1,)) for l in range(lmax + 1)])  # noqa
        )
        update[f"{prefix}conv_to_output_hidden_irreps_out"] = repr(
            # num_features // 2  scalars
            o3.Irreps([(max(1, num_features // 2), (0, 1))])
        )

    # check update is consistant with config
    # (this is necessary since it is not possible
    #  to delete keys from config, so instead of
    #  making simple and full styles mutually
    #  exclusive, we just insist that if full
    #  and simple are provided, full must be
    #  consistant with simple)
    for k, v in update.items():
        if k in config:
            assert config[k] == v, (
                f"For key {k}, the full irreps options had value `{config[k]}` inconsistant "
                f"with the value derived from the simple irreps options `{v}`"
            )
        config[k] = v


class AtomTypeMapping(torch.nn.Module):
    """Maps atomic numbers to a contiguous range of integers.

    Args:
        species: List of atomic species to consider.
    """

    def __init__(self, species: list[int]):
        super().__init__()
        sorted_species = sorted(species)
        sorted_numbers = [ase.data.atomic_numbers[s] for s in sorted_species]
        mapping = torch.full((max(sorted_numbers) + 1,), -1, dtype=torch.long)
        for i, n in enumerate(sorted_numbers):
            mapping[n] = i
        self.register_buffer("mapping", mapping)

    def forward(self, atoms_graph: DataDict) -> DataDict:
        atoms_graph[AK.elem_map] = self.mapping[atoms_graph[K.elems]]
        return atoms_graph


@registry.register_energy_model("nequip")
class NequIP(GraphModuleMixin, BaseEnergyModel):
    """The NequIP model, which applies E(3) equivariant convolution layers using e3nn.
    Provides "node_features" and "node_vec_features" embeddings.

    Args:
        species (List[str]): List of atomic species to consider.
        cutoff (float): Cutoff radius for the ACSF. Default is 5.0.
        num_layers (int): Number of convolution layers. Default is 4.
        l_max (int): Maximum spherical harmonic degree. Default is 2.
        num_features (int): Number of hidden features for atom. Default is 64.
        parity (bool): Whether to include both even and odd parities in irreps. Default is True.
        invariant_layers (int): Number of invariant layers. Default is 2.
        invariant_neurons (int): Number of invariant neurons. Default is 64.
        use_sc (bool): Whether to use skip connections. Default is True.
        num_radial_basis (int): Number of radial basis functions. Default is 8.
        trainable_radial_basis (bool): Whether to train the radial basis functions. Default is True.
        polynomial_cutoff_p (float): Cutoff exponent for polynomial cutoff. Default is 6.0.
        resnet (bool): Whether to use ResNet-style skip connections. Default is False.
        nonlinearity_scalars (dict[str, str]): Nonlinearity to use for scalar features.
            Different nonlinearities are used for even and odd parities.
            Default is {"e": "silu", "o": "tanh"}.
        nonlinearity_gates (dict[str, str]): Nonlinearity to use for gates.
            Different nonlinearities are used for even and odd parities.
            Default is {"e": "silu", "o": "tanh"}.
        avg_num_neighbors (float): Average number of neighbors per atom. Required.
    """

    embedding_keys = [K.node_features, K.node_vec_features]

    def __init__(
        self,
        species: list[str],
        cutoff: float = 5.0,
        num_layers: int = 4,
        l_max: int = 2,
        num_features: int = 64,
        parity: bool = True,
        invariant_layers: int = 2,
        invariant_neurons: int = 64,
        use_sc: bool = True,
        num_radial_basis: int = 8,
        trainable_radial_basis: bool = True,
        polynomial_cutoff_p: float = 6.0,
        resnet: bool = False,
        nonlinearity_scalars: dict[str, str] = {"e": "silu", "o": "tanh"},  # noqa
        nonlinearity_gates: dict[str, str] = {"e": "silu", "o": "tanh"},  # noqa
        avg_num_neighbors: float | None = None,  # required
    ):
        BaseEnergyModel.__init__(self, species, cutoff)
        if avg_num_neighbors is None:
            raise ValueError("avg_num_neighbors is required")
        self.num_types = len(species)
        self.l_max = l_max
        self.num_features = num_features
        self.parity = parity
        self.num_layers = num_layers
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.use_sc = use_sc
        self.r_max = cutoff
        self.num_radial_basis = num_radial_basis
        self.trainable_radial_basis = trainable_radial_basis
        self.polynomial_cutoff_p = polynomial_cutoff_p
        self.resnet = resnet
        self.nonlinearity_scalars = nonlinearity_scalars
        self.nonlinearity_gates = nonlinearity_gates
        self.avg_num_neighbors = avg_num_neighbors

        self.irreps_config = {"l_max": l_max, "num_features": num_features, "parity": parity}
        SimpleIrrepsConfig(self.irreps_config)
        layers = self._build_layers()
        if isinstance(layers, dict):
            module_list = list(layers.values())
        else:
            module_list = list(layers)
        # check in/out irreps compatible
        for m1, m2 in zip(module_list, module_list[1:], strict=False):
            assert _irreps_compatible(m1.irreps_out, m2.irreps_in), (
                f"Incompatible irreps_out from {type(m1).__name__} for input to {type(m2).__name__}: "
                f"{m1.irreps_out} -> {m2.irreps_in}"
            )
        self._init_irreps(
            irreps_in=module_list[0].irreps_in,
            my_irreps_in=module_list[0].irreps_in,
            irreps_out=module_list[-1].irreps_out,
        )
        # torch.nn.Sequential will name children correctly if passed an OrderedDict
        if isinstance(layers, dict):
            layers = OrderedDict(layers)
        else:
            layers = OrderedDict((f"module{i}", m) for i, m in enumerate(module_list))

        self.layers = torch.nn.ModuleDict(layers)
        self.atom_type_mapping = AtomTypeMapping(species)
        self.species_energy_scale = PerSpeciesScaleShift(species)

        hidden_irreps = o3.Irreps(self.irreps_config["feature_irreps_hidden"])
        self.num_scalar_features = hidden_irreps.count("0e")
        self.num_vector_features = hidden_irreps.count("1e")

    def _build_layers(self) -> dict:
        layers = {}
        layers["one_hot"] = OneHotAtomEncoding(num_types=self.num_types, set_features=True, irreps_in=None)
        layers["spharm_edges"] = SphericalHarmonicEdgeAttrs(
            irreps_in=layers["one_hot"].irreps_out, irreps_edge_sh=self.irreps_config["irreps_edge_sh"]
        )
        layers["radial_basis"] = RadialBasisEdgeEncoding(
            irreps_in=layers["spharm_edges"].irreps_out,
            basis_kwargs={
                "r_max": self.r_max,
                "num_basis": self.num_radial_basis,
                "trainable": self.trainable_radial_basis,
            },
            cutoff_kwargs={"r_max": self.r_max, "p": self.polynomial_cutoff_p},
        )
        layers["chemical_embedding"] = AtomwiseLinear(
            irreps_in=layers["radial_basis"].irreps_out,
            irreps_out=self.irreps_config["chemical_embedding_irreps_out"],
            out_field=K.node_features,
        )

        # add convnet layers
        # insertion preserves order
        for layer_i in range(self.num_layers):
            if layer_i == 0:
                irreps_in = layers["chemical_embedding"].irreps_out
            else:
                irreps_in = layers[f"layer{layer_i - 1}_convnet"].irreps_out
            layers[f"layer{layer_i}_convnet"] = ConvNetLayer(
                irreps_in=irreps_in,
                feature_irreps_hidden=self.irreps_config["feature_irreps_hidden"],
                nonlinearity_scalars=self.nonlinearity_scalars,
                nonlinearity_gates=self.nonlinearity_gates,
                resnet=self.resnet,
                convolution_kwargs={
                    "invariant_layers": self.invariant_layers,
                    "invariant_neurons": self.invariant_neurons,
                    "avg_num_neighbors": self.avg_num_neighbors,
                    "nonlinearity_scalars": self.nonlinearity_scalars,
                    "use_sc": self.use_sc,
                },
            )
        layers["convnet_to_output_hidden"] = AtomwiseLinear(
            out_field="node_hidden_features",
            irreps_in=list(layers.values())[-1].irreps_out,
            irreps_out=self.irreps_config["conv_to_output_hidden_irreps_out"],
        )
        layers["output_hidden_to_scalar"] = AtomwiseLinear(
            field="node_hidden_features",
            out_field=K.atomic_energy,
            irreps_in=layers["convnet_to_output_hidden"].irreps_out,
            irreps_out="1x0e",
        )
        return layers

    def forward(self, data: DataDict) -> Tensor:
        compute_neighbor_vecs(data)
        data = self.atom_type_mapping(data)
        for module in self.layers.values():
            data = module(data)
        energy_i = data[K.atomic_energy].squeeze(-1)
        energy_i = self.species_energy_scale(data, energy_i)
        # Compute system energy
        energy = scatter(energy_i, data[K.batch], dim=0, reduce="sum")

        # Embeddings
        node_features = data[K.node_features]
        data[K.node_features] = node_features[:, : self.num_scalar_features]
        data[K.node_vec_features] = node_features[
            :, self.num_scalar_features : self.num_scalar_features + self.num_vector_features * 3  # noqa
        ].view(-1, self.num_vector_features, 3)

        return energy
