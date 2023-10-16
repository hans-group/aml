from aml.common.registry import registry
from aml.common.scatter import scatter
from aml.data import keys as K
from aml.nn.equivariant_transformer import EquivariantTransformerRepresentation
from aml.nn.mlp import MLP
from aml.typing import DataDict, Tensor

from .base import BaseEnergyModel


@registry.register_energy_model("equivariant_transformer")
class EquivariantTransformer(BaseEnergyModel):
    """The TorchMD-net equivariant Transformer model.
    This model applies graph attention layers with PaiNN-style equivariant convolutions.
    Provides "node_features" and "node_vec_features" embeddings.
    Code taken and modified from https://github.com/torchmd/torchmd-net.

    Args:
        species: List of atomic species to consider.
        cutoff: Cutoff radius for the ACSF.
        hidden_channels: Number of hidden channels in the Transformer.
        num_layers: Number of Transformer layers.
        num_rbf: Number of radial basis functions.
        rbf_type: Type of radial basis functions.
        trainable_rbf: Whether to train the radial basis functions.
        activation: Activation function to use in the Transformer.
        attn_activation: Activation function to use in the attention layers.
        neighbor_embedding: Whether to use neighbor embedding.
        num_heads: Number of attention heads.
        distance_influence: Whether to use distance influence.
        cutoff_lower: Lower cutoff radius for the Transformer.
    """

    embedding_keys = [K.node_features, K.node_vec_features]

    def __init__(
        self,
        species,
        cutoff: float = 5.0,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 50,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = True,
        activation: str = "silu",
        attn_activation: str = "silu",
        neighbor_embedding: bool = True,
        num_heads: int = 8,
        distance_influence: str = "both",
        cutoff_lower: float = 0.0,
    ):
        super().__init__(species, cutoff)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower

        # Layers
        self.representation = EquivariantTransformerRepresentation(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            attn_activation=attn_activation,
            neighbor_embedding=neighbor_embedding,
            num_heads=num_heads,
            distance_influence=distance_influence,
            cutoff_lower=cutoff_lower,
            max_z=100,
        )

        self.energy_output = MLP(
            n_input=hidden_channels,
            n_output=1,
            hidden_layers=(hidden_channels // 2,),
            activation="silu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.energy_output.reset_parameters()
        self.representation.reset_parameters()

    def forward(self, data: DataDict) -> Tensor:
        data = self.representation(data)
        # Compute per-atom energy
        energy_i = self.energy_output(data[K.node_features]).squeeze(-1)
        energy_i = self.species_energy_scale(data, energy_i)
        # Compute system energy
        energy = scatter(energy_i, data[K.batch], dim=0, reduce="sum")
        return energy
