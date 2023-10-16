from aml.common.registry import registry
from aml.common.scatter import scatter
from aml.common.utils import compute_neighbor_vecs
from aml.data import keys as K
from aml.nn.mlp import MLP
from aml.nn.schnet.representation import SchNetRepresentation
from aml.typing import DataDict, Tensor

from .base import BaseEnergyModel


@registry.register_energy_model("schnet")
class SchNet(BaseEnergyModel):
    """SchNet model.
    This model applies continuous-filter convolutional layers to atomic features.
    Provides "node_features" embeddings.

    Args:
        species (list[str]): List of atomic species to consider.
        cutoff (float): Cutoff radius for interactions. Defaults to 5.0.
        hidden_channels (int): Number of hidden channels in the convolutional layers. Defaults to 128.
        n_filters (int): Number of convolutional filters. Defaults to 128.
        n_interactions (int): Number of convolutional layers. Defaults to 6.
        rbf_type (str): Type of radial basis functions. One of "gaussian" or "bessel".
            Defaults to "gaussian".
        n_rbf (int): Number of radial basis functions. Defaults to 50.
        trainable_rbf (bool): Whether to train the radial basis functions. Defaults to False.
    """
    embedding_keys = [K.node_features]

    def __init__(
        self,
        species,
        cutoff: float = 5.0,
        hidden_channels: int = 128,
        n_filters: int = 128,
        n_interactions: int = 6,
        rbf_type: str = "gaussian",
        n_rbf: int = 50,
        trainable_rbf: bool = False,
    ):
        super().__init__(species, cutoff)
        self.hidden_channels = hidden_channels
        self.n_filters = n_filters
        self.n_interactions = n_interactions
        self.rbf_type = rbf_type
        self.n_rbf = n_rbf
        self.trainable_rbf = trainable_rbf

        self.representation = SchNetRepresentation(
            hidden_channels=hidden_channels,
            n_filters=n_filters,
            n_interactions=n_interactions,
            rbf_type=rbf_type,
            n_rbf=n_rbf,
            trainable_rbf=trainable_rbf,
            cutoff=cutoff,
        )
        self.energy_output = MLP(
            n_input=hidden_channels,
            n_output=1,
            hidden_layers=(hidden_channels // 2,),
            activation="shifted_softplus",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.energy_output.reset_parameters()
        self.representation.reset_parameters()

    def forward(self, data: DataDict) -> Tensor:
        compute_neighbor_vecs(data)
        data = self.representation(data)
        # Compute per-atom energy
        energy_i = self.energy_output(data[K.node_features]).squeeze(-1)
        energy_i = self.species_energy_scale(data, energy_i)
        # Compute system energy
        energy = scatter(energy_i, data[K.batch], dim=0, reduce="sum")
        return energy
