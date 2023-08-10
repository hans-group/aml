import torch

from aml.data import keys as K
from aml.typing import DataDict

from ..radial_basis import BesselRBF, GaussianRBF
from .interaction import SchnetInteractionBlock


class SchNetRepresentation(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        n_filters: int = 128,
        n_interactions: int = 6,
        rbf_type: str = "gaussian",
        n_rbf: int = 50,
        trainable_rbf: bool = False,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_filters = n_filters
        self.n_interactions = n_interactions
        self.n_rbf = n_rbf
        self.trainable_rbf = trainable_rbf
        self.cutoff = cutoff

        # Layers
        self.embedding = torch.nn.Embedding(100, hidden_channels, padding_idx=0)
        if rbf_type == "gaussian":
            self.rbf = GaussianRBF(n_rbf, self.cutoff, trainable=trainable_rbf)
        elif rbf_type == "bessel":
            self.rbf = BesselRBF(n_rbf, self.cutoff, trainable=trainable_rbf)
        else:
            raise ValueError("Unknown radial basis function type: {}".format(rbf_type))
        self.interactions = torch.nn.ModuleList()
        for _ in range(n_interactions):
            block = SchnetInteractionBlock(hidden_channels, n_rbf, n_filters, cutoff)
            self.interactions.append(block)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()

    def forward(self, data: DataDict) -> DataDict:
        z = data[K.elems]
        edge_index = data[K.edge_index]  # neighbors
        edge_vec = data[K.edge_vec]  # vectors to neighbors

        edge_weight = torch.linalg.norm(edge_vec, dim=1)  # distances to neighbors
        edge_attr = self.rbf(edge_weight)  # gaussian smearing expansion

        # Representation task
        h = self.embedding(z)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        data[K.node_features] = h

        return data
