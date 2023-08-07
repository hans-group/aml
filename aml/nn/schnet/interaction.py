import torch
from torch_geometric.nn import MessagePassing

from aml.nn.activation import ShiftedSoftplus
from aml.typing import Tensor


class CFConv(MessagePassing):
    """Continuous-filter convolutional layer used in Schnet.
    # Add a reference to the paper here.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_filters (int): Number of filters.
        nn (torch.nn.Module): An filter-generating neural network.
        cutoff (float): Cutoff distance. used in computing cosine cutoff function.
    """

    propagate_type = {"x": Tensor, "W": Tensor}

    def __init__(self, in_channels: int, out_channels: int, num_filters: int, nn: torch.nn.Module, cutoff: float):
        super().__init__(aggr="add")
        self.lin1 = torch.nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = torch.nn.Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:
        """Compute continuous-filter convolution.

        Args:
            x (Tensor): Node feature embeddings.
            edge_index (Tensor): Edge indices. Bond or neighbor list.
            edge_weight (Tensor): Edge weights.
            edge_attr (Tensor): Edge attributes.

        Returns:
            Tensor: Updated node feature embeddings.
        """
        C = 0.5 * (torch.cos(edge_weight * torch.pi / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W, size=(x.size(0), x.size(0)))
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class SchnetInteractionBlock(torch.nn.Module):
    """Interaction block used by SchNet model.

    Args:
        hidden_channels (int): Number of hidden channels in CFConv.
        num_basis (int): Number of expanded basis of distance.
        num_filters (int): Number of convolutional filters.
        cutoff (float): Cutoff radius for distance.
    """

    def __init__(self, hidden_channels: int, num_basis: int, num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_basis, num_filters),
            ShiftedSoftplus(),
            torch.nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff).jittable()
        self.act = ShiftedSoftplus()
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:
        """Compute interaction block.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices (neighbor list).
            edge_weight (Tensor): Edge weights
            edge_attr (Tensor): Edge attributes

        Returns:
            Tensor: Updated node features.
        """
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x
