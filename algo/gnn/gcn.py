import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, dropout_ratio=0.6):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index)
        return x