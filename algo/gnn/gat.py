import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(input_channels, hidden_channels, heads) 
        self.conv2 = GATConv(hidden_channels * heads, out_channels) 

    def forward(self, x, edge_index, dropout_ratio=0.6):
        x = F.dropout(x, p=dropout_ratio, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index)
        return x