import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSage(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, num_layers=3, dropout_ratio=0., aggr='mean'):
        super().__init__()
        torch.manual_seed(1234567)
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_channels, hidden_channels, aggr=aggr))
        for i in range(num_layers-2):
            self.convs.append(SAGEConv(hidden_channels,hidden_channels, aggr=aggr))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x, edge_index):
        for i in range(self.num_layers-1):
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

if __name__ == '__main__':

    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    dataset = Planetoid(root='../../data/', name='Cora', transform=NormalizeFeatures())

    model = GraphSage(dataset.num_features, 64, dataset.num_classes, 3, dropout_ratio=0.5, aggr='mean')
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    data = dataset[0]  # Get the first graph object.
    print(len(data.val_mask), len(data.test_mask))

    def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def test(mask):
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
        return acc

    best_test_acc = 0
    for epoch in range(1, 301):
        loss = train()
        val_acc = test(data.val_mask)
        test_acc = test(data.test_mask)
        best_test_acc = test_acc if test_acc > best_test_acc else best_test_acc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    print(f'Best test accuracy: {best_test_acc}')