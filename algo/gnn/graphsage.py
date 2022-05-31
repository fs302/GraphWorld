import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSage(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, aggr='add'):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = SAGEConv(input_channels, hidden_channels) 
        self.conv2 = SAGEConv(hidden_channels, out_channels) 

    def forward(self, x, edge_index, dropout_ratio=0.6):
        x = F.dropout(x, p=dropout_ratio, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index)
        return x

if __name__ == '__main__':

    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    dataset = Planetoid(root='../../data/', name='Cora', transform=NormalizeFeatures())

    model = GraphSage(dataset.num_features, 16, dataset.num_classes, aggr='max')
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    data = dataset[0]  # Get the first graph object.

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


    for epoch in range(1, 501):
        loss = train()
        val_acc = test(data.val_mask)
        test_acc = test(data.test_mask)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')