import os
import torch
import sys, time
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

projct_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projct_root_path)

from algo.gnn.gcn import GCN
from algo.gnn.graphsage import GraphSage
from algo.gnn.gat import GAT
from algo.gnn.appnp import APPNP_Net


dataset = Planetoid(root='../data/', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# Heterophily Analysis

match, total = 0, 0
for edge in data.edge_index.t():
    total += 1
    match += 1 if data.y[edge[0]] == data.y[edge[1]] else 0
print('homo_ratio:', match/total)

def train(model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def test(mask, model):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    return acc

def run_exp(num_epoch, model_name, model, optimizer, criterion):
    st = time.time()
    best_test_acc = 0
    for epoch in range(1, num_epoch+1):
        loss = train(model, criterion, optimizer)
        val_acc = test(data.val_mask, model)
        test_acc = test(data.test_mask, model)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if epoch % 100 == 0:
            print(f'model: {model_name}, Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    et = time.time()
    p_time = (et-st)/num_epoch*100
    print(f'model: {model_name}, time_used_100_epoch:{p_time:.2f}, best_test_acc:{best_test_acc:.4f}') 

num_epoch = 500
# GCN
model = GCN(dataset.num_features, hidden_channels=16, out_channels=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
run_exp(num_epoch, 'GCN', model, optimizer, criterion)

# GraphSage
model = GraphSage(dataset.num_features, hidden_channels=16, out_channels=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
run_exp(num_epoch, 'GraphSage', model, optimizer, criterion)

# GAT
model = GAT(dataset.num_features, hidden_channels=16, out_channels=dataset.num_classes, heads=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
run_exp(num_epoch, 'GAT', model, optimizer, criterion)

# APPNP
model = APPNP_Net(dataset.num_features, hidden_channels=16, out_channels=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
run_exp(num_epoch, 'APPNP', model, optimizer, criterion)