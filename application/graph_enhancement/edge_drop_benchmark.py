import os, sys, time
import torch
import numpy as np
import copy
os.environ['TORCH'] = torch.__version__

projct_root_path = os.path.dirname(os.path.dirname(os.path.abspath('')))
sys.path.append(projct_root_path)
from algo.gnn.gcn import GCN
from algo.gnn.graphsage import GraphSage
from algo.gnn.gat import GAT
from algo.gnn.appnp import APPNP_Net
from common.graph_utils import homo_ratio
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from sparsification import global_sparsification_by_aa, global_sparsification_by_feature_sim

# using different dataset accrossing diff homophily
# CORA/Citeseer/Pubmed/Cornell/Texas

def data_prepare(dataset_name='Cora'):
    '''
        Args:
            dataset_name
        Returns:
            an object contains properties and data of the graph
    '''
    if dataset_name in ('Cora','CiteSeer','PubMed'):
        dataset = Planetoid(root='../../data/', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]  # Get the first graph object.
        hr = homo_ratio(data.edge_index.t(), data.y)
        print('homo ratio:',hr)
    return data, dataset.num_features, dataset.num_classes
    

def train(model, optimizer, criterion, x, edge_index, y, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test(model, mask, x, edge_index, y):
    model.eval()
    out = model(x, edge_index)
    pred = out.argmax(dim=1)
    correct = pred[mask] == y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc

def run_exp(data, model_name, args, debug=True):
    st = time.time()
    best_test_acc = 0
    if model_name == 'GCN':
        model = GCN(args['num_features'], 
            hidden_channels=args['hidden_channels'], 
            out_channels=args['num_classes'])
    elif model_name == 'GraphSage':
        model = GraphSage(args['num_features'], 
            hidden_channels=args['hidden_channels'],
            out_channels=args['num_classes'],
            num_layers=args['num_layers'],
            dropout_ratio=args['model_dropout_ratio']
            )
    elif model_name == 'GAT':
        model = GAT(args['num_features'],
                hidden_channels=args['hidden_channels'],
                out_channels=args['num_classes'],
                heads=args['attention_heads'],
                dropout_ratio=args['model_dropout_ratio']
        )
    elif model_name == 'APPNP':
        model = APPNP_Net(args['num_features'],
                hidden_channels=args['hidden_channels'],
                out_channels=args['num_classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    num_epoch = args['num_epoch'] if 'num_epoch' in args else 1
    for epoch in range(1, num_epoch+1):
        loss = train(model, optimizer, criterion, 
            data.x, data.edge_index, data.y, data.train_mask)
        val_acc = test(model, data.val_mask, data.x, data.edge_index, data.y)
        test_acc = test(model, data.test_mask, data.x, data.edge_index, data.y)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if epoch % 100 == 0 and debug==True:
            print(f'model: {model_name}, Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    et = time.time()
    p_time = (et-st)/num_epoch*100
    drop_rate = args['drop_rate']
    dataset_name = args['dataset']
    edge_size = len(data.edge_index.t())
    if debug:
        print(f'{dataset_name}\t{drop_rate:.3f}\t{edge_size}\t{model_name}\t{p_time:.2f}\t{best_test_acc:.4f}') 
    return best_test_acc

dataset_name = 'Cora'
data, num_features, num_classes = data_prepare(dataset_name=dataset_name)
args = {
    'dataset': dataset_name,
    'rounds': 10,
    'num_epoch': 200,
    'hidden_channels': 32,
    'num_layers': 3, 
    'model_dropout_ratio': 0.3,
    'attention_heads': 4,
    'num_features': num_features,
    'num_classes': num_classes,
    'drop_rate': 0.,
    'drop_func': 'cos'
}
print(args)
models = ['GCN','GraphSage','GAT']
for model_name in models:
    for drop_rate in np.arange(0, 0.2, 0.02):
        test_accs = []
        for r in range(args['rounds']):
            args['drop_rate'] = drop_rate
            new_edge_index = data.edge_index
            if args['drop_func'] == 'AA':
                new_edge_index = torch.tensor(global_sparsification_by_aa(
                                            np.array(data.edge_index.t()), 
                                            drop_rate, 
                                            add_ratio=0)
                                        ).t()
            elif args['drop_func'] == 'cos':
                new_edge_index = torch.tensor(global_sparsification_by_feature_sim(
                np.array(data.edge_index.t()),
                data.x,
                drop_rate,
                0,
                sim_func='cos'
                )).t()
            backup_edge_index = data.edge_index
            data.edge_index = new_edge_index
            test_acc = run_exp(data, model_name, args, debug=False)
            test_accs.append(test_acc)
            data.edge_index = backup_edge_index
        avg_test_acc = sum(test_accs)/len(test_accs)
        std_test_acc = np.std(np.array(test_accs))
        print(f'{dataset_name}\t{drop_rate:.3f}\t{model_name}\t{avg_test_acc:.4f}\t±{std_test_acc:.6f}')
