import networkx as nx
import numpy as np
import os, sys
projct_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projct_root_path)
import common.graph_utils as graph_utils
import common.data_utils as data_utils
import algo.node2vec.node2vec as node2vec 
from algo.influence.lgm import Local_gravity_model
from algo.influence.sir import sir_ranking
from algo.deep_inf.deepinf_dataset import deepinf_dataset,ChunkSampler
from algo.deep_inf.gat import BatchGAT
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

def evaluate(epoch, loader, thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    for i_batch, batch in enumerate(loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)

        # if args.cuda:
        #     features = features.cuda()
        #     graph = graph.cuda()
        #     labels = labels.cuda()
        #     vertices = vertices.cuda()

        output = model(features, vertices, graph)
        output = output[:, -1, :]
        loss_batch = F.nll_loss(output, labels)
        loss += bs * loss_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("%sloss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
            log_desc, loss / total, auc, prec, rec, f1)

    # tensorboard_logger.log_value(log_desc + 'loss', loss / total, epoch + 1)
    # tensorboard_logger.log_value(log_desc + 'auc', auc, epoch + 1)
    # tensorboard_logger.log_value(log_desc + 'prec', prec, epoch + 1)
    # tensorboard_logger.log_value(log_desc + 'rec', rec, epoch + 1)
    # tensorboard_logger.log_value(log_desc + 'f1', f1, epoch + 1)

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return best_thr
    else:
        return None


def train(epoch, train_loader, valid_loader, test_loader, log_desc='train_'):
    model.train()

    loss = 0.
    total = 0.
    for i_batch, batch in enumerate(train_loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)
        # print(graph, features, labels, vertices)
        # if args.cuda:
        #     features = features.cuda()
        #     graph = graph.cuda()
        #     labels = labels.cuda()
        #     vertices = vertices.cuda()

        optimizer.zero_grad()
        output = model(features, vertices, graph)
        output = output[:, -1, :]
        loss_train = F.nll_loss(output, labels)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss in this epoch %f", loss / total)
    # tensorboard_logger.log_value('train_loss', loss / total, epoch + 1)
    check_point = 10
    if (epoch + 1) % check_point == 0:
        logger.info("epoch %d, checkpoint!", epoch)
        best_thr = evaluate(epoch, valid_loader, return_best_thr=True, log_desc='valid_')
        evaluate(epoch, test_loader, thr=best_thr, log_desc='test_')

network_name = 'facebook'
net_file = data_utils.get_data_path(network_name)
g = graph_utils.load_basic_network(net_file)
influence_dataset = deepinf_dataset(g, neighbor_size=10)
# influence_dataset.make()
target_path = projct_root_path+"/algo/deep_inf/"+network_name+"_preprocess"
influence_dataset.load(target_path)

# print(influence_dataset.embedding)
# print(influence_dataset.ego_virtices)

# gat
n_heads = [1,1,1]
feature_dim = influence_dataset.get_feature_dimension()
hidden_units = '16,8'
n_classes = 2
n_units = [feature_dim] + [int(x) for x in hidden_units.strip().split(",")] + [n_classes]
use_vertex_feature = False
model_name = 'gat'
model = BatchGAT(pretrained_emb=influence_dataset.embedding,
        vertex_feature=influence_dataset.graph_node_features,
        use_vertex_feature=use_vertex_feature,
        n_units=n_units, n_heads=n_heads,
        dropout=0.2, instance_normalization=False)

params = [{'params': filter(lambda p: p.requires_grad, model.parameters())
    if model_name == "pscn" else model.layer_stack.parameters()}]
learning_rate = 1e-4
weight_decay = 5e-4
optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)

N = len(influence_dataset.ego_graphs)
logger.info("ego_graphs num:"+str(N))
train_ratio, valid_ratio = 50, 20
batch_size = 2048
train_start,  valid_start, test_start = \
        0, int(N * train_ratio / 100), int(N * (train_ratio + valid_ratio) / 100)
train_loader = DataLoader(influence_dataset, batch_size=batch_size,
                        sampler=ChunkSampler(valid_start - train_start, 0))
valid_loader = DataLoader(influence_dataset, batch_size=batch_size,
                        sampler=ChunkSampler(test_start - valid_start, valid_start))
test_loader = DataLoader(influence_dataset, batch_size=batch_size,
                        sampler=ChunkSampler(N - test_start, test_start))

# Train model
t_total = time.time()
logger.info("training...")
num_epochs = 500
for epoch in range(num_epochs):
    train(epoch, train_loader, valid_loader, test_loader)
logger.info("optimization Finished!")
logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

logger.info("retrieve best threshold...")
best_thr = evaluate(num_epochs, valid_loader, return_best_thr=True, log_desc='valid_')

# Testing
logger.info("testing...")
evaluate(num_epochs, test_loader, thr=best_thr, log_desc='test_')