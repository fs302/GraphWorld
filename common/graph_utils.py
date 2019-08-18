import numpy as np
import networkx as nx
import scipy.sparse as ss
import sys
import common.data_utils as data_utils
import random
import torch
from sklearn.model_selection import StratifiedKFold

def load_graph_adj_map(file_path):
    context_matrix = dict()
    max_source = 0
    max_target = 0
    with open(file_path) as f:
        for l in f:
            data = l.strip().split()
            id1 = int(data[0])
            if id1 > max_source:
                max_source = id1
            id2 = int(data[1])
            if id2 > max_target:
                max_target = id2
            if len(data) > 2:
                weight = float(data[2])
            else:
                weight = 1.0
            context_matrix.setdefault(id1, {})
            context_matrix[id1][id2] = weight
    return context_matrix, max_source, max_target


def load_networkx_format(file_path):
    g = nx.DiGraph()
    try:
        with open(file_path) as f:
            print("loading file:", file_path)
            for l in f:
                data = l.strip().split()
                id1 = int(data[0])
                id2 = int(data[1])
                if len(data) > 2:
                    weight = float(data[2])
                else:
                    weight = 1.0
                g.add_edge(id1, id2)
    except IOError:
        print("error:", sys.exc_info()[0])
        raise
    print("#nodes:",g.number_of_nodes(),",#edges:",g.number_of_edges())
    nx_adj = nx.adj_matrix(g)
    return nx_adj


def sparse_to_tuple(sparse_mx):
    if not ss.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def train_test_split(nx_adj, test_frac=.2, val_frac=.1, pos_neg_ratio=.5):

    # Remove diagonal elements
    nx_adj = nx_adj - ss.dia_matrix((nx_adj.diagonal()[np.newaxis, :], [0]), shape=nx_adj.shape)
    nx_adj.eliminate_zeros()

    g = nx.from_scipy_sparse_matrix(nx_adj)

    adj_triu = ss.triu(nx_adj)
    edges, weights, adj_shape = sparse_to_tuple(adj_triu)

    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    np.random.shuffle(edge_tuples)

    num_test = int(np.floor(edges.shape[0] * test_frac))
    num_val = int(np.floor(edges.shape[0] * val_frac))
    test_edges = edge_tuples[:num_test]
    val_edges = edge_tuples[num_test:num_test+num_val]
    train_edges = edge_tuples[num_test+num_val:]

    # TODO: optimize for connected component, not to break bridge

    print("Negative Sampling.")
    all_edge_set = set(edge_tuples)
    neg_edge_set = set()
    nodes_set = g.nodes
    while len(neg_edge_set) < len(edge_tuples)/pos_neg_ratio:
        idx_i = np.random.randint(0, nx_adj.shape[0])
        idx_j = np.random.randint(0, nx_adj.shape[0])
        if idx_i == idx_j or idx_i not in nodes_set or idx_j not in nodes_set:
            continue
        neg_edge = (idx_i, idx_j)
        if neg_edge in all_edge_set or neg_edge in neg_edge_set:
            continue
        neg_edge_set.add(neg_edge)

    neg_edge_tuples = list(neg_edge_set)
    num_test_neg = int(num_test/pos_neg_ratio)
    num_val_neg = int(num_val/pos_neg_ratio)
    test_neg_edges = neg_edge_tuples[:num_test_neg]
    val_neg_edges = neg_edge_tuples[num_test_neg:num_test_neg + num_val_neg]
    train_neg_edges = neg_edge_tuples[num_test_neg + num_val_neg:]

    return train_edges, test_edges, val_edges, train_neg_edges, test_neg_edges, val_neg_edges

def dump_network2data(g, file_name):
    out_file = open(file_name, "w")
    for u, v in list(g.edges):
        out_file.writelines(str(u)+"\t"+str(v)+"\n")
    out_file.close()


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_graphs(file_path, degree_as_tag):
    '''
            file_path: path of dataset
            test_proportion: ratio of test train split
            seed: random seed for random splitting of dataset
        '''

    print('loading graphs:',file_path)
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open(file_path, 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def separate_graph_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list



def main():
    net_file = data_utils.get_data_path("twitter")
    adj = load_networkx_format(net_file)
    # g = nx.from_scipy_sparse_matrix(adj)
    # g_int = nx.convert_node_labels_to_integers(g)
    # out_file = "/Users/shenfan/Code/Project/GraphWorld/data/twitter/twitter.txt"
    # dump_network2data(g_int, out_file)

if __name__ == '__main__':
    main()