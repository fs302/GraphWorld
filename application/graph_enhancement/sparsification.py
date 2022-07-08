import networkx as nx 
import numpy as np
from numpy.linalg import norm

def global_sparsification_by_aa(edge_list, del_ratio=0.5, add_ratio=0.5, debug=False):
    # construct networkx graph
    g = nx.from_edgelist(edge_list, create_using=nx.Graph)
    del_edge_cnt = min(int(del_ratio * len(edge_list)), len(edge_list))
    add_edge_cnt = int(add_ratio * len(edge_list))
    
    # remove edges
    if del_ratio > 0:
        preds = nx.adamic_adar_index(g, edge_list)
        results = []
        for u, v, p in preds:
            results.append((p, u, v))
        results.sort()
        edge_list_cut = np.array(results[del_edge_cnt:], dtype=int)[:,1:3]
        if debug:
            print('highest score removed:{}', results[del_edge_cnt-1][0])
    else:
        edge_list_cut = edge_list
    
    # add edges
    if add_ratio > 0:
        candidate_links = set()
        for node_i in g.nodes:
            for node_j in nx.neighbors(g, node_i):
                for node_k in nx.neighbors(g, node_j):
                    if node_i < node_k and not g.has_edge(node_i, node_k):
                        candidate_links.add((node_i, node_k))
        preds = nx.adamic_adar_index(g, list(candidate_links))

        results = []
        for u, v, p in preds:
            if u != v:
                results.append((p, u, v))
        results.sort(reverse=True)
        add_edge_list = np.array(results[:add_edge_cnt], dtype=int)[:,1:3]
        if debug:
            print('lowest score added:{}', results[add_edge_cnt][0])
    
    if del_ratio >= 1:
        result_edges = add_edge_list
    elif add_ratio == 0:
        result_edges = edge_list_cut
    else:
        result_edges = np.concatenate((edge_list_cut,add_edge_list), axis=0)
    return result_edges


def global_sparsification_by_feature_sim(edge_list, feature, del_ratio=0.5, add_ratio=0.5, sim_func='cos'):
    
    def similarity_func(x1, x2):
        if sim_func == 'cos':
            return np.dot(x1, x2)/(norm(x1)*norm(x2))
        elif sim_func == 'euc':
            return 1-norm(x1-x2)
        elif sim_func == 'rand':
            return np.random.rand()
        return 1
    
    # construct networkx graph
    g = nx.from_edgelist(edge_list, create_using=nx.Graph)
    del_edge_cnt = min(int(del_ratio * len(edge_list)), len(edge_list))
    add_edge_cnt = int(add_ratio * len(edge_list))
    
    # remove edges
    if del_ratio > 0:
        results = []
        for u, v in edge_list:
            p = similarity_func(feature[u], feature[v])
            results.append((p, u, v))
        results.sort()
        edge_list_cut = np.array(results[del_edge_cnt:], dtype=int)[:,1:3]
    else:
        edge_list_cut = edge_list
    
    # add edges
    if add_ratio > 0:
        candidate_links = set()
        for node_i in g.nodes:
            for node_j in nx.neighbors(g, node_i):
                for node_k in nx.neighbors(g, node_j):
                    if node_i < node_k and not g.has_edge(node_i, node_k):
                        candidate_links.add((node_i, node_k))

        results = []
        for u, v in list(candidate_links):
            if u != v:
                p = similarity_func(feature[u], feature[v])
                results.append((p, u, v))
        results.sort(reverse=True)
        add_edge_list = np.array(results[:add_edge_cnt], dtype=int)[:,1:3]
    
    if del_ratio >= 1:
        result_edges = add_edge_list
    elif add_ratio == 0:
        result_edges = edge_list_cut
    else:
        result_edges = np.concatenate((edge_list_cut,add_edge_list), axis=0)
    return result_edges