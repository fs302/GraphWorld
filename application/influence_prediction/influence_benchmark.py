import networkx as nx
import numpy as np
import common.graph_utils as graph_utils
import common.data_utils as data_utils

def main():
    net_file = data_utils.get_data_path("lyb")
    nx_adj = graph_utils.load_networkx_format(net_file)
    g = nx.from_scipy_sparse_matrix(nx_adj)
    centralities = [nx.degree_centrality, nx.betweenness_centrality, nx.eigenvector_centrality, nx.pagerank, nx.closeness_centrality]
    for c in centralities:
        if c.__name__ == 'pagerank':
            res = c(g, alpha=0.8, tol=0.001)
        else:
            res = c(g)

        res_list = [(v, '{:0.4f}'.format(c)) for v, c in res.items()]
        print(c.__name__, sorted(res_list, key=lambda v: v[1], reverse=True))

if __name__ == '__main__':
    main()