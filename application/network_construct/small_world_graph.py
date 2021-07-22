import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def draw_degree_dist(g, degree_base=0, title='Network Distribution'):
    degree_freq = nx.degree_histogram(g)
    degree_x = range(len(degree_freq))
    plt.loglog(degree_x[degree_base:], degree_freq[degree_base:], 'go-')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

def network_stats(graph, network_name='Network'):

    print(network_name)
    print("边的条数："+str(graph.number_of_edges()))
    print("点的个数："+str(graph.number_of_nodes()))

    #计算平均度
    a_e = graph.number_of_edges() * 2.0 / graph.number_of_nodes()
    print("平均度："+str(a_e))

    #计算同配系数
    a = nx.degree_assortativity_coefficient(graph)
    print("同配系数："+str(a))

    #计算连通性
    n = nx.is_connected(graph)
    print("连通性："+str(n))

    #计算聚类系数
    c = nx.average_clustering(graph)
    print("聚类系数："+str(c))

    # #计算平均最短路径
    if nx.is_connected(graph):
        p = nx.average_shortest_path_length(graph)
        print("平均最短路径："+str(p))

n = 1000
# ER-Network
p = 0.005
g = nx.erdos_renyi_graph(n, p)
g = g.to_undirected()
draw_degree_dist(g, title='ER-Network Distribution')
network_stats(g, 'ER-Network')

# BA-Network
m = 10
g = nx.barabasi_albert_graph(n, m)
g = g.to_undirected()
draw_degree_dist(g, m, title='BA-Network Distribution')
network_stats(g, 'BA-Network')

# WS-Network
k = 20
p = 0.01
g = nx.watts_strogatz_graph(n, k, p)
g = g.to_undirected()
draw_degree_dist(g, k, title='WS-Network Distribution')
network_stats(g, 'WS-Network')

# powerlaw_cluster network
m = 10
p = 1
g = nx.powerlaw_cluster_graph(n, m, p)
g = g.to_undirected()
draw_degree_dist(g, m, title='PC-Network Distribution')
network_stats(g, 'PC-Network')

# Facebook Network
import os, sys
projct_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projct_root_path)
import common.graph_utils as graph_utils
import common.data_utils as data_utils

net_file = data_utils.get_data_path("facebook")
g = graph_utils.load_basic_network(net_file)
g = g.to_undirected()
draw_degree_dist(g, m, title='Facebook-Network Distribution')
network_stats(g, 'Facebook-Network')