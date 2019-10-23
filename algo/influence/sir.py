import networkx as nx
import numpy as np
from tqdm import tqdm
import multiprocessing as mp 
import time
import os, sys
projct_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projct_root_path)
import common.graph_utils as graph_utils
import common.data_utils as data_utils
import random

# Thanks Sara for a better virsion
def individual_SIR(G,node,beta):
    # individual infection: calculate the influence power of targetnode, G is the networkx structure
    # beta is the infection ratio
    importance = 0 
    infectnumber = 0
    state = dict.fromkeys(G,0) # 0 for susceptible; 1 for infective; -1 for recovery
    infectnodes = set()

    state[node] = 1
    infectnodes.add(node)
    infectnumber += 1 


    while infectnumber > 0: # only if existed infected nodes, the process should continues
        node = infectnodes.pop()
        state[node] = -1
        importance += 1
        infectnumber -= 1

        # consider the current infective  node used to infective its neighbors
        neighbors = G.neighbors # for directed network, got the predecessor node
        neighbors_num = len(list(neighbors(node)))

        if neighbors_num != 0:
            for neighbor in list(neighbors(node)):
                if state[neighbor] == 0: #node-number
                    p = random.uniform(0,1)
                    if p <= beta: 
                        state[neighbor] = 1
                        infectnodes.add(neighbor)
                        infectnumber += 1       
    return importance 


def sir_ranking(g, beta=None, gamma=1.0, num_epoch=100):
    sir_score = {}
    nodes = list(g.nodes())
    n = len(nodes)
    if beta is None:
        beta = opt_beta(g)
    print("nodes:",n,"beta=%f,gamma=%f" % (beta,gamma))
    for i in tqdm(range(n)):
        node = nodes[i]
        node_influence = 0
        for epoch in range(num_epoch):
            sir_inf = individual_SIR(g,node,beta)
            node_influence += sir_inf
            # sir = sir_simulation(g, node, beta)
            # node_influence += len(sir)/n
        avg_node_inf = node_influence/num_epoch if num_epoch>0 else 0.0
        sir_score[node] = avg_node_inf
    return sir_score


def opt_beta(g):
    k1s = 0
    k2s = 0
    for node in g.nodes():
        k1s += len(list(g.neighbors(node)))
        k2s += len(list(nx.bfs_tree(g, source=node, depth_limit=2).edges()))
    if k2s > k1s:
        beta = k1s/(k2s-k1s)
    else:
        beta = 0.1 
    return beta


if __name__ == '__main__':
    # g = nx.karate_club_graph()
    # sir_score = sir_ranking(g,num_epoch=1000)
    # print(sorted(sir_score.items(), key=lambda v: v[1], reverse=True))\
    net_file = data_utils.get_data_path("BlogCatalog")
    g = graph_utils.load_basic_network(net_file)
    st = time.time()
    sir = sir_ranking(g, beta=opt_beta(g), gamma=1.0, num_epoch=100)
    out_file = net_file.split('.')[0]+'-sir.txt'
    with open(out_file,'w') as f:
        for i, v in sir.items():
            f.write(str(i)+'\t'+str(round(v,6))+'\n')
    print('time used:',time.time()-st)