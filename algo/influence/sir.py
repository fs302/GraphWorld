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

available_statuses = {
    "Susceptible": 0,
    "Infected": 1,
    "Recovered": 2,
    "Blocked": -1
}

def sir_simulation(g, source=0, beta=0.2, gamma=1.0, max_iter=100):
    """
    g: networkx graph
    source: source node id
    beta: The infection rate (float value in [0,1]), default 0.2
    gamma: The recovery rate (float value in [0,1]), default 1.0
    max_iter: iteration threshold, default 100
    """
    # Init
    node_status = {}
    for node in g.nodes():
        node_status[node] = available_statuses["Susceptible"]
    node_status[source] = available_statuses["Infected"]
    infected_set = set([source])
    # Iteration
    for it in range(max_iter):
        status_change = False 
        candidate_set = set()
        for source in infected_set:
            for target in list(g.adj[source]):
                # Infecte Stage
                if node_status[target] == available_statuses["Susceptible"]:
                    if np.random.rand() < beta:
                        node_status[target] = available_statuses["Infected"]
                        candidate_set.add(target)
                        status_change = True
            node_status[source] = available_statuses["Recovered"] # Recover Stage
            status_change = True
        infected_set = candidate_set
        if not status_change:
            break
    
    succ_infected = [i for i, v in node_status.items() if v == available_statuses["Recovered"]]

    return succ_infected

def node_sir_score(g, source, beta, gamma, num_epoch):
    n = len(list(g.nodes()))
    node_influence = 0.0
    print("Processing: ", source)
    for epoch in range(num_epoch):
        sir_results = sir_simulation(g,source=source,beta=beta,gamma=gamma)
        f = len(sir_results)/n
        node_influence += f
    return node_influence/num_epoch

def sir_ranking(g, beta=0.02, gamma=1.0, num_epoch=100):
    sir_score = {}
    nodes = list(g.nodes())
    n = len(nodes)
    for i in tqdm(range(n)):
        node = nodes[i]
        node_influence = 0
        for epoch in range(num_epoch):
            sir_results = sir_simulation(g,source=node,beta=beta,gamma=gamma)
            f = len(sir_results)/n
            node_influence += f
        avg_node_inf = node_influence/num_epoch if num_epoch>0 else 0.0
        sir_score[node] = avg_node_inf
    return sir_score

def sir_parallel(g, beta, gamma, num_epoch):
    pool = mp.Pool(processes=8)
    nodes = list(g.nodes())
    results = [pool.apply(node_sir_score, args=(g, node, beta, gamma, num_epoch)) for node in nodes]
    return results

if __name__ == '__main__':
    # g = nx.karate_club_graph()
    # sir_score = sir_ranking(g,num_epoch=1000)
    # print(sorted(sir_score.items(), key=lambda v: v[1], reverse=True))\
    net_file = data_utils.get_data_path("facebook")
    g = graph_utils.load_basic_network(net_file)
    st = time.time()
    sir = sir_ranking(g, beta=0.2, gamma=1.0, num_epoch=100)
    out_file = net_file.split('.')[0]+'-sir.txt'
    with open(out_file,'w') as f:
        for i, v in sir.items():
            f.write(str(i)+'\t'+str(round(v,6))+'\n')
    print('time used:',time.time()-st)