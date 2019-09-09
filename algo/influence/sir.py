import networkx as nx
import numpy as np

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

    # Iteration
    for it in range(max_iter):
        status_change = False 
        for source in [i for i, v in node_status.items() if v == available_statuses["Infected"]]:
            for target in list(g.adj[source]):
                # Infecte Stage
                if node_status[target] == available_statuses["Susceptible"]:
                    if np.random.rand() < beta:
                        node_status[target] = available_statuses["Infected"]
                        status_change = True
            node_status[source] = available_statuses["Recovered"] # Recover Stage
            status_change = True
        if status_change == False:
            break
    
    succ_infected = [i for i, v in node_status.items() if v == available_statuses["Recovered"]]

    return succ_infected

def sir_ranking(g, num_epoch=100):
    sir_score = {}
    n = len(list(g.nodes()))
    for node in g.nodes():
        node_influence = 0
        for epoch in range(num_epoch):
            sir_results = sir_simulation(g,source=node)
            f = len(sir_results)/n
            node_influence += f
        avg_node_inf = node_influence/num_epoch if num_epoch>0 else 0.0
        sir_score[node] = avg_node_inf
    return sir_score

if __name__ == '__main__':
    g = nx.karate_club_graph()
    sir_score = sir_ranking(g,1000)
    print(sorted(sir_score.items(), key=lambda v: v[1], reverse=True))