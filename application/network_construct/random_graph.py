# toy-model
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# init network
nodes = 100
G = nx.Graph() # undirected Graph
for node in range(nodes):
    G.add_node(node)

# simulation: each round, a person talk to his stranger, and build a link with him/her.
round_num = 3

for rd in range(round_num):
    candidate_nodes = set([i for i in range(nodes)])
    for node_i in range(nodes):
        if node_i not in candidate_nodes:
            continue
        candidate_nodes.remove(node_i)
        node_j = np.random.choice(list(candidate_nodes),1)[0]
        candidate_nodes.remove(node_j)
        G.add_edge(node_j, node_i)
        print(node_i, node_j)

# evalutation: how much broken are the network
cc = nx.connected_components(G)
node_community = [0 for i in range(nodes)]
for i, c_set in enumerate(cc):
    print(c_set)
    for node_id in c_set:
        node_community[node_id] = i+1
print("# Connected Components:", max(node_community))
nx.draw(G, node_color=node_community, node_size=50)
plt.show()
