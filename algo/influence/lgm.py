import networkx as nx
import os, sys
projct_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projct_root_path)
import common.graph_utils as graph_utils
import common.data_utils as data_utils

def Local_gravity_model(g, depth=2):
    lgm_results = {}
    degrees = dict(g.degree())
    for node in g.nodes():
        candidates = {}
        neighbors = g.neighbors
        depth_now = 1
        cur_level = list(neighbors(node))
        visited = set(cur_level)
        while depth_now <= depth and len(cur_level) > 0:
            next_level = set()
            for target in cur_level:
                if target not in candidates:
                    candidates[target] = depth_now
                for child in neighbors(target):
                    if child not in visited:
                        visited.add(child)
                        next_level.add(child)
            cur_level = next_level
            depth_now += 1
        gravity = 0
        for target in candidates:
            distance = candidates[target]
            if target != node and distance <= depth:
                partial_gravity = degrees[node]*degrees[target]/(distance**2)
                gravity += partial_gravity
        lgm_results[node] = gravity
    return lgm_results

if __name__ == '__main__':
    # g = nx.karate_club_graph()
    net_file = data_utils.get_data_path("lyb")
    g = graph_utils.load_basic_network(net_file)
    lgm = Local_gravity_model(g)
    print("Local Gravity Model:")
    print(sorted(lgm.items(), key=lambda v: v[1], reverse=True))