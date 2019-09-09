import networkx as nx

def Local_gravity_model(g, depth=3):
    lgm_results = {}
    degrees = dict(g.degree())
    for node in g.nodes():
        candidates = nx.shortest_path_length(g, source=node)
        gravity = 0
        for target in candidates:
            distance = candidates[target]
            if target != node and distance <= depth:
                partial_gravity = degrees[node]*degrees[target]/(distance**2)
                gravity += partial_gravity
        lgm_results[node] = gravity
    return lgm_results


if __name__ == '__main__':
    g = nx.karate_club_graph()
    lgm = Local_gravity_model(g)
    print("Local Gravity Model:")
    print(sorted(lgm.items(), key=lambda v: v[1], reverse=True))