import networkx as nx
import numpy as np

import os, sys
projct_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projct_root_path)
import algo.node2vec.node2vec as node2vec 
from algo.influence.lgm import Local_gravity_model
from algo.influence.sir import sir_ranking

'''
data list
    1. ego-net graphs
    2. influence features(ego-view for influecned state and central nodes)
    3. vertice ids
    4. vertex raw features
    5. vertex embedding
    6. labels
''' 

class deepinf_dataset():

    def __init__(self, graph, active_states=None, labels=None):
        self.g = graph 
        self.n2v_p = 0.5
        self.n2v_q = 2
        self.embedding_dim =  16
        self.n2v_walks = 20
        self.ego_graphs = None
        self.ego_vertices = None

        # degree, pagerank, lgm, embedding
        self.n2v_emb = node2vec.node2vec_emb(self.g, p=self.n2v_p,q=self.n2v_q, out_dim=self.embedding_dim, num_walks=self.n2v_walks)
        self.n2v_emb.learn_embedding()
        self.degree_influence = nx.degree_centrality(self.g)
        self.pagerank_influence =nx.pagerank(self.g, alpha=0.95)
        self.lgm = Local_gravity_model(self.g)
        self.labels = self.generate_label_by_sir()
        

    def single_source_randomwalk(self, source, radius=3, restart_prob=0.8, sample_size=10):
        G = self.g
        sub_graph = nx.bfs_tree(G, source, depth_limit=radius)
        if len(sub_graph.nodes()) < sample_size:
            return list(sub_graph.nodes())
        sample = set()
        cur_node = source
        sample.add(cur_node)
        while len(sample) < sample_size:
            if np.random.random() < restart_prob:
                cur_node = source 
                continue 
            else:
                neighbor = list(sub_graph.neighbors(cur_node))
                if len(neighbor) > 0:
                    next_node = np.random.choice(neighbor,1)[0]
                    sample.add(next_node)
                    cur_node = next_node
                else:
                    cur_node = source
        return sample

    def sample_near_neighbors_rwr(self, sample_size=3):
        graphs = []
        vertices = []
        influence_features = []
        graph_node_features = []
        sample_labels = []
        for node in self.g.nodes:
            sample_nodes = list(self.single_source_randomwalk(node, sample_size=sample_size))
            graphs.append(nx.adjacency_matrix(g.subgraph(sample_nodes)))
            vertices.append(sample_nodes)
            state_tag = []
            node_features = []
            for v in sample_nodes:
                if v == node:
                    state_tag.append([0,1]) # [influenced_state, is_center_node]
                else:
                    state_tag.append([0,0]) # [influenced_state, is_center_node]
                node_feature = np.concatenate(
                            (np.array([self.degree_influence[v],self.pagerank_influence[v],self.lgm[v]]),self.n2v_emb.model.wv[str(v)])
                            ,axis=None)
                node_features.append(node_feature)
            influence_features.append(state_tag)
            sample_labels.append(self.labels[node])
            graph_node_features.append(node_features)
        return graphs, vertices, influence_features, graph_node_features, sample_labels

    
    def generate_label_by_sir(self, num_epoch=100):
        sir = sir_ranking(self.g, num_epoch=num_epoch) 
        rank_score = sorted(sir.items(),key=lambda x:x[1],reverse=True)
        split_pos = len(rank_score)/2
        labels = {}
        for pos, item in enumerate(rank_score):
            node = item[0]
            if pos < split_pos:
                labels[node] = 1
            else:
                labels[node] = 0
        return labels


if __name__ == '__main__':
    g = nx.karate_club_graph()
    dataset = deepinf_dataset(g)
    # print(dataset.n2v_emb.model.wv['0'])
    graphs, vertices, influence_features, graph_node_features, labels = dataset.sample_near_neighbors_rwr(sample_size=5)
    
