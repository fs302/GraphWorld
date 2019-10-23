import networkx as nx
import numpy as np

import os, sys
projct_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projct_root_path)
import common.graph_utils as graph_utils
import common.data_utils as data_utils
import algo.node2vec.node2vec as node2vec 
from algo.influence.lgm import Local_gravity_model
from algo.influence.sir import sir_ranking
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

'''
data list
    1. ego-net graphs
    2. influence features(ego-view for influecned state and central nodes)
    3. vertice ids
    4. vertex raw features
    5. vertex embedding
    6. labels
''' 

class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

class deepinf_dataset(Dataset):

    def __init__(self, graph, emb_dim=64, neighbor_size = 10, sir_file=None):
        self.g = graph 
        self.n2v_p = 0.5
        self.n2v_q = 2
        self.embedding_dim =  emb_dim
        self.n2v_walks = 20
        self.neighbor_sample_size = neighbor_size
        self.n_classes = 2
        self.sir_file = sir_file

        self.ego_graphs = None 
        self.ego_virtices = None
        self.influence_features = None
        self.graph_node_features = None
        self.graph_labels = None

    def make(self):
        # degree, pagerank, lgm, embedding
        logger.info("learning node2vec embedding.")
        self.n2v_emb = node2vec.node2vec_emb(self.g, p=self.n2v_p,q=self.n2v_q, out_dim=self.embedding_dim, num_walks=self.n2v_walks)
        self.n2v_emb.learn_embedding()
        logger.info("counting node influence:degree_centrality")
        self.degree_influence = nx.degree_centrality(self.g)
        logger.info("counting node influence:page_rank")
        self.pagerank_influence =nx.pagerank(self.g, alpha=0.95)
        logger.info("counting node influence:local_gravity_model")
        self.lgm = Local_gravity_model(self.g)
        logger.info("making label by sir model.")
        self.sir_labels = self.generate_label_by_sir()

        # neighbor sample
        logger.info("sample neighbors and attach features")
        self.sample_near_neighbors_rwr()

        self.N = self.ego_graphs.shape[0]
        logger.info("%d ego networks loaded, each with size %d" % (self.N, self.ego_graphs.shape[1]))
        
    def generate_label_by_sir(self, num_epoch=100):
        sir = {}
        if self.sir_file:
            with open(self.sir_file,'r') as f:
                for l in f:
                    data = l.split()
                    id = int(data[0])
                    score = float(data[1])
                    sir[id] = score 
        else:
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
        return list(sample)

    def sample_near_neighbors_rwr(self):
        graphs = []
        vertices = []
        influence_features = []
        graph_node_features = []
        sample_labels = []
        embeddings = np.zeros((max(self.g.nodes)+1,self.embedding_dim))
        for node in self.g.nodes:
            sample_nodes = self.single_source_randomwalk(node, sample_size=self.neighbor_sample_size)
            # assert len(sample_nodes) == self.neighbor_sample_size
            # in case when sample nodes can not meet the requirements
            sub_graph_adj = nx.adjacency_matrix(self.g.subgraph(sample_nodes)).todense()
            mask_adj = np.zeros((self.neighbor_sample_size,self.neighbor_sample_size))
            mask_adj[:sub_graph_adj.shape[0],:sub_graph_adj.shape[1]] += sub_graph_adj
            graphs.append(mask_adj)
            vertices.append(sample_nodes)
            state_tag = np.zeros((self.neighbor_sample_size,2)) # [influenced_state, is_center_node]
            node_features = np.zeros((self.neighbor_sample_size,3))
            for i,v in enumerate(sample_nodes):
                if v == node:
                    state_tag[i,] = [0,1] # [influenced_state, is_center_node]
                else:
                    state_tag[i,] = [0,0] # [influenced_state, is_center_node]
                node_feature = np.concatenate(
                            (np.array([self.degree_influence[v],self.pagerank_influence[v],self.lgm[v]]))
                            ,axis=None)
                node_features[i,] = node_feature
            
            influence_features.append(state_tag)
            sample_labels.append(self.sir_labels[node])
            graph_node_features.append(node_features)
            embeddings[node,] = self.n2v_emb.model.wv[str(node)]

        self.ego_graphs = np.array(graphs).astype(np.dtype('B'))
        self.ego_virtices = np.array(vertices)
        self.influence_features = np.array(influence_features).astype(np.float32)
        self.graph_node_features = graph_node_features
        self.graph_labels = sample_labels
        self.embedding = torch.FloatTensor(embeddings)
        
    def get_feature_dimension(self):
        return self.influence_features.shape[-1]

    def get_vertex_features(self):
        return self.graph_node_features

    def get_embedding(self):
        return self.embedding

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.ego_graphs[idx], self.influence_features[idx], self.graph_labels[idx], self.ego_virtices[idx]

    def save(self, file_dir):
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        np.save(os.path.join(file_dir, "adjacency_matrix.npy"),self.ego_graphs)
        print('%d ego graph saved to adjacency_matrix.npy.' % (len(self.ego_graphs)))
        np.save(os.path.join(file_dir, "influence_feature.npy"),self.influence_features)
        print('influence_feature.npy saved.')
        np.save(os.path.join(file_dir,"label.npy"),self.graph_labels)
        print('label.npy saved.')
        np.save(os.path.join(file_dir, "vertex_id.npy"), self.ego_virtices)
        print('vertex_id.npy saved.')
        np.save(os.path.join(file_dir, "vertex_feature.npy"), self.graph_node_features)
        print('vertex_feature.npy saved.')
        np.save(os.path.join(file_dir, "embedding.npy"),self.embedding)
        print('embedding.npy saved.')

    def load(self, file_dir):
        self.ego_graphs = np.load(os.path.join(file_dir, "adjacency_matrix.npy"))
        self.influence_features = np.load(os.path.join(file_dir, "influence_feature.npy")).astype(np.float32)
        self.graph_labels = np.load(os.path.join(file_dir, "label.npy"))
        self.ego_virtices = np.load(os.path.join(file_dir, "vertex_id.npy"))
        self.graph_node_features = torch.FloatTensor(np.load(os.path.join(file_dir, "vertex_feature.npy")))
        self.embedding = torch.FloatTensor(np.load(os.path.join(file_dir, "embedding.npy")))
        print("%s dataset loaded." % (file_dir))

if __name__ == '__main__':
    network_name = 'facebook'
    net_file = data_utils.get_data_path(network_name)
    g = graph_utils.load_basic_network(net_file)
    dataset = deepinf_dataset(g, sir_file=net_file.split('.')[0]+'-sir.txt')
    target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),network_name+"_preprocess")
    dataset.make()
    dataset.save(target_path)
    # dataset.load(target_path)
