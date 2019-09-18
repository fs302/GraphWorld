import os, sys
projct_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projct_root_path)
import common.graph_utils as graph_utils
import common.data_utils as data_utils
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
import algo.node2vec.node2vec as node2vec 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['font.family'] ='sans-serif'
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False 

class node2vec_emb():
    def __init__(self, 
                    g, 
                    p = 1, 
                    q = 1, 
                    out_dim=16, 
                    num_walks = 10, 
                    walk_length = 80, 
                    window_size=10):
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.G = node2vec.Graph(g, g.is_directed(), p, q)
        self.dimensions = out_dim
        self.window_size = window_size
        self.workers = 8
        self.iter = 1
        self.model = None
        self.emb = {}

    def learn_embedding(self):
        self.G.preprocess_transition_probs()
        simulate_walks = self.G.simulate_walks(self.num_walks, self.walk_length)
        walks = [list(map(str, walk)) for walk in simulate_walks]
        self.model = Word2Vec(walks, size=self.dimensions, window=self.window_size, min_count=0, sg=1, workers=self.workers, iter=self.iter)

    def output_embedding(self, outfile):
        self.model.wv.save_word2vec_format(outfile)

def tsne_plot(model, node_labels):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(node_labels[int(word)])
    
    tsne_model = TSNE(n_components=2, perplexity=50, learning_rate=20.0, n_iter=1000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(12, 9)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

def main():
    # g = nx.karate_club_graph()
    net_file = data_utils.get_data_path("lyb")
    g = graph_utils.load_basic_network(net_file)
    n2v_emb = node2vec_emb(g,p=1,q=1, out_dim=128, num_walks=20)
    n2v_emb.learn_embedding()
    out_file = net_file.split('.')[0]+'-n2v_emb.txt'
    n2v_emb.output_embedding(out_file)
    # node_labels = graph_utils.load_node_labels(data_utils.get_node_path("lyb"))
    # print(n2v_emb.model.most_similar(positive=['23']))
    #tsne_plot(n2v_emb.model, node_labels)
    

if __name__ == '__main__':
    main()