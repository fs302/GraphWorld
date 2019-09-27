import os, sys
projct_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projct_root_path)
import common.graph_utils as graph_utils
import common.data_utils as data_utils
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
import algo.node2vec.node2vec as node2vec 
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['font.family'] ='sans-serif'
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False 

def tsne_plot(model, node_labels):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(node_labels[int(word)])
    
    tsne_model = TSNE(n_components=2, perplexity=100, learning_rate=20.0, n_iter=1000, random_state=23)
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

def explore_lyb():
    net_file = data_utils.get_data_path("lyb")
    g = graph_utils.load_basic_network(net_file)
    n2v_emb = node2vec.node2vec_emb(g,p=0.5,q=2, out_dim=32, num_walks=20)
    n2v_emb.learn_embedding()

    # out_file = net_file.split('.')[0]+'-n2v_emb.txt'
    # n2v_emb.output_embedding(out_file)
    
    node_labels = graph_utils.load_node_labels(data_utils.get_node_path("lyb"))
    sus_best = n2v_emb.model.most_similar(positive=['23'])
    for item in sus_best:
        print(node_labels[int(item[0])], item[1])
    com = list(greedy_modularity_communities(g))
    node_community = {}
    for i,c in enumerate(com):
        for node in c:
            node_community[node_labels[node]] = i
    print(node_community)
    # tsne_plot(n2v_emb.model, node_labels)

def main():
    net_file = data_utils.get_data_path("lyb")
    g = graph_utils.load_basic_network(net_file)
    n2v_emb = node2vec.node2vec_emb(g, p=0.5,q=2, out_dim=64, num_walks=20)
    n2v_emb.learn_embedding()
    # out_file = net_file.split('.')[0]+'-n2v_emb.txt'
    # n2v_emb.output_embedding(out_file)
    

if __name__ == '__main__':
    #main()
    explore_lyb()