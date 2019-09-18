#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, sys
projct_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projct_root_path)
import networkx as nx
import numpy as np
import common.graph_utils as graph_utils
import common.data_utils as data_utils
from common.eval_utils import *
import matplotlib.pyplot as plt
from algo.influence.lgm import Local_gravity_model
from algo.influence.sir import sir_ranking

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['font.family'] ='sans-serif'
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False 

def main():
    test_network = ["karate_club","facebook"]
    for net in test_network:
        net_file = data_utils.get_data_path(net)
        g = graph_utils.load_basic_network(net_file)
        sir_file = net_file.split('.')[0]+'-sir.txt'
        sir = {}
        if os.path.exists(sir_file):
            with open(sir_file,'r') as f:
                for l in f:
                    data = l.split()
                    id = int(data[0])
                    score = float(data[1])
                    sir[id] = score 
        else:
            print("SIR Simulation start.")
            sir = sir_ranking(g, gamma=1.0, num_epoch=100)
            print("SIR Simulation end.")
        centralities = [nx.degree_centrality, nx.closeness_centrality, nx.eigenvector_centrality, 
            nx.pagerank, Local_gravity_model]
        for c in centralities:
            if c.__name__ == 'pagerank':
                res = c(g, alpha=0.95)
            elif c.__name__=='Local_gravity_model':
                res = c(g, depth=2)
            else:
                res = c(g)
            tau, p = kendallTau(res, sir)
            print("%s\t%s\t%f" % (net,c.__name__, tau))
if __name__ == '__main__':
    main()