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

font_path = '/Users/shenfan/anaconda2/envs/py3/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf'
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=font_path)
from matplotlib.font_manager import _rebuild
_rebuild()
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['font.family'] ='sans-serif'
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False 

def main():
    test_network = ["lyb","facebook","twitter","BlogCatalog"]
    for net in test_network:
        net_file = data_utils.get_data_path("lyb")
        g = graph_utils.load_basic_network(net_file)
        sir = sir_ranking(g,num_epoch=250)
        centralities = [nx.degree_centrality, nx.closeness_centrality, nx.eigenvector_centrality, 
            nx.pagerank, Local_gravity_model]
        for c in centralities:
            if c.__name__ == 'pagerank':
                res = c(g, alpha=0.8, tol=0.001)
            else:
                res = c(g)
            # res_list = [(v, round(c,4)) for v, c in res.items()]
            tau, p = kendallTau(res, sir)
            print("%s\t%s\t%f" % (net,c.__name__, tau))
            # visialize
            # v_max = max([v for k,v in res_list])
            # vmap = [v/v_max for k,v in res_list]
            # nx.draw_spring(g, with_labels=True,labels=node_labels, node_color=vmap,
            #                 font_size=8, node_size=350)
            # plt.show()

if __name__ == '__main__':
    main()