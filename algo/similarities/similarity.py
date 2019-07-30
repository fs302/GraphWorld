import numpy as np
import copy


def adamic_adar(node1, node2, context_matrix):
    '''
    cite: Adamic L A, Adar E. Friends and neighbors on the web[J]. Social networks, 2003, 25(3): 211-230.
    :param node1:
    :param node2:
    :param context_matrix:
    :return: AA similarity
    '''
    sim = 0
    if node1 in context_matrix and node2 in context_matrix:
        set1 = set(context_matrix[node1])
        set2 = set(context_matrix[node2])
        interset = set1&set2
        for node in interset:
            if node in context_matrix:
                sim += 1/np.log(len(context_matrix[node]))
    return sim

def common_neighbors(node1, node2, context_matrix):
    sim = 0
    if node1 in context_matrix and node2 in context_matrix:
        set1 = set(context_matrix[node1])
        set2 = set(context_matrix[node2])
        sim = len(set1&set2)
    return sim


def local_path_index(node1, node2, context_matrix):
    '''
    Lü L, Jin C H, Zhou T. Similarity index based on local paths for link prediction of complex networks[J].
    :param node1: 
    :param node2: 
    :param context_matrix: 
    :return: 
    '''    
    sim = 0
    eps = 0.001
    if node1 in context_matrix and node2 in context_matrix:
        set1 = set(context_matrix[node1])
        set2 = set(context_matrix[node2])
        sim = len(set1&set2)
    if node1 in context_matrix:
        a3 = 0
        for agent1 in context_matrix[node1]:
            for agent2 in context_matrix[agent1]:
                if node2 in context_matrix[agent2]:
                    a3 += 1
        sim += eps * a3
        # print eps * a3
    return sim 


def common_attributes(node1, node2, context_matrix):
    if node1 in context_matrix and node2 in context_matrix[node1]:
        return 1.0*context_matrix[node1][node2]
    return 0.0
    

def pathsim(node1, node2, context_matrix):
    if node1 in context_matrix and node1 in context_matrix[node1]:
        p_xx = 1.0*context_matrix[node1][node1]
    else:
        p_xx = 0.0
    if node2 in context_matrix and node2 in context_matrix[node2]:
        p_yy = 1.0*context_matrix[node2][node2]
    else:
        p_yy = 0.0
    if node1 in context_matrix and node2 in context_matrix[node1]:
        p_xy = 1.0*context_matrix[node1][node2]
    else:
        p_xy = 0.0
    if p_xx+p_yy>0:
        return 2*p_xy/(p_xx+p_yy)
    else:
        return 0


def jaccard_sim(node1, node2, context_matrix):
    sim = 0
    if node1 in context_matrix and node2 in context_matrix:
        set1 = set(context_matrix[node1])
        set2 = set(context_matrix[node2])
        sim = len(set1&set2)/len(set1|set2)
    return sim


def jaccard_sim_attributes(node1, node2, context_matrix):
    if node1 in context_matrix and node1 in context_matrix[node1]:
        p_xx = 1.0*context_matrix[node1][node1]
    else:
        p_xx = 0.0
    if node2 in context_matrix and node2 in context_matrix[node2]:
        p_yy = 1.0*context_matrix[node2][node2]
    else:
        p_yy = 0.0
    if node1 in context_matrix and node2 in context_matrix[node1]:
        p_xy = 1.0*context_matrix[node1][node2]
    else:
        p_xy = 0.0
    if p_xx+p_yy-p_xy>0:
        return p_xy/(p_xx+p_yy-p_xy)
    else:
        return 0


def cosine_sim(node1, node2, context_matrix):
    sim = 0
    if node1 in context_matrix and node2 in context_matrix:
        set1 = set(context_matrix[node1])
        set2 = set(context_matrix[node2])
        sim = len(set1&set2)/(np.sqrt(len(set1))*np.sqrt(len(set2)))
    return sim


def cosine_sim_attributes(node1, node2, context_matrix):
    if node1 in context_matrix and node1 in context_matrix[node1]:
        p_xx = 1.0*context_matrix[node1][node1]
    else:
        p_xx = 0.0
    if node2 in context_matrix and node2 in context_matrix[node2]:
        p_yy = 1.0*context_matrix[node2][node2]
    else:
        p_yy = 0.0
    if node1 in context_matrix and node2 in context_matrix[node1]:
        p_xy = 1.0*context_matrix[node1][node2]
    else:
        p_xy = 0.0
    if p_xx*p_yy>0:
        return p_xy/(np.sqrt(p_xx)*np.sqrt(p_yy))
    else:
        return 0


def simrank(context_matrix, K=3, c=0.8):
    sim_matrix = dict()
    node = set(context_matrix)
    # O(KN^2d^2)
    for a in node:
            sim_matrix.setdefault(a,{})
            sim_matrix[a][a] = 1.0
    for k in range(K):
        last_sim_mat = copy.copy(sim_matrix)
        accu_sim = dict()
        for a in last_sim_mat:
            for b in last_sim_mat[a]:
                for i in context_matrix[a]:
                    for j in context_matrix[b]:
                        if (i < j):
                            accu_sim.setdefault(i,{})
                            accu_sim[i].setdefault(j,0.0)
                            accu_sim[i][j] += last_sim_mat[a][b]

        for i in node:
            for j in node:
                if (i < j and i in accu_sim and j in accu_sim[i] and context_matrix[i]>0 and context_matrix[j]>0):
                    sim_matrix[i][j] = c * accu_sim[i][j] / (len(context_matrix[i]) * len(context_matrix[j]))
                    sim_matrix[j][i] = sim_matrix[i][j]
    return sim_matrix


def personalized_pagerank(context_matrix, K=3, epsilon=0.1):
    node = set(context_matrix)
    sim_matrix = dict()
    for a in node:
        sim_matrix.setdefault(a,{})
        sim_matrix[a][a] = 1.0
    # O(KN^2d)
    for k in range(K):
        last_sim_mat = copy.copy(sim_matrix)
        accu_sim = dict()
        for x in node:
            for k in last_sim_mat[x]:
                for y in context_matrix[k]:
                    accu_sim.setdefault(x,{})
                    accu_sim[x].setdefault(y,0.0)
                    if (x in last_sim_mat and k in last_sim_mat[x]):
                        accu_sim[x][y] += (1-epsilon) * last_sim_mat[x][k]*context_matrix[k][y]
                    if y == x:
                        accu_sim[x][y] += epsilon
        for x in accu_sim:
            for y in accu_sim[x]:
                sim_matrix.setdefault(x,{})
                sim_matrix[x][y] = accu_sim[x][y]
    return sim_matrix


def panther(context_matrix, T=10, R=1000):
    node = context_matrix.keys()
    transition_candidate = dict()
    transition_prob = dict()
    # conduct transition probability matrix
    for x in node:
        transition_candidate[x] = []
        transition_prob[x] = []
        sum_prob = 0.0
        for y in context_matrix[x]:
            transition_candidate[x].append(y)
            transition_prob[x].append(context_matrix[x][y])
            sum_prob += context_matrix[x][y]
        for i in range(len(transition_prob[x])):
            transition_prob[x][i] = transition_prob[x][i]/sum_prob
    
    # RandomPath
    path = []
    node_path = dict()
    for r in range(R):
        path.append([])
        curr = np.random.choice(node)
        for i in range(T):
            nexts = np.random.choice(transition_candidate[curr],1,p=transition_prob[curr])
            if len(nexts)>0:
                next = nexts[0]
                path[r].append(next)             # add v into p_r
                node_path.setdefault(next,[])
                node_path[next].append(r)         # add p_r into P_v
                curr = next
    # accumulate
    accu_sim = dict()
    for v_i in node_path:
        for p in node_path[v_i]:
            for v_j in path[p]:
                if v_i != v_j:
                    accu_sim.setdefault(v_i,{})
                    accu_sim[v_i].setdefault(v_j,0.0)
                    accu_sim[v_i][v_j] += 1.0/R
    return accu_sim



