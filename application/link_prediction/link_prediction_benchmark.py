import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import common.graph_utils as graph_utils
import common.data_utils as data_utils
import logging

# create logger
logger_name = "link_prediction_benchmark"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

# create file handler
log_path = "../../logs/link_prediction_benchmark.log"
fh = logging.FileHandler(log_path)
fh.setLevel(logging.INFO)

# create formatter
fmt = "%(asctime)-15s %(levelname)s %(filename)s %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

# add handler and formatter to logger
fh.setFormatter(formatter)
logger.addHandler(fh)

def get_roc_score(edges_pos, edges_neg, score_matrix):
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(score_matrix[edge[0], edge[1]])  # predicted score
        pos.append(1)  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(score_matrix[edge[0], edge[1]])  # predicted score
        neg.append(0)  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def main():
    for dataset_name in data_utils.data_file:
        net_file = data_utils.get_data_path(dataset_name)

        nx_adj = graph_utils.load_networkx_format(net_file)
        g = nx.from_scipy_sparse_matrix(nx_adj)
        train, test, val, train_neg, test_neg, val_neg = graph_utils.train_test_split(nx_adj, pos_neg_ratio=0.5)
        logger.info("[%s] train 1#%d 0#%d" % (dataset_name, len(train), len(train_neg)))
        logger.info("[%s] test 1#%d 0#%d" % (dataset_name, len(test), len(test_neg)))
        logger.info("[%s] valid 1#%d 0#%d" % (dataset_name, len(val), len(val_neg)))
        # Compute basic link prediction indexes from g_train
        aa_matrix = np.zeros(nx_adj.shape)
        g_train = nx.from_edgelist(train)
        train_nodes = g_train.nodes
        candidate_edges = []
        for u, v in test:
            if u in train_nodes and v in train_nodes:
                candidate_edges.append((u, v))
        for u, v in test_neg:
            if u in train_nodes and v in train_nodes:
                candidate_edges.append((u, v))

        # Run Algos

        lp_baselines = {"Adamic-Adar": nx.adamic_adar_index,
                        "Resouce Allocation": nx.resource_allocation_index,
                        "Jaccard": nx.jaccard_coefficient,
                        "Preferential Attachment": nx.preferential_attachment}
        print("#============================")
        print("Method\tAUC\tAP")
        for baseline in lp_baselines:
            for u, v, p in lp_baselines[baseline](g_train, candidate_edges):
                aa_matrix[u][v] = p
                aa_matrix[v][u] = p  # make sure it's symmetric
            # Calculate ROC AUC and Average Precision
            roc, ap = get_roc_score(test, test_neg, aa_matrix)
            print("%s\t%.6f\t%.6f" % (baseline, roc, ap))
            logger.info("[%s]\t%s\t%.6f\t%.6f" % (dataset_name, baseline, roc, ap))
        print("#============================")


if __name__ == '__main__':
    main()