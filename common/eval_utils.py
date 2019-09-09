import numpy
from scipy import stats

def get_AP(k, ideal, test):
    """
        compute AP
    """
    ideal = set(ideal)
    accumulation = 0.0
    count = 0
    for i in range(len(test)):
        if i >= k:
            break
        if test[i] in ideal:
            count += 1
            accumulation += count / (i + 1.0)
    m = len(ideal)
    n = k
    x = 0
    if m > n:
        x = n
    else:
        x = m
    if x == 0:
        return 0
        # if accumulation/x < 1.0:
    #    print test, ideal, k
    return accumulation / x


def get_MAP(k, ideal_map, test_map):
    """
        compute MAP
    """
    accumulation = 0.0
    for key in ideal_map.keys():
        accumulation += get_AP(k, ideal_map[key], test_map[key])
    if len(ideal_map) == 0:
        return 0
    return accumulation / len(ideal_map)


def get_nDCG(k, ideal, test):
    """
        compute NDCG
    """
    ideal = set(ideal)
    accumulation = 0.0
    for i in range(len(test)):
        if i >= k:
            break
        if test[i] in ideal:
            if i == 0:
                accumulation += 1.0
            else:
                accumulation += 1.0 / numpy.log2(i + 1)
    normalization = 0.0
    for i in range(len(ideal)):
        if i >= k:
            break
        if i == 0:
            normalization += 1.0
        else:
            normalization += 1.0 / numpy.log2(i + 1)
    if normalization == 0:
        return 0
    return accumulation / normalization


def get_MnDCG(k, ideal_map, test_map):
    """
        compute mean NDCG
    """
    accumulation = 0.0
    for key in ideal_map.keys():
        accumulation += get_nDCG(k, ideal_map[key], test_map[key])
    if len(ideal_map) == 0:
        return 0
    return accumulation / len(ideal_map)

def kendallTau(map1, map2):
    x = []
    y = []
    for k in map1:
        x.append(map1[k])
        y.append(map2[k])
    tau, p_value = stats.kendalltau(x, y)
    return tau, p_value