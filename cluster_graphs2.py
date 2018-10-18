import matplotlib
matplotlib.use('agg')
from os import cpu_count
from numba import jit, njit
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Parallel, delayed
import numpy as np

prot = 'cox1'
path_to_graphs = '../res/' + prot + '_Aledo_igraph_enc_merged.random_graphs'
out_path = '../res/random_graphs_identity_enc_Aledo_' + prot
reverse_shuffle = True
thread_num = cpu_count()


@jit(nopython=True, nogil=True, cache=True)
def check_last_cluster(g_list, index_list, centroid, threshold, res):
    l = centroid.size
    for i in index_list:
        j1 = 0
        j2 = 0
        c = 0
        while j1 < l and j2 < l:
            if g_list[i, j1] < centroid[j2]:
                j1 += 1
            elif g_list[i, j1] == centroid[j2]:
                c += 1
                j1 += 1
                j2 += 1
            else:
                j2 += 1
        if c / l > threshold:
            res[i] = 1
        else:
            res[i] = 0
        # if np.intersect1d(g_list[i,:], centroid, assume_unique=True).size / l > threshold:
        #     res[i] = 1
        # else:
        #     res[i] = 0
    return 0


def clustering(graphs_list, threshold):
    centroids = [graphs_list[0]]
    graphs_array = np.empty((len(graphs_list), graphs_list[0].size), dtype=int)
    for i in range(graphs_array.shape[0]):
        for j in range(graphs_array.shape[1]):
            graphs_array[i, j] = graphs_list[i][j]
    remaining = np.array(range(1, len(graphs_list)), dtype=int)
    cluster_info = np.empty(len(graphs_list), dtype=int)
    cluster_info[0] = 1
    while remaining.size > 0:
        nums = []
        for i in range(thread_num):
            nums.append(remaining.size//thread_num)
        for i in range(remaining.size%thread_num):
            nums[i] += 1
        sum_num = [0]
        sum_num.extend(np.cumsum(nums))
        tasks = Parallel(n_jobs=1, backend='threading')(
            delayed(check_last_cluster)(graphs_array, remaining[sum_num[j]: sum_num[j + 1]], centroids[-1], threshold, cluster_info)
            for j in range(thread_num))
        remaining_new = list(filter(lambda x: cluster_info[x] == 0, remaining))
        if len(remaining_new) == 0:
            break
        last = remaining_new.pop()
        centroids.append(graphs_list[last])
        cluster_info[last] = 1
        remaining = np.array(remaining_new, dtype=int)
    return len(centroids)/len(graphs_list)


if __name__ == '__main__':
    graphs = None
    with open(path_to_graphs, 'r') as f:
        for line in f.readlines():
            random_graphs = []
            s = line.strip().split('\t')
            for g in s[1].split(';'):
                nodes = [int(p) for p in g.split(',')]
                nodes.sort()
                random_graphs.append(np.array(nodes, dtype=int))
            shuffled_indices = [int(i) for i in s[3].split(';')]
            if reverse_shuffle:
                shuffled_indices.reverse()
            if graphs is None:
                graphs = [[] for x in range(len(shuffled_indices))]
            for j in range(len(random_graphs)):
                graphs[shuffled_indices[j]].append(random_graphs[j])
    print('reading done')
    for i in range(3, len(graphs)):
        comp_graphs = graphs[i]
        effective_numbers = []
        thresholds = [j/100 for j in range(95, 30, -5)]
        for j in thresholds:
            effective_numbers.append(clustering(comp_graphs, j))
        plt.title('Effective number of random graphs')
        plt.xlabel('Identity to centroid')
        plt.ylabel('Proportion of graphs')
        plt.plot(thresholds, effective_numbers)
        # plt.axis([0, 0.002, 0, 6000])
        plt.savefig(out_path + '_' + str(i) + '.png')
        plt.clf()
