import matplotlib
matplotlib.use('agg')
from os import cpu_count
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Parallel, delayed

prot = 'cox1'
path_to_graphs = '../res/' + prot + '_Aledo_igraph_enc_merged.random_graphs'
out_path = '../res/random_graphs_identity_' + prot
comp_num = 4

thread_num = cpu_count()


def check_last_cluster(g_list, centroid, threshold):
    in_cluster = []
    out_cluster = []
    l = len(centroid)
    for g in g_list:
        if len(g.intersection(centroid)) / l > threshold:
            in_cluster.append(g)
        else:
            out_cluster.append(g)
    return in_cluster, out_cluster


def clustering(graphs_list, threshold):
    clusters = [[graphs_list[0]]]
    centroids = [graphs_list[0]]
    remaining = graphs_list[1:]
    while len(remaining) >= 0:
        print(len(remaining))
        nums = []
        for i in range(thread_num):
            nums.append(len(remaining)//thread_num)
        for i in range(len(remaining)%thread_num):
            nums[i] += 1
        sum_num = [0]
        for i in range(0, thread_num):
            sum_num.append(sum_num[-1] + nums[i])
        tasks = Parallel(n_jobs=1, batch_size=1)(
            delayed(check_last_cluster)(remaining[sum_num[j]: sum_num[j + 1]], centroids[-1], threshold)
            for j in range(thread_num))
        remaining = []
        for in_cluster, out_cluster in tasks:
            clusters[-1].extend(in_cluster)
            remaining.extend(out_cluster)
        centroids.append(remaining.pop())
    return len(clusters)/len(graphs_list)


if __name__ == '__main__':
    graphs = [[] for i in range(comp_num)]
    with open(path_to_graphs, 'r') as f:
        for line in f.readlines():
            sampled_graphs = []
            random_graphs = []
            s = line.strip().split('\t')
            for g in s[1].split(';'):
                nodes = set(int(p) for p in g.split(','))
                random_graphs.append(nodes)
            shuffled_indices = [int(i) for i in s[3].split(';')]
            for j in range(len(random_graphs)):
                graphs[shuffled_indices[j]].append(random_graphs[j])
    for i in range(comp_num):
        comp_graphs = graphs[i]
        effective_numbers = []
        thresholds = [j/100 for j in range(95, 30, -5)]
        for j in thresholds:
            effective_numbers.append(clustering(comp_graphs, j))
        plt.title('Histogram of effective number of random graphs')
        plt.xlabel('identity to centroid')
        plt.ylabel('Percent of graphs')
        plt.plot(thresholds, effective_numbers)
        # plt.axis([0, 0.002, 0, 6000])
        plt.savefig(out_path + '_' + str(i) + '.png')
        plt.clf()
