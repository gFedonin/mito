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


def get_cluster_id(g, centroids, threshold):
    res = -1
    for i in range(len(centroids)):
        c = centroids[i]
        if len(g.intersection(c)) / len(g) > threshold:
            res = i
            break
    return res


def cluster_graphs(g_list, clusters, centroids, threshold):
    cl_list = [[g_list[0]]]
    cent_list = [g_list[0]]
    for g in g_list[1:]:
        clustered = False
        for i in range(len(cent_list)):
            c = cent_list[i]
            if len(g.intersection(c))/len(g) > threshold:
                cl_list[i].append(g)
                clustered = True
                break
        if not clustered:
            cent_list.append(g)
            cl_list.append([g])
    clusters.extend(cl_list)
    centroids.extend(cent_list)


def clustering(graphs_list, threshold):
    clusters = []
    centroids = []
    for i in range(len(graphs_list)//thread_num):
        g_list = []
        tasks = Parallel(n_jobs=thread_num)(delayed(get_cluster_id)(graphs_list[i*thread_num + j], centroids, threshold)
                                            for j in range(thread_num))
        for j in range(thread_num):
            cl_id = tasks[j]
            if cl_id == -1:
                g_list.append(graphs_list[i*thread_num + j])
            else:
                clusters[cl_id].append(graphs_list[i*thread_num + j])
        if len(g_list) > 0:
            cluster_graphs(g_list, clusters, centroids, threshold)
    g_list = []
    rest = len(graphs_list) % thread_num
    tasks = Parallel(n_jobs=thread_num)(delayed(get_cluster_id)(graphs_list[len(graphs_list) - rest + j], centroids, threshold)
                                        for j in range(rest))
    for j in range(rest):
        cl_id = tasks[j]
        if cl_id == -1:
            g_list.append(graphs_list[len(graphs_list) - rest + j])
        else:
            clusters[cl_id].append(graphs_list[len(graphs_list) - rest + j])
    if len(g_list) > 0:
        cluster_graphs(g_list, clusters, centroids, threshold)
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
