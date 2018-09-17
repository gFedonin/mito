import os
import numpy as np
import math

from os.path import exists

path_to_pdb = './COX/1occ.pdb1'# './pdb/5ara.pdb1'
path_to_colors = './Coloring/cox2.simap.partitions/'
out_path = './Coloring/graphs/'

chain_to_cox = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}# {'W': 'atp6'}
dist_threshold = 8


def parse_pdb():
    chain_to_site_coords = {}
    with open(path_to_pdb, 'r') as f:
        curr_chain = ''
        pos_to_coords = {}
        for line in f.readlines():
            s = line.split()
            if s[0] == 'ATOM' and s[2] == 'CA':
                chain = s[4]
                if chain in chain_to_cox.keys():
                    if curr_chain != '':
                        if chain != curr_chain:
                            chain_to_site_coords[chain_to_cox[curr_chain]] = pos_to_coords
                            curr_chain = chain
                            pos_to_coords = {}
                    else:
                        curr_chain = chain
                    pos_to_coords[int(s[5])] = (float(s[6]), float(s[7]), float(s[8]))
        chain_to_site_coords[chain_to_cox[curr_chain]] = pos_to_coords
    return chain_to_site_coords


def parse_colors():
    chain_to_clusters = []
    for (dirpath, dirnames, filenames) in os.walk(path_to_colors):
        for filename in filenames:
            i = filename.find('.colors')
            if i != -1:
                prot_name = filename[:filename.index('.')]
                if i != len(prot_name):
                    method_name = filename[len(prot_name) + 1:i]
                else:
                    method_name = ''
                pos_to_cluster_id = []
                with open(path_to_colors + filename, 'r') as f:
                    max_pos = 0
                    for line in f.readlines()[1:]:
                        s = line.split('\t')
                        pos = int(s[1])
                        if pos == 0:
                            continue
                        if pos > max_pos:
                            max_pos = pos
                        pos_to_cluster_id.append((pos, int(s[2])))
                cluster_ids = np.zeros(max_pos + 1, dtype=int)
                for pos, c in pos_to_cluster_id:
                    cluster_ids[pos] = c
                chain_to_clusters.append((prot_name, method_name, cluster_ids))
    return chain_to_clusters


def dist(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)


def get_neighbors(pos_to_coords):
    pos_to_c = []
    for p, c in pos_to_coords.items():
        pos_to_c.append((p, c))
    neighbors = {}
    for p, c in pos_to_c:
        neighbors[p] = []
    for i in range(len(pos_to_c)):
        p_i, c_i = pos_to_c[i]
        for j in range(i + 1, len(pos_to_c)):
            p_j, c_j = pos_to_c[j]
            if dist(c_i, c_j) < dist_threshold:
                neighbors[p_i].append(p_j)
                neighbors[p_j].append(p_i)
    return neighbors


def print_graph():
    if not exists(out_path):
        os.makedirs(out_path)
    chain_to_site_coords = parse_pdb()
    chain_to_clusters = parse_colors()
    for prot_name, method_name, cluster_ids in chain_to_clusters:
    # for prot_name, pos_to_coords in chain_to_site_coords.items():
        print(prot_name + ' ' + method_name)
        pos_to_coords = chain_to_site_coords[prot_name]
        neighbors = get_neighbors(pos_to_coords)
        with open(out_path + prot_name + ' ' + method_name + '.sif', 'w') as f:
            for pos in range(len(cluster_ids)):
                if cluster_ids[pos] > 0:
                    for n in neighbors[pos]:
                        f.write('%s\tc\t%s\n' % (pos, n))


def main():
    if not exists(out_path):
        os.makedirs(out_path)
    chain_to_site_coords = parse_pdb()
    chain_to_clusters = parse_colors()
    for prot_name, method_name, cluster_ids in chain_to_clusters:
    # for prot_name, pos_to_coords in chain_to_site_coords.items():
        print(prot_name + ' ' + method_name)
        pos_to_coords = chain_to_site_coords[prot_name]
        neighbors = get_neighbors(pos_to_coords)
        with open(out_path + prot_name + ' ' + method_name + '.ncount', 'w') as f:
            max_n = 0
            av_n = 0
            av_av_nn = 0
            c = 0
            n_counts = []
            for pos in range(len(cluster_ids)):
                if cluster_ids[pos] > 0:
                    n = len(neighbors[pos])
                    if n > max_n:
                        max_n = n
                    av_n += n
                    c += 1
                    n_counts.append(n)
                    av_nn = 0
                    for npos in neighbors[pos]:
                        av_nn += len(neighbors[npos])
                    av_av_nn += av_nn/n
                    f.write('%s\t%d\t%d\n' % (pos, n, av_nn/n))
            f.write('\nav neighbor count %d\n' % (av_n/c))
            f.write('\nmedian neighbor count %d\n' % (np.median(n_counts)))
            f.write('\nav av neighbor count %d\n' % (av_av_nn / c))
            f.write('\nhistogram:\n')
            f.write('neighbor_num pos_num\n')
            counts = np.zeros(max_n + 1)
            for pos in range(len(cluster_ids)):
                if cluster_ids[pos] > 0:
                    counts[len(neighbors[pos])] += 1
            for i in range(max_n + 1):
                f.write('%d %d\n' % (i, counts[i]))


if __name__ == '__main__':
    main()