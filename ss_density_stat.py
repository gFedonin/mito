import os
from os.path import exists
import matplotlib.pyplot as plt

import numpy as np
import math

from numpy.random.mtrand import shuffle
from sklearn.externals.joblib import Parallel, delayed

from assimptotic_tests import parse_dssp

# path_to_pdb = '../pdb/5ara.pdb1'
path_to_pdb = '../pdb/1occ.pdb1'
# path_to_pdb = '../pdb/1bgy.pdb1'
path_to_colors = '../Coloring/internal_gaps.2/'

# chain_to_prot = {'W': 'atp6'}
chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
# chain_to_prot = {'C': 'cytb'}
dist_threshold = 8
permutations_num = 10000

print_hist = False
out_path = './res/density_hist/'

use_colors = False


def parse_pdb(chain_to_prot, path_to_pdb):
    prot_to_site_coords = {}
    with open(path_to_pdb, 'r') as f:
        curr_chain = ''
        pos_to_coords = {}
        for line in f.readlines():
            s = line.split()
            if s[0] == 'ATOM' and s[2] == 'CA':
                chain = s[4]
                if chain in chain_to_prot.keys():
                    if curr_chain != '':
                        if chain != curr_chain:
                            prot_to_site_coords[chain_to_prot[curr_chain]] = pos_to_coords
                            curr_chain = chain
                            pos_to_coords = {}
                    else:
                        curr_chain = chain
                    pos_to_coords[int(s[5])] = (float(s[6]), float(s[7]), float(s[8]))
        prot_to_site_coords[chain_to_prot[curr_chain]] = pos_to_coords
    return prot_to_site_coords


def parse_colors(chain_to_prot, path_to_colors):
    prot_to_clusters = []
    for (dirpath, dirnames, filenames) in os.walk(path_to_colors):
        for filename in filenames:
            i = filename.find('.colors')
            if i != -1:
                prot_name = filename[:filename.index('.')]
                if prot_name not in chain_to_prot.values():
                    continue
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
                prot_to_clusters.append((prot_name, method_name, cluster_ids))
    return prot_to_clusters


def parse_site2pdb(chain_to_prot, path_to_colors):
    prot_to_site_map = {}
    for (dirpath, dirnames, filenames) in os.walk(path_to_colors):
        for filename in filenames:
            i = filename.find('.site2pdb')
            if i != -1:
                prot_name = filename[:filename.index('.')]
                if prot_name not in chain_to_prot.values():
                    continue
                site2pdb = {}
                prot_to_site_map[prot_name] = site2pdb
                with open(path_to_colors + filename, 'r') as f:
                    f.readline()
                    for line in f.readlines():
                        s = line.strip().split('\t')
                        site2pdb[s[0]] = int(s[1])
    return prot_to_site_map


def parse_out(prot_to_site_map, chain_to_prot, path_to_colors):
    prot_to_clusters = []
    for (dirpath, dirnames, filenames) in os.walk(path_to_colors):
        for filename in filenames:
            i = filename.find('.out')
            if i == -1:
                i = filename.find('.partition')
            if i != -1:
                prot_name = filename[:filename.index('.')]
                if prot_name not in chain_to_prot.values():
                    continue
                if i != len(prot_name):
                    method_name = filename[len(prot_name) + 1:i]
                else:
                    method_name = ''
                pos_to_cluster_id = []
                site2pdb = prot_to_site_map[prot_name]
                with open(path_to_colors + filename, 'r') as f:
                    max_pos = 0
                    for line in f.readlines()[5:]:
                        s = line.strip().split()
                        pos = site2pdb[s[0]]
                        if pos == 0:
                            continue
                        if pos > max_pos:
                            max_pos = pos
                        pos_to_cluster_id.append((pos, int(s[1])))
                cluster_ids = np.zeros(max_pos + 1, dtype=int)
                for pos, c in pos_to_cluster_id:
                    cluster_ids[pos] = c
                prot_to_clusters.append((prot_name, method_name, cluster_ids))
    return prot_to_clusters


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


def cluster_stat(neighbors, cluster_id, permute):
    pos_to_cluster_id = cluster_id
    if permute:
        # print(cluster_id)
        non_zeroes = []
        for i in range(len(cluster_id)):
            if cluster_id[i] > 0:
                non_zeroes.append(cluster_id[i])
        shuffle(non_zeroes)
        shuffled = np.zeros(len(cluster_id), dtype=int)
        j = 0
        for i in range(len(cluster_id)):
            if cluster_id[i] > 0:
                shuffled[i] = non_zeroes[j]
                j += 1
        pos_to_cluster_id = shuffled
        # print(pos_to_cluster_id)
    cluster_id_to_stat = {}
    cl_num = 0
    pos_num = 0
    for id in pos_to_cluster_id:
        if id > cl_num:
            cl_num = id
        if id > 0:
            pos_num += 0
    cl_num += 1
    av_in_cluster = np.zeros(cl_num, dtype=float)
    av_normalized_edge_num = np.zeros(cl_num, dtype=float)
    std_normalized_edge_num = np.zeros(cl_num, dtype=float)
    cluster_population = np.zeros(cl_num, dtype=int)
    cluster_densities = np.zeros(cl_num, dtype=float)
    for pos, n_list in neighbors.items():
        if pos < len(pos_to_cluster_id):
            internal_edge_count = 0
            cl_id = pos_to_cluster_id[pos]
            total_neighbor_count = 0
            for p in n_list:
                if p < len(pos_to_cluster_id):
                    if pos_to_cluster_id[p] == cl_id:
                        internal_edge_count += 1
                    if pos_to_cluster_id[p] > 0:
                        total_neighbor_count += 1
            if total_neighbor_count > 0:
                av_in_cluster[cl_id] += internal_edge_count/total_neighbor_count
                cluster_population[cl_id] += 1
                cluster_densities[cl_id] += internal_edge_count/2
            av_normalized_edge_num[cl_id] += internal_edge_count/(pos_num - 1)
            std_normalized_edge_num[cl_id] += internal_edge_count/(pos_num - 1)**2
    av_in_cluster_total = 0
    cluster_population_total = 0
    for cl_id in range(1, cl_num):
        av_in_cluster_total += av_in_cluster[cl_id]
        cluster_population_total += cluster_population[cl_id]
        n = cluster_population[cl_id]
        if n > 0:
            cluster_id_to_stat[cl_id] = av_in_cluster[cl_id]/n
            av_normalized_edge_num[cl_id] /= n
            std_normalized_edge_num[cl_id] = np.sqrt(std_normalized_edge_num[cl_id]/n - av_normalized_edge_num[cl_id]**2)
            cluster_densities[cl_id] /= n*(n - 1)/2
        else:
            cluster_id_to_stat[cl_id] = 0
    cluster_id_to_stat['total'] = av_in_cluster_total/cluster_population_total

    return cluster_id_to_stat, cluster_population, cluster_densities


def cluster_stat1(neighbors, pos_to_ss_array, ss_types, permute):
    pos_to_ss = pos_to_ss_array
    if permute:
        # print(cluster_id)
        non_zeroes = []
        for i in range(len(pos_to_ss_array)):
            if pos_to_ss_array[i] is not None:
                non_zeroes.append(pos_to_ss_array[i])
        shuffle(non_zeroes)
        shuffled = np.empty(len(pos_to_ss_array), dtype=object)
        j = 0
        for i in range(len(pos_to_ss_array)):
            if pos_to_ss_array[i] is not None:
                shuffled[i] = non_zeroes[j]
                j += 1
        pos_to_ss = shuffled
        # print(pos_to_cluster_id)
    cluster_id_to_stat = {}
    internal_edges_count = {}
    external_edges_count = {}
    cluster_population = {}
    cluster_densities = {}
    for ss in ss_types:
        internal_edges_count[ss] = 0
        external_edges_count[ss] = 0
        cluster_population[ss] = 0
        cluster_densities[ss] = 0
    for pos, n_list in neighbors.items():
        if pos < len(pos_to_ss):
            internal_edge_count = 0
            ss = pos_to_ss[pos]
            external_edge_count = 0
            for p in n_list:
                if p < len(pos_to_ss):
                    if pos_to_ss[p] == ss:
                        internal_edge_count += 1
                    else:
                        if pos_to_ss[p] is not None:
                            external_edge_count += 1
            internal_edge_count /= 2
            internal_edges_count[ss] += internal_edge_count
            external_edges_count[ss] += external_edge_count
            cluster_population[ss] += 1
    total_internal = 0
    total_external = 0
    for ss in ss_types:
        inter = internal_edges_count[ss]
        exter = external_edges_count[ss]
        total_internal += inter
        total_external += exter
        n = cluster_population[ss]
        if inter + exter > 0:
            cluster_id_to_stat[ss] = inter/(inter + exter)
            cluster_densities[ss] = inter/(n*(n - 1)/2)
        else:
            cluster_id_to_stat[ss] = 0
    cluster_id_to_stat['total'] = total_internal/(total_internal + total_external/2)
    return cluster_id_to_stat, cluster_population, cluster_densities


def pvalue(perm_array, stat_val):
    c = 0
    for val in perm_array:
        if val >= stat_val:
            c += 1
    return c / permutations_num


def print_hist(perm_array, name, cl_num):
    plt.title('Histogram of density scores on random permutations')
    plt.xlabel('Density score')
    plt.ylabel('Percent of sites')
    n, bins, patches = plt.hist(perm_array, 50, density=True, facecolor='g', alpha=0.75)
    # plt.axis([0, 0.002, 0, 6000])
    plt.savefig(out_path + name + '_' + str(cl_num) + '.png')
    plt.clf()


def main():
    if print_hist:
        if not exists(out_path):
            os.makedirs(out_path)
    chain_to_site_coords = parse_pdb(chain_to_prot, path_to_pdb)
    chain_to_ss = parse_dssp()
    for chain, pos_to_ss in chain_to_ss.items():
        prot_name = chain_to_prot[chain]
        print(prot_name)
        pos_to_coords = chain_to_site_coords[prot_name]
        neighbors = get_neighbors(pos_to_coords)
        ss_types = set()
        pos_to_ss_array = np.empty(len(pos_to_ss) + 1, dtype=object)
        for pos, ss in pos_to_ss.items():
            pos_to_ss_array[pos] = ss
            ss_types.add(ss)
        cluster_id_to_stat, cluster_population, cluster_densities = cluster_stat1(neighbors, pos_to_ss_array, ss_types, False)
        permutations = Parallel(n_jobs=-1)(delayed(cluster_stat1)(neighbors, pos_to_ss_array, ss_types, True)
                                           for i in range(permutations_num))
        perm_array = {}
        density_array = {}
        for ss in ss_types:
            perm_array[ss] = []
            density_array[ss] = []
        perm_array['total'] = []
        for cl_id_to_stat, cl_population, cl_density in permutations:
            for ss in ss_types:
                perm_array[ss].append(cl_id_to_stat[ss])
                density_array[ss].append(cl_density[ss])
            perm_array['total'].append(cl_id_to_stat['total'])
        print('cluster n_sites stat_value expected_value pvalue density_value expected_density pvalue_density')
        for ss in ss_types:
            stat_val = cluster_id_to_stat[ss]
            av = np.mean(perm_array[ss])
            pval = pvalue(perm_array[ss], stat_val)
            dens = cluster_densities[ss]
            av_dens = np.mean(density_array[ss])
            pval_dens = pvalue(density_array[ss], dens)
            if print_hist:
                print_hist(perm_array[ss], prot_name, ss)
            print("%s %d %1.2f %1.2f %1.4f %1.2f %1.2f %1.4f" % (ss, cluster_population[ss], stat_val, av, pval, dens,
                                                                 av_dens, pval_dens))
        stat_val = cluster_id_to_stat['total']
        av = np.mean(perm_array['total'])
        pval = pvalue(perm_array['total'], stat_val)
        print('total %d %1.2f %1.2f %1.4f' % (sum(cluster_population.values()), stat_val, av, pval))
        print()


if __name__ == '__main__':
    main()