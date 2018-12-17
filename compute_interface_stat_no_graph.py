import os
from bisect import bisect_left
from random import randint, choice

import numpy as np
import math

from numpy.random.mtrand import shuffle
from sklearn.externals.joblib import Parallel, delayed

from assimptotic_tests import parse_pdb, read_cox_data, get_interface

path_to_pdb = './pdb/1occ.pdb1'
path_to_cox_data = './Coloring/COXdata.txt'
path_to_colors = './Coloring/internal_gaps.2/'

chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}
dist_threshold = 8

use_colors = False
only_selected_chains = True
only_mitochondria_to_nuclear = False
thread_num = 44
permutations_num = 10000


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
                        s = line.strip().split('\t')
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


def parse_site2pdb(chain_to_prot):
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


def cluster_centers(cluster_ids, cl_num, pos_to_coords):
    centers = np.empty(cl_num + 1, dtype=object)
    for i in range(1, cl_num + 1):
        centers[i] = np.zeros(3, dtype=float)
    cl_pop = np.zeros(cl_num + 1, dtype=int)
    for i in range(len(cluster_ids)):
        cl_id = cluster_ids[i]
        if cl_id > 0:
            cent = centers[cl_id]
            coord = pos_to_coords[i]
            cent[0] += coord[0]
            cent[1] += coord[1]
            cent[2] += coord[2]
            cl_pop[cl_id] += 1
    for cl_id in range(1, cl_num + 1):
        n = cl_pop[cl_id]
        cent = centers[cl_id]
        cent[0] /= n
        cent[1] /= n
        cent[2] /= n
    return centers


def gen_random_coloring(cluster_id, cl_num, pos_to_coords):

    shuffled = np.zeros(len(cluster_id), dtype=int)
    sum_dist_in_cluster, centers = get_cluster_stat(cl_num, pos_to_coords, cluster_id)
    non_zeroes = []
    for i in range(len(cluster_id)):
        if cluster_id[i] > 0:
            non_zeroes.append(cluster_id[i])
    shuffle(non_zeroes)
    j = 0
    for i in range(len(cluster_id)):
        if cluster_id[i] > 0:
            shuffled[i] = non_zeroes[j]
            j += 1
    sum_dist_in_cluster_random, centers = get_cluster_stat(cl_num, pos_to_coords, shuffled)
    cluster_to_vertices = {}
    cluster_to_rest_vertices = {}
    for i in range(1, cl_num + 1):
        cluster_to_vertices[i] = []
        cluster_to_rest_vertices[i] = []
    for i in range(len(shuffled)):
        if shuffled[i] > 0:
            cluster_to_vertices[shuffled[i]].append(i)
    for cl_id in range(1, cl_num + 1):
        for i in range(len(shuffled)):
            if shuffled[i] > 0 and shuffled[i] != cl_id:
                cluster_to_rest_vertices[cl_id].append(i)
    bad_clusters = []
    for i in range(cl_num + 1):
        if sum_dist_in_cluster_random[i] > sum_dist_in_cluster[i]:
            bad_clusters.append(i)
    while len(bad_clusters) > 0:
        repeat = True
        v = 0
        u = 0
        cl_id1 = 0
        cl_id2 = 0
        while repeat:
            cl_id1 = choice(bad_clusters)
            vertices1 = cluster_to_rest_vertices[cl_id1]
            v = choice(vertices1)
            cl_id2 = shuffled[v]
            vertices2 = cluster_to_vertices[cl_id1]
            u = choice(vertices2)
            dist_added_to_1 = dist(pos_to_coords[v], centers[cl_id1]) - dist(pos_to_coords[u], centers[cl_id1])
            if dist_added_to_1 < 0:
                dist_added_to_2 = dist(pos_to_coords[u], centers[cl_id2]) - dist(pos_to_coords[v], centers[cl_id2])
                if sum_dist_in_cluster_random[cl_id2] + dist_added_to_2 <= sum_dist_in_cluster[cl_id2]:
                    repeat = False
                elif dist_added_to_1 + dist_added_to_2 < 0:
                    repeat = False
        exchange_vertices(v, u, cl_id1, cl_id2, shuffled, cluster_to_vertices, cluster_to_rest_vertices)
        sum_dist_in_cluster_random, centers = get_cluster_stat(cl_num, pos_to_coords, shuffled)
        bad_clusters = []
        for i in range(cl_num + 1):
            if sum_dist_in_cluster_random[i] > sum_dist_in_cluster[i]:
                bad_clusters.append(i)
    return shuffled


def exchange_vertices(v, u, cl_id1, cl_id2, shuffled, cluster_to_vertices, cluster_to_rest_vertices):
    shuffled[v] = cl_id1
    shuffled[u] = cl_id2
    cluster1 = cluster_to_vertices[cl_id1]
    i = cluster1.index(u)
    cluster1[i] = v
    cluster2 = cluster_to_vertices[cl_id2]
    i = cluster2.index(v)
    cluster2[i] = u
    neighbors1 = cluster_to_rest_vertices[cl_id1]
    neighbors2 = cluster_to_rest_vertices[cl_id2]
    i = neighbors1.index(v)
    neighbors1[i] = u
    i = neighbors2.index(u)
    neighbors2[i] = v


def get_cluster_stat(cl_num, pos_to_coords, cluster_id):
    sum_dist_in_cluster = np.zeros(cl_num + 1, dtype=int)
    centers = cluster_centers(cluster_id, cl_num, pos_to_coords)
    for i in range(len(cluster_id)):
        cl_id = cluster_id[i]
        if cl_id > 0:
            sum_dist_in_cluster[cl_id] += dist(centers[cl_id, ], pos_to_coords[i])
    return sum_dist_in_cluster, centers


def compute_stat_on_random_colorings(pos_to_coords, cluster_id, n, interface_set, filtered_set):
    res = []
    for i in range(n):
        cl_num = 0
        for id in cluster_id:
            if id > cl_num:
                cl_num = id
        random_coloring = gen_random_coloring(cluster_id, cl_num, pos_to_coords)
        res.append(chi_sqr(random_coloring, interface_set, filtered_set))
    return res


def chi_sqr(cluster_ids, interface_set, filter_set):
    res = 0
    cl_to_poses = {}
    for pos in filter_set:
        cl = cluster_ids[pos]
        l = cl_to_poses.get(cl)
        if l is None:
            l = []
            cl_to_poses[cl] = l
        l.append(pos)
    p_exp = len(interface_set)/len(filter_set)
    for pos_list in cl_to_poses.values():
        c = 0
        for pos in pos_list:
            if pos in interface_set:
                c += 1
        p_obs = c/len(pos_list)
        res += (p_obs - p_exp)*(p_obs - p_exp)/p_exp
    return len(filter_set)*res


def test_independence(pos_to_coords, cluster_ids, interface, filter_set):
    print('computing p_value')
    filtered_set = set()
    for i in range(len(cluster_ids)):
        if cluster_ids[i] > 0 and i in filter_set:
            filtered_set.add(i)

    stat = chi_sqr(cluster_ids, interface, filtered_set)
    iter_nums = []
    for i in range(thread_num):
        iter_nums.append(permutations_num//thread_num)
    for i in range(permutations_num%thread_num):
        iter_nums[i] += 1
    tasks = Parallel(n_jobs=thread_num)(delayed(compute_stat_on_random_colorings)(pos_to_coords, cluster_ids, n,
                                                                                  interface, filtered_set) for n in iter_nums)
    i = 0
    for task in tasks:
        for s in task:
            if s >= stat:
                i += 1
    return i/permutations_num


def count(cluster_ids, interface, filter_set=None):
    cl_num = 0
    for id in cluster_ids:
        if id > cl_num:
            cl_num = id
    non_int_counts = np.zeros(cl_num, dtype=int)
    int_counts = np.zeros(cl_num, dtype=int)
    for pos in range(len(cluster_ids)):
        cl = cluster_ids[pos]
        if cl != 0:
            if pos in interface:
                int_counts[cl - 1] += 1
            else:
                if filter_set == None or pos in filter_set:
                    non_int_counts[cl - 1] += 1
    return non_int_counts, int_counts


def print_unified_intefaces():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot), chain_to_prot, path_to_colors)
    prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data()
        # prot_to_buried, prot_to_non_buried = read_dssp_data()
    interfaces = {}
    for prot_name1, chain1 in prot_to_chain.items():
        coords1 = chain_to_site_coords[chain1]
        non_burried1 = prot_to_non_buried[prot_name1]
        for chain2, coords2 in chain_to_site_coords.items():
            prot_name2 = chain_to_prot.get(chain2)
            if prot_name2 is not None:
                non_burried2 = prot_to_non_buried[prot_name2]
                if only_mitochondria_to_nuclear:
                    continue
                if prot_name1 < prot_name2:
                    int1, int2 = get_interface(coords1, coords2, non_burried1, non_burried2)
                    l = interfaces.get(prot_name1)
                    if l is not None:
                        l.update(int1)
                    else:
                        interfaces[prot_name1] = int1
                    l = interfaces.get(prot_name2)
                    if l is not None:
                        l.update(int2)
                    else:
                        interfaces[prot_name2] = int2
            else:
                int1, int2 = get_interface(coords1, coords2, non_burried1)
                l = interfaces.get(prot_name1)
                if l is not None:
                    l.update(int1)
                else:
                    interfaces[prot_name1] = int1
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        print(prot_name + '.' + method_name)
        int = interfaces[prot_name]
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        filter_set = prot_to_non_buried[prot_name]
        for i in range(len(cluster_ids)):
            if i not in filter_set:
                cluster_ids[i] = 0
        cl_counts, int_counts = count(cluster_ids, int, filter_set)
        p_value = test_independence(coords, cluster_ids, int, filter_set)
        if method_name != '':
            print_table(cl_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)
        else:
            print_table(cl_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)


def print_unified_intefaces_enc():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot), chain_to_prot, path_to_colors)
    prot_to_burried, prot_to_non_buried, prot_to_non_interface = read_cox_data()
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        print(prot_name + ' ' + method_name)
        non_buried = prot_to_non_buried[prot_name]
        non_interface = prot_to_non_interface[prot_name]
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        for i in range(len(cluster_ids)):
            if i not in non_buried:
                cluster_ids[i] = 0
        cl_counts, int_counts = count(cluster_ids, non_interface, non_buried)
        p_value = test_independence(coords, cluster_ids, non_interface, non_buried)
        if method_name != '':
            print_table(cl_counts, int_counts, 'ENC_noninterf', 'CONT + ENC_interface', p_value)
        else:
            print_table(cl_counts, int_counts, 'ENC_noninterf', 'CONT + ENC_interface', p_value)


def print_table(cl_counts, int_counts, label1, label2, p_value):
    count_list = []
    group_num = []
    for c in cl_counts:
        group_num.append(str(len(group_num) + 1))
        count_list.append(str(c))
    print('номер группы\t' + '\t'.join(group_num))
    print(label1 + '\t' + '\t'.join(count_list))
    count_list = []
    for c in int_counts:
        count_list.append(str(c))
    print(label2 + '\t' + '\t'.join(count_list))
    print('p_value = %1.4f\n' % p_value)


def main():
    # print_unified_intefaces()
    print_unified_intefaces_enc()


if __name__ == '__main__':
    main()