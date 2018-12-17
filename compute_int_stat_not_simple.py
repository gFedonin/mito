import os
from bisect import bisect_left
from random import choice

import numpy as np
import math

from numpy.random.mtrand import shuffle
from sklearn.externals.joblib import Parallel, delayed

from compute_interface_stat_not_so_simple import exchange_vertices, get_cluster_neighbors, get_neighbors, parse_colors, \
    parse_out, parse_site2pdb, print_table, count, chi_sqr
from assimptotic_tests import parse_pdb, read_cox_data, get_interface

path_to_pdb = './pdb/1occ.pdb1'
path_to_cox_data = './Coloring/COXdata.txt'
path_to_colors = './Coloring/internal_gaps.2/'

chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}
dist_threshold = 8

use_colors = False
only_selected_chains = False
only_mitochondria_to_nuclear = True
thread_num = 44
permutations_num = 10000


def gen_random_coloring(neighbors, cluster_id, cl_num):

    shuffled = np.zeros(len(cluster_id), dtype=int)
    cl_num += 1

    edge_numbers, cluster_to_neighbors = get_cluster_neighbors(cl_num, neighbors, cluster_id)
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
    edge_numbers_random, cluster_to_neighbors = get_cluster_neighbors(cl_num, neighbors, shuffled)
    cluster_to_vertices = {}
    for i in range(1, cl_num):
        cluster_to_vertices[i] = []
    for i in range(len(shuffled)):
        if shuffled[i] > 0:
            cluster_to_vertices[shuffled[i]].append(i)
    bad_clusters = []
    for i in range(cl_num):
        if edge_numbers_random[i] != edge_numbers[i]:
            bad_clusters.append(i)
    while len(bad_clusters) > 0:
        repeat = True
        v = 0
        u = 0
        edges_added_to_1 = 0
        edges_added_to_2 = 0
        cl_id1 = 0
        cl_id2 = 0
        while repeat:
            edges_added_to_1 = 0
            edges_added_to_2 = 0
            cl_id1 = choice(bad_clusters)
            vertices1 = cluster_to_neighbors[cl_id1]
            v = choice(vertices1)
            cl_id2 = shuffled[v]
            vertices2 = cluster_to_vertices[cl_id1]
            u = choice(vertices2)
            for n in neighbors[v]:
                if shuffled[n] == cl_id1:
                    edges_added_to_1 += 1
                elif shuffled[n] == cl_id2:
                    edges_added_to_2 -= 1
            for n in neighbors[u]:
                if shuffled[n] == cl_id1:
                    edges_added_to_1 -= 1
                elif shuffled[n] == cl_id2:
                    edges_added_to_2 += 1

            diff1 = edges_added_to_1*(2*edge_numbers_random[cl_id1] - 2*edge_numbers[cl_id1] + edges_added_to_1)
            diff2 = edges_added_to_2*(2*edge_numbers_random[cl_id2] - 2*edge_numbers[cl_id2] + edges_added_to_2)

            if diff1 + diff2 <= 0:
                repeat = False
        edge_numbers_random[cl_id1] += edges_added_to_1
        edge_numbers_random[cl_id2] += edges_added_to_2
        exchange_vertices(v, u, cl_id1, cl_id2, shuffled, cluster_to_vertices, cluster_to_neighbors, neighbors)
        bad_clusters = []
        for i in range(cl_num):
            if edge_numbers_random[i] != edge_numbers[i]:
                bad_clusters.append(i)
    return shuffled


def compute_stat_on_random_colorings(neighbors, cluster_id, n, interface_set, filtered_set):
    res = []
    for i in range(n):
        cl_num = 0
        for id in cluster_id:
            if id > cl_num:
                cl_num = id
        random_coloring = gen_random_coloring(neighbors, cluster_id, cl_num)
        res.append(chi_sqr(random_coloring, interface_set, filtered_set))
    return res


def test_independence(pos_to_coords, cluster_ids, interface, filter_set):
    print('computing p_value')
    filtered_set = set()
    for i in range(len(cluster_ids)):
        if cluster_ids[i] > 0 and i in filter_set:
            filtered_set.add(i)
    neighbors = get_neighbors(pos_to_coords, filtered_set)

    stat = chi_sqr(cluster_ids, interface, filtered_set)
    iter_nums = []
    for i in range(thread_num):
        iter_nums.append(permutations_num//thread_num)
    for i in range(permutations_num%thread_num):
        iter_nums[i] += 1
    tasks = Parallel(n_jobs=thread_num)(delayed(compute_stat_on_random_colorings)(neighbors, cluster_ids, n,
                                                                                  interface, filtered_set) for n in iter_nums)
    i = 0
    for task in tasks:
        for s in task:
            if s >= stat:
                i += 1
    return i/permutations_num


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


def main():
    print_unified_intefaces()
    # print_unified_intefaces_enc()


if __name__ == '__main__':
    main()