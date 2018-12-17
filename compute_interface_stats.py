import bisect
import os
from bisect import bisect_left
from os import makedirs

from os.path import exists
from random import randint, choice, random

import numpy as np
import math
import networkx as nx

from numpy.random.mtrand import permutation, shuffle
from scipy.stats import chi2_contingency
from sklearn.externals.joblib import Parallel, delayed

from compute_cluster_stats import dist, parse_colors, parse_out, parse_site2pdb
from assimptotic_tests import parse_pdb, get_interface, read_cox_data

path_to_pdb = './COX/1occ.pdb1'
path_to_cox_data = './Coloring/COXdata.txt'
path_to_colors = './Coloring/internal_gaps.2/'

chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}
dist_threshold = 8

use_colors = False
debug = True
only_selected_chains = True
only_mitochondria_to_nuclear = False
thread_num = 8
permutations_num = 100
max_iter_num = 100000


def gen_random_subgraphs(big_graph, small_graph, n):
    target_node_num = small_graph.number_of_nodes()
    target_edge_num = small_graph.number_of_edges()
    nodes = list(nx.node_connected_component(big_graph, list(small_graph)[0]))
    connected_component = nx.induced_subgraph(big_graph, nodes)
    res = [gen_random_subgraph(connected_component, target_node_num, target_edge_num) for i in range(n)]
    return res


def parallel_gen_random_subgraphs(big_graph, small_graph, n):
    subset_sizes = []
    for i in range(thread_num):
        subset_sizes.append(n//thread_num)
    for i in range(n - (n//thread_num)*thread_num):
        subset_sizes[i] += 1
    tasks = Parallel(n_jobs=thread_num)(delayed(gen_random_subgraphs)(big_graph, small_graph, c) for c in subset_sizes)
    res = []
    for task in tasks:
        res.extend(task)
    return res


def chi_sqr(pos_lists, interface_set, total_pos_num):
    res = 0
    p_exp = len(interface_set)/total_pos_num
    for pos_list in pos_lists:
        c = 0
        for pos in pos_list:
            if pos in interface_set:
                c += 1
        p_obs = c/len(pos_list)
        res += (p_obs - p_exp)*(p_obs - p_exp)/p_exp
    return total_pos_num*res


def gen_random_subgraph(connected_graph, target_node_num, target_edge_num):
    target_graph = None
    i = 0
    edge_num = -1
    while edge_num < target_edge_num and i < max_iter_num:
        selected_nodes = set()
        neighbors = set()
        nodes = list(connected_graph.nodes().keys())
        node = choice(nodes)
        selected_nodes.add(node)
        for n in connected_graph[node].keys():
            neighbors.add(n)
        success = True
        for i in range(1, target_node_num):
            if len(neighbors) == 0:
                success =False
                break
            node = choice(list(neighbors))
            neighbors.remove(node)
            selected_nodes.add(node)
            for n in connected_graph[node].keys():
                if n not in selected_nodes:
                    neighbors.add(n)
        if success:
            target_graph = nx.induced_subgraph(connected_graph, selected_nodes)
            edge_num = nx.number_of_edges(target_graph)
        else:
            edge_num = -1
        i += 1
    return target_graph


def compute_stat_on_random_subgraphs(big_graph, small_graphs, n, interface_set):
    res = []
    for i in range(n):
        pos_lists = []
        for small_graph in small_graphs:
            pos_list = []
            for comp in nx.connected_components(small_graph):
                small_connected_comp = nx.induced_subgraph(small_graph, comp)
                target_node_num = small_connected_comp.number_of_nodes()
                target_edge_num = small_connected_comp.number_of_edges()
                nodes = list(nx.node_connected_component(big_graph, next(iter(comp))))
                for pos in pos_list:
                    if pos in nodes:
                        nodes.remove(pos)
                filtered_graph = nx.induced_subgraph(big_graph, nodes)
                connected_comps = nx.connected_components(filtered_graph)
                connected_comps_filtered = []
                for comp in connected_comps:
                    g = nx.induced_subgraph(filtered_graph, comp)
                    if g.number_of_nodes() >= target_node_num and g.number_of_edges() >= target_edge_num:
                        connected_comps_filtered.append(g)
                shuffle(connected_comps_filtered)
                for g in connected_comps_filtered:
                    random_graph = gen_random_subgraph(g, target_node_num, target_edge_num)
                connected_component = nx.induced_subgraph(big_graph, nodes)
                random_graph = gen_random_subgraph(connected_component, target_node_num, target_edge_num)
                pos_list.extend(random_graph.nodes)
            pos_lists.append(pos_list)
        res.append(chi_sqr(pos_lists, interface_set, big_graph.number_of_nodes()))
    return res


def compute_hit_stat_on_random_subgraphs(big_graph, small_graphs, n, interface_set):
    res = [0]*len(small_graphs)
    for i in range(n):
        for j in range(len(small_graphs)):
            small_graph = small_graphs[j]
            largest_comp = None
            max_len = 0
            for comp in nx.connected_components(small_graph):
                if len(comp) > max_len:
                    max_len = len(comp)
                    largest_comp = comp
            small_connected_comp = nx.induced_subgraph(small_graph, largest_comp)
            target_node_num = small_connected_comp.number_of_nodes()
            target_edge_num = small_connected_comp.number_of_edges()
            nodes = list(nx.node_connected_component(big_graph, next(iter(largest_comp))))
            connected_component = nx.induced_subgraph(big_graph, nodes)
            random_graph = gen_random_subgraph(connected_component, target_node_num, target_edge_num)
            if random_graph.nodes.keys() == largest_comp:
                res[j] += 1
    return res


def compute_intersect_stat_on_random_subgraphs(big_graph, small_graphs, n, interface_set):
    res = [0]*len(small_graphs)
    for i in range(n):
        for j in range(len(small_graphs)):
            small_graph = small_graphs[j]
            pos_list = set()
            for comp in nx.connected_components(small_graph):
                small_connected_comp = nx.induced_subgraph(small_graph, comp)
                target_node_num = small_connected_comp.number_of_nodes()
                target_edge_num = small_connected_comp.number_of_edges()
                nodes = list(nx.node_connected_component(big_graph, next(iter(comp))))
                connected_component = nx.induced_subgraph(big_graph, nodes)
                random_graph = gen_random_subgraph(connected_component, target_node_num, target_edge_num)
                pos_list.update(random_graph.nodes)
            res[j] += len(set(small_graph.nodes.keys()).intersection(pos_list))/len(set(small_graph.nodes.keys()).union(pos_list))
    return res


def compute_vertex_coverage_stat_on_random_subgraphs(big_graph, small_graphs, n, interface_set):
    res = {}
    for i in range(n):
        for j in range(len(small_graphs)):
            small_graph = small_graphs[j]
            pos_list = set()
            for comp in nx.connected_components(small_graph):
                small_connected_comp = nx.induced_subgraph(small_graph, comp)
                target_node_num = small_connected_comp.number_of_nodes()
                target_edge_num = small_connected_comp.number_of_edges()
                nodes = list(nx.node_connected_component(big_graph, next(iter(comp))))
                connected_component = nx.induced_subgraph(big_graph, nodes)
                random_graph = gen_random_subgraph(connected_component, target_node_num, target_edge_num)
                pos_list.update(random_graph.nodes)
            for pos in pos_list:
                c = res.get(pos)
                if c is None:
                    res[pos] = 1
                else:
                    res[pos] = c + 1
    return res


def create_graph(pos_to_coords, poses):
    g = nx.Graph()
    g.add_nodes_from(poses)
    for i in range(len(poses)):
        p_i = poses[i]
        for j in range(i + 1, len(poses)):
            p_j = poses[j]
            if dist(pos_to_coords[p_i], pos_to_coords[p_j]) < dist_threshold:
                g.add_edge(p_i, p_j)
    return g


def test_independence(pos_to_coords, cluster_ids, interface, filter_set):
    print('computing p_value')
    filtered_poses = list(filter_set)
    big_graph = create_graph(pos_to_coords, filtered_poses)

    connected_comps = nx.connected_components(big_graph)
    print('big graph:')
    lens = [str(len(comp)) for comp in connected_comps]
    print('connected comp lens: ' + ' '.join(lens))

    cl_to_poses = {}
    for pos in filtered_poses:
        if pos < len(cluster_ids):
            cl = cluster_ids[pos]
            if cl > 0:
                l = cl_to_poses.get(cl)
                if l is None:
                    l = []
                    cl_to_poses[cl] = l
                l.append(pos)

    small_graphs = [nx.induced_subgraph(big_graph, pos_list) for pos_list in cl_to_poses.values()]
    print('%d small graphs:' % len(small_graphs))
    for small_graph in small_graphs:
        print('node num: %d' % len(small_graph))
        connected_comps = nx.connected_components(small_graph)
        lens = [str(len(comp)) for comp in connected_comps]
        print('connected comp lens: ' + ' '.join(lens))

    stat = chi_sqr(cl_to_poses.values(), interface, len(filtered_poses))
    iter_nums = []
    for i in range(thread_num):
        iter_nums.append(permutations_num//thread_num)
    for i in range(permutations_num - (permutations_num//thread_num)*thread_num):
        iter_nums[i] += 1
    if debug:
        tasks = Parallel(n_jobs=thread_num)(delayed(compute_vertex_coverage_stat_on_random_subgraphs)(big_graph, small_graphs,
                                                                                      n, interface) for n in iter_nums)
        random_stats = {}
        for task in tasks:
            for pos, count in task.items():
                c = random_stats.get(pos)
                if c is None:
                    random_stats[pos] = count
                else:
                    random_stats[pos] = count + c
        print('random hit rates:')
        for pos, c in random_stats.items():
            print('%d %f' % (pos, c/permutations_num))
    tasks = Parallel(n_jobs=thread_num)(delayed(compute_stat_on_random_subgraphs)(big_graph, small_graphs,
                                                                                 n, interface) for n in iter_nums)
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
    chain_to_site_coords = parse_pdb(path_to_pdb)
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
        cl_counts, int_counts = count(cluster_ids, int, filter_set)
        p_value = test_independence(coords, cluster_ids, int, filter_set)
        if method_name != '':
            print_table(cl_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)
        else:
            print_table(cl_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)


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
    print_unified_intefaces()


if __name__ == '__main__':
    main()