from os import makedirs, cpu_count
from os.path import exists
from random import shuffle, random, randrange

import numpy as np
import pandas as pd
from numpy.random.mtrand import choice
import matplotlib

from Graph import Graph

matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.externals.joblib import Parallel, delayed

from compute_cluster_stats import dist, parse_colors, parse_out, parse_site2pdb
from assimptotic_tests import parse_pdb, get_interface, read_cox_data

path_to_pdb = '../pdb/1occ.pdb1'# '../pdb/1be3.pdb1'
path_to_cox_data = '../Coloring/COXdata.txt'
path_to_cytb_data = '../aledo.csv'
path_to_surf_racer_data = '../surf_racer/burried/1bgy.csv'
path_to_dssp_data = '../dssp/1be3.csv'
path_to_colors = '../Coloring/internal_gaps.2/'

chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}# {'C': 'cytb'}
prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}# {'cytb': 'C'}
dist_threshold = 8

use_colors = False
use_cox_data = False
use_dssp = True
debug = False
only_selected_chains = True
only_mitochondria_to_nuclear = False
print_random_graphs = True
random_graph_stat_hist_path = '../res/random_graph_stat_hist_ABC_Aledo_my/'
temp_path = random_graph_stat_hist_path + 'temp/'
if debug:
    thread_num = 1
else:
    thread_num = cpu_count()
if debug:
    permutations_num = 1
else:
    permutations_num = 64
max_iter = 10000


def read_surf_racer_data(path_to_surf_racer_data):
    prot_to_non_buried = {}
    prot_to_buried = {}
    for prot_name in prot_to_chain.keys():
        prot_to_non_buried[prot_name] = set()
        prot_to_buried[prot_name] = set()
    with open(path_to_surf_racer_data, 'r') as f:
        for line in f.readlines():
            s = line.strip().split('\t')
            if s[-1] == 'burried':
                prot_to_buried[chain_to_prot[s[0]]].add(int(s[1]))
            else:
                prot_to_non_buried[chain_to_prot[s[0]]].add(int(s[1]))
    return prot_to_buried, prot_to_non_buried


def read_dssp_data(path_to_dssp_data):
    prot_to_non_buried = {}
    prot_to_buried = {}
    for prot_name in prot_to_chain.keys():
        prot_to_non_buried[prot_name] = set()
        prot_to_buried[prot_name] = set()
    with open(path_to_dssp_data, 'r') as f:
        f.readline()
        for line in f.readlines():
            s = line.strip().split('\t')
            if s[-1] == '1':
                prot_to_buried[chain_to_prot[s[0]]].add(int(s[1]))
            else:
                prot_to_non_buried[chain_to_prot[s[0]]].add(int(s[1]))
    return prot_to_buried, prot_to_non_buried


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


def gen_random_subgraph_new1(connected_graph, target_node_num, target_edge_num):
    target_graph = None
    edge_num = -1
    iterNum = 0
    max_node = connected_graph.size
    while edge_num < target_edge_num and iterNum < max_iter:
        iterNum += 1
        selected_nodes = np.zeros(max_node + 1, dtype=int)
        outgoing_edges = np.zeros(max_node + 1, dtype=int)
        edge_weights = np.zeros(max_node + 1, dtype=float)
        node_weights = np.zeros(max_node + 1, dtype=float)
        node = choice(connected_graph.nodes)
        selected_nodes[node] = 1
        sel_nodes_num = 1
        outgoing_edges_num = 0
        for n in connected_graph.neighbors[node]:
            outgoing_edges[n] = 1
            outgoing_edges_num += 1
        for i in range(1, target_node_num):
            if outgoing_edges_num == 0:
                break
            np.copyto(edge_weights, outgoing_edges)
            edge_weights /= outgoing_edges_num
            n = choice(max_node + 1, p=edge_weights)
            outgoing_edges_num -= outgoing_edges[n]
            outgoing_edges[n] = 0
            selected_nodes[n] = 1
            sel_nodes_num += 1
            for n1 in connected_graph.neighbors[n]:
                if selected_nodes[n1] == 0:
                    c = 0
                    for n2 in connected_graph.neighbors[n1]:
                        if selected_nodes[n2] == 1:
                            c += 1
                    outgoing_edges_num += c - outgoing_edges[n1]
                    outgoing_edges[n1] = c

        if sel_nodes_num < target_node_num:
            continue
        unprocessed = np.zeros(max_node + 1, dtype=int)
        unprocessed_num = 0
        for n in connected_graph.nodes:
            if selected_nodes[n] == 0:
                unprocessed[n] = 1
                unprocessed_num += 1
        i = target_node_num
        sel_nodes = list(n for n in range(max_node + 1) if selected_nodes[n] == 1)
        g = connected_graph.subgraph(sel_nodes)
        while outgoing_edges_num > 0:
            i += 1
            np.copyto(edge_weights, outgoing_edges)
            edge_weights /= outgoing_edges_num
            v = choice(max_node + 1, p=edge_weights)
            unprocessed[v] = 0
            unprocessed_num -= 1
            if random() < target_node_num/i:
                np.copyto(node_weights, selected_nodes)
                node_weights /= sel_nodes_num
                u = choice(max_node + 1, p=node_weights)
                isolated = False
                for u_n in g.neighbors[u]:
                    if len(g.neighbors[u_n]) == 1:
                        isolated = True
                        break
                if not isolated:
                    selected_nodes[u] = 0
                    g.delete_node(u)
                    selected_nodes[v] = 1
                    g.add_node(v)
                    for n in connected_graph.neighbors[v]:
                        if selected_nodes[n] == 1:
                            g.neighbors[n].append(v)
                            g.neighbors[v].append(n)
                    if not g.is_connected():
                        selected_nodes[u] = 1
                        g.add_node(u)
                        selected_nodes[v] = 0
                        g.delete_node(v)
                        for n in connected_graph.neighbors[u]:
                            if selected_nodes[n] == 1:
                                g.neighbors[n].append(u)
                                g.neighbors[u].append(n)
                    else:
                        for n1 in connected_graph.neighbors[v]:
                            if unprocessed[n1] == 1:
                                c = 0
                                for n2 in connected_graph.neighbors[n1]:
                                    if selected_nodes[n2] == 1:
                                        c += 1
                                outgoing_edges_num += c - outgoing_edges[n1]
                                outgoing_edges[n1] = c

                        for n1 in connected_graph.neighbors[u]:
                            c1 = outgoing_edges[n1]
                            if c1 > 0:
                                outgoing_edges[n1] = c1 - 1
                                outgoing_edges_num -= 1
            c = outgoing_edges[v]
            if c > 0:
                outgoing_edges_num -= c
                outgoing_edges[v] = 0
        sel_nodes = list(n for n in range(max_node + 1) if selected_nodes[n] == 1)
        target_graph = connected_graph.subgraph(sel_nodes)
        edge_num = target_graph.count_edges()
    if edge_num < target_edge_num:
        return None
    return target_graph


def compute_stat_on_random_subgraphs(thread_id, big_graph, small_graphs, n, pos_lists, prot_name):
    chi_sqr_stat = []
    jaccard_index_stat = [[] for i in range(len(small_graphs))]
    max_iter_stops = []
    if exists(temp_path + prot_name + '/' + str(thread_id) + '.random_graphs'):
        with open(temp_path + prot_name + '/' + str(thread_id) + '.random_graphs', 'r') as f:
            for line in f.readlines():
                sampled_graphs = []
                random_graphs = []
                s = line.strip().split('\t')
                for g in s[0].split(';'):
                    nodes = [int(p) for p in g.split(',')]
                    sampled_graphs.append(nodes)
                for g in s[1].split(';'):
                    nodes = [int(p) for p in g.split(',')]
                    random_graphs.append(nodes)
                max_iter_stops.append(int(s[2]))
                shuffled_indices = [int(i) for i in s[3].split(';')]
                int_set = set()
                for node_list in random_graphs:
                    int_set.update(node_list)
                chi_sqr_stat.append(chi_sqr(pos_lists, int_set, len(big_graph.nodes)))
                for j in range(len(sampled_graphs)):
                    gr = set(random_graphs[j])
                    g = set(sampled_graphs[j])
                    jaccard_index_stat[shuffled_indices[j]].append(len(gr.intersection(g))/len(gr.union(g)))
    iter_done = len(chi_sqr_stat)
    for i in range(iter_done, n):
        sampled_graphs = []
        random_graphs = []
        c = 0
        shuffled_indices = list(range(len(small_graphs)))
        shuffle(shuffled_indices)
        graphs_to_sample = []
        for j in shuffled_indices:
            graphs_to_sample.append(small_graphs[j])
        while len(graphs_to_sample) > 0:
            small_graph = graphs_to_sample.pop()
            target_node_num = len(small_graph.nodes)
            target_edge_num = small_graph.count_edges()
            nodes = set(big_graph.nodes)
            for g in random_graphs:
                for n in g.nodes:
                    nodes.remove(n)
            filtered_graph = big_graph.subgraph(nodes)
            connected_comps = filtered_graph.connected_components()
            connected_comps_filtered = []
            for g in connected_comps:
                if len(g.nodes) >= target_node_num and g.count_edges() >= target_edge_num:
                    connected_comps_filtered.append(g)
            shuffle(connected_comps_filtered)
            random_graph = None
            for g in connected_comps_filtered:
                random_graph = gen_random_subgraph_new1(g, target_node_num, target_edge_num)
                if random_graph is not None:
                    break
            if random_graph is None:
                c += 1
                graphs_to_sample.append(small_graph)
                if len(sampled_graphs) > 0:
                    graphs_to_sample.append(sampled_graphs.pop())
                    random_graphs.pop()
            else:
                sampled_graphs.append(small_graph)
                random_graphs.append(random_graph)
        with open(temp_path + prot_name + '/' + str(thread_id) + '.random_graphs', 'a') as f:
            r_graphs = []
            for g in random_graphs:
                g_str = [str(node) for node in g.nodes]
                r_graphs.append(','.join(g_str))
            s_graphs = []
            for g in sampled_graphs:
                g_str = [str(node) for node in g.nodes]
                s_graphs.append(','.join(g_str))
            sh_indices = ';'.join([str(i) for i in shuffled_indices])
            dump = [';'.join(s_graphs), ';'.join(r_graphs), str(c), sh_indices]
            f.write('\t'.join(dump) + '\n')
        int_set = set()
        for g in random_graphs:
            int_set.update(g.nodes)
        chi_sqr_stat.append(chi_sqr(pos_lists, int_set, len(big_graph.nodes)))
        for j in range(len(sampled_graphs)):
            gr = set(random_graphs[j].nodes)
            g = set(sampled_graphs[j].nodes)
            jaccard_index_stat[shuffled_indices[j]].append(len(gr.intersection(g))/len(gr.union(g)))
        max_iter_stops.append(c)
    return chi_sqr_stat, jaccard_index_stat, max_iter_stops


def create_graph(pos_to_coords, poses):
    g = Graph(nodes=poses)
    for i in range(len(poses)):
        p_i = poses[i]
        for j in range(i + 1, len(poses)):
            p_j = poses[j]
            if dist(pos_to_coords[p_i], pos_to_coords[p_j]) < dist_threshold:
                g.neighbors[p_i].append(p_j)
                g.neighbors[p_j].append(p_i)
    return g


def test_independence(pos_to_coords, cluster_ids, interface, filter_set, prot_name):
    if debug:
        print('computing p_value')
    filtered_poses = list(filter_set)
    big_graph = create_graph(pos_to_coords, filtered_poses)

    if debug:
        connected_comps = big_graph.connected_components()
        print('big graph:')
        lens = [str(len(comp.nodes)) for comp in connected_comps]
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

    interface_graph = big_graph.subgraph(interface)
    small_graphs = interface_graph.connected_components()
    if debug:
        print('interface:')
        lens = [str(len(comp.nodes)) for comp in small_graphs]
        print('connected comp lens: ' + ' '.join(lens))

    stat = chi_sqr(cl_to_poses.values(), interface, len(filtered_poses))
    iter_nums = []
    for i in range(thread_num):
        iter_nums.append(permutations_num//thread_num)
    for i in range(permutations_num%thread_num):
        iter_nums[i] += 1
    if not exists(temp_path + prot_name):
        makedirs(temp_path + prot_name)
    tasks = Parallel(n_jobs=thread_num)(delayed(compute_stat_on_random_subgraphs)(i, big_graph, small_graphs,
                                                                                 iter_nums[i],
                                                                                  list(cl_to_poses.values()), prot_name)
                                        for i in range(thread_num))
    if print_random_graphs:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        jaccad_indices = [[] for i in range(len(small_graphs))]
        chi_sqr_stats = []
        max_iter_stops_arr = []
        for chi_sqr_stat, jaccard_index_stat, max_iter_stops in tasks:
            for i in range(len(small_graphs)):
                jaccad_indices[i].extend(jaccard_index_stat[i])
            chi_sqr_stats.extend(chi_sqr_stat)
            max_iter_stops_arr.extend(max_iter_stops)
        for i in range(len(small_graphs)):
            plt.title('Histogram of Jaccard index of random graphs')
            plt.xlabel('Jaccard index')
            plt.ylabel('Percent of graphs')
            n, bins, patches = plt.hist(jaccad_indices[i], 50, density=True, facecolor='g', alpha=0.75)
            # plt.axis([0, 0.002, 0, 6000])
            plt.savefig(random_graph_stat_hist_path + prot_name + '_' + str(i) + '.png')
            plt.clf()
        plt.title('Histogram of ChiSqr stat of random graphs')
        plt.xlabel('ChiSqr')
        plt.ylabel('Percent of graphs')
        n, bins, patches = plt.hist(chi_sqr_stats, 50, density=True, facecolor='g', alpha=0.75)
        # plt.axis([0, 0.002, 0, 6000])
        plt.savefig(random_graph_stat_hist_path + prot_name + '_chi.png')
        plt.clf()
        plt.title('Histogram of number of rejects during generation of random graphs')
        plt.xlabel('number of rejects')
        plt.ylabel('Percent of graphs')
        n, bins, patches = plt.hist(max_iter_stops_arr, 50, density=True, facecolor='g', alpha=0.75)
        # plt.axis([0, 0.002, 0, 6000])
        plt.savefig(random_graph_stat_hist_path + prot_name + '_rejects.png')
        plt.clf()
    i = 0
    for chi_sqr_stat, jaccard_index_stat, max_iter_stops in tasks:
        for s in chi_sqr_stat:
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
                if filter_set is None or pos in filter_set:
                    non_int_counts[cl - 1] += 1
    return non_int_counts, int_counts


def print_unified_intefaces():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    if use_cox_data:
        prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    else:
        if use_dssp:
            prot_to_buried, prot_to_non_buried = read_dssp_data(path_to_dssp_data)
        else:
            prot_to_buried, prot_to_non_buried = read_surf_racer_data(path_to_surf_racer_data)
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
        if method_name != '':
            print(prot_name)
        else:
            print(prot_name + '.' + method_name)
        int = interfaces[prot_name]
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        filter_set = prot_to_non_buried[prot_name]
        cl_counts, int_counts = count(cluster_ids, int, filter_set)
        if method_name != '':
            p_value = test_independence(coords, cluster_ids, int, filter_set, prot_name)
        else:
            p_value = test_independence(coords, cluster_ids, int, filter_set, prot_name + '.' + method_name)
        print_table(cl_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)


def print_unified_intefaces_aledo():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    cox_data = pd.read_csv(path_to_cox_data, sep='\t', decimal='.')
    cox_data['Prot'] = cox_data['Chain'].apply(lambda x: chain_to_prot[x])
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name)
        else:
            print(prot_name + '.' + method_name)
        non_burried = cox_data[(cox_data['Prot'] == prot_name) & (cox_data['BCEE'] != 'BURIED')]
        if only_mitochondria_to_nuclear:
            interface = set(non_burried.loc[(non_burried['Cont'] == 2) | (non_burried['Cont'] == 3), 'Pos'])
        else:
            interface = set(non_burried.loc[(non_burried['Cont'] == 1) | (non_burried['Cont'] == 3), 'Pos'])
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        non_int_counts, int_counts = count(cluster_ids, interface, set(non_burried['Pos']))
        if method_name != '':
            p_value = test_independence(coords, cluster_ids, interface, set(non_burried['Pos']), prot_name)
        else:
            p_value = test_independence(coords, cluster_ids, interface, set(non_burried['Pos']), prot_name + '.' + method_name)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)


def print_unified_intefaces_aledo1():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    cytb_data = pd.read_csv(path_to_cytb_data, sep=',', decimal='.')
    cytb_data['Prot'] = cytb_data['Chain'].apply(lambda x: chain_to_prot[x])
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name)
        else:
            print(prot_name + '.' + method_name)
        non_burried = cytb_data[(cytb_data['Prot'] == prot_name) & (cytb_data['acc.subunit'] >= 0.05)]
        interface = set(non_burried.loc[non_burried['InterContact'] > 0, 'ResidNr'])
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        non_int_counts, int_counts = count(cluster_ids, interface, set(non_burried['ResidNr']))
        if method_name != '':
            p_value = test_independence(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name)
        else:
            p_value = test_independence(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name + '.' + method_name)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)


def print_unified_intefaces_aledo1_enc():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    cytb_data = pd.read_csv(path_to_cytb_data, sep=',', decimal='.')
    cytb_data['Prot'] = cytb_data['Chain'].apply(lambda x: chain_to_prot[x])
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name)
        else:
            print(prot_name + '.' + method_name)
        non_burried = cytb_data[(cytb_data['Prot'] == prot_name) & (cytb_data['acc.subunit'] >= 0.05)]
        interface = set(non_burried.loc[(non_burried['InterContact'] > 0) | (non_burried['DeltaSASA'] > 0), 'ResidNr'])
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        non_int_counts, int_counts = count(cluster_ids, interface, set(non_burried['ResidNr']))
        if method_name != '':
            p_value = test_independence(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name)
        else:
            p_value = test_independence(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name + '.' + method_name)
        print_table(non_int_counts, int_counts, 'ENC_noninterf', 'CONT + ENC_interface', p_value)


def print_unified_intefaces_enc():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    prot_to_burried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name + '.' + method_name)
        else:
            print(prot_name)
        non_buried = prot_to_non_buried[prot_name]
        non_interface = prot_to_non_interface[prot_name]
        interface = [pos for pos in non_buried if pos not in non_interface]
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        for i in range(len(cluster_ids)):
            if i not in non_buried:
                cluster_ids[i] = 0
        non_int_counts, int_counts = count(cluster_ids, interface, non_buried)
        if method_name != '':
            p_value = test_independence(coords, cluster_ids, interface, non_buried, prot_name + '.' + method_name)
        else:
            p_value = test_independence(coords, cluster_ids, interface, non_buried, prot_name)
        print_table(non_int_counts, int_counts, 'ENC_noninterf', 'CONT + ENC_interface', p_value)


def print_separate_intefaces():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    # prot_to_buried, prot_to_non_buried = read_dssp_data()
    prot_name_to_clusters = {}
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        prot_name_to_clusters[prot_name] = (method_name, cluster_ids)
    for chain1, coords1 in chain_to_site_coords.items():
        prot_name1 = chain_to_prot[chain1]
        non_burried1 = prot_to_non_buried[prot_name1]
        for chain2, coords2 in chain_to_site_coords.items():
            prot_name2 = chain_to_prot[chain2]
            non_burried2 = prot_to_non_buried[prot_name2]
            if prot_name1 < prot_name2:
                int1, int2 = get_interface(coords1, coords2, non_burried1, non_burried2)
                if len(int1) == 0:
                    continue
                method_name, cluster_ids = prot_name_to_clusters[prot_name1]
                coords = chain_to_site_coords[prot_to_chain[prot_name1]]
                filter_set = non_burried1
                for i in range(len(cluster_ids)):
                    if i not in filter_set:
                        cluster_ids[i] = 0
                cl_counts, int_counts = count(cluster_ids, int1, filter_set)

                if method_name != '':
                    print(prot_name1 + ' vs ' + prot_name2 + ' ' + method_name)
                    p_value = test_independence(coords, cluster_ids, int1, filter_set, prot_name1 + ' vs ' + prot_name2 + ' ' + method_name)
                else:
                    print(prot_name1 + ' vs ' + prot_name2)
                    p_value = test_independence(coords, cluster_ids, int1, filter_set, prot_name1 + ' vs ' + prot_name2)
                print_table(cl_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)
                method_name, cluster_ids = prot_name_to_clusters[prot_name2]
                coords = chain_to_site_coords[prot_to_chain[prot_name2]]
                filter_set = non_burried2
                for i in range(len(cluster_ids)):
                    if i not in filter_set:
                        cluster_ids[i] = 0
                cl_counts, int_counts = count(cluster_ids, int2, filter_set)

                if method_name != '':
                    print(prot_name2 + ' vs ' + prot_name1 + ' ' + method_name)
                    p_value = test_independence(coords, cluster_ids, int2, filter_set, prot_name2 + ' vs ' + prot_name1 + ' ' + method_name)
                else:
                    print(prot_name2 + ' vs ' + prot_name1)
                    p_value = test_independence(coords, cluster_ids, int2, filter_set, prot_name2 + ' vs ' + prot_name1)
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


if __name__ == '__main__':
    # print_unified_intefaces()
    # print_unified_intefaces_enc()
    # print_separate_intefaces()
    print_unified_intefaces_aledo()
    # print_unified_intefaces_aledo1_enc()
