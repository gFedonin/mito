from os import makedirs, cpu_count
from os.path import exists
from random import shuffle, random, randrange
from igraph import *

import numpy as np
import pandas as pd
from numpy.random.mtrand import choice
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.externals.joblib import Parallel, delayed

from compute_cluster_stats import dist, parse_colors, parse_out, parse_site2pdb
from print_xnomial_table import parse_pdb, read_cox_data

pdb_id = '1occ'
path_to_pdb = '../pdb/' + pdb_id + '.pdb1'
# path_to_pdb = '../pdb/1be3.pdb1'
path_to_cox_data = '../Coloring/COXdata.txt'
path_to_cytb_data = '../aledo.csv'
path_to_dssp_dir = '../dssp/'
path_to_dssp_raw = path_to_dssp_dir + pdb_id + '.dssp'
path_to_colors = '../Coloring/internal_gaps.2/'

# chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}#{'A': 'cox1'} {'C': 'cytb'}
chain_to_prot = {'C': 'cytb'}
# prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}#{'cox1': 'A'} {'cytb': 'C'}
prot_to_chain = {'cytb': 'C'}
dist_threshold = 8

use_colors = False
filter_burried = False
debug = False
random_graph_stat_hist_path = '../res/random_graph_stat_hist_cytb_Aledo_igraph_enc/'
temp_path = random_graph_stat_hist_path + 'temp/'
if debug:
    thread_num = 1
else:
    thread_num = cpu_count()
if debug:
    permutations_num = 1
else:
    permutations_num = 10000
max_iter = 10000


def gen_random_subgraph(connected_graph, target_node_num, target_edge_num):
    target_graph = None
    edge_num = -1
    iterNum = 0
    node_num = connected_graph.vcount()
    max_name = max(connected_graph.vs['name'])
    while edge_num < target_edge_num and iterNum < max_iter:
        iterNum += 1
        selected_nodes = np.zeros(node_num, dtype=int)
        outgoing_edges = np.zeros(node_num, dtype=int)
        edge_weights = np.zeros(node_num, dtype=float)
        node_index = choice(node_num)
        selected_nodes[node_index] = 1
        sel_nodes_num = 1
        outgoing_edges_num = 0
        for n in connected_graph.neighbors(node_index):
            outgoing_edges[n] = 1
            outgoing_edges_num += 1
        for i in range(1, target_node_num):
            if outgoing_edges_num == 0:
                break
            np.copyto(edge_weights, outgoing_edges)
            edge_weights /= outgoing_edges_num
            n = choice(node_num, p=edge_weights)
            outgoing_edges_num -= outgoing_edges[n]
            outgoing_edges[n] = 0
            selected_nodes[n] = 1
            sel_nodes_num += 1
            for n1 in connected_graph.neighbors(n):
                if selected_nodes[n1] == 0:
                    c = 0
                    for n2 in connected_graph.neighbors(n1):
                        if selected_nodes[n2] == 1:
                            c += 1
                    outgoing_edges_num += c - outgoing_edges[n1]
                    outgoing_edges[n1] = c

        if sel_nodes_num < target_node_num:
            continue
        unprocessed = np.zeros(node_num, dtype=int)
        unprocessed_num = 0
        for n in range(node_num):
            if selected_nodes[n] == 0:
                unprocessed[n] = 1
                unprocessed_num += 1
        i = target_node_num
        sel_nodes = [n for n in range(node_num) if selected_nodes[n] == 1]
        g = connected_graph.subgraph(sel_nodes)
        g_name_to_index = np.zeros(max_name + 1, dtype=int)
        for i in range(target_node_num):
            g_name_to_index[g.vs['name'][i]] = i
        while outgoing_edges_num > 0:
            i += 1
            np.copyto(edge_weights, outgoing_edges)
            edge_weights /= outgoing_edges_num
            v = choice(node_num, p=edge_weights)
            unprocessed[v] = 0
            unprocessed_num -= 1
            if random() < target_node_num/i:
                u_index = choice(target_node_num)
                u = sel_nodes[u_index]
                u_name = connected_graph.vs['name'][u]
                u_g_index = g_name_to_index[u_name]
                isolated = False
                for n_u in g.neighbors(u_g_index):
                    if len(g.neighbors(n_u)) == 1:
                        isolated = True
                        break
                if not isolated:
                    selected_nodes[u] = 0
                    # u_g_index = g.vs.find(name=u_name).index
                    # g.delete_vertices(g.vs.find(name=u_name))
                    u_g_edge_list = [(u_g_index, n) for n in g.neighbors(u_g_index)]
                    g.delete_edges([g.get_eid(*edge) for edge in u_g_edge_list])
                    selected_nodes[v] = 1
                    sel_nodes[u_index] = v
                    v_name = connected_graph.vs['name'][v]
                    g.vs['name'][u_g_index] = v_name
                    g_name_to_index[v_name] = u_g_index
                    # g.add_vertex(name=v_name)
                    v_g_edge_list = []
                    for n in connected_graph.neighbors(v):
                        if selected_nodes[n] == 1:
                            n_name = connected_graph.vs['name'][n]
                            # n_g_index = g.vs.find(name=n_name).index
                            n_g_index = g_name_to_index[n_name]
                            v_g_edge_list.append((u_g_index, n_g_index))
                            # g[n_name, v_name] = 1
                    g.add_edges(v_g_edge_list)
                    if not g.is_connected():
                        selected_nodes[v] = 0
                        # g.delete_vertices(g.vs.find(name=v_name))
                        g.vs['name'][u_g_index] = u_name
                        g.delete_edges([g.get_eid(*edge) for edge in v_g_edge_list])
                        selected_nodes[u] = 1
                        sel_nodes[u_index] = u
                        # g.add_vertex(name=u_name)
                        g.add_edges(u_g_edge_list)
                        # for n in connected_graph.neighbors(u):
                        #     if selected_nodes[n] == 1:
                        #         n_name = connected_graph.vs['name'][n]
                        #         g[n_name, u_name] = 1
                    else:
                        for n1 in connected_graph.neighbors(v):
                            if unprocessed[n1] == 1:
                                c = 0
                                for n2 in connected_graph.neighbors(n1):
                                    if selected_nodes[n2] == 1:
                                        c += 1
                                outgoing_edges_num += c - outgoing_edges[n1]
                                outgoing_edges[n1] = c

                        for n1 in connected_graph.neighbors(u):
                            c1 = outgoing_edges[n1]
                            if c1 > 0:
                                outgoing_edges[n1] = c1 - 1
                                outgoing_edges_num -= 1
            c = outgoing_edges[v]
            if c > 0:
                outgoing_edges_num -= c
                outgoing_edges[v] = 0
        sel_nodes = list(n for n in range(node_num) if selected_nodes[n] == 1)
        target_graph = connected_graph.subgraph(sel_nodes)
        edge_num = target_graph.ecount()
    if edge_num < target_edge_num:
        return None
    return target_graph


def gen_random_subgraphs_for_each_group(thread_id, big_graph, small_graphs, n, prot_name):
    iter_done = 0
    if exists(temp_path + prot_name + '/' + str(thread_id) + '.random_graphs'):
        with open(temp_path + prot_name + '/' + str(thread_id) + '.random_graphs', 'r') as f:
            for line in f.readlines():
                iter_done += 1

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
            target_node_num = small_graph.vcount()
            target_edge_num = small_graph.ecount()
            nodes = set(big_graph.vs['name'])
            for g in random_graphs:
                for n in g.vs['name']:
                    nodes.remove(n)
            nodes = big_graph.vs.select(name_in=nodes)
            filtered_graph = big_graph.induced_subgraph(nodes)
            connected_comps = filtered_graph.components().subgraphs()
            connected_comps_filtered = []
            for g in connected_comps:
                if g.vcount() >= target_node_num and g.ecount() >= target_edge_num:
                    connected_comps_filtered.append(g)
            shuffle(connected_comps_filtered)
            random_graph = None
            for g in connected_comps_filtered:
                random_graph = gen_random_subgraph(g, target_node_num, target_edge_num)
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
                l = [str(n) for n in g.vs['name']]
                r_graphs.append(','.join(l))
            s_graphs = []
            for g in sampled_graphs:
                l = [str(n) for n in g.vs['name']]
                s_graphs.append(','.join(l))
            shuffled_indices.reverse()
            sh_indices = ';'.join([str(i) for i in shuffled_indices])
            dump = [';'.join(s_graphs), ';'.join(r_graphs), str(c), sh_indices]
            f.write('\t'.join(dump) + '\n')
    return 0


def create_graph(pos_to_coords, poses):
    g = Graph()
    g.add_vertices(len(poses))
    g.vs['name'] = list(poses)
    for i in range(len(poses)):
        p_i = poses[i]
        for j in range(i + 1, len(poses)):
            p_j = poses[j]
            if dist(pos_to_coords[p_i], pos_to_coords[p_j]) < dist_threshold:
                g.add_edges([(i, j)])
    return g


def compute_graphs(pos_to_coords, pos_set, filter_set):
    filtered_poses = list(filter_set)
    big_graph = create_graph(pos_to_coords, filtered_poses)

    if debug:
        connected_comps = big_graph.components().subgraphs()
        print('big graph:')
        lens = [str(comp.vcount()) for comp in connected_comps]
        print('connected comp lens: ' + ' '.join(lens))

    pos_list = big_graph.vs.select(name_in=pos_set)
    pos_set_graph = big_graph.subgraph(pos_list)
    small_graphs = pos_set_graph.components().subgraphs()
    if debug:
        print('secondary structure:')
        lens = [str(comp.vcount()) for comp in small_graphs]
        print('connected comp lens: ' + ' '.join(lens))
    return big_graph, small_graphs


def gen_all_random_graphs(pos_to_coords, pos_set, filter_set, prot_name):
    big_graph, small_graphs = compute_graphs(pos_to_coords, pos_set, filter_set)
    iter_nums = []
    for i in range(thread_num):
        iter_nums.append(permutations_num//thread_num)
    for i in range(permutations_num%thread_num):
        iter_nums[i] += 1
    if not exists(temp_path + prot_name):
        makedirs(temp_path + prot_name)
    tasks = Parallel(n_jobs=thread_num)(delayed(gen_random_subgraphs_for_each_group)(i, big_graph, small_graphs,
                                                                                     iter_nums[i], prot_name)
                                        for i in range(thread_num))
    c = 0
    for task in tasks:
        c += 1


def parse_dssp():
    chain_to_ss = {}
    for chain in chain_to_prot.keys():
        chain_to_ss[chain] = {}
    with open(path_to_dssp_raw) as f:
        for line in f.readlines()[28:]:
            s = line.strip().split()
            if s[2] in chain_to_ss:
                pos_to_ss = chain_to_ss[s[2]]
                if s[4] in 'HISTGBEC':
                    pos_to_ss[int(s[1])] = s[4]
                else:
                    pos_to_ss[int(s[1])] = 'C'
    return chain_to_ss


def build_secondary_structure_random_graphs():
    chain_to_ss = parse_dssp()
    prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    if filter_burried:
        prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    chain_to_site_coords = parse_pdb(path_to_pdb, True, chain_to_prot)
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name)
        else:
            print(prot_name + '.' + method_name)
        if filter_burried:
            non_burried = prot_to_non_buried[prot_name]
        else:
            non_burried = None
        pos_to_ss = chain_to_ss[prot_to_chain[prot_name]]
        ss_to_pos = {}
        for pos, ss in pos_to_ss.items():
            pos_set = ss_to_pos.get(ss)
            if pos_set is None:
                pos_set = set()
                ss_to_pos[ss] = pos_set
            pos_set.add(pos)
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        for ss, pos_set in ss_to_pos.items():
            gen_all_random_graphs(coords, pos_set, non_burried, prot_name)


if __name__ == '__main__':
    build_secondary_structure_random_graphs()
