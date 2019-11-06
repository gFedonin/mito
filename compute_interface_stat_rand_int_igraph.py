from os import makedirs, cpu_count
from os.path import exists
from random import shuffle, random, randrange

import Bio.PDB
from igraph import *

import numpy as np
import pandas as pd
from numpy.random.mtrand import choice
import matplotlib
matplotlib.use('agg')

from sklearn.externals.joblib import Parallel, delayed

from compute_cluster_stats import dist, parse_colors, parse_out, parse_site2pdb
from assimptotic_tests import parse_pdb, get_interface, read_cox_data

# pdb_id = '5ara'
# pdb_id = '1be3'
# pdb_id = '1bgy'
pdb_id = '1occ'
path_to_pdb = '../pdb/' + pdb_id + '.pdb'
path_to_cox_data = '../Coloring/COXdata.txt'
path_to_cytb_data = '../aledo.csv'
path_to_atp6_data = '../Coloring/cytb_1bgy_Aledo_4ang.csv'
# path_to_surf_racer_data = '../surf_racer/burried/1bgy.csv'
# path_to_dssp_data = '../dssp/1be3.csv'
path_to_colors = '../Coloring/internal_gaps.2/'

# chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}#{'A': 'cox1'} {'C': 'cytb'}
# chain_to_prot = {'C': 'cytb'}
# chain_to_prot = {'W': 'atp6'}
chain_to_prot = {'A': 'cox1'}
# prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}#{'cox1': 'A'} {'cytb': 'C'}
# prot_to_chain = {'cytb': 'C'}
# prot_to_chain = {'atp6': 'W'}
prot_to_chain = {'cox1': 'A'}
aledo_dist = True
dist_threshold = 4

use_colors = False
use_cox_data = True
use_dssp = False
debug = False
only_selected_chains = False
only_mitochondria_to_nuclear = True
random_graph_stat_hist_path = '../res/random_graph_stat_hist_cox1_enc_Aledo_igraph/'
# temp_path = random_graph_stat_hist_path + 'temp/'
if debug:
    thread_num = 1
else:
    thread_num = cpu_count()
if debug:
    permutations_num = 1
else:
    permutations_num = 10000
max_iter = 10000


# def read_surf_racer_data(path_to_surf_racer_data):
#     prot_to_non_buried = {}
#     prot_to_buried = {}
#     for prot_name in prot_to_chain.keys():
#         prot_to_non_buried[prot_name] = set()
#         prot_to_buried[prot_name] = set()
#     with open(path_to_surf_racer_data, 'r') as f:
#         for line in f.readlines():
#             s = line.strip().split('\t')
#             if s[-1] == 'burried':
#                 prot_to_buried[chain_to_prot[s[0]]].add(int(s[1]))
#             else:
#                 prot_to_non_buried[chain_to_prot[s[0]]].add(int(s[1]))
#     return prot_to_buried, prot_to_non_buried


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
    edge_num = -1
    i = 0
    nodes_num = connected_graph.vcount()
    while edge_num < target_edge_num and i < max_iter:
        i += 1
        selected_nodes = set()
        neighbors = set()
        node_index = randrange(nodes_num)
        selected_nodes.add(node_index)
        for e in connected_graph.es.select(_from=node_index):
            neighbors.add(e.target)
        for i in range(1, target_node_num):
            if len(neighbors) == 0:
                break
            node_index = choice(list(neighbors))
            neighbors.remove(node_index)
            selected_nodes.add(node_index)
            for e in connected_graph.es.select(_from=node_index):
                if e.target not in selected_nodes:
                    neighbors.add(e.target)
        if len(selected_nodes) < target_node_num:
            continue
        target_graph = connected_graph.subgraph(list(selected_nodes))
        edge_num = target_graph.ecount()
    if edge_num < target_edge_num:
        return None
    return target_graph


def gen_random_subgraph_new1(connected_graph, target_node_num, target_edge_num):
    target_graph = None
    edge_num = -1
    iterNum = 0
    node_num = connected_graph.vcount()
    while edge_num < target_edge_num and iterNum < max_iter:
        iterNum += 1
        selected_nodes = np.zeros(node_num, dtype=int)
        outgoing_edges = np.zeros(node_num, dtype=int)
        edge_weights = np.zeros(node_num, dtype=float)
        node_weights = np.zeros(node_num, dtype=float)
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
        sel_nodes = list(n for n in range(node_num) if selected_nodes[n] == 1)
        g = connected_graph.subgraph(sel_nodes)
        while outgoing_edges_num > 0:
            i += 1
            np.copyto(edge_weights, outgoing_edges)
            edge_weights /= outgoing_edges_num
            v = choice(node_num, p=edge_weights)
            unprocessed[v] = 0
            unprocessed_num -= 1
            if random() < target_node_num/i:
                np.copyto(node_weights, selected_nodes)
                node_weights /= sel_nodes_num
                u = choice(node_num, p=node_weights)
                u_vertex = connected_graph.vs['name'][u]
                selected_nodes[u] = 0
                g.delete_vertices(g.vs.find(name=u_vertex))
                selected_nodes[v] = 1
                v_vertex = connected_graph.vs['name'][v]
                g.add_vertex(name=v_vertex)
                for n in connected_graph.neighbors(v):
                    if selected_nodes[n] == 1:
                        n_vertex = connected_graph.vs['name'][n]
                        g[n_vertex, v_vertex] = 1
                if not g.is_connected():
                    selected_nodes[v] = 0
                    g.delete_vertices(g.vs.find(name=v_vertex))
                    selected_nodes[u] = 1
                    g.add_vertex(name=u_vertex)
                    for n in connected_graph.neighbors(u):
                        if selected_nodes[n] == 1:
                            n_vertex = connected_graph.vs['name'][n]
                            g[n_vertex, u_vertex] = 1
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


def gen_random_subgraph_new2(connected_graph, target_node_num, target_edge_num):
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


def print_random_subgraphs(thread_id, big_graph, small_graphs, n, prot_name):
    iter_done = 0
    if exists(random_graph_stat_hist_path + prot_name + '/' + str(thread_id) + '.random_graphs'):
        with open(random_graph_stat_hist_path + prot_name + '/' + str(thread_id) + '.random_graphs', 'r') as f:
            for line in f.readlines():
                sampled_graphs = []
                random_graphs = []
                s = line.strip().split('\t')
                for g in s[0].split(';'):
                    nodes = [int(n) for n in g.split(',')]
                    sampled_graphs.append(nodes)
                for g in s[1].split(';'):
                    nodes = [int(n) for n in g.split(',')]
                    random_graphs.append(nodes)
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
                random_graph = gen_random_subgraph_new2(g, target_node_num, target_edge_num)
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
        with open(random_graph_stat_hist_path + prot_name + '/' + str(thread_id) + '.random_graphs', 'a') as f:
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
    return 1


def create_graph(pos_to_coords, poses, dist_f):
    g = Graph()
    g.add_vertices(len(poses))
    g.vs['name'] = list(poses)
    for i in range(len(poses)):
        p_i = poses[i]
        for j in range(i + 1, len(poses)):
            p_j = poses[j]
            if dist_f(pos_to_coords[p_i], pos_to_coords[p_j]) < dist_threshold:
                g.add_edges([(i, j)])
    return g


def dist_aledo(heavy_atoms1, heavy_atoms2):
    n1 = len(heavy_atoms1)
    n2 = len(heavy_atoms2)
    return min(dist(heavy_atoms1[i], heavy_atoms2[j]) for i in range(n1) for j in range(n2))


def parse_pdb_Aledo_biopython(pdb_name, path_to_pdb, only_selected_chains, chain_to_prot):
    chain_to_site_coords = {}
    structure = Bio.PDB.PDBParser().get_structure(pdb_name, path_to_pdb)
    model = structure[0]
    for chn in model:
        if only_selected_chains and chn.id not in chain_to_prot:
            continue
        pos_to_coords = {}
        for residue in chn:
            pos = int(residue.id[1])
            atoms = [tuple(atom.coord) for atom in residue]
            pos_to_coords[pos] = atoms
        chain_to_site_coords[chn.id] = pos_to_coords
    return chain_to_site_coords


def compute_graphs(pos_to_coords, cluster_ids, interface, filter_set, dist_f):
    if debug:
        print('computing p_value')
    filtered_poses = list(filter_set)
    big_graph = create_graph(pos_to_coords, filtered_poses, dist_f)

    if debug:
        connected_comps = big_graph.components().subgraphs()
        print('big graph:')
        lens = [str(comp.vcount()) for comp in connected_comps]
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
    # interface_names = set(str(p) for p in interface)
    int_list = big_graph.vs.select(name_in=interface)
    interface_graph = big_graph.subgraph(int_list)
    small_graphs = interface_graph.components().subgraphs()
    if debug:
        print('interface:')
        lens = [str(comp.vcount()) for comp in small_graphs]
        print('connected comp lens: ' + ' '.join(lens))
    return cl_to_poses, len(filtered_poses), big_graph, small_graphs


def print_all_random_subgraphs(pos_to_coords, cluster_ids, interface, filter_set, prot_name, dist_f):
    cl_to_poses, filtered_poses_num, big_graph, small_graphs = compute_graphs(pos_to_coords, cluster_ids, interface, filter_set, dist_f)

    iter_nums = []
    for i in range(thread_num):
        iter_nums.append(permutations_num//thread_num)
    for i in range(permutations_num%thread_num):
        iter_nums[i] += 1
    if not exists(random_graph_stat_hist_path + prot_name):
        makedirs(random_graph_stat_hist_path + prot_name)
    tasks = Parallel(n_jobs=thread_num)(delayed(print_random_subgraphs)(i, big_graph, small_graphs,
                                                                        iter_nums[i], prot_name)
                                        for i in range(thread_num))
    c = 0
    for task in tasks:
        c += task
    print('%d jobs done' % c)


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
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
        chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
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
        if method_name != '':
            print_all_random_subgraphs(coords, cluster_ids, int, filter_set, prot_name, dist_f)
        else:
            print_all_random_subgraphs(coords, cluster_ids, int, filter_set, prot_name + '.' + method_name, dist_f)


def print_unified_intefaces_aledo():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
        chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    cox_data = pd.read_csv(path_to_cox_data, sep='\t', decimal='.')
    # cox_data['Prot'] = cox_data['Chain'].apply(lambda x: chain_to_prot[x])
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name)
        else:
            print(prot_name + '.' + method_name)
        chain = prot_to_chain[prot_name]
        non_burried = cox_data[(cox_data['Chain'] == chain) & (cox_data['BCEE'] != 'BURIED')]
        if only_mitochondria_to_nuclear:
            interface = set(non_burried.loc[(non_burried['Cont'] == 2) | (non_burried['Cont'] == 3), 'Pos'])
        else:
            interface = set(non_burried.loc[(non_burried['Cont'] == 1) | (non_burried['Cont'] == 3), 'Pos'])
        coords = chain_to_site_coords[chain]
        if method_name != '':
            print_all_random_subgraphs(coords, cluster_ids, interface, set(non_burried['Pos']), prot_name, dist_f)
        else:
            print_all_random_subgraphs(coords, cluster_ids, interface, set(non_burried['Pos']), prot_name + '.' + method_name, dist_f)


def print_unified_intefaces_aledo_cytb():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
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
        if method_name != '':
            print_all_random_subgraphs(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name, dist_f)
        else:
            print_all_random_subgraphs(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name + '.' + method_name, dist_f)


def print_unified_intefaces_aledo_cytb_enc():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
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
        if method_name != '':
            print_all_random_subgraphs(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name, dist_f)
        else:
            print_all_random_subgraphs(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name + '.' + method_name, dist_f)


def print_unified_intefaces_aledo_atp6():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
        chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    atp6_data = pd.read_csv(path_to_atp6_data, sep='\t', decimal='.')
    atp6_data['Prot'] = atp6_data['chain'].apply(lambda x: chain_to_prot[x])
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name)
        else:
            print(prot_name + '.' + method_name)
        non_burried = atp6_data[(atp6_data['Prot'] == prot_name) & (atp6_data['BCEE'] != 'BURIED')]
        interface = set(non_burried.loc[non_burried['InterContact'] > 0, 'pos'])
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        if method_name != '':
            print_all_random_subgraphs(coords, cluster_ids, interface, set(non_burried['pos']), prot_name, dist_f)
        else:
            print_all_random_subgraphs(coords, cluster_ids, interface, set(non_burried['pos']), prot_name + '.' + method_name, dist_f)


def print_unified_intefaces_aledo_atp6_enc():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
        chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    atp6_data = pd.read_csv(path_to_atp6_data, sep='\t', decimal='.')
    atp6_data['Prot'] = atp6_data['chain'].apply(lambda x: chain_to_prot[x])
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name)
        else:
            print(prot_name + '.' + method_name)
        non_burried = atp6_data[(atp6_data['Prot'] == prot_name) & (atp6_data['BCEE'] != 'BURIED')]
        interface = set(non_burried.loc[(non_burried['BCEE'] == 'CONT') | (non_burried['BCEE'] == 'ENC_interface'), 'pos'])
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        if method_name != '':
            print_all_random_subgraphs(coords, cluster_ids, interface, set(non_burried['pos']), prot_name, dist_f)
        else:
            print_all_random_subgraphs(coords, cluster_ids, interface, set(non_burried['pos']), prot_name + '.' + method_name, dist_f)


def print_unified_intefaces_enc():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
        chain_to_site_coords = parse_pdb(path_to_pdb + pdb_id, only_selected_chains, chain_to_prot)
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
        if method_name != '':
            print_all_random_subgraphs(coords, cluster_ids, interface, non_buried, prot_name + '.' + method_name, dist_f)
        else:
            print_all_random_subgraphs(coords, cluster_ids, interface, non_buried, prot_name, dist_f)


def print_separate_intefaces():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
        chain_to_site_coords = parse_pdb(path_to_pdb + pdb_id, only_selected_chains, chain_to_prot)
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

                if method_name != '':
                    print(prot_name1 + ' vs ' + prot_name2 + ' ' + method_name)
                    print_all_random_subgraphs(coords, cluster_ids, int1, filter_set, prot_name1 + ' vs ' + prot_name2 + ' ' + method_name, dist_f)
                else:
                    print(prot_name1 + ' vs ' + prot_name2)
                    print_all_random_subgraphs(coords, cluster_ids, int1, filter_set, prot_name1 + ' vs ' + prot_name2, dist_f)
                method_name, cluster_ids = prot_name_to_clusters[prot_name2]
                coords = chain_to_site_coords[prot_to_chain[prot_name2]]
                filter_set = non_burried2
                for i in range(len(cluster_ids)):
                    if i not in filter_set:
                        cluster_ids[i] = 0

                if method_name != '':
                    print(prot_name2 + ' vs ' + prot_name1 + ' ' + method_name)
                    print_all_random_subgraphs(coords, cluster_ids, int2, filter_set, prot_name2 + ' vs ' + prot_name1 + ' ' + method_name, dist_f)
                else:
                    print(prot_name2 + ' vs ' + prot_name1)
                    print_all_random_subgraphs(coords, cluster_ids, int2, filter_set, prot_name2 + ' vs ' + prot_name1, dist_f)


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
    print_unified_intefaces_enc()
    # print_separate_intefaces()
    # print_unified_intefaces_aledo()
    # print_unified_intefaces_aledo_cytb_enc()
    # print_unified_intefaces_aledo_atp6()
    # print_unified_intefaces_aledo_atp6_enc()