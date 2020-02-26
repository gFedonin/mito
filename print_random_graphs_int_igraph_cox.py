from os import makedirs, cpu_count
from os.path import exists
from random import shuffle, random, randrange

from igraph import *

import numpy as np
import matplotlib

from print_random_graphs_int_igraph import parse_pdb_Aledo_biopython, gen_random_subgraph_new2

matplotlib.use('agg')

from sklearn.externals.joblib import Parallel, delayed

from compute_cluster_stats import dist, get_pdb_neighbors, dist_aledo
from assimptotic_tests import parse_pdb, get_interface, read_cox_data


pdb_id = '1occ'
path_to_pdb = '../pdb/' + pdb_id + '.pdb'
path_to_cox_data = '../Coloring/COXdata.txt'

chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}

prot1 = 'cox3'
prot2 = 'cox2'



aledo_dist = True
dist_threshold = 4
use_internal_contacts = True
use_external_contacts = False

debug = False
only_selected_chains = True
only_mitochondria_to_nuclear = False
random_graph_path = '../res/random_graph_ABC_ind_Aledo_igraph_fixed/'
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


def print_random_subgraphs(thread_id, big_graph, small_graphs, n, prot_name):
    iter_done = 0
    if exists(random_graph_path + prot_name + '/' + str(thread_id) + '.random_graphs'):
        with open(random_graph_path + prot_name + '/' + str(thread_id) + '.random_graphs', 'r') as f:
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
                random_graph = gen_random_subgraph_new2(g, target_node_num, target_edge_num, max_iter)
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
        with open(random_graph_path + prot_name + '/' + str(thread_id) + '.random_graphs', 'a') as f:
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


# def create_graph(pos_to_coords, poses, dist_f):
#     g = Graph()
#     g.add_vertices(len(poses))
#     g.vs['name'] = list(poses)
#     for i in range(len(poses)):
#         p_i = poses[i]
#         for j in range(i + 1, len(poses)):
#             p_j = poses[j]
#             if dist_f(pos_to_coords[p_i], pos_to_coords[p_j]) < dist_threshold:
#                 g.add_edges([(i, j)])
#     return g


def create_graph(prot_name, prot_to_chain, chain_to_site_coords, poses, dist_f, use_internal_contacts, use_external_contacts):
    neighbors = get_pdb_neighbors(prot_name, prot_to_chain, chain_to_site_coords, dist_f, use_internal_contacts, use_external_contacts)
    g = Graph()
    g.add_vertices(len(poses))
    g.vs['name'] = list(poses)
    for i in range(len(poses)):
        p_i = poses[i]
        n_set = set(neighbors[p_i])
        for j in range(i + 1, len(poses)):
            p_j = poses[j]
            if p_j in n_set:
                g.add_edges([(i, j)])
    return g


def compute_graphs(prot_name, prot_to_chain, chain_to_site_coords, interface, filter_set, dist_f, use_internal_contacts, use_external_contacts):
    if debug:
        print('computing p_value')
    filtered_poses = list(filter_set)
    big_graph = create_graph(prot_name, prot_to_chain, chain_to_site_coords, filtered_poses, dist_f, use_internal_contacts, use_external_contacts)

    if debug:
        connected_comps = big_graph.components().subgraphs()
        print('big graph:')
        lens = [str(comp.vcount()) for comp in connected_comps]
        print('connected comp lens: ' + ' '.join(lens))

    # interface_names = set(str(p) for p in interface)
    int_list = big_graph.vs.select(name_in=interface)
    interface_graph = big_graph.subgraph(int_list)
    small_graphs = interface_graph.components().subgraphs()
    if debug:
        print('interface:')
        lens = [str(comp.vcount()) for comp in small_graphs]
        print('connected comp lens: ' + ' '.join(lens))
    return len(filtered_poses), big_graph, small_graphs


def print_all_random_subgraphs(prot_name, chain_to_site_coords, interface, filter_set, dist_f, folder_name):
    filtered_poses_num, big_graph, small_graphs = compute_graphs(prot_name, prot_to_chain, chain_to_site_coords,
                                                                 interface, filter_set, dist_f, use_internal_contacts, use_external_contacts)

    iter_nums = []
    for i in range(thread_num):
        iter_nums.append(permutations_num//thread_num)
    for i in range(permutations_num%thread_num):
        iter_nums[i] += 1
    if not exists(random_graph_path + folder_name):
        makedirs(random_graph_path + folder_name)
    tasks = Parallel(n_jobs=thread_num)(delayed(print_random_subgraphs)(i, big_graph, small_graphs,
                                                                        iter_nums[i], folder_name)
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


def print_separate_intefaces():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
        chain_to_site_coords = parse_pdb(path_to_pdb + pdb_id, only_selected_chains, chain_to_prot)
    prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    # prot_to_buried, prot_to_non_buried = read_dssp_data()
    coords1 = chain_to_site_coords[prot_to_chain[prot1]]
    coords2 = chain_to_site_coords[prot_to_chain[prot2]]

    non_burried1 = prot_to_non_buried[prot1]
    non_burried2 = prot_to_non_buried[prot2]
    int1, int2 = get_interface(coords1, coords2, dist_f, dist_threshold, non_burried1, non_burried2)
    if len(int1) == 0:
        return
    filter_set = non_burried1
    print(prot1 + ' vs ' + prot2)
    print_all_random_subgraphs(prot1, chain_to_site_coords, int1, filter_set, dist_f, prot1 + ' vs ' + prot2)


def print_interface_pos_in_file():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
        chain_to_site_coords = parse_pdb(path_to_pdb + pdb_id, only_selected_chains, chain_to_prot)
    prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    # prot_to_buried, prot_to_non_buried = read_dssp_data()
    coords1 = chain_to_site_coords[prot_to_chain[prot1]]
    coords2 = chain_to_site_coords[prot_to_chain[prot2]]

    non_burried1 = prot_to_non_buried[prot1]
    non_burried2 = prot_to_non_buried[prot2]
    int1, int2 = get_interface(coords1, coords2, dist_f, dist_threshold, non_burried1, non_burried2)
    if len(int1) == 0:
        return
    with open('../res/' + prot1 + '_' + prot2 + '.interface', 'w') as f:
        for pos in int1:
            f.write(str(pos) + '\n')


if __name__ == '__main__':
    print_separate_intefaces()
    # print_interface_pos_in_file()