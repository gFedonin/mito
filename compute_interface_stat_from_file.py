from os import makedirs
from os.path import exists
from igraph import *
import numpy as np
import pandas as pd
import matplotlib


matplotlib.use('agg')
import matplotlib.pyplot as plt


from compute_cluster_stats import dist, parse_colors, parse_out, parse_site2pdb
from assimptotic_tests import parse_pdb, get_interface, read_cox_data

pdb_id = '1occ'
# path_to_pdb = '../pdb/' + pdb_id + '.pdb1'
path_to_pdb = '../pdb/1be3.pdb1'
path_to_cox_data = '../Coloring/COXdata.txt'
path_to_cytb_data = '../aledo.csv'
path_to_surf_racer_data = '../surf_racer/burried/1bgy.csv'
path_to_dssp_dir = '../dssp/'
path_to_dssp_data = path_to_dssp_dir + pdb_id + '.csv'
path_to_dssp_raw = path_to_dssp_dir + pdb_id + '.dssp'
path_to_colors = '../Coloring/internal_gaps.2/'

# chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
chain_to_prot = {'C': 'cytb'}
# prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}
prot_to_chain = {'cytb': 'C'}
dist_threshold = 8

debug = True
only_selected_chains = False
only_mitochondria_to_nuclear = True
random_graph_stat_hist_path = '../res/random_graph_stat_hist_Aledo_igraph_cytb_enc_rep/'
random_graph_suffix = '_Aledo_igraph_enc_merged.random_graphs'
random_graph_path = '../res/'
permutations_num = 10000
reverse_shuffle = False

print_stat_hist = False
print_identity_hist = False
print_rejects_stat = False


def chi_sqr(pos_lists, interface_set, total_pos_num):
    res = 0
    p_int = len(interface_set)/total_pos_num
    for pos_list in pos_lists:
        c = 0
        for pos in pos_list:
            if pos in interface_set:
                c += 1
        exp = len(pos_list)*p_int
        res += (c - exp)*(c - exp)/exp
    return total_pos_num*res


def chi_sqr_ind(pos_lists, interface_set, total_pos_num):
    res = 0
    p_int = len(interface_set)/total_pos_num
    for pos_list in pos_lists:
        c = 0
        for pos in pos_list:
            if pos in interface_set:
                c += 1
        p_obs = c/total_pos_num
        p_group = len(pos_list)/total_pos_num
        res += (p_obs - p_group*p_int)**2/(p_group*p_int)
        p_obs = (len(pos_list) - c)/total_pos_num
        res += (p_obs - p_group*(1 - p_int))**2/(p_group*(1 - p_int))
    return total_pos_num*res


def jaccard_index(cl_to_poses, interface_set):
    res = {}
    for cl, pos_list in cl_to_poses.items():
        intersection = 0
        for pos in pos_list:
            if pos in interface_set:
                intersection += 1
        res[cl] = intersection/(len(pos_list) + len(interface_set) - intersection)
    return res


def contigency_table(cl_to_poses, interface_set):
    interface = {}
    noninterface = {}
    for cl, pos_list in cl_to_poses.items():
        c_in = 0
        c_out = 0
        for pos in pos_list:
            if pos in interface_set:
                c_in += 1
            else:
                c_out += 1
        interface[cl] = c_in
        noninterface[cl] = c_out
    return interface, noninterface


def compute_stat_on_random_subgraphs(big_graph, small_graphs, pos_lists, prot_name):
    chi_sqr_stat = []
    identity_stat = [[] for i in range(len(small_graphs))]
    max_iter_stops = []
    with open(random_graph_path + prot_name + random_graph_suffix, 'r') as f:
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
            max_iter_stops.append(int(s[2]))
            shuffled_indices = [int(i) for i in s[3].split(';')]
            if reverse_shuffle:
                shuffled_indices.reverse()
            int_set = set()
            for g in random_graphs:
                int_set.update(g)
            chi_sqr_stat.append(chi_sqr_func(pos_lists, int_set, big_graph.vcount()))
            for j in range(len(sampled_graphs)):
                gr = set(random_graphs[j])
                g = set(sampled_graphs[j])
                identity_stat[shuffled_indices[j]].append(len(gr.intersection(g))/len(g))
    return chi_sqr_stat, identity_stat, max_iter_stops


def compute_stat_on_random_subgraphs_for_each_group(small_graphs, cl_to_poses, prot_name):
    jaccard_index_stat = {cl: [] for cl in cl_to_poses.keys()}
    identity_stat = [[] for i in range(len(small_graphs))]
    max_iter_stops = []
    with open(random_graph_path + prot_name + random_graph_suffix, 'r') as f:
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
            max_iter_stops.append(int(s[2]))
            shuffled_indices = [int(i) for i in s[3].split(';')]
            if reverse_shuffle:
                shuffled_indices.reverse()
            int_set = set()
            for g in random_graphs:
                int_set.update(g)
            stat = jaccard_index(cl_to_poses, int_set)
            for cl, jaccard in stat.items():
                jaccard_index_stat[cl].append(jaccard)
            for j in range(len(sampled_graphs)):
                gr = set(random_graphs[j])
                g = set(sampled_graphs[j])
                identity_stat[shuffled_indices[j]].append(len(gr.intersection(g))/len(g))
    return jaccard_index_stat, identity_stat, max_iter_stops


def compute_stat_on_random_subgraphs_all(big_graph, small_graphs, cl_to_poses, prot_name):
    chi_sqr_stat = []
    identity_stat = [[] for i in range(len(small_graphs))]
    jaccard_index_stat = {cl: [] for cl in cl_to_poses.keys()}
    max_iter_stops = []
    interface = {cl: [] for cl in cl_to_poses.keys()}
    with open(random_graph_path + prot_name + random_graph_suffix, 'r') as f:
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
            max_iter_stops.append(int(s[2]))
            shuffled_indices = [int(i) for i in s[3].split(';')]
            if reverse_shuffle:
                shuffled_indices.reverse()
            int_set = set()
            for g in random_graphs:
                int_set.update(g)
            chi_sqr_stat.append(chi_sqr_func(cl_to_poses.values(), int_set, big_graph.vcount()))
            stat = jaccard_index(cl_to_poses, int_set)
            inter, noninter = contigency_table(cl_to_poses, int_set)
            for cl, c in inter.items():
                interface[cl].append(c)
            for cl, jaccard in stat.items():
                jaccard_index_stat[cl].append(jaccard)
            for j in range(len(sampled_graphs)):
                gr = set(random_graphs[j])
                g = set(sampled_graphs[j])
                identity_stat[shuffled_indices[j]].append(len(gr.intersection(g))/len(g))
    return chi_sqr_stat, jaccard_index_stat, identity_stat, max_iter_stops, interface


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


def compute_graphs(pos_to_coords, cluster_ids, interface, filter_set):
    if debug:
        print('computing p_value')
    filtered_poses = list(filter_set)
    big_graph = create_graph(pos_to_coords, filtered_poses)

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


def print_random_graphs_identity(small_graphs, identities, prot_name):
    for i in range(len(small_graphs)):
        plt.title('Identity of random graphs to interface graph component with %d nodes' % len(small_graphs[i].vs))
        plt.xlabel('Identity')
        plt.ylabel('Proportion of graphs')
        weights = np.ones_like(identities[i]) / float(len(identities[i]))
        plt.hist(identities[i], 50, weights=weights, facecolor='g', alpha=0.75)
        # n, bins, patches = plt.hist(identities[i], 50, density=True, facecolor='g', alpha=0.75)#
        # plt.axis([0, 0.002, 0, 6000])
        plt.savefig(random_graph_stat_hist_path + prot_name + '_' + str(i) + '.png')
        plt.clf()


def print_chi_sqr_stat(chi_sqr_stats, stat, prot_name):
    plt.title('ChiSqr statistics of random graphs, our stat = %1.1f' % stat)
    plt.xlabel('ChiSqr value')
    plt.ylabel('Proportion of graphs')
    weights = np.ones_like(chi_sqr_stats) / float(len(chi_sqr_stats))
    plt.hist(chi_sqr_stats, 50, weights=weights, facecolor='g', alpha=0.75)
    # n, bins, patches = plt.hist(chi_sqr_stats, 50, density=True, facecolor='g', alpha=0.75)
    # plt.axis([0, 0.002, 0, 6000])
    plt.savefig(random_graph_stat_hist_path + prot_name + '_chi.png')
    plt.clf()


def print_jaccard_index_stat(jaccard_index_stats, stat, prot_name):
    for cl, jaccard in stat.items():
        plt.title('Jaccard statistics of random graphs, group %d, our stat = %1.1f' % (cl, jaccard))
        plt.xlabel('Jaccard index')
        plt.ylabel('Proportion of graphs')
        weights = np.ones_like(jaccard_index_stats[cl]) / float(len(jaccard_index_stats[cl]))
        plt.hist(jaccard_index_stats[cl], 50, weights=weights, facecolor='g', alpha=0.75)
        plt.savefig(random_graph_stat_hist_path + prot_name + '_' + str(cl) + '_jaccard.png')
        plt.clf()


def print_rejection_freqs(max_iter_stops, prot_name):
    plt.title('Frequency of rejects during generation of random graphs')
    plt.xlabel('Number of rejects')
    plt.ylabel('Proportion of graphs')
    weights = np.ones_like(max_iter_stops) / float(len(max_iter_stops))
    plt.hist(max_iter_stops, 50, weights=weights, facecolor='g', alpha=0.75)
    # n, bins, patches = plt.hist(max_iter_stops, 50, density=True, facecolor='g', alpha=0.75)
    # plt.axis([0, 0.002, 0, 6000])
    plt.savefig(random_graph_stat_hist_path + prot_name + '_rejects.png')
    plt.clf()


chi_sqr_func = chi_sqr_ind


def test_independence_chi_sqr(pos_to_coords, cluster_ids, interface, filter_set, prot_name):
    cl_to_poses, filtered_poses_num, big_graph, small_graphs = compute_graphs(pos_to_coords, cluster_ids, interface, filter_set)

    stat = chi_sqr_func(cl_to_poses.values(), interface, filtered_poses_num)

    chi_sqr_stats, identities, max_iter_stops = compute_stat_on_random_subgraphs(big_graph, small_graphs,
                                                                                 list(cl_to_poses.values()), prot_name)

    if print_identity_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_random_graphs_identity(small_graphs, identities, prot_name)
    if print_stat_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_chi_sqr_stat(chi_sqr_stats, stat, prot_name)
    if print_rejects_stat:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_rejection_freqs(max_iter_stops, prot_name)

    i = 0
    for s in chi_sqr_stats:
        if s >= stat:
            i += 1
    return i/len(chi_sqr_stats)


def test_independence_jaccard(pos_to_coords, cluster_ids, interface, filter_set, prot_name):
    cl_to_poses, filtered_poses_num, big_graph, small_graphs = compute_graphs(pos_to_coords, cluster_ids, interface,
                                                                              filter_set)

    stat = jaccard_index(cl_to_poses, interface)

    jaccard_index_stats, identities, max_iter_stops = compute_stat_on_random_subgraphs_for_each_group(small_graphs,
                                                                                 cl_to_poses, prot_name)

    if print_identity_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_random_graphs_identity(small_graphs, identities, prot_name)
    if print_stat_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_jaccard_index_stat(jaccard_index_stats, stat, prot_name)
    if print_rejects_stat:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_rejection_freqs(max_iter_stops, prot_name)

    cl_num = 0
    for id in cluster_ids:
        if id > cl_num:
            cl_num = id
    p_values = np.zeros(cl_num, dtype=float)
    for cl, val in stat.items():
        i = 0
        for s in jaccard_index_stats[cl]:
            if s >= val:
                i += 1
        p_values[cl - 1] = i/len(jaccard_index_stats[cl])
    return p_values


def test_independence_all(pos_to_coords, cluster_ids, interface, filter_set, prot_name):
    cl_to_poses, filtered_poses_num, big_graph, small_graphs = compute_graphs(pos_to_coords, cluster_ids, interface, filter_set)

    group_stat = chi_sqr_func(cl_to_poses.values(), interface, filtered_poses_num)
    individual_stats = jaccard_index(cl_to_poses, interface)

    chi_sqr_stats, jaccard_index_stats, identities, max_iter_stops, inter = compute_stat_on_random_subgraphs_all(big_graph,
                                                                                                          small_graphs,
                                                                                 cl_to_poses, prot_name)

    if print_identity_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_random_graphs_identity(small_graphs, identities, prot_name)
    if print_stat_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_chi_sqr_stat(chi_sqr_stats, group_stat, prot_name)
        print_jaccard_index_stat(jaccard_index_stats, individual_stats, prot_name)
    if print_rejects_stat:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_rejection_freqs(max_iter_stops, prot_name)

    chi_bigger_count = 0
    for s in chi_sqr_stats:
        if s >= group_stat:
            chi_bigger_count += 1
    cl_num = 0
    for id in cluster_ids:
        if id > cl_num:
            cl_num = id
    group_p_value = chi_bigger_count/len(chi_sqr_stats)
    individual_up_p_values = np.zeros(cl_num, dtype=float)
    individual_down_p_values = np.zeros(cl_num, dtype=float)
    expected_interface = np.zeros(cl_num, dtype=float)
    for cl, val in individual_stats.items():
        jaccard_bigger_count = 0
        jaccard_smaller_count = 0
        sum_inter = sum(inter[cl])
        for s in jaccard_index_stats[cl]:
            if s >= val:
                jaccard_bigger_count += 1
            if s <= val:
                jaccard_smaller_count += 1
        expected_interface[cl - 1] = sum_inter/len(jaccard_index_stats[cl])
        individual_up_p_values[cl - 1] = jaccard_bigger_count/len(jaccard_index_stats[cl])
        individual_down_p_values[cl - 1] = jaccard_smaller_count / len(jaccard_index_stats[cl])
    return group_p_value, individual_up_p_values, individual_down_p_values, expected_interface


p_values_func = test_independence_chi_sqr#test_independence_jaccard


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
        cl_counts, int_counts = count(cluster_ids, int, filter_set)
        if method_name != '':
            p_value = p_values_func(coords, cluster_ids, int, filter_set, prot_name)
        else:
            p_value = p_values_func(coords, cluster_ids, int, filter_set, prot_name + '.' + method_name)
        print_table(cl_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)


def print_unified_intefaces_aledo():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
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
        # p_value = p_values_func(coords, cluster_ids, interface, set(non_burried['Pos']), prot_name)
        # print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)
        group_p_value, individual_up_p_values, individual_down_p_values, individual_expected_vals = \
            test_independence_all(coords, cluster_ids, interface, set(non_burried['Pos']), prot_name)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', group_p_value,
                    individual_up_p_values, individual_down_p_values, individual_expected_vals, 'ожидаемо не в интерфейсе',
                    'ожидаемо в интерфейсе')


def print_unified_intefaces_aledo_cytb():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
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
        # p_value = p_values_func(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name)
        # print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)
        group_p_value, individual_up_p_values, individual_down_p_values, individual_expected_vals = \
            test_independence_all(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', group_p_value,
                    individual_up_p_values, individual_down_p_values, individual_expected_vals, 'ожидаемо не в интерфейсе',
                    'ожидаемо в интерфейсе')


def print_unified_intefaces_aledo_cytb_enc():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
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
        # p_value = p_values_func(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name)
        # print_table(non_int_counts, int_counts, 'ENC_noninterf', 'CONT + ENC_interface', p_value)
        group_p_value, individual_up_p_values, individual_down_p_values, individual_expected_vals = \
            test_independence_all(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', group_p_value,
                    individual_up_p_values, individual_down_p_values, individual_expected_vals, 'ожидаемо не в интерфейсе',
                    'ожидаемо в интерфейсе')


def print_unified_intefaces_enc():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
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
        # p_value = p_values_func(coords, cluster_ids, interface, non_buried, prot_name)
        # print_table(non_int_counts, int_counts, 'ENC_noninterf', 'CONT + ENC_interface', p_value)
        group_p_value, individual_up_p_values, individual_down_p_values, individual_expected_vals = \
            test_independence_all(coords, cluster_ids, interface, non_buried, prot_name)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', group_p_value,
                    individual_up_p_values, individual_down_p_values, individual_expected_vals, 'ожидаемо не в интерфейсе',
                    'ожидаемо в интерфейсе')


def print_separate_intefaces():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
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
                    p_value = p_values_func(coords, cluster_ids, int1, filter_set, prot_name1 + ' vs ' + prot_name2 + ' ' + method_name)
                else:
                    print(prot_name1 + ' vs ' + prot_name2)
                    p_value = p_values_func(coords, cluster_ids, int1, filter_set, prot_name1 + ' vs ' + prot_name2)
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
                    p_value = p_values_func(coords, cluster_ids, int2, filter_set, prot_name2 + ' vs ' + prot_name1 + ' ' + method_name)
                else:
                    print(prot_name2 + ' vs ' + prot_name1)
                    p_value = p_values_func(coords, cluster_ids, int2, filter_set, prot_name2 + ' vs ' + prot_name1)
                print_table(cl_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)


def print_table(cl_counts, int_counts, label1, label2, total_p_value=None, individual_up_p_values=None,
                individual_down_p_values=None, individual_expected_vals=None, expected_label1=None, expected_label2=None):
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
    if expected_label1 is not None:
        print(expected_label1, end='')
        i = 0
        for val in individual_expected_vals:
            print('\t%1.1f' % (int_counts[i] + cl_counts[i] - val), end='')
            i += 1
        print()
    if expected_label2 is not None:
        print(expected_label2, end='')
        for val in individual_expected_vals:
            print('\t%1.1f' % val, end='')
        print()
    if individual_up_p_values is not None:
        print('upper_p_value', end='')
        for val in individual_up_p_values:
            print('\t%1.4f' % val, end='')
        print()
    if individual_down_p_values is not None:
        print('lower_p_value', end='')
        for val in individual_down_p_values:
            print('\t%1.4f' % val, end='')
        print()
    if total_p_value is not None:
        print('total p_value = %1.4f\n' % total_p_value)



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


def print_secondary_structure_enrichment():
    chain_to_ss = parse_dssp()
    prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name)
        else:
            print(prot_name + '.' + method_name)
        non_burried = prot_to_non_buried[prot_name]
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
            non_int_counts, int_counts = count(cluster_ids, pos_set, non_burried)
            p_value = p_values_func(coords, cluster_ids, pos_set, non_burried, prot_name)
            print_table(non_int_counts, int_counts, 'not ' + ss, ss, p_value)


if __name__ == '__main__':
    # print_unified_intefaces()
    # print_unified_intefaces_enc()
    # print_separate_intefaces()
    # print_unified_intefaces_aledo()
    # print_unified_intefaces_aledo_cytb()
    print_unified_intefaces_aledo_cytb_enc()
    # print_secondary_structure_enrichment()
