from os import makedirs
from os.path import exists
from igraph import *
import numpy as np
import pandas as pd
import matplotlib
from scipy.stats import chisquare

from compute_interface_stat_rand_int_igraph import compute_graphs, parse_pdb_Aledo_biopython, dist_aledo

matplotlib.use('agg')
import matplotlib.pyplot as plt


from compute_cluster_stats import dist, parse_colors, parse_out, parse_site2pdb
from assimptotic_tests import parse_pdb, get_interface, read_cox_data

pdb_id = '1occ'
# pdb_id = '1bgy'
# pdb_id = '1be3'
# pdb_id = '5ara'
path_to_pdb = '../pdb/' + pdb_id + '.pdb'
path_to_cox_data = '../Coloring/COXdata.txt'
path_to_cytb_data = '../aledo.csv'
path_to_atp6_data = '../Coloring/atp6_5ara_Aledo_4ang.csv'
# path_to_atp6_data = '../Coloring/cytb_1bgy_Aledo_4ang.csv'
path_to_surf_racer_data = '../surf_racer/burried/1bgy.csv'
path_to_dssp_dir = '../dssp/'
path_to_dssp_data = path_to_dssp_dir + pdb_id + '.csv'
path_to_dssp_raw = path_to_dssp_dir + pdb_id + '.dssp'
path_to_colors = '../Coloring/internal_gaps.2/'

chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
# chain_to_prot = {'C': 'cytb'}
# chain_to_prot = {'W': 'atp6'}
# chain_to_prot = {'B': 'cox2'}
# chain_to_prot = {'C': 'cox3'}
prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}
# prot_to_chain = {'cytb': 'C'}
# prot_to_chain = {'atp6': 'W'}
# prot_to_chain = {'cox2': 'B'}
# prot_to_chain = {'cox3': 'C'}
aledo_dist = True
dist_threshold = 4

debug = True
only_selected_chains = False
only_mitochondria_to_nuclear = True
random_graph_stat_hist_path = '../res/random_graph_stat_hist_cox1_ABC_Aledo_igraph/'
random_graph_suffix = '_Aledo_igraph_enc_merged.random_graphs'
random_graph_path = '../res/random_graphs/'
permutations_num = 10000
reverse_shuffle = False

print_stat_hist = False
print_identity_hist = False
print_rejects_stat = False
print_jaccard_and_chi_stats = True


# def chi_sqr_old(pos_lists, interface_set, total_pos_num):
#     res = 0
#     p_int = len(interface_set)/total_pos_num
#     for pos_list in pos_lists:
#         c = 0
#         for pos in pos_list:
#             if pos in interface_set:
#                 c += 1
#         exp = len(pos_list)*p_int
#         res += (c - exp)*(c - exp)/exp
#     return total_pos_num*res


def chi_sqr(pos_lists, interface_set):
    res = 0
    in_interface = 0
    total_pos_num = 0
    for pos_list in pos_lists:
        for pos in pos_list:
            total_pos_num += 1
            if pos in interface_set:
                in_interface += 1
    p_int = in_interface/total_pos_num
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


def chi_with_given_p_int(cl_to_poses, interface_set, cl_to_p_int):
    res = 0
    in_interface = 0
    total_pos_num = 0
    for cl, pos_list in cl_to_poses.items():
        for pos in pos_list:
            total_pos_num += 1
            if pos in interface_set:
                in_interface += 1
    expected = []
    observed = []
    cl_stats = {}
    for cl, pos_list in cl_to_poses.items():
        c = 0
        for pos in pos_list:
            if pos in interface_set:
                c += 1
        observed.append(c)
        exp = cl_to_p_int[cl - 1]
        expected.append(exp)
        res += (c - exp)*(c - exp)/exp
        observed.append(len(pos_list) - c)
        res += (c - exp)*(c - exp)/(len(pos_list) - exp)
        expected.append(len(pos_list) - exp)
        cl_stats[cl] = (c - exp)*(c - exp)/exp + (c - exp)*(c - exp)/(len(pos_list) - exp)
    return res, observed, expected, cl_stats


def jaccard_index(cl_to_poses, interface_set):
    res = {}
    for cl, pos_list in cl_to_poses.items():
        intersection = 0
        for pos in pos_list:
            if pos in interface_set:
                intersection += 1
        res[cl] = intersection/(len(pos_list) + len(interface_set) - intersection)
    return res


def individual_chi(cl_to_poses, interface_set):
    res = {}
    total_pos_num = 0
    total_in_interface = 0
    for cl, pos_list in cl_to_poses.items():
        for pos in pos_list:
            total_pos_num += 1
            if pos in interface_set:
                total_in_interface += 1
    p_int = total_in_interface / total_pos_num
    for cl, pos_list in cl_to_poses.items():
        in_group = len(pos_list)
        p_group = in_group / total_pos_num
        in_int_in_group = 0
        for pos in pos_list:
            if pos in interface_set:
                in_int_in_group += 1
        p_obs = in_int_in_group/total_pos_num
        chi = (p_obs - p_group*p_int)**2/(p_group*p_int)
        p_obs = (in_group - in_int_in_group)/total_pos_num
        chi += (p_obs - p_group*(1 - p_int))**2/(p_group*(1 - p_int))
        p_obs = (total_in_interface - in_int_in_group) / total_pos_num
        chi += (p_obs - (1 - p_group)*p_int)**2/((1 - p_group)*p_int)
        p_obs = (total_pos_num - total_in_interface - in_group + in_int_in_group)/total_pos_num
        chi += (p_obs - (1 - p_group)*(1 - p_int))**2/((1 - p_group)*(1 - p_int))
        res[cl] = total_pos_num*chi
    return res


def individual_chi_with_given_p_int(cl_to_poses, interface_set, cl_to_p_int):
    res = {}
    total_pos_num = 0
    total_in_interface = 0
    for cl, pos_list in cl_to_poses.items():
        for pos in pos_list:
            total_pos_num += 1
            if pos in interface_set:
                total_in_interface += 1
    for cl, pos_list in cl_to_poses.items():
        in_group = len(pos_list)
        in_int_in_group = 0
        for pos in pos_list:
            if pos in interface_set:
                in_int_in_group += 1
        chi = (in_int_in_group - cl_to_p_int[cl - 1])**2/cl_to_p_int[cl - 1]
        obs = in_group - in_int_in_group
        exp = in_group - cl_to_p_int[cl - 1]
        chi += (obs - exp)**2/exp
        obs = total_in_interface - in_int_in_group
        exp = total_in_interface - cl_to_p_int[cl - 1]
        chi += (obs - exp)**2/exp
        obs = total_pos_num - total_in_interface - in_group + in_int_in_group
        exp = total_pos_num - total_in_interface - in_group + cl_to_p_int[cl - 1]
        chi += (obs - exp)**2/exp
        res[cl] = chi
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


def compute_stat_on_random_subgraphs(small_graphs, pos_lists, prot_name):
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
            chi_sqr_stat.append(chi_sqr_func(pos_lists, int_set))
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


def compute_stat_on_random_subgraphs_all_ind_chi(small_graphs, cl_to_poses, prot_name):
    chi_sqr_stat = []
    identity_stat = [[] for i in range(len(small_graphs))]
    chi_sqr_ind_stat = {cl: [] for cl in cl_to_poses.keys()}
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
            chi_sqr_stat.append(chi_sqr_func(cl_to_poses.values(), int_set))
            ind_stat = individual_chi(cl_to_poses, int_set)
            inter, noninter = contigency_table(cl_to_poses, int_set)
            for cl, c in inter.items():
                interface[cl].append(c)
            for cl, chi in ind_stat.items():
                chi_sqr_ind_stat[cl].append(chi)
            for j in range(len(sampled_graphs)):
                gr = set(random_graphs[j])
                g = set(sampled_graphs[j])
                identity_stat[shuffled_indices[j]].append(len(gr.intersection(g))/len(g))
    return chi_sqr_stat, chi_sqr_ind_stat, identity_stat, max_iter_stops, interface


def compute_stat_on_random_subgraphs_all_ind_chi_perm(small_graphs, cl_to_poses, prot_name, cl_to_p_int):
    chi_sqr_stat = []
    identity_stat = [[] for i in range(len(small_graphs))]
    chi_sqr_ind_stat = {cl: [] for cl in cl_to_poses.keys()}
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
            stat, observed, expected = chi_with_given_p_int(cl_to_poses, int_set, cl_to_p_int)
            chi_sqr_stat.append(stat)
            ind_stat = individual_chi_with_given_p_int(cl_to_poses, int_set, cl_to_p_int)
            inter, noninter = contigency_table(cl_to_poses, int_set)
            for cl, c in inter.items():
                interface[cl].append(c)
            for cl, chi in ind_stat.items():
                chi_sqr_ind_stat[cl].append(chi)
            for j in range(len(sampled_graphs)):
                gr = set(random_graphs[j])
                g = set(sampled_graphs[j])
                identity_stat[shuffled_indices[j]].append(len(gr.intersection(g))/len(g))
    return chi_sqr_stat, chi_sqr_ind_stat, identity_stat, max_iter_stops, interface


def compute_stat_on_random_subgraphs_all(small_graphs, cl_to_poses, prot_name):
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
            chi_sqr_stat.append(chi_sqr_func(cl_to_poses.values(), int_set))
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


def compute_stat_on_random_subgraphs_all_no_chi(small_graphs, cl_to_poses, prot_name):
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
    return jaccard_index_stat, identity_stat, max_iter_stops, interface


def compute_stat_on_random_subgraphs_chi(cl_to_poses, prot_name, cl_to_p_int):
    chi_sqr_stat = []
    cl_stats_array = []
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
            shuffled_indices = [int(i) for i in s[3].split(';')]
            if reverse_shuffle:
                shuffled_indices.reverse()
            int_set = set()
            for g in random_graphs:
                int_set.update(g)
            stat, observed, expected, cl_stats = chi_with_given_p_int(cl_to_poses, int_set, cl_to_p_int)
            chi_sqr_stat.append(stat)
            cl_stats_array.append(cl_stats)
    return chi_sqr_stat, cl_stats_array


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


chi_sqr_func = chi_sqr


def test_independence_chi_sqr(pos_to_coords, cluster_ids, interface, filter_set, prot_name, dist_f):

    cl_to_poses, filtered_poses_num, big_graph, small_graphs = compute_graphs(pos_to_coords, cluster_ids, interface,
                                                                              filter_set, dist_f)

    stat = chi_sqr_func(cl_to_poses.values(), interface)

    chi_sqr_stats, identities, max_iter_stops = compute_stat_on_random_subgraphs(small_graphs,
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


def test_independence_jaccard(pos_to_coords, cluster_ids, interface, filter_set, prot_name, dist_f):
    cl_to_poses, filtered_poses_num, big_graph, small_graphs = compute_graphs(pos_to_coords, cluster_ids, interface,
                                                                              filter_set, dist_f)

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


def test_independence_all(pos_to_coords, cluster_ids, interface, filter_set, prot_name, dist_f):
    cl_to_poses, filtered_poses_num, big_graph, small_graphs = compute_graphs(pos_to_coords, cluster_ids, interface,
                                                                              filter_set, dist_f)

    group_stat = chi_sqr_func(cl_to_poses.values(), interface)
    individual_stats = jaccard_index(cl_to_poses, interface)

    chi_sqr_stats, jaccard_index_stats, identities, max_iter_stops, inter = \
        compute_stat_on_random_subgraphs_all(small_graphs,cl_to_poses, prot_name)

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


def test_independence_all_chi(pos_to_coords, cluster_ids, interface, filter_set, prot_name, dist_f):
    cl_to_poses, filtered_poses_num, big_graph, small_graphs = compute_graphs(pos_to_coords, cluster_ids, interface,
                                                                              filter_set, dist_f)

    group_stat = chi_sqr_func(cl_to_poses.values(), interface)
    individual_stats = individual_chi(cl_to_poses, interface)

    chi_sqr_stats, chi_sqr_ind_stats, identities, max_iter_stops, inter = \
        compute_stat_on_random_subgraphs_all_ind_chi(small_graphs, cl_to_poses, prot_name)

    if print_identity_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_random_graphs_identity(small_graphs, identities, prot_name)
    if print_stat_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_chi_sqr_stat(chi_sqr_stats, group_stat, prot_name)
        print_jaccard_index_stat(chi_sqr_ind_stats, individual_stats, prot_name)
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
    sample_num = len(chi_sqr_stats)
    group_p_value = chi_bigger_count/sample_num
    individual_up_p_values = np.zeros(cl_num, dtype=float)
    individual_down_p_values = np.zeros(cl_num, dtype=float)
    expected_interface = np.zeros(cl_num, dtype=float)
    for cl, val in individual_stats.items():
        individual_bigger_count = 0
        individual_smaller_count = 0
        sum_inter = sum(inter[cl])
        for s in chi_sqr_ind_stats[cl]:
            if s >= val:
                individual_bigger_count += 1
            if s <= val:
                individual_smaller_count += 1
        expected_interface[cl - 1] = sum_inter/sample_num
        individual_up_p_values[cl - 1] = individual_bigger_count/sample_num
        individual_down_p_values[cl - 1] = individual_smaller_count/sample_num
    return group_p_value, individual_up_p_values, individual_down_p_values, expected_interface


def test_independence_all_modified_chi_assimptotic(pos_to_coords, cluster_ids, interface, filter_set, prot_name, dist_f):
    cl_to_poses, filtered_poses_num, big_graph, small_graphs = compute_graphs(pos_to_coords, cluster_ids, interface,
                                                                              filter_set, dist_f)
    individual_stats = jaccard_index(cl_to_poses, interface)

    jaccard_index_stats, identities, max_iter_stops, inter = \
        compute_stat_on_random_subgraphs_all_no_chi(small_graphs, cl_to_poses, prot_name)
    # group_stat = chi_sqr(list(cl_to_poses.values()), interface)
    if print_identity_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_random_graphs_identity(small_graphs, identities, prot_name)
    if print_stat_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        # print_chi_sqr_stat(chi_sqr_stats, group_stat, prot_name)
        print_jaccard_index_stat(jaccard_index_stats, individual_stats, prot_name)
    if print_rejects_stat:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_rejection_freqs(max_iter_stops, prot_name)

    cl_num = 0
    for id in cluster_ids:
        if id > cl_num:
            cl_num = id
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
    group_stat, observed, expected = chi_with_given_p_int(cl_to_poses, interface, expected_interface)
    chi_val, group_p_value = chisquare(observed, expected)
    # print('my chi=%1.2f their_chi=%1.2f' % (group_stat, chi_val))
    return group_p_value, individual_up_p_values, individual_down_p_values, expected_interface


def print_random_graphs_stats(individual_stats, chi_sqr_stat, cl_stats_array, prot_name):
    random_graph_num = len(chi_sqr_stat)
    res = [[] for i in range(random_graph_num)]
    for cl, val in individual_stats.items():
        for i in range(random_graph_num):
            res[i].append(str(val[i]))
            res[i].append(str(cl_stats_array[i][cl]))
    with open(random_graph_stat_hist_path + prot_name + '_random_graph_stats.csv', 'w') as fout:
        fout.write('graph\group')
        for cl, val in individual_stats.items():
            fout.write('\tg' + str(cl) + '_jaccard')
            fout.write('\tg' + str(cl) + '_chi')
        fout.write('\tgroup_chi\n')
        for i in range(random_graph_num):
            fout.write(str(i) + '\t')
            fout.write('\t'.join(res[i]))
            fout.write('\t' + str(chi_sqr_stat[i]) + '\n')


def test_independence_all_modified_chi_perm(pos_to_coords, cluster_ids, interface, filter_set, prot_name, dist_f):
    cl_to_poses, filtered_poses_num, big_graph, small_graphs = compute_graphs(pos_to_coords, cluster_ids, interface,
                                                                              filter_set, dist_f)
    individual_stats = jaccard_index(cl_to_poses, interface)

    jaccard_index_stats, identities, max_iter_stops, inter = \
        compute_stat_on_random_subgraphs_all_no_chi(small_graphs, cl_to_poses, prot_name)
    # group_stat = chi_sqr(list(cl_to_poses.values()), interface)
    if print_identity_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_random_graphs_identity(small_graphs, identities, prot_name)
    if print_stat_hist:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        # print_chi_sqr_stat(chi_sqr_stats, group_stat, prot_name)
        print_jaccard_index_stat(jaccard_index_stats, individual_stats, prot_name)
    if print_rejects_stat:
        if not exists(random_graph_stat_hist_path):
            makedirs(random_graph_stat_hist_path)
        print_rejection_freqs(max_iter_stops, prot_name)

    cl_num = 0
    for id in cluster_ids:
        if id > cl_num:
            cl_num = id
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
    group_stat, observed, expected, cl_stats = chi_with_given_p_int(cl_to_poses, interface, expected_interface)
    chi_sqr_stat, cl_stats_array = compute_stat_on_random_subgraphs_chi(cl_to_poses, prot_name, expected_interface)
    c = 0
    for stat in chi_sqr_stat:
        if stat > group_stat:
            c += 1
    group_p_value = c / len(chi_sqr_stat)
    if print_jaccard_and_chi_stats:
        print_random_graphs_stats(jaccard_index_stats, chi_sqr_stat, cl_stats_array, prot_name)
    return group_p_value, individual_up_p_values, individual_down_p_values, expected_interface


def test_independence_all_modified_chi_ind_chi_perm(pos_to_coords, cluster_ids, interface, filter_set, prot_name, dist_f):
    cl_to_poses, filtered_poses_num, big_graph, small_graphs = compute_graphs(pos_to_coords, cluster_ids, interface,
                                                                              filter_set, dist_f)


    jaccard_index_stats, identities, max_iter_stops, inter = \
        compute_stat_on_random_subgraphs_all_no_chi(small_graphs, cl_to_poses, prot_name)
    # group_stat = chi_sqr(list(cl_to_poses.values()), interface)


    cl_num = 0
    for id in cluster_ids:
        if id > cl_num:
            cl_num = id
    individual_up_p_values = np.zeros(cl_num, dtype=float)
    individual_down_p_values = np.zeros(cl_num, dtype=float)
    expected_interface = np.zeros(cl_num, dtype=float)
    for cl in range(1, cl_num + 1):
        sum_inter = sum(inter[cl])
        expected_interface[cl - 1] = sum_inter/len(inter[cl])

    chi_sqr_stats, chi_sqr_ind_stat, identity_stat, max_iter_stops, inter = \
        compute_stat_on_random_subgraphs_all_ind_chi_perm(small_graphs, cl_to_poses, prot_name, expected_interface)

    individual_stats = individual_chi_with_given_p_int(cl_to_poses, interface, expected_interface)
    group_stat, observed, expected = chi_with_given_p_int(cl_to_poses, interface, expected_interface)

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

    for cl, val in individual_stats.items():
        ind_bigger_count = 0
        ind_smaller_count = 0
        for s in chi_sqr_ind_stat[cl]:
            if s >= val:
                ind_bigger_count += 1
            if s <= val:
                ind_smaller_count += 1
        individual_up_p_values[cl - 1] = ind_bigger_count/len(jaccard_index_stats[cl])
        individual_down_p_values[cl - 1] = ind_smaller_count / len(jaccard_index_stats[cl])

    c = 0
    for stat in chi_sqr_stats:
        if stat > group_stat:
            c += 1
    group_p_value = c / len(chi_sqr_stats)
    return group_p_value, individual_up_p_values, individual_down_p_values, expected_interface


p_values_func = test_independence_all_modified_chi_perm#test_independence_jaccard


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
            p_value = p_values_func(coords, cluster_ids, int, filter_set, prot_name, dist_f)
        else:
            p_value = p_values_func(coords, cluster_ids, int, filter_set, prot_name + '.' + method_name, dist_f)
        print_table(cl_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)


def print_unified_intefaces_aledo():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
        chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
    prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    cox_data = pd.read_csv(path_to_cox_data, sep='\t', decimal='.')
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
        non_int_counts, int_counts = count(cluster_ids, interface, set(non_burried['Pos']))
        # p_value = p_values_func(coords, cluster_ids, interface, set(non_burried['Pos']), prot_name)
        # print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)
        group_p_value, individual_up_p_values, individual_down_p_values, individual_expected_vals = \
            p_values_func(coords, cluster_ids, interface, set(non_burried['Pos']), prot_name, dist_f)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', group_p_value,
                    individual_up_p_values, individual_down_p_values, individual_expected_vals,
                    'ожидаемо не в интерфейсе', 'ожидаемо в интерфейсе')


def print_unified_intefaces_aledo_cytb():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
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
            p_values_func(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name, dist_f)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', group_p_value,
                    individual_up_p_values, individual_down_p_values, individual_expected_vals, 'ожидаемо не в интерфейсе',
                    'ожидаемо в интерфейсе')


def print_unified_intefaces_aledo_cytb_enc():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
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
            p_values_func(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name, dist_f)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', group_p_value,
                    individual_up_p_values, individual_down_p_values, individual_expected_vals, 'ожидаемо не в интерфейсе',
                    'ожидаемо в интерфейсе')


def print_unified_intefaces_aledo_atp6():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
        chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
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
        non_int_counts, int_counts = count(cluster_ids, interface, set(non_burried['pos']))
        # p_value = p_values_func(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name)
        # print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)
        group_p_value, individual_up_p_values, individual_down_p_values, individual_expected_vals = \
            p_values_func(coords, cluster_ids, interface, set(non_burried['pos']), prot_name, dist_f)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', group_p_value,
                    individual_up_p_values, individual_down_p_values, individual_expected_vals, 'ожидаемо не в интерфейсе',
                    'ожидаемо в интерфейсе')


def print_unified_intefaces_aledo_atp6_enc():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
        chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot)
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
        non_int_counts, int_counts = count(cluster_ids, interface, set(non_burried['pos']))
        # p_value = p_values_func(coords, cluster_ids, interface, set(non_burried['ResidNr']), prot_name)
        # print_table(non_int_counts, int_counts, 'ENC_noninterf', 'CONT + ENC_interface', p_value)
        group_p_value, individual_up_p_values, individual_down_p_values, individual_expected_vals = \
            p_values_func(coords, cluster_ids, interface, set(non_burried['pos']), prot_name, dist_f)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', group_p_value,
                    individual_up_p_values, individual_down_p_values, individual_expected_vals, 'ожидаемо не в интерфейсе',
                    'ожидаемо в интерфейсе')


def print_unified_intefaces_enc():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
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
            p_values_func(coords, cluster_ids, interface, non_buried, prot_name, dist_f)
        print_table(non_int_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', group_p_value,
                    individual_up_p_values, individual_down_p_values, individual_expected_vals, 'ожидаемо не в интерфейсе',
                    'ожидаемо в интерфейсе')


def print_separate_intefaces():
    if aledo_dist:
        dist_f = dist_aledo
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, only_selected_chains, chain_to_prot)
    else:
        dist_f = dist
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
                    p_value = p_values_func(coords, cluster_ids, int1, filter_set, prot_name1 + ' vs ' + prot_name2 + ' ' + method_name, dist_f)
                else:
                    print(prot_name1 + ' vs ' + prot_name2)
                    p_value = p_values_func(coords, cluster_ids, int1, filter_set, prot_name1 + ' vs ' + prot_name2, dist_f)
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
                    p_value = p_values_func(coords, cluster_ids, int2, filter_set, prot_name2 + ' vs ' + prot_name1 + ' ' + method_name, dist_f)
                else:
                    print(prot_name2 + ' vs ' + prot_name1)
                    p_value = p_values_func(coords, cluster_ids, int2, filter_set, prot_name2 + ' vs ' + prot_name1, dist_f)
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
    print_unified_intefaces_enc()
    # print_separate_intefaces()
    # print_unified_intefaces_aledo()
    # print_unified_intefaces_aledo_cytb()
    # print_unified_intefaces_aledo_cytb_enc()
    # print_secondary_structure_enrichment()
    # print_unified_intefaces_aledo_atp6()
    # print_unified_intefaces_aledo_atp6_enc()
