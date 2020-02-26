import os
from random import shuffle

from os.path import exists

import numpy as np
import math

from scipy.stats import entropy
from sklearn.externals.joblib import Parallel, delayed

from compute_cluster_stats import parse_colors, parse_out, parse_site2pdb

pdb_id = '1occ'
path_to_pdb = '../pdb/' + pdb_id + '.pdb1'
path_to_colors = '../Coloring/internal_gaps.2/'
path_to_cox_data = '../Coloring/COXdata.txt'
path_to_dssp_dir = '../dssp/'
path_to_dssp_data = path_to_dssp_dir + pdb_id + '.csv'
path_to_dssp_raw = path_to_dssp_dir + pdb_id + '.dssp'

chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}
dist_threshold = 8
reduced_ss_states = {'H':'H', 'G':'H', 'I':'H', 'E':'E', 'B':'E', 'S':'C', 'C':'C', 'T':'C'}

only_selected_chains = True
print_contingency = True
only_mitochondria_to_nuclear = False
only_non_burried = False
use_reduced_ss_states = True
iter_num = 10000
thread_num = 8

out_path = '../fisher/ABC_internal_gaps.2_ss/'

use_colors = False


def read_cox_data(path_to_cox_data):
    prot_to_non_buried = {}
    prot_to_non_interface = {}
    prot_to_buried = {}
    for prot_name in prot_to_chain.keys():
        prot_to_non_buried[prot_name] = set()
        prot_to_non_interface[prot_name] = set()
        prot_to_buried[prot_name] = set()
    with open(path_to_cox_data, 'r') as f:
        f.readline()
        for line in f.readlines():
            s = line.strip().split('\t')
            if s[-3] == 'BURIED':
                prot_to_buried[chain_to_prot[s[7]]].add(int(s[0]))
            else:
                prot_to_non_buried[chain_to_prot[s[7]]].add(int(s[0]))
                if s[-3] == 'ENC_noninterf':
                    prot_to_non_interface[chain_to_prot[s[7]]].add(int(s[0]))
    return prot_to_buried, prot_to_non_buried, prot_to_non_interface


def read_dssp_data():
    prot_to_buried = {}
    prot_to_non_buried = {}
    for prot_name in prot_to_chain.keys():
        prot_to_non_buried[prot_name] = set()
        prot_to_buried[prot_name] = set()
    with open(path_to_dssp_data, 'r') as f:
        f.readline()
        for line in f.readlines():
            s = line.strip().split('\t')
            if s[3] == '1':
                prot_to_buried[chain_to_prot[s[0]]].add(int(s[1]))
            else:
                prot_to_non_buried[chain_to_prot[s[0]]].add(int(s[1]))
    return prot_to_buried, prot_to_non_buried


def parse_pdb(path_to_pdb, only_selected_chains, chain_to_prot):
    chain_to_site_coords = {}
    with open(path_to_pdb, 'r') as f:
        curr_chain = ''
        pos_to_coords = {}
        for line in f.readlines():
            s = line.split()
            if s[0] == 'ATOM' and s[2] == 'CA':
                chain = s[4]
                if not only_selected_chains or chain in chain_to_prot.keys():
                    if curr_chain != '':
                        if chain != curr_chain:
                            chain_to_site_coords[curr_chain] = pos_to_coords
                            curr_chain = chain
                            pos_to_coords = {}
                    else:
                        curr_chain = chain
                    pos_to_coords[int(s[5])] = (float(s[6]), float(s[7]), float(s[8]))
        chain_to_site_coords[curr_chain] = pos_to_coords
    return chain_to_site_coords


def dist(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)


def get_interface(pos_to_coords1, pos_to_coords2, dist_f, dist_threshold, filter_set1=None, filter_set2=None):
    pos_to_c1 = []
    for p, c in pos_to_coords1.items():
        if filter_set1 is None or p in filter_set1:
            pos_to_c1.append((p, c))
    pos_to_c2 = []
    for p, c in pos_to_coords2.items():
        if filter_set2 is None or p in filter_set2:
            pos_to_c2.append((p, c))
    interface1 = set()
    interface2 = set()
    for p1, c1 in pos_to_c1:
        for p2, c2 in pos_to_c2:
            if dist_f(c1, c2) < dist_threshold:
                interface1.add(p1)
                interface2.add(p2)
    return interface1, interface2


def count(cluster_ids, interface, filter_set=None):
    cl_num = 0
    for id in cluster_ids:
        if id > cl_num:
            cl_num = id
    cl_counts = np.zeros(cl_num, dtype=int)
    int_counts = np.zeros(cl_num, dtype=int)
    for pos in range(len(cluster_ids)):
        cl = cluster_ids[pos]
        if cl != 0:
            if pos in interface:
                int_counts[cl - 1] += 1
            else:
                if print_contingency:
                    if filter_set is None or pos in filter_set:
                        cl_counts[cl - 1] += 1
            if not print_contingency:
                if filter_set is None or pos in filter_set:
                    cl_counts[cl - 1] += 1
    return cl_counts, int_counts


def group_entropy(cluster_ids, group1, group2):
    cl_num = 0
    for id in cluster_ids:
        if id > cl_num:
            cl_num = id
    group1_counts = np.zeros(cl_num, dtype=int)
    group2_counts = np.zeros(cl_num, dtype=int)
    for pos in range(len(cluster_ids)):
        cl = cluster_ids[pos]
        if cl != 0:
            if pos in group1:
                group1_counts[cl - 1] += 1
            else:
                if pos in group2:
                    group2_counts[cl - 1] += 1
    return entropy(group1_counts), entropy(group2_counts)


def print_table(path, cl_counts, int_counts, label1, label2):
    with open(path, 'w') as f:
        count_list = []
        group_num = []
        for c in cl_counts:
            group_num.append(str(len(group_num) + 1))
            count_list.append(str(c))
        print('номер группы\t' + '\t'.join(group_num))
        print(label1 + '\t' + '\t'.join(count_list))
        f.write('\t'.join(count_list) + '\n')
        count_list = []
        for c in int_counts:
            count_list.append(str(c))
        print(label2 + '\t' + '\t'.join(count_list) + '\n\n')
        f.write('\t'.join(count_list) + '\n')


def print_ss_table(cl_ss_counts, cl_num, ss_types, path=None):
    if path is not None:
        f = open(path, 'w')
    group_num = []
    for c in range(1, cl_num + 1):
        group_num.append(str(c))
    print('ss\group_id\t' + '\t'.join(group_num))
    for ss in ss_types:
        print(ss, end='')
        for c in range(1, cl_num + 1):
            ss_counts = cl_ss_counts[c]
            print('\t' + str(ss_counts[ss]), end='')
        if path is not None:
            ss_counts = cl_ss_counts[1]
            f.write(str(ss_counts[ss]))
            for c in range(2, cl_num + 1):
                ss_counts = cl_ss_counts[c]
                f.write('\t' + str(ss_counts[ss]))
            f.write('\n')
        print()
    print()
    print()
    if path is not None:
        f.close()


def print_unified_intefaces():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    if only_non_burried:
        prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
        # prot_to_buried, prot_to_non_buried = read_dssp_data()
    interfaces = {}
    for prot_name1, chain1 in prot_to_chain.items():
        coords1 = chain_to_site_coords[chain1]
        if only_non_burried:
            non_burried1 = prot_to_non_buried[prot_name1]
        for chain2, coords2 in chain_to_site_coords.items():
            prot_name2 = chain_to_prot.get(chain2)
            if prot_name2 is not None:
                if only_non_burried:
                    non_burried2 = prot_to_non_buried[prot_name2]
                if only_mitochondria_to_nuclear:
                    continue
                if prot_name1 < prot_name2:
                    if only_non_burried:
                        int1, int2 = get_interface(coords1, coords2, non_burried1, non_burried2)
                    else:
                        int1, int2 = get_interface(coords1, coords2)
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
                if only_non_burried:
                    int1, int2 = get_interface(coords1, coords2, non_burried1)
                else:
                    int1, int2 = get_interface(coords1, coords2)
                l = interfaces.get(prot_name1)
                if l is not None:
                    l.update(int1)
                else:
                    interfaces[prot_name1] = int1
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        print(prot_name + '.' + method_name)
        int = interfaces[prot_name]
        # cluster_ids = chain_to_clusters[prot_name]
        if only_non_burried:
            cl_counts, int_counts = count(cluster_ids, int, prot_to_non_buried[prot_name])
        else:
            cl_counts, int_counts = count(cluster_ids, int)
        if method_name != '':
            print_table(out_path + prot_name + '_' + method_name + '.txt', cl_counts, int_counts, 'не в интерфейсе',
                        'в интерфейсе')
        else:
            print_table(out_path + prot_name + '.txt', cl_counts, int_counts, 'не в интерфейсе',
                        'в интерфейсе')


def print_unified_intefaces_enc():
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    if only_non_burried:
        prot_to_burried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        print(prot_name + ' ' + method_name)
        non_buried = prot_to_non_buried[prot_name]
        non_interface = prot_to_non_interface[prot_name]
        # cluster_ids = chain_to_clusters[prot_name]
        cl_counts, int_counts = count(cluster_ids, non_interface, non_buried)
        if method_name != '':
            print_table(out_path + prot_name + '_' + method_name + '.txt', cl_counts, int_counts, 'ENC_noninterf',
                        'CONT + ENC_interface')
        else:
            print_table(out_path + prot_name + '.txt', cl_counts, int_counts, 'ENC_noninterf',
                        'CONT + ENC_interface')


def print_unified_intefaces_enc_burried():
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        print(prot_name + ' ' + method_name)
        buried = prot_to_buried[prot_name]
        non_buried = prot_to_non_buried[prot_name]
        non_interface = prot_to_non_interface[prot_name]
        buried_and_non_interface = set()
        non_buried_interface = set()
        for site in buried:
            buried_and_non_interface.add(site)
        for site in non_interface:
            buried_and_non_interface.add(site)
        for site in non_buried:
            if site not in non_interface:
                non_buried_interface.add(site)
        # cluster_ids = chain_to_clusters[prot_name]
        cl_counts, int_counts = count(cluster_ids, buried_and_non_interface, non_buried_interface)
        if method_name != '':
            print_table(out_path + prot_name + '_' + method_name + '.txt', cl_counts, int_counts,
                        'Buried + ENC_noninterf', 'CONT + ENC_interface')
        else:
            print_table(out_path + prot_name + '.txt', cl_counts, int_counts,
                        'Buried + ENC_noninterf', 'CONT + ENC_interface')


def print_unified_intefaces_enc_burried_entropy():
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        print(prot_name + ' ' + method_name)
        buried = prot_to_buried[prot_name]
        non_buried = prot_to_non_buried[prot_name]
        non_interface = prot_to_non_interface[prot_name]
        buried_and_non_interface = set()
        non_buried_interface = set()
        for site in buried:
            buried_and_non_interface.add(site)
        for site in non_interface:
            buried_and_non_interface.add(site)
        for site in non_buried:
            if site not in non_interface:
                non_buried_interface.add(site)
        e1, e2 = group_entropy(cluster_ids, buried_and_non_interface, non_buried_interface)
        print(e1)
        print(e2)


def print_separate_intefaces():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    prot_name_to_clusters = {}
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        prot_name_to_clusters[prot_name] = (method_name, cluster_ids)
    for chain1, coords1 in chain_to_site_coords.items():
        prot_name1 = chain_to_prot[chain1]
        for chain2, coords2 in chain_to_site_coords.items():
            prot_name2 = chain_to_prot[chain2]
            if prot_name1 < prot_name2:
                int1, int2 = get_interface(coords1, coords2)
                if len(int1) == 0:
                    continue
                method_name, cluster_ids = prot_name_to_clusters[prot_name1]
                cl_counts, int_counts = count(cluster_ids, int1)
                if method_name != '':
                    print_table(out_path + prot_name1 + '_' + prot_name2 + '_' + method_name + '.txt', cl_counts,
                                int_counts, 'не в интерфейсе', 'в интерфейсе')
                else:
                    print_table(out_path + prot_name1 + '_' + prot_name2 + '.txt', cl_counts, int_counts,
                                'не в интерфейсе', 'в интерфейсе')
                method_name, cluster_ids = prot_name_to_clusters[prot_name2]
                cl_counts, int_counts = count(cluster_ids, int2)
                if method_name != '':
                    print_table(out_path + prot_name2 + '_' + prot_name1 + '_' + method_name + '.txt', cl_counts,
                                int_counts, 'не в интерфейсе', 'в интерфейсе')
                else:
                    print_table(out_path + prot_name2 + '_' + prot_name1 + '.txt', cl_counts, int_counts,
                                'не в интерфейсе', 'в интерфейсе')


def parse_dssp():
    chain_to_ss = {}
    for chain in chain_to_prot.keys():
        chain_to_ss[chain] = {}
    with open(path_to_dssp_raw) as f:
        for line in f.readlines()[28:]:
            s = line.strip().split()
            if s[2] in chain_to_ss:
                pos_to_ss = chain_to_ss[s[2]]
                if use_reduced_ss_states:
                    if s[4] in reduced_ss_states:
                        pos_to_ss[int(s[1])] = reduced_ss_states[s[4]]
                    else:
                        pos_to_ss[int(s[1])] = 'C'
                else:
                    if s[4] in 'HISTGBEC':
                        pos_to_ss[int(s[1])] = s[4]
                    else:
                        pos_to_ss[int(s[1])] = 'C'
    return chain_to_ss


def chi_sqr(cluster_ids, pos_to_ss_arr, cl_num, ss_types, permute=False, ss_permute=True):
    pos_to_cluster_id = cluster_ids
    pos_to_ss = pos_to_ss_arr
    if permute:
        if ss_permute:
            ss_list = []
            c_type = None
            length = 0
            for ss in pos_to_ss_arr[1:]:
                if c_type != ss:
                    if length > 0:
                        ss_list.append((c_type, length))
                    c_type = ss
                    length = 1
                else:
                    length += 1
            ss_list.append((c_type, length))
            shuffle(ss_list)
            shuffled = np.empty(len(pos_to_ss_arr), dtype=object)
            j = 1
            for t, l in ss_list:
                for i in range(l):
                    shuffled[j] = t
                    j += 1
            pos_to_ss = shuffled
        else:
            non_zeroes = []
            for i in range(len(cluster_ids)):
                if cluster_ids[i] > 0:
                    non_zeroes.append(cluster_ids[i])
            shuffle(non_zeroes)
            shuffled = np.zeros(len(cluster_ids), dtype=int)
            j = 0
            for i in range(len(cluster_ids)):
                if cluster_ids[i] > 0:
                    shuffled[i] = non_zeroes[j]
                    j += 1
            pos_to_cluster_id = shuffled
            # print(pos_to_cluster_id)
    cl_probs = np.zeros(cl_num + 1, dtype=float)
    for cl_id in pos_to_cluster_id:
        if cl_id > 0:
            cl_probs[cl_id] += 1
    n = cl_probs.sum()
    cl_probs /= n
    ss_probs = {ss: 0 for ss in ss_types}
    for pos in range(len(pos_to_cluster_id)):
        ss = pos_to_ss[pos]
        if pos_to_cluster_id[pos] > 0:
            ss_probs[ss] += 1
    ss_probs = {ss: prob/n for ss, prob in ss_probs.items()}
    cl_ss_counts = np.empty(cl_num + 1, dtype=object)
    for cl_id in range(1, cl_num + 1):
        cl_ss_counts[cl_id] = {ss: 0 for ss in ss_probs.keys()}
    for pos in range(0, len(pos_to_cluster_id)):
        cl_id = pos_to_cluster_id[pos]
        if cl_id > 0:
            ss = pos_to_ss[pos]
            ss_counts = cl_ss_counts[cl_id]
            ss_counts[ss] += 1
    stat = 0
    for cl_id in range(1, cl_num + 1):
        ss_counts = cl_ss_counts[cl_id]
        cl_prob = cl_probs[cl_id]
        for ss, ss_prob in ss_probs.items():
            stat += (ss_counts[ss]/n - cl_prob*ss_prob)**2
    return stat, cl_ss_counts


def extreme_counts(n, stat, cluster_ids, pos_to_ss, cl_num, ss_types):
    c = 0
    for i in range(n):
        s, cl_ss_counts = chi_sqr(cluster_ids, pos_to_ss, cl_num, ss_types, True)
        if s >= stat:
            c += 1
    return c


def ss_pvalue(cluster_ids, pos_to_ss_filtered, cl_num, ss_types):
    stat, cl_ss_counts = chi_sqr(cluster_ids, pos_to_ss_filtered, cl_num, ss_types)
    c = 0
    tasks = Parallel(n_jobs=thread_num)(delayed(extreme_counts)(iter_num//thread_num + 1, stat, cluster_ids,
                                                                pos_to_ss_filtered, cl_num, ss_types)
                                        for i in range(thread_num))
    for task in tasks:
        c += task
    return stat, c/((iter_num//thread_num + 1)*thread_num), cl_ss_counts


def print_secondary_structure_enrichment_chi_sqr():
    chain_to_ss = parse_dssp()
    prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name)
        else:
            print(prot_name + '.' + method_name)
        if only_non_burried:
            non_burried = prot_to_non_buried[prot_name]
        else:
            non_burried = None
        pos_to_ss = chain_to_ss[prot_to_chain[prot_name]]
        pos_to_ss_filtered = np.empty(len(pos_to_ss) + 1, dtype=object)
        ss_types = set()
        for pos, ss in pos_to_ss.items():
            # if pos < len(cluster_ids) and cluster_ids[pos] > 0:
            ss_types.add(ss)
            pos_to_ss_filtered[pos] = ss
        cl_num = 0
        for cl_id in cluster_ids:
            if cl_id > cl_num:
                cl_num = cl_id
        stat, p_value, cl_ss_counts = ss_pvalue(cluster_ids, pos_to_ss_filtered, cl_num, ss_types)
        print_ss_table(cl_ss_counts, cl_num, ss_types)
        print('chi_sqr p_value = %1.4f' % p_value)


def print_secondary_structure_enrichment_fisher(binary=False):
    chain_to_ss = parse_dssp()
    prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
    prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name)
        else:
            print(prot_name + '.' + method_name)
        if only_non_burried:
            non_burried = prot_to_non_buried[prot_name]
        else:
            non_burried = None
        pos_to_ss = chain_to_ss[prot_to_chain[prot_name]]
        pos_to_ss_filtered = np.empty(len(cluster_ids), dtype=object)
        ss_types = set()
        for pos, ss in pos_to_ss.items():
            if pos < len(cluster_ids) and cluster_ids[pos] > 0:
                ss_types.add(ss)
                pos_to_ss_filtered[pos] = ss
        cl_num = 0
        for cl_id in cluster_ids:
            if cl_id > cl_num:
                cl_num = cl_id
        if binary:
            ss_to_pos = {}
            for pos in range(len(cluster_ids)):
                ss = pos_to_ss_filtered[pos]
                if ss is not None:
                    pos_set = ss_to_pos.get(ss)
                    if pos_set is None:
                        pos_set = set()
                        ss_to_pos[ss] = pos_set
                    pos_set.add(pos)
            for ss, pos_set in ss_to_pos.items():
                non_int_counts, int_counts = count(cluster_ids, pos_set, non_burried)
                print_table(out_path + prot_name + '_' + ss + '.txt', non_int_counts, int_counts, 'not ' + ss, ss)
        else:
            stat, cl_ss_counts = chi_sqr(cluster_ids, pos_to_ss_filtered, cl_num, ss_types)
            print_ss_table(cl_ss_counts, cl_num, ss_types, out_path + prot_name + '.txt')


if __name__ == '__main__':
    if not exists(out_path):
        os.makedirs(out_path)
    # print_unified_intefaces_enc_burried()
    # print_unified_intefaces()
    # print_unified_intefaces_enc()
    # print_secondary_structure_enrichment_fisher()
    print_secondary_structure_enrichment_chi_sqr()