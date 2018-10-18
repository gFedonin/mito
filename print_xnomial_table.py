import os
from os.path import exists

import numpy as np
import math

from scipy.stats import entropy

from compute_cluster_stats import parse_colors, parse_out, parse_site2pdb

path_to_pdb = './pdb/1occ.pdb1'
path_to_colors = './Coloring/nogaps.pcor.p05/'
path_to_cox_data = './Coloring/COXdata.txt'
path_to_dssp = './Coloring/buried/1occ.csv'

chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}
dist_threshold = 8

only_selected_chains = True
print_contingency = True
only_mitochondria_to_nuclear = False
only_non_burried = True

out_path = './fisher/non_burried_ABC_nogaps.pcor.p05_enc/'

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
    with open(path_to_dssp, 'r') as f:
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


def get_interface(pos_to_coords1, pos_to_coords2, filter_set1=None, filter_set2=None):
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
            if dist(c1, c2) < dist_threshold:
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


def main():
    if not exists(out_path):
        os.makedirs(out_path)
    # print_unified_intefaces_enc_burried()
    # print_unified_intefaces()
    print_unified_intefaces_enc()


if __name__ == '__main__':
    main()