from os import cpu_count
from os.path import exists
from random import choice

import numpy as np
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from numpy.random.mtrand import shuffle
from sklearn.externals.joblib import Parallel, delayed
from numba import jit


path_to_pdb = '../pdb/1occ.pdb1'
path_to_cox_data = '../Coloring/COXdata.txt'
path_to_colors = '../Coloring/internal_gaps.2/'

chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
prot_to_chain = {'cox1': 'A', 'cox2': 'B', 'cox3': 'C'}
dist_threshold = 8

use_colors = False
only_selected_chains = True
only_mitochondria_to_nuclear = False
only_non_burried = True
thread_num = cpu_count()
permutations_num = 1
print_hists = True
random_coloring_stat_hist_path = '../res/random_coloring_stat_hist/'


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


def parse_pdb(path_to_pdb, only_selected_chains):
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


def read_cox_data():
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


def parse_colors(chain_to_prot, path_to_colors):
    prot_to_clusters = []
    for (dirpath, dirnames, filenames) in os.walk(path_to_colors):
        for filename in filenames:
            i = filename.find('.colors')
            if i != -1:
                prot_name = filename[:filename.index('.')]
                if prot_name not in chain_to_prot.values():
                    continue
                if i != len(prot_name):
                    method_name = filename[len(prot_name) + 1:i]
                else:
                    method_name = ''
                pos_to_cluster_id = []
                with open(path_to_colors + filename, 'r') as f:
                    max_pos = 0
                    for line in f.readlines()[1:]:
                        s = line.strip().split('\t')
                        pos = int(s[1])
                        if pos == 0:
                            continue
                        if pos > max_pos:
                            max_pos = pos
                        pos_to_cluster_id.append((pos, int(s[2])))
                cluster_ids = np.zeros(max_pos + 1, dtype=int)
                for pos, c in pos_to_cluster_id:
                    cluster_ids[pos] = c
                prot_to_clusters.append((prot_name, method_name, cluster_ids))
    return prot_to_clusters


def parse_site2pdb(chain_to_prot):
    prot_to_site_map = {}
    for (dirpath, dirnames, filenames) in os.walk(path_to_colors):
        for filename in filenames:
            i = filename.find('.site2pdb')
            if i != -1:
                prot_name = filename[:filename.index('.')]
                if prot_name not in chain_to_prot.values():
                    continue
                site2pdb = {}
                prot_to_site_map[prot_name] = site2pdb
                with open(path_to_colors + filename, 'r') as f:
                    f.readline()
                    for line in f.readlines():
                        s = line.strip().split('\t')
                        site2pdb[s[0]] = int(s[1])
    return prot_to_site_map


def parse_out(prot_to_site_map, chain_to_prot, path_to_colors):
    prot_to_clusters = []
    for (dirpath, dirnames, filenames) in os.walk(path_to_colors):
        for filename in filenames:
            i = filename.find('.out')
            if i == -1:
                i = filename.find('.partition')
            if i != -1:
                prot_name = filename[:filename.index('.')]
                if prot_name not in chain_to_prot.values():
                    continue
                if i != len(prot_name):
                    method_name = filename[len(prot_name) + 1:i]
                else:
                    method_name = ''
                pos_to_cluster_id = []
                site2pdb = prot_to_site_map[prot_name]
                with open(path_to_colors + filename, 'r') as f:
                    max_pos = 0
                    for line in f.readlines()[5:]:
                        s = line.strip().split()
                        pos = site2pdb[s[0]]
                        if pos == 0:
                            continue
                        if pos > max_pos:
                            max_pos = pos
                        pos_to_cluster_id.append((pos, int(s[1])))
                cluster_ids = np.zeros(max_pos + 1, dtype=int)
                for pos, c in pos_to_cluster_id:
                    cluster_ids[pos] = c
                prot_to_clusters.append((prot_name, method_name, cluster_ids))
    return prot_to_clusters


@jit(nopython=True)
def dist(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)


def get_neighbors(pos_to_coords, filter_set):
    pos_to_c = []
    for p, c in pos_to_coords.items():
        if p in filter_set:
            pos_to_c.append((p, c))
    neighbors = {}
    for p, c in pos_to_c:
        neighbors[p] = []
    for i in range(len(pos_to_c)):
        p_i, c_i = pos_to_c[i]
        for j in range(i + 1, len(pos_to_c)):
            p_j, c_j = pos_to_c[j]
            if dist(c_i, c_j) < dist_threshold:
                neighbors[p_i].append(p_j)
                neighbors[p_j].append(p_i)
    return neighbors


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
        if edge_numbers_random[i] < edge_numbers[i]:
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
            if edges_added_to_1 > 0:
                if edge_numbers_random[cl_id2] + edges_added_to_2 >= edge_numbers[cl_id2]:
                   repeat = False
                elif edges_added_to_1 + edges_added_to_2 > 0:
                    repeat = False
        edge_numbers_random[cl_id1] += edges_added_to_1
        edge_numbers_random[cl_id2] += edges_added_to_2
        exchange_vertices(v, u, cl_id1, cl_id2, shuffled, cluster_to_vertices, cluster_to_neighbors, neighbors)
        bad_clusters = []
        for i in range(cl_num):
            if edge_numbers_random[i] < edge_numbers[i]:
                bad_clusters.append(i)
    return shuffled


def exchange_vertices(v, u, cl_id1, cl_id2, shuffled, cluster_to_vertices, cluster_to_neighbors, neighbors):
    shuffled[v] = cl_id1
    shuffled[u] = cl_id2

    cluster1 = cluster_to_vertices[cl_id1]
    i = cluster1.index(u)
    cluster1[i] = v
    cluster2 = cluster_to_vertices[cl_id2]
    i = cluster2.index(v)
    cluster2[i] = u

    neighbors1 = set(cluster_to_neighbors[cl_id1])
    neighbors2 = set(cluster_to_neighbors[cl_id2])
    neighbors1.remove(v)
    if u in neighbors2:
        neighbors2.remove(u)
    for n in neighbors[v]:
        if shuffled[n] != cl_id1:
            neighbors1.add(n)
        if shuffled[n] == cl_id2:
            neighbors2.add(v)
        if n in neighbors2:
            delete = True
            for nn in neighbors[n]:
                if shuffled[nn] == cl_id2:
                    delete = False
                    break
            if delete:
                neighbors2.remove(n)
    for n in neighbors[u]:
        if shuffled[n] != cl_id2:
            neighbors2.add(n)
        if shuffled[n] == cl_id1:
            neighbors1.add(u)
        if n in neighbors1:
            delete = True
            for nn in neighbors[n]:
                if shuffled[nn] == cl_id1:
                    delete = False
                    break
            if delete:
                neighbors1.remove(n)
    cluster_to_neighbors[cl_id1] = list(neighbors1)
    cluster_to_neighbors[cl_id2] = list(neighbors2)


def get_cluster_neighbors(cl_num, neighbors, cluster_id):
    edge_num_in_cluster = np.zeros(cl_num, dtype=int)
    cluster_to_neighbors = {}
    for i in range(1, cl_num):
        cluster_to_neighbors[i] = set()
    for pos, n_list in neighbors.items():
        same_cluster_count = 0
        cl_id = cluster_id[pos]
        cl_neighbors = cluster_to_neighbors[cl_id]
        for p in n_list:
            if cluster_id[p] == cl_id:
                same_cluster_count += 1
            else:
                cl_neighbors.add(p)
        edge_num_in_cluster[cl_id] += same_cluster_count#/len(n_list)
    for i in range(1, cl_num):
        edge_num_in_cluster[i] /= 2
        cluster_to_neighbors[i] = list(cluster_to_neighbors[i])
    return edge_num_in_cluster, cluster_to_neighbors


def compute_stat_on_random_colorings(neighbors, cluster_id, n, interface_set, filtered_set):
    res = []
    cl_num = 0
    for id in cluster_id:
        if id > cl_num:
            cl_num = id
    for i in range(n):
        random_coloring = gen_random_coloring(neighbors, cluster_id, cl_num)
        res.append(chi_sqr(random_coloring, interface_set, filtered_set))
    return res


def compute_JI_on_random_colorings(neighbors, cluster_id, n):
    cl_num = 0
    for id in cluster_id:
        if id > cl_num:
            cl_num = id
    res = [[] for i in range(cl_num)]
    for i in range(n):
        random_coloring = gen_random_coloring(neighbors, cluster_id, cl_num)
        for cl_id in range(1, cl_num + 1):
            real = set(pos for pos in range(len(cluster_id)) if cluster_id[pos] == cl_id)
            rand = set(pos for pos in range(len(random_coloring)) if random_coloring[pos] == cl_id)
            res[cl_id - 1].append(len(real.intersection(rand))/len(real.union(rand)))
    return res


def chi_sqr(cluster_ids, interface_set, filter_set):
    res = 0
    cl_to_poses = {}
    for pos in filter_set:
        cl = cluster_ids[pos]
        l = cl_to_poses.get(cl)
        if l is None:
            l = []
            cl_to_poses[cl] = l
        l.append(pos)
    p_exp = len(interface_set)/len(filter_set)
    for pos_list in cl_to_poses.values():
        c = 0
        for pos in pos_list:
            if pos in interface_set:
                c += 1
        p_obs = c/len(pos_list)
        res += (p_obs - p_exp)*(p_obs - p_exp)/p_exp
    return len(filter_set)*res


def test_independence(pos_to_coords, cluster_ids, interface, filter_set, prot_name):
    print('computing p_value')
    filtered_set = set()
    if filter_set is not None:
        for i in range(len(cluster_ids)):
            if cluster_ids[i] > 0 and i in filter_set:
                filtered_set.add(i)
    else:
        for i in range(len(cluster_ids)):
            if cluster_ids[i] > 0:
                filtered_set.add(i)
    neighbors = get_neighbors(pos_to_coords, filtered_set)

    stat = chi_sqr(cluster_ids, interface, filtered_set)
    iter_nums = []
    for i in range(thread_num):
        iter_nums.append(permutations_num//thread_num)
    for i in range(permutations_num%thread_num):
        iter_nums[i] += 1
    if print_hists:
        if not exists(random_coloring_stat_hist_path):
            os.makedirs(random_coloring_stat_hist_path)
        cl_num = 0
        for id in cluster_ids:
            if id > cl_num:
                cl_num = id
        tasks = Parallel(n_jobs=thread_num)(delayed(compute_JI_on_random_colorings)(neighbors, cluster_ids, n) for n in
                                            iter_nums)
        jaccad_indices = [[] for i in range(cl_num)]
        for task in tasks:
            for i in range(cl_num):
                jaccad_indices[i].extend(task[i])
        for i in range(cl_num):
            plt.title('Histogram of Jaccard index of random graphs')
            plt.xlabel('Jaccard index')
            plt.ylabel('Percent of graphs')
            n, bins, patches = plt.hist(jaccad_indices[i], 50, density=True, facecolor='g', alpha=0.75)
            # plt.axis([0, 0.002, 0, 6000])
            plt.savefig(random_coloring_stat_hist_path + prot_name + '_' + str(i + 1) + '.png')
            plt.clf()
    tasks = Parallel(n_jobs=thread_num)(delayed(compute_stat_on_random_colorings)(neighbors, cluster_ids, n,
                                                                                  interface, filtered_set) for n in iter_nums)
    i = 0
    for task in tasks:
        for s in task:
            if s >= stat:
                i += 1
    return i/permutations_num


def count(cluster_ids, interface, filter_set=None):
    cl_num = 0
    for cl_id in cluster_ids:
        if cl_id > cl_num:
            cl_num = cl_id
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
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot), chain_to_prot, path_to_colors)
    prot_to_buried = None
    prot_to_non_buried = None
    prot_to_non_interface = None
    if only_non_burried:
        prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data()
        # prot_to_buried, prot_to_non_buried = read_dssp_data()
    interfaces = {}
    for prot_name1, chain1 in prot_to_chain.items():
        coords1 = chain_to_site_coords[chain1]
        non_burried1 = None
        if only_non_burried:
            non_burried1 = prot_to_non_buried[prot_name1]
        for chain2, coords2 in chain_to_site_coords.items():
            prot_name2 = chain_to_prot.get(chain2)
            if prot_name2 is not None:
                non_burried2 = None
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
        if method_name != '':
            print(prot_name + '.' + method_name)
        else:
            print(prot_name)
        int = interfaces[prot_name]
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        if only_non_burried:
            filter_set = prot_to_non_buried[prot_name]
            for i in range(len(cluster_ids)):
                if i not in filter_set:
                    cluster_ids[i] = 0
        else:
            filter_set = None
        cl_counts, int_counts = count(cluster_ids, int, filter_set)
        if method_name != '':
            p_value = test_independence(coords, cluster_ids, int, filter_set, prot_name + '.' + method_name)
        else:
            p_value = test_independence(coords, cluster_ids, int, filter_set, prot_name)
        print_table(cl_counts, int_counts, 'не в интерфейсе', 'в интерфейсе', p_value)


def print_unified_intefaces_enc():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot), chain_to_prot, path_to_colors)
    prot_to_burried, prot_to_non_buried, prot_to_non_interface = read_cox_data()
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        if method_name != '':
            print(prot_name + '.' + method_name)
        else:
            print(prot_name)
        non_buried = prot_to_non_buried[prot_name]
        non_interface = prot_to_non_interface[prot_name]
        coords = chain_to_site_coords[prot_to_chain[prot_name]]
        for i in range(len(cluster_ids)):
            if i not in non_buried:
                cluster_ids[i] = 0
        cl_counts, int_counts = count(cluster_ids, non_interface, non_buried)
        if method_name != '':
            p_value = test_independence(coords, cluster_ids, non_interface, non_buried, prot_name + '.' + method_name)
        else:
            p_value = test_independence(coords, cluster_ids, non_interface, non_buried, prot_name)
        print_table(cl_counts, int_counts, 'ENC_noninterf', 'CONT + ENC_interface', p_value)


def print_separate_intefaces():
    chain_to_site_coords = parse_pdb(path_to_pdb, only_selected_chains)
    if use_colors:
        prot_to_clusters = parse_colors(chain_to_prot, path_to_colors)
    else:
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot), chain_to_prot, path_to_colors)
    prot_to_buried = None
    prot_to_non_buried = None
    prot_to_non_interface = None
    if only_non_burried:
        prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data()
        # prot_to_buried, prot_to_non_buried = read_dssp_data()
    prot_name_to_clusters = {}
    for prot_name, method_name, cluster_ids in prot_to_clusters:
        prot_name_to_clusters[prot_name] = (method_name, cluster_ids)
    for chain1, coords1 in chain_to_site_coords.items():
        prot_name1 = chain_to_prot[chain1]
        non_burried1 = None
        if only_non_burried:
            non_burried1 = prot_to_non_buried[prot_name1]
        for chain2, coords2 in chain_to_site_coords.items():
            prot_name2 = chain_to_prot[chain2]
            non_burried2 = None
            if only_non_burried:
                non_burried2 = prot_to_non_buried[prot_name2]
            if prot_name1 < prot_name2:
                int1, int2 = get_interface(coords1, coords2, non_burried1, non_burried2)
                if len(int1) == 0:
                    continue
                method_name, cluster_ids = prot_name_to_clusters[prot_name1]
                coords = chain_to_site_coords[prot_to_chain[prot_name1]]
                if only_non_burried:
                    filter_set = prot_to_non_buried[prot_name1]
                    for i in range(len(cluster_ids)):
                        if i not in filter_set:
                            cluster_ids[i] = 0
                else:
                    filter_set = None
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
                if only_non_burried:
                    filter_set = prot_to_non_buried[prot_name2]
                    for i in range(len(cluster_ids)):
                        if i not in filter_set:
                            cluster_ids[i] = 0
                else:
                    filter_set = None
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


def main():
    print_unified_intefaces()
    # print_unified_intefaces_enc()
    # print_separate_intefaces()


if __name__ == '__main__':
    main()