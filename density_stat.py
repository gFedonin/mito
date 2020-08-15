import os
from os.path import exists
from sys import float_info
import pandas as pd
import numpy as np
import math

from Bio.Alphabet import ThreeLetterProtein
from Bio.PDB import PDBParser

from numpy.random.mtrand import shuffle
from sklearn.externals.joblib import Parallel, delayed

# pdb_id = '5ara'
# pdb_id = '1be3'
pdb_id = '1bgy'
# pdb_id = '1occ'
from assimptotic_tests import read_cox_data
from compute_interface_stat_rand_int import path_to_cox_data


path_to_pdb = '../pdb/' + pdb_id + '.pdb'
# path_to_pos_list = '../res/cox1_nonABC.interface'
path_to_pos_list = '../res/cytb.interface'
# path_to_atp6_data = '../Coloring/atp6_5ara_Aledo_4ang_fixed.csv'
path_to_atp6_data = '../Coloring/cytb_1bgy_Aledo_4ang_CO.csv'
selected_chain = 'C'

# chain_to_prot = {'W': 'atp6'}
# prot_to_chain = {'atp6': ['W']}
# chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
# chain_to_prot = {'A': 'cox1'}
# chain_to_prot = {'A': 'cox1', 'N': 'cox1', 'B': 'cox2', 'O': 'cox2', 'C': 'cox3', 'P': 'cox3'}
# prot_to_chain = {'cox1': ['A', 'N'], 'cox2': ['B', 'O'], 'cox3': ['C', 'P']}
# prot_to_chain = {'cox1': 'A'}
# chain_to_prot = {'C': 'cytb'}
chain_to_prot = {'C': 'cytb', 'O': 'cytb'}
prot_to_chain = {'cytb': ['C', 'O']}
# aledo_dist = True
dist_threshold = 4
permutations_num = 10000


use_colors = False
use_internal_contacts = True
use_external_contacts = True


def parse_pdb_Aledo_biopython(pdb_name, path_to_pdb, chain_to_prot):
    # prot_to_site_coords = {}
    chain_to_site_coords = {}
    structure = PDBParser().get_structure(pdb_name, path_to_pdb)
    model = structure[0]
    alphabet = set(aa.upper() for aa in ThreeLetterProtein().letters)
    for chn in model:
        if chn.id not in chain_to_prot:
            continue
        pos_to_coords = {}
        pos = 1
        for residue in chn:
            # pos = int(residue.id[1])
            if residue.resname not in alphabet:
                continue
            r1, r2, r3 = residue.id
            if r1 != ' ':
                continue
            atoms = [tuple(atom.coord) for atom in residue]
            pos_to_coords[pos] = atoms
            pos += 1
        # prot_to_site_coords[chain_to_prot[chn.id]] = pos_to_coords
        chain_to_site_coords[chn.id] = pos_to_coords
    # return prot_to_site_coords
    return chain_to_site_coords


def dist(c1, c2):
    return math.sqrt((c1[0] - c2[0])*(c1[0] - c2[0]) + (c1[1] - c2[1])*(c1[1] - c2[1]) + (c1[2] - c2[2])*(c1[2] - c2[2]))


def dist_aledo(heavy_atoms1, heavy_atoms2):
    n1 = len(heavy_atoms1)
    n2 = len(heavy_atoms2)
    return min(dist(heavy_atoms1[i], heavy_atoms2[j]) for i in range(n1) for j in range(n2))


def get_internals_dist(pos_to_coords, dist_f, max_pos):
    pos_to_c = [(p, c) for p, c in pos_to_coords.items()]
    internal_dist = np.zeros((max_pos + 1, max_pos + 1), dtype=float)
    for i in range(len(pos_to_c)):
        p_i, c_i = pos_to_c[i]
        for j in range(i + 1, len(pos_to_c)):
            p_j, c_j = pos_to_c[j]
            d = dist_f(c_i, c_j)
            internal_dist[p_i, p_j] = d
            internal_dist[p_j, p_i] = d
    return internal_dist


def get_external_dist(prot_name, chain_to_site_coords, prot_to_chain, dist_f, max_pos):
    chains = prot_to_chain[prot_name]
    external_dist = np.ndarray((max_pos + 1, max_pos + 1), dtype=float)
    external_dist.fill(float_info.max)
    for i in range(len(chains)):
        pos_to_c_i = [(p, c) for p, c in chain_to_site_coords[chains[i]].items()]
        for j in range(i + 1, len(chains)):
            pos_to_c_j = [(p, c) for p, c in chain_to_site_coords[chains[j]].items()]
            for p_k, c_k in pos_to_c_i:
                for p_n, c_n in pos_to_c_j:
                    d = dist_f(c_k, c_n)
                    external_dist[p_k, p_n] = min(d, external_dist[p_k, p_n])
    return external_dist


def get_pdb_neighbors(prot_name, prot_to_chain, chain_to_site_coords, dist_f, use_internal_contacts, use_external_contacts):
    chains = prot_to_chain[prot_name]
    neighbors = {}
    max_pos = 0
    poses = []
    for p, c in chain_to_site_coords[chains[0]].items():
        neighbors[p] = []
        poses.append(p)
        if p > max_pos:
            max_pos = p

    if use_internal_contacts and not use_external_contacts:
        internal_dist = get_internals_dist(chain_to_site_coords[chains[0]], dist_f, max_pos)
        for i in range(1, len(chains)):
            internal_dist += get_internals_dist(chain_to_site_coords[chains[i]], dist_f, max_pos)
        internal_dist /= len(chains)
        for i in range(len(poses)):
            p_i = poses[i]
            for j in range(i + 1, len(poses)):
                p_j = poses[j]
                d = internal_dist[p_i, p_j]
                if 0 < d < dist_threshold:
                    neighbors[p_i].append(p_j)
                    neighbors[p_j].append(p_i)
        return neighbors

    if not use_internal_contacts and use_external_contacts:
        external_dist = get_external_dist(prot_name, chain_to_site_coords, prot_to_chain, dist_f, max_pos)
        for i in range(len(poses)):
            p_i = poses[i]
            for j in range(i + 1, len(poses)):
                p_j = poses[j]
                if external_dist[p_i, p_j] < dist_threshold:
                    neighbors[p_i].append(p_j)
                    neighbors[p_j].append(p_i)
        return neighbors

    if use_internal_contacts and use_external_contacts:
        internal_dist = get_internals_dist(chain_to_site_coords[chains[0]], dist_f, max_pos)
        for i in range(1, len(chains)):
            internal_dist += get_internals_dist(chain_to_site_coords[chains[i]], dist_f, max_pos)
        internal_dist /= len(chains)

        external_dist = get_external_dist(prot_name, chain_to_site_coords, prot_to_chain, dist_f, max_pos)

        for i in range(len(poses)):
            p_i = poses[i]
            for j in range(i + 1, len(poses)):
                p_j = poses[j]
                d_int = internal_dist[p_i, p_j]
                if d_int == 0:
                    if external_dist[p_i, p_j] < dist_threshold:
                        neighbors[p_i].append(p_j)
                        neighbors[p_j].append(p_i)
                else:
                    if min(d_int, external_dist[p_i, p_j]) < dist_threshold:
                        neighbors[p_i].append(p_j)
                        neighbors[p_j].append(p_i)
        return neighbors


def get_evolution_neighbors(graph, positive=True):
    neighbors = {}
    for node in graph.nodes:
        neighbors[int(node)] = []
    for i in graph.nodes:
        p_i = int(i)
        for j in graph.adj[i]:
            p_j = int(j)
            w = float(graph[i][j]['weight'])
            if w > 0:
                if positive:
                    neighbors[p_i].append(p_j)
                    neighbors[p_j].append(p_i)
            else:
                if not positive:
                    neighbors[p_i].append(p_j)
                    neighbors[p_j].append(p_i)
    return neighbors


def cluster_stat(neighbors, cluster_id, permute):
    pos_to_cluster_id = cluster_id
    if permute:
        # print(cluster_id)
        non_zeroes = []
        for i in range(len(cluster_id)):
            if cluster_id[i] > 0:
                non_zeroes.append(cluster_id[i])
        shuffle(non_zeroes)
        shuffled = np.zeros(len(cluster_id), dtype=int)
        j = 0
        for i in range(len(cluster_id)):
            if cluster_id[i] > 0:
                shuffled[i] = non_zeroes[j]
                j += 1
        pos_to_cluster_id = shuffled
        # print(pos_to_cluster_id)
    cluster_id_to_stat = {}
    cl_num = 0
    pos_num = 0
    for id in pos_to_cluster_id:
        if id > cl_num:
            cl_num = id
        if id > 0:
            pos_num += 0
    cl_num += 1
    av_in_cluster = np.zeros(cl_num, dtype=float)
    # av_normalized_edge_num = np.zeros(cl_num, dtype=float)
    # std_normalized_edge_num = np.zeros(cl_num, dtype=float)
    cluster_population = np.zeros(cl_num, dtype=int)
    cluster_densities = np.zeros(cl_num, dtype=float)
    for pos, n_list in neighbors.items():
        if pos < len(pos_to_cluster_id):
            internal_edge_count = 0
            cl_id = pos_to_cluster_id[pos]
            total_neighbor_count = 0
            for p in n_list:
                if p < len(pos_to_cluster_id):
                    if pos_to_cluster_id[p] == cl_id:
                        internal_edge_count += 1
                    if pos_to_cluster_id[p] > 0:
                        total_neighbor_count += 1
            if total_neighbor_count > 0:
                av_in_cluster[cl_id] += internal_edge_count/total_neighbor_count
                cluster_population[cl_id] += 1
                cluster_densities[cl_id] += internal_edge_count/2
            # av_normalized_edge_num[cl_id] += internal_edge_count/(pos_num - 1)
            # std_normalized_edge_num[cl_id] += internal_edge_count/(pos_num - 1)**2
    av_in_cluster_total = 0
    cluster_population_total = 0
    for cl_id in range(1, cl_num):
        av_in_cluster_total += av_in_cluster[cl_id]
        cluster_population_total += cluster_population[cl_id]
        n = cluster_population[cl_id]
        if n > 0:
            cluster_id_to_stat[cl_id] = av_in_cluster[cl_id]/n
            # av_normalized_edge_num[cl_id] /= n
            # std_normalized_edge_num[cl_id] = np.sqrt(std_normalized_edge_num[cl_id]/n - av_normalized_edge_num[cl_id]**2)
            cluster_densities[cl_id] /= n*(n - 1)/2
        else:
            cluster_id_to_stat[cl_id] = 0
    cluster_id_to_stat['total'] = av_in_cluster_total/cluster_population_total

    return cluster_id_to_stat, cluster_population, cluster_densities


def cluster_stat2(neighbors, cluster_id, permute):
    # for each vertex stat = internal/total; average for all vertices in each cluster
    pos_to_cluster_id = cluster_id
    if permute:
        # print(cluster_id)
        non_zeroes = []
        for i in range(len(cluster_id)):
            if cluster_id[i] > 0:
                non_zeroes.append(cluster_id[i])
        shuffle(non_zeroes)
        shuffled = np.zeros(len(cluster_id), dtype=int)
        j = 0
        for i in range(len(cluster_id)):
            if cluster_id[i] > 0:
                shuffled[i] = non_zeroes[j]
                j += 1
        pos_to_cluster_id = shuffled
        # print(pos_to_cluster_id)
    cluster_id_to_stat = {}
    cl_num = 0
    pos_num = 0
    for id in pos_to_cluster_id:
        if id > cl_num:
            cl_num = id
        if id > 0:
            pos_num += 0
    cl_num += 1
    av_in_cluster = np.zeros(cl_num, dtype=float)
    # av_normalized_edge_num = np.zeros(cl_num, dtype=float)
    # std_normalized_edge_num = np.zeros(cl_num, dtype=float)
    cluster_population = np.zeros(cl_num, dtype=int)
    cluster_densities = np.zeros(cl_num, dtype=float)
    for pos, n_list in neighbors.items():
        if pos < len(pos_to_cluster_id):
            internal_edge_count = 0
            cl_id = pos_to_cluster_id[pos]
            total_neighbor_count = 0
            for p in n_list:
                if p < len(pos_to_cluster_id):
                    if pos_to_cluster_id[p] == cl_id:
                        internal_edge_count += 1
                    if pos_to_cluster_id[p] > 0:
                        total_neighbor_count += 1
            if total_neighbor_count > 0:
                av_in_cluster[cl_id] += internal_edge_count/total_neighbor_count
                cluster_population[cl_id] += 1
                cluster_densities[cl_id] += internal_edge_count/2
            # av_normalized_edge_num[cl_id] += internal_edge_count/(pos_num - 1)
            # std_normalized_edge_num[cl_id] += internal_edge_count/(pos_num - 1)**2
    av_in_cluster_total = 0
    cluster_population_total = 0
    for cl_id in range(1, cl_num):
        av_in_cluster_total += av_in_cluster[cl_id]
        cluster_population_total += cluster_population[cl_id]
        n = cluster_population[cl_id]
        if n > 0:
            cluster_id_to_stat[cl_id] = av_in_cluster[cl_id]/n
            # av_normalized_edge_num[cl_id] /= n
            # std_normalized_edge_num[cl_id] = np.sqrt(std_normalized_edge_num[cl_id]/n - av_normalized_edge_num[cl_id]**2)
            # cluster_densities[cl_id] /= n*(n - 1)/2
        else:
            cluster_id_to_stat[cl_id] = 0
    cluster_id_to_stat['total'] = av_in_cluster_total/cluster_population_total

    return cluster_id_to_stat, cluster_population, cluster_densities


def cluster_stat1(neighbors, cluster_id, permute):
    pos_to_cluster_id = cluster_id
    if permute:
        # print(cluster_id)
        non_zeroes = []
        for i in range(len(cluster_id)):
            if cluster_id[i] > 0:
                non_zeroes.append(cluster_id[i])
        shuffle(non_zeroes)
        shuffled = np.zeros(len(cluster_id), dtype=int)
        j = 0
        for i in range(len(cluster_id)):
            if cluster_id[i] > 0:
                shuffled[i] = non_zeroes[j]
                j += 1
        pos_to_cluster_id = shuffled
        # print(pos_to_cluster_id)
    cluster_id_to_stat = {}
    cl_num = 0
    for id in pos_to_cluster_id:
        if id > cl_num:
            cl_num = id
    cl_num += 1
    internal_edges_count = np.zeros(cl_num, dtype=float)
    external_edges_count = np.zeros(cl_num, dtype=int)
    cluster_population = np.zeros(cl_num, dtype=int)
    cluster_densities = np.zeros(cl_num, dtype=float)
    for pos, n_list in neighbors.items():
        if pos < len(pos_to_cluster_id):
            internal_edge_count = 0
            cl_id = pos_to_cluster_id[pos]
            external_edge_count = 0
            for p in n_list:
                if p < len(pos_to_cluster_id):
                    if pos_to_cluster_id[p] == cl_id:
                        internal_edge_count += 1
                    else:
                        if pos_to_cluster_id[p] > 0:
                            external_edge_count += 1
            internal_edge_count /= 2
            internal_edges_count[cl_id] += internal_edge_count
            external_edges_count[cl_id] += external_edge_count
            cluster_population[cl_id] += 1
    total_internal = 0
    total_external = 0
    for cl_id in range(1, cl_num):
        inter = internal_edges_count[cl_id]
        exter = external_edges_count[cl_id]
        total_internal += inter
        total_external += exter
        n = cluster_population[cl_id]
        if inter + exter > 0:
            cluster_id_to_stat[cl_id] = inter/(inter + exter)
            cluster_densities[cl_id] = inter/(n*(n - 1)/2)
        else:
            cluster_id_to_stat[cl_id] = 0
    cluster_id_to_stat['total'] = total_internal/(total_internal + total_external/2)
    return cluster_id_to_stat, cluster_population, cluster_densities


def cluster_stat11(neighbors, cluster_id, permute):
    pos_to_cluster_id = cluster_id
    if permute:
        # print(cluster_id)
        non_zeroes = []
        for i in range(len(cluster_id)):
            if cluster_id[i] > 0:
                non_zeroes.append(cluster_id[i])
        shuffle(non_zeroes)
        shuffled = np.zeros(len(cluster_id), dtype=int)
        j = 0
        for i in range(len(cluster_id)):
            if cluster_id[i] > 0:
                shuffled[i] = non_zeroes[j]
                j += 1
        pos_to_cluster_id = shuffled
        # print(pos_to_cluster_id)
    cluster_id_to_stat = {}
    cl_num = 0
    for id in pos_to_cluster_id:
        if id > cl_num:
            cl_num = id
    cl_num += 1
    internal_edges_count = np.zeros(cl_num, dtype=float)
    external_edges_count = np.zeros(cl_num, dtype=int)
    cluster_population = np.zeros(cl_num, dtype=int)
    cluster_sum_degree = np.zeros(cl_num, dtype=int)
    cluster_densities = np.zeros(cl_num, dtype=float)
    pos_to_degree = {pos: len(n_list) for pos, n_list in neighbors.items()}
    total_degree = 0
    for pos, n_list in neighbors.items():
        if pos < len(pos_to_cluster_id):
            internal_edge_count = 0
            cl_id = pos_to_cluster_id[pos]
            external_edge_count = 0
            for p in n_list:
                if p < len(pos_to_cluster_id):
                    if pos_to_cluster_id[p] == cl_id:
                        internal_edge_count += 1
                    else:
                        if pos_to_cluster_id[p] > 0:
                            external_edge_count += 1
            internal_edge_count /= 2
            internal_edges_count[cl_id] += internal_edge_count
            external_edges_count[cl_id] += external_edge_count
            cluster_population[cl_id] += 1
            cluster_sum_degree[cl_id] += pos_to_degree[pos]
            total_degree += pos_to_degree[pos]
    total_internal = 0
    total_external = 0
    for cl_id in range(1, cl_num):
        inter = internal_edges_count[cl_id]
        exter = external_edges_count[cl_id]
        total_internal += inter
        total_external += exter
        n = cluster_sum_degree[cl_id]
        if inter + exter > 0:
            cluster_id_to_stat[cl_id] = inter/(inter + exter)
            cluster_densities[cl_id] = 2*inter*total_degree/n/n
        else:
            cluster_id_to_stat[cl_id] = 0
    cluster_id_to_stat['total'] = total_internal/(total_internal + total_external/2)
    return cluster_id_to_stat, cluster_population, cluster_densities


def cluster_stat12(neighbors, cluster_id, permute):
    # in each cluster compute stat = internal edges / total edges
    pos_to_cluster_id = cluster_id
    if permute:
        # print(cluster_id)
        non_zeroes = [cluster_id[i] for i in range(len(cluster_id)) if cluster_id[i] > 0]
        shuffle(non_zeroes)
        shuffled = np.zeros(len(cluster_id), dtype=int)
        j = 0
        for i in range(len(cluster_id)):
            if cluster_id[i] > 0:
                shuffled[i] = non_zeroes[j]
                j += 1
        pos_to_cluster_id = shuffled
        # print(pos_to_cluster_id)
    cluster_id_to_stat = {}
    cl_num = 0
    for id in pos_to_cluster_id:
        if id > cl_num:
            cl_num = id
    cl_num += 1
    internal_edges_count = np.zeros(cl_num, dtype=float)
    external_edges_count = np.zeros(cl_num, dtype=int)
    cluster_population = np.zeros(cl_num, dtype=int)
    cluster_densities = np.zeros(cl_num, dtype=int)
    for pos, n_list in neighbors.items():
        if pos < len(pos_to_cluster_id):
            internal_edge_count = 0
            cl_id = pos_to_cluster_id[pos]
            external_edge_count = 0
            for p in n_list:
                if p < len(pos_to_cluster_id):
                    if pos_to_cluster_id[p] == cl_id:
                        internal_edge_count += 1
                    else:
                        if pos_to_cluster_id[p] > 0:
                            external_edge_count += 1
            internal_edge_count /= 2
            internal_edges_count[cl_id] += internal_edge_count
            external_edges_count[cl_id] += external_edge_count
            cluster_population[cl_id] += 1
    total_internal = 0
    total_external = 0
    for cl_id in range(1, cl_num):
        inter = internal_edges_count[cl_id]
        exter = external_edges_count[cl_id]
        total_internal += inter
        total_external += exter
        if inter + exter > 0:
            cluster_id_to_stat[cl_id] = inter/(inter + exter)
            cluster_densities[cl_id] = inter
        else:
            cluster_id_to_stat[cl_id] = 0
    cluster_id_to_stat['total'] = total_internal/(total_internal + total_external/2)
    return cluster_id_to_stat, cluster_population, cluster_densities


def pvalue(perm_array, stat_val):
    c = 0
    for val in perm_array:
        if val >= stat_val:
            c += 1
    return c / permutations_num


def pdb_contact_density():
    # if aledo_dist:
    dist_f = dist_aledo
        # prot_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, chain_to_prot)
    chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, chain_to_prot)
    pos_to_coords = chain_to_site_coords[selected_chain]
    max_pos = max(pos_to_coords.keys())
    prot_name = chain_to_prot[selected_chain]
    prot_to_buried, prot_to_non_buried, prot_to_non_interface = read_cox_data(path_to_cox_data)
    non_burried = prot_to_non_buried[prot_name]
    cluster_ids = np.zeros(max_pos + 1, dtype=int)
    for pos in non_burried:
        cluster_ids[pos] = 1
    for l in open(path_to_pos_list).readlines():
        cluster_ids[int(l)] = 2
    print(prot_name)
    # pos_to_coords = prot_to_site_coords[prot_name]
    neighbors = get_pdb_neighbors(prot_name, prot_to_chain, chain_to_site_coords, dist_f, use_internal_contacts, use_external_contacts)
    # cluster_ids = chain_to_clusters[prot_name]
    cl_num = 3
    cluster_id_to_stat, cluster_population, cluster_densities = cluster_stat12(neighbors, cluster_ids, False)
    # cluster_id_to_stat, cluster_population, cluster_densities = cluster_stat2(neighbors, cluster_ids, False)
    permutations = Parallel(n_jobs=-1)(delayed(cluster_stat12)(neighbors, cluster_ids, True) for i in range(permutations_num))
    # permutations = Parallel(n_jobs=-1)(delayed(cluster_stat2)(neighbors, cluster_ids, True) for i in range(permutations_num))
    perm_array = {}
    density_array = {}
    for i in range(1, cl_num):
        perm_array[i] = []
        density_array[i] = []
    perm_array['total'] = []
    for cl_id_to_stat, cl_population, cl_density in permutations:
        for i in range(1, cl_num):
            perm_array[i].append(cl_id_to_stat[i])
            density_array[i].append(cl_density[i])
        perm_array['total'].append(cl_id_to_stat['total'])
    print('cluster n_sites stat_value expected_value pvalue density_value expected_density pvalue_density')
    for i in range(1, cl_num):
        stat_val = cluster_id_to_stat[i]
        av = np.mean(perm_array[i])
        pval = pvalue(perm_array[i], stat_val)
        dens = cluster_densities[i]
        av_dens = np.mean(density_array[i])
        pval_dens = pvalue(density_array[i], dens)
        print("%d %d %1.2f %1.2f %1.4f %1.2f %1.2f %1.4f" % (i, cluster_population[i], stat_val, av, pval, dens,
                                                             av_dens, pval_dens))
    stat_val = cluster_id_to_stat['total']
    av = np.mean(perm_array['total'])
    pval = pvalue(perm_array['total'], stat_val)
    print('total %d %1.2f %1.2f %1.4f' % (cluster_population.sum() - cluster_population[0], stat_val, av, pval))
    print()


def pdb_contact_density_enc():
    # if aledo_dist:
    dist_f = dist_aledo
        # prot_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, chain_to_prot)
    chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, chain_to_prot)
    pos_to_coords = chain_to_site_coords[selected_chain]
    max_pos = max(pos_to_coords.keys())
    prot_name = chain_to_prot[selected_chain]
    atp6_data = pd.read_csv(path_to_atp6_data, sep='\t', decimal='.')
    atp6_data['Prot'] = atp6_data['chain'].apply(lambda x: chain_to_prot[x])
    non_burried = atp6_data[(atp6_data['Prot'] == prot_name) & (atp6_data['BCEE'] != 'BURIED')]
    interface = non_burried.loc[(non_burried['BCEE'] == 'CONT') | (non_burried['BCEE'] == 'ENC_interface'), 'pos']
    cluster_ids = np.zeros(max_pos + 1, dtype=int)
    for pos in non_burried['pos']:
        cluster_ids[pos] = 1
    for pos in interface:
        cluster_ids[pos] = 2
    print(prot_name)
    # pos_to_coords = prot_to_site_coords[prot_name]
    neighbors = get_pdb_neighbors(prot_name, prot_to_chain, chain_to_site_coords, dist_f, use_internal_contacts, use_external_contacts)
    # cluster_ids = chain_to_clusters[prot_name]
    cl_num = 3
    cluster_id_to_stat, cluster_population, cluster_densities = cluster_stat12(neighbors, cluster_ids, False)
    # cluster_id_to_stat, cluster_population, cluster_densities = cluster_stat2(neighbors, cluster_ids, False)
    permutations = Parallel(n_jobs=-1)(delayed(cluster_stat12)(neighbors, cluster_ids, True) for i in range(permutations_num))
    # permutations = Parallel(n_jobs=-1)(delayed(cluster_stat2)(neighbors, cluster_ids, True) for i in range(permutations_num))
    perm_array = {}
    density_array = {}
    for i in range(1, cl_num):
        perm_array[i] = []
        density_array[i] = []
    perm_array['total'] = []
    for cl_id_to_stat, cl_population, cl_density in permutations:
        for i in range(1, cl_num):
            perm_array[i].append(cl_id_to_stat[i])
            density_array[i].append(cl_density[i])
        perm_array['total'].append(cl_id_to_stat['total'])
    print('cluster n_sites stat_value expected_value pvalue density_value expected_density pvalue_density')
    for i in range(1, cl_num):
        stat_val = cluster_id_to_stat[i]
        av = np.mean(perm_array[i])
        pval = pvalue(perm_array[i], stat_val)
        dens = cluster_densities[i]
        av_dens = np.mean(density_array[i])
        pval_dens = pvalue(density_array[i], dens)
        print("%d %d %1.2f %1.2f %1.4f %1.2f %1.2f %1.4f" % (i, cluster_population[i], stat_val, av, pval, dens,
                                                             av_dens, pval_dens))
    stat_val = cluster_id_to_stat['total']
    av = np.mean(perm_array['total'])
    pval = pvalue(perm_array['total'], stat_val)
    print('total %d %1.2f %1.2f %1.4f' % (cluster_population.sum() - cluster_population[0], stat_val, av, pval))
    print()


def connected_components():
    dist_f = dist_aledo
        # prot_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, chain_to_prot)
    chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, chain_to_prot)
    prot_name = chain_to_prot[selected_chain]
    atp6_data = pd.read_csv(path_to_atp6_data, sep='\t', decimal='.')
    atp6_data['Prot'] = atp6_data['chain'].apply(lambda x: chain_to_prot[x])
    non_burried = atp6_data[(atp6_data['Prot'] == prot_name) & (atp6_data['BCEE'] != 'BURIED')]
    interface = non_burried.loc[(non_burried['BCEE'] == 'CONT') | (non_burried['BCEE'] == 'ENC_interface'), 'pos']
    nonburied_set = set(non_burried['pos'])
    print(prot_name)
    # pos_to_coords = prot_to_site_coords[prot_name]
    neighbors = get_pdb_neighbors(prot_name, prot_to_chain, chain_to_site_coords, dist_f, use_internal_contacts, use_external_contacts)
    # cluster_ids = chain_to_clusters[prot_name]
    components = []
    for pos in interface:
        found = False
        comp_i = None
        for comp in components:
            if pos in comp:
                comp_i = comp
                found = True
        if not found:
            comp_i = set()
            comp_i.add(pos)
            components.append(comp_i)
        for n in neighbors[pos]:
            if n in nonburied_set:
                comp_i.add(n)
    for comp in components:
        for p in comp:
            print(str(p), end=' ')
        print()


if __name__ == '__main__':
    # pdb_contact_density()
    # pdb_contact_density_enc()
    connected_components()