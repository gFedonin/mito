from os import makedirs
from os.path import exists
from random import shuffle

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import igraph
import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from compute_cluster_stats import parse_pdb, parse_site2pdb, parse_out, get_pdb_neighbors, dist_aledo, dist, \
    parse_pdb_Aledo_biopython
from plot_graph_color import MidpointNormalize

prot_names = ['cox1', 'cox2', 'cox3', 'cytb', 'atp6']
path_to_colors = '../Coloring/internal_gaps.2/'
path_to_pdb = '../pdb/'
out_dir = '../res/graphs_width_and_color_perm/'

pdb_to_chain_to_prot = {'1occ': {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}, '1bgy': {'C': 'cytb'}, '5ara': {'W': 'atp6'}}

aledo_dist = True
dist_threshold = 4
iter_num = 10000
thread_num = 32


def read_data(prot_name):
    path_to_pajek_net = '../res/graphs/' + prot_name + '/' + prot_name + '.pcor.up05.net'
    path_to_pajek_clu = '../res/graphs/' + prot_name + '/' + prot_name + '.pcor.up05.louvain.modularity.clu'
    graph = nx.Graph(nx.read_pajek(path_to_pajek_net))
    node_names = []
    with open(path_to_pajek_net) as f:
        node_num = int(f.readline().strip().split(' ')[1])
        for l in f.readlines()[:node_num]:
            node_names.append(l.strip().split(' ')[1][1:-1])
    node_to_group = {}
    i = 0
    max_node_id = 0
    for line in open(path_to_pajek_clu).readlines()[1:]:
        group = int(line.strip())
        node_id = int(node_names[i])
        node_to_group[node_id] = group
        if node_id > max_node_id:
            max_node_id = node_id
        i += 1
    cluster_id = np.zeros(max_node_id + 1, dtype=int)
    for node_id, group in node_to_group.items():
        cluster_id[node_id] = group
    return graph, cluster_id


def create_small_graph_with_normalization(graph, cluster_id, permute=False):
    nodes_to_groups = cluster_id
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
        nodes_to_groups = shuffled
        # print(pos_to_cluster_id)
    group_to_nodes = {}
    node_num = 0
    for i in range(len(nodes_to_groups)):
        cl_id = nodes_to_groups[i]
        if cl_id > 0:
            node_num += 1
            nodes = group_to_nodes.get(cl_id)
            if nodes is None:
                nodes = []
                group_to_nodes[cl_id] = nodes
            nodes.append(i)
    group_num = len(group_to_nodes)
    small_pos = igraph.Graph()
    small_neg = igraph.Graph()
    small_pos.add_vertices(group_num)
    small_neg.add_vertices(group_num)
    i = 0
    groups = list(group_to_nodes.keys())
    groups.sort()
    groups.reverse()
    for group in groups:
        nodes = group_to_nodes[group]
        small_pos.vs[i]['name'] = str(group)
        small_pos.vs[i]['weight'] = len(nodes)/node_num
        small_neg.vs[i]['name'] = str(group)
        small_neg.vs[i]['weight'] = len(nodes)/node_num
        i += 1
        # small.add_node(group, weight=len(nodes)/len(nodes_to_groups))
    i = 0
    j = 0
    for group, nodes in group_to_nodes.items():
        gstr = str(group)
        group_edges_pos = {}
        group_edges_neg = {}
        for n1 in nodes:
            n1str = str(n1)
            for n2 in graph.neighbors(n1str):
                w = float(graph[n1str][n2]['weight'])
                if w > 0:
                    group2 = nodes_to_groups[int(n2)]
                    if group2 in group_edges_pos:
                        group_edges_pos[group2] += w
                    else:
                        group_edges_pos[group2] = w
                else:
                    group2 = nodes_to_groups[int(n2)]
                    if group2 in group_edges_neg:
                        group_edges_neg[group2] += w
                    else:
                        group_edges_neg[group2] = w

        for group2, w in group_edges_pos.items():
            if group <= group2:
                small_pos.add_edge(gstr, str(group2))
                small_pos.es[i]['weight'] = w
                i += 1
        for group2, w in group_edges_neg.items():
            if group <= group2:
                small_neg.add_edge(gstr, str(group2))
                small_neg.es[j]['weight'] = w
                j += 1
    return small_pos, small_neg


def create_a_contact_graph(cluster_id, neighbors, permute=False):
    nodes_to_groups = cluster_id
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
        nodes_to_groups = shuffled
        # print(pos_to_cluster_id)
    group_to_nodes = {}
    node_num = 0
    for i in range(len(nodes_to_groups)):
        cl_id = nodes_to_groups[i]
        if cl_id > 0:
            node_num += 1
            nodes = group_to_nodes.get(cl_id)
            if nodes is None:
                nodes = []
                group_to_nodes[cl_id] = nodes
            nodes.append(i)

    small = igraph.Graph()
    small.add_vertices(len(group_to_nodes))
    i = 0
    groups = list(group_to_nodes.keys())
    groups.sort()
    groups.reverse()
    for group in groups:
        nodes = group_to_nodes[group]
        small.vs[i]['name'] = str(group)
        small.vs[i]['weight'] = len(nodes) / node_num
        i += 1
        # small.add_node(group, weight=len(nodes)/len(nodes_to_groups))
    i = 0
    for group, nodes in group_to_nodes.items():
        edge_counts = {}
        for n1 in nodes:
            if n1 in neighbors:
                for n2 in neighbors[n1]:
                    if n2 < node_num:
                        group2 = nodes_to_groups[n2]
                        if group2 == 0:
                            continue
                        if group2 in edge_counts:
                            edge_counts[group2] += 1
                        else:
                            edge_counts[group2] = 1
        for group2, w in edge_counts.items():
            if group <= group2:
                small.add_edge(str(group), str(group2))
                small.es[i]['weight'] = w
                i += 1
    return small


def create_contact_graph_with_normalization2():
    prot_to_graph = {}
    for pdb_id, chain_to_prot in pdb_to_chain_to_prot.items():
        if aledo_dist:
            chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb + pdb_id + '.pdb', chain_to_prot)
        else:
            chain_to_site_coords = parse_pdb(chain_to_prot, path_to_pdb + pdb_id + '.pdb')


        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)

        for prot_name, method_name, cluster_ids in prot_to_clusters:
            if aledo_dist:
                neighbors = get_pdb_neighbors(chain_to_site_coords[prot_name], dist_aledo)
            else:
                neighbors = get_pdb_neighbors(chain_to_site_coords[prot_name], dist)
            real_graph = create_a_contact_graph(cluster_ids, neighbors)
            edge_num = len(real_graph.es)
            random_graphs = Parallel(n_jobs=thread_num, batch_size=iter_num//thread_num + 1)(delayed(create_a_contact_graph)(cluster_ids, neighbors, True)
                                                for k in range(iter_num))
            real_graph.es['expected'] = [0 for j in range(edge_num)]
            for random_graph in random_graphs:
                for j in range(edge_num):
                    edge = real_graph.es[j]
                    eid = random_graph.get_eid(edge.source, edge.target, directed=False, error=False)
                    if eid != -1:
                        edge['expected'] += random_graph.es[eid]['weight']
            for j in range(edge_num):
                real_graph.es[j]['weight'] = iter_num*real_graph.es[j]['weight']/real_graph.es[j]['expected']
            prot_to_graph[prot_name] = real_graph
    return prot_to_graph


def plot_graph(prot_name, graph, base_edge_width):
    min_weight = min(graph.es['weight'])
    max_weight = max(graph.es['weight'])
    norm = MidpointNormalize(vmin=min_weight, vmax=max_weight, midpoint=1.0)
    m = cm.ScalarMappable(norm=norm, cmap=cm.bwr)#seismic
    m.set_array(np.asarray(graph.es['weight']))
    fig = plt.figure(figsize=(10, 10))
    axes = fig.add_subplot(111)
    plt.axis('off')
    cb = plt.colorbar(m, ax=axes, orientation="horizontal")
    cb.ax.tick_params(labelsize=22)
    fig.savefig(out_dir + prot_name + '_colorbar.png')
    visual_style = {}
    visual_style["vertex_size"] = [400 * w for w in graph.vs['weight']]
    visual_style["vertex_label"] = [n for n in graph.vs["name"]]
    visual_style['vertex_color'] = 'white'
    visual_style["label_size"] = 100
    visual_style["edge_width"] = [base_edge_width*w for w in graph.es['weight']]
    visual_style['edge_color'] = [m.to_rgba(w) for w in graph.es['weight']]
    visual_style["layout"] = graph.layout_circle()
    visual_style["bbox"] = (1000, 1000)
    visual_style["margin"] = 200
    igraph.plot(graph, out_dir + prot_name + '_color.png', **visual_style)


if __name__ == '__main__':
    if not exists(out_dir):
        makedirs(out_dir)
    contact_graphs = create_contact_graph_with_normalization2()
    for prot_name in prot_names:
        graph, cluster_ids = read_data(prot_name)
        small_pos, small_neg = create_small_graph_with_normalization(graph, cluster_ids)
        edge_num_pos = len(small_pos.es)
        small_pos.es['expected'] = [0 for j in range(edge_num_pos)]
        edge_num_neg = len(small_neg.es)
        small_neg.es['expected'] = [0 for j in range(edge_num_neg)]
        random_graphs = Parallel(n_jobs=thread_num, batch_size=iter_num//thread_num + 1)(delayed(create_small_graph_with_normalization)(graph, cluster_ids, True)
                                            for i in range(iter_num))
        for random_pos, random_neg in random_graphs:
            for j in range(edge_num_pos):
                edge = small_pos.es[j]
                eid = random_pos.get_eid(edge.source, edge.target, directed=False, error=False)
                if eid != -1:
                    small_pos.es[j]['expected'] += random_pos.es[eid]['weight']
            for j in range(edge_num_neg):
                edge = small_neg.es[j]
                eid = random_neg.get_eid(edge.source, edge.target, directed=False, error=False)
                if eid != -1:
                    small_neg.es[j]['expected'] += random_neg.es[eid]['weight']
        for j in range(edge_num_pos):
            small_pos.es[j]['weight'] = iter_num * small_pos.es[j]['weight'] / small_pos.es[j]['expected']
        for j in range(edge_num_neg):
            small_neg.es[j]['weight'] = iter_num * small_neg.es[j]['weight'] / small_neg.es[j]['expected']
        print(prot_name)
        plot_graph(prot_name + '_pos', small_pos, 10)
        plot_graph(prot_name + '_neg', small_neg, 10)
        plot_graph(prot_name + '_cont', contact_graphs[prot_name], 10)