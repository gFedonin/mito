from os import makedirs, listdir
from os.path import exists

import networkx as nx
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import igraph
# import numpy as np
# from matplotlib import colors
# from matplotlib.artist import Artist
from sklearn.externals.joblib import Parallel, delayed

# matplotlib.use("cairo")

from compute_cluster_stats import parse_site2pdb, parse_out, get_pdb_neighbors, dist_aledo, dist, \
    parse_pdb_Aledo_biopython

prot_names = ['cox1', 'cox2', 'cox3', 'cytb', 'atp6']


# pdb_id = '5ara'
# pdb_id = '1be3'
# pdb_id = '1bgy'
# pdb_id = '1occ'
path_to_pdb = '../pdb/'
# path_to_colors = '../Coloring/internal_gaps.2/'
path_to_colors = '../Coloring/G10.4/'
path_to_coevolution_graphs = '../res/G10.4_graphs/'

pdb_to_chain_to_prot = {'1occ': {'A': 'cox1', 'N': 'cox1', 'B': 'cox2', 'O': 'cox2', 'C': 'cox3', 'P': 'cox3'},
                        '1bgy': {'C': 'cytb', 'O': 'cytb'},
                        '5ara': {'W': 'atp6'}}
pdb_to_prot_to_chain = {'1occ': {'cox1': ['A', 'N'], 'cox2': ['B', 'O'], 'cox3': ['C', 'P']},
                        '1bgy': {'cytb': ['C', 'O']},
                        '5ara': {'atp6': ['W']}}
# chain_to_prot = {'W': 'atp6'}
# chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
# chain_to_prot = {'C': 'cytb', 'O': 'cytb'}

out_path = '../res/graphs_multichain_gml/'

aledo_dist = True
dist_threshold = 4

use_internal_contacts = True
use_external_contacts = True


# def read_data(prot_name):
#     path_to_pajek_net = path_to_coevolution_graphs + prot_name + '/' + prot_name + '.pcor.up05.net'
#     path_to_pajek_clu = path_to_coevolution_graphs + prot_name + '/' + prot_name + '.pcor.up05.louvain.modularity.clu'
#     graph = nx.Graph(nx.read_pajek(path_to_pajek_net))
#     node_names = []
#     with open(path_to_pajek_net) as f:
#         node_num = int(f.readline().strip().split(' ')[1])
#         for l in f.readlines()[:node_num]:
#             node_names.append(l.strip().split(' ')[1][1:-1])
#     group_to_nodes = {}
#     nodes_to_groups = {}
#     i = 0
#     for line in open(path_to_pajek_clu).readlines()[1:]:
#         group = line.strip()
#         nodes_to_groups[node_names[i]] = group
#         if group in group_to_nodes:
#             group_to_nodes[group].append(node_names[i])
#         else:
#             group_to_nodes[group] = [node_names[i]]
#         i += 1
#     return graph, group_to_nodes, nodes_to_groups


def read_data_G10(prot_name):
    for fname in listdir(path_to_coevolution_graphs):
        if fname.startswith(prot_name) and fname.endswith('.net'):
            path_to_pajek_net = path_to_coevolution_graphs + fname
    for fname in listdir(path_to_colors):
        if fname.startswith(prot_name) and fname.endswith('.out'):
            path_to_out = path_to_colors + fname
    graph = nx.Graph(nx.read_pajek(path_to_pajek_net))
    group_to_nodes = {}
    nodes_to_groups = {}
    for line in open(path_to_out).readlines()[5:]:
        s = line.strip().split()
        group = s[1]
        node = s[0]
        nodes_to_groups[node] = group
        if group in group_to_nodes:
            group_to_nodes[group].append(node)
        else:
            group_to_nodes[group] = [node]
    print(prot_name)
    for group, nodes in group_to_nodes.items():
        print(group + ' have ' + str(len(nodes)))
    return graph, group_to_nodes, nodes_to_groups


def create_small_graph(graph, group_to_nodes, nodes_to_groups):
    # small = nx.MultiDiGraph()
    small = igraph.Graph(directed=True)
    small.add_vertices(len(group_to_nodes))
    i = 0
    for group, nodes in group_to_nodes.items():
        small.vs[i]['name'] = group
        small.vs[i]['weight'] = len(nodes)/len(nodes_to_groups)
        i += 1
        # small.add_node(group, weight=len(nodes)/len(nodes_to_groups))
    for group, nodes in group_to_nodes.items():
        sum_pos = 0
        sum_neg = 0
        group_edges_pos = {}
        group_edges_neg = {}
        for n1 in nodes:
            for n2 in graph.neighbors(n1):
                w = float(graph[n1][n2]['weight'])
                if w > 0:
                    sum_pos += w
                    group2 = nodes_to_groups[n2]
                    if group2 in group_edges_pos:
                        group_edges_pos[group2] += w
                    else:
                        group_edges_pos[group2] = w
                else:
                    sum_neg += w
                    group2 = nodes_to_groups[n2]
                    if group2 in group_edges_neg:
                        group_edges_neg[group2] += w
                    else:
                        group_edges_neg[group2] = w
        # for group2, w in group_edges_pos.items():
        #     small.add_edge(group, group2)
        #     small.es[len(small.es) - 1]['key'] = 'pos'
        #     small.es[len(small.es) - 1]['weight'] = w/sum_pos
        #     # small.add_edge(group, group2, key='pos', weight=w/sum_pos)
        for group2, w in group_edges_neg.items():
            small.add_edge(group, group2)
            small.es[len(small.es) - 1]['key'] = 'neg'
            small.es[len(small.es) - 1]['weight'] = w/sum_neg
            # small.add_edge(group, group2, key='neg', weight=w/sum_neg)
    return small


def create_small_graph_no_normalization(graph, group_to_nodes, nodes_to_groups, pos_only):
    # small = nx.MultiDiGraph()
    small = igraph.Graph()
    small.add_vertices(len(group_to_nodes))
    i = 0
    for group, nodes in group_to_nodes.items():
        small.vs[i]['name'] = group
        small.vs[i]['weight'] = len(nodes)/len(nodes_to_groups)
        i += 1
        # small.add_node(group, weight=len(nodes)/len(nodes_to_groups))
    sum_pos = 0
    sum_neg = 0
    for group, nodes in group_to_nodes.items():
        for n1 in nodes:
            for n2 in graph.neighbors(n1):
                w = float(graph[n1][n2]['weight'])
                if w > 0:
                    sum_pos += w
                else:
                    sum_neg += w
    for group, nodes in group_to_nodes.items():
        group_edges_pos = {}
        group_edges_neg = {}
        for n1 in nodes:
            for n2 in graph.neighbors(n1):
                w = float(graph[n1][n2]['weight'])
                if w > 0:
                    group2 = nodes_to_groups[n2]
                    if group2 in group_edges_pos:
                        group_edges_pos[group2] += w
                    else:
                        group_edges_pos[group2] = w
                else:
                    group2 = nodes_to_groups[n2]
                    if group2 in group_edges_neg:
                        group_edges_neg[group2] += w
                    else:
                        group_edges_neg[group2] = w
        if pos_only:
            for group2, w in group_edges_pos.items():
                if group <= group2:
                    small.add_edge(group, group2)
                    small.es[len(small.es) - 1]['key'] = 'pos'
                    small.es[len(small.es) - 1]['weight'] = w/sum_pos
                    # small.add_edge(group, group2, key='pos', weight=w/sum_pos)
        else:
            for group2, w in group_edges_neg.items():
                if group <= group2:
                    small.add_edge(group, group2)
                    small.es[len(small.es) - 1]['key'] = 'neg'
                    small.es[len(small.es) - 1]['weight'] = w/sum_neg
                    # small.add_edge(group, group2, key='neg', weight=w/sum_neg)
    return small


def create_small_graph_with_normalization(prot_name, graph, group_to_nodes, nodes_to_groups, pos_only):
    # small = nx.MultiDiGraph()
    small = igraph.Graph()
    small.add_vertices(len(group_to_nodes))
    i = 0
    groups = list(group_to_nodes.keys())
    groups.sort()
    groups.reverse()
    for group in groups:
        nodes = group_to_nodes[group]
        small.vs[i]['name'] = str(group)
        small.vs[i]['weight'] = len(nodes)/len(nodes_to_groups)
        i += 1
        # small.add_node(group, weight=len(nodes)/len(nodes_to_groups))
    sum_pos = 0
    sum_neg = 0
    nodes_weights_pos = {}
    nodes_weights_neg = {}
    for group, nodes in group_to_nodes.items():
        for n1 in nodes:
            sum_w_pos = 0
            sum_w_neg = 0
            for n2 in graph.neighbors(n1):
                w = float(graph[n1][n2]['weight'])
                if w > 0:
                    sum_pos += w
                    sum_w_pos += w
                else:
                    sum_neg += w
                    sum_w_neg += w
            nodes_weights_pos[n1] = sum_w_pos
            nodes_weights_neg[n1] = sum_w_neg
    sum_pos /= 2
    sum_neg /= 2
    group_weights_pos = {}
    group_weights_neg = {}
    for group, nodes in group_to_nodes.items():
        s = 0
        for n in nodes:
            s += nodes_weights_pos[n]
        group_weights_pos[group] = s
        s = 0
        for n in nodes:
            s += nodes_weights_neg[n]
        group_weights_neg[group] = s
    for group, nodes in group_to_nodes.items():
        group_edges_pos = {}
        group_edges_neg = {}
        for n1 in nodes:
            for n2 in graph.neighbors(n1):
                w = float(graph[n1][n2]['weight'])
                if w > 0:
                    group2 = nodes_to_groups[n2]
                    if group2 in group_edges_pos:
                        group_edges_pos[group2] += w
                    else:
                        group_edges_pos[group2] = w
                else:
                    group2 = nodes_to_groups[n2]
                    if group2 in group_edges_neg:
                        group_edges_neg[group2] += w
                    else:
                        group_edges_neg[group2] = w
        if pos_only:
            for group2, w in group_edges_pos.items():
                if group == group2:
                    small.add_edge(group, group2)
                    small.es[len(small.es) - 1]['key'] = 'pos'
                    n = group_weights_pos[group]
                    small.es[len(small.es) - 1]['weight'] = 2*w*sum_pos/n/n
                    # small.add_edge(group, group2, key='pos', weight=w/sum_pos)
                if group < group2:
                    small.add_edge(group, group2)
                    small.es[len(small.es) - 1]['key'] = 'pos'
                    small.es[len(small.es) - 1]['weight'] = 2*w*sum_pos/group_weights_pos[group2]/group_weights_pos[group]
                    # small.add_edge(group, group2, key='pos', weight=w/sum_pos)
        else:
            for group2, w in group_edges_neg.items():
                if group == group2:
                    small.add_edge(group, group2)
                    small.es[len(small.es) - 1]['key'] = 'neg'
                    n = group_weights_neg[group]
                    small.es[len(small.es) - 1]['weight'] = 2*w*sum_neg/n/n
                    # small.add_edge(group, group2, key='neg', weight=w/sum_neg)
                if group < group2:
                    small.add_edge(group, group2)
                    small.es[len(small.es) - 1]['key'] = 'neg'
                    small.es[len(small.es) - 1]['weight'] = 2*w*sum_neg/group_weights_neg[group2]/group_weights_neg[group]
                    # small.add_edge(group, group2, key='neg', weight=w/sum_neg)
    return prot_name, small



# def create_contact_graph_no_normalization():
#     if aledo_dist:
#         chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, chain_to_prot)
#     else:
#         chain_to_site_coords = parse_pdb(chain_to_prot, path_to_pdb)
#     prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
#     res = {}
#     for prot_name, method_name, cluster_ids in prot_to_clusters:
#         pos_to_coords = chain_to_site_coords[prot_name]
#         group_to_nodes = {}
#         node_num = 0
#         for i in range(len(cluster_ids)):
#             g = cluster_ids[i]
#             if g > 0:
#                 node_num += 1
#                 if g in group_to_nodes:
#                     group_to_nodes[g].append(i)
#                 else:
#                     group_to_nodes[g] = [i]
#         if aledo_dist:
#             neighbors = get_pdb_neighbors(pos_to_coords, dist_aledo)
#         else:
#             neighbors = get_pdb_neighbors(pos_to_coords, dist)
#         small = igraph.Graph()
#         small.add_vertices(len(group_to_nodes))
#         i = 0
#         for group, nodes in group_to_nodes.items():
#             small.vs[i]['name'] = str(group)
#             small.vs[i]['weight'] = len(nodes)/node_num
#             i += 1
#             # small.add_node(group, weight=len(nodes)/len(nodes_to_groups))
#         sum = 0
#         for group, nodes in group_to_nodes.items():
#             for n1 in nodes:
#                 sum += len(neighbors[n1])
#         for group, nodes in group_to_nodes.items():
#             edge_weights = {}
#             for n1 in nodes:
#                 for n2 in neighbors[n1]:
#                     if n2 < len(cluster_ids):
#                         group2 = cluster_ids[n2]
#                         if group2 in edge_weights:
#                             edge_weights[group2] += 1
#                         else:
#                             edge_weights[group2] = 1
#             for group2, w in edge_weights.items():
#                 if group <= group2:
#                     small.add_edge(str(group), str(group2))
#                     small.es[len(small.es) - 1]['weight'] = w/sum
#         res[prot_name] = small
#     return res
#
#
# def create_contact_graph_with_normalization():
#     if aledo_dist:
#         chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, chain_to_prot)
#     else:
#         chain_to_site_coords = parse_pdb(chain_to_prot, path_to_pdb)
#     prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
#     res = {}
#     for prot_name, method_name, cluster_ids in prot_to_clusters:
#         pos_to_coords = chain_to_site_coords[prot_name]
#         group_to_nodes = {}
#         node_num = 0
#         for i in range(len(cluster_ids)):
#             g = cluster_ids[i]
#             if g > 0:
#                 node_num += 1
#                 if g in group_to_nodes:
#                     group_to_nodes[g].append(i)
#                 else:
#                     group_to_nodes[g] = [i]
#         if aledo_dist:
#             neighbors = get_pdb_neighbors(pos_to_coords, dist_aledo)
#         else:
#             neighbors = get_pdb_neighbors(pos_to_coords, dist)
#         small = igraph.Graph()
#         small.add_vertices(len(group_to_nodes))
#         i = 0
#         for group, nodes in group_to_nodes.items():
#             small.vs[i]['name'] = str(group)
#             small.vs[i]['weight'] = len(nodes)/node_num
#             i += 1
#             # small.add_node(group, weight=len(nodes)/len(nodes_to_groups))
#         sum = 0
#         for group, nodes in group_to_nodes.items():
#             for n1 in nodes:
#                 sum += len(neighbors[n1])
#         for group, nodes in group_to_nodes.items():
#             edge_counts = {}
#             for n1 in nodes:
#                 for n2 in neighbors[n1]:
#                     if n2 < len(cluster_ids):
#                         group2 = cluster_ids[n2]
#                         if group2 in edge_counts:
#                             edge_counts[group2] += 1
#                         else:
#                             edge_counts[group2] = 1
#             for group2, w in edge_counts.items():
#                 if group == group2:
#                     n1 = len(group_to_nodes[group])
#                     small.add_edge(str(group), str(group2))
#                     small.es[len(small.es) - 1]['weight'] = 2*w/(n1*(n1 - 1))
#                 if group < group2:
#                     n1 = len(group_to_nodes[group])
#                     n2 = len(group_to_nodes[group2])
#                     small.add_edge(str(group), str(group2))
#                     small.es[len(small.es) - 1]['weight'] = w/(n1*n2)
#         res[prot_name] = small
#     return res
#
#
# def create_contact_graph_with_normalization1():
#     if aledo_dist:
#         chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, chain_to_prot)
#     else:
#         chain_to_site_coords = parse_pdb(chain_to_prot, path_to_pdb)
#     prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)
#     res = {}
#     for prot_name, method_name, cluster_ids in prot_to_clusters:
#         pos_to_coords = chain_to_site_coords[prot_name]
#         group_to_nodes = {}
#         node_num = 0
#         for i in range(len(cluster_ids)):
#             g = cluster_ids[i]
#             if g > 0:
#                 node_num += 1
#                 if g in group_to_nodes:
#                     group_to_nodes[g].append(i)
#                 else:
#                     group_to_nodes[g] = [i]
#         if aledo_dist:
#             neighbors = get_pdb_neighbors(pos_to_coords, dist_aledo)
#         else:
#             neighbors = get_pdb_neighbors(pos_to_coords, dist)
#         small = igraph.Graph()
#         small.add_vertices(len(group_to_nodes))
#         i = 0
#         for group, nodes in group_to_nodes.items():
#             small.vs[i]['name'] = str(group)
#             small.vs[i]['weight'] = len(nodes)/node_num
#             i += 1
#             # small.add_node(group, weight=len(nodes)/len(nodes_to_groups))
#         sum = 0
#         for group, nodes in group_to_nodes.items():
#             for n1 in nodes:
#                 sum += len(neighbors[n1])
#         max_edge_num = 0
#         for n_list in neighbors.values():
#             if len(n_list) > max_edge_num:
#                 max_edge_num = len(n_list)
#         for group, nodes in group_to_nodes.items():
#             edge_counts = {}
#             for n1 in nodes:
#                 for n2 in neighbors[n1]:
#                     if n2 < len(cluster_ids):
#                         group2 = cluster_ids[n2]
#                         if group2 in edge_counts:
#                             edge_counts[group2] += 1
#                         else:
#                             edge_counts[group2] = 1
#             for group2, w in edge_counts.items():
#                 if group == group2:
#                     n1 = len(group_to_nodes[group])
#                     small.add_edge(str(group), str(group))
#                     small.es[len(small.es) - 1]['weight'] = w/min(n1*(n1 - 1)/2, max_edge_num*n1)
#                 if group < group2:
#                     n1 = len(group_to_nodes[group])
#                     n2 = len(group_to_nodes[group2])
#                     small.add_edge(str(group), str(group2))
#                     small.es[len(small.es) - 1]['weight'] = w/(min(max_edge_num*n1, max_edge_num*n2, n1*n2))
#         res[prot_name] = small
#     return res


def create_a_contact_graph(prot_name, cluster_ids, prot_to_chain, chain_to_site_coords, dist_f):
    group_to_nodes = {}
    node_num = 0
    for i in range(len(cluster_ids)):
        g = cluster_ids[i]
        if g > 0:
            node_num += 1
            if g in group_to_nodes:
                group_to_nodes[g].append(i)
            else:
                group_to_nodes[g] = [i]
    # if aledo_dist:
    #     neighbors = get_pdb_neighbors(pos_to_coords, dist_aledo)
    # else:
    #     neighbors = get_pdb_neighbors(pos_to_coords, dist)
    neighbors = get_pdb_neighbors(prot_name, prot_to_chain, chain_to_site_coords, dist_f, use_internal_contacts,
                      use_external_contacts)
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
    sum = 0
    group_weights = {}
    for group, nodes in group_to_nodes.items():
        weight = 0
        for n1 in nodes:
            if n1 in neighbors:
                weight += len(neighbors[n1])
                sum += len(neighbors[n1])
        group_weights[group] = weight
    sum /= 2
    for group, nodes in group_to_nodes.items():
        edge_counts = {}
        for n1 in nodes:
            if n1 in neighbors:
                for n2 in neighbors[n1]:
                    if n2 < len(cluster_ids):
                        group2 = cluster_ids[n2]
                        if group2 in edge_counts:
                            edge_counts[group2] += 1
                        else:
                            edge_counts[group2] = 1
        for group2, w in edge_counts.items():
            if group == group2:
                n1 = group_weights[group]
                small.add_edge(str(group), str(group2))
                small.es[len(small.es) - 1]['weight'] = 2 * w * sum / n1 / n1
            if group < group2:
                small.add_edge(str(group), str(group2))
                small.es[len(small.es) - 1]['weight'] = 2 * w * sum / group_weights[group] / group_weights[group2]
    return prot_name, small


def create_contact_graph_with_normalization2():
    dist_f = dist_aledo
    contact_data = []
    for pdb_id, chain_to_prot in pdb_to_chain_to_prot.items():
        prot_to_chain = pdb_to_prot_to_chain[pdb_id]
        # if aledo_dist:
        chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb + pdb_id + '.pdb', chain_to_prot)
        # else:
        #     chain_to_site_coords = parse_pdb(chain_to_prot, path_to_pdb + pdb_id + '.pdb')
        prot_to_clusters = parse_out(parse_site2pdb(prot_to_chain, path_to_colors), prot_to_chain, path_to_colors)

        for prot_name, method_name, cluster_ids in prot_to_clusters:

            contact_data.append((prot_name, cluster_ids, prot_to_chain, chain_to_site_coords))
    tasks = Parallel(n_jobs=len(contact_data))(delayed(create_a_contact_graph)(prot_name, cluster_ids, prot_to_chain,
                                                                               chain_to_site_coords, dist_f)
                                       for prot_name, cluster_ids, prot_to_chain, chain_to_site_coords in contact_data)
    return {prot_name: graph for prot_name, graph in tasks}


def plot_graph_nx(graph):
    pos = nx.circular_layout(graph)
    # node_size = [600*len(graph.nodes)*graph.nodes[node]['weight'] for node in graph.nodes]
    # plt.plot()
    nx.draw_networkx_nodes(graph, pos)#, node_size=node_size
    nx.draw_networkx_labels(graph, pos, labels={node: 'G'+str(node) for node in graph.nodes})
    nx.draw_networkx_edges(graph, pos, edgelist=[e for e in graph.edges if e[2] == 'pos'])
    nx.draw_networkx_edges(graph, pos, edgelist=[e for e in graph.edges if e[2] == 'neg'], style='dashed')
    # plt.show()


def plot_igraph(graph, out_path, edge_width):
    visual_style = {}
    # graph.vs['size'] = [100 * len(graph.vs) * w for w in graph.vs['weight']]
    visual_style["vertex_size"] = [75 * len(graph.vs) * w for w in graph.vs['weight']]
    visual_style["vertex_label"] = ['G' + n for n in graph.vs["name"]]
    visual_style["edge_width"] = [edge_width*w for w in graph.es['weight']]
    # graph.es['width'] = [10*w for w in graph.es['weight']]
    visual_style["layout"] = graph.layout_circle()
    # color_dict = {"pos": "blue", "neg": "green"}
    # visual_style["edge_color"] = [color_dict[key] for key in graph.es["key"]]
    visual_style["bbox"] = (1000, 1000)
    visual_style["margin"] = 175
    igraph.plot(graph, out_path, **visual_style)


def plot_igraph_contact(prot_name, graph, base_edge_width):
    visual_style = {}
    # graph.vs['size'] = [100 * len(graph.vs) * w for w in graph.vs['weight']]
    visual_style["vertex_size"] = [75 * len(graph.vs) * w for w in graph.vs['weight']]
    visual_style["vertex_label"] = ['G' + n for n in graph.vs["name"]]
    visual_style["edge_width"] = [base_edge_width*w for w in graph.es['weight']]
    # graph.es['width'] = [10*w for w in graph.es['weight']]
    visual_style["layout"] = graph.layout_circle()
    visual_style["bbox"] = (1000, 1000)
    visual_style["margin"] = 175
    igraph.plot(graph, '../res/graphs/' + prot_name + '_contacts_with_norm.png', **visual_style)


def plot_graph(prot_name, graph, base_edge_width, edge_color):
    # Create the figure
    visual_style = {}
    visual_style["vertex_size"] = [400 * w for w in graph.vs['weight']]
    visual_style["vertex_label"] = [n for n in graph.vs["name"]]
    visual_style["label_size"] = 14
    visual_style['vertex_color'] = 'white'
    visual_style["edge_width"] = [base_edge_width*w for w in graph.es['weight']]
    visual_style['edge_color'] = edge_color
    visual_style["layout"] = graph.layout_circle()
    visual_style["bbox"] = (1000, 1000)
    visual_style["margin"] = 300
    igraph.plot(graph, '../res/graphs_thick/' + prot_name + '_with_norm.png', **visual_style)


# def merge_plots(prot_name):
#     plot1 = plt.imread('../res/graphs_thick/' + prot_name + '_pos_with_norm.png', format='png')
#     plot2 = plt.imread('../res/graphs_thick/' + prot_name + '_neg_with_norm.png', format='png')
#     plot3 = plt.imread('../res/graphs_thick/' + prot_name + '_cont_with_norm.png', format='png')
#     fig = plt.figure(figsize=(10, 10))
#     axes = fig.add_subplot(131)
#     axes.imshow(plot1, interpolation='nearest')
#     plt.axis('off')
#     axes = fig.add_subplot(132)
#     axes.imshow(plot2, interpolation='nearest')
#     plt.axis('off')
#     axes = fig.add_subplot(133)
#     axes.imshow(plot3, interpolation='nearest')
#     plt.axis('off')
#     fig.savefig('../res/graphs_thick/' + prot_name + '.png')


def print_gml(out_path, graph):
    graph.write_gml(out_path)


def print_graphviz(out_path, graph):
    graph.write_dot(out_path)


def print_pajek(out_path, graph):
    graph.write_pajek(out_path)


if __name__ == '__main__':
    if not exists(out_path):
        makedirs(out_path)
    # for prot_name in prot_names:
    #     print(prot_name)
    #     graph, group_to_nodes, nodes_to_groups = read_data(prot_name)
    #     out_path = '../res/graphs/' + prot_name + '_with_norm_pos.png'
    #     small_graph = create_small_graph_with_normalization(graph, group_to_nodes, nodes_to_groups, True)
    #     plot_igraph(small_graph, out_path, 5)
    #     out_path = '../res/graphs/' + prot_name + '_with_norm_neg.png'
    #     small_graph = create_small_graph_with_normalization(graph, group_to_nodes, nodes_to_groups, False)
    #     plot_igraph(small_graph, out_path, 5)
    contact_graphs = create_contact_graph_with_normalization2()
    coevolution_data = [(prot_name, *read_data_G10(prot_name)) for prot_name in prot_names]
    tasks = Parallel(n_jobs=len(coevolution_data))(delayed(create_small_graph_with_normalization)
                                          (prot_name, graph, group_to_nodes, nodes_to_groups, True)
                                          for prot_name, graph, group_to_nodes, nodes_to_groups in coevolution_data)
    positive_graphs = {prot_name: g for prot_name, g in tasks}
    tasks = Parallel(n_jobs=len(coevolution_data))(delayed(create_small_graph_with_normalization)
                                          (prot_name, graph, group_to_nodes, nodes_to_groups, False)
                                          for prot_name, graph, group_to_nodes, nodes_to_groups in coevolution_data)
    negative_graphs = {prot_name: g for prot_name, g in tasks}
    # for prot_name, graph in small_conact_graphs.items():
    #     plot_graphs(prot_name, [graph], 50)
    for prot_name in prot_names:
        print(prot_name)
        # graphs = [positive_graphs[prot_name], negative_graphs[prot_name], contact_graphs[prot_name]]
        # graph, group_to_nodes, nodes_to_groups = read_data(prot_name)
        # graphs.append(create_small_graph_with_normalization(graph, group_to_nodes, nodes_to_groups, True))
        # graphs.append(create_small_graph_with_normalization(graph, group_to_nodes, nodes_to_groups, False))
        # plot_graphs(prot_name, graphs, 10)
        # plot_graph(prot_name + '_pos', positive_graphs[prot_name], 10, 'red')
        # plot_graph(prot_name + '_neg', negative_graphs[prot_name], 10, 'blue')
        # plot_graph(prot_name + '_cont', contact_graphs[prot_name], 10, 'black')
        # merge_plots(prot_name)
        print_gml(out_path + prot_name + '_pos.gml', positive_graphs[prot_name])
        print_gml(out_path + prot_name + '_neg.gml', negative_graphs[prot_name])
        print_gml(out_path + prot_name + '_cont.gml', contact_graphs[prot_name])
        # print_pajek('../res/graphs_dot/' + prot_name + '_pos.net', positive_graphs[prot_name])
        # print_pajek('../res/graphs_dot/' + prot_name + '_neg.net', negative_graphs[prot_name])
        # print_pajek('../res/graphs_dot/' + prot_name + '_cont.net', contact_graphs[prot_name])