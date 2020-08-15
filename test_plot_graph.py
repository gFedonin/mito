from os import makedirs, listdir
from os.path import exists

import networkx as nx
from sklearn.externals.joblib import Parallel, delayed

# matplotlib.use("cairo")

from compute_cluster_stats import parse_site2pdb, parse_out, get_pdb_neighbors, dist_aledo, dist, \
    parse_pdb_Aledo_biopython

prot_names = ['atp6']


# pdb_id = '5ara'
# pdb_id = '1be3'
# pdb_id = '1bgy'
# pdb_id = '1occ'
path_to_pdb = '../pdb/'
# path_to_colors = '../Coloring/internal_gaps.2/'
path_to_colors = '../Coloring/G10.4/'
path_to_coevolution_graphs = '../res/G10.4_graphs/'

pdb_to_chain_to_prot = {'5ara': {'W': 'atp6'}}
pdb_to_prot_to_chain = {'5ara': {'atp6': ['W']}}
# chain_to_prot = {'W': 'atp6'}
# chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
# chain_to_prot = {'C': 'cytb', 'O': 'cytb'}

out_path = '../res/graphs_multichain_gml_test/'

aledo_dist = True
dist_threshold = 4

use_internal_contacts = True
use_external_contacts = True


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


def get_filtered_pos_lists():
    prot_to_pos_set = {}
    for pdb_id, chain_to_prot in pdb_to_chain_to_prot.items():
        prot_to_chain = pdb_to_prot_to_chain[pdb_id]
        prot_to_site_map = parse_site2pdb(prot_to_chain, path_to_colors)
        for prot, site2pdb in prot_to_site_map.items():
            sites_in_pdb = site2pdb.keys()
            prot_to_pos_set[prot] = sites_in_pdb
    return prot_to_pos_set


def create_small_graph_with_normalization(prot_name, graph, group_to_nodes, nodes_to_groups, pos_set, pos_only):
    # small = igraph.Graph()
    # small.add_vertices(len(group_to_nodes))
    # i = 0
    # groups = list(group_to_nodes.keys())
    # groups.sort()
    # groups.reverse()
    # for group in groups:
    #     nodes = group_to_nodes[group]
    #     small.vs[i]['name'] = str(group)
    #     small.vs[i]['weight'] = len(nodes)/len(nodes_to_groups)
    #     i += 1
    sum_pos = 0
    sum_neg = 0
    nodes_weights_pos = {}
    nodes_weights_neg = {}
    for group, nodes in group_to_nodes.items():
        for n1 in nodes:
            if n1 not in pos_set:
                continue
            sum_w_pos = 0
            sum_w_neg = 0
            for n2 in graph.neighbors(n1):
                if n2 not in pos_set:
                    continue
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
            if n not in pos_set:
                continue
            s += nodes_weights_pos[n]
        group_weights_pos[group] = s
        s = 0
        for n in nodes:
            if n not in pos_set:
                continue
            s += nodes_weights_neg[n]
        group_weights_neg[group] = s
    for group, nodes in group_to_nodes.items():
        group_edges_pos = {}
        group_edges_neg = {}
        for n1 in nodes:
            if n1 not in pos_set:
                continue
            for n2 in graph.neighbors(n1):
                if n2 not in pos_set:
                    continue
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
                    a = 0
                    # small.add_edge(group, group2)
                    # small.es[len(small.es) - 1]['key'] = 'pos'
                    # n = group_weights_pos[group]
                    # small.es[len(small.es) - 1]['weight'] = 2*w*sum_pos/n/n
                if group < group2:
                    a = 0
                    # small.add_edge(group, group2)
                    # small.es[len(small.es) - 1]['key'] = 'pos'
                    # small.es[len(small.es) - 1]['weight'] = 2*w*sum_pos/group_weights_pos[group2]/group_weights_pos[group]
        else:
            for group2, w in group_edges_neg.items():
                if group == group2:
                    a = 0
                    # small.add_edge(group, group2)
                    # small.es[len(small.es) - 1]['key'] = 'neg'
                    # n = group_weights_neg[group]
                    # small.es[len(small.es) - 1]['weight'] = 2*w*sum_neg/n/n
                if group < group2:
                    a = 0
                    # small.add_edge(group, group2)
                    # small.es[len(small.es) - 1]['key'] = 'neg'
                    # small.es[len(small.es) - 1]['weight'] = 2*w*sum_neg/group_weights_neg[group2]/group_weights_neg[group]
    # return prot_name, small
    return prot_name, None



def plot_graph_nx(graph):
    pos = nx.circular_layout(graph)
    # node_size = [600*len(graph.nodes)*graph.nodes[node]['weight'] for node in graph.nodes]
    # plt.plot()
    nx.draw_networkx_nodes(graph, pos)#, node_size=node_size
    nx.draw_networkx_labels(graph, pos, labels={node: 'G'+str(node) for node in graph.nodes})
    nx.draw_networkx_edges(graph, pos, edgelist=[e for e in graph.edges if e[2] == 'pos'])
    nx.draw_networkx_edges(graph, pos, edgelist=[e for e in graph.edges if e[2] == 'neg'], style='dashed')
    # plt.show()


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
    # contact_graphs = create_contact_graph_with_normalization2()
    prot_to_pos_set = get_filtered_pos_lists()
    coevolution_data = [(prot_name, *read_data_G10(prot_name)) for prot_name in prot_names]
    tasks = Parallel(n_jobs=len(coevolution_data))(delayed(create_small_graph_with_normalization)
                                          (prot_name, graph, group_to_nodes, nodes_to_groups,
                                           prot_to_pos_set[prot_name], True)
                                          for prot_name, graph, group_to_nodes, nodes_to_groups in coevolution_data)
    positive_graphs = {prot_name: g for prot_name, g in tasks}
    tasks = Parallel(n_jobs=len(coevolution_data))(delayed(create_small_graph_with_normalization)
                                          (prot_name, graph, group_to_nodes, nodes_to_groups,
                                           prot_to_pos_set[prot_name], False)
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
        # print_pajek('../res/graphs_dot/' + prot_name + '_pos.net', positive_graphs[prot_name])
        # print_pajek('../res/graphs_dot/' + prot_name + '_neg.net', negative_graphs[prot_name])
        # print_pajek('../res/graphs_dot/' + prot_name + '_cont.net', contact_graphs[prot_name])