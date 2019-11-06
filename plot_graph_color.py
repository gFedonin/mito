import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import igraph
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from matplotlib.colors import Normalize

from compute_cluster_stats import parse_pdb, parse_site2pdb, parse_out, get_pdb_neighbors, dist_aledo, dist, \
    parse_pdb_Aledo_biopython

prot_names = ['cox1', 'cox2', 'cox3', 'cytb', 'atp6']
path_to_colors = '../Coloring/internal_gaps.2/'
path_to_pdb = '../pdb/'

pdb_to_chain_to_prot = {'1occ': {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}, '1bgy': {'C': 'cytb'}, '5ara': {'W': 'atp6'}}

aledo_dist = True
dist_threshold = 4


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def read_data(prot_name):
    path_to_pajek_net = '../res/graphs/' + prot_name + '/' + prot_name + '.pcor.up05.net'
    path_to_pajek_clu = '../res/graphs/' + prot_name + '/' + prot_name + '.pcor.up05.louvain.modularity.clu'
    graph = nx.Graph(nx.read_pajek(path_to_pajek_net))
    node_names = []
    with open(path_to_pajek_net) as f:
        node_num = int(f.readline().strip().split(' ')[1])
        for l in f.readlines()[:node_num]:
            node_names.append(l.strip().split(' ')[1][1:-1])
    group_to_nodes = {}
    nodes_to_groups = {}
    i = 0
    for line in open(path_to_pajek_clu).readlines()[1:]:
        group = line.strip()
        nodes_to_groups[node_names[i]] = group
        if group in group_to_nodes:
            group_to_nodes[group].append(node_names[i])
        else:
            group_to_nodes[group] = [node_names[i]]
        i += 1
    return graph, group_to_nodes, nodes_to_groups


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


def create_a_contact_graph(prot_name, cluster_ids, pos_to_coords):
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
    if aledo_dist:
        neighbors = get_pdb_neighbors(pos_to_coords, dist_aledo)
    else:
        neighbors = get_pdb_neighbors(pos_to_coords, dist)
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
                        if group2 == 0:
                            continue
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
    contact_data = []
    for pdb_id, chain_to_prot in pdb_to_chain_to_prot.items():
        if aledo_dist:
            chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb + pdb_id + '.pdb', chain_to_prot)
        else:
            chain_to_site_coords = parse_pdb(chain_to_prot, path_to_pdb + pdb_id + '.pdb')
        prot_to_clusters = parse_out(parse_site2pdb(chain_to_prot, path_to_colors), chain_to_prot, path_to_colors)

        for prot_name, method_name, cluster_ids in prot_to_clusters:
            contact_data.append((prot_name, cluster_ids, chain_to_site_coords[prot_name]))
    tasks = Parallel(n_jobs=len(contact_data))(delayed(create_a_contact_graph)(prot_name, cluster_ids, pos_to_coords)
                                       for prot_name, cluster_ids, pos_to_coords in contact_data)
    return {prot_name: graph for prot_name, graph in tasks}


def plot_graph(out_dir, prot_name, graph, base_edge_width, m=None):
    if m is None:
        min_weight = min(graph.es['weight'])
        max_weight = max(graph.es['weight'])
        norm = MidpointNormalize(vmin=min_weight, vmax=max_weight, midpoint=1.0)
        m = cm.ScalarMappable(norm=norm, cmap=cm.seismic)#bwr
        m.set_array(np.asarray(graph.es['weight']))
        fig = plt.figure(figsize=(10,10))
        axes = fig.add_subplot(111)
        plt.axis('off')
        plt.colorbar(m, ax=axes, orientation="horizontal")
        fig.savefig(out_dir + prot_name + '_colorbar.png')
    visual_style = {}
    # graph.vs['size'] = [100 * len(graph.vs) * w for w in graph.vs['weight']]
    visual_style["vertex_size"] = [400 * w for w in graph.vs['weight']]
    visual_style["vertex_label"] = [n for n in graph.vs["name"]]
    visual_style['vertex_color'] = 'white'
    visual_style["label_size"] = 14
    visual_style["edge_width"] = base_edge_width
    visual_style['edge_color'] = [m.to_rgba(w) for w in graph.es['weight']]
    # graph.es['width'] = [10*w for w in graph.es['weight']]
    visual_style["layout"] = graph.layout_circle()
    visual_style["bbox"] = (1000, 1000)
    visual_style["margin"] = 300
    igraph.plot(graph, out_dir + prot_name + '_color.png', **visual_style)


def merge_plots(prot_name, m):
    plot1 = plt.imread('../res/graphs_color/' + prot_name + '_pos_with_norm_color.png', format='png')
    plot2 = plt.imread('../res/graphs_color/' + prot_name + '_neg_with_norm_color.png', format='png')
    plot3 = plt.imread('../res/graphs_color/' + prot_name + '_cont_with_norm_color.png', format='png')
    fig = plt.figure(figsize=(20, 20))
    axes = fig.add_subplot(131)

    axes.imshow(plot1, interpolation='nearest')
    plt.colorbar(m, ax=axes, orientation="horizontal", pad=0.01)
    plt.axis('off')
    axes = fig.add_subplot(132)

    axes.imshow(plot2, interpolation='nearest')
    plt.colorbar(m, ax=axes, orientation="horizontal", pad=0.01)
    plt.axis('off')
    axes = fig.add_subplot(133)

    axes.imshow(plot3, interpolation='nearest')
    plt.colorbar(m, ax=axes, orientation="horizontal", pad=0.01)
    plt.axis('off')
    fig.savefig('../res/graphs_color/' + prot_name + '.png')


def merge_plots1(out_dir, prot_name):
    plot1 = plt.imread(out_dir + prot_name + '_pos_color.png', format='png')
    plot2 = plt.imread(out_dir + prot_name + '_neg_color.png', format='png')
    plot3 = plt.imread(out_dir + prot_name + '_cont_color.png', format='png')
    plot4 = plt.imread(out_dir + prot_name + '_pos_colorbar.png', format='png')
    plot5 = plt.imread(out_dir + prot_name + '_neg_colorbar.png', format='png')
    plot6 = plt.imread(out_dir + prot_name + '_cont_colorbar.png', format='png')
    fig = plt.figure(figsize=(100, 100))
    axes = fig.add_subplot(131)
    axes.imshow(plot1, interpolation='nearest')
    plt.axis('off')
    axes = fig.add_subplot(132)
    axes.imshow(plot2, interpolation='nearest')
    plt.axis('off')
    axes = fig.add_subplot(133)
    axes.imshow(plot3, interpolation='nearest')
    plt.axis('off')
    fig.savefig(out_dir + prot_name + '.png', dpi=80)
    fig.clear()
    axes = fig.add_subplot(131)
    axes.imshow(plot4, interpolation='nearest')
    plt.axis('off')
    axes = fig.add_subplot(132)
    axes.imshow(plot5, interpolation='nearest')
    plt.axis('off')
    axes = fig.add_subplot(133)
    axes.imshow(plot6, interpolation='nearest')
    plt.axis('off')
    fig.savefig(out_dir + prot_name + '_colormap.png', dpi=80)


out_dir = '../res/graphs_color/'

if __name__ == '__main__':
    contact_graphs = create_contact_graph_with_normalization2()
    coevolution_data = [(prot_name, *read_data(prot_name)) for prot_name in prot_names]
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
        # weights = []
        # for graph in graphs:
        #     weights.extend(graph.es['weight'])
        # min_weight = min(weights)
        # max_weight = max(weights)
        # norm = colors.Normalize(vmin=min_weight, vmax=max_weight)
        # m = cm.ScalarMappable(norm=norm, cmap=cm.winter)
        # m.set_array(np.asarray(weights))
        plot_graph(out_dir, prot_name + '_pos', positive_graphs[prot_name], 10)
        plot_graph(out_dir, prot_name + '_neg', negative_graphs[prot_name], 10)
        plot_graph(out_dir, prot_name + '_cont', contact_graphs[prot_name], 10)
        merge_plots1(out_dir, prot_name)