from os import makedirs
from os.path import exists

import igraph
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize


# path_to_graphs = '../res/graphs_gml/'
path_to_graphs = '../res/graphs_multichain_gml_test/'
# out_dir = '../res/graphs_width_and_color1/'
out_dir = '../res/graphs_width_and_color_multichain_test/'


prot_names = ['cox1', 'cox2', 'cox3', 'cytb', 'atp6']

color_only = True


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def read_graph_from_file():
    return {prot_name: (igraph.Graph.Read_GML(path_to_graphs + prot_name + '_pos.gml'),
                        igraph.Graph.Read_GML(path_to_graphs + prot_name + '_neg.gml'),
                        igraph.Graph.Read_GML(path_to_graphs + prot_name + '_cont.gml')
                        ) for prot_name in prot_names}


def plot_graph(prot_name, graph, base_node_size, base_edge_width, margin):
    min_weight = min(graph.es['weight'])
    max_weight = max(graph.es['weight'])
    norm = MidpointNormalize(vmin=min_weight, vmax=max_weight, midpoint=1.0)
    m = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)#seismic bwr
    m.set_array(np.asarray(graph.es['weight']))
    fig = plt.figure(figsize=(10, 10))
    axes = fig.add_subplot(111)
    plt.axis('off')
    cb = plt.colorbar(m, ax=axes, orientation="horizontal")
    cb.ax.tick_params(labelsize=28)
    fig.savefig(out_dir + prot_name + '_colorbar.png')
    visual_style = {}
    # graph.vs['size'] = [100 * len(graph.vs) * w for w in graph.vs['weight']]
    if not color_only:
        visual_style["vertex_size"] = [base_node_size * w for w in graph.vs['weight']]
        visual_style["edge_width"] = [base_edge_width * w for w in graph.es['weight']]
        visual_style["margin"] = margin
    else:
        visual_style["vertex_size"] = 100
        visual_style["edge_width"] = 10
        visual_style["margin"] = (70, 140, 140, 70)
    visual_style["vertex_label"] = [n for n in graph.vs["name"]]
    visual_style['vertex_color'] = 'white'
    visual_style["label_size"] = 36
    visual_style['edge_color'] = [m.to_rgba(w) for w in graph.es['weight']]
    # graph.es['width'] = [10*w for w in graph.es['weight']]
    visual_style["layout"] = graph.layout_circle()
    visual_style["bbox"] = (1000, 1000)
    igraph.plot(graph, out_dir + prot_name + '_color.png', **visual_style)


if __name__ == '__main__':
    if not exists(out_dir):
        makedirs(out_dir)
    prot_to_style = {}
    prot_to_style['cox1'] = {'base_node_size': 820, 'base_edge_width': 13, 'margin': (120, 320, 170, 70)}
    prot_to_style['cox2'] = {'base_node_size': 600, 'base_edge_width': 15, 'margin': (120, 320, 260, 110)}
    # prot_to_style['cox3'] = {'base_node_size': 800, 'base_edge_width': 11, 'margin': (130, 340, 120, 50)}
    prot_to_style['cox3'] = {'base_node_size': 800, 'base_edge_width': 11, 'margin': (130, 340, 120, 100)}
    # prot_to_style['atp6'] = {'base_node_size': 690, 'base_edge_width': 5, 'margin': (120, 310, 140, 110)}
    prot_to_style['atp6'] = {'base_node_size': 600, 'base_edge_width': 5, 'margin': (120, 320, 140, 100)}
    prot_to_style['cytb'] = {'base_node_size': 870, 'base_edge_width': 6, 'margin': (120, 340, 90, 80)}
    for prot_name, (pos_graph, neg_graph, cont_graph) in read_graph_from_file().items():
        print(prot_name)
        # graphs = [positive_graphs[prot_name], negative_graphs[prot_name], contact_graphs[prot_name]]
        # graph, group_to_nodes, nodes_to_groups = read_data(prot_name)
        # graphs.append(create_small_graph_with_normalization(graph, group_to_nodes, nodes_to_groups, True))
        # graphs.append(create_small_graph_with_normalization(graph, group_to_nodes, nodes_to_groups, False))
        # plot_graphs(prot_name, graphs, 10)
        plot_graph(prot_name + '_pos', pos_graph, **prot_to_style[prot_name])
        plot_graph(prot_name + '_neg', neg_graph, **prot_to_style[prot_name])
        plot_graph(prot_name + '_cont', cont_graph, **prot_to_style[prot_name])