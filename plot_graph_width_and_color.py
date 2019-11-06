import igraph
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from plot_graph_color import MidpointNormalize

path_to_graphs = '../res/graphs_gml/'
out_dir = '../res/graphs_width_and_color/'


prot_names = ['cox1', 'cox2', 'cox3', 'cytb', 'atp6']


def read_graph_from_file():
    return {prot_name: (igraph.Graph.Read_GML(path_to_graphs + prot_name + '_pos.gml'),
                        igraph.Graph.Read_GML(path_to_graphs + prot_name + '_neg.gml'),
                        igraph.Graph.Read_GML(path_to_graphs + prot_name + '_cont.gml')
                        ) for prot_name in prot_names}


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
    # graph.vs['size'] = [100 * len(graph.vs) * w for w in graph.vs['weight']]
    visual_style["vertex_size"] = [400 * w for w in graph.vs['weight']]
    visual_style["vertex_label"] = [n for n in graph.vs["name"]]
    visual_style['vertex_color'] = 'white'
    visual_style["label_size"] = 100
    visual_style["edge_width"] = [base_edge_width*w for w in graph.es['weight']]
    visual_style['edge_color'] = [m.to_rgba(w) for w in graph.es['weight']]
    # graph.es['width'] = [10*w for w in graph.es['weight']]
    visual_style["layout"] = graph.layout_circle()
    visual_style["bbox"] = (1000, 1000)
    visual_style["margin"] = 200
    igraph.plot(graph, out_dir + prot_name + '_color.png', **visual_style)


if __name__ == '__main__':

    for prot_name, (pos_graph, neg_graph, cont_graph) in read_graph_from_file().items():
        print(prot_name)
        # graphs = [positive_graphs[prot_name], negative_graphs[prot_name], contact_graphs[prot_name]]
        # graph, group_to_nodes, nodes_to_groups = read_data(prot_name)
        # graphs.append(create_small_graph_with_normalization(graph, group_to_nodes, nodes_to_groups, True))
        # graphs.append(create_small_graph_with_normalization(graph, group_to_nodes, nodes_to_groups, False))
        # plot_graphs(prot_name, graphs, 10)
        plot_graph(prot_name + '_pos', pos_graph, 10)
        plot_graph(prot_name + '_neg', neg_graph, 10)
        plot_graph(prot_name + '_cont', cont_graph, 10)