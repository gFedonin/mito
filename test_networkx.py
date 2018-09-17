import networkx as nx

from compute_interface_stats import parallel_gen_random_subgraphs

small_nodes_num = 10
big_nodes_num = 1000
graph_num = 1000

if __name__ == '__main__':
    big = nx.fast_gnp_random_graph(big_nodes_num, 0.1)
    print('generation done')
    small = nx.Graph()
    for i in range(small_nodes_num):
        small.add_node(i)
    for i in range(small_nodes_num):
        for j in range(i + 1, small_nodes_num):
            if big.has_edge(i, j):
                small.add_edge(i, j)
    random_subgraphs = parallel_gen_random_subgraphs(big, small, graph_num)

    for g in random_subgraphs:
        if not nx.is_connected(g):
            print('not connected!')
        if g.number_of_nodes() != small_nodes_num:
            print('wrong number of nodes')
        if g.number_of_edges() < nx.number_of_edges(small):
            print('wrong number of edges')
        nodes = list(g.nodes)
        for i in range(small_nodes_num):
            for j in range(i + 1, small_nodes_num):
                if g.has_edge(nodes[i], nodes[j]):
                    if not big.has_edge(nodes[i], nodes[j]):
                        print('fake edge!')
                else:
                    if big.has_edge(nodes[i], nodes[j]):
                        print('missing edge!')

