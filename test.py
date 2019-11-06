import pygraphviz as PG

G = PG.AGraph()
nlist = "A B C D E".split()
a, b = "A A B", "B C D"
elist = zip(a.split(), b.split())

G.add_nodes_from(nlist)
G.add_edges_from(elist)
G.add_edge('A','A',color='blue')
G.node_attr.update(color="red", style="filled")
G.edge_attr.update(color="blue", len="2.0", width="2.0")

G.draw('./test.png', format='png', prog='circo')