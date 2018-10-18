import numpy as np


class Graph(object):

    def __init__(self, size=None, nodes=None):
        if size is None:
            max_node = 0
            for node in nodes:
                if node > max_node:
                    max_node = node
            self.size = max_node + 1
            self.nodes = list(nodes)
            self.node_set = np.zeros(self.size, dtype=int)
            for n in self.nodes:
                self.node_set[n] = 1
        else:
            self.size = size
            self.nodes = []
            self.node_set = np.zeros(self.size, dtype=int)
        self.neighbors = [[] for i in range(self.size)]
        self.nextlevel = np.empty(self.size, dtype=int)
        self.thislevel = np.empty(self.size, dtype=int)

    def subgraph(self, nodes):
        g = Graph(size=self.size)
        g.nodes = list(nodes)
        for n in g.nodes:
            g.node_set[n] = 1
        for n in g.nodes:
            for neighbor in self.neighbors[n]:
                if g.node_set[neighbor] == 1:
                    g.neighbors[n].append(neighbor)
        return g

    def count_edges(self):
        c = 0
        for n in range(self.size):
            c += len(self.neighbors[n])
        return c/2

    def delete_node(self, node):
        for n in self.neighbors[node]:
            self.neighbors[n].remove(node)
        self.neighbors[node].clear()
        self.node_set[node] = 0
        self.nodes.remove(node)

    def add_node(self, node, neighbors=None):
        self.node_set[node] = 1
        self.nodes.append(node)
        if neighbors is not None:
            self.neighbors[node].extend(neighbors)
            for n in neighbors:
                self.neighbors[n].append(node)

    def is_connected(self):
        """ determines if the graph is connected """
        seen = np.zeros(self.size, dtype=int)
        seen[self.nodes[0]] = 1
        nlen = 0
        tlen = 0
        thislevel = self.thislevel
        nextlevel = self.nextlevel
        thislevel[tlen] = self.nodes[0]
        tlen += 1
        repeat = True
        while repeat:
            repeat = False
            for j in range(tlen):
                v = thislevel[j]
                for n in self.neighbors[v]:
                    if seen[n] == 0:
                        seen[n] = 1
                        nextlevel[nlen] = n
                        nlen += 1
            if nlen > 0:
                repeat = True
            temp = thislevel
            thislevel = nextlevel
            tlen = nlen
            nextlevel = temp
            nlen = 0

        for n in self.nodes:
            if seen[n] == 0:
                return False
        return True

    def connected_components(self):
        start = self.nodes[0]
        res = []
        seen = np.zeros(self.size, dtype=int)
        nextlevel = self.nextlevel
        thislevel = self.thislevel
        curr_comp = np.zeros(self.size, dtype=int)
        while start != -1:
            curr_comp[start] = 1
            seen[start] = 1
            nlen = 0
            thislevel[0] = start
            tlen = 1
            while tlen > 0:
                for j in range(tlen):
                    v = thislevel[j]
                    for n in self.neighbors[v]:
                        if curr_comp[n] == 0:
                            curr_comp[n] = 1
                            seen[n] = 1
                            nextlevel[nlen] = n
                            nlen += 1
                temp = thislevel
                thislevel = nextlevel
                tlen = nlen
                nextlevel = temp
                nlen = 0

            comp = Graph(size=self.size)
            start = -1
            for n in self.nodes:
                if seen[n] == 0:
                    start = n
                elif curr_comp[n] == 1:
                    comp.nodes.append(n)
                    comp.node_set[n] = 1
                    comp.neighbors[n] = self.neighbors[n].copy()
                    curr_comp[n] = 0
            res.append(comp)
        return res