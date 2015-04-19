#!/usr/bin/env python

import networkx as nx

nx.write_edgelist(nx.DiGraph(nx.complete_graph(100)), 'complete-small.txt', data=False)
nx.write_edgelist(nx.DiGraph(nx.barabasi_albert_graph(100, 3)), 'social-small.txt', data=False)
