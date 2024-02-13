import networkx as nx

def draw_network(g: nx.DiGraph, with_labels=False):
    nx.draw(g, pos=nx.circular_layout(g), node_size=5, width=0.1, with_labels=with_labels)