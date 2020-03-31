import networkx as nx

def gc(graph: nx.Graph) -> nx.Graph:
    """Return giant component of graph."""
    return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()