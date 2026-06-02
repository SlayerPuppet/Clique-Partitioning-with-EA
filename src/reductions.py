import networkx as nx


# --- Contribution 1: Partial Optimality Preprocessing ---
import networkx as nx

def apply_partial_optimality(G):
    """
    Applies reduction rules to shrink the graph before bounding.
    """
    reduced_G = G.copy()
    
    # 1. Extract attractive subgraph
    G_plus = nx.Graph()
    G_plus.add_nodes_from(reduced_G.nodes())
    G_plus.add_edges_from([(u, v) for u, v, d in reduced_G.edges(data=True) if d['c_paper'] > 0])
    
    # Determine connected components of the attractive subgraph
    components = list(nx.connected_components(G_plus))
    node_to_comp = {}
    for idx, comp in enumerate(components):
        for node in comp:
            node_to_comp[node] = idx

    edges_to_remove = []
    for u, v, d in reduced_G.edges(data=True):
        if d['c_paper'] < 0:
            # Rule: Delete repulsive edges spanning different attractive components
            if node_to_comp[u] != node_to_comp[v]:
                edges_to_remove.append((u, v))
    
    reduced_G.remove_edges_from(edges_to_remove)
    
    # 2. Bridge Edge Reductions
    bridges = list(nx.bridges(reduced_G))
    for u, v in bridges:
        c_e = reduced_G[u][v]['c_paper']
        if c_e < 0:
            # Repulsive bridge -> delete
            reduced_G.remove_edge(u, v)
        elif c_e > 0:
            # Attractive bridge -> contract
            pass 
            
    return reduced_G