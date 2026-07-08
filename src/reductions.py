import networkx as nx


# --- Contribution 1: Partial Optimality Preprocessing ---
import networkx as nx

def apply_partial_optimality(G):
    """
    Applies reduction rules to shrink the graph before bounding.
    Runs as a fixpoint loop: contracting an attractive bridge can expose new
    bridges, so we repeat until no rule fires.
    """
    reduced_G = G.copy()

    changed = True
    while changed:
        changed = False

        # Rule 1: Delete repulsive edges that span different attractive components.
        G_plus = nx.Graph()
        G_plus.add_nodes_from(reduced_G.nodes())
        G_plus.add_edges_from(
            [(u, v) for u, v, d in reduced_G.edges(data=True) if d['c_paper'] > 0]
        )
        components = list(nx.connected_components(G_plus))
        node_to_comp = {node: idx for idx, comp in enumerate(components) for node in comp}

        edges_to_remove = [
            (u, v) for u, v, d in reduced_G.edges(data=True)
            if d['c_paper'] < 0 and node_to_comp[u] != node_to_comp[v]
        ]
        if edges_to_remove:
            reduced_G.remove_edges_from(edges_to_remove)
            changed = True

        # Rule 2: Bridge reductions.
        for u, v in list(nx.bridges(reduced_G)):
            if not reduced_G.has_edge(u, v):
                continue
            c_e = reduced_G[u][v]['c_paper']
            if c_e < 0:
                # Repulsive bridge: endpoints must be in different clusters -> delete.
                reduced_G.remove_edge(u, v)
                changed = True
            elif c_e > 0:
                # Attractive bridge: endpoints must be in the same cluster -> contract v into u.
                for neighbor in list(reduced_G.neighbors(v)):
                    if neighbor == u:
                        continue
                    if reduced_G.has_edge(u, neighbor):
                        reduced_G[u][neighbor]['cost'] += reduced_G[v][neighbor]['cost']
                        reduced_G[u][neighbor]['c_paper'] += reduced_G[v][neighbor]['c_paper']
                    else:
                        reduced_G.add_edge(
                            u, neighbor,
                            cost=reduced_G[v][neighbor]['cost'],
                            c_paper=reduced_G[v][neighbor]['c_paper'],
                        )
                reduced_G.remove_node(v)
                changed = True
                break  # restart bridge detection after structural change

    return reduced_G