# def gaec(G):
#     """Greedy Additive Edge Contraction"""
#     H = G.copy()
#     partition = {n: {n} for n in H.nodes()}
    
#     while True:
#         # Find edge with the minimum negative cost
#         edges = [(u, v, d['cost']) for u, v, d in H.edges(data=True) if d['cost'] < 0]
#         if not edges:
#             break  # No more beneficial contractions
            
#         u, v, min_cost = min(edges, key=lambda x: x[2])
        
#         # Contract u and v
#         partition[u] = partition[u].union(partition[v])
#         del partition[v]
        
#         # Update edges and costs
#         for neighbor in list(H.neighbors(v)):
#             if neighbor != u:
#                 if H.has_edge(u, neighbor):
#                     H[u][neighbor]['cost'] += H[v][neighbor]['cost']
#                 else:
#                     H.add_edge(u, neighbor, cost=H[v][neighbor]['cost'])
#         H.remove_node(v)
        
#     return list(partition.values())

import networkx as nx

def calculate_original_cost(G, partition):
    """Calculates true cost based on the original graph."""
    cost = 0
    for cluster in partition:
        nodes = list(cluster)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if G.has_edge(nodes[i], nodes[j]):
                    cost += G[nodes[i]][nodes[j]]['cost']
    return cost

def gaec(G):
    """Original Greedy Additive Edge Contraction"""
    H = G.copy()
    partition = {n: {n} for n in H.nodes()}
    
    while True:
        edges = [(u, v, d['cost']) for u, v, d in H.edges(data=True) if d['cost'] < 0]
        if not edges: break
            
        u, v, min_cost = min(edges, key=lambda x: x[2])
        partition[u] = partition[u].union(partition[v])
        del partition[v]
        
        for neighbor in list(H.neighbors(v)):
            if neighbor != u:
                if H.has_edge(u, neighbor):
                    H[u][neighbor]['cost'] += H[v][neighbor]['cost']
                else:
                    H.add_edge(u, neighbor, cost=H[v][neighbor]['cost'])
        H.remove_node(v)
        
    return list(partition.values())

# --- Contribution 3: Re-weighting & Primal Heuristics ---
def reweight_and_gaec(reduced_G, residuals, lam=0.5):
    """Generates re-weighted costs and runs GAEC."""
    H = reduced_G.copy()
    for u, v, d in H.edges(data=True):
        e = tuple(sorted((u, v)))
        c_e = d['c_paper']
        w_e = residuals.get(e, abs(c_e))
        
        magnitude = lam * abs(c_e) + (1 - lam) * w_e
        d['reweighted'] = magnitude if c_e > 0 else -magnitude

    partition = {n: {n} for n in H.nodes()}
    while True:
        edges = [(u, v, d['reweighted']) for u, v, d in H.edges(data=True) if d['reweighted'] > 0]
        if not edges: break
        
        u, v, max_val = max(edges, key=lambda x: x[2])
        
        partition[u] = partition[u].union(partition[v])
        del partition[v]
        
        for neighbor in list(H.neighbors(v)):
            if neighbor != u:
                if H.has_edge(u, neighbor):
                    H[u][neighbor]['reweighted'] += H[v][neighbor]['reweighted']
                else:
                    H.add_edge(u, neighbor, reweighted=H[v][neighbor]['reweighted'])
        H.remove_node(v)
        
    return list(partition.values())

def klj_local_search(original_G, partition):
    """Kernighan-Lin with joins (KLj). Locally improves a partition.

    Uses O(deg) delta evaluation per candidate move instead of recomputing
    the full partition cost. Alternates single-node relocations (1-opt,
    including moves to a new singleton) with cluster joins until neither
    move type improves the cost.
    """
    clusters = {}
    node_to_cid = {}
    next_cid = 0
    for c in partition:
        if c:
            clusters[next_cid] = set(c)
            for n in c:
                node_to_cid[n] = next_cid
            next_cid += 1
    # Nodes missing from the partition become singletons
    for n in original_G.nodes():
        if n not in node_to_cid:
            clusters[next_cid] = {n}
            node_to_cid[n] = next_cid
            next_cid += 1

    improved = True
    while improved:
        improved = False

        # Move type 1: Single-node relocation (1-opt)
        moved = True
        while moved:
            moved = False
            for node in original_G.nodes():
                src = node_to_cid[node]

                # Edge-weight sum from node to each adjacent cluster
                conn = {}
                for nb in original_G.neighbors(node):
                    cid = node_to_cid[nb]
                    conn[cid] = conn.get(cid, 0.0) + original_G[node][nb]['cost']

                cost_stay = conn.get(src, 0.0)

                # Best alternative: an adjacent cluster, or a new singleton (0.0)
                best_cid, best_cost = None, 0.0
                for cid, w in conn.items():
                    if cid != src and w < best_cost:
                        best_cid, best_cost = cid, w

                if best_cost - cost_stay < -1e-12:
                    clusters[src].discard(node)
                    if not clusters[src]:
                        del clusters[src]
                    if best_cid is None:
                        clusters[next_cid] = {node}
                        node_to_cid[node] = next_cid
                        next_cid += 1
                    else:
                        clusters[best_cid].add(node)
                        node_to_cid[node] = best_cid
                    moved = True
                    improved = True

        # Move type 2: Cluster joins — merge the pair whose inter-cluster
        # edge weight sum is most negative (attraction dominates).
        inter = {}
        for u, v, d in original_G.edges(data=True):
            cu, cv = node_to_cid[u], node_to_cid[v]
            if cu == cv:
                continue
            key = (cu, cv) if cu < cv else (cv, cu)
            inter[key] = inter.get(key, 0.0) + d['cost']

        if inter:
            (ca, cb), w = min(inter.items(), key=lambda kv: kv[1])
            if w < -1e-12:
                clusters[ca] |= clusters[cb]
                for n in clusters[cb]:
                    node_to_cid[n] = ca
                del clusters[cb]
                improved = True

    final = list(clusters.values())
    return final, calculate_original_cost(original_G, final)