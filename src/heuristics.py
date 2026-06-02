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
    """Kernighan-Lin with joins (KLj). Locally improves a partition."""
    current_partition = [set(c) for c in partition]
    improved = True
    
    while improved:
        improved = False
        current_cost = calculate_original_cost(original_G, current_partition)
        
        for node in list(original_G.nodes()):
            source_idx = next(i for i, cluster in enumerate(current_partition) if node in cluster)
            target_indices = [i for i in range(len(current_partition)) if i != source_idx] + [-1]
            
            best_new_partition = None
            best_new_cost = current_cost
            
            for target_idx in target_indices:
                test_partition = [set(c) for c in current_partition]
                test_partition[source_idx].remove(node)
                
                if target_idx == -1: 
                    test_partition.append({node})
                else: 
                    test_partition[target_idx].add(node)
                    
                test_partition = [c for c in test_partition if c]
                test_cost = calculate_original_cost(original_G, test_partition)
                
                if test_cost < best_new_cost:
                    best_new_cost = test_cost
                    best_new_partition = test_partition
                    improved = True
                    
            if improved:
                current_partition = best_new_partition
                break 
                
    return current_partition, calculate_original_cost(original_G, current_partition)