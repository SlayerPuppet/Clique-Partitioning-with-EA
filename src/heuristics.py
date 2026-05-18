def gaec(G):
    """Greedy Additive Edge Contraction"""
    H = G.copy()
    partition = {n: {n} for n in H.nodes()}
    
    while True:
        # Find edge with the minimum negative cost
        edges = [(u, v, d['cost']) for u, v, d in H.edges(data=True) if d['cost'] < 0]
        if not edges:
            break  # No more beneficial contractions
            
        u, v, min_cost = min(edges, key=lambda x: x[2])
        
        # Contract u and v
        partition[u] = partition[u].union(partition[v])
        del partition[v]
        
        # Update edges and costs
        for neighbor in list(H.neighbors(v)):
            if neighbor != u:
                if H.has_edge(u, neighbor):
                    H[u][neighbor]['cost'] += H[v][neighbor]['cost']
                else:
                    H.add_edge(u, neighbor, cost=H[v][neighbor]['cost'])
        H.remove_node(v)
        
    return list(partition.values())