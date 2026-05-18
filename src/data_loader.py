import networkx as nx

def load_benchmark_graph(filepath):
    """
    Parses standard edge-list files from CP-lib or SPPA.
    Expects format: node1 node2 cost
    """
    G = nx.Graph()
    print(f"Loading graph from: {filepath}")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comment headers
            if not line or line.startswith('#') or line.startswith('c') or line.startswith('p'):
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    cost = float(parts[2])
                    G.add_edge(u, v, cost=cost)
                except ValueError:
                    continue 

    print(f"Successfully loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G