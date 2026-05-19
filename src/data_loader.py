import networkx as nx

def load_benchmark_graph(filepath):
    """
    Parses CP-Lib dense upper-triangular format.
    First token is the number of nodes (N).
    The remaining N(N-1)/2 tokens are the edge weights.
    """
    G = nx.Graph()
    print(f"Loading graph from: {filepath}")
    
    # Read the entire file and split into a flat list of tokens
    with open(filepath, 'r') as f:
        tokens = f.read().split()
        
    if not tokens:
        print("Error: File is empty.")
        return G
        
    # The first number is the total node count
    num_nodes = int(tokens[0])
    G.add_nodes_from(range(1, num_nodes + 1))
    
    # The rest are the weights
    weights = [float(w) for w in tokens[1:]]
    
    expected_edges = (num_nodes * (num_nodes - 1)) // 2
    if len(weights) != expected_edges:
        print(f"Warning: Expected {expected_edges} edge weights, but found {len(weights)}.")
        
    # Reconstruct the upper-triangular connections
    weight_idx = 0
    for i in range(1, num_nodes):
        for j in range(i + 1, num_nodes + 1):
            if weight_idx < len(weights):
                G.add_edge(i, j, cost=weights[weight_idx])
                weight_idx += 1
                
    print(f"Successfully loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G