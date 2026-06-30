import os
import networkx as nx

def load_benchmark_graph(filepath):
    """
    Dynamically loads a graph, automatically detecting between 
    CP-Lib dense matrices and SPPA Multicut sparse edge-lists,
    while providing helpful terminal logging.
    """
    filename = os.path.basename(filepath)
    print(f"Loading dataset: {filename}")
    
    G = nx.Graph()
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"-> Error reading file: {e}")
        return None
        
    # 1. Clean the data: remove empty lines and comments (# or c or header text)
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(('#', 'c', 'MULTICUT')):
            continue
        clean_lines.append(line)
        
    if not clean_lines:
        print("-> Error: File contains no valid data after stripping comments.")
        return G

    # 2. Format Detection Heuristic
    # Look at the first data line. If it has exactly 3 elements, it's an edge list.
    # If the first line is just "Nodes Edges" (2 elements), check the second line.
    first_line_tokens = clean_lines[0].split()
    is_edge_list = False
    
    if len(first_line_tokens) == 3:
        is_edge_list = True
    elif len(first_line_tokens) == 2 and len(clean_lines) > 1:
        if len(clean_lines[1].split()) == 3:
            is_edge_list = True

    # 3. Parse based on detected format
    if is_edge_list:
        print("-> Format Detected: SPPA Multicut (Sparse Edge List)")
        start_idx = 1 if len(first_line_tokens) == 2 else 0
            
        for line in clean_lines[start_idx:]:
            tokens = line.split()
            if len(tokens) >= 3:
                u, v, cost = int(tokens[0]), int(tokens[1]), float(tokens[2])
                G.add_edge(u, v, cost=cost)
                
    else:
        print("-> Format Detected: CP-Lib (Dense Upper-Triangular Matrix)")
        node_i = 0
        for line in clean_lines:
            tokens = line.split()
            node_j = node_i + 1
            for token in tokens:
                cost = float(token)
                if cost != 0: 
                    G.add_edge(node_i, node_j, cost=cost)
                node_j += 1
            node_i += 1
            
    print(f"-> Successfully built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G