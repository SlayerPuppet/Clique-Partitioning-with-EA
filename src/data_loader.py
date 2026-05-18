import networkx as nx
import numpy as np
import random
from scipy.optimize import linprog

def load_benchmark_graph(filepath):
    """
    Dummy loader for CP-lib or SPPA datasets.
    """
    G = nx.Graph()
    # Simulated parsing: Node 1, Node 2, Weight (Cost)
    # Negative cost = similar (want to be together)
    # Positive cost = dissimilar (want to be apart)
    sample_edges = [
        (1, 2, -5.0), (1, 3, 2.0), (2, 3, -1.0),
        (3, 4, -8.0), (1, 4, 10.0), (2, 4, 3.0)
    ]
    G.add_weighted_edges_from(sample_edges, weight='cost')
    return G