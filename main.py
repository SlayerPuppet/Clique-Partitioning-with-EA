# main.py
from src.data_loader import load_benchmark_graph
from src.bounds import simplex_lower_bound
from src.evolutionary import CliquePartitioningEA

def run_experiment():
    print("Loading graph...")
    G = load_benchmark_graph("data/dummy_path")

    print("Computing Simplex Lower Bound...")
    lb = simplex_lower_bound(G)
    print(f"Lower Bound: {lb}")

    print("Running Evolutionary Algorithm...")
    ea = CliquePartitioningEA(G, pop_size=10, generations=50)
    best_partition, best_cost = ea.optimize()
    
    print(f"EA Best Cost: {best_cost}")
    print(f"Clusters: {best_partition}")

if __name__ == "__main__":
    run_experiment()