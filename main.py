import sys
from src.data_loader import load_benchmark_graph
from src.heuristics import gaec
from src.bounds import simplex_lower_bound
from src.evolutionary import CliquePartitioningEA

def run_experiment(filepath):
    print("\n--- Starting Clique Partitioning Pipeline ---")
    
    # 1. Load the real data
    G = load_benchmark_graph(filepath)
    node_count = G.number_of_nodes()

    if node_count == 0:
        print("Error: Graph is empty. Check your dataset file.")
        return

    # 2. Safety Check for Simplex (O(N^3) constraint explosion)
    if node_count <= 20:
        print("\n[Phase 1] Computing Simplex Lower Bound...")
        lb = simplex_lower_bound(G)
        print(f"-> Theoretical Minimum Cost: {lb}")
    else:
        print(f"\n[Phase 1] Skipping Simplex Lower Bound (Graph too large: {node_count} nodes).")
        print("-> Implement lazy constraint generation (cutting planes) to evaluate bounds on this graph.")

    # 3. Baseline Heuristic
    print("\n[Phase 2] Running GAEC Baseline...")
    ea_temp = CliquePartitioningEA(G) # Instantiate just to use the cost calculator
    baseline_partition = gaec(G)
    baseline_cost = ea_temp.calculate_cost(baseline_partition)
    print(f"-> GAEC Cost: {baseline_cost}")

    # 4. Evolutionary Algorithm
    print("\n[Phase 3] Running Evolutionary Algorithm...")
    ea = CliquePartitioningEA(G, pop_size=20, generations=100)
    best_partition, best_cost = ea.optimize()
    print(f"-> EA Best Cost: {best_cost}")
    print(f"-> Improvement over baseline: {baseline_cost - best_cost}")
    
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    # Allow passing the dataset path via command line
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_dataset>")
        sys.exit(1)
        
    target_file = sys.argv[1]
    run_experiment(target_file)