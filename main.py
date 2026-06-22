import sys
import os
from src.data_loader import load_benchmark_graph
from src.advanced_solver import AdvancedSolver

def run_experiment(filepath):
    print("\n" + "="*50)
    print("   Advanced Clique Partitioning Pipeline")
    print("="*50)
    
    # 1. Verify file exists
    if not os.path.exists(filepath):
        print(f"Error: Could not find dataset at '{filepath}'")
        return

    # 2. Load the benchmark graph
    G = load_benchmark_graph(filepath)
    if G.number_of_nodes() == 0:
        print("Error: Graph is empty. Check your dataset parser.")
        return

    # 3. Initialize and run the Advanced Solver
    solver = AdvancedSolver(G)
    final_partition, UB, LB, gap = solver.run_pipeline()

    # 4. Display Final Metrics
    if final_partition is not None:
        print("\n" + "="*50)
        print("                 FINAL RESULTS")
        print("="*50)
        print(f"Theoretical Lower Bound (LB): {LB}")
        print(f"Heuristic Upper Bound (UB):   {UB}")
        print(f"Optimality Gap:               {gap:.4f}")
        print(f"Total Clusters Formed:        {len(final_partition)}")
        print("="*50 + "\n")

if __name__ == "__main__":
    # Default to the sample data if no terminal argument is provided
    target_file = sys.argv[1] if len(sys.argv) > 1 else "data/sample_data.txt"
    run_experiment(target_file)