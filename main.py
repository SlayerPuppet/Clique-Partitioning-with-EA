import sys
import os
import matplotlib.pyplot as plt
from src.data_loader import load_benchmark_graph
from src.advanced_solver import AdvancedSolver

def run_experiment(filepath):
    print("\n" + "="*50)
    print("   Advanced Memetic Clique Partitioning Pipeline")
    print("="*50)
    
    if not os.path.exists(filepath):
        print(f"Error: Could not find dataset at '{filepath}'")
        return

    G = load_benchmark_graph(filepath)
    if G is None or G.number_of_nodes() == 0:
        print("Error: Graph is empty. Check your dataset parser.")
        return

    solver = AdvancedSolver(G)

    # Extract the actual filename (e.g., 'corr40-10.txt') from the path
    filename = os.path.basename(filepath)
    
    # Pass it to the pipeline
    final_partition, UB, LB, gap, history = solver.run_pipeline(filename=filename)
    
    # # Unpack the history variable
    # final_partition, UB, LB, gap, history = solver.run_pipeline()

    if final_partition is not None:
        print("\n" + "="*50)
        print("                 FINAL RESULTS")
        print("="*50)
        print(f"Theoretical Lower Bound (LB): {LB}")
        print(f"Heuristic Upper Bound (UB):   {UB}")
        print(f"Optimality Gap:               {gap:.4f}")
        print(f"Total Clusters Formed:        {len(final_partition)}")
        print("="*50 + "\n")
        
        # --- NEW: Plotting Logic ---
        if history:
            print("Generating convergence plot...")
            plt.figure(figsize=(8, 5))
            # Plot the history array
            plt.plot(range(len(history)), history, marker='o', markersize=4, linestyle='-', color='#2ca02c')
            
            # Format the chart
            dataset_name = os.path.basename(filepath)
            plt.title(f"Memetic Algorithm Convergence\nDataset: {dataset_name}")
            plt.xlabel("Generation")
            plt.ylabel("Best Upper Bound (Cost)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the image to your folder
            plot_filename = f"convergence_{dataset_name.split('.')[0]}.png"
            plt.savefig(plot_filename)
            print(f"Plot saved successfully as '{plot_filename}'")
            
    else:
        print("\nPipeline terminated early (Graph was fully reduced).")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "data/sample_data.txt"
    run_experiment(target_file)