import os
import csv
import time
from src.data_loader import load_benchmark_graph
from src.advanced_solver import AdvancedSolver

def run_batch_experiments(target_folders, output_csv="batch_results.csv"):
    print("\n" + "="*50)
    print("   Initializing Batch Experiment Runner")
    print("="*50)
    
    # 1. Prepare the CSV file and write the header row
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Instance", "Nodes", "Edges", 
                         "Lower Bound (LB)", "Memetic Cost (UB)", 
                         "Relative Gap", "Clusters", "Time (s)"])

        # 2. Loop through each specified folder
        for folder in target_folders:
            if not os.path.exists(folder):
                print(f"Warning: Skipping '{folder}' - Directory not found.")
                continue

            # 3. Process every text file in the folder
            for filename in os.listdir(folder):
                if not filename.endswith(".txt"): 
                    continue

                filepath = os.path.join(folder, filename)
                dataset_name = os.path.basename(folder)

                print(f"\nProcessing {dataset_name} / {filename}...")
                start_time = time.time()

                try:
                    # Load the graph
                    G = load_benchmark_graph(filepath)
                    if G is None or G.number_of_nodes() == 0:
                        print(" -> Empty graph, skipping.")
                        continue

                    nodes = G.number_of_nodes()
                    edges = G.number_of_edges()

                    # Run the pipeline
                    solver = AdvancedSolver(G)
                    final_partition, UB, LB, gap = solver.run_pipeline()

                    # Calculate metrics
                    exec_time = round(time.time() - start_time, 2)
                    num_clusters = len(final_partition) if final_partition else 0

                    # Write results to CSV
                    writer.writerow([dataset_name, filename, nodes, edges, 
                                     LB, UB, round(gap, 4), num_clusters, exec_time])
                    
                    print(f"-> Success! Time: {exec_time}s | Gap: {gap:.4f}")

                except Exception as e:
                    # If an instance fails, log it and keep the batch running
                    print(f"-> ERROR on {filename}: {e}")
                    writer.writerow([dataset_name, filename, "ERROR", "ERROR", "-", "-", "-", "-", "-"])

    print("\n" + "="*50)
    print(f"Batch testing complete! Results saved to: {output_csv}")
    print("="*50 + "\n")

if __name__ == "__main__":
    folders_to_test = [
        "data/CP-Lib-main/Correlation",
        "data/CP-Lib-main/ClusEdit"
    ]

    run_batch_experiments(folders_to_test, output_csv="batch_results_correlation_clusedit.csv")