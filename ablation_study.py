import os
import sys
import copy
import matplotlib.pyplot as plt
from src.data_loader import load_benchmark_graph
from src.reductions import apply_partial_optimality
from src.bounds import iterative_cycle_packing
from src.heuristics import reweight_and_gaec, klj_local_search, calculate_original_cost
from src.evolutionary import MemeticAlgorithm

def run_ablation_study(filepath):
    print("\n" + "="*50)
    print("   Starting Ablation Study (Component Analysis)")
    print("="*50)
    
    if not os.path.exists(filepath):
        print(f"Error: Could not find dataset at '{filepath}'")
        return

    original_G = load_benchmark_graph(filepath)
    if original_G is None or original_G.number_of_nodes() == 0:
        return

    # Internalize sign convention
    G = copy.deepcopy(original_G)
    for u, v, d in G.edges(data=True):
        d['c_paper'] = -d['cost']

    # --- Phase 1 & 2: Setup and Bounds ---
    reduced_G = apply_partial_optimality(G)
    paper_lb, residuals = iterative_cycle_packing(reduced_G)
    sum_reduced_costs = sum(d['cost'] for u, v, d in reduced_G.edges(data=True))
    LB = sum_reduced_costs + paper_lb

    # --- Study 1: Baseline (GAEC Only) ---
    print("\nRunning Baseline: GAEC Only...")
    gaec_partition = reweight_and_gaec(reduced_G, residuals)
    gaec_cost = calculate_original_cost(original_G, gaec_partition)
    
    # --- Study 2: GAEC + Local Search (KLj) ---
    print("Running Configuration 2: GAEC + KLj Local Search...")
    klj_partition, klj_cost = klj_local_search(original_G, gaec_partition)
    
    # --- Study 3: The Full Engine (Memetic EA) ---
    print("Running Configuration 3: Full Memetic EA...")
    ea = MemeticAlgorithm(original_G, pop_size=10, generations=30)
    final_partition, final_cost, _ = ea.optimize(seed_partition=klj_partition)
    
    print("\n" + "="*50)
    print("   ABLATION RESULTS")
    print("="*50)
    print(f"Lower Bound (Mathematical Floor): {LB}")
    print(f"1. GAEC Only Cost:                {gaec_cost}")
    print(f"2. GAEC + KLj Cost:               {klj_cost}")
    print(f"3. Full Memetic EA Cost:          {final_cost}")
    
    # --- Generate the Visual Evidence ---
    generate_ablation_chart(os.path.basename(filepath), LB, gaec_cost, klj_cost, final_cost)

def generate_ablation_chart(filename, lb, gaec, klj, memetic):
    labels = ['GAEC (Baseline)', 'GAEC + KLj', 'Full Memetic EA']
    costs = [gaec, klj, memetic]
    
    plt.figure(figsize=(9, 6))
    
    # Create bar chart
    bars = plt.bar(labels, costs, color=['#d62728', '#ff7f0e', '#2ca02c'], edgecolor='black')
    
    # Add the Lower Bound as a strict horizontal line
    plt.axhline(y=lb, color='blue', linestyle='--', linewidth=2, label=f'Lower Bound ({lb})')
    
    # Format the chart
    plt.title(f"Ablation Study: Algorithmic Contribution\nDataset: {filename}", fontsize=14)
    plt.ylabel("Cost (Lower is Better)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add exact values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval}', 
                 ha='center', va='bottom' if yval > 0 else 'top', fontweight='bold')
        
    plt.tight_layout()
    plot_filename = f"ablation_study_{filename.split('.')[0]}.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved successfully as '{plot_filename}'")

if __name__ == "__main__":
    # Default to a tough graph that requires the EA to work hard
    target_file = sys.argv[1] if len(sys.argv) > 1 else "data/CP-Lib-main/Correlation/corr40-10.txt"
    run_ablation_study(target_file)