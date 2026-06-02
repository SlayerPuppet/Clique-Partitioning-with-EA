import copy
from src.reductions import apply_partial_optimality
from src.bounds import iterative_cycle_packing
from src.heuristics import reweight_and_gaec, klj_local_search, calculate_original_cost
from src.evolutionary import MemeticAlgorithm

class AdvancedSolver:
    def __init__(self, original_G):
        """
        Initializes the solver and aligns the sign convention.
        Repo convention: negative = attractive, positive = repulsive.
        Paper convention: positive = attractive, negative = repulsive.
        """
        self.original_G = original_G
        self.G = copy.deepcopy(original_G)
        
        # Internalize paper's sign convention for the math
        for u, v, d in self.G.edges(data=True):
            d['c_paper'] = -d['cost']

    def run_pipeline(self):
        print("\n[Phase 1] Partial Optimality Preprocessing...")
        reduced_G = apply_partial_optimality(self.G)
        print(f"-> Reduced Graph: {reduced_G.number_of_nodes()} nodes")
        
        if reduced_G.number_of_edges() == 0:
            return None, 0, 0, 0

        print("\n[Phase 2] Iterative Cycle Packing (ICP)...")
        paper_lb, residuals = iterative_cycle_packing(reduced_G)
        LB = -paper_lb 
        print(f"-> ICP Lower Bound: {LB}")
        
        print("\n[Phase 3] Reweighting and GAEC...")
        gaec_partition = reweight_and_gaec(reduced_G, residuals)
        gaec_cost = calculate_original_cost(self.original_G, gaec_partition)
        print(f"-> Reweighted GAEC Cost: {gaec_cost}")
        
        print("\n[Phase 4] Kernighan-Lin with Joins (KLj)...")
        klj_partition, klj_cost = klj_local_search(self.original_G, gaec_partition)
        print(f"-> Final Primal Cost (KLj Upper Bound): {klj_cost}")

        print("\n[Phase 5] Memetic Evolutionary Search...")
        # We pass the original graph to the EA so it evaluates true costs
        ea = MemeticAlgorithm(self.original_G, pop_size=10, generations=30)
        final_partition, final_cost = ea.optimize(seed_partition=klj_partition)
        print(f"-> Memetic Final Cost: {final_cost}")
        
        # Calculate final gap based on the EA's best result
        UB = final_cost
        gap = (UB - LB) / abs(UB) if UB != 0 else 0
        
        return final_partition, UB, LB, gap