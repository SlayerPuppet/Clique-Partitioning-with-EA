import copy
from src.reductions import apply_partial_optimality
from src.bounds import iterative_cycle_packing
from src.heuristics import reweight_and_gaec, klj_local_search, calculate_original_cost
from src.evolutionary import MemeticAlgorithm

# --- THE CP-LIB OPTIMUM LOOKUP TABLE ---
# Values extracted from Sørensen & Letchford (2024).
# Converted to negative values to match the minimization objective.
KNOWN_OPTIMUMS = {
    # Table 1: ABR Instances (Grötschel & Wakabayashi)
    "cars.txt": -1501.0, "cetacea.txt": -967.0, "companies.txt": -81802.0, "micro.txt": -966.0, 
    "UNO.txt": -798.0, "UNO_1a.txt": -12197.0, "UNO_1b.txt": -11775.0, "UNO_2a.txt": -72820.0, 
    "UNO_2b.txt": -71818.0, "UNO_3a.txt": -73068.0, "UNO_3b.txt": -72629.0, "wildcats.txt": -1304.0, 
    "workers.txt": -964.0,

    # Table 2: More ABR instances
    "bridges.txt": -3867.0, "Hayes-Roth.txt": -2800.0, "lecturers.txt": -14317.0, "lung-cancer.txt": -3472.0, 
    "lymphography.txt": -19174.0, "primary-tumor.txt": -323614.0, "soup.txt": -4625.0, "soybean-21.txt": -3041.0, 
    "soybean-35.txt": -14613.0, "soybean-large.txt": -316469.0, "sponge.txt": -25677.0, "ta-evaluation.txt": -1108.0, 
    "zoo.txt": -16948.0,

    # Table 3: Machine cell formation
    "BOC_1.txt": -58.0, "BOC_2.txt": -61.0, "BOC_3.txt": -60.0, "BOC_4.txt": -50.0, "BOC_5.txt": -72.0, 
    "BOC_6.txt": -76.0, "BOC_7.txt": -78.0, "BOC_8.txt": -61.0, "BOC_9.txt": -89.0, "BOC_10.txt": -70.0, 
    "BOE_91.txt": -80.0, "BUR_69.txt": -98.0, "BUR_73.txt": -126.0, "BUR_75.txt": -67.0, "BUR_91.txt": -72.0, 
    "CAN_97.txt": -157.0, "CHA_86.txt": -102.0, "CHA_87.txt": -347.0, "GRO_80.txt": -53.0, "IRA_95.txt": -38.0, 
    "KAT_97.txt": -175.0, "KIN_80.txt": -41.0, "LEE_97.txt": -115.0, "MAS_97.txt": -41.0, "MCC_72.txt": -43.0, 
    "MIL_91.txt": -46.0, "NAI_96a.txt": -117.0, "NAI_96b.txt": -93.0, "NAI_96c.txt": -91.0, "NAI_96d.txt": -74.0, 
    "ROG_05.txt": -60.0, "SEI_88.txt": -54.0, "SUL_91.txt": -46.0, "Wang250.txt": -419.0, "Wang800.txt": -1177.0, 
    "Wang1150.txt": -3236.0,

    # Table 4: Random cluster editing
    "ce50-20.txt": -58.0, "ce50-30.txt": -79.0, "ce50-40.txt": -105.0, "ce50-50.txt": -163.0, "ce50-60.txt": -257.0, 
    "ce60-20.txt": -73.0, "ce60-30.txt": -100.0, "ce60-40.txt": -151.0, "ce60-50.txt": -200.0, "ce60-60.txt": -373.0, 
    "ce70-20.txt": -93.0, "ce70-30.txt": -128.0, "ce70-40.txt": -177.0, "ce70-50.txt": -266.0, "ce70-60.txt": -491.0, 
    "ce80-20.txt": -107.0, "ce80-30.txt": -157.0, "ce80-40.txt": -227.0, "ce80-50.txt": -325.0, "ce80-60.txt": -657.0,

    # Table 5: Charon & Hudry and Brusco & Köhn random
    "rand100-5.txt": -1407.0, "rand100-100.txt": -24296.0, "rand200-5.txt": -4079.0, "rand200-100.txt": -74924.0, 
    "rand300-5.txt": -7732.0, "rand300-100.txt": -152709.0, "rand400-5.txt": -12133.0, "rand400-100.txt": -222757.0, 
    "rand500-5.txt": -17127.0, "rand500-100.txt": -309125.0, "regnier300-50.txt": -32164.0, "sym300-50.txt": -17592.0, 
    "zahn300.txt": -2504.0,

    # Table 6: Palubeckis et al. random instances
    "p500-5-1.txt": -17691.0, "p500-5-2.txt": -17169.0, "p500-5-3.txt": -16816.0, "p500-5-4.txt": -16808.0, 
    "p500-5-5.txt": -16957.0, "p500-5-6.txt": -16615.0, "p500-5-7.txt": -16649.0, "p500-5-8.txt": -16756.0, 
    "p500-5-9.txt": -16629.0, "p500-5-10.txt": -17360.0, "p500-100-1.txt": -308896.0, "p500-100-2.txt": -310241.0, 
    "p500-100-3.txt": -310477.0, "p500-100-4.txt": -309567.0, "p500-100-5.txt": -309135.0, "p500-100-6.txt": -310280.0, 
    "p500-100-7.txt": -310063.0, "p500-100-8.txt": -303148.0, "p500-100-9.txt": -305305.0, "p500-100-10.txt": -314864.0, 
    "p1000-1.txt": -885281.0, "p1000-2.txt": -881751.0, "p1000-3.txt": -866488.0, "p1000-4.txt": -869374.0, 
    "p1000-5.txt": -888960.0, "p1500-1.txt": -1619470.0, "p1500-2.txt": -1649778.0, "p1500-3.txt": -1611197.0, 
    "p1500-4.txt": -1641933.0, "p1500-5.txt": -1595627.0, "p2000-1.txt": -2508005.0, "p2000-2.txt": -2495730.0, 
    "p2000-3.txt": -2544728.0, "p2000-4.txt": -2528721.0, "p2000-5.txt": -2514009.0,

    # Table 7: Zhou et al. random instances
    "gauss500-100-1.txt": -265070.0, "gauss500-100-2.txt": -269076.0, "gauss500-100-3.txt": -257700.0, 
    "gauss500-100-4.txt": -267683.0, "gauss500-100-5.txt": -271567.0, "unif700-100-1.txt": -515016.0, 
    "unif700-100-2.txt": -519441.0, "unif700-100-3.txt": -512351.0, "unif700-100-4.txt": -513582.0, 
    "unif700-100-5.txt": -510585.0, "unif800-100-1.txt": -639675.0, "unif800-100-2.txt": -630704.0, 
    "unif800-100-3.txt": -629375.0, "unif800-100-4.txt": -624728.0, "unif800-100-5.txt": -625905.0,

    # Table 8: Lu et al. random instances
    "b2500-1.txt": -1063621.0, "b2500-2.txt": -1064144.0, "b2500-3.txt": -1082946.0, "b2500-4.txt": -1066239.0, 
    "b2500-5.txt": -1066387.0, "b2500-6.txt": -1066978.0, "b2500-7.txt": -1068377.0, "b2500-8.txt": -1070060.0, 
    "b2500-9.txt": -1071272.0, "b2500-10.txt": -1066770.0,

    # Table 9: Du et al. and Jovanovic et al.
    "CPn35-1.txt": -7837.0, "CPn35-2.txt": -7215.0, "CPn35-3.txt": -7633.0, "CPn35-4.txt": -7652.0, 
    "CPn45-1.txt": -11545.0, "CPn45-2.txt": -12345.0, "CPn45-3.txt": -11880.0, "CPn45-4.txt": -10506.0, 
    "CPn50-1.txt": -13562.0, "CPn50-2.txt": -14080.0, "CPn50-3.txt": -13172.0, "CPn50-4.txt": -13728.0, 
    "CPn65-1.txt": -20028.0, "CPn65-2.txt": -20753.0, "CPn65-3.txt": -20463.0, "CPn65-4.txt": -20000.0, 
    "CPn100-1.txt": -37188.0, "CPn100-2.txt": -37460.0, "CPn100-3.txt": -39766.0, "CPn100-4.txt": -38192.0,

    # Table 10: Artificial instances
    "am-25-3.txt": -625.0, "am-25-10.txt": -800.0, "am-25-20.txt": -1050.0, "am-50-3.txt": -2500.0, 
    "am-50-10.txt": -2850.0, "am-50-20.txt": -3350.0, "am-75-3.txt": -5625.0, "am-75-10.txt": -6150.0, 
    "am-75-20.txt": -6900.0, "am-100-3.txt": -10000.0, "am-100-10.txt": -10700.0, "am-100-20.txt": -11700.0, 
    "am-125-3.txt": -15625.0, "am-125-10.txt": -16500.0, "am-125-20.txt": -17750.0, "am-150-3.txt": -22500.0, 
    "am-150-10.txt": -23550.0, "am-150-20.txt": -25050.0,

    # Table 11: Equicut "negative" instances
    "neg-c-00.txt": -752.0, "neg-c-10.txt": -649.0, "neg-c-20.txt": -604.0, "neg-c-30.txt": -582.0, 
    "neg-c-40.txt": -577.0, "neg-c-50.txt": -549.0, "neg-c-60.txt": -463.0, "neg-c-70.txt": -452.0, 
    "neg-c-80.txt": -317.0, "neg-s-80.txt": -473.0, "neg-tt-80.txt": -592.0,

    # Table 12: Correlation instances
    "corr40-1.txt": -2191.0, "corr40-2.txt": -1852.0, "corr40-3.txt": -2310.0, "corr40-4.txt": -2084.0, 
    "corr40-5.txt": -2245.0, "corr40-6.txt": -2516.0, "corr40-7.txt": -2294.0, "corr40-8.txt": -2184.0, 
    "corr40-9.txt": -2129.0, "corr40-10.txt": -2301.0, "corr60-1.txt": -3678.0, "corr60-2.txt": -3445.0, 
    "corr60-3.txt": -3595.0, "corr60-4.txt": -3565.0, "corr60-5.txt": -3313.0, "corr60-6.txt": -3295.0, 
    "corr60-7.txt": -3506.0, "corr60-8.txt": -3540.0, "corr60-9.txt": -3372.0, "corr60-10.txt": -3570.0, 
    "corr80-1.txt": -4724.0, "corr80-2.txt": -4667.0, "corr80-3.txt": -4993.0, "corr80-4.txt": -4504.0, 
    "corr80-5.txt": -5090.0, "corr80-6.txt": -4465.0, "corr80-7.txt": -5088.0, "corr80-8.txt": -4757.0, 
    "corr80-9.txt": -4430.0, "corr80-10.txt": -5071.0
}

# Filenames on disk vary in case (e.g. 'boc_1.txt' vs 'BOC_1.txt'), so all
# lookups go through this lowercase view of the table.
_OPTIMA_LC = {k.lower(): v for k, v in KNOWN_OPTIMUMS.items()}


def lookup_known_optimum(filename):
    """Case-insensitive lookup of the literature optimum for an instance file."""
    if not filename:
        return None
    return _OPTIMA_LC.get(filename.lower())


class AdvancedSolver:
    def __init__(self, original_G):
        self.original_G = original_G
        self.G = copy.deepcopy(original_G)
        
        for u, v, d in self.G.edges(data=True):
            d['c_paper'] = -d['cost']

    def run_pipeline(self, filename=None):
        print("\n[Phase 1] Partial Optimality Preprocessing...")
        reduced_G = apply_partial_optimality(self.G)
        print(f"-> Reduced Graph: {reduced_G.number_of_nodes()} nodes")

        print("\n[Phase 2] Iterative Cycle Packing (ICP)...")
        paper_lb, residuals = iterative_cycle_packing(reduced_G)
        
        # FIX: CP Lower Bound = Sum of all costs in reduced graph + Multicut LB
        sum_reduced_costs = sum(d['cost'] for u, v, d in reduced_G.edges(data=True))
        LB = sum_reduced_costs + paper_lb 
        print(f"-> True ICP Lower Bound: {LB}")
        
        print("\n[Phase 3] Reweighting and GAEC...")
        gaec_partition = reweight_and_gaec(reduced_G, residuals)
        gaec_cost = calculate_original_cost(self.original_G, gaec_partition)
        print(f"-> Reweighted GAEC Cost: {gaec_cost}")
        
        print("\n[Phase 4] Kernighan-Lin with Joins (KLj)...")
        klj_partition, klj_cost = klj_local_search(self.original_G, gaec_partition)
        print(f"-> Final Primal Cost (KLj Upper Bound): {klj_cost}")

        # --- THE EARLY TERMINATION CHECK ---
        if abs(klj_cost - LB) < 1e-6:
            print("\n[Phase 5] Skipped! (Global Optimum mathematically proven by LB)")
            return klj_partition, klj_cost, LB, 0.0, []

        print("\n[Phase 5] Memetic Evolutionary Search with LB Pruning...")
        # Run 3 independent EA restarts and keep the best result.
        # This dramatically reduces variance without changing the algorithm.
        NUM_RESTARTS = 3
        best_partition_overall = None
        best_cost_overall = float('inf')
        history = []

        for restart in range(NUM_RESTARTS):
            print(f"   [Restart {restart + 1}/{NUM_RESTARTS}]")
            ea = MemeticAlgorithm(self.original_G, pop_size=20, generations=100)
            partition, cost, h = ea.optimize(seed_partition=klj_partition)
            if cost < best_cost_overall:
                best_cost_overall = cost
                best_partition_overall = partition
                history = h  # keep convergence history of the best run

        final_partition = best_partition_overall
        final_cost = best_cost_overall
        print(f"-> Memetic Final Cost: {final_cost}")
        
        UB = final_cost
        # gap = abs(UB - LB) / abs(LB) if LB != 0 else 0.0
        
        # # Return history to main.py
        # return final_partition, UB, LB, gap, history
        
        # --- Automated Optimum Lookup (case-insensitive) ---
        known_optimum = lookup_known_optimum(filename)
        if known_optimum is not None:
            print(f"\n-> Note: Using literature known optimum ({known_optimum}) for gap calculation.")
        else:
            print("\n-> Note: Optimum unknown. Falling back to ICP Lower Bound for gap calculation.")
        
        # Formula: (c(x) - c(x*)) / |c(x*)|
        c_x_star = known_optimum if known_optimum is not None else LB
        
        if c_x_star != 0:
            gap = (UB - c_x_star) / abs(c_x_star)
        else:
            gap = 0.0

        # A UB strictly better than the known optimum is impossible — it means
        # the data parsing or sign convention is broken. Never hide this.
        if known_optimum is not None and UB < known_optimum - 1e-6:
            print(f"-> WARNING: UB ({UB}) is BETTER than the literature optimum "
                  f"({known_optimum}). This indicates a data or sign-convention bug!")

        return final_partition, UB, LB, gap, history