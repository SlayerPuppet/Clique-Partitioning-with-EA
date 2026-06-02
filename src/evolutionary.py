import random
from src.heuristics import klj_local_search

class MemeticAlgorithm:
    def __init__(self, original_G, pop_size=10, generations=50):
        self.G = original_G
        self.pop_size = pop_size
        self.generations = generations

    def calculate_cost(self, partition):
        cost = 0
        for cluster in partition:
            nodes = list(cluster)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if self.G.has_edge(nodes[i], nodes[j]):
                        cost += self.G[nodes[i]][nodes[j]]['cost']
        return cost

    def initialize_population(self, seed_partition):
        """
        Seeds the population with the highly optimized KLj result,
        plus random valid partitions.
        """
        population = [seed_partition]
        nodes = list(self.G.nodes())
        
        for _ in range(self.pop_size - 1):
            random.shuffle(nodes)
            split_idx = random.randint(1, len(nodes) - 1)
            random_partition = [{n} for n in nodes[:split_idx]] + [set(nodes[split_idx:])]
            # Refine the random guess using local search before it enters the gene pool
            refined_partition, _ = klj_local_search(self.G, random_partition)
            population.append(refined_partition)
            
        return population

    def mutate_and_refine(self, partition):
        """
        The Memetic step: Mutate to jump out of a local minimum, 
        then use KLj to slide into a new one.
        """
        new_partition = [set(c) for c in partition]
        if len(new_partition) < 2: 
            return new_partition
        
        # 1. Mutate: Randomly shift a node
        source_idx = random.randint(0, len(new_partition) - 1)
        if not new_partition[source_idx]: 
            return new_partition
            
        node = new_partition[source_idx].pop()
        target_idx = random.choice([i for i in range(len(new_partition)) if i != source_idx])
        new_partition[target_idx].add(node)
        
        mutated_partition = [c for c in new_partition if c]
        
        # 2. Refine: Apply KLj to the mutation
        refined_partition, _ = klj_local_search(self.G, mutated_partition)
        return refined_partition

    def optimize(self, seed_partition):
        print("   -> Initializing Memetic Population...")
        population = self.initialize_population(seed_partition)
        
        for gen in range(self.generations):
            scored_pop = [(self.calculate_cost(p), p) for p in population]
            scored_pop.sort(key=lambda x: x[0])
            
            # Elitism: Keep top 50%
            survivors = [p for cost, p in scored_pop[:self.pop_size // 2]]
            
            # Generate offspring
            next_gen = list(survivors)
            while len(next_gen) < self.pop_size:
                parent = random.choice(survivors)
                next_gen.append(self.mutate_and_refine(parent))
                
            population = next_gen
            
            if gen % 10 == 0:
                print(f"   -> Generation {gen} Best Cost: {scored_pop[0][0]}")
            
        best_cost, best_partition = min([(self.calculate_cost(p), p) for p in population], key=lambda x: x[0])
        return best_partition, best_cost
    
    def run_pipeline(self):
        print("\n[Phase 1] Partial Optimality Preprocessing...")
        reduced_G = self.partial_optimality()
        print(f"-> Reduced Graph: {reduced_G.number_of_nodes()} nodes")
        
        if reduced_G.number_of_edges() == 0:
            return None, 0, 0, 0

        print("\n[Phase 2] Iterative Cycle Packing (ICP)...")
        paper_lb, residuals = self.iterative_cycle_packing(reduced_G)
        LB = -paper_lb 
        print(f"-> ICP Lower Bound: {LB}")
        
        print("\n[Phase 3] Reweighting and GAEC...")
        gaec_partition = self.reweight_and_gaec(reduced_G, residuals)
        gaec_cost = self._calculate_original_cost(gaec_partition)
        print(f"-> Reweighted GAEC Cost: {gaec_cost}")
        
        print("\n[Phase 4] Kernighan-Lin with Joins (KLj)...")
        klj_partition, klj_cost = self.klj_local_search(gaec_partition)
        print(f"-> Final Primal Cost (KLj Upper Bound): {klj_cost}")

        print("\n[Phase 5] Memetic Evolutionary Search...")
        # Pass the original graph to the EA so it evaluates true costs
        ea = MemeticAlgorithm(self.original_G, pop_size=10, generations=30)
        final_partition, final_cost = ea.optimize(seed_partition=klj_partition)
        print(f"-> Memetic Final Cost: {final_cost}")
        
        # Calculate final gap based on the EA's best result
        UB = final_cost
        gap = (UB - LB) / abs(UB) if UB != 0 else 0
        
        return final_partition, UB, LB, gap