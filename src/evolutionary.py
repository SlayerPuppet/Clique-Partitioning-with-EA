import random
from src.heuristics import klj_local_search
from src.reductions import apply_partial_optimality
from src.bounds import iterative_cycle_packing

class MemeticAlgorithm:
    def __init__(self, original_G, pop_size=10, generations=30):
        self.original_G = original_G
        # Create a working copy for partial bound calculations
        self.G = original_G.copy()
        for u, v, d in self.G.edges(data=True):
            if 'c_paper' not in d:
                d['c_paper'] = -d['cost']
                
        self.pop_size = pop_size
        self.generations = generations
        self.best_UB = float('inf')

    def calculate_cost(self, partition):
        cost = 0
        for cluster in partition:
            nodes = list(cluster)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if self.original_G.has_edge(nodes[i], nodes[j]):
                        cost += self.original_G[nodes[i]][nodes[j]]['cost']
        return cost

    def get_partial_lb(self, fixed_partition):
            """Calculates the lower bound of a partially assigned graph."""
            H = self.G.copy()
            
            # 1. Accurately calculate the cost of the fixed clusters (Fix for missed internal edges)
            fixed_cost = 0
            for cluster in fixed_partition:
                nodes = list(cluster)
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        if self.original_G.has_edge(nodes[i], nodes[j]):
                            fixed_cost += self.original_G[nodes[i]][nodes[j]]['cost']
            
            # 2. Fix variables by contracting nodes that are grouped together
            for cluster in fixed_partition:
                if len(cluster) < 2: continue
                nodes = list(cluster)
                u = nodes[0]
                for v in nodes[1:]:
                    if H.has_node(v):
                        for neighbor in list(H.neighbors(v)):
                            if neighbor != u:
                                if H.has_edge(u, neighbor):
                                    H[u][neighbor]['cost'] += H[v][neighbor]['cost']
                                    H[u][neighbor]['c_paper'] += H[v][neighbor]['c_paper']
                                else:
                                    H.add_edge(u, neighbor, cost=H[v][neighbor]['cost'], c_paper=H[v][neighbor]['c_paper'])
                        H.remove_node(v)
            
            # 3. Apply Partial Optimality to the fixed graph
            reduced_H = apply_partial_optimality(H)
            if reduced_H.number_of_edges() == 0:
                return fixed_cost
                
            paper_lb, _ = iterative_cycle_packing(reduced_H)
            sum_reduced = sum(d['cost'] for u,v,d in reduced_H.edges(data=True))
            return fixed_cost + sum_reduced + paper_lb

    def initialize_population(self, seed_partition):
        population = [seed_partition]
        nodes = list(self.original_G.nodes())
        
        while len(population) < self.pop_size:
            random.shuffle(nodes)
            split_idx = random.randint(1, len(nodes) - 1)
            random_partition = [{n} for n in nodes[:split_idx]] + [set(nodes[split_idx:])]
            refined_partition, cost = klj_local_search(self.original_G, random_partition)
            if cost < self.best_UB:
                self.best_UB = cost
            population.append(refined_partition)
            
        return population

    def mutate_and_refine(self, partition):
        new_partition = [set(c) for c in partition]
        if len(new_partition) < 2: return new_partition
        
        num_nodes_to_move = max(3, int(self.original_G.number_of_nodes() * 0.10))
        unassigned = set()
        
        # 1. KICK: Unassign a few nodes
        for _ in range(num_nodes_to_move):
            valid_sources = [i for i, c in enumerate(new_partition) if c]
            if not valid_sources: break
            source_idx = random.choice(valid_sources)
            node = random.choice(list(new_partition[source_idx]))
            new_partition[source_idx].remove(node)
            unassigned.add(node)
            
        new_partition = [c for c in new_partition if c]
        
        # 2. PRUNE: Find new lower bound. If higher (worse) than best UB, discard!
        partial_lb = self.get_partial_lb(new_partition)
        if partial_lb >= self.best_UB:
            return None # Discard this pool instantly
            
        # 3. REASSIGN randomly if it passed the bounding check
        for node in unassigned:
            target_indices = [i for i in range(len(new_partition))] + [-1]
            target_idx = random.choice(target_indices)
            if target_idx == -1:
                new_partition.append({node})
            else:
                new_partition[target_idx].add(node)
        
        mutated_partition = [c for c in new_partition if c]
        
        # 4. REFINE with KLj
        refined_partition, refined_cost = klj_local_search(self.original_G, mutated_partition)
        if refined_cost < self.best_UB:
            self.best_UB = refined_cost
            
        return refined_partition

    def optimize(self, seed_partition):
        print("   -> Initializing Memetic Population...")
        self.best_UB = self.calculate_cost(seed_partition)
        population = self.initialize_population(seed_partition)
        
        for gen in range(self.generations):
            scored_pop = [(self.calculate_cost(p), p) for p in population]
            scored_pop.sort(key=lambda x: x[0])
            
            survivors = [p for cost, p in scored_pop[:self.pop_size // 2]]
            next_gen = list(survivors)
            
            pruned_count = 0
            while len(next_gen) < self.pop_size and pruned_count < 50:
                parent = random.choice(survivors)
                offspring = self.mutate_and_refine(parent)
                
                if offspring is not None:
                    next_gen.append(offspring)
                else:
                    pruned_count += 1 # The LB check saved us from running KLj!
                
            while len(next_gen) < self.pop_size:
                 next_gen.append(random.choice(survivors))

            population = next_gen
            
            if gen % 10 == 0:
                print(f"   -> Generation {gen} Best Cost: {self.best_UB} (Bad mutations pruned: {pruned_count})")
            
        best_cost, best_partition = min([(self.calculate_cost(p), p) for p in population], key=lambda x: x[0])
        return best_partition, best_cost