class CliquePartitioningEA:
    def __init__(self, G, pop_size=10, generations=50):
        self.G = G
        self.pop_size = pop_size
        self.generations = generations
        
    def calculate_cost(self, partition):
        cost = 0
        for cluster in partition:
            nodes = list(cluster)
            # Add cost of edges within the same cluster
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if self.G.has_edge(nodes[i], nodes[j]):
                        cost += self.G[nodes[i]][nodes[j]]['cost']
        return cost

    def initialize_population(self):
        population = []
        # Seed with GAEC
        population.append(gaec(self.G))
        # Fill remainder with random valid partitions
        nodes = list(self.G.nodes())
        for _ in range(self.pop_size - 1):
            random.shuffle(nodes)
            split_idx = random.randint(1, len(nodes) - 1)
            population.append([{n} for n in nodes[:split_idx]] + [set(nodes[split_idx:])])
        return population

    def mutate(self, partition):
        # Randomly move a node to a different cluster
        new_partition = [set(c) for c in partition]
        if len(new_partition) < 2: return new_partition
        
        source_idx = random.randint(0, len(new_partition) - 1)
        if not new_partition[source_idx]: return new_partition
        
        node = new_partition[source_idx].pop()
        target_idx = random.choice([i for i in range(len(new_partition)) if i != source_idx])
        new_partition[target_idx].add(node)
        
        return [c for c in new_partition if c]

    def optimize(self):
        population = self.initialize_population()
        
        for gen in range(self.generations):
            # Evaluate fitness (lower cost is better)
            scored_pop = [(self.calculate_cost(p), p) for p in population]
            scored_pop.sort(key=lambda x: x[0])
            
            # Select top 50%
            survivors = [p for cost, p in scored_pop[:self.pop_size // 2]]
            
            # Generate offspring via mutation
            next_gen = list(survivors)
            while len(next_gen) < self.pop_size:
                parent = random.choice(survivors)
                next_gen.append(self.mutate(parent))
                
            population = next_gen
            
        best_cost, best_partition = min([(self.calculate_cost(p), p) for p in population], key=lambda x: x[0])
        return best_partition, best_cost

# --- Execution ---
G = load_benchmark_graph("path/to/dataset")

# 1. Get Lower Bound via Simplex
lb = simplex_lower_bound(G)
print(f"Simplex Lower Bound: {lb}")

# 2. Run EA
ea = CliquePartitioningEA(G)
best_partition, best_cost = ea.optimize()
print(f"EA Best Cost: {best_cost}")
print(f"Clusters: {best_partition}")