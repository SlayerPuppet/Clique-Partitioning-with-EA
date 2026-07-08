import random
from src.heuristics import klj_local_search
from src.reductions import apply_partial_optimality
from src.bounds import iterative_cycle_packing

class MemeticAlgorithm:
    def __init__(self, original_G, pop_size=20, generations=100,
                 crossover_rate=0.3, max_kick=0.40, use_lb_pruning=True,
                 verbose=True):
        self.original_G = original_G
        self.G = original_G.copy()
        for u, v, d in self.G.edges(data=True):
            if 'c_paper' not in d:
                d['c_paper'] = -d['cost']

        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.max_kick = max_kick
        self.use_lb_pruning = use_lb_pruning
        self.verbose = verbose
        self.best_UB = float('inf')
        self.best_partition = None

        # Adaptive disruption parameters
        self.stagnation = 0
        self.high_stagnation = 0  # consecutive generations at max kick
        self.kick_pct = 0.05

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
        H = self.G.copy()
        fixed_cost = 0

        for cluster in fixed_partition:
            nodes = list(cluster)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if self.original_G.has_edge(nodes[i], nodes[j]):
                        fixed_cost += self.original_G[nodes[i]][nodes[j]]['cost']

        for cluster in fixed_partition:
            if len(cluster) < 2:
                continue
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
                                H.add_edge(u, neighbor,
                                           cost=H[v][neighbor]['cost'],
                                           c_paper=H[v][neighbor]['c_paper'])
                    H.remove_node(v)

        reduced_H = apply_partial_optimality(H)
        if reduced_H.number_of_edges() == 0:
            return fixed_cost

        paper_lb, _ = iterative_cycle_packing(reduced_H)
        sum_reduced = sum(d['cost'] for u, v, d in reduced_H.edges(data=True))
        return fixed_cost + sum_reduced + paper_lb

    def initialize_population(self, seed_partition):
        population = [seed_partition]
        if self.verbose:
            print("      -> Generating diverse starting pool from seed...")

        attempts = 0
        while len(population) < self.pop_size and attempts < self.pop_size * 5:
            attempts += 1
            original_kick = self.kick_pct
            self.kick_pct = 0.20
            result = self.mutate_and_refine(seed_partition)
            self.kick_pct = original_kick

            if result is not None:
                offspring, cost = result
                population.append(offspring)
                if cost < self.best_UB:
                    self.best_UB = cost
                    self.best_partition = offspring

        while len(population) < self.pop_size:
            population.append(seed_partition)

        return population

    def crossover(self, parent1, parent2):
        """
        Alternating cluster inheritance crossover.
        Sorts each parent's clusters by internal cohesion (most attractive first),
        then alternately picks the best remaining cluster from P1 and P2.
        Nodes already assigned are stripped from later-picked clusters.
        """
        def cluster_quality(cluster):
            return sum(
                self.original_G[u][v]['cost']
                for u in cluster for v in cluster
                if u < v and self.original_G.has_edge(u, v)
            )

        p1 = sorted([set(c) for c in parent1], key=cluster_quality)
        p2 = sorted([set(c) for c in parent2], key=cluster_quality)

        child = []
        assigned = set()
        sources = [p1, p2]
        turn = 0

        while p1 or p2:
            src = sources[turn % 2]
            turn += 1
            if not src:
                continue
            cluster = src.pop(0)
            new_cluster = cluster - assigned
            if new_cluster:
                child.append(new_cluster)
                assigned |= new_cluster

        for node in self.original_G.nodes():
            if node not in assigned:
                child.append({node})

        return child

    def mutate_and_refine(self, partition):
        new_partition = [set(c) for c in partition]
        if len(new_partition) < 2:
            return None

        num_nodes_to_move = max(3, int(self.original_G.number_of_nodes() * self.kick_pct))
        unassigned = set()

        # RUIN: random selection (random ruin preserves population diversity)
        for _ in range(num_nodes_to_move):
            valid_sources = [i for i, c in enumerate(new_partition) if c]
            if not valid_sources:
                break
            src_idx = random.choice(valid_sources)
            node = random.choice(list(new_partition[src_idx]))
            new_partition[src_idx].remove(node)
            unassigned.add(node)

        new_partition = [c for c in new_partition if c]

        # PRUNE (skippable for ablation studies)
        if self.use_lb_pruning:
            partial_lb = self.get_partial_lb(new_partition)
            if partial_lb >= self.best_UB:
                return None

        # RECREATE: greedy insertion — always join least-cost cluster (original behaviour)
        for node in unassigned:
            best_target = -1
            best_delta = float('inf')

            for target_idx in range(len(new_partition)):
                delta = sum(
                    self.original_G[node][v]['cost']
                    for v in new_partition[target_idx]
                    if self.original_G.has_edge(node, v)
                )
                if delta < best_delta:
                    best_delta = delta
                    best_target = target_idx

            # Create singleton only if node is net-repulsive to every cluster
            if best_target == -1 or best_delta >= 0:
                new_partition.append({node})
            else:
                new_partition[best_target].add(node)

        mutated_partition = [c for c in new_partition if c]

        # REFINE
        refined_partition, refined_cost = klj_local_search(self.original_G, mutated_partition)
        return refined_partition, refined_cost

    def optimize(self, seed_partition):
        if self.verbose:
            print("   -> Initializing Memetic Population...")
        self.best_UB = self.calculate_cost(seed_partition)
        self.best_partition = seed_partition
        population = self.initialize_population(seed_partition)

        history = [self.best_UB]

        for gen in range(self.generations):
            scored_pop = [(self.calculate_cost(p), p) for p in population]
            scored_pop.sort(key=lambda x: x[0])

            current_gen_best_cost, current_gen_best_part = scored_pop[0]

            if current_gen_best_cost < self.best_UB:
                self.best_UB = current_gen_best_cost
                self.best_partition = current_gen_best_part
                self.stagnation = 0
                self.high_stagnation = 0
                self.kick_pct = 0.05
            else:
                self.stagnation += 1
                if self.stagnation > 3:
                    self.kick_pct = min(self.max_kick, self.kick_pct + 0.05)
                if self.kick_pct >= self.max_kick:
                    self.high_stagnation += 1

            # SOFT RESTART: reset the kick cycle without changing the population.
            # This lets the adaptive disruption re-escalate from small kicks,
            # which often finds improvements that maxed-out kicks miss.
            if self.high_stagnation > 5:
                self.stagnation = 0
                self.high_stagnation = 0
                self.kick_pct = 0.05

            survivors = [p for _, p in scored_pop[:self.pop_size // 2]]
            next_gen = list(survivors)

            pruned_count = 0
            while len(next_gen) < self.pop_size and pruned_count < 50:
                # crossover → mutate+refine, else direct mutate+refine
                if len(survivors) >= 2 and random.random() < self.crossover_rate:
                    p1, p2 = random.sample(survivors, 2)
                    child = self.crossover(p1, p2)
                    result = self.mutate_and_refine(child)
                else:
                    parent = random.choice(survivors)
                    result = self.mutate_and_refine(parent)

                if result is not None:
                    offspring, off_cost = result
                    next_gen.append(offspring)
                    if off_cost < self.best_UB:
                        self.best_UB = off_cost
                        self.best_partition = offspring
                else:
                    pruned_count += 1

            while len(next_gen) < self.pop_size:
                next_gen.append(random.choice(survivors))

            population = next_gen
            history.append(self.best_UB)

            if self.verbose and gen % 10 == 0:
                print(f"   -> Generation {gen} Best Cost: {self.best_UB} "
                      f"(Kick: {int(self.kick_pct*100)}%, Pruned: {pruned_count})")

        return self.best_partition, self.best_UB, history
