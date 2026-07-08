from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import networkx as nx

def simplex_lower_bound(G):
    """
    Computes a lower bound using the LP relaxation over triangle inequalities.
    Formulation (x_e = 1 means edge e is joined):
        min sum(c_e * x_e)
        s.t. x_ij + x_jk - x_ik <= 1   (all 3 rotations, per triangle)
             0 <= x_e <= 1
    Uses a sparse constraint matrix (3 nonzeros per row) and enumerates
    only triangles that actually exist in G.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return 0.0

    edge_idx = {}
    for i, (u, v, _) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i

    c = [d['cost'] for _, _, d in edges]

    rows, cols, vals = [], [], []
    n_rows = 0
    # Enumerate real triangles: for each edge (u,v), common neighbors w > max(u,v)
    for u, v, _ in edges:
        for w in nx.common_neighbors(G, u, v):
            if w <= u or w <= v:
                continue
            e1, e2, e3 = edge_idx[(u, v)], edge_idx[(u, w)], edge_idx[(v, w)]
            for signs in ((1, 1, -1), (1, -1, 1), (-1, 1, 1)):
                rows.extend([n_rows] * 3)
                cols.extend([e1, e2, e3])
                vals.extend(signs)
                n_rows += 1

    if n_rows == 0:
        # No triangles: LP separates into independent edges
        return sum(min(ce, 0.0) for ce in c)

    A_ub = coo_matrix((vals, (rows, cols)), shape=(n_rows, len(edges))).tocsr()
    b_ub = [1.0] * n_rows

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')
    return res.fun if res.success else None


# --- Contribution 2: Iterative Cycle Packing (ICP) ---
def iterative_cycle_packing(reduced_G, max_rounds=50):
    """Computes the lower bound by iterative greedy cycle packing.

    A conflicted cycle is a repulsive edge closing a path of attractive
    edges. Each packed cycle consumes residual capacity on its edges;
    saturated attractive edges are removed from the residual graph so new
    (longer) paths are discovered in later rounds. Repeats until no
    packable cycle remains.
    """
    residuals = {tuple(sorted((u, v))): abs(d['c_paper']) for u, v, d in reduced_G.edges(data=True)}
    L_triv = sum(d['c_paper'] for u, v, d in reduced_G.edges(data=True) if d['c_paper'] < 0)

    repulsive = [tuple(sorted((u, v))) for u, v, d in reduced_G.edges(data=True) if d['c_paper'] < 0]
    attractive = [tuple(sorted((u, v))) for u, v, d in reduced_G.edges(data=True) if d['c_paper'] > 0]

    L_dual = 0.0
    for _ in range(max_rounds):
        # Residual attractive graph: only edges with capacity left
        G_plus = nx.Graph()
        G_plus.add_nodes_from(reduced_G.nodes())
        G_plus.add_edges_from(e for e in attractive if residuals[e] > 0)

        packed_any = False
        for e_rep in repulsive:
            if residuals[e_rep] <= 0:
                continue
            u, v = e_rep
            try:
                path = nx.shortest_path(G_plus, source=u, target=v)
            except nx.NetworkXNoPath:
                continue

            cycle = [e_rep] + [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]
            y_C = min(residuals[e] for e in cycle)
            if y_C > 0:
                for e in cycle:
                    residuals[e] -= y_C
                    # Remove saturated attractive edges so later searches reroute
                    if e != e_rep and residuals[e] <= 0 and G_plus.has_edge(*e):
                        G_plus.remove_edge(*e)
                L_dual += y_C
                packed_any = True

        if not packed_any:
            break

    return L_triv + L_dual, residuals