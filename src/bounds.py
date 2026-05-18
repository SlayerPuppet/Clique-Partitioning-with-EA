from scipy.optimize import linprog

def simplex_lower_bound(G):
    """
    Computes a lower bound using LP relaxation with Simplex.
    Formulation: min sum(c_e * x_e)
    s.t. x_ij + x_jk - x_ik <= 1  (Triangle inequalities)
    0 <= x_e <= 1
    """
    edges = list(G.edges(data=True))
    edge_idx = {(u, v): i for i, (u, v, _) in enumerate(edges)}
    edge_idx.update({(v, u): i for i, (u, v, _) in enumerate(edges)})
    
    c = [data['cost'] for u, v, data in edges]
    
    # Construct sparse constraints for triangle inequalities
    A_ub = []
    b_ub = []
    nodes = list(G.nodes())
    
    # Generate a subset of triangle inequalities for performance
    for i in nodes:
        for j in nodes:
            for k in nodes:
                if i < j and j < k:
                    if (i, j) in edge_idx and (j, k) in edge_idx and (i, k) in edge_idx:
                        e1, e2, e3 = edge_idx[(i,j)], edge_idx[(j,k)], edge_idx[(i,k)]
                        
                        row1 = [0]*len(edges); row1[e1]=1; row1[e2]=1; row1[e3]=-1
                        A_ub.append(row1); b_ub.append(1)
                        
                        row2 = [0]*len(edges); row2[e1]=1; row2[e2]=-1; row2[e3]=1
                        A_ub.append(row2); b_ub.append(1)
                        
                        row3 = [0]*len(edges); row3[e1]=-1; row3[e2]=1; row3[e3]=1
                        A_ub.append(row3); b_ub.append(1)

    # Use Highs-DS (Dual Simplex) via SciPy
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs-ds')
    return res.fun if res.success else None