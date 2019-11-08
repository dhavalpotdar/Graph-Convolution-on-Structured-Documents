import numpy as np

# the call function implements the logic of the Adjacency Layer
def call(adj, node_vec):
    '''
        Generates an adjacency matrix that consolidates the neighborhood
        information

        if adj_ij == 0, new_adj_ij = 0
        if adj_ij == 1, new_adj_ig = |node_vec_i - node_vec_j|
    '''
    new_adj = np.zeros_like(adj, dtype='float')

    # iterate rows
    for ik, i in enumerate(adj):
        # iterate columns
        for jk,j in enumerate(i):
            adj_ij = j

            if adj_ij == 0:
                new_adj[ik,jk] = 0

            elif adj_ij == 1:
                new_adj[ik,jk] = np.linalg.norm(node_vec[ik] - node_vec[jk])

    return new_adj
