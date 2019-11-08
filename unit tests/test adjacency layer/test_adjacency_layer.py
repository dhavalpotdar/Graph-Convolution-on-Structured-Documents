import unittest
from adjacency_layer import call

import numpy as np

# test input feature matrix
node_feat_mat = np.array([[1, 1, 1, 1], 
                            [2, 2, 2, 2], 
                            [3, 3, 3, 3], 
                            [4, 4, 4, 4], 
                            [5, 5, 5, 5]])

class TestAdjacencyLayer(unittest.TestCase):
    def test_all_zeros(self):
        
        # test inputs
        adj = np.zeros(shape=(5,5))
        
        # test answer
        adj_ans = np.zeros(shape=(5,5))

        # test logic
        A = call(adj, node_feat_mat)

        self.assertTupleEqual(adj_ans.shape, A.shape)
        self.assertListEqual(adj_ans.flatten().tolist(), 
                                A.flatten().tolist())
        
    def test_case1(self):

        # test inputs
        adj = np.array([[0,1,0,0,1], 
                        [1,0,0,0,0], 
                        [0,0,0,1,1], 
                        [0,0,1,0,0], 
                        [1,0,1,0,0]])
        
        # test answer
        adj_ans = np.array([[0,2,0,0,8], 
                            [2,0,0,0,0], 
                            [0,0,0,2,4], 
                            [0,0,2,0,0], 
                            [8,0,4,0,0]])

        # test logic
        A = call(adj, node_feat_mat)

        self.assertTupleEqual(adj_ans.shape, A.shape)
        self.assertListEqual(adj_ans.flatten().tolist(), 
                                A.flatten().tolist())
