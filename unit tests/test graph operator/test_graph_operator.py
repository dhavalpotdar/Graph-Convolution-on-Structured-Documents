import unittest
from graph_operator import GraphOperator
import numpy as np

# test_adj = np.array([[0,1,1],
#                      [1,0,0],
#                      [1,0,0]])


class GraphOperatorTest(unittest.TestCase):

    def test_empty_case(self):
        POWER = 2
        # define test case
        test_adj = np.zeros(shape=(5,5))

        ANSWER_pow0 = np.eye(5)
        ANSWER_pow1 = test_adj
        ANSWER_pow2 = np.matmul(test_adj, test_adj)

        # define test logic
        g_op = GraphOperator(POWER)
        adj_lst = g_op(test_adj)

        mat0 = adj_lst[0]
        mat1 = adj_lst[1]
        mat2 = adj_lst[2]

        assert ANSWER_pow0.flatten().tolist() == mat0.flatten().tolist()
        assert ANSWER_pow1.flatten().tolist() == mat1.flatten().tolist()
        assert ANSWER_pow2.flatten().tolist() == mat2.flatten().tolist()

        assert self.assertListEqual(ANSWER_pow0.flatten().tolist(), mat0.flatten().tolist())
        assert self.assertListEqual(ANSWER_pow1.flatten().tolist(), mat1.flatten().tolist())
        assert self.assertListEqual(ANSWER_pow2.flatten().tolist(), mat2.flatten().tolist())


    def test_case1(self):
        POWER = 2
        # define test case
        test_adj = np.array([[0,1,1],
                             [1,0,0],
                             [1,0,0]], dtype='int')

        ANSWER_pow0 = np.eye(3, dtype='int')
        ANSWER_pow1 = test_adj
        ANSWER_pow2 = np.matmul(test_adj, test_adj)

        # test logic
        g_op = GraphOperator(2)
        adj_lst = g_op(test_adj)

        mat0 = adj_lst[0]
        mat1 = adj_lst[1]
        mat2 = adj_lst[2]

        assert ANSWER_pow0.flatten().tolist() == mat0.flatten().tolist()
        assert ANSWER_pow1.flatten().tolist() == mat1.flatten().tolist()
        assert ANSWER_pow2.flatten().tolist() == mat2.flatten().tolist()

        # even this is failing:
        # assert self.assertEqual(3, 3)

        assert self.assertAlmostEqual(ANSWER_pow0.flatten().tolist(), mat0.flatten().tolist())
        assert self.assertListEqual(ANSWER_pow1.flatten().tolist(), mat1.flatten().tolist())
        assert self.assertListEqual(ANSWER_pow2.flatten().tolist(), mat2.flatten().tolist())

    def test_case_power_5(self):
        POWER = 5
        # define test case
        test_adj = np.array([[0, 1, 1, 0, 0, 1],
                             [1, 0, 0, 1, 1, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 1, 0, 0]])

        ANSWER_pow0 = np.eye(6)
        ANSWER_pow1 = test_adj
        ANSWER_pow2 = np.matmul(test_adj, test_adj)
        ANSWER_pow3 = np.matmul(ANSWER_pow2, test_adj)
        ANSWER_pow4 = np.matmul(ANSWER_pow3, test_adj)
        ANSWER_pow5 = np.matmul(ANSWER_pow4, test_adj)

        # define test logic
        g_op = GraphOperator(POWER)
        adj_lst = g_op(test_adj)

        assert ANSWER_pow0.flatten().tolist() == adj_lst[0].ravel().tolist()
        assert ANSWER_pow1.flatten().tolist() == adj_lst[1].ravel().tolist()
        assert ANSWER_pow2.flatten().tolist() == adj_lst[2].ravel().tolist()
        assert ANSWER_pow3.flatten().tolist() == adj_lst[3].ravel().tolist()
        assert ANSWER_pow4.flatten().tolist() == adj_lst[4].ravel().tolist()
        assert ANSWER_pow5.flatten().tolist() == adj_lst[5].ravel().tolist()

        # assert self.assertEqual(POWER, len(adj_lst))
        assert self.assertListEqual(ANSWER_pow0.ravel().tolist(), adj_lst[0].ravel().tolist())
        assert self.assertListEqual(ANSWER_pow1.ravel().tolist(), adj_lst[1].ravel().tolist())
        assert self.assertListEqual(ANSWER_pow2.ravel().tolist(), adj_lst[2].ravel().tolist())
        assert self.assertListEqual(ANSWER_pow3.ravel().tolist(), adj_lst[3].ravel().tolist())
        assert self.assertListEqual(ANSWER_pow4.ravel().tolist(), adj_lst[4].ravel().tolist())
