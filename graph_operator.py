from tensorflow.keras.layers import Layer
import numpy as np

class GraphOperator(Layer):
    '''
        Layer to generate Adjacency matrices raised to specified power.

        Input:
            power: int, power to raise the input adjacency matrix to such that\
                 a set is generated:
                    A(k) = [A^0, A^1, .., A^power]

        Output:
            A list of adjacency matrices generated as above.
    '''
    def __init__(self, power):
        super(GraphOperator, self).__init__()
        self.power = power

    # TODO: test logic
    def call(self, adj):
        
        power_range = range(0, self.power+1)
        
        adj_list = [np.linalg.matrix_power(adj, p) for p in power_range]

        return adj_list

    def __eq__(self, other):
        if type(self) == type(other):
            return self.node_one == other.node_one and self.node_two == other.node_two
        else:
            raise TypeError