from tensorflow import keras
from layers import GraphOperator, GNN, Adjacency
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import tensorflow
from tensorflow.keras.utils import plot_model
# import keras

tensorflow.enable_eager_execution()

######################## model logic ##########################################


# ================== input ====================================================
adjacency_input = keras.Input(shape=(50,50), batch_size=1, name='Adjacency Input')
node_input = keras.Input(shape=(50, 5), batch_size=1, name='Node Input')

# ================== preprocessing ============================================
node_embeddings = layers.Dense(units=50, name='Embedding')(node_input)
adj_list = GraphOperator()(adjacency_input)

# ========================== residual block 1 =================================
# adj_list_1 contains the adjacency matrices for the input graph and the node
# vector of the input layer
adj_list_1 = [adj_tensor for adj_tensor in adj_list]
adj_list_1.append(node_embeddings)

# AL
adj_0_1, adj_1_1, adj_2_1 = Adjacency(n_features=50, max_nodes=50)(adj_list_1)

# GNN 1                                                          
gnn_1_1 = GNN(n_features=50, n_nodes=50)(inputs = [node_embeddings, adj_0_1, 
                                                    adj_1_1, adj_2_1])
# BN1 + ReLU
bn_1_1 = layers.BatchNormalization(name='BN_1.1')(gnn_1_1)
relu_1 = layers.ReLU(name='ReLU_1')(bn_1_1)

# GNN 2
gnn_2_1 = GNN(n_features=50, n_nodes=50)(inputs = [relu_1, adj_0_1, adj_1_1, 
                                                    adj_2_1])

# BN2
bn_2_1 = layers.BatchNormalization(name='BN_1.2')(gnn_2_1)

# Addition
add_1 = layers.add([bn_2_1, node_embeddings], name='Add_1')


# =========================== residual block 2 =================================
# adj_list_2 contains the adjacency matrices for the input graph and the node
# vector of the previous
adj_list_2 = [adj_tensor for adj_tensor in adj_list]
adj_list_2.append(add_1)

# AL
adj_0_2, adj_1_2, adj_2_2 = Adjacency(n_features=50, max_nodes=50)(adj_list_2) 
                                                            
# GNN 1                                                          
gnn_1_2 = GNN(n_features=50, n_nodes=50)(inputs = [add_1, adj_0_2, adj_1_2, 
                                                            adj_2_2])

# BN1 + ReLU
bn_1_2 = layers.BatchNormalization(name='BN_2.1')(gnn_1_2)
relu_2 = layers.ReLU(name='ReLU_2')(bn_1_2)

# GNN 2
gnn_2_2 = GNN(n_features=50, n_nodes=50)(inputs = [relu_2, adj_0_2, 
                                                    adj_1_2, adj_2_2])

# BN2
bn_2_2 = layers.BatchNormalization(name='BN_2.2')(gnn_2_2)

# Addition
add_2 = layers.add([bn_2_2, add_1], name='Add_2')


# ========================= residual block 3 ===================================
# adj_list_2 contains the adjacency matrices for the input graph and the node
# vector of the previous
adj_list_3 = [adj_tensor for adj_tensor in adj_list]
adj_list_3.append(add_2)

# AL
adj_0_3, adj_1_3, adj_2_3 = Adjacency(n_features=50, max_nodes=50)(adj_list_3)                                                            

# GNN 1                                                          
gnn_1_3 = GNN(n_features=50, n_nodes=50)(inputs = [add_2, adj_0_3, 
                                                    adj_1_3, adj_2_3])
# BN1 + ReLU
bn_1_3 = layers.BatchNormalization(name='BN_3.1')(gnn_1_3)
relu_3 = layers.ReLU(name='ReLU_3')(bn_1_3)

# GNN 2
gnn_2_3 = GNN(n_features=50, n_nodes=50)(inputs = [relu_3, adj_0_3, 
                                                    adj_1_3, adj_2_3])

# BN2
bn_2_3 = layers.BatchNormalization(name='BN_3.2')(gnn_2_3)

# Addition
add_3 = layers.add([bn_2_3, add_2], name='Add_3')


# ===================== linear classifier =====================================

node_labels = layers.Dense(14, activation='sigmoid', name='Classifier')(add_3)


# ======================= output adjacency learning layer =====================

# not implemented


##################### model compilation #######################################

opt = SGD(lr = 0.001, momentum=0.9)

model = keras.Model(inputs=[adjacency_input, node_input], 
                    outputs=[node_labels])


# plot_model(model, './gcn_model.png', show_shapes=True)

# model.compile(optimizer=opt, 
#             loss='categorical_crossentropy', 
#             metrics=['accuracy'])

# import keras
# import pydotplus
# from keras.utils.vis_utils import model_to_dot
# keras.utils.vis_utils.pydot = pydot
plot_model(model, to_file='model.png', show_shapes=True, rankdir='TB')