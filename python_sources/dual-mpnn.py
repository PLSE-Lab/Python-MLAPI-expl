#!/usr/bin/env python
# coding: utf-8

# ## Message Passing Neural Network
# 
# So, as many of you might have surmised by now the dataset for this challenge is essentially the QM9 dataset with some new values calculated for it. 
# 
# The first thing I though of when seeing this challenge was the [Gilmer paper](https://arxiv.org/abs/1704.01212), as it uses the QM9 dataset. ([see this talk](https://vimeo.com/238221016))
# 
# The major difference in this challenge is that we are asked to calulate bond properties (thus edges in a graph) as opposed to bulk properties in the paper. 
# 
# Here the model is laid out in a modular way so the parts can easily be replaced
# 

# In[ ]:


# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[ ]:


# Internet needs to be on
get_ipython().system('pip install tensorflow-gpu==2.0a0')


# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle
import os
print(os.listdir("../input"))
import pickle 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Make sure tf 2.0 alpha has been installed
print(tf.__version__)


# In[ ]:


#is it using the gpu?
tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)


# In[ ]:


tf.random.set_seed(42)
datadir = "../input/"


# ## Message passer
# 
# The message passer here is a MLP that takes $concat([node_i, edge_{ij}, node_j])$ as input and returns a message of the same dimension of the node

# In[ ]:


class Message_Passer_1(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, state_dim):
        super(Message_Passer_1, self).__init__()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=state_dim, activation=tf.nn.relu)


        
    def call(self, node_i, node_j, edge_ij):
        concat = self.concat_layer([node_i, node_j, edge_ij])
        activation = self.hidden_layer_1(concat)
        return self.output_layer(activation)


# ## Aggregator
# 
# Define the message aggregator (just sum)  
# Probably overkill to have it as its own layer, but good if you want to replace it with something more complex
# 

# In[ ]:


class Message_Agg(tf.keras.layers.Layer):
    def __init__(self):
        super(Message_Agg, self).__init__()
    
    def call(self, messages):
        return tf.math.reduce_sum(messages, 2)


# ## Node Update function
# 
# The node update function is an MLP that takes $[old\_node, agg\_messages]$ as input and return the new node value

# In[ ]:


class Update_Func_1(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, state_dim):
        super(Update_Func_1, self).__init__()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.hidden_layer_1  = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=state_dim, activation =  tf.nn.relu)

        
    def call(self, old_state, agg_messages):
        concat = self.concat_layer([old_state, agg_messages])
        activation = self.hidden_layer_1(concat)
        return self.output_layer(activation)


# ## Edge update 
# 
# The edge update function is a MLP that takes $concat([node_i, edge_{ij}, node_j])$ as input and produces a new edge value

# In[ ]:


class Adj_Updater_1(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, state_dim):
        super(Adj_Updater_1, self).__init__()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=state_dim, activation = tf.nn.relu)

    def call(self, node_i, node_j, edge_ij):
        concat = self.concat_layer([node_i, node_j, edge_ij])
        activation = self.hidden_layer_1(concat)
        return self.output_layer(activation)


# ## Output layer
# 
# This is where the model diverges with the paper.   
# As the paper predicts bulk properties, but we are interested in edges, we need something different.   
# 
# Here the each edge is passed through a MLP which is used to regress the scalar coupling for each edge

# In[ ]:


class Edge_Regressor(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Edge_Regressor, self).__init__()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.hidden_layer_3 = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=1)#, activation=tf.nn.tanh)

        
    def call(self, edges):
            
        activation_1 = self.hidden_layer_1(edges)
        activation_2 = self.hidden_layer_2(activation_1)
        activation_3 = self.hidden_layer_3(activation_2)

        return self.output_layer(activation_3)


# ## Message passing layer
# 
# Put all of the above together to make a message passing layer which does one round of message passing and node updating

# In[ ]:


class MP_Layer(tf.keras.layers.Layer):
    def __init__(self, mp_int_dim, up_int_dim, out_int_dim, state_dim):
        super(MP_Layer, self).__init__(self)
        self.state_dim = state_dim  
        self.message_passers_1  = Message_Passer_1(intermediate_dim = mp_int_dim, state_dim = state_dim) 
        self.update_functions_1 = Update_Func_1(intermediate_dim = up_int_dim, state_dim = state_dim)
        self.adj_updaters_1     = Adj_Updater_1(intermediate_dim = up_int_dim, state_dim = state_dim)
        self.message_passers_2  = Message_Passer_1(intermediate_dim = mp_int_dim, state_dim = state_dim) 
        self.update_functions_2 = Update_Func_1(intermediate_dim = up_int_dim, state_dim = state_dim)
        self.adj_updaters_2     = Adj_Updater_1(intermediate_dim = up_int_dim, state_dim = state_dim)
        self.message_aggs    = Message_Agg()   
        self.batch_norm_n_1 = tf.keras.layers.BatchNormalization() 
        self.batch_norm_e_1 = tf.keras.layers.BatchNormalization() 
        self.batch_norm_e_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_n_2 = tf.keras.layers.BatchNormalization()

        
    def call(self, nodes_branch_1, edges_branch_1, nodes_branch_2, edges_branch_2, mask):
        
        nodes_1          = nodes_branch_1
        edges_1          = edges_branch_1
        
        n_nodes  = tf.shape(nodes_1)[1]
        node_dim = tf.shape(nodes_1)[2]
        
        state_i_1 = tf.tile(nodes_1, [1, n_nodes, 1])
        state_j_1 = tf.reshape(tf.tile(nodes_1, [1, 1, n_nodes]),[-1,n_nodes*n_nodes, node_dim ])

        new_edges_1 = self.adj_updaters_1(state_i_1, state_j_1, edges_1)
        new_edges_1 = tf.math.multiply(new_edges_1, mask)

        
        messages_1  = self.message_passers_1(state_i_1, state_j_1, new_edges_1)
        #Do this to ignore messages from non-existant nodes
        masked_1 =  tf.math.multiply(messages_1, mask)
        masked_1 = tf.reshape(masked_1, [tf.shape(messages_1)[0], tf.shape(nodes_1)[1], tf.shape(nodes_1)[1], tf.shape(messages_1)[2]])
        agg_m_1 = self.message_aggs(masked_1)
        
        # Update states
        state_1 = self.update_functions_1(nodes_1, agg_m_1)
        
        nodes_2          = nodes_branch_2
        edges_2          = edges_branch_2
        
        n_nodes  = tf.shape(nodes_2)[1]
        node_dim = tf.shape(nodes_2)[2]
        
        state_i_2 = tf.tile(nodes_2, [1, n_nodes, 1])
        state_j_2 = tf.reshape(tf.tile(nodes_2, [1, 1, n_nodes]),[-1,n_nodes*n_nodes, node_dim ])

        new_edges_2 = self.adj_updaters_2(state_i_2, state_j_2, edges_2)
        new_edges_2 = tf.math.multiply(new_edges_2, mask)

        
        messages_2  = self.message_passers_2(state_i_2, state_j_2, new_edges_2)
        #Do this to ignore messages from non-existant nodes
        masked_2 =  tf.math.multiply(messages_2, mask)
        masked_2 = tf.reshape(masked_2, [tf.shape(messages_2)[0], tf.shape(nodes_2)[1], tf.shape(nodes_2)[1], tf.shape(messages_2)[2]])
        agg_m_2 = self.message_aggs(masked_2)
        
        # Update states
        state_2 = self.update_functions_2(nodes_2, agg_m_2)
        
        # Update state 1 with state 2 value
        state_1 = tf.math.multiply(state_1, tf.math.sigmoid(state_2))
      
        # Batch norm and output
        nodes_out_1 = self.batch_norm_n_1(state_1)
        edges_out_1 = self.batch_norm_e_1(new_edges_1)    
        nodes_out_2 = self.batch_norm_n_2(state_2)
        edges_out_2 = self.batch_norm_e_2(new_edges_2)    

        return nodes_out_1, edges_out_1, nodes_out_2, edges_out_2


# ## Define an edge only version of the MPL to do a final edge update. 

# In[ ]:


class MP_Layer_edge_only(tf.keras.layers.Layer):
    def __init__(self, mp_int_dim, up_int_dim, out_int_dim, state_dim):
        super(MP_Layer_edge_only, self).__init__(self)
        self.adj_updaters     = Adj_Updater_1(intermediate_dim = up_int_dim, state_dim = state_dim)
        self.message_aggs    = Message_Agg()
        self.state_dim = state_dim         

        
    def call(self, nodes, edges, mask):
     
        nodes_0          = nodes
        edges_0          = edges
        
        n_nodes  = tf.shape(nodes_0)[1]
        node_dim = tf.shape(nodes_0)[2]
        
        state_i = tf.tile(nodes_0, [1, n_nodes, 1])
        state_j = tf.reshape(tf.tile(nodes_0, [1, 1, n_nodes]),[-1,n_nodes*n_nodes, node_dim ])

        new_edges = self.adj_updaters(state_i, state_j, edges_0)
        new_edges = tf.math.multiply(new_edges, mask)
        
        edges_out = new_edges

        return edges_out


# ## Put it all together to form a MPNN
# 
# Defines the full mpnn that does T message passing steps, where T is a hyperparameter.   
# Here each layer has it's own weights, but weights can be shared across layers. 

# In[ ]:


# Define the MPNN here using the parts defined earlier
adj_input = tf.keras.Input(shape=(None,), name='adj_input')
nod_input = tf.keras.Input(shape=(None,), name='nod_input')
class MPNN(tf.keras.Model):
    def __init__(self, mp_int_dim, up_int_dim, out_int_dim, state_dim, T):
        super(MPNN, self).__init__(self)        
        self.MP = [MP_Layer(mp_int_dim, up_int_dim, out_int_dim, state_dim) for _ in range(T)]        
        self.MP_edge = MP_Layer_edge_only(mp_int_dim, up_int_dim, out_int_dim, state_dim) 
        self.embed_node = tf.keras.layers.Dense(units=state_dim, activation=tf.nn.relu)
        self.embed_edge = tf.keras.layers.Dense(units=state_dim, activation=tf.nn.relu)        
        self.edge_regressor  = Edge_Regressor(mp_int_dim)
        
    def call(self, inputs =  [adj_input, nod_input]):
      
      
        nodes            = inputs['nod_input']
        edges            = inputs['adj_input']
        
#         print(nodes)
        
        edges_0    = edges

        len_edges = tf.shape(edges)[-1]
        
        _, x = tf.split(edges, [len_edges -1, 1], 2)

        mask =  tf.where(tf.equal(x, 0), x, tf.ones_like(x))

        nodes = tf.cast(nodes, dtype=tf.float32)
        edges = tf.cast(edges, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        
        nodes_1 = self.embed_node(nodes) 
        edges_1 = self.embed_edge(edges)
        nodes_2 = nodes_1
        edges_2 = edges_1

        nodes_1_ = nodes_1
        edges_1_ = edges_1
        nodes_2_ = nodes_2
        edges_2_ = edges_2
              
        for i, mp in enumerate(self.MP):
            index = i + 1
            if index%2 == 0:
                nodes_1, edges_1, nodes_2, edges_2 =  mp(nodes_1, edges_1, nodes_2, edges_2, mask)
                nodes_1 = nodes_1 - nodes_1_
                edges_1 = edges_1 - edges_1_
                nodes_2 = nodes_2 - nodes_2_
                edges_2 = edges_2 - edges_2_
                nodes_1_ = nodes_1
                edges_1_ = edges_1
                nodes_2_ = nodes_2
                edges_2_ = edges_2             
            else:
                nodes_1, edges_1, nodes_2, edges_2 =  mp(nodes_1, edges_1, nodes_2, edges_2, mask)
            
        
        edges = self.MP_edge(nodes_1, edges_1, mask)
        
        con_edges = self.edge_regressor(edges_1)
    
        return con_edges


# ## Define the loss functions. 
# 
# (**note**: that for LMAE, as the output values have been scaled down values will be much smaller than for unscaled values)

# In[ ]:


def log_mae(orig , preds):
 
    # Mask values for which no scalar coupling exists
    mask  = tf.where(tf.equal(orig, 0), orig, tf.ones_like(orig))

    nums  = tf.boolean_mask(orig,  mask)
    preds = tf.boolean_mask(preds,  mask)

    reconstruction_error = tf.math.log(tf.reduce_mean(tf.abs(tf.subtract(nums, preds))))

    return reconstruction_error


# ## Define some callbacks, the initial learning rate and the optimizer

# In[ ]:


learning_rate = 0.001
def warmup(epoch):
    initial_lrate = learning_rate   
    if epoch == 0:
        lrate = 0.00001
    if epoch == 1:
        lrate = 0.0001
    if epoch > 1:
        lrate = 0.001   
    if epoch > 20:
        lrate = 0.0001
    if epoch > 25:
        lrate = 0.00001
        
    tf.print("Learning rate: ", lrate)
    return lrate

lrate = tf.keras.callbacks.LearningRateScheduler(warmup)


opt = tf.optimizers.Adam(learning_rate=learning_rate)


# ## Finally create the model, and compile

# In[ ]:


mpnn = MPNN(mp_int_dim = 512, up_int_dim = 1024, out_int_dim = 512, state_dim = 256, T = 3)
#mpnn = MPNN(mp_int_dim = 128, up_int_dim = 128, out_int_dim = 256, state_dim = 64, T = 5)

mpnn.compile(opt, log_mae)
# mpnn.summary()


# Define some hyperparameters

# In[ ]:


batch_size = 64
epochs = 32


# ## Let the learning begin!

# In[ ]:


# Wrap in a function so that memory is freed after calling
def train():
    nodes_train     = np.load(datadir + "internalgraphdata/nodes_train.npz" )['arr_0']
    in_edges_train  = np.load(datadir + "internalgraphdata/in_edges_train.npz")['arr_0']
    out_edges_train = np.load(datadir + "internalgraphdata/out_edges_train.npz" )['arr_0']

    out_labels = out_edges_train.reshape(-1,out_edges_train.shape[1]*out_edges_train.shape[2],1)
    in_edges_train = in_edges_train.reshape(-1,in_edges_train.shape[1]*in_edges_train.shape[2],in_edges_train.shape[3])


    train_size = int(len(out_labels)*0.8)

    mpnn.call({'adj_input' : in_edges_train[:10], 'nod_input': nodes_train[:10]})
    
    mpnn.load_weights(datadir + "dual-mpnn/model.h5")

    history = mpnn.fit({'adj_input' : in_edges_train[:train_size], 'nod_input': nodes_train[:train_size]}, y = out_labels[:train_size], batch_size = batch_size, epochs = epochs, 
             callbacks = [lrate], use_multiprocessing = True, initial_epoch = 0, verbose = 2, 
             validation_data = ({'adj_input' : in_edges_train[train_size:], 'nod_input': nodes_train[train_size:]},out_labels[train_size:]) )
    
    preds = mpnn.predict({'adj_input' : in_edges_train[train_size:], 'nod_input': nodes_train[train_size:]}, verbose = 0)
    
    return preds, train_size, history


# In[ ]:


get_ipython().run_cell_magic('time', '', 'preds, train_size, history = train()')


# In[ ]:



mpnn.save_weights("model.h5")


# ## Show history of training

# In[ ]:


# list all data in history
print(history.history.keys())

# plt.plot(history.history['lr'])
# plt.title('learning rate')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ## Save history of training

# In[ ]:


with open('/trainHistoryDict.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


# ## Predict on val set

# In[ ]:


train = pd.read_csv(datadir + "champs-scalar-coupling/train.csv")
test = pd.read_csv(datadir + "champs-scalar-coupling/test.csv")

train_mol_names = train['molecule_name'].unique()

val = train[train.molecule_name.isin(train_mol_names[train_size:])]
val_group = val.groupby('molecule_name')


# In[ ]:


def make_outs(test_group, preds):
    i = 0
    x = np.array([])
    for test_gp, preds in zip(test_group, preds):
        if (not i%1000):
            print(i)

        gp = test_gp[1]
        
        x = np.append(x, (preds[gp['atom_index_0'].values, gp['atom_index_1'].values] + preds[gp['atom_index_1'].values, gp['atom_index_0'].values])/2.0)
        
        i = i+1
    return x

def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


# In[ ]:


max_size = 29
preds = preds.reshape((-1,max_size, max_size))
out_unscaled = make_outs(val_group, preds)


# In[ ]:


val['pred_scalar_coupling_constant'] = out_unscaled


coups_to_isolate = ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']
for i, coup in enumerate(coups_to_isolate):
    
    
    scale_min = train['scalar_coupling_constant'].loc[train.type == coup].min()
    scale_max = train['scalar_coupling_constant'].loc[train.type == coup].max()
    scale_mid = (scale_max + scale_min)/2
    scale_norm = scale_max - scale_mid

    val.loc[val.type == coup, 'pred_scalar_coupling_constant'] = val['pred_scalar_coupling_constant'].loc[val.type == coup]*scale_norm + scale_mid

    
    val.loc[val.type == coup, 'pred_scalar_coupling_constant'] = val['pred_scalar_coupling_constant'].loc[val.type == coup]


# In[ ]:


for coup in coups_to_isolate:
    log_mae = group_mean_log_mae(val['scalar_coupling_constant'], val['pred_scalar_coupling_constant'], val['type'][val.type == coup])
    print(coup,"\t", log_mae)
    
total = group_mean_log_mae(val['scalar_coupling_constant'], val['pred_scalar_coupling_constant'], val['type'])
print("")
print("Total:","\t", total)


# ## Predict on the test set

# In[ ]:


nodes_test     = np.load(datadir + "internalgraphdata/nodes_test.npz" )['arr_0']
in_edges_test  = np.load(datadir + "internalgraphdata/in_edges_test.npz")['arr_0']
in_edges_test  = in_edges_test.reshape(-1,in_edges_test.shape[1]*in_edges_test.shape[2],in_edges_test.shape[3])


# In[ ]:


preds = mpnn.predict({'adj_input' : in_edges_test, 'nod_input': nodes_test}, verbose=1)


# In[ ]:


np.save("preds_kernel.npy" , preds)


# # Prediction done!
# 
# Now rescale outputs and create submission.csv

# In[ ]:


test_group = test.groupby('molecule_name')


# In[ ]:


preds = preds.reshape((-1,max_size, max_size))
out_unscaled = make_outs(test_group, preds)


# In[ ]:


test['scalar_coupling_constant'] = out_unscaled

coups_to_isolate = ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']
for i, coup in enumerate(coups_to_isolate):
    
    
    scale_min = train['scalar_coupling_constant'].loc[train.type == coup].min()
    scale_max = train['scalar_coupling_constant'].loc[train.type == coup].max()
    scale_mid = (scale_max + scale_min)/2
    scale_norm = scale_max - scale_mid

    test.loc[test.type == coup, 'scalar_coupling_constant'] = test['scalar_coupling_constant'].loc[test.type == coup]*scale_norm + scale_mid

    
    test.loc[test.type == coup, 'pred_scalar_coupling_constant'] = test['scalar_coupling_constant'].loc[test.type == coup]




# In[ ]:


test[['id','scalar_coupling_constant']].to_csv('submission.csv', index=False)


# In[ ]:


test[['id','scalar_coupling_constant']].head()

