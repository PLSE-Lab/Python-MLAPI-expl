#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as netx
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from keras.layers import Dense, Input, Lambda
from keras.layers import Activation, Reshape
from keras.layers import AveragePooling1D
from keras.layers import Dropout,BatchNormalization
from keras.layers import Concatenate, Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from keras.callbacks import EarlyStopping

# specification of a Graph Convolution Layer
class Kipf_GNN(Layer):

    def __init__(self, output_dim, activation, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(Kipf_GNN, self).__init__(**kwargs)

    def build(self, input_shape):
        #assert isinstance(input_shape, list)

        self.kernel = self.add_weight(name='kernel', 
                shape=(input_shape[-1], self.output_dim),
                initializer='he_normal',
                trainable=True)
        self.bias = self.add_weight(name='bias',
                shape=(self.output_dim,),
                initializer='zeros',
                trainable=True)
        super(Kipf_GNN, self).build(input_shape)

    def call(self, x):
        #assert isinstance(x, list)

        #adj = x[0]
        #feat = x[1]
        #h = tf.linalg.matmul(adj, feat)
        return self.activation(K.dot(x, self.kernel) + self.bias)

    def compute_output_shape(self, input_shape):
        #assert isinstance(input_shape, list)

        #shape1 = input_shape[0]
        shape = list(input_shape)
        shape[-1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
                'kernel': self.kernel,
                'bias': self.bias,
                }
        base_config = super(Kipf_GNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
                

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# functions for simulating random isomorphic and non-isomorphic graph instances
def generate_isomorphic(adj):
    I = np.eye(adj.shape[0])
    P = np.random.permutation(I)
    PMat = np.matrix(P)
    del(P)
    del(I)
    return PMat * adj * PMat**-1

def generate_non_isomorphic(adj):
    return np.random.permutation(adj)


# In[3]:


# Generate the reference graph
def generate_reference_graph(vertices):
    I = np.eye(vertices)
    result = np.eye(vertices)
    max_edges = vertices * (vertices - 1);
    edges = np.random.randint(max_edges // 4, 3 * max_edges // 4)
    
    for i in range(edges):
        row = np.random.randint(0, vertices)
        col = np.random.randint(0, vertices)
        result[row][col] = 1
    
    result -= I
    del(I)
    resultMat = np.matrix(result)
    del(result)
    return resultMat

def unzip(lst):  
    return [(lambda tup: list(tup))(tup) for tup in list(zip(*lst))]
            

        


# In[4]:


# Generating the data
vertices = 40
#reference = generate_reference_graph(vertices)
data = []
for i in range(100000):    
    ref1 = generate_reference_graph(vertices)
    ref2 = generate_reference_graph(vertices)
    non_iso = generate_non_isomorphic(ref1)
    iso = generate_isomorphic(ref2)
    data.append([ref1, non_iso, 0])
    data.append([ref2, iso, 1])
    


# In[5]:


# shuffle data and split into training and test datasets
np.random.shuffle(data)

train_data = data[:100000]
test_data = data[100000:]

# split the datasets into xs and ys

train_g1s = [row[0] for row in train_data]
#train_x1s, train_a1s = unzip(train_g1s)
train_g2s = [row[1] for row in train_data]
#train_x2s, train_a2s = unzip(train_g2s)
train_ys = [row[2] for row in train_data]

test_g1s = [row[0] for row in test_data]
#test_x1s, test_a1s = unzip(test_g1s)
test_g2s = [row[1] for row in test_data]
#test_x2s, test_a2s = unzip(test_g2s)
test_ys = [row[2] for row in test_data]


# In[6]:


def average_flat_tensors(tensors):
    x1 = tensors[0]
    x2 = tensors[1]
    return x1 + x2 / 2

def average_flat_tensors_shape(input_shapes):
    return input_shapes[0]



# define the model architecture
X1 = Input(shape=(vertices, vertices))
#A1 = Input(shape=(vertices, vertices))
X2 = Input(shape=(vertices, vertices))
#A2 = Input(shape=(vertices, vertices))

identity = lambda x: x

conv1 = Kipf_GNN(output_dim=vertices, activation=identity)
bn1 = BatchNormalization()
act1 = Activation('relu')
conv2 = Kipf_GNN(output_dim=2*vertices, activation=identity)
bn2 = BatchNormalization()
act2 = Activation('relu')
concat = Concatenate(axis=1)
lam = Lambda(average_flat_tensors, average_flat_tensors_shape)
flat = Flatten()
d1 = Dense(2 * vertices, activation='relu')
drop1 = Dropout(0.5)
d2 = Dense(16, activation='relu')
drop2 = Dropout(0.5)
out = Dense(1, activation='sigmoid')
reshape = Reshape(target_shape=(1, ))

H11 = act1(bn1(conv1(X1)))
H12 = act1(bn1(conv1(X2)))
H21 = act2(bn2(conv2(H11)))
H22 = act2(bn2(conv2(H12)))
H31 = flat(H21)
H32 = flat(H22)
#H4 = concat([H31, H32])
H4 = lam([H31, H32])
H41 = drop1(d1(H31))
H42 = drop1(d1(H32))
H51 = drop2(d2(H41))
H52 = drop2(d2(H42))
H6 = lam([H51, H52])
Y = reshape(out(H6))

# create the model, summarize, and compile
model = Model(inputs=[X1, X2], outputs=Y)
model.trainable = True
model.summary()
model.compile(
        loss='binary_crossentropy', 
        metrics=['accuracy'], 
        optimizer=Adam(lr=0.0005)
)


# In[7]:


# early stopping callback
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
# fit the model to the training data
model.fit(
    x=[train_g1s, train_g2s], 
    y=train_ys, validation_split=0.2, 
    verbose=1, batch_size=100, epochs=20,
    callbacks=[es]
)

plot_model(model, to_file='model.png')


# In[8]:


metrics = model.evaluate(x=[test_g1s, test_g2s], y=test_ys, verbose=1)
print(metrics)


# In[9]:


print(len([y for y in test_ys if y == 1]))
print(len([y for y in train_ys if y == 1]))


# In[10]:


sizes = [np.sum(adj) for adj in train_g1s + train_g2s + test_g1s + test_g2s]
sizes[:20]

