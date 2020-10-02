import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train_data = pd.read_csv("../input/train.csv")
test_data  = pd.read_csv("../input/test.csv")


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
import time
import os
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize



target = train_data.iloc[:,0].values.ravel().astype(np.uint8)
train = np.array(train_data.iloc[:,1:].values).reshape((-1,1,28,28)).astype(np.uint8)
test = np.array(test_data.iloc[:,:].values).reshape((-1,1,28,28)).astype(np.uint8)

#get only digits of a specific target


def CNN(n_epochs):
    net1 = NeuralNet(
        layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),      #Convolutional layer.  Params defined below
        ('pool1', layers.MaxPool2DLayer),   # Like downsampling, for execution speed
        ('conv2', layers.Conv2DLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 1, 28, 28),
    conv1_num_filters=7, 
    conv1_filter_size=(3, 3), 
    conv1_nonlinearity=lasagne.nonlinearities.rectify,
        
    pool1_pool_size=(2, 2),
        
    conv2_num_filters=12, 
    conv2_filter_size=(2, 2),    
    conv2_nonlinearity=lasagne.nonlinearities.rectify,
        
    hidden3_num_units=200,
    output_num_units=10, 
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update_learning_rate=0.005,
    update_momentum=0.9,

    max_epochs=n_epochs,
    verbose=1,
    )
    return net1

cnn = CNN(10).fit(train,target) # train the CNN model for 15 epochs


# use the NN model to classify test data
pred = cnn.predict(test)

# save results
np.savetxt('convolutional_nn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')