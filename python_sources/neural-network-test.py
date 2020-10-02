import pandas as pd
import numpy as np

# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("../input/train.csv") #reading the training file
labels = dataset[[0]].values.ravel() #reading labels from training dataset 
pixels = dataset.iloc[:,1:].values #skipping the header row
test = pd.read_csv("../input/test.csv").values #reading the test file

#target=labels
#train=pixels
#reshaping :

labels = labels.astype(np.uint8)
pixels = np.array(pixels).reshape((-1, 1, 28, 28)).astype(np.uint8)
test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

#creating class net1

net1 = NeuralNet(
        layers=[('input', layers.InputLayer), #defining layer 
                ('hidden', layers.DenseLayer), #defining layer
                ('output', layers.DenseLayer), #defining layer
                ],
        # layer parameters :
        input_shape=(None,1,28,28),
        hidden_num_units=1000, # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method :
        update = nesterov_momentum,
        update_learning_rate = 0.0001,
        update_momentum = 0.9,

        max_epochs = 15,
        verbose = 1,
        )
#training the network :
net2 = net1.fit(pixels, labels)
#using NN to classify the data
pred = net2.predict(test)

#saving results

np.savetxt('submission_NN.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageID,Label', comments = '', fmt='%d')
