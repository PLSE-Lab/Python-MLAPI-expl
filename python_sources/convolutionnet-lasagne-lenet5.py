#######
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:15:32 2016

@author: jm.espin@biometricvox.com
"""

# Generate output files with write_csv(), plot() or ggplot()
# Any files you write to the current directory get shown as outputs
import os
os.environ["THEANO_FLAGS"] = "base_compiledir=kaggle/compiledir_Linux-3.19--generic-x86_64-with-debian-8.5--3.5.2-64/"
import theano

import pandas as pd
import numpy as np

# Packages imported
from lasagne import layers
from lasagne.nonlinearities import  softmax, rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.metrics import accuracy_score


np.random.seed(seed=123)
tipe = 'random' # 'separate', 'random', 'random_label'
factorDivision = 0.3

data = pd.read_csv('../input/train.csv')
data_y = data.ix[:,0].values.astype('int32')
# Divide pixel brightnesses by max brightness so they are between 0 and 1
# This helps the network optimizer make changes on the right order of magnitude
pixel_brightness_scaling_factor = float(data.max().max())
data_x = (data.ix[:,1:].values/pixel_brightness_scaling_factor).astype('float32')


# Reshape the array of pixels so a convolutional neural network knows where
# each pixel belongs in the image
image_width = image_height = int(data_x.shape[1] ** .5)
data_x_reshaped = data_x.reshape(data_x.shape[0], 1, image_height, image_width)

size = data_x_reshaped.shape[0] # Number of samples

if tipe=='separate':
    idx_val = np.array(range(size))< int(size*factorDivision) # This is for get the first "size_val, 30%" to validation
    
if tipe == 'random':
    idx_val = np.random.randn(size)<factorDivision# This get random samples to validation, no see the label

if tipe=='random_label':
    # Make the random separation but seeing the label. 
    train_x_reshaped = np.array([]).reshape(0,1,28,28)
    vali_x_reshaped = np.array([]).reshape(0,1,28,28)
    train_y = np.array([])
    vali_y = np.array([])
    for i in range(10):
        idx_label = data_y== i 
        data_x_reshaped_label = data_x_reshaped[idx_label,]
        data_y_label = data_y[idx_label,]
        size_label = data_x_reshaped_label.shape[0] # Number of samples
        idx_val_label = np.random.random_sample(size_label)<factorDivision# This get random samples to validation, no see the label
        train_x_reshaped_label = data_x_reshaped_label[~idx_val_label,]
        vali_x_reshaped_label = data_x_reshaped_label[idx_val_label,]
        train_y_label = data_y_label[~idx_val_label]
        vali_y_label = data_y_label[idx_val_label]
        
        train_x_reshaped = np.concatenate((train_x_reshaped, train_x_reshaped_label),axis=0) 
        vali_x_reshaped = np.concatenate((vali_x_reshaped, vali_x_reshaped_label),axis=0)
        train_y = np.concatenate((train_y,train_y_label))
        vali_y = np.concatenate((vali_y, vali_y_label))
    train_y = train_y.astype('int32')    
    vali_y = vali_y.astype('int32')
    del data_x_reshaped_label, data_y_label, size_label, idx_val_label, train_x_reshaped_label, vali_x_reshaped_label, train_y_label, vali_y_label
else:
    # Divide the data to the train and validation group
    train_x_reshaped = data_x_reshaped[~idx_val,]
    vali_x_reshaped = data_x_reshaped[idx_val,]

    # Divide the label as same as the data
    train_y = data_y[~idx_val]
    vali_y = data_y[idx_val]
    

# Print Info about the separation of the data.

print ("\n \n Separation of de data")
print (" \t Original number of samples \t %d " % data_y.shape[0])
print (" \t Number of samples to train \t %d " % train_y.shape[0])
print (" \t Number of samples to validate \t %d " % vali_y.shape[0])

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": range(1,len(preds)+1), "Label": preds}).to_csv(fname, index=False, header=True)

#runfile('C:/JuanManuel/Kaggle_MNIST/code/prepareData_train_simple.py', wdir='C:/JuanManuel/Kaggle_MNIST/code')


numberHidden_opt = 150
size_output = 10
learning_rate = 0.05
momentum = 0.7
number_max_epoch = 20
print (" with the configuration %d neuron in the hidden layer " % numberHidden_opt)



print (" Train the optimal Convolutional Nnet ")
net_opt = NeuralNet(
            layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),      #Convolutional layer.  Params defined below
            ('pool1', layers.MaxPool2DLayer),   #Like downsampling, for execution speed
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('hidden3', layers.DenseLayer),
            ('hidden4', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
            
    
        input_shape=(None, 1, image_width, image_height),
        conv1_num_filters=6, conv1_filter_size=(5, 5), conv1_nonlinearity=rectify,
        pool1_pool_size=(2, 2),
        conv2_num_filters=16, conv2_filter_size=(5, 5), conv2_nonlinearity=rectify,
        pool2_pool_size=(2, 2),
        hidden3_num_units=numberHidden_opt,
        hidden4_num_units=84,
        output_num_units=10, output_nonlinearity=softmax,
    
        update_learning_rate=learning_rate,
        update_momentum=momentum,
    
        regression=False,
        max_epochs=number_max_epoch,
        verbose=1
        )

net_opt.fit(data_x_reshaped,data_y)
#Read de data test and divide by the max value of brightness, 255

test_x = (pd.read_csv('../input/test.csv').values/pixel_brightness_scaling_factor).astype('float32')
test_x_reshaped = test_x.reshape(test_x.shape[0], 1, image_height, image_width)

pred = net_opt.predict(test_x_reshaped)
write_preds(pred, "convolutional_lasagne_lenet5.csv")

