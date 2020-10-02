import pandas as pd
import keras.callbacks as cb
from keras.layers.core import Activation, Dense, Dropout

from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

import numpy as np
import time

def PreprocessDataset():
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    train = pd.read_csv("../input/train.csv")
    test  = pd.read_csv("../input/test.csv")

    # Write to the log:
    print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
    print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
    # Any files you write to the current directory get shown as outputs
    
    if type(train) is pd.core.frame.DataFrame:
        train = train.as_matrix()


    if type(test) is pd.core.frame.DataFrame:
        test = test.as_matrix()

    x_train = train[:,1:]
    y_train = train[:,:1]

    x_test = test[:,1:]
    y_test = test[:,:1]
    
    ## Transform labels to one-hot encoding
    ## i.e., from '7' to [0,0,0,0,0,0,0,1,0,0]
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    ## Process features. Set numeric type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Activity 1 (Pre-processing):
    # Group A: w/o pre-processing datasets.
    #
    # Group B: Min-Max Normalize value to [0, 1]
    # x_train /= 255
    # x_test /= 255
    #
    # Group C: proceed w/ standardizing datasets by z-scoring (de-mean, uni-variance).
    # x_train = preprocessing.scale(x_train)
    # x_test = preprocessing.scale(x_test)
    ################################################################  
    ## YOUR TURN: CHANGE HERE
    x_train /= 255
    x_test /= 255
    return x_train, x_test, y_train, y_test
    
    
x_train, x_test, y_train, y_test = PreprocessDataset()
print("x_train type: " + str(x_train.shape))
print("x_test type: " + str(x_test.shape))
print("y_train type: " + str(y_train.shape))
print("y_test type: " + str(y_test.shape))

def DefineModel():

    ################################################################
    # Activity 2 (Network Structure):
    # Group A: uses only 1 layer
    # second_layer_width = 0
    #
    # Group B: uses 2 layers of a tower-shaped (same width) network.
    # second_layer_width = 128
    #
    # Group C: uses 2 layers of a pyramid-shaped (shrink width) network.
    # second_layer_width = 64
    ################################################################
    first_layer_width = 128
    second_layer_width = 64   
    
    ################################################################
    # Activity 3 (Activation Function):
    # Group A uses ReLU.
    # activation_func = 'relu' 
    # 
    # Group B uses Sigmoid.
    # activation_func = 'sigmoid'
    #
    # Group C uses Tanh.
    # activation_func = 'tanh'
    ################################################################
    activation_func = 'relu' 

    ################################################################    
    # Activity 4 (Loss Function):
    # Group A uses cross entropy.
    # loss_function = 'categorical_crossentropy'
    # 
    # Group B uses cross entropy.
    # loss_function = 'categorical_crossentropy'
    # 
    # Group C uses squared error.
    # loss_function = 'mean_squared_error'
    ################################################################    
    loss_function = 'categorical_crossentropy'
    
    #################################################################    
    # Activity 5 (Dropout):
    # Group A uses 0% dropout.
    #
    # Group B uses 50% dropout.
    # dropout_rate = 0.5
    #
    # Group C uses 90% dropout.
    # dropout_rate = 0.9
    #################################################################    
    dropout_rate = 0
    
    ################################################################    
    # Activity 6 (Regularization):
    # Group A uses L1 regularizer
    # weight_regularizer = l1(0.01)
    #
    # Group B uses L2 regularizer
    # weight_regularizer = l2(0.01)
    # 
    # Group C uses no regularizer
    # weight_regularizer = None
    ################################################################
    weight_regularizer = None

    ################################################################    
    # Activity 8 (Learning Rate):
    # Group A uses learning rate of 0.1.
    # learning_rate = 0.1
    # 
    # Group B uses learning rate of 0.01.
    # learning_rate = 0.01
    #
    # Group C uses learning rate of 0.5.    
    # learning_rate = 0.5
    ################################################################
    learning_rate = 0.1
    
    ## Initialize model.
    model = Sequential()

    ## First hidden layer with 'first_layer_width' neurons. 
    ## Also need to specify input dimension.
    ## 'Dense' means fully-connected.
    model.add(Dense(first_layer_width, input_dim=784, W_regularizer=weight_regularizer))
    model.add(Activation(activation_func))
    if dropout_rate > 0:
        model.add(Dropout(0.5))

    ## Second hidden layer.
    if second_layer_width > 0:
        model.add(Dense(second_layer_width))
        model.add(Activation(activation_func))
        if dropout_rate > 0:
            model.add(Dropout(0.5))         
    
    ## Last layer has the same dimension as the number of classes
    ## For classification, the activation is softmax
    model.add(Dense(10, activation='softmax'))
    ## Define optimizer. In this tutorial/codelab, we select SGD.
    ## You can also use other methods, e.g., opt = RMSprop()
    opt = SGD(lr=learning_rate, clipnorm=5.)
    ## Define loss function = 'categorical_crossentropy' or 'mean_squared_error'
    model.compile(loss=loss_function, optimizer=opt, metrics=["accuracy"])

    return model

def TrainModel(data=None, epochs=20):
    ################################################################
    # Activity 7 (Mini-batch):
    # Group A uses mini-batch of size 128.
    # batch = 128
    #
    # Group B uses mini-batch of size 256.
    # batch = 256
    # 
    # Group C uses mini-batch of size 512.
    # batch = 512
    ################################################################
    batch=128
    start_time = time.time()
    model = DefineModel()
    if data is None:
        print("Must provide data.")
        return
    x_train, x_test, y_train, y_test = data
    print('Start training.')
    ## Use the first 55,000 (out of 60,000) samples to train, last 5,500 samples to validate.
    history = model.fit(x_train[:55000], y_train[:55000], nb_epoch=epochs, batch_size=batch,
              validation_data=(x_train[55000:], y_train[55000:]))
    print("Training took {0} seconds.".format(time.time() - start_time))
    return model, history

trained_model, training_history = TrainModel(data=[x_train, x_test, y_train, y_test])