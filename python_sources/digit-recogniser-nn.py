import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train_data = pd.read_csv("../input/train.csv")
test_data  = pd.read_csv("../input/test.csv")


import numpy as np
import pandas as pd
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

def get_digit(digit):
    return train_data[train_data['label'] == digit]

def show_digit(data,n):
    plt.title(data.iloc[n,0], fontsize = 40)
    plt.imshow(np.reshape(1.-data.iloc[n,1:].values,(28,28))/255., cmap = mpl.cm.Greys_r)
    plt.show()

def plot_digit(array,shape,plot_name):
    plot_name.imshow(np.reshape(array,shape),cmap = mpl.cm.Greys_r)

def plot_digit_binary(array,shape,plot_name):
    plot_name.imshow(np.reshape(array,shape),cmap = mpl.cm.binary)


def draw_hundred_zeroes():
        
    number_x = 10
    number_y = 10
    fig = plt.figure()
    gridspec1 = gridspec.GridSpec(10,10)
    gridspec1.update(wspace = 0.0,hspace = 0.0)

    for plt_i in range(100):
        f = plt.subplot(gridspec1[plt_i])
        plot_digit(1. - get_digit(0).iloc[plt_i,1:].values/255.,(28,28),f)
        f.set_xticklabels([])
        f.set_yticklabels([])
        plt.axis('off')
        f.set_aspect('equal')
        fig.subplots_adjust(hspace=0.0,wspace=0.0)
        plt.show()
'''
    for xlabel in f.axes.get_xticklabels():
        xlabel.set_visible(False)
        xlabel.set_fontsize(0.0)
    for xlabel in f.axes.get_yticklabels():
        xlabel.set_fontsize(0.0)
        xlabel.set_visible(False)
    for tick in f.axes.get_xticklines():
        tick.set_visible(False)
    for tick in f.axes.get_yticklines():
        tick.set_visible(False)
'''

def get_average_digit():
    average_digit = [0]*10
    for j in range(10):
        average_digit[j] = [0]*(28**2)
        for i in range(len(get_digit(j))):
            average_digit[j] += 1-get_digit(j).iloc[i,1:].values/255.
    return average_digit

def draw_averages():
    number_x =5
    number_y = 2
    fig = plt.figure()
    gridspec1 = gridspec.GridSpec(number_y,number_x)
    gridspec1.update(wspace = 0.0,hspace = 0.0)
    for plt_i in range(10):
        f = plt.subplot(gridspec1[plt_i])
        plot_digit(get_average_digit()[plt_i],(28,28),f)
        f.set_xticklabels([])
        f.set_yticklabels([])
        plt.axis('off')
        f.set_aspect('equal')
        fig.subplots_adjust(hspace=0.0,wspace=0.0)
    plt.show()

def get_rand_digit():
    rand= np.random.randint(42000)
    plot_digit(255-train[rand],(28,28),plt)
    plt.title(target[rand],fontsize=36)
    plt.show()
    
    
    
net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None,1,28,28),
        hidden_num_units=1000, # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=15,
        verbose=1,
        )

# let's train the network
net1.fit(train, target)

pred = net1.predict(test)
np.savetxt('Basic_NN_Guess.csv', np.c_[range(1,len(test)+1),pred], 
           delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')