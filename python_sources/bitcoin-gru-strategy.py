#!/usr/bin/env python
# coding: utf-8

# Here is the model for RNN-based Bitcoin price prediction
# The main steps are the next:
# 1. Loading and preprocessing the historical data
# make training/validation split
# Train at first 2/3 of the samples, validate on the last. Absence of randomization during split is to meet the reality (we can train only on the past)
# 2. Build generator for the data tot train RNN
# t is the current time, T is the timeshift
#     Build inputs as a sequences of length L of the Open prices relative change compared to time -T
#     Build outputs as price change at moment +T in future compared to the t
# 3, Build RNN
#     First layer is 1D convolutional layer  enhances the learning rate and allow to  find simple dependences 
#     Second layer is GRU for detection of more complicated dependences (can be used CuDNNGRU for NVIDIA GPUs)
#     Couple of dense layers
#     Output layer with 3 units and Softmax activation:
#         Output 1 - bet to sell
#         Output 2-  bet to buy
#         Output 3-  bet to do nothing
#     Batch normalization layers speed up the training
#     Loss function is constructed to maximaze the value of (gain) during trading

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
import keras.backend as K
import random
from keras import regularizers
from keras.layers import Reshape, multiply ,Activation, Conv2D, Multiply, Input, MaxPooling2D, SpatialDropout2D,BatchNormalization, Flatten, Dense, Lambda, Dropout, LSTM,CuDNNGRU,CuDNNLSTM,GRU,Conv1D
from keras.optimizers import SGD, Adam, RMSprop
import os
print(os.listdir("../input/"))


# In[ ]:


data=pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv', delimiter=',')


# Check the data

# In[ ]:


data.head(5)


# fill the NaN values

# In[ ]:


data[data.columns.values] = data[data.columns.values].ffill()
data[data.columns.values]=data[data.columns.values].fillna(0)


# Build the function for the Data calculation. (+gain and -gain is the income if we by or sell BTC - model that can be easilly improved by looking at High and Low prices for +-gain instead of Close  in order to take in account the spreads in case of closing the deals "by Market" ) +-gains are in percentes related to Weighted_Price

# In[ ]:


def preparedata(data, periodtochange=10, periodtopredict=100):
    data['change']=(data['Weighted_Price']-data['Weighted_Price'].shift(periodtochange))/data['Weighted_Price']
    data['+gain']=(data['Close'].shift(-periodtopredict)-data['Close'])/data['Weighted_Price']*100
    data['-gain']=(-data['Close'].shift(-periodtopredict)+data['Close'])/data['Weighted_Price']*100
    return data


# remove the Nan we get after calculations with shift

# In[ ]:


data=preparedata(data)
data[data.columns.values]=data[data.columns.values].fillna(0)


# make training/val split

# In[ ]:


datatrain=data[:1990342]
dataval=data[1990342:-300]


# Create generator in order to save the operative memory during training. 
# Here we got the randomisation inside the train and validation data.
# Every labels have 3 dimensions - 1st - gain in case of 'Buy'  at this moment, second in case of 'Sell', last equals to 0 gain in case of 'Do Nothing'

# In[ ]:


def data_generator(df, batchsize=10240, length=200):
    xsez=np.array(df['change'])
    ysez=np.zeros((len(df),3))
    ysez[:,:2]=np.array(df[['+gain','-gain']])
    sequences=np.zeros((batchsize, length, 1))
    answer=np.zeros((batchsize, 3))
    j=1
    k=0
    while True:
        k=random.randint(1,xsez.shape[0]-length-2)
        seq=xsez[k:k+length]
        ans=ysez[k+length, :]
        sequences[j%batchsize,:,0]=seq
        answer[j%batchsize,:]=ans
        j=j+1
        if j>len(df)-length-2: j=0
        if j%batchsize==0:
            yield sequences, answer


# Create one  batch from validation data 

# In[ ]:


genvalregr=data_generator(dataval, batchsize=10240, length=200)
toval=next(genvalregr)


# Creation of the Keras Loss as the mean gain per one decicion   

# In[ ]:


def Gain_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    toret = -K.mean(y_true_f * y_pred_f)
    return toret


#  L1 and L2 regularization values for the RNN are necessary in order not to stuck in always 'do nothing' and avoid gradients explosions. L2 regularization used in this version

# In[ ]:


biasl1=0
kernell1=0
biasl2=0.01
kernell2=0.01


# Build the model. Here different activations, length of convolutional core and number of units/layers can be used

# In[ ]:


model = Sequential()
model.add(Conv1D(16, 32, activation='sigmoid', input_shape=(None, 1), kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2)))
model.add(BatchNormalization())
model.add(GRU(16, kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2)))
model.add(BatchNormalization())
model.add(Dense(32,  activation='sigmoid',kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2)))
model.add(BatchNormalization())
model.add(Dense(32,  activation='sigmoid',kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2)))
model.add(BatchNormalization())
model.add(Dense(3,  activation='softmax',kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2)))
model.compile(loss=Gain_loss, optimizer=Adam(lr=0.001))


# Check the number of parameters is not huge to avoid the overfitting

# In[ ]:


model.summary()


# Learning rate can be used as other hyperparameter as well as learning rate change during training is also can be played with, for example

# In[ ]:


from keras.callbacks import LearningRateScheduler
def annealing(x):
    initial_lrate = 0.001
    return initial_lrate/(x+1)*np.random.rand()
lrate = [LearningRateScheduler(annealing)]


# In[ ]:


K.set_value(model.optimizer.lr, 0.0001)
hist=model.fit_generator(data_generator(datatrain, batchsize=1024, length=200),
    samples_per_epoch = 10, 
    epochs = 10,
    validation_data=(toval[0], toval[1]),
    verbose=1,
    callbacks=lrate,
    )


# Negative loss (-0.0001)  on validation set  can be reached during training with some types of activation/regularization parameters which means that this model potentially can be used for trading.  
# Anyway in case of  loss>0 on validation, if it is stable, and there is no L1/L2 regularization, you can just do the opposite of the model output and still use it for decision   (if you are not taking in account spreads during +-gain calculations) o_O
