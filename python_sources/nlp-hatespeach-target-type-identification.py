#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# tensorflow, keras, scikit
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras import backend as K
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
import sklearn.model_selection as sk
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# helper imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# hyperparamter gird space options
optimizer = ['Adam', 'SGD']
activation = ['relu', 'sigmoid']
batch_size = [50, 100] 
epochs = [5, 10]
neurons = [64,128,284]

# hyperparamter gird space
hyperparamter_space = dict(
    batch_size=batch_size, 
    epochs=epochs,
    optimizer=optimizer,
    activation=activation,
    neurons=neurons
)

# f1_score calculation function for model evaluation
def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

# reading csv    
def read_csv(filepath, has_header=None):
    # data import
    data = pd.read_csv(filepath)
    return data

# model creation functoin
def create_model(f1_score=f1_score,neurons=128,activation='relu',optimizer='sgd'):

    # model init; sequential MLP 
    model = Sequential()

    # layer1: input layer, with kernel initiazer function same as class distriubtion function        
    model.add(
        Dense(256, activation=activation, kernel_initializer = 'normal')
    )

    # layer2: dropout for preventing overfitting 
    model.add(Dropout(0.5))

    # layer3: activation        
    model.add(
        Dense(128, activation=activation)
    )

    # layer4: dropout
    model.add(Dropout(0.2))        

    # layer5: output with softmax        
    model.add(
        Dense(5, activation=tf.nn.softmax)
    )

    # print model summary 
    print(model.summary())

    # compile model
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy',
        metrics=[f1_score, 'accuracy'] # f1 score
    )

    return model

# fit model
def fit_model(model, x_train, y_train, epochs=10, verbose=1):

    # fit model and retrieve history 
    history = model.fit(x_train, y_train, epochs=epochs,verbose=verbose)

    # Plot training & validation f1 values
    plt.plot(history.history['f1_score'])
    plt.title('Model F1')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    return history

# evaluate model
def evaluate_model(model, x_test, y_test):

    test_loss, test_f1, test_acc = model.evaluate(x_test, y_test)

    print('Test F1 Score:', test_f1)

    return test_loss, test_f1, test_acc

# split for cross validation     
def split_by_test_size(x, y, test_size=0.33):

    # split 1: train, test
    x_train, x_test, y_train, y_test = sk.train_test_split(
        x,
        y,
        test_size=test_size
    )

    return x_train, x_test, y_train, y_test


# In[ ]:


dataset = read_csv('../input/train.csv', True)
dataset.head()


# In[ ]:



# one-hot encode labels
def one_hot_encode_by_label(dataset, new_col, col_list, threshold_val=0.1):
    
    for col in col_list:
        dataset[new_col] = (
            dataset[col] > threshold_val
        )*1.0  
    
    return dataset

#
religions = [
    'atheist',
    'buddhist',
    'christian',
    'hindu',
    'jewish',
    'muslim',
    'other_religion'
]
dataset = one_hot_encode_by_label(dataset, 'religion', religions)

#
ethnicity = [
    'asian',
    'black',
    'latino',
    'white',
    'other_race_or_ethnicity'
]
dataset = one_hot_encode_by_label(dataset, 'ethnicity', ethnicity)

# 
sexualOrientation = [
    'bisexual',
    'heterosexual',
    'homosexual_gay_or_lesbian',
    'other_gender',
    'transgender',
    'other_sexual_orientation'    
]
dataset = one_hot_encode_by_label(dataset, 'sexualOrientation', sexualOrientation)

# dataset.hist(column="religion")
# dataset.hist(column="ethnicity")
# dataset.hist(column="sexualOrientation")

# dataset = (
#     dataset['religion'] == 0 and 
#     dataset['ethnicity'] == 0 and
#     dataset['sexualOrientation'] == 0
# )
dataset[
    dataset['religion'] == 1
].tail()
dataset[
    dataset['ethnicity'] == 1
].tail()

# # 1804874
# # len(dataset)


# In[ ]:


# model creation functoin
def create_model(f1_score=f1_score,neurons=128,activation='relu',optimizer='sgd'):

    # model init; sequential MLP 
    model = Sequential()

    # layer1: input layer, with kernel initiazer function same as class distriubtion function        
    model.add(
        Dense(256, activation=activation, input_shape=(284,), kernel_initializer = 'normal')
    )

    # layer2: dropout for preventing overfitting 
    model.add(Dropout(0.5))

    # layer3: activation        
    model.add(
        Dense(128, activation=activation)
    )

    # layer4: dropout
    model.add(Dropout(0.2))        

    # layer5: output with softmax        
    model.add(
        Dense(5, activation=tf.nn.softmax)
    )

    # print model summary 
    print(model.summary())

    # compile model
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy',
        metrics=[f1_score, 'accuracy'] # f1 score
    )

    return model

# fit model
def fit_model(model, x_train, y_train, epochs=10, verbose=1):

    # fit model and retrieve history 
    history = model.fit(x_train, y_train, epochs=epochs,verbose=verbose)

    # Plot training & validation f1 values
    plt.plot(history.history['f1_score'])
    plt.title('Model F1')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    return history

# evaluate model
def evaluate_model(model, x_test, y_test):

    test_loss, test_f1, test_acc = model.evaluate(x_test, y_test)

    print('Test F1 Score:', test_f1)

    return test_loss, test_f1, test_acc


# In[ ]:


dataset.head()

features = dataset['comment_text']
labels = dataset[['religion','ethnicity','sexualOrientation']]

x_train, x_test, y_train, y_test = split_by_test_size(
    features, 
    labels
)

model = create_model()
model = fit_model(model,x_train,y_train)

