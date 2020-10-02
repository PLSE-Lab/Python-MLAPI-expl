#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# baseline cnn model for mnist
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import seaborn as sns


# In[ ]:


from keras.datasets import mnist
from matplotlib import pyplot
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))


# In[ ]:


#import the data into the system
df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


df_test.shape


# In[ ]:


df_test.columns


# In[ ]:


df_train.columns


# In[ ]:


def load_dataset():
    d = df_train.columns[1:]
    trainX = df_train[d].values
    trainY =  df_train['label'].values
    testX = df_test[d].values
    #testY =  df_test['label'].values
    trainX = trainX.reshape(trainX.shape[0], 28, 28,1)
    testX = testX.reshape(testX.shape[0], 28, 28,1)
    trainY = to_categorical(trainY)
    #testY = to_categorical(testY)
    return trainX, trainY,testX


# In[ ]:


def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm


# In[ ]:


def define_model():
    #create the neural network model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
    # define model
        model = define_model()
        # select rows for train and test
        train_X, train_Y, test_X, test_Y = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(train_X, train_Y, epochs=10, batch_size=32, validation_data=(test_X, test_Y), verbose=0)
        # evaluate model
        _, acc = model.evaluate(test_X, test_Y, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        model.save('final_model.h5')
 
        scores.append(acc)
        histories.append(history)
        return scores, histories


# In[ ]:


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')


# In[ ]:


def run_test_harness():
    #load the dataset
    trainX, trainY, testX = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    #summarize_performance(scores)

# entry point, run the test harness
run_test_harness()


# In[ ]:


trainX.shape


# In[ ]:


testX.shape


# In[ ]:


testX = testX.reshape(testX.shape[0], 28, 28,1)


# In[ ]:


from keras import models    
model = models.load_model('final_model.h5')

pred_y = np.argmax((model.predict(testX)),axis=1)


# In[ ]:


pred_y


# In[ ]:


d = list()
pred = [[d.append([p]) for i in range(27)] for p in pred_y]


# In[ ]:


os.chdir('/kaggle/')


# In[ ]:


cd working


# In[ ]:


pred = pd.DataFrame(d,columns = ['label']) 


# In[ ]:


pred['ImageID'] = pred.index + 1


# In[ ]:


pred = pred[['ImageID','label']]


# In[ ]:


pred


# In[ ]:


pred.to_csv('r.csv', index=False, header=True)


# In[ ]:


pred.shape


# In[ ]:




