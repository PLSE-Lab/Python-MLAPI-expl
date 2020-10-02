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


# ### What is Machine Learning
# #### Machine Learning is the extraction of knowledge from data. This gained knowledge can be used to:
# - predict the stock market (or at least attempt)
# - provide a recommendation to you or someone on an item (such as Amazon on purchases)
# - teach a computer what a cat looks like 
# - teach a computer language and sentence structure 
# 
# <img src="https://media.geeksforgeeks.org/wp-content/cdn-uploads/machineLearning3.png" width="400">
# 
# ### This kernel gives a very brief overview of machine learning, its various divisions and a few simple examples for you to expland on

# ### But first, lets go through some knowledge and definitions
# 
# This kernel doesn't cover all the topics involved with machine learning but it's nice to have a good idea of how 
# much this field, artificial intelligence and others apply to various industries and applications
# 
# <img src="https://storage.googleapis.com/cdn.thenewstack.io/media/2018/07/53726660-marconi.png" width=600>
# 
# 

# ### Neural Networks 
# 
# A neural network is a supervised learning model that infers on patterns within the data without needing feature enginnering methods. The neural netowrk, commonly shortened as NN, is made up of perceptrons that perform calculations on the input weights and feed their calculations into another layer of perceptrons
# 
# <img src="https://miro.medium.com/max/2870/1*n6sJ4yZQzwKL9wnF5wnVNg.png" width=600>
# 
# The critical piece to learn when jumping into neural networks is to understand what a perceptron is and how it works. The following example was referrenced from, https://medium.com/@Job_Collins/a-simple-quick-guide-in-deep-learning-with-tensorflow-and-python-1a801e910cf
# 
# In this tutorial, Job does a great job (no pun intended) of explainging the magic behind a perceptron with an example we all can understand, conversion.

# ### Creating the Fahrenheit Data
# We'll use numpy to create some random data to simulate various fahrenheit degrees

# ### F = C * 1.8 + 32

# In[ ]:


fahrenheit = np.random.random((100, 1)) * 100
celcius = (fahrenheit - 32) / 1.8


# In[ ]:


print ("Fahrenheit: {}".format(fahrenheit[0]))
print ("Celcius: {}".format(celcius[0]))


# In[ ]:


#Lets split this data up into a train and test set
train_split = 0.7

train_f = fahrenheit[0:int(len(fahrenheit)*train_split)]
train_c = celcius[0:int(len(celcius)*train_split)]

test_f = fahrenheit[len(train_f):]
test_c = celcius[len(train_c):]


# ### Use the Keras library to build the Neural Network (single perceptron)

# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt


# ### Lets build a single perceptron to learn the celcius to fahrenheit conversion 

# In[ ]:


#Early stop the training
callball = EarlyStopping(monitor='val_loss', mode='min', verbose=1)


# ### Creating a single perceptron

# In[ ]:


perceptron = Sequential()
perceptron.add(Dense(1, input_dim=1, activation = 'linear'))
perceptron.compile(loss = 'mse', learning_rate=0.001)


# In[ ]:


training = perceptron.fit(train_c, train_f, epochs=10000, verbose=False, callbacks=[callball])


# In[ ]:


perceptron.get_weights()


# In[ ]:


plt.plot(training.history['loss'])


# In[ ]:


perceptron.predict([20])


# ### If we want to work this backwards: Fahrenheit to Celcius we need to regenerate a neural net

# In[ ]:


perceptron = Sequential()
perceptron.add(Dense(3, input_dim=1, activation = 'linear'))
perceptron.add(Dense(1, activation = 'linear'))
perceptron.compile(loss = 'mse', learning_rate=0.01)


# In[ ]:


training = perceptron.fit(train_f, train_c, epochs=8000, verbose=False)


# In[ ]:


plt.plot(training.history['loss'])


# In[ ]:


perceptron.predict([25])


# In[ ]:


print (perceptron.layers[0].get_weights())
print (perceptron.layers[1].get_weights())


# In[ ]:


#Begin format of the classification NN model with classification data generate from sklearn
from sklearn.datasets import make_classification, make_circles, make_blobs

X, y = make_circles(n_samples=1000, noise=0.05, factor=0.5)

colors=['red', 'blue']

plt.scatter(X[:, 0], X[:, 1], c=y)


# In[ ]:




