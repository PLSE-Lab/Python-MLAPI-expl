#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sb
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.layers.core import Dropout
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import random
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.

There is my function to show graphs with that, code will look simplier now
# In[ ]:


import operator
def showGraphs(speciesList,feature,style):
    rgb = [0.2,0.1,0.4]
    for species in speciesList:
        method = getattr(plt,style)
        method(species.Id,getattr(species,feature),color=(rgb[0], rgb[1], rgb[2], 1), label=species.Species)
        plt.xlabel("Id")
        plt.ylabel(feature)
        rgb[0]+=0.3
        rgb[1] +=0.1
        rgb[2] -=0.1
        


# In[ ]:


data = pd.read_csv("../input/iris/Iris.csv")


# In[ ]:


print(data.columns)


# In[ ]:


data.info()


# In[ ]:


print(data.Species.unique())


# In[ ]:


SpeciesList = [data[data.Species == "Iris-setosa"],
               data[data.Species == "Iris-versicolor"],
               data[data.Species == "Iris-virginica"]]


# In[ ]:


showGraphs(SpeciesList,"PetalWidthCm","plot")


# In[ ]:


showGraphs(SpeciesList,"PetalWidthCm","scatter")

#We will use it to classify if iris is setosa or not
maxP = (data[data.Species== "Iris-setosa"].PetalWidthCm).max()


# These graphs shows us that we can understand which species is the given Iris clearly by looking its Petal Width. On the other hand, unlike Iris-Setosa, Iris-versicolor and Iris-virginica have a possibility of having same Petal width. So that, we can find some way to clarify this situation by looking these 2 species' other features.

# In[ ]:


#I designed species list for these 2 species we need to clarify
IrisSetosaForTest = SpeciesList[:1]
SpeciesList = SpeciesList[1:]
for species in SpeciesList:
    print(species.Species.unique())


# Now let's compare these two species with another features

# In[ ]:


showGraphs(SpeciesList,"SepalLengthCm","scatter")


# We can't use this too, there is one feature left

# In[ ]:


showGraphs(SpeciesList,"SepalWidthCm","scatter")


# Wow... Can't use that too :(

# In[ ]:


showGraphs(SpeciesList,"PetalLengthCm","scatter")


# I was waiting for better difference but it is better than other comparations, i am going to use neural network to identify if type is versicolor or virginica.

# In[ ]:


np.array(SpeciesList[0].SepalLengthCm)


# In[ ]:


#Firstly, we must set dataset to train network
X_Train = np.zeros((50,2),dtype=float)
Y_Train = np.zeros((50,2),dtype=int)
versiSepal = np.array(SpeciesList[0].SepalLengthCm)
versiPetal = np.array(SpeciesList[0].PetalLengthCm)
virgiSepal = np.array(SpeciesList[1].SepalLengthCm)
virgiPetal = np.array(SpeciesList[1].PetalLengthCm)
#X_Train = SpeciesList[0].drop(["Id","SepalWidthCm","PetalWidthCm"],axis=1,inplace = True)
for i in range(0,50):
    X_Train[i][0]= versiSepal[i] if i%2==0 else virgiSepal[i]
    X_Train[i][1]= versiPetal[i] if i%2==0 else virgiPetal[i]
    Y_Train[i][0]=(i%2==0)
    Y_Train[i][1]=(i%2==1)
print(X_Train, Y_Train)
X_Test = np.copy(X_Train)


# In[ ]:


model = Sequential()
X_Test = X_Train[:10,:]
Y_Test = Y_Train[:10,:]
X_Train = X_Train[10:,:]
Y_Train = Y_Train[10:,:]
model.add(Dense(32,input_dim=2,
                kernel_initializer  =   'uniform',
                activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(32,input_dim=2,
                kernel_initializer  =   'uniform',
                activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2 , kernel_initializer='uniform')) # outputs for one hot encoding
model.add(Activation('softmax'))
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam')
model.fit(X_Train,
          Y_Train,
          epochs = 200,
          batch_size = 10,
          validation_split = 0.1,
          verbose = 1)
results = model.evaluate(X_Test,Y_Test)
print(results)


# We created an artificial neural network to classificate iris species as iris-versicolor or iris-virginica. Iris-setosa can be classified easily just looking at PetalWidthCm feature. So last function to classify an iris sample:

# In[ ]:


def classify_iris(specy,model):
    if specy.PetalWidthCm <= maxP:
        print("Iris-Setosa")
    else:
        inputs = np.zeros((1,2),dtype=float)
        inputs[0][0]=specy.SepalLengthCm
        inputs[0][1]=specy.PetalLengthCm
        result = np.zeros((1,2),dtype="float")
        result = model.predict(inputs)
        if(result[0][0]>result[0][1]): print("Iris-Versicolor")
        else: print("Iris-Virginica")
        
param = SpeciesList[0].iloc[0] #versicolor
classify_iris(param,model)
param = SpeciesList[1].iloc[0] #virginica
classify_iris(param,model)
param = IrisSetosaForTest[0].iloc[0] #Setosa
classify_iris(param,model)

