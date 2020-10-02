#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import numpy as np 
import pandas as pd 
import pandas as pd 
import numpy as np 
import scipy as sp 
import sklearn
import random 
import time 

from sklearn import preprocessing, model_selection


from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle


# 
# This is a very basic example of a construction of a neural network that allows for a multiclass classification with tensorflow keras.

# In[ ]:


data = pd.read_csv('../input/Iris.csv')
data = data.drop(['Id'], axis =1)


# 
# We are going to separate the data. One part will be used to make predictions in the end, the other part, the most important will be used for training and testing the neural network.
# This part is not mandatory, but it is for fun, and especially to show how to predict from an input.

# In[ ]:


data = shuffle(data)


i = 8
data_to_predict = data[:i].reset_index(drop = True)
predict_species = data_to_predict.Species 
predict_species = np.array(predict_species)
prediction = np.array(data_to_predict.drop(['Species'],axis= 1))

data = data[i:].reset_index(drop = True)




# In[ ]:


X = data.drop(['Species'], axis = 1)
X = np.array(X)
Y = data['Species']



# 
# We must transform the column of classes, because we have a format 'str', and it is a multiclass situation. We must first convert the names of species into numerical values, then into vectors for the output of the neuron network. 
# 

# In[ ]:


# Transform name species into numerical values 
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y)
#print(Y)

# We have 3 classes : the output looks like : 
#0,0,1 : Class 1
#0,1,0 : Class 2
#1,0,0 : Class 3


# In[ ]:


train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size = 0.1, random_state = 0)


# 
# It's time to build our neural network. The dimension in input is the number of features of the dataframe (without the class to predict!).
# 
# We are on a multiclass classification situation, so the activation function for the last most suitable layer is "softmax", and "categorical_crossentropy" for the loss.
# 
# We have to do several tests to find the best architecture, but this one works pretty well

# In[ ]:


input_dim = len(data.columns) - 1

model = Sequential()
model.add(Dense(8, input_dim = input_dim , activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_x, train_y, epochs = 10, batch_size = 2)

scores = model.evaluate(test_x, test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# 
# It is time to make predictions with the small sample removed from the base at the beginning.
# To train the neural network it was necessary to convert the species into vectors. So after the prediction it is necessary to carry out the opposite operation to recover the name of the associated species

# In[ ]:


predictions = model.predict_classes(prediction)
prediction_ = np.argmax(to_categorical(predictions), axis = 1)
prediction_ = encoder.inverse_transform(prediction_)

for i, j in zip(prediction_ , predict_species):
    print( " the nn predict {}, and the species to find is {}".format(i,j))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




