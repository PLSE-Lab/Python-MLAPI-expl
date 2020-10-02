#!/usr/bin/env python
# coding: utf-8

# ## Gender Voice recognition using Tensorflow and Keras
# 
# This is my first Kernel for a Kaggle dataset. Will perform some exploratory data analysis and then move ahead with classification using Tensorflow and Keras.
# 
# Always happy to learn so please feel free to give feedback and thoughts! Thanks

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import losses,optimizers,metrics

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# #### Read in data
# 
# Look at standard description and information on dataset

# In[ ]:


voice = pd.read_csv('../input/voice.csv')

print(voice.columns)
voice.head()


# In[ ]:


voice.describe()


# ### Preprocessing 1
# The above shows us that all the data is numerical except for the 'label' column. In order to train the neural netowrk we will need to change the labels to a numeric form. Let's do this by using Male == 0 and Female == 1.

# In[ ]:


voice = pd.get_dummies(voice)
voice.drop('label_male',axis=1,inplace=True)
voice.head()


# In[ ]:


# change label if you want
voice['label'] = voice['label_female']
voice.drop('label_female',axis=1,inplace=True)


# ### Data exploration
# 
# Let's begin exporing the data. First we can check for correlations.

# In[ ]:


plt.figure(figsize=[16,9])
mask = np.ones_like(voice.corr())
mask[np.tril_indices_from(mask)] = False
sns.heatmap(voice.corr(),mask=mask,vmin=-1,vmax=1,cmap='coolwarm',annot=True,linewidths=0.5)


# From the correlation heatmap we can see a few things:
# 1. *meanfun* seems to have the strongest correlation with the label
# 2. Other feautures with correlations above abs(0.3) are *meanfreq, sd, Q25, IQR, sp.ent, sfm,* and * centroid*
# 3. *Q75, skew, kurt* and *modindx* essentially have no correlation i.e. they are "independent" with regards to sex
# 4. *centroid* and *meanfreq* are pefectly correlated as they are the same

# We can remove *centroid* as we dont want to train on the "same" feature twice

# In[ ]:


voice.drop('centroid',axis=1,inplace=True)


# What would be cool is to check how seperable the label is for some of these higher correlation features. Can use some boxplots for this. Or if you want a massive pairplot. We will only do two boxplots here.

# In[ ]:


plt.figure(figsize=[9,6])
sns.boxplot(x='label',y='meanfreq',data=voice)
plt.xticks([0,1],['male','female'])
plt.xlabel(xlabel=None)


# In[ ]:


plt.figure(figsize=[9,6])
sns.boxplot(x='label',y='meanfun',data=voice)
plt.xticks([0,1],['male','female'])
plt.xlabel(xlabel=None)


# *meanfun* seems to be able to separate the data really well.

# ### Preprocessing 2
# 
# Now onto the second part of preprocessing and separation of data from labels

# In[ ]:


voice_data = voice.drop('label',axis=1)
voice_label = voice['label']


# Split the data into training and testing sets. For now just do training and testing but later split into 3: train, validation, and test.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(voice_data,voice_label,test_size=0.3)


# Now scale the data for use in a neural network

# In[ ]:


scaler = MinMaxScaler()
# only train the scaler on the training data
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

print('Scaled training data shape: ',scaled_x_train.shape)


# Now start neural network model definition. We have 19 input features and we want our output to classify the data for either male or female. We therefore need two outputs using 'softmax' activation function. The hidden layers will use the standard 'relu' activation function.

# In[ ]:


dnn_keras_model = models.Sequential()


# In[ ]:


# can play around with the number of hidden layers but I found that one hidden layer was more than enough to give great metrics
dnn_keras_model.add(layers.Dense(units=30,input_dim=19,activation='relu'))
# dnn_keras_model.add(layers.Dense(units=30,activation='relu'))
dnn_keras_model.add(layers.Dense(units=20,activation='relu'))
dnn_keras_model.add(layers.Dense(units=10,activation='relu'))
dnn_keras_model.add(layers.Dense(units=2,activation='softmax'))


# In[ ]:


# compile model by selecting optimizer and loss function
dnn_keras_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


# train/fit the model
dnn_keras_model.fit(scaled_x_train,y_train,epochs=50)


# Now we can make our predictions with the test set

# In[ ]:


predictions = dnn_keras_model.predict_classes(scaled_x_test)


# In[ ]:


print('Metric for ')
print('Classification report:')
print(classification_report(predictions,y_test))
print('\n')
print('Confusion matrix:')
print(confusion_matrix(predictions,y_test))
print('\n')
print('Accuracy score is {:6.3f}.'.format(accuracy_score(predictions,y_test)))


# So with the above setup, we are getting about 98% accuracy, precision and recall which is pretty good!

# In[ ]:




