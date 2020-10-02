#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# In[ ]:


import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# This kernel will use an artificial neural net to classify voices by gender.  The neural net will have two hidden layers, will use a rectifier activation function for each node in the hidden layers, and use a logistic activation function to recieve probabilities of each class in the output layer and will use stochastic gradient descent to minimize the objective function (cross entropy).  We begin by reading the dataset and do some elementary data exploration.

# In[ ]:


dataset = pd.read_csv('../input/voice.csv')
dataset.head()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.corr()


# All are numeric columns, and the final column gives us the labels (male or female) so lets lazily determine which column that is and use that to split our dataset into the independent and dependent variables:

# In[ ]:


num_columns = dataset.shape[1]
x = dataset.iloc[:,:20].values
y = dataset.iloc[:,20].values


# The one non numeric category (labels) needs to be encoded using dummy variables,when we assign integer values to gender (binary variable in this case), it becomes a dummy variable.

# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
gender_labels = LabelEncoder()
y = gender_labels.fit_transform(y)
# lets see which is 0 and which is 1
print(list(gender_labels.inverse_transform([0,1])))


# Generate testing and training datasets

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# These variables are between 0 and 1, but we normalize them so that they have comparable weightings.

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Creating an artificial neural network using keras, by building the layers two hidden layers and an output layer, then train the neural network on the training set.

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 11, activation = 'relu', kernel_initializer = 'uniform', input_shape = (20,)))
classifier.add(Dense(units = 11, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform', input_shape = (20,)))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)


# The neural network is now built and trained, the weights are set, now we can evaluate the generalization performance of the neural network on the testing dataset.  We then use the threshold of 0.50 to convert output probilities from the sigmoid activation function into binary predictions for gender.

# In[ ]:


y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)


# Evaluate the performance of the neural network using a confusion matrix.

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# A more fashionable plot, recall that 0 is female and 1 is male.

# In[ ]:


import matplotlib.pyplot as plt
plt.matshow(cm)
plt.colorbar()


# Results on the testing dataset are fantastic the confusion matrix shows that the neural net predicts gender on the testing dataset with 98.3% accuracy.
