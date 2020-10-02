#!/usr/bin/env python
# coding: utf-8

# Creating a Deep learning predictive model from the data set to get the prediction whether a country can have banking crisis in future with the given set of data with simple hold out validation.
# 
# Note: However I will be using simple hold out validation but K fold validation and iterated K fold will be a better choice as we have a small dataset of 1059 rows.

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


df = pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')


# In[ ]:


df.head()


# Checking for null values in dataframe

# In[ ]:


df.isnull().sum()


# Feature engineering => dropping unneccessary columns which do not have effect of predictions

# In[ ]:


df.drop(['cc3'],axis=1,inplace=True)
df.head()


# Label encoding to covert string values to integers

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df.country = le.fit_transform(df.country)
df.banking_crisis = le.fit_transform(df.banking_crisis)


# Randomizing the data for non bias and doing train and test split in 80% and 20%

# In[ ]:


train_test_split = np.random.rand(len(df)) < 0.7
train = df[train_test_split]
test = df[~train_test_split]


# looking few rows of train and test data

# In[ ]:


train[:10]


# In[ ]:


len(train)


# In[ ]:


test[:10]


# In[ ]:


len(test)


# Column 12 - "banking_crisis" is what i have to predict from the given dataset so I will map 'train_data' from col 0 to col 11 and will map 'train_label' to col 12 which we have to predict and the same for test data as well

# In[ ]:


train_data = train.iloc[:,0:12]
train_data.head()


# In[ ]:


train_label = train.iloc[:,12]
train_label.head()


# In[ ]:


test_data = test.iloc[:,0:12]
test_data.head()


# In[ ]:


test_label = test.iloc[:,12]
test_label.head()


# Creating the network architecture: selecting 3 layer model (one input, one hidden and one output) with 8 hidden units of layer for input and hidden layer (to lessen the complexity) and one for output layer with activation function as sigmoid

# In[ ]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(8, activation = 'relu', input_shape = (12,)))
model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))


# Compiling the model and fixing the required metrics

# In[ ]:


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])


# Seggregating 100 validation samples from training data to check training accuracy and loss and validation accuracy and loss

# In[ ]:


x_val = train_data[:100]
partial_x_train = train_data[100:]
y_val = train_label[:100]
partial_y_train = train_label[100:]


# Running model fit to train the model and test on validation data, after the 9th epochs the model started to show appreciable and consistant accuracy

# In[ ]:


history = model.fit(partial_x_train, partial_y_train, epochs = 60, batch_size = 10, validation_data = (x_val, y_val))


# Finally running the model on the test data

# In[ ]:


model.fit(train_data,train_label, epochs = 60, batch_size = 10)
results = model.evaluate(test_data, test_label)


# The final accuracy turned out to be 96.04%

# In[ ]:


print(results)


# A fairly naive approach achieved an accuracy of 96.04% which can be increased with other state of art approaches such as K fold validation and iterated K fold.
