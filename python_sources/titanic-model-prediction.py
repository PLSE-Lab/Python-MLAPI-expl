#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential


# In[ ]:


train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


# Handling missing data in training data by imputation (age)
age=train_data['Age']
age_mean=age.mean()
age.fillna(age_mean,inplace=True)
# Handling missing data in training data by imputation (fare)
fare=train_data['Fare']
fare_mean=fare.mean()
fare.fillna(fare_mean,inplace=True)


# In[ ]:


# Handling missing data in test data by imputation (age)
age=test_data['Age']
age_mean=age.mean()
age.fillna(age_mean,inplace=True)
# Handling missing data in test data by imputation (fare)
fare=test_data['Fare']
fare_mean=fare.mean()
fare.fillna(fare_mean,inplace=True)


# In[ ]:


train_data = train_data.drop('Cabin', axis=1)
train_data = train_data.drop('Ticket', axis=1)
train_data = train_data.drop('Name', axis=1)


# In[ ]:


test_data = test_data.drop('Cabin', axis=1)
test_data = test_data.drop('Ticket', axis=1)
test_data = test_data.drop('Name', axis=1)


# In[ ]:


for i in range(len(train_data['Sex'])):
    if train_data['Sex'][i]=='male':
        train_data['Sex'][i]=0
    elif train_data['Sex'][i]=='female' :
        train_data['Sex'][i]=1

for j in range(len(train_data['Embarked'])):
    if j!=61 and j!=829:
        train_data['Embarked'][j]=ord(train_data['Embarked'][j])


# In[ ]:


for i in range(len(test_data['Sex'])):
    if test_data['Sex'][i]=='male':
        test_data['Sex'][i]=0
    elif test_data['Sex'][i]=='female' :
        test_data['Sex'][i]=1
for j in range(len(test_data['Embarked'])):
    test_data['Embarked'][j]=ord(test_data['Embarked'][j])


# In[ ]:


#Dropping Nan values 
train_data=train_data.dropna()
test_data=test_data.dropna()


# In[ ]:


Y=np.array(train_data['Survived'],np.float32)
train_data=train_data.drop('Survived',axis=1)
train_data=train_data.drop('PassengerId',axis=1)
X_train = np.array(train_data,np.float32)

test_data=test_data.drop('PassengerId',axis=1)
X_test = np.array(test_data,np.float32)


# In[ ]:


num_features = 7
Batch_size = 891
training_steps = 25000


# In[ ]:


model = Sequential()
model.add(layers.Dense(1, activation='sigmoid', input_dim=num_features))
model.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


model.fit(X_train, Y, epochs=training_steps, batch_size=Batch_size)


# In[ ]:


weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]


# In[ ]:


L=np.array(tf.sigmoid(np.dot(X_test,weights)+biases))
for i in range(len(L)):
    if list(L[i])[0]<0.5:
        L[i]=0
    else :
        L[i]=1
# L is the prediction array

