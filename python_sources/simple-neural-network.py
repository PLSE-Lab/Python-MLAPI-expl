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


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


x = train.iloc[:,2:].values
y = train.iloc[:, 1].values


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc = StandardScaler()
x = sc.fit_transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=42)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Dense(80,kernel_initializer='normal', activation='relu', input_dim=200))
model.add(Dense(1, kernel_initializer='normal', activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train, Y_train, epochs=2, validation_data=(X_test, Y_test))


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.roc_auc_score(Y_test, y_pred)


# In[ ]:


x_test = test.iloc[:, 1:].values


# In[ ]:


x_test = sc.transform(x_test)


# In[ ]:


x_test.shape


# In[ ]:


Y_pred = model.predict(x_test)


# In[ ]:


sub['target'] = Y_pred


# In[ ]:


sub.to_csv('submission.csv', index=False)


# In[ ]:




