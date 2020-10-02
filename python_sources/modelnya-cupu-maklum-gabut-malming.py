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


train = pd.read_csv('/kaggle/input/hmif-data-science-bootcamp-2019/train-data.csv')
test = pd.read_csv('/kaggle/input/hmif-data-science-bootcamp-2019/test-data.csv')


# In[ ]:


test.head()


# In[ ]:


train.columns


# In[ ]:


train.dtypes


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train['bentuk'] = train['bentuk'].astype('category').cat.codes
test['bentuk'] = test['bentuk'].astype('category').cat.codes
train['status'] = train['status'].astype('category').cat.codes
test['status'] = test['status'].astype('category').cat.codes
train['kurikulum'] = train['kurikulum'].astype('category').cat.codes
test['kurikulum'] = test['kurikulum'].astype('category').cat.codes
train['penyelenggaraan'] = train['penyelenggaraan'].astype('category').cat.codes
test['penyelenggaraan'] = test['penyelenggaraan'].astype('category').cat.codes
train['akses_internet'] = train['akses_internet'].astype('category').cat.codes
test['akses_internet'] = test['akses_internet'].astype('category').cat.codes
train['sumber_listrik'] = train['sumber_listrik'].astype('category').cat.codes
test['sumber_listrik'] = test['sumber_listrik'].astype('category').cat.codes


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


X_train = train.iloc[:,:46].values
y_train = train.iloc[:,46:47].values

X_test = test.iloc[:,:].values


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


y_train


# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train).toarray()


# In[ ]:


y_train


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


model = Sequential()
model.add(Dense(64, input_dim=46, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train, epochs=100, batch_size=64)


# In[ ]:


y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))


# In[ ]:


pred


# In[ ]:


sub = pd.read_csv('/kaggle/input/hmif-data-science-bootcamp-2019/sample-submission.csv')


# In[ ]:


sub['akreditasi'] = pred


# In[ ]:


sub.to_csv('sub.csv', index = False)


# In[ ]:




