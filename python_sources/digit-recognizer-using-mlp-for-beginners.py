#!/usr/bin/env python
# coding: utf-8

# In[204]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

# Any results you write to the current directory are saved as output.


# In[205]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
sub=pd.read_csv('../input/sample_submission.csv')


# In[206]:


if train.isna().sum().all()==0:
  print("yes")


# In[207]:


if test.isna().sum().all()==0:
  print("yes")


# In[208]:


x=train.drop(['label'],axis=1)
x_test=test.copy()


# In[209]:


y=train['label']


# In[210]:


x.shape


# In[211]:


y.shape


# In[212]:


y=pd.Categorical(y)


# In[213]:


y=pd.get_dummies(y)


# In[214]:


y.head()


# In[215]:


from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=0)


# In[216]:


x_train.shape


# In[217]:


x_val.shape


# In[218]:


y_train.shape


# In[219]:


y_val.shape


# In[220]:


x_train=x_train.values
x_val=x_val.values
y_train=y_train.values
y_val=y_val.values
x_test=x_test.values


# In[221]:


scale=np.max(x_train)
x_train=x_train/scale
x_val=x_val/scale
x_test=x_test/scale


# In[222]:


mean=np.mean(x_train)
x_train=x_train-mean
x_val=x_val-mean
x_test=x_test-mean


# In[223]:


model=Sequential()
model.add(Dense(128,activation='relu',input_dim=784))
model.add(Dropout(0.15))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(10, activation='softmax'))


# In[224]:


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# In[225]:


model.fit(x_train, y_train,
          epochs=25,
          batch_size=16)
score = model.evaluate(x_val, y_val, batch_size=16)


# In[226]:


score


# In[227]:


pred=model.predict(x_test,verbose=0)
new_pred = [np.argmax(y, axis=None, out=None) for y in pred]
output=pd.DataFrame({'ImageId':sub['ImageId'],'Label':new_pred})
output.to_csv('Digit_recognizer.csv', index=False)

