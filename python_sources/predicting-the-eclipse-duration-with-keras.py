#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
import keras
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


train_ = pd.read_csv('../input/dark-side-of-moon-dataset/train.csv')
test_ = pd.read_csv('../input/dark-side-of-moon-dataset/test.csv')


# In[ ]:


train_.head(10)


# In[ ]:


test_.head(10)


# In[ ]:


print(train_.shape)
print(test_.shape)


# In[ ]:


train_['Lunation Number'].value_counts()


# In[ ]:


train_nans = train_.shape[0] - train_.dropna().shape[0]
test_nans = test_.shape[0] - test_.dropna().shape[0]
print(train_nans)
print(test_nans)


# In[ ]:


train_.isnull().sum()


# In[ ]:


cat = train_.select_dtypes(include= ['O'])
cat.apply(pd.Series.nunique)


# In[ ]:





# In[ ]:


t_data = [train_,test_]
for data in t_data:
    for x in data.columns:
        if data[x].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[x].values))
            data[x] = lbl.transform(list(data[x].values))


# In[ ]:


train_.head(10)


# In[ ]:


def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))*20
    return dataNorm


# In[ ]:


train_norm= normalize(train_)
test_norm = normalize(test_)   


# In[ ]:


train_norm.head(10)


# In[ ]:


test_norm.head(10)


# In[ ]:


y = train_['Eclipse Duration (m)']


# In[ ]:


X = train_norm.drop(columns = ['Eclipse Duration (m)'])


# In[ ]:


X.shape


# In[ ]:


print(y.shape)


# In[ ]:


sc = StandardScaler()
X = sc.fit_transform(X)
test = sc.fit_transform(test_norm)


# In[ ]:


model = Sequential()
model.add(Dense(11,kernel_initializer='uniform',activation='relu',input_dim=11))
model.add(Dense(11,kernel_initializer='uniform',activation='relu'))
model.add(Dense(5,kernel_initializer='uniform',activation='relu'))
model.add(Dense(1,kernel_initializer='uniform',activation='linear'))
          


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam',loss ='mean_squared_error',metrics=['accuracy'])
model.fit(X,y,batch_size=32,nb_epoch=300)


# In[ ]:


test_norm.shape


# In[ ]:


test_norm.head(10)


# In[ ]:


test_norm = test_norm.drop(columns = ['ID'])


# In[ ]:


train_norm.head()


# In[ ]:


samp = pd.read_csv('../input/dark-side-of-moon-dataset/sample_submission.csv')
y_pred = model.predict(test_norm)


# In[ ]:


y_final = y_pred.astype(int).reshape(test_norm.shape[0])


# In[ ]:


samp['Eclipse Duration (m)'] = y_final
samp.to_csv('sample_ans.csv',index = False)


# In[ ]:




