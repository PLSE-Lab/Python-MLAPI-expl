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


# # Time series forecasting

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# In[ ]:


tf.__version__


# In[ ]:


train = pd.read_csv('/kaggle/input/ltfs-2/train_fwYjLYX.csv')
test = pd.read_csv('/kaggle/input/ltfs-2/test_1eLl9Yf.csv')


# In[ ]:


train['application_date']=pd.to_datetime(train['application_date'],format="%Y-%m-%d")
test['application_date']=pd.to_datetime(test['application_date'],format="%Y-%m-%d")


# In[ ]:


train=train.groupby(['application_date','segment']).sum().reset_index()


# In[ ]:


train['weekday']=train['application_date'].apply(lambda x : x.weekday())
train['day']=train['application_date'].apply(lambda x : x.day)
train['month']=train['application_date'].apply(lambda x : x.month)
train['year']=train['application_date'].apply(lambda x : x.year)

test['weekday']=test['application_date'].apply(lambda x : x.weekday())
test['day']=test['application_date'].apply(lambda x : x.day)
test['month']=test['application_date'].apply(lambda x : x.month)
test['year']=test['application_date'].apply(lambda x : x.year)


# In[ ]:


train_seg1 = train[train['segment']==1]
train_seg2 = train[train['segment']==2]


# In[ ]:


idx = pd.date_range('2017-04-01', '2019-07-05')
train_seg1.index=train_seg1.application_date
train_seg1.drop(['application_date'],axis=1,inplace=True)
train_seg1=train_seg1.reindex(idx,method='ffill')
train_seg1.head()


# In[ ]:


idx = pd.date_range('2017-04-01', '2019-07-23')
train_seg2.index=train_seg2.application_date
train_seg2.drop(['application_date'],axis=1,inplace=True)
train_seg2=train_seg2.reindex(idx,method='ffill')
train_seg2.head()


# In[ ]:


idx = pd.date_range('2019-07-06', '2019-10-24')
test.index=test.application_date
test.drop(['application_date'],axis=1,inplace=True)
#test=test.reindex(idx,method='ffill')
test.head()


# In[ ]:


features_considered = ['case_count','weekday','day','month','year']
target = ['case_count']


# In[ ]:



def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
    # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# In[ ]:


raw_seq_s1 = list(train_seg1['case_count'])
n_steps = 9
X_s1, y_s1 = split_sequence(raw_seq_s1, n_steps)


# In[ ]:


raw_seq_s2 = list(train_seg2['case_count'])
n_steps = 9
X_s2, y_s2 = split_sequence(raw_seq_s2, n_steps)


# In[ ]:


initial_input_s1 = list(X_s1[-1])
initial_input_s2 = list(X_s2[-1])


# In[ ]:


initial_input_s2


# In[ ]:


test_s1 = test[test['segment']==1]
test_s2 = test[test['segment']==2]


# In[ ]:


test_s1.shape


# In[ ]:


X_s1.shape


# In[ ]:


n_features = 1
n_seq = 3
n_steps = 3
X_s1 = X_s1.reshape((X_s1.shape[0], n_seq, n_steps, n_features))
X_s2 = X_s2.reshape((X_s2.shape[0], n_seq, n_steps, n_features))


# In[ ]:


X_s1.shape


# In[ ]:


X_s2.shape


# In[ ]:


train_seg1.head()


# In[ ]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),input_shape=(None, n_steps, n_features)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
model.add(tf.keras.layers.LSTM(50, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X_s1, y_s1, epochs=1000,verbose=0)


# In[ ]:


test_s1.shape[0]


# In[ ]:


results_s1 = []
n_steps_1=9
n_steps_2 = 3
for i in range(test_s1.shape[0]):
    val = np.array(initial_input_s1[i:i+n_steps_1])
    #print(val)
    #val = val.reshape((1, n_steps, n_features))
    val = val.reshape((1, n_seq, n_steps_2, n_features))
    yhat = model.predict(val)
    initial_input_s1.append(yhat[0][0])
    results_s1.append(yhat[0][0])


# In[ ]:


results_s1


# In[ ]:


train_seg1.tail(10)


# In[ ]:


model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),input_shape=(None, n_steps, n_features)))
model2.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)))
model2.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
model2.add(tf.keras.layers.LSTM(50, activation='relu'))
model2.add(tf.keras.layers.Dense(1))

model2.compile(optimizer='adam', loss='mse')

model2.fit(X_s2, y_s2, epochs=1000,verbose=0)


# In[ ]:


results_s2 = []
n_steps_1=9
n_steps_2 = 3
for i in range(test_s2.shape[0]):
    val = np.array(initial_input_s2[i:i+n_steps_1])
    #print(val)
    #val = val.reshape((1, n_steps, n_features))
    val = val.reshape((1, n_seq, n_steps_2, n_features))
    yhat = model2.predict(val)
    initial_input_s2.append(yhat[0][0])
    results_s2.append(yhat[0][0])


# In[ ]:


results_s2


# In[ ]:


train_seg2.tail()


# In[ ]:


test_s1['case_count']= results_s1
test_s2['case_count']= results_s2
submission = pd.concat([test_s1,test_s2])


# In[ ]:


submission = submission.reset_index().sort_values('id')[['id','application_date','segment','case_count']]
submission.head()


# In[ ]:


submission.to_csv('submission.csv',index=False)


# In[ ]:


submission.shape


# In[ ]:




