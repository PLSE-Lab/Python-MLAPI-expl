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
#train['year']=train['application_date'].apply(lambda x : x.year)

test['weekday']=test['application_date'].apply(lambda x : x.weekday())
test['day']=test['application_date'].apply(lambda x : x.day)
test['month']=test['application_date'].apply(lambda x : x.month)
#test['year']=test['application_date'].apply(lambda x : x.year)


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


test.head()


# In[ ]:


train_seg1.head()


# In[ ]:


features_considered = ['case_count','weekday','day','month']
target = ['case_count']


# In[ ]:


features_s1 = train_seg1[features_considered]
features_s1.head()


# In[ ]:


features_s2 = train_seg2[features_considered]
features_s2.head()


# In[ ]:


features_s1.plot(subplots=True)


# In[ ]:


features_s2.plot(subplots=True)


# In[ ]:


train_seg1_split = int(int(train_seg1.shape[0])/1.2)
train_seg2_split = int(int(train_seg2.shape[0])/1.2)


# In[ ]:


TRAIN_SPLIT_s1=train_seg1_split
TRAIN_SPLIT_s2=train_seg2_split


# In[ ]:


dataset_s1 = features_s1.values
#data_mean_s1 = dataset_s1[:TRAIN_SPLIT_s1].mean(axis=0)
#data_std_s1 = dataset_s1[:TRAIN_SPLIT_s1].std(axis=0)
#dataset_s1 = (dataset_s1-data_mean_s1)/data_std_s1


# In[ ]:


dataset_s2 = features_s2.values
#data_mean_s2 = dataset_s2[:TRAIN_SPLIT_s2].mean(axis=0)
#data_std_s2 = dataset_s2[:TRAIN_SPLIT_s2].std(axis=0)
#dataset_s2 = (dataset_s2-data_mean_s2)/data_std_s2


# In[ ]:


dataset_s2[0]


# In[ ]:


#data_mean_s2


# In[ ]:


dataset_s1[4]


# In[ ]:



def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
    # find the end of this pattern
        end_ix = i + n_steps
    # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# In[ ]:


train_seg1.head()


# In[ ]:


dataset_s1[0]


# In[ ]:


dataset_s2[0]


# In[ ]:


in_seq1_s1 = np.array(list(train_seg1['weekday']))
in_seq2_s1 = np.array(list(train_seg1['day']))
in_seq3_s1 = np.array(list(train_seg1['month']))
#in_seq4_s1 = np.array(list(train_seg1['year']))
out_seq_s1 = np.array(list(train_seg1['case_count']))
#in_seq1 = np.array([1,2,3])
#in_seq2 = np.array([5,6,7])
#in_seq3 = np.array([8,9,0])

in_seq1_s1 = in_seq1_s1.reshape((len(in_seq1_s1), 1))
in_seq2_s1 = in_seq2_s1.reshape((len(in_seq2_s1), 1))
in_seq3_s1 = in_seq3_s1.reshape((len(in_seq3_s1), 1))
#in_seq4_s1 = in_seq4_s1.reshape((len(in_seq4_s1), 1))
out_seq_s1 = out_seq_s1.reshape((len(out_seq_s1), 1))

#dataset_s1 = np.hstack((in_seq1_s1, in_seq2_s1, in_seq3_s1,in_seq4_s1,out_seq_s1))
dataset_s1 = np.hstack((in_seq1_s1, in_seq2_s1, in_seq3_s1,out_seq_s1))
n_steps = 10
# convert into input/output
X_s1, y_s1 = split_sequences(dataset_s1, n_steps)


# In[ ]:


in_seq1_s2 = np.array(list(train_seg2['weekday']))
in_seq2_s2 = np.array(list(train_seg2['day']))
in_seq3_s2 = np.array(list(train_seg2['month']))
#in_seq4_s2 = np.array(list(train_seg2['year']))
out_seq_s2 = np.array(list(train_seg2['case_count']))
#in_seq1 = np.array([1,2,3])
#in_seq2 = np.array([5,6,7])
#in_seq3 = np.array([8,9,0])

in_seq1_s2 = in_seq1_s2.reshape((len(in_seq1_s2), 1))
in_seq2_s2 = in_seq2_s2.reshape((len(in_seq2_s2), 1))
in_seq3_s2 = in_seq3_s2.reshape((len(in_seq3_s2), 1))
#in_seq4_s2 = in_seq4_s2.reshape((len(in_seq4_s2), 1))
out_seq_s2 = out_seq_s2.reshape((len(out_seq_s2), 1))

#dataset_s2 = np.hstack((in_seq1_s2, in_seq2_s2, in_seq3_s2,in_seq4_s2,out_seq_s2))
dataset_s2 = np.hstack((in_seq1_s2, in_seq2_s2, in_seq3_s2,out_seq_s2))
n_steps = 10
# convert into input/output
X_s2, y_s2 = split_sequences(dataset_s2, n_steps)
#n_features_s2 = X_s2.shape[2]
#n_input_s2 = X_s2.shape[1] * X_s2.shape[2]
#X_s2 = X_s2.reshape((X_s2.shape[0], n_input_s2))



# In[ ]:


X_s2.shape


# In[ ]:


test_s1 = test[test['segment']==1]
#x_input_test_s1 = [[x,y,z,z1] for x,y,z,z1 in zip(list(test_s1['weekday']),list(test_s1['day']),list(test_s1['month']),list(test_s1['year']))]
x_input_test_s1 = [[x,y,z] for x,y,z in zip(list(test_s1['weekday']),list(test_s1['day']),list(test_s1['month']))]

test_s2 = test[test['segment']==2]
x_input_test_s2 = [[x,y,z] for x,y,z in zip(list(test_s2['weekday']),list(test_s2['day']),list(test_s2['month']))]


# In[ ]:


x_input_test_final_s1 = []
i = 1
thval = len(x_input_test_s1)-n_steps
#print(thval)
for x in range(len(x_input_test_s1)):
    #print(thval-x,thval)
    if (thval-x)<0:
        #print(x_input_test_final_s1[-1])
        val = x_input_test_s1[x:x+10] + list([x_input_test_s1[x]])*np.abs(thval-x)
        x_input_test_final_s1.append(val)
    else:
        x_input_test_final_s1.append(x_input_test_s1[x:x+10])     


# In[ ]:


x_input_test_final_s1[-3]


# In[ ]:


len(x_input_test_final_s1[-1])


# In[ ]:


x_input_test_final_s2 = []
i = 1
thval = len(x_input_test_s2)-n_steps
#print(thval)
for x in range(len(x_input_test_s2)):
    #print(thval-x,thval)
    if (thval-x)<0:
        #print(x_input_test_final_s1[-1])
        val = x_input_test_s2[x:x+10] + list([x_input_test_s2[x]])*np.abs(thval-x)
        x_input_test_final_s2.append(val)
    else:
        x_input_test_final_s2.append(x_input_test_s2[x:x+10])     


# In[ ]:


print(X_s1.shape, y_s1.shape)
# summarize the data
for i in range(len(X_s1)):
    print(X_s1[i], y_s1[i])


# In[ ]:


print(X_s2.shape, y_s2.shape)
# summarize the data
for i in range(len(X_s2)):
    print(X_s2[i], y_s2[i])


# In[ ]:


train_seg2.tail()


# In[ ]:


#n_features_s1,n_features_s2


# In[ ]:


n_input_s1 = X_s1.shape[1] * X_s1.shape[2]
X_s1 = X_s1.reshape((X_s1.shape[0], n_input_s1))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100, activation='relu', input_dim=n_input_s1))
#model.add(tf.keras.layers.Dense(1))
#model.add(tf.keras.layers.GRU(20, return_sequences=True))
#model.add(tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
#model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_s1, y_s1, epochs=2000,verbose=0)


# In[ ]:


results_s1 = []
for input_val in x_input_test_final_s1:
    val = np.array(input_val)
    yhat = val.reshape((1, n_input_s1))
    yhat = model.predict(yhat)
    results_s1.append(yhat[0][0])


# In[ ]:


results_s1


# In[ ]:


train_seg1.tail(25)


# In[ ]:


x_input_test_final_s1


# In[ ]:


n_input_s2 = X_s2.shape[1] * X_s2.shape[2]
X_s2 = X_s2.reshape((X_s2.shape[0], n_input_s2))

model_s2 = tf.keras.Sequential()
model_s2.add(tf.keras.layers.Dense(100, activation='relu', input_dim=n_input_s2))
model_s2.add(tf.keras.layers.Dense(1))
model_s2.compile(optimizer='adam', loss='mse')

model_s2.fit(X_s2, y_s2, epochs=2000,verbose=0)


# In[ ]:


x_input = np.array([[ 5,  6,  7],[ 6,  7,  7],[ 0,  8,  7]])


# In[ ]:


results_s2 = []
for input_val in x_input_test_final_s2:
    val = np.array(input_val)
    yhat = val.reshape((1, n_input_s2))
    yhat = model_s2.predict(yhat)
    results_s2.append(yhat[0][0])


# In[ ]:


results_s2


# In[ ]:


x_input_test_final_s2


# In[ ]:


train_seg2.tail(45)


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




