#!/usr/bin/env python
# coding: utf-8

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#References:
#https://www.kaggle.com/yuliagm/talkingdata-eda-plus-time-patterns
#https://arxiv.org/abs/1604.06737
#https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os,gc
print(os.listdir("../input"))

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.layers import Embedding, Input, Dense, concatenate, Flatten,Reshape
from keras.models import Model
from keras.initializers import glorot_normal
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

# Any results you write to the current directory are saved as output.


# In[6]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[4]:


N = 184903890
size = 400
sample = np.random.choice(N,size)
# pd.read_csv('../input/train.csv',index_col=sample)


# In[ ]:


traindf = pd.read_csv('../input/train_sample.csv')
# traindf_sample = pd.read_csv("../input/train.csv", skiprows=00000000, nrows=40000000, header=None)
traindf_sample.columns = traindf.columns
testdf = pd.read_csv('../input/test.csv')
traindf.columns


# In[ ]:


print(traindf.loc[~pd.isnull(traindf['attributed_time']),:].shape[0]/traindf.shape[0])
traindf['click_time_f'] = pd.to_datetime(traindf['click_time'])
traindf['attributed_time_f'] = pd.to_datetime(traindf['attributed_time'],errors='coerce')

testdf['click_time_f'] = pd.to_datetime(testdf['click_time'])
# testdf['attributed_time_f'] = pd.to_datetime(testdf['attributed_time'],errors='coerce')


# In[ ]:


deltas = (traindf['attributed_time_f'] - traindf['click_time_f']).dt.seconds
deltas = deltas.replace(to_replace=np.NaN,value=0.0)
deltas.plot()


# In[ ]:


def get_deltas(data):
    clk_time = pd.to_datetime(data['click_time'])
    att_time = pd.to_datetime(data['attributed_time'],errors='coerce')
    deltas = (att_time-clk_time).dt.seconds
    deltas = deltas.replace(to_replace=np.NaN,value=0.0)
    deltas = (deltas/deltas.max()).values.astype(np.float32)
    data['reg_deltas'] = deltas
    return data


# In[ ]:


traindf_sample = get_deltas(traindf_sample)


# In[ ]:


print('Max time delta: {} seconds'.format(deltas.max()));
print('Max Delta datapoint: \n{}'.format(traindf.loc[np.argmax(deltas.values),:]))


# In[ ]:


#Hourly click count feature:
traindf['click_time'] = pd.to_datetime(traindf['click_time'])
traindf_sample['click_time'] = pd.to_datetime(traindf_sample['click_time'])
testdf['click_time'] = pd.to_datetime(testdf['click_time'])
traindf['click_time'].dt.hour.value_counts().reset_index().plot(kind='scatter',x='index',y='click_time')

traindf['hour'] = traindf['click_time'].dt.hour
traindf_sample['hour'] = traindf_sample['click_time'].dt.hour
testdf['hour'] = testdf['click_time'].dt.hour


# In[ ]:


#Values only on four days:
traindf['click_time'].dt.dayofweek.value_counts().reset_index().plot(kind='scatter',x='index',y='click_time')


# In[ ]:


traindf['click_time'].dt.dayofweek.value_counts().reset_index().plot(kind='scatter',x='index',y='click_time')


# In[ ]:


testdf['click_time'].dt.dayofweek.value_counts().reset_index().plot(kind='scatter',x='index',y='click_time')


# In[ ]:


traindf['click_time'].dt.second.value_counts().reset_index().plot(kind='scatter',x='index',y='click_time')


# In[ ]:


cols = ['ip','app','device', 'os', 'channel','hour']


# In[ ]:


def split_t(X):
    return np.hsplit(X,X.shape[1])


# In[ ]:


X = split_t(traindf_sample[cols])
Xtest = split_t(testdf[cols])


# In[ ]:


Y = traindf_sample['is_attributed'].values
reg_deltas = traindf_sample['reg_deltas'].values


# In[ ]:


def make_emb(data,col):
    col_dim = data[col].unique().shape[0]
    if col_dim>1000:
        out_dim = 5
        col_dim = 600000
    elif col_dim<1000 and col_dim>100:
        out_dim = 5
        col_dim = 10000
    elif col_dim<100 and col_dim>10:
        out_dim = 5
        col_dim = 5000
    else:
        out_dim = 5
    emb = Embedding(col_dim,out_dim,input_length=1,embeddings_initializer='glorot_normal',name=col+'_emb')
    inp = Input((1,),name=col+'_input')
    return inp,emb(inp)


# In[ ]:


embs = [make_emb(traindf,col) for col in cols]
inps = [inp for inp,emb in embs]
x = concatenate([emb for inp,emb in embs])
x = Flatten()(x)
x = Dense(20,activation='relu')(x)
x = Dense(10,activation='relu')(x)
is_attributed = Dense(1,activation='sigmoid',name='is_attributed')(x)
deltas_regr = Dense(1,activation='sigmoid',name='reg_deltas')(x)
model = Model(inps,[is_attributed,deltas_regr])


# In[ ]:


model.summary()


# In[ ]:


#Compile and fit model:
model.compile('adam',['binary_crossentropy','mse'])
model.fit(X,[Y,reg_deltas],batch_size=512*16,validation_split=0.1,epochs=1)


# In[ ]:


test_preds = model.predict(Xtest)


# In[ ]:


#Create submission:
outdf = pd.DataFrame()
outdf['click_id'] = testdf['click_id']
outdf['is_attributed'] = test_preds[0]
outdf.to_csv('sub.csv',index=False)


# In[ ]:




