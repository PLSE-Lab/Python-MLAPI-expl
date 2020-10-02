#!/usr/bin/env python
# coding: utf-8

# In[2]:


mf = '../input/fork-of-fork-of-densenet201/model.hdf5'
import os
os.system('ls '+mf)


# In[ ]:


#os.system(f'ls {mf}1')


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


x = np.load('../input/prepare-for-submission/0.npy')


# In[ ]:


from keras.models import load_model
import tensorflow as tf
def logloss(y, y_):
    return tf.losses.log_loss(y,y_)

model = load_model(mf, custom_objects={'logloss':logloss})


# In[ ]:


import gc


# In[ ]:


y1 = model.predict(x)


# In[ ]:


del x
gc.collect()


# In[ ]:


x = np.load('../input/prepare-for-submission-2/1.npy')
y2 = model.predict(x)
del x
gc.collect()


# In[ ]:


x = np.load('../input/prepare-for-submission-3/2.npy')
y3 = model.predict(x)
del x
gc.collect()


# In[ ]:


y = np.concatenate([y1, y2, y3])


# In[ ]:


y.shape


# In[ ]:


df = pd.DataFrame()
df['ID'] = pd.read_csv('../input/gendered-pronoun-resolution/test_stage_2.tsv', delimiter='\t')['ID']


# In[ ]:


df_s = pd.read_csv('../input/gendered-pronoun-resolution/sample_submission_stage_2.csv')


# In[ ]:


df['A'] = y[:,0] #, 'B', 'NEITHER'


# In[ ]:


df['B'] = y[:,1]
df['NEITHER'] = y[:,2]


# In[ ]:


df.head()


# In[ ]:


df.to_csv('submission.csv', index=False)

