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


train = []
for i in range(40):
    train.append(np.load('../input/tf-embeddings-processed-train-files-first-half/train_embeddings'+str(i)+'.npy'))


# In[ ]:


for i in range(40, 80):
    train.append(np.load('../input/tf-embeddings-processed-train-files-second-half/train_embeddings'+str(i)+'.npy'))


# In[ ]:


len(train)


# In[ ]:


train = np.vstack(train)


# In[ ]:


train.shape


# In[ ]:


target = np.load('../input/tf-embeddings-processed-train-files-first-half/train_target.npy')


# In[ ]:


target.shape


# In[ ]:


test = []
for i in range(40):
    test.append(np.load('../input/tf-embeddings-processed-test-files-first-half/test_embeddings'+str(i)+'.npy'))


# In[ ]:


for i in range(40, 77):
    test.append(np.load('../input/tf-embeddings-processed-test-files-second-half/test_embeddings'+str(i)+'.npy'))


# In[ ]:


test = np.vstack(test)


# In[ ]:


test.shape


# In[ ]:


np.save('train', train)
np.save('test', test)
np.save('target', target)


# In[ ]:


feature_columns = ['C'+str(i) for i in range(512)]


# In[ ]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[ ]:


target = pd.DataFrame(data=target, columns=class_names)
target.head()


# In[ ]:


train = pd.DataFrame(data= train, columns=feature_columns)
test = pd.DataFrame(data= test, columns=feature_columns)


# In[ ]:


train[class_names] = target[class_names]


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)


# In[ ]:




