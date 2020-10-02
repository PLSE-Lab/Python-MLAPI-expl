#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


digits=pd.read_csv('../input/digit-recognizer/train.csv')


# In[ ]:


digits.info()


# In[ ]:


digits.head()


# In[3]:


labels=digits['label']


# In[ ]:


labels


# In[4]:


digits.drop('label',axis=1,inplace=True)


# In[5]:


import sklearn.neural_network as nn
import sklearn.model_selection as ms


# In[6]:


x_train,x_test,y_train,y_test=ms.train_test_split(digits,labels,test_size=0.2,random_state=22)


# In[ ]:


x_train.shape


# In[7]:


ANN=nn.MLPClassifier()


# In[8]:


ANN.fit(x_train,y_train)


# In[9]:


ANN.score(x_test,y_test)


# In[10]:


test=pd.read_csv('../input/digit-recognizer/test.csv')


# In[11]:


answers=ANN.predict(test)


# In[12]:


submission=pd.read_csv('../input/digit-recognizer/sample_submission.csv')


# In[15]:


submission.head()


# In[14]:


submission['Label']=answers


# In[16]:


submission.to_csv('1st.csv',index=False)

