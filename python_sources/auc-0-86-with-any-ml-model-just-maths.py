#!/usr/bin/env python
# coding: utf-8

# In[24]:


# This kernel might be the shortest code in Kaggle competitions
# I will use no machine learning in the predictions but only mathematics ! (maths not hot ! just sauce...)
# From the kernels of other competitors approximately all linear regression 
# hypotheses are satistified (normal distribution of features, Homoscedasticity, ...)
# So We can solve the problem where 
# y = Ax (1)
# y' = A'x (2)
# y = (ground thruth i.e known target), A = training data, 
# and b = bias (y - Ax); y' = target to predict, A' = testing data and b' = bias of the predicted target: 
# Solution: (1) => x = A^-1y and replace it in (2)=> y' = A'A^-1y


# In[7]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings #filtering warnings
warnings.filterwarnings('ignore')


# In[17]:


df_train = pd.read_csv('../input/train.csv', low_memory=False)
df_test = pd.read_csv('../input/test.csv',low_memory=False)


# In[18]:


ids_test = df_test['ID_code']
y = df_train['target']
del df_train['target']
del df_train['ID_code']
del df_test['ID_code']


# In[25]:


# Min Max scaling
A = (df_train - df_train.min()) / (df_train.min() - df_train.max())
A_prime = (df_test - df_test.min()) / (df_test.min() - df_test.max())


# In[20]:


# solving (1) with numpy linalg.lstsq
x,_,_,_ = np.linalg.lstsq(A, y)


# In[30]:


# predicting test data target
y_prime = np.dot(A_prime, x)


# In[31]:


submission = pd.DataFrame({'ID_code':ids_test, 'target':y_prime})
submission.head(20)


# In[29]:


submission.to_csv('submission.csv', index = False)


# Yes you are right ! using the linear model in Scikit-Learn with solver = least-sqarre will give approximately the same score.
