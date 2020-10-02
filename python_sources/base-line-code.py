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


df = pd.read_csv('../input/train.csv')


# In[ ]:


y = df['Volume']
X = df.drop(['Volume','Date'],axis=1)


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg = LinearRegression()


# In[ ]:


reg.fit(X,y)


# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


testdf = test.drop(['Date'],axis=1)


# In[ ]:


testdf.head()


# In[ ]:


prediction = reg.predict(testdf)


# In[ ]:


serial = test['Date']
data = {'Date': serial, 'Volume': prediction}
submission = pd.DataFrame(data)
submission.to_csv('Submission.csv', index=False)

