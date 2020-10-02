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


import numpy as np


# In[ ]:


#creation/ initialization

#list

a = np.array([1,2,3,4,5,6])

#all zeros

b = np.zeros(10)

#all 5's matrix of size 4X4

c = np.full((4,4),5)


# In[ ]:


a


# In[ ]:


b


# In[ ]:


c


# In[ ]:


#creating sequence

a =  np.arange(10,100,5)

#equal spaced 10 numbers between 1 & 2

b = np.linspace(start=1, stop=2,num=10)

#random whole numbers between 5 and 10

c = np.random.randint(5,10,size=(3,4))

#random 10 numbers between 0 and 1

d = np.random.rand(10)


# In[ ]:


d


# In[ ]:


#access numpy array

h = np.random.randint(5,10,size=(5,5))
h


# In[ ]:


#2nd row onwards rest of all
h[2:]


# In[ ]:


h[1:4]


# In[ ]:


h[:,1:3]


# In[ ]:


#shape & reshape

i = np.random.randint(1,5,size=(4,5))


# In[ ]:


i


# In[ ]:


i.shape


# In[ ]:


i.size


# In[ ]:


i.ndim


# In[ ]:


j = i.reshape(2,10)


# In[ ]:


j


# In[ ]:


j.shape


# In[ ]:


k = j.ravel()


# In[ ]:


k


# In[ ]:


k.shape


# In[ ]:


m =  np.arange(5)


# In[ ]:


m


# In[ ]:


np.sum(m)


# In[ ]:


#pandas basics


# In[ ]:


import pandas as pd


# In[ ]:


s1 = pd.Series(data=[1,2,3,4], index=["a","b","c","d"])


# In[ ]:


s1


# In[ ]:


s1["a"]


# In[ ]:


s2 = pd.Series(data=[11,12,13,14],index=["a","b","c","d"])


# In[ ]:


s2


# In[ ]:


df1 = pd.DataFrame({'col1':s1,'col2':s2})


# In[ ]:


df1


# In[ ]:




