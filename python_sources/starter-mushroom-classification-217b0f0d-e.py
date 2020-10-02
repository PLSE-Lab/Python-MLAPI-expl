#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd


# In[ ]:


os.listdir("../input")


# In[ ]:


df = pd.read_csv("../input/mushrooms.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


from sklearn import preprocessing


# In[ ]:


enc = preprocessing.OrdinalEncoder()
enc.fit(df)


# In[ ]:


tf = enc.transform(df)


# In[ ]:


type(tf)


# In[ ]:


x = tf[:,1:]


# In[ ]:


y = tf[:,0]


# In[ ]:


y


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


c = LinearRegression()


# In[ ]:


c.fit(x,y)


# In[ ]:


c.score(x,y)


# In[ ]:


from sklearn import svm
c2 = svm.SVC()
c2.fit(x,y)
c2.score(x,y)


# In[ ]:




