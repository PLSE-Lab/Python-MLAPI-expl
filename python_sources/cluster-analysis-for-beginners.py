#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


train=pd.read_csv("../input/train.csv")


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.shape


# In[ ]:


train["Age"].mean()


# In[ ]:


train.drop("Cabin", axis=1,inplace=True)


# In[ ]:


train.describe()


# In[ ]:


train.mean()


# In[ ]:


train.fillna(round(train.mean(),2), inplace=True)


# In[ ]:


train.head(10)


# In[ ]:


train.dropna(axis=0,inplace=True) 


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


train_numeric=train.select_dtypes(exclude=['object'])


# In[ ]:


train_numeric.drop("Survived",axis=1,inplace=True)


# In[ ]:


kmeans = KMeans(n_clusters=2)


# In[ ]:


kmeans.fit(train_numeric)


# In[ ]:


train["cluster"]=kmeans.predict(train_numeric)


# In[ ]:


train.head()


# In[ ]:


test=pd.read_csv("../input/test.csv")


# In[ ]:


test.head()


# In[ ]:


test.info()


# In[ ]:


test.drop("Cabin", axis=1,inplace=True)


# In[ ]:


test.fillna(round(test.mean(),2), inplace=True)


# In[ ]:


test.info()


# In[ ]:


test_numeric=test.select_dtypes(exclude=['object'])


# In[ ]:


test_numeric.info()


# In[ ]:


test["cluster"]=kmeans.predict(test_numeric)

