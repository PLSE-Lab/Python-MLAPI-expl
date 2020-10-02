#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[26]:


base_dataset= pd.read_csv("../input/application_train.csv")


# In[27]:


base_dataset.head()


# In[22]:


### This is wrong base_dataset_var=base_dataset.var().sort_values(ascending=False).index[0:5]


# In[28]:


base_dataset_var=base_dataset[base_dataset.var().sort_values(ascending=False).index[0:5]]


# In[29]:


base_dataset_var.head()


# In[ ]:


### this is not requirede base_dataset_var=base_dataset[['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED','SK_ID_CURR']]


# In[25]:


base_dataset_var


# In[30]:


new_base_dataset=base_dataset_var.drop('SK_ID_CURR', axis=1)
##base_dataset_var.drop('SK_ID_CURR', axis=1,inplace=True) we can write in this way


# In[31]:


new_base_dataset.head()


# In[32]:


### Remove null values###

for i in new_base_dataset.columns:
    new_base_dataset[i].fillna(new_base_dataset[i].median(),inplace=True)


# In[33]:


###Normalization###
from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler()
mn.fit(new_base_dataset)
x=mn.transform(new_base_dataset)


# In[34]:


x


# In[35]:


x=pd.DataFrame(x)


# In[37]:


x.head()


# In[38]:


new_base_dataset=x


# In[39]:


new_base_dataset.head()


# In[40]:


### K Elbow Method ###

from sklearn.cluster import KMeans
x1 = []
for i in range(1,20):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_base_dataset)
    x1.append(kmeans.inertia_) 


# In[41]:


x1


# In[42]:


from sklearn.cluster import KMeans
km=KMeans(n_clusters=4)
x=km.fit(new_base_dataset)


# In[43]:


len(x.labels_)


# In[45]:


new_base_dataset1=new_base_dataset


# In[ ]:


new_base_dataset.head()


# In[46]:


new_base_dataset1=base_dataset[['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED']]


# In[47]:


new_base_dataset1.head()


# In[48]:


new_base_dataset1['cluster']=x.labels_


# In[49]:


new_base_dataset1.head()


# In[50]:


new_base_dataset1['cluster'].value_counts()


# In[ ]:


Test


# In[ ]:


pd.DataFrame(Test)


# In[ ]:


#### Interpretations ###


# In[ ]:


## Findout better performing clusters 


# In[ ]:


#### Agglomerative ###


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
Ag=AgglomerativeClustering(n_clusters=4)
Ag.fit(new_base_dataset)
Ag.labels_


# In[ ]:


new_base_dataset.head()


# In[ ]:


new_base_dataset.shape

