#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/BO_2016.csv')


# In[ ]:


df[['ID_DELEGACIA','MES','RUBRICA']].head()


# In[ ]:


df[['ID_DELEGACIA','MES','RUBRICA']].describe()


# In[ ]:


sns.pairplot(data=df[['ID_DELEGACIA','MES']])


# In[ ]:


clusters = 10

kmeans = KMeans(n_clusters=clusters)
kmeans = kmeans.fit(df[['ID_DELEGACIA','MES']])
labels = kmeans.predict(df[['ID_DELEGACIA','MES']])
C_center = kmeans.cluster_centers_
print(labels,"\n",C_center)


# In[ ]:


dfGroup = pd.concat([df[['ID_DELEGACIA','MES']],pd.DataFrame(labels, columns= ['Group'])], axis=1, join='inner')
dfGroup.head()


# In[ ]:


dfGroup.groupby("Group").aggregate("mean").plot.bar()

