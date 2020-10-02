#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy  as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans





# In[ ]:


dataframe=pd.read_csv('../input/Iris.csv',index_col=0)


# In[ ]:


sb.pairplot(dataframe)


# In[ ]:


x = np.array(dataframe.drop('Species',axis = 1))
x=np.round(dataframe.drop('Species',axis = 1),2)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(x)
kmeans.labels_
dataframe['classes'] = kmeans.labels_


# In[ ]:


sb.pairplot(dataframe,vars=dataframe.columns[:4],hue="classes")

