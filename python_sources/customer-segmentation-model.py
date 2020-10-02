#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
df = pd.read_csv('../input/customer-segmentation/BoasterResponses.csv')
df.head(6)


# In[ ]:


k_elb = range(1,10)
sse = []
for k in k_elb:
    km = KMeans(n_clusters=k)
    km.fit(df[['Prefered_Preice','Rating','Age','Family_members','Frequency_of_use_Per_day']])
    sse.append(km.inertia_)


# In[ ]:


plt.plot(k_elb, sse)


# In[ ]:


km = KMeans(n_clusters=4)
km


# In[ ]:


Y=km.fit_predict(df[['Prefered_Preice','Rating','Age','Family_members','Frequency_of_use_Per_day']])
Y


# In[ ]:


df['Cluster']=Y
df.head(6)


# In[ ]:


df1 = df[df.Cluster==0]
df2 = df[df.Cluster==1]
df3 = df[df.Cluster==2]
df3 = df[df.Cluster==3]


# In[ ]:


a = df1.Survey_ID
b = df1.Survey_ID
c = df1.Survey_ID
d = df1.Survey_ID


# In[ ]:


print('Following Customer Belong to Segment No."1"', a)
print('Following Customer Belong to Segment No."2"', b)
print('Following Customer Belong to Segment No."3"', c)
print('Following Customer Belong to Segment No."4"', d)

