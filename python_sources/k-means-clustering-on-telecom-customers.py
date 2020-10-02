#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering on Telecom Customers

# **This notebook is for basic understanding of K-means clustering on real telecom industry datasent. Although the dataset is very low but the primary purpose is to understand and work with the real industry data.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')


# Let's load the dataset

# In[ ]:


df=pd.read_excel('/kaggle/input/social-profile-of-customers/Social profile of customers_without header.xlsx')
df.head()


# In[ ]:


df.columns=[c.replace(' ','_') for c in df.columns]
df.head(1)


# In[ ]:


df.drop(columns=['Name_','Profile','Profession_','Location','Prority__level_1','Priority_level_2','Priority_level_3'],inplace=True)
df.head(2)


# In[ ]:


df.info()


# In[ ]:


df['Type_of_Location_'][0]=1
df['Type_of_Location_']=pd.to_numeric(df['Type_of_Location_'])
df.info()


# Let's build the model using k-means clustering (k=3)

# In[ ]:


from sklearn.cluster import KMeans
k_means=KMeans(n_clusters=3).fit(df)
df1=pd.read_excel('/kaggle/input/social-profile-of-customers/Social profile of customers_without header.xlsx')
clustering=pd.DataFrame({'Name':df1['Name '],'Clusters':k_means.labels_})
clustering.to_csv('/kaggle/working/clustering_output.csv', index=False)
cl=pd.read_csv('/kaggle/working/clustering_output.csv')
cl.head()


# **Please upvote if you like or find this notebook useful, Thanks.**
