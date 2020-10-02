#!/usr/bin/env python
# coding: utf-8

# Just a Simple Attempt to the unsupervised learning and finding the solution for the problem statement 
# 
# 
# Do upvote if this notebook was helpful!!!!

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


# importing the necessary library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


# In[ ]:


# reading the data
data = pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')
df = pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')
# df = 


# In[ ]:


# dropping the column
df.drop(columns = 'health',inplace= True)


# In[ ]:


# first 5 rows of the dataset
data.head()


# In[ ]:


# check for the null values
sns.heatmap(data.isnull())


# In[ ]:


# looking for the realtions of columns with each other
sns.heatmap(data.corr(),annot = True)


# In[ ]:


# dropping the column
data.drop(columns = 'health',inplace=True)
data.drop(columns = 'country',inplace=True)


# In[ ]:


# seeing the income and gdp distribution

sns.scatterplot(x = 'income',y = 'gdpp',data=data)


# In[ ]:


#  seeing how import and exports are related to gdp
sns.scatterplot(x = 'imports',y = 'exports',hue = 'gdpp',data=data)


# In[ ]:


# histrogram based on the child mortalitry rate per 1000 capita
sns.distplot(data['child_mort'],bins = 10,kde= False)


# In[ ]:


data.describe()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# to scale the data
scalar = MinMaxScaler()
data = scalar.fit_transform(data)


# In[ ]:


df = pd.DataFrame(data = data,columns=df.columns[1:])


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


# to get the sum of distance
clf = KMeans()
ssd = []
K = range(1,9)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data)
    ssd.append(km.inertia_) 


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(K, ssd, 'bx-')
plt.xlabel('Clusters')
plt.ylabel('Distance')
plt.title('Elbow Method For Optimization')
plt.show()


# In[ ]:


# dividing the the dataset into clusters of 5
kmean = KMeans(n_clusters=5)
kmean.fit(data)


# In[ ]:


# distributed labels
pred = kmean.labels_
print(pred)


# In[ ]:


df1 = pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')


# In[ ]:


# gdp and income based on the clusters
sns.scatterplot(data= df1,x = 'gdpp',y = 'income',hue=kmean.labels_)


# In[ ]:


''' list of countries which require utmost need for the money based on 
the income less than 1000 noticed from the above diagram'''
df1['country'][df1['income']<1000]


# In[ ]:




