#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pa # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pa.read_csv('../input/Mall_Customers.csv')


# In[ ]:


df['Gender_code']=df['Gender'].apply(lambda a:1 if a=='Male' else 0) 


# In[ ]:


df2= df[['Age','Annual Income (k$)','Spending Score (1-100)','Gender_code']]


# In[ ]:


df2.isnull().values.any()
df2.head()


# In[ ]:


import seaborn as sns
sns.lineplot(data=df2,x ='Age',y='Annual Income (k$)')
sns.lineplot(data=df2,x ='Age',y='Spending Score (1-100)')
#sns.scatterplot(data=df3)


# In[ ]:


sns.scatterplot(data=df2,y='Spending Score (1-100)',x='Annual Income (k$)',hue='Gender_code')


# from the above plot we can see there are a group of people those are having less income but they are spending alot likewise there are people who are having high income and they are sending alot and we can see if a person is earning from 40K to 65k these people are nither spending much nor less. so according to there clusterd they can advertice the sales or new products to a particuler set of peoples only that will reduce their advertisement cost and will help them to attract similer type customer. 

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


df3=df2[['Annual Income (k$)', 'Spending Score (1-100)', 'Gender_code']]


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mms= MinMaxScaler()
df3=mms.fit_transform(df3)


# In[ ]:



inertia=[]
for i in range(2,10):
    model = KMeans(n_clusters=i)
    model.fit(df2)
    inertia.append(model.inertia_)
    


# In[ ]:


import matplotlib.pyplot as plt 
plt.figure(figsize=(16,6))
plt.plot(np.arange(2,10),inertia)


# In[ ]:


model = KMeans(n_clusters=5)
model.fit(df2)


# In[ ]:


print(model.labels_)
print(model.cluster_centers_)


# In[ ]:




