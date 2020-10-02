#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORITNG ALL THE ESSENTIAL PACKAGES
import pandas as pd
import numpy as np
import statsmodels.api as sns
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


raw_data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

raw_data.shape


# # Preprocessing

# In[ ]:


raw_data.columns


# In[ ]:


host_id=raw_data['host_id']
host_name=raw_data['host_name']


# In[ ]:


host_id.value_counts()


# In[ ]:


host_name.value_counts()


# In[ ]:


data=raw_data[['id','name','host_id','neighbourhood_group','room_type','price','number_of_reviews']]
data.head()


# # Analysis 
# By plotting a sctter plot we can get a rough idea of realtion betweeen number of reviews and price
# 

# In[ ]:


plt.scatter(data['number_of_reviews'],data['price'],alpha=0.2)
plt.xlabel('NUMBER OF REVIEWS')
plt.ylabel('PRICE')
plt.title('RELATIONSHIP BETWEEN PRICE AND NUMBER OF REVIEWS')
plt.show()


# # Some more preprocessing
# 

# In[ ]:


np.mean(data['price'])


# In[ ]:


np.std(data['price'])


# In[ ]:


sns.distplot(data['price'],bins=200,color='red')


# In[ ]:


data.shape


# In[ ]:


data1=data[data['price']<=300]
data1.shape


# In[ ]:


sns.distplot(data1['price'],bins=200,color='red')


# In[ ]:


data1['price'].mean()


# In[ ]:


data1['price'].std()


# # Useful visualization

# In[ ]:


count_each=np.asarray(np.unique(data1['neighbourhood_group'],return_counts=1))

plt.bar(count_each[0],count_each[1],color='red')
plt.xlabel('NEIGBOURS')
plt.ylabel('COUNT')
plt.title('NO OF HOTELS NEAR TO DIFFERENT LOCALITIES')
plt.show()


# In[ ]:


count_each_room=np.asarray(np.unique(data1['room_type'],return_counts=1))
plt.bar(count_each_room[0],count_each_room[1],color='red')
plt.xlabel('ROOM TYPES')
plt.ylabel('COUNT')
plt.title('DIFFERENT ROOM TYPES AND THEIR FREQUENCY')
plt.show()


# 

# In[ ]:


sns.violinplot(x='neighbourhood_group',y='price',data=data1,inner=None)


# In[ ]:


sns.boxplot(x='neighbourhood_group',y='price',data=data1,whis=10)


# In[ ]:


sns.violinplot(x='room_type',y='price',data=data1,inner=None)


# In[ ]:




