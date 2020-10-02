#!/usr/bin/env python
# coding: utf-8

# importing libraries

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# **

# In[ ]:


data = pd.read_csv("../input/listings.csv")


# In[ ]:


data.head(15) #first 15 datas


# In[ ]:


f,ax = plt.subplots(figsize=(9,9))
sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt ='.2f',ax = ax)


# room types in the project

# In[ ]:


room_types = data.groupby(['room_type'])
for name, rdata in room_types:
    print(name)


# In[ ]:


room_price = room_types['price'].agg(np.mean)


# In[ ]:


print(room_price)


# show room types of prices mean

# In[ ]:


room_types['price'].agg(np.mean).plot(kind='bar')
plt.show()


# In[ ]:


neighbourhood = data.groupby(['neighbourhood'])
#for i,y in neighbourhood : 
#    print(i)


# In[ ]:


data.groupby(['neighbourhood'])['price'].agg(['mean','count'])


# In[ ]:


data.groupby(['neighbourhood','room_type'])['price'].agg(['mean'])


# In[ ]:


data.groupby(['neighbourhood','room_type'])['price'].agg(['mean', 'count'])


# In[ ]:


plt.rcParams["figure.figsize"] = [20, 20]
data.groupby(['neighbourhood','room_type'])['price'].agg(['mean', 'count']).plot.bar(stacked=True)

