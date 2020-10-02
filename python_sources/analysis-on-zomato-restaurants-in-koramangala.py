#!/usr/bin/env python
# coding: utf-8

# <h3> Zomato Restaurants Data Exploration and Analysis </h3>

# In[ ]:


import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np


# In[ ]:


PATH = '../input/zomato-bangalore-restaurants/'
df = pd.read_csv(f"{PATH}/zomato.csv")


# In[ ]:


df.head(5)


# In[ ]:


#checking all the columns of the data
df.columns


# In[ ]:


#finding all the restaurants in Koramangala 
df_koramangala = df[df['location'] ==                      'Koramangala']


# In[ ]:


df_koramangala.columns


# In[ ]:


popular_df = df_koramangala[['name','rate', 'approx_cost(for two people)', 'online_order', 'votes', 'cuisines']]


# In[ ]:


popular_df['rating'] = popular_df.rate.astype('str')
popular_df['rating'] = popular_df['rating'].apply(lambda x : float(x.split('/')[0]) if x!= '-' else 0.0)
popular_df = popular_df.fillna(0)


# In[ ]:


popular_df.head(5)


# In[ ]:


#Number of outlets
pop_counts = popular_df.name.value_counts()
pop_counts


# In[ ]:


sns.barplot(x = pop_counts, y = pop_counts.index)


# In[ ]:





# In[ ]:


#checking the most popular cuisines in Koramangala
plt.figure(figsize = (8,8))
cuisines = df_koramangala['cuisines'].value_counts()[:10]
sns.barplot(cuisines, cuisines.index)
plt.xlabel('Count')
plt.ylabel('Cuisine')


# In[ ]:


df_koramangala[df['name'] == 'Hunger Hitman'].iloc[1][0]

