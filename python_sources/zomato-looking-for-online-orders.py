#!/usr/bin/env python
# coding: utf-8

# ## **Objective**
# 
# 1. Loading the Dataset
# 2. Data Cleansing
# 3. Analysis on Online Order
# 4. Finding the best restaurants 

# In[ ]:


import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings as war
war.filterwarnings('ignore')

import os
print(os.listdir("../input/zomato-bangalore-restaurants/"))


# In[ ]:


data=pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


print("Percentage of null values in df-")
(data.isnull().sum()*100/data.index.size).round(2)


# In[ ]:


data['rate']=data['rate'].replace('NEW',np.nan)
data['rate'].unique()


# In[ ]:


data.dropna(how='any',inplace=True)


# In[ ]:


data['rate']=data['rate'].apply(lambda x: x.split('/')[0])


# In[ ]:


data.head(2)


# In[ ]:


del data['url']
del data['address']
del data['location']
del data['phone']
del data['menu_item']


# In[ ]:


data.head()


# In[ ]:


data.rename(columns={'listed_in(type)': 'Restaurant_type','listed_in(city)':'location','approx_cost(for two people)':'average_cost'},inplace=True)


# In[ ]:


data.head()


# In[ ]:


data['average_cost']=data['average_cost'].apply(lambda x: x.replace(',',''))
data['average_cost']=data['average_cost'].astype(int)


# In[ ]:


data['average_cost'].dtype


# In[ ]:


data.head()


# In[ ]:


print(data['Restaurant_type'].unique())


# In[ ]:


data['rate']=data['rate'].astype('float')


# ## Restaurant who prefers online orders..

# In[ ]:


sns.countplot(data['online_order'])


# In[ ]:


data['online_order'].value_counts()


# There are 6749 restaurants who are listed on zomato but doesn't take online orders.

# In[ ]:


plt.figure(figsize=(15,7))
g=sns.countplot(data['location'],palette='Set1',hue=data['online_order'])
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha='right')
g


# In[ ]:


data[data['online_order']=='Yes']['online_order'].count()


# In[ ]:


New_df=data[data['online_order']=='Yes'].sort_values(by='votes',ascending=False)
#New_df.groupby('name').first()


# In[ ]:


data['Restaurant_type'].value_counts()


# In[ ]:


data[data['online_order']=='Yes']['Restaurant_type'].value_counts()


# In[ ]:


print('No of Restaurant doesnot accept Online Orders are: ',10575-9179)


# In[ ]:


data[data['Restaurant_type']=='Delivery']['online_order'].value_counts()


# In[ ]:


Online_No=data[(data['Restaurant_type']=='Delivery') & (data['online_order']=='No')]
Online_No.groupby('name').first()


# In[ ]:


Online_No['name'].value_counts()


# In[ ]:


Online_No[Online_No['book_table']=='No']['name'].value_counts()


# ## Finding the best Resturent

# In[ ]:


Top1000_byrate=data.sort_values(by='rate',ascending=False).head(1000)
print('Top 20 Restaurants with the highest rates')
Top1000_byrate[['name','rate','votes','rest_type','average_cost','location']].head(20)


# **Observation:** We can clearly see that the Asia Kitchen By Mainland China has multiple occurances.

# In[ ]:


Economical_Restaurents=Top1000_byrate.sort_values(by='average_cost')
Economical_Restaurents.head()


# In[ ]:


Economical_Restaurents['cuisines'].value_counts()


# In[ ]:


Economical_Restaurents[Economical_Restaurents['cuisines']=='Desserts']


# In[ ]:


Economical_Restaurents[Economical_Restaurents['cuisines']=='Desserts']['name'].unique()


# In[ ]:


Economical_Restaurents[Economical_Restaurents['cuisines']=='Desserts'].sort_values(by='votes',ascending=False).head(2)


# **Observation:** The Belgian Waffle factory of MG Road and BelgYum of Whitefield is the most popular and ecomonical dessert parlour.

# In[ ]:


Top1000_byprice=data.sort_values(by='average_cost',ascending=False).head(1000)


# In[ ]:


print ('Top 20 most expensive Restaurent')
Top1000_byprice[['name','rate','votes','rest_type','average_cost','location']].head(20)


# In[ ]:


Best_and_expensive=Top1000_byprice.sort_values(by='rate',ascending=False)
Best_and_expensive.head()

