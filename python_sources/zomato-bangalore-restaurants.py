#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd

import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/zomato.csv')
data.tail(30)


# In[ ]:


data['name'].nunique()


# **Data Wrangling and Cleaning**

# In[ ]:


data.isnull().sum()


# In[ ]:


data.drop(columns = ['phone','location'],inplace = True)


# In[ ]:


data['rate'].value_counts()


# In[ ]:


data['rate'] = data['rate'].replace('NEW',np.NaN)
data['rate'] = data['rate'].replace('-',np.NaN)
data.dropna(how = 'any', inplace = True)


# In[ ]:


data['rate'] = data.loc[:,'rate'].replace('[ ]','',regex = True)
data['rate'] = data['rate'].astype(str)
data['rate'] = data['rate'].apply(lambda r: r.replace('/5',''))
data['rate'] = data['rate'].apply(lambda r: float(r))


# In[ ]:


print("Unique ratings : {}".format(data['rate'].nunique()))
print("Unique value counts : \n{}".format(data['rate'].value_counts()))


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dtypes


# In[ ]:


data.rename(columns = {'listed_in(type)':'service_type','listed_in(city)': 'location', 'approx_cost(for two people)':'cost_for_two'}, inplace = True)


# In[ ]:


data['cost_for_two'] = data['cost_for_two'].str.replace(',','')
data['cost_for_two'] = data['cost_for_two'].astype(int)


# **Exploratory Data Analysis (EDA)**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


data['online_order'].value_counts().plot(kind = 'bar', figsize = (10,6), color = ['r','b'], title = 'Online orders')


# In[ ]:


print("Unique restaurant type: {}".format(data['rest_type'].nunique()))
print("Top 5 restaurant types: \n{}".format(data['rest_type'].value_counts().sort_values(ascending = False).head()))


# In[ ]:


data['cuisines'].value_counts().head()


# In[ ]:


data.head()


# In[ ]:


data['service_type'].value_counts().plot(kind = 'bar', figsize = (12,6),
                                         color = ['r','b','g','c','y'],
                                         title = 'Number of restaurants by service type')


# In[ ]:


data['location'].value_counts().plot(kind = 'barh', figsize = (12,12), title = 'Number of restaurants by location')


# In[ ]:


data['cost_for_two'].value_counts().plot(kind = 'hist', figsize = (10,5))


# In[ ]:


data_1 = data.loc[(data.book_table == 'Yes') & (data.rate > 4.0), ['name', 'rate','cost_for_two', 'address']]
data_1.head()


# In[ ]:


data_1.groupby(['rate'])['cost_for_two'].agg(['mean','median']).plot(kind = 'bar',
                                                                                figsize = (12,6),
                                                                                color = ['r','g'],
                                                                                title = 'Mean and median cost for two for top rated restaurants')


# In[ ]:


top_rated_cheap = data_1.loc[(data_1.rate > 4.5) & (data_1.cost_for_two < 1000), ['name','rate', 'cost_for_two']]
top_rated_cheap


# In[ ]:


top_rated_cheap.name.unique()


# **Onesta**,** Baar Union** and **Zamindar** are the three top rated (rating > 4.5) and cheapest resaturants (cost for two people  of 600,850 and 800)

# This was an introductory kernel that would help one find cheaper and high rated restaurants in Bangalore. 
# Any comments or suggestions regarding the analysis and code is welcome.
# Please do upvote if you like the kernel.
# Happy Learning!
