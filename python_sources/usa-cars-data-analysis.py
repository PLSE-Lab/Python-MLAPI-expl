#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv',index_col=0)


# # EXPLORATORY DDATA ANALYSIS AND BASIC PLOTS

# In[ ]:


data.head(10)    


# In[ ]:


data.info()


# In[ ]:


data.describe()


# Since prices cannot be zero, replacing all those cells with median of prices according to the brand.

# In[ ]:


median=data.groupby('brand')['price'].median()
def fill_median(cols):
    price=cols[0]
    brand=cols[1]
    if price==0:
        return median[brand]
    else:
        return price


# In[ ]:


data['price']=data[['price','brand']].apply(fill_median,axis=1)


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(8,5))
sns.distplot(data['price'])


# In[ ]:


data[data['price']>65000]


# Ford, dodge and mercedes-benz are among the most expensive cars.

# In[ ]:


brands= data.groupby('brand')['model'].count().sort_values(ascending = False).head(10)
brands=brands.reset_index()
fig=plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(brands['brand'],brands['model'],color='teal')
ax.set_xlabel('Brand')
ax.set_ylabel('Count')

Ford, dodge, nissan, chevrolet are the popular brands.
# In[ ]:


colors=data.groupby('color')['brand'].count().sort_values(ascending=False).head(10)
colors=colors.reset_index()
fig=plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(colors['color'],colors['brand'],color='pink')
ax.set_xlabel('Color')
ax.set_ylabel('Count')


# White and black coloured cars are preferred.

# In[ ]:


years= data.groupby('year')['model'].count().sort_values(ascending = False).head(10)
years=years.reset_index()
fig=plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(years['year'],brands['model'],color='purple')
ax.set_xlabel('Year')
ax.set_ylabel('Count')


# Car sales have increased over the years.

# In[ ]:


plt.figure(figsize=(8,6))
sns.heatmap(data.corr(),annot=True)


# Price is highly correlated with year and mileage.
# 

# In[ ]:


plt.figure(figsize=(8,5))
sns.scatterplot(x='year',y='price',data=data,hue='title_status',alpha=0.2)


# The newer the car, the higher the price.

# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='mileage',y='price',data=data,color='purple')


# Lesser the mileage more the price.
# 

# In[ ]:


sns.pairplot(data[['price','year','mileage']])
plt.tight_layout()


# # Do upvote my work if you like!

# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:




