#!/usr/bin/env python
# coding: utf-8

# # **Introduction**

# This open dataset describes the listing activity and metrics in NYC. In this kernel, I have tried to gain insights about the price of various listings with respect to different features by visualizing data.
# 
# I hope you like this kernel and would appreciate your upvotes :)

# # **Importing Packages and Data**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import category_scatter
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("whitegrid")


# In[ ]:


df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# # **EDA and Data Visualization**

#  We can already see sum null values in the data.Let's get some more information about the data

# In[ ]:


df.info()


# Checking the number of null values in each column and printing in descending order
# 

# In[ ]:


df.isnull().sum().sort_values(ascending = False)


# As we will be exploring the features that affect the price of the listing, columns like name are not relevant for our analysis.

# In[ ]:


df.drop(['host_name','last_review','name'],axis = 1, inplace = True)


# All of the null values in the 'reviews_per_month' column corresponds to the listings where 'number_of_reviews' is equal to zero. Hence we go ahead and fill those values with 0.

# In[ ]:


df[df['number_of_reviews']==0].shape


# In[ ]:


df['reviews_per_month'].fillna(0, inplace = True)


# Checking if there are any other null values left out.

# In[ ]:


df.isnull().sum().sort_values(ascending = False)


# Checking the various categories which gives us an idea of how to go about with the analysis

# In[ ]:


df['neighbourhood_group'].unique()


# In[ ]:


df['room_type'].unique()


# Grouping the neighbourhoods and having a look at the price distributions

# In[ ]:


n_group = df.groupby('neighbourhood_group').describe()
n_group.xs('price',axis = 1)


# The price ranges from as low as 10 to as high as 10000 although the mean price remains below 200. 

# In[ ]:


sns.catplot(x = 'neighbourhood_group', y = 'price', data = df)


# The plot shows that although there are some high values, most of the data lies below 1000.

# In[ ]:


df1 =df[df['price']<500]
plt.figure(figsize = (10,5))
sns.violinplot(x = 'neighbourhood_group', y = 'price', data = df1, scale = 'count', linewidth = 0.3)


# This violin plot is a depiction of the density and distribution of price in different neighbourhoods. The width of the violins is scaled by the number of observations in that bin. Both Brooklyn and Manhattan has many listings around the price 100 but Manhattan has more listings with a greater price.

# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(df['neighbourhood_group'],hue=df['room_type'])


# From the plot, we can tell that Brooklyn and Manhattan are the most popular neighbourhoods with greater number of listings. Brooklyn has almost equal number of 'private room' and 'entire home' whereas Manhattan shows greater number of homes/apartments. However, when it comes to less popular areas such as Staten Island and Bronx, there's no significant difference between them.

# In[ ]:


plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='room_type', y='price')


# As expected, homes/apartments are more expensive than private room/shared room.

# In[ ]:


df2= df1.groupby('neighbourhood_group')
fig, ax = plt.subplots(2,3,figsize =(20,10))
ax = ax.flatten().T
sns.scatterplot('number_of_reviews','price',data = df2.get_group('Brooklyn'),ax = ax[0], label ='Brooklyn')
sns.scatterplot('number_of_reviews','price',data = df2.get_group('Manhattan'),ax = ax[1],color = 'orange',label ='Manhattan')
sns.scatterplot('number_of_reviews','price',data = df2.get_group('Bronx'),ax = ax[2],color = 'purple',label ='Bronx')
sns.scatterplot('number_of_reviews','price',data = df2.get_group('Queens'),ax = ax[3],color = 'g',label ='Queens')
sns.scatterplot('number_of_reviews','price',data = df2.get_group('Staten Island'),ax = ax[4],color = 'r',label ='Staten Island')


# The number of reviews shows no correlation with the price for any neighbourhood

# In[ ]:


plt.figure(figsize=(8, 6))
sns.countplot('minimum_nights', data = df1)
plt.xlim(0, 40)
tick = [1,5,10,15,20,25,30,35,40]
plt.xticks(tick, tick)


# Most of the listings for minimum nights are less 5 days whereas some guests stay for about a month.

# In[ ]:


plt.figure(figsize= (10,8))
plt.scatter(df1.longitude, df1.latitude, c = df1.price, alpha = 0.7, cmap ='jet',edgecolor = 'black')
cbar = plt.colorbar()
cbar.set_label('Price')


# As seen earlier, listings in manhattan are comparatively more expensive than others.

# In[ ]:


plt.figure(figsize= (10,8))
plt.hist2d(df1.longitude, df1.latitude, bins=(100,100),cmap =plt.cm.jet)
c_bar = plt.colorbar()
c_bar.set_label('Density')


# The second plot gives us an idea about the density of the points according to the location. It can be seen that some areas in Manhatten are most dense followed by the one's in Brooklyn

# In[ ]:


plt.figure(figsize= (10,8))
plt.scatter(df1.longitude, df1.latitude, c = df1.availability_365, alpha = 0.7,cmap ='summer',edgecolor = 'black')
c_bar = plt.colorbar()
c_bar.set_label('Availability')


# It can be seen that availability is almost evenly distributed throughout

# In[ ]:




