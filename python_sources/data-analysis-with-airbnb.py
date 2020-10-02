#!/usr/bin/env python
# coding: utf-8

# Hello,My English is pretty limited,so my grammar have some wrong,sorry for the inconvenience caused. I will use airbnb(NewYork) open data to run and Observe.
# 

# **#import  Libraries**

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# **#read data**

# In[ ]:


df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# In[ ]:


df.head()


# **#check sum(NAN) **

# In[ ]:


df.isnull().sum()


# **#check data information**

# In[ ]:


df.info()


# #Replacing null values in the column  with 0 in the dataset

# In[ ]:


dff=df.fillna(0)


# **#check variable**

# In[ ]:


df.neighbourhood_group.unique()


# In[ ]:


df.neighbourhood.unique()


# **#run crosstab to observe**

# In[ ]:


pd.crosstab([df.neighbourhood_group, df.room_type], df.calculated_host_listings_count, margins=True) 


# ****#first crosstab show (calculated_host_listings_count = 1)is the most   (32303/48895)=  0.66066                                                 ******

# In[ ]:


aviail = pd.crosstab([df.neighbourhood_group, df.room_type], df.availability_365, margins=True)


# In[ ]:


aviail


# **second crosstab (availability_365 = 0) is the most (17533/48895 = 0.35858****)**
# 

# **#pivot table**

# In[ ]:


df.pivot_table('minimum_nights', index='neighbourhood_group', columns='room_type', aggfunc='sum', margins=True)


# In[ ]:


df.pivot_table('calculated_host_listings_count', index='neighbourhood_group', columns='room_type', aggfunc='count', margins=True)


# In[ ]:


df.pivot_table('availability_365', index='neighbourhood_group', columns='room_type', aggfunc='count', margins=True)


# In[ ]:


mean_nei = df.pivot_table('price', index='neighbourhood_group', columns='room_type', aggfunc='count', margins=True)


# In[ ]:


mean_nei


# In[ ]:


sum_ratings = df.pivot_table('price', index='neighbourhood_group', columns='room_type', aggfunc='mean', margins=True)


# In[ ]:


sum_ratings


# In[ ]:


room_typesum = df.groupby('room_type').size()


# In[ ]:


room_typesum


# In[ ]:


neighbourhood_groupsum = df.groupby('neighbourhood_group').size()


# In[ ]:


neighbourhood_groupsum


# In[ ]:


minimum_nights_mean = df.pivot_table('price', index='minimum_nights', columns='room_type', aggfunc='mean')


# In[ ]:


sns.set(rc={'figure.figsize':(10,8)})


# In[ ]:


viz_1=room_typesum.plot(kind='bar')
viz_1.set_title('room type')
viz_1.set_ylabel('Count of listings')
viz_1.set_xlabel('Host type')
viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=45)


# # first bar plot show entire home/apt and Private room is optimal selection

# In[ ]:


viz_1=neighbourhood_groupsum.plot(kind='bar')
viz_1.set_title('room type')
viz_1.set_ylabel('Count of listings')
viz_1.set_xlabel('Host type')
viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=45)


# # Second bar plot show Manhattan and Brooklyn is the best choice

# In[ ]:



sns.catplot(x="room_type", y="price", data=df);


# first catplot show budget price and entire home/apt gain favor

# In[ ]:


sns.catplot(x="neighbourhood_group", y="price", data=df);


# second catplot show bodget price and Manhattan is gain favor with passenger

# In[ ]:


ax = sns.countplot(x="neighbourhood_group", hue="room_type", data=df)


# countplot show obviously Manhattan and Brooklyn is the most

# **#check correlation**

# In[ ]:


plt.figure(figsize=(10,6)) #manage the size of the plot
sns.heatmap(df.corr(),annot=True) #df.corr() makes a correlation matrix and sns.heatmap is used to show the correlations heatmap
plt.show()


# **#check scatterplot **

# In[ ]:


ax = sns.scatterplot(x="latitude", y="longitude", hue="room_type",data=df)
                    


# In[ ]:


ax = sns.scatterplot(x="latitude", y="longitude", hue="availability_365",data=df)


# **#cut when you need to segment and sort data values into bins**

# In[ ]:


bins = np.array([0,30,60,90,120,150,180,210,240,270,300])


# In[ ]:


labels = pd.cut(df.price, bins)


# In[ ]:


labels


# In[ ]:


grouped = df.groupby(['room_type', labels])


# In[ ]:


rl = grouped.size().unstack(0)


# In[ ]:


rl


# In[ ]:


rl.plot(kind='barh')


# In[ ]:


groupednei = df.groupby(['neighbourhood_group', labels])


# In[ ]:


nl = groupednei.size().unstack(0)


# In[ ]:


nl


# In[ ]:


nl.plot(kind='barh')


# **conclusion******

# Most people prefer to private space like entire home/apt ,Manhattan and Brooklyn are gaining favor with passenger,
