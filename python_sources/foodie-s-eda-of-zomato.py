#!/usr/bin/env python
# coding: utf-8

# Hello There ! I am a foodie and so, this dataset is one of my favourites :D. The EDA is still in progress and so I'll be updating the kernel . **Please upvote it if you liked it. Your upvotes will motivate me to code more**. **New suggestions are always welcomed**. Please mention them in comments. Thank you for visiting my kernel :D

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# In[ ]:


data=pd.read_csv('../input/zomato.csv')
data.head()


# In[ ]:


data=data.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',
                         'listed_in(city)':'city'})


# In[ ]:


data.info()


# In[ ]:


round((data.isnull().sum()/data.shape[0])*100,2)


# In[ ]:


data.describe()


# In[ ]:


url=data.pop('url')
address=data.pop('address')
phone=data.pop('phone')
menu_item=data.pop('menu_item')
reviews_list=data.pop('reviews_list')
type_hotel=data.pop('type')


# In[ ]:


data['online_order']=data['online_order'].apply(lambda x: '1' if str(x)=='Yes' else '0')
data['book_table']=data['book_table'].apply(lambda x: '1' if str(x)=='Yes' else '0')
data['rate']=data['rate'].apply(lambda x: str(x).split('/')[0])
data['cost']=data['cost'].apply(lambda x: str(x).replace(',',''))
data.dropna(subset=['rate','cost'])
data=data[data['rate']!='nan']
data=data[data['rate']!='NEW']
data=data[data['rate']!='-']
data=data[data['cost']!='nan']
data['rate']=data['rate'].astype(float)
data['votes']=data['votes'].astype(int)
data['cost']=data['cost'].astype(int)


# In[ ]:


plt.subplots(3,2,figsize=(10,10))
plt.subplot(3,2,1)
sns.countplot(data['online_order'])
plt.subplot(3,2,2)
sns.countplot(data['book_table'])
plt.subplot(3,2,3)
sns.distplot(data['rate'],kde=True)
plt.subplot(3,2,4)
sns.distplot(data['votes'],kde=True)
plt.subplot(3,2,5)
sns.distplot(data['cost'])

plt.tight_layout()


# In[ ]:


data.sample(5)


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(data['rate'])


# Looks like there are many duplicates. The location as well as city is same and so they are the same restaurants. Let us see the counts of each restaurant. 

# In[ ]:


data[data['rate']==4.9][:5]


# In[ ]:


data[data['rate']==1.8][:5]


# In[ ]:


hotel_counts=data['name'].value_counts()
unique_hotels=data['name'].unique()


# In[ ]:


hotel_counts


# In[ ]:


data[data['name']=='Onesta'].head(10)


# In[ ]:


data.drop_duplicates(keep='first',inplace=True)


# In[ ]:


data[data['name']=='KFC'].head(10)


# In[ ]:


data['name'].value_counts()


# In[ ]:


plt.subplots(3,2,figsize=(10,10))
plt.subplot(3,2,1)
sns.countplot(data['online_order'])
plt.subplot(3,2,2)
sns.countplot(data['book_table'])
plt.subplot(3,2,3)
sns.distplot(data['rate'],kde=True)
plt.subplot(3,2,4)
sns.distplot(data['votes'],kde=True)
plt.subplot(3,2,5)
sns.distplot(data['cost'])

plt.tight_layout()


# Hotels having large count of online order must be outlets of KFC, McDonalds,etc. 

# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(data['rate'],hue='online_order',data=data)


# We see that only the hotels with high ratings have table booking facility. This seems fairly logical based on the demand they might be having. 

# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(data['rate'],hue='book_table',data=data)


# Let us check average rating for hotels which provide delivery vs those who dont and which have booking facility vs those who dont.

# In[ ]:


plt.figure(figsize=(6,6))
data.groupby('online_order')['rate'].mean().plot.bar()
plt.ylabel('Average rating')


# There is not much of difference between the average ratings of hotels which provide online_order facilities and which dont. But the hotels which have table booking facility certainly have significant amount of greater rating compared to those who dont provide table booking facility. 

# In[ ]:


plt.figure(figsize=(6,6))
data.groupby('book_table')['rate'].mean().plot.bar()
plt.ylabel('Average rating')


# Clearly, hotels which have high ratings have more number of votes.
# From the hex plot we see that most restaurants have cost for two under 1000 rupees.
# Restaurants with rating between 4 and 4.5 have very high cost. 
# 

# In[ ]:



sns.lmplot(x='votes',y='rate',data=data)

sns.jointplot(x='votes',y='rate',data=data,kind='hex',gridsize=15,color='orange')

sns.lmplot(x='cost',y='rate',data=data)

sns.jointplot(x='cost',y='rate',data=data,color='red',kind='hex',gridsize=15)


# Let us see which location has the highest average rating. It seems Lavelle Road is the best location for foodies in Banglore. 

# In[ ]:


plt.figure(figsize=(10,6))
data.groupby('location')['rate'].mean().sort_values(ascending=False)[:10].plot.bar()


# Checking the variation in rating of the top 5 locations. It seems that Lavelle Road has large variation in ratings.

# In[ ]:


top_locations=data[data['location'].isin(['Lavelle Road','Koramangala 3rd Block','Koramangala 5th block',
                                         'St. Marks Road','Sankey Road'])]
sns.violinplot(x='location',y='rate',data=top_locations)
plt.xticks(rotation=90)


# MG road can be seen in almost every big city. The area around that road is good for shopping and best for foodies in almost every city. Same is true for Banglore as well. 

# In[ ]:


plt.figure(figsize=(10,6))
data.groupby('city')['rate'].mean().sort_values(ascending=False)[:10].plot.bar()


# Let us see which restaurant type has highest average rating. Pubs, Bars and Cafes seem to rule Banglore!

# Each city has lot of variation .

# In[ ]:


top_cities=data[data['city'].isin(['MG Road','Brigade Road','Koramangala 4th block',
                                       'Lavelle Road','Koramangala 7th Block'])]
                                         
sns.violinplot(x='city',y='rate',data=top_cities)
plt.xticks(rotation=90)


# In[ ]:


plt.figure(figsize=(10,6))
data.groupby('rest_type')['rate'].mean().sort_values(ascending=False)[:10].plot.bar()


# Let us create new feature 'afford' which is ratio of rate and cost. Basavanagudi provides places with decent rating and cost for two.    

# In[ ]:


data['afford']=data['rate']/data['cost']
data.groupby('location')['afford'].mean().sort_values(ascending=False)[:10].plot.bar()
plt.ylabel('Affordability')


# Let us check variation in affordability in top affordability locations.
# Shivajinagar has extreme variation. 

# In[ ]:


top_afford=data[data['location'].isin(['Basavanagudi','City Market','Commercial Street',
                                      'Shivajinagar','Vijay Nagar'])]
sns.violinplot(x='location',y='afford',data=top_afford)
plt.xticks(rotation=90)


# In[ ]:


data.sample(5)


# In[ ]:


data['rest_type'].unique()


# Let us see restaurants which are family oriented and affordable.

# In[ ]:


dining=data[data['rest_type'].isin(['Casual Dining','Fine Dining'])]
dining.groupby('location')['afford'].mean().sort_values(ascending=False)[:10].plot.bar()


# Not much variation is seen in afforability. KR Puram has only 1 restaurant with Casual Dining I guess.

# In[ ]:


top_afford_dining=dining[dining['location'].isin(['City Market','Uttarahalli','Jalahalli',
                                                 'Mysore Road','KR Puram'])]
sns.violinplot(x='location',y='afford',data=top_afford_dining)
plt.xticks(rotation=90)


# In[ ]:





# In[ ]:





# In[ ]:




