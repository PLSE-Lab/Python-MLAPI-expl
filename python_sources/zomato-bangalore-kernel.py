#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import datetime # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


restaurants = pd.read_csv('../input/zomato.csv')


# In[ ]:


print(restaurants.shape)


# In[ ]:


restaurants.info()


# In[ ]:


restaurants.head()


# In[ ]:


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


# In[ ]:


def get_rating(name):
    if type(name) == str and hasNumbers(name):
        rating = name.split('/');
        return float(rating[0]);

#Get all the ratings
ratings = restaurants['rate'].apply(get_rating)
restaurants['ratings'] = ratings


# In[ ]:


restaurants['listed_in(type)'].value_counts()


# In[ ]:


restaurants['rest_type'].value_counts()


# In[ ]:


restaurants.iloc[0,10]


# In[ ]:


restaurants.loc[0,'dish_liked']


# In[ ]:


restaurants['rest_type'].value_counts()


# In[ ]:


restaurants['listed_in(city)'].value_counts().plot(kind='bar', title='Restaurant count by Area',figsize=(20,8)) 


# In[ ]:


restaurants['listed_in(type)'].value_counts().plot(kind='bar', title='Restaurant count by Type')


# In[ ]:


#instead of appending data to dataframe , it gives much better performance when data is appended to lists and later converted to dataframe
restTypesArr = []
cuisineTypesArr = []
for index, row in restaurants.iterrows():
    if type(row['rest_type']) == str:
        rest_type_arr = row['rest_type'].split(',')
        for rest_type in rest_type_arr:
            restTypesArr.append([index,rest_type.strip(),row['listed_in(city)']])
            
    if type(row['cuisines']) == str:
        cuisine_type_arr = row['cuisines'].split(',')
        for cuisine_type in cuisine_type_arr:
            cuisineTypesArr.append([index,cuisine_type.strip(),row['listed_in(city)']])


# In[ ]:


#Convert list of tuples to dataframe and set column names and indexes
restaurantTypedf = pd.DataFrame(restTypesArr,columns=['rest_id', 'rest_type','listed_in(city)'])
cuisineTypedf = pd.DataFrame(cuisineTypesArr,columns=['rest_id', 'cuisine_type','listed_in(city)'])


# In[ ]:


restaurantCounts = restaurantTypedf.groupby('rest_type')['rest_type'].agg(['count']).reset_index()
restaurantCounts = restaurantCounts.rename(columns = {'count': 'restaurant_counts'})
restaurantCounts = restaurantCounts.sort_values('restaurant_counts',ascending=False)
restaurantCounts.head()


# In[ ]:


cuisineCounts = cuisineTypedf.groupby('cuisine_type')['cuisine_type'].agg(['count']).reset_index()
cuisineCounts = cuisineCounts.rename(columns = {'count': 'cuisine_counts'})
cuisineCounts = cuisineCounts.sort_values('cuisine_counts',ascending=False)
cuisineCounts.head(10)


# ****There are double the restaurants serving North indian food compared to South indian food in Bangalore****

# In[ ]:


topCuisines = cuisineCounts.head(10)
index = np.arange(10)
f, ax = plt.subplots(figsize=(10,5))
plt.bar(index, topCuisines['cuisine_counts'])
plt.xlabel('Cuisine', fontsize=10)
plt.ylabel('No of restaurants', fontsize=10)
plt.xticks(index, topCuisines['cuisine_type'], fontsize=10, rotation=30)


# In[ ]:


# df = pd.DataFrame({'category': list('XYZXY'), 'B': range(5,10),'sex': list('mfmff')})
# df.head()


# In[ ]:


# df.groupby(['category','sex']).B.count().unstack().reset_index()\
# .plot.bar(x = 'category', y = ['f', 'm'])
# print(df.groupby(['category','sex']).B.count())
# print(df.groupby(['category','sex']).B.count().unstack())
# print(df.groupby(['category','sex']).B.count().unstack().reset_index())


# In[ ]:


def topCuisine(group,rank):
    df= group.nlargest(rank).iloc[-2:]
    return df
    
#     return group.loc[group['ranks']==3 ,:]

s = cuisineTypedf.groupby(['listed_in(city)','cuisine_type'])['rest_id'].size()
s = s.groupby(level=0).apply(topCuisine,rank=4)
# print(s.reset_index(level=0, drop=True).unstack())
s.reset_index(level=0, drop=True).unstack().plot.bar(figsize=(14,10))
plt.xlabel("Area", labelpad=14)
plt.ylabel("Count of Restaurants", labelpad=14)
plt.title("the most popular cuisines in Each Area", y=1.02);


# The below code gives top selling food in an area based on number (x) . i.e if 2 is given , it will return 2nd highest available cuisine in an area

# **North Indian and Chinese cuisines are two most popular foods in all areas in Bangalore**

# In[ ]:


# print(cuisineTypedf.groupby(['listed_in(city)','cuisine_type'])['cuisine_type'].count())
s = cuisineTypedf.groupby(['listed_in(city)','cuisine_type'])['rest_id'].count()
# print(s.groupby(level=0).apply(lambda group: group.nlargest(4).iloc[-2:]))
s.groupby(level=0).apply(lambda group: group.nlargest(4).iloc[-2:]).reset_index(level=0, drop=True).unstack().reset_index().plot.bar(x = 'listed_in(city)',  figsize=(20, 6), rot=30);
plt.xlabel("Area", labelpad=14)
plt.ylabel("Count of Restaurants", labelpad=14)
plt.title("Two most popular cuisines in Each Area", y=1.02);


# 1. top mexican restaurants
# 2. top italian restaurants
# 3. top chinese restaurant
# 4. top biryani restaurants
# 5. top breweries with dine in
# 6. number of restros servig food vs drinks
# 7. area, cuisine_type and price range gives top 5 restaurants in area with cuisine type below range
# 

# In[ ]:


def cuisineRest(cusineType):
    mexicanRestDf = cuisineTypedf.loc[cuisineTypedf['cuisine_type']==cusineType,:]
    result = pd.merge(mexicanRestDf, restaurants, left_on='rest_id', right_index=True, how='left', sort=False);
    result['rating_votes'] = result['votes'] * result['ratings']
    result = result.drop(columns = ['listed_in(city)_x'])
    result = result[['rating_votes','address','listed_in(city)_y','rest_id']].sort_values('rating_votes', ascending=False)
    # print(result.groupby('listed_in(city)_y')['rating_votes'].sum().reset_index())
    result.groupby('listed_in(city)_y')['rating_votes'].sum().reset_index().head(10).sort_values('rating_votes', ascending=False).plot.bar(x = 'listed_in(city)_y', y = 'rating_votes',figsize=(20, 6), rot=30);
    plt.xlabel("Area", labelpad=14)
    plt.ylabel("top ratings", labelpad=14)
    plt.title( "{} restaurants in Each Area".format(cusineType), y=1.02);

# restaurants.iloc[result.head()['rest_id'],1]
cuisineRest('Biryani')
cuisineRest('Mexican')
cuisineRest('Italian')


# In[ ]:




