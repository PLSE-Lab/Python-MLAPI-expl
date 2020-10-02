#!/usr/bin/env python
# coding: utf-8

# This kernal shares with some data cleaning and prosessing problem I encountered when I was exploring the Google Play Store Apps datasets. It is hoped that it would be helpful if someone just begin to look through the data had the same problems as I did. All feedback and improvement are welcome as well.
# 
# *Lasted Updated: May 18th 2018*
# 
# **Cleaning and Preprocessing**
# * Data Exploration
# * Convert and clean data types
# 
# **Univariate Exploration**
# * Genre Distribution (How to split and calculate apps with multiple genres?)
# * Rating Distribution
# * Paid Apps Deep Dive
# 
# **Bivariate Exploration and Analysis**
# * Rate vs. Review
# 
# (More works will be done)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Cleaning and Preprocessing**
# * Data Exploration
# * Convert and clean data types

# In[ ]:


dt_store = pd.read_csv("../input/googleplaystore.csv")
dt_store.head()


# In[ ]:


dt_store.info()


# In[ ]:


print(dt_store.shape)


# After a sneak peek into our store data, we have a few findings:
# * Altogether there are 10.8k apps records. And there are 13 varaibles describing the apps.
# * There are some missing values in **Rating** variables. Besides, other varaibles have missing variables as well on different levels but it not a huge concerning number.
# * There are a variables needed to convert its data type. They are supposed to be numeric however they are string type right now. Example: **Reveiws** (Integer), **Price **(Float), **Last Updated** (Datestamp)

# In[ ]:


#data type cleaning
#clean reviews data
#We find one record containing 3.0M that leads us not being able to convert it right to numeric
dt_store.loc[dt_store.Reviews == '3.0M', 'Reviews'] = "3000000"
dt_store['Reviews'] = pd.to_numeric(dt_store['Reviews'])
#Besides, we find some price value is recorded as 'Everyone', I'm not sure what does it mean, here I simply convert all of them to 0
dt_store.loc[dt_store.Price == 'Everyone', 'Price'] = "0"
dt_store['Price'] = pd.to_numeric(dt_store['Price'].apply(lambda x: x.strip("$")))
dt_store.info()


# In[ ]:


dt_store.head()


# **Univariate Exploration**
# * Genre Distribution (How to split and calculate apps with multiple genres?)
# * Free and Priced Apps Distribution
# * Review Distribution

# In order to get the distribution of genres, we realized that we needed to first clean the genre columns as we realized. There are some apps possessing more than one genres and they are currently stored in the same columns. 
# * First split the genres by ';' and get a full unique list of them
# * Create a dummy columns for each genres
# * Caculate the sum of each columns and append them into a dictionary
# * Sorted by their number and plot as a horizontal histogram
# 
# **Finding**: the top 5 largest categories are Tools (843), Education (782), Entertaiment (667), Action (503) and Medical (463)

# In[ ]:


#create a whole list of genres
new_genre = dt_store["Genres"].str.split(";", n = 1, expand = True) 
new_genre.head(100)
#create a full unique list of genres
first_list = list(new_genre[0].unique())
second_list = list(new_genre[1].unique())
in_first = set(new_genre[0].unique())
in_second = set(new_genre[1].unique())
in_second_but_not_in_first = in_second - in_first
full_genre_list = list(first_list) + list(in_second_but_not_in_first)

#for each genres, we create a dummy columns for them
for g in full_genre_list:
    dt_store[str(g)] = dt_store["Genres"].apply(lambda x: 1 if str(g) in x else 0)
dt_store.head(30)


# In[ ]:


#make a plot about the distribution of genre categories
import matplotlib.pyplot as plt
list(dt_store.columns)[13:]
g_dic = {}
for g in full_genre_list:
    a = {str(g): dt_store[str(g)].sum()}
    g_dic.update(a)
lists = sorted(g_dic.items(), key=lambda x: x[1])
# equiv# sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.rcParams["figure.figsize"] = (10,16)
plt.barh(x, y)
plt.title("Number of Apps Distribution across Genres",fontsize=15)
plt.show()


# **Free/Paid Distribution:** About 92.6% of the apps on Google Play Stores are free. Only 7.4% of them are paid.
# 
# Someone may wonder if so many of them are free, how are those apps getting profit? It turned out to be a strategy that the developer set the apps free to entice download. If the users would like to use some premium service, they need to pay or subscribe for it. The other way is that the apps sell ads in it to turn profitable.

# In[ ]:


#free/paid distribution
type_c = dt_store.groupby(["Type"]).size()/dt_store.shape[0]
plt.rcParams["figure.figsize"] = (10,5)
type_c[1:].plot(kind='barh')
plt.title("Proportion of Paid/Free Apps",fontsize=15)
plt.show()


# **Rating Distribution**: with the first histogram of rating, we find that there are a few outliers in it, given app store usually only use scale of 0-5 to rate their apps. Therefore, we set a cap at 5. Within the 9.3k records that has rating value, the mean is about 4.3. And it is highly left skewed.

# In[ ]:


#Review Distribution
plt.hist(dt_store['Rating'],bins=20)


# In[ ]:


dt_store.loc[dt_store['Rating'] > 5.0, 'Rating'] = 5
plt.hist(dt_store['Rating'],bins=20)
print(dt_store['Rating'].describe())


# **Paid Apps Deep Dive**
# Even though paid apps are only a small fraction of it. We would like to see how they distribute across genres and categories and how they set their price. We find that:
# * Medical, Personalize, Tools, Education and Action are at the top 5
# * For most of the apps, even though they set the price, they are only 0.99 dollar or below.
# * We do see a small spike between the price of 350 and 400. After digging in a bit, we found that those are just 'I am rich' apps people downloaded to show off LOL.

# In[ ]:


#paid apps distribution
paid_app = dt_store.loc[dt_store['Type'] == 'Paid']

import matplotlib.pyplot as plt
list(dt_store.columns)[13:]
g_dic = {}
for g in full_genre_list:
    a = {str(g): paid_app[str(g)].sum()}
    g_dic.update(a)
lists = sorted(g_dic.items(), key=lambda x: x[1])
# equiv# sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.rcParams["figure.figsize"] = (10,16)
plt.barh(x, y)
plt.title("Number of Apps Distribution across Genres for Paid Apps",fontsize=15)
plt.show()


# In[ ]:


plt.rcParams["figure.figsize"] = (16,10)
plt.hist(paid_app.Price, bins=40)


# In[ ]:


paid_app.loc[paid_app['Price'] > 350].head()


# **Bivariate Exploration and Analysis**
# * **Rating vs. Review**: It is hard to conclude that more review would lead to higher rating. However, after we take a log transoformation of review. The patterns seems more dicernable. For those that got very low rating, all of them seems to have small number of review. Higher review to some extent would correlate with higher rating, but there is the diminishing return effect here.

# In[ ]:


#rate and review
plt.scatter(dt_store['Reviews'],dt_store['Rating'])
plt.ylim(top=5.5, bottom =0.5)
#plt.xlim(right=3000000, left=-1)
plt.rcParams["figure.figsize"] = (10,10)
plt.show()


# In[ ]:


#rate and review
plt.scatter(np.log(dt_store['Reviews']),dt_store['Rating'])
plt.ylim(top=5.5, bottom =0.5)
#plt.xlim(right=3000000, left=-1)
plt.rcParams["figure.figsize"] = (10,10)
plt.show()


# More work will be done and updated! Feel free to leave any comments :) Thanks!
