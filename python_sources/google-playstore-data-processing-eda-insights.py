#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# We can see that there are two files available in this dataset. Let's load them in. 

# In[ ]:


playstore = pd.read_csv('../input/googleplaystore.csv')
reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')


# In[ ]:


playstore.head()


# In[ ]:


playstore.isnull().sum()


# In[ ]:


reviews.head()


# I notice that **playstore** dataframe is ideal for analysing the stats in general while the **reviews** dataframe is good choice for sentiment analysis.
# 
# As I am not yet exposed to NLP techniques, I'll leave **reviews** dataframe alone for now.  
# 
# Lets dive into **playstore**

# getting the shape of **playstore**

# In[ ]:


playstore.shape


# Well **playstore** seems to be in good shape to me. Not too large and not too small with 10841 rows and 13 cols.

# Let's checkout what datatypes exist in this dataframe.

# In[ ]:


playstore.dtypes


# Well, this is one boring thing and I dont like it. A lot of* objects*. That means a lot of data cleaning and conversions. :(

# At the onset we can see that there are a few variables that should have been either integers or floats but are objects instead. Such parameters are :
# * Reviews
# * Size
# * Installs
# * Price
# 
# Let's convert them one by one.

# In[ ]:


def get_reviews(reviews):
    if reviews.endswith('.0M'):
        reviews = reviews[:-3] + '000000'
        return int(reviews)
    else:
        return int(reviews)
playstore['Reviews'] = playstore['Reviews'].apply(get_reviews)


# In[ ]:


def get_size(size):
    if size == 'Varies with device':
        return 10
    elif size.endswith('M'):
        size = size[:-1]
        return float(size)
playstore['Size'] = playstore['Size'].apply(get_size)


# In[ ]:


x = np.unique(playstore['Installs'])
x


# It can be noted here that the values contain a **+** sign and a few **,** some installs are even mentioned as Free which makes no sense at all.
# 
# We need to remove all the **+** signs and **,** in order to proceed as this operation will convert the strings into covertable format that will allow us to transform the data in **integer** format.
# 
# We can remove the commas using the re libraray as is used below. This will save us a lot of pain that would be caused if we loop through all the reviews removing commas and deleteing spaces so created as we go.
# 
# Everything (almost) has a libraray. That is why python is love! <3

# In[ ]:


def get_installs(installs):
    installs = re.sub(',', '', installs)
    if installs.endswith('+'):
        installs = installs[:-1]
        return int(installs)
    else:
        return 0
playstore['Installs'] = playstore['Installs'].apply(get_installs)


# see? how easy it was. Let's move on!

# In[ ]:


p = np.unique(playstore['Price'])
p


# Same thing here. we need to remove **$** sign and I will label the **Everyone** column as **0**

# In[ ]:


def get_price(price):
    price = re.sub('\$', '', price)
    if price == 'Everyone':
        return float(0)
    else:
        return float(price)
playstore['Price'] = playstore['Price'].apply(get_price)


# # **Missing Values**

# Missing values exist in :
# * Rating - Mean
# * Size - Mean
# * Type - Free if Price = 0 / else paid
# * Current Ver
# * Android Ver

# In[ ]:


mean_rating = playstore['Rating'].mean()
mean_size = playstore['Size'].mean()
playstore['Rating'] = playstore['Rating'].fillna(mean_rating)
playstore['Size'] = playstore['Size'].fillna(mean_size)


# We need to check if the missing values in Type do have non zero price or not.  

# In[ ]:


playstore.loc[playstore['Type'].isnull()]


# Yes! We can fill NaN value of this **Type as Free** as it can be seen that the **Price as well is 0.0**! Let's do this already!

# In[ ]:


playstore['Type'] = playstore['Type'].fillna('Free')


# Let's see which app has a **missing Content Rating**. Then maybe by looking at the **Category** of the app and other such features we can identify what should be the Content Rating.

# In[ ]:


playstore.loc[playstore['Content Rating'].isnull()]


# This is one weird app with category 1.9! Funny. Although by name, I can imagine fron its name* Life Made WI-Fi Touchscreen Photo Frame*, it should be a pretty general nature. Lets give it a** Content Rating of Everyone**

# In[ ]:


playstore['Content Rating'] = playstore['Content Rating'].fillna('Everyone')


# I will drop the remaining NA values.

# In[ ]:


playstore.dropna(how = 'any', inplace = True)


# In[ ]:


playstore.isnull().sum()


# Data Cleaning done and dusted!

# # **Exploratory Data Analysis**

# Right off the top of the head, I can think that it would be very intresting and insightfull to know what is the relationship between **Category and Installs** Let's see what we find when we dive deep into this relationship.

# It would be worthwhile to find out what all categories are available on playstore.

# In[ ]:


cat = np.unique(playstore['Category'])
cat


# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Category','Installs', data = playstore)
plt.show()


# A few key takeaways : 
# * Highest downloads are for COMMUNICATION category. Agreed! Whatsapp! 
# * SOCIAL category has the second highest install rates. This is is not a shocker as apps like Facebook, Twitter, Instagram are very very dominant in the smartphone arena.
# * What really disappoints me is that HEALTH_AND_FITNESS apps are among the lowest installed apps. This is a bitter remiender of the very unhealthy lifestyle that we have decended into. I personally believe that we need to be much more concerned about out fitness. I will look into this issue more as I proceed with the analysis.
# * It is great to know NEWS_AND_MAGAZINES apps have a substantial number of installs
# * PHOTOGRAPHY/TRAVEL_AND_LOCAL apps have a greater number of installs that SHOPPING. Looks wierd to me as almost everthing now sells online and almost every person that I have met has Amazon installed in their phones. But on the other side int can be that the number of good PHOTOGRAPHY apps are flatout more than the number of good SHOPPING apps. We will see.
# * GAME is almost equal to PHOTOGRAPHY. Cool!
# * VIDEO_PLAYERS also has a good number of Installs. Pretty simple. For those who like to watch videos on our phone. 

# I will explore the HEALTH_AND _FITNESS category a little more.

# In[ ]:


health = playstore.loc[playstore['Category'] == 'HEALTH_AND_FITNESS']
health


# 341 such apps exist. Lets compute the mean of their RATING

# In[ ]:


health['Rating'].mean()


# Pretty good Rating there! On further thought I believe that people like to take their fitness in thier own hands by going to gyms and runs and consulting their trainers and friends and doctors regarding it instead of an app. If this is the case then it is good as we have not become digital zombies atleast in this aspect of life! :)

# **Number of Apps in Each Category**

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.countplot('Category', data = playstore)
plt.show()


# Family has the highest number of apps available on PlayStore. But it is very insightfull to know that COMMUNICATIONS which has a very low number of apps available has the highest number of Installs as visualised earlier while FAMILY is one of the lowest downloaded apps. This is very well attributed to the popularity of apps like WhatsApp, Messanger, etc. But, why so many FAMILY apps are being developed?

# Next I would like to explore the **Rating vs Category** relationship.

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Category','Rating', data = playstore)
plt.show()


# All the Categories have an almost equal rating.

# Let's see how Categories are distributed with respect to Reviews

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Category','Reviews', data = playstore)
plt.show()


# The most installed categories (COMMUNICATION, GAME, SOCIAL) have the highest number of Reviews.
# Great! 

# Let's see how Size is distributed across the data.

# In[ ]:


plt.subplots(figsize = (8,4))
sns.distplot(playstore['Size'], bins = 15)
plt.show()


# It is evident that most of the apps range between 0 to 20 MB and a few others from 20 to 60.

# Let's see what category of apps have the largest size.

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Category','Size', data = playstore)
plt.show()


# As is evident that GAME category has the largest size. Not uncomman at all. We have seen that an average game size exceeds 35MB. Some games are also in the order of GBs. 
# 
# The FAMILY apps have the second largest size. I do not have these apps and so I cannot comment. :P
# 
# So, are you running low on your phone space? You know what apps to uninstall! :P

# Lets see how many people are willing to spead a dime on these apps. **Installs vs Type**! Lets go! Woohoo!!

# In[ ]:


plt.subplots(figsize = (8,4))
plt.xticks(rotation = 90)
sns.barplot('Type','Installs', data = playstore)
plt.show()


# Not a lot of people (including me :P) it seems!

# An intresting visualisation would be how Categories and Installs relate with respect to Type. **Category vs Installs with Hue = Type**

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Category','Installs',hue = 'Type', data = playstore)
plt.show()


# Nothing to see here. Almost all apps are free. Disappointing.. :(

# I will now visualise the distribution of Price across the dataset.

# In[ ]:


plt.subplots(figsize = (8,4))
sns.distplot(playstore['Price'], bins = 2)
plt.show()


# Most of the paid apps are less than 50 bucks. Cool!
# 

# **Content Rating vs Installs**

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Content Rating','Installs', data = playstore)
plt.show()


# **Content Rating vs Reviews**

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Content Rating', 'Reviews', data = playstore)
plt.show()


# The trend is similar as to what we have witnessed in Content Rating vs Installs. An intresting obervation can be that almost everyone who installed the apps with **Content Rating Everyone 10+ and Adults only 18+ left a review**.

# Let's see how many apps are there in each category by drawing a countplot.

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.countplot('Content Rating', data = playstore)
plt.show()


# An intresting observation here is that although the category **Everyone** has the highest number of apps, **it gets beaten by Everone 10+ and Teen which are less than Everyone even when combined!**

# Lets see how Ratings affect Reviews. **Rating vs Reviews**.

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Rating', 'Reviews', data = playstore)
plt.show()


# Ignore the Rating 19. The apps with a rating of 4.5 have the highest umber of reviews.
# 
# I'll now draw a countplot of Ratings to see how ratings are distributed across the dataset.

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.countplot('Rating', data = playstore)
plt.show()


# This is a very beautiful distribution! **majority of apps are rated in the range of 4.0 to 4.7**. Cool!

# There are two features that I have rather overlooked. **Android Ver and Current Ver**.

# **Android Ver vs Rating**

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Android Ver', 'Rating', data = playstore)
plt.show()


# **Android Ver vs Reviews**

# In[ ]:


plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Android Ver', 'Reviews', data = playstore)
plt.show()


# There is a lot of redundency in this parameter (like 7.1 and up & 7.0 - 7.1.1) as we can see. This is making this parameter look rather bad. Or maybe I am wrong. I do require some help here! 

# That's it for now! I will be adding more visualisations in the coming days.
# 
# If you like my work, do cast an upvote, it helps!
# 
# Any suggestions are welcome.
