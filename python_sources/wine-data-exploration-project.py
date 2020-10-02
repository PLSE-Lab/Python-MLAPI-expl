#!/usr/bin/env python
# coding: utf-8

# # <font color=pink>Data Exploration on Wine Dataset</font>
#  
#  Hello!
# 
# As a recent wine enthusiast and hopefully future data scientist I was excited to come across this dataset! 
#  
#  I am curious to explore the different results of 150k wine reviews with variety, location, winery, price, and description. Lets get started!

# ## <font color=green>Importing the Modules</font>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt

plt.figure()


# Any results you write to the current directory are saved as output.


# ## <font color=green>Data Importing</font>

# In[ ]:


wine = pd.read_csv('../input/winemag-data_first150k.csv')


# ## <font color=green>Organizing the Data</font>
# 
# We want to get the top 5 countries, varieties, and wineries

# In[ ]:


country = wine['country'].value_counts()
country1 = country.iloc[0:5]


# In[ ]:


variety = wine['variety'].value_counts()
variety1 = variety.iloc[0:5]


# In[ ]:


winery = wine['winery'].value_counts()
winery1 = winery.iloc[0:5]


# ## <font color=red>What country had the most wine reviews?</font> 

# In[ ]:


country1.plot(kind='bar')


# The top country that got the most reviews was the US then Italy then France. One could argue that the US could be at the top simply because of the convience of location if the critics lived in the US.

# ## <font color=red>What winery had the most reviews?</font> 
# 

# In[ ]:


winery1.plot(kind='bar')


# Williams Selyem winery had the most reviews.

# ## <font color=red>What was the most reviewed type of wine?</font>

# In[ ]:


variety1.plot(kind='bar')


# The most reviewed wine was Chardonnay. 

# ## <font color=green>Gather Data Based on Points Given</font>

# In[ ]:


points = wine['points']
best = points == 100


# ## <font color=red>Of the 150k wine reviews, how many recieved a perfect score (100 points)? </font>

# In[ ]:


wine[best].shape


# There were 24 wine reviews given 100 points. These were the best of the best wines rated. 

# ## <font color=red>Of the best wines, which were the cheapest and what kind were they? </font>

# In[ ]:


best1 = wine[best]
price1 = best1['price'].value_counts()
price1.plot(kind='bar')


# The best wines were more than $300, even reaching up to $1400. The cheapest best wine was $65 and was a Syrah (type of red wine).

# ## <font color=red>Of the best wines, what kind of wine were they? </font>

# In[ ]:


type = best1['variety'].value_counts()
type.plot(kind='bar')


# The wines that got the most perfect scores was the Syrah, Merlot, Muscat, Cabernet Sauvignon, and Chardonnay.

# ## <font color=red>Of the best wines, what country did a majority of them belong to?</font>

# In[ ]:


country2 = best1['country'].value_counts()
country2.plot(kind='pie')


# The wines with the best reviews were found in the US

# ## <font color=red>What was the price of the majority of wines reviewed?</font>

# In[ ]:


price = wine['price'].value_counts()
price.head()


# Most of the wines reviewed were around $20

# ## <font color=red>Is there a correlation between wine scores and price?</font>

# In[ ]:


df = pd.DataFrame({'points': points,
                       'price': price}).dropna()
df.corr()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




