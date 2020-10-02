#!/usr/bin/env python
# coding: utf-8

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

import warnings
warnings.filterwarnings(action = 'ignore')

# Any results you write to the current directory are saved as output.


# In[ ]:


_data = pd.read_csv("../input/zomato.csv")

_data.head()


# In[ ]:


# Preprocessing : removing the useless columns

print("Original set of columns:{}".format(_data.columns))
data = _data.drop(columns = ['url', 'address', 'phone'], axis = 1 )

columns = data.columns

print("New columns : {}".format(columns))
data.head()


# In[ ]:


# Basic data visualization

# What kind of foods are served in Bangalore?

splist = []
cuisine = []
for i in range(0, data['cuisines'].count()):
    splist = str(data['cuisines'][i]).split(', ')
    for item in splist:
        if item not in cuisine:
            cuisine.append(item)
cuisine


# In[ ]:


cuisineCount = pd.DataFrame(columns = ['cuisines', 'count'])
i = 0;
for c in cuisine:
    restaurant = data['cuisines'].str.contains(c, case = False, regex = True, na = False)
    #print( "{} : {}".format(c, restaurant[ restaurant == True].count() ) 
    cuisineCount.loc[i] = [c, restaurant[ restaurant == True].count()]
    i = i+1

cuisineCount.sort_values(by = 'count', axis = 0, ascending = False, inplace = True)

print("The top 10 cuisines sold in bangalore:\n{}".format(cuisineCount.head(25)))


# In[ ]:


#vote count
dataVote = []
for c in cuisineCount['cuisines'].head(10):
    restaurantVote = data[data['cuisines'].str.contains(c, 
                                                      case = False, 
                                                      regex = True, 
                                                      na = False)][data['rate'] 
                                                                   != 'NEW'][data['rate'] 
                                                                             != '-'][['votes', 'cuisines']]
    dataVote.append(sum(restaurantVote['votes']))
dataVote


# In[ ]:


# bar plot on the top 25 cuisines sold in Bangalore

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure (1, figsize = (30,30))

data_CuisineCount = cuisineCount.iloc[0:10, :]

plt.subplot(3,1,1)
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
plt.bar(data_CuisineCount['cuisines'], data_CuisineCount['count'], label = 'Count')

plt.subplot(3,1,2)
#plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
plt.bar(data_CuisineCount['cuisines'], dataVote, label = 'Votes')

plt.subplot(3,1,3)
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
plt.pie(x = data_CuisineCount['count'], labels = data_CuisineCount['cuisines'])

plt.legend()
plt.show()


# The above graphs clearly shows that North Indian and Chinese restaurants dominate the market, whereas, South Indian, Fast Foods and Biryani etc are lagging behind. Also, the number of votes by the customer is an indication of the number of footfalls in the restaurant. Going by the Vote count, we can see that the number of restaurants serving a cuisine is somewhat proportional to the number of footfalls recieved by the restaurant, for most cuisines, which is expected. However, from the plots, it is apparent that the footfall of the continental restaurants are more than chinese restaurants, even though they are less in number. The same arguments goes for cafes.
# Hence, the following conclusion can be drawn. 
# 
# *Conclusion 1:* **Bangalore loves continental foods. Bangalore has a prevalent cafe culture**

# In[ ]:


# getting the top restaurants who serve "Indian"

n = 0
plt.figure (1, figsize = (10,6))
for c in cuisineCount['cuisines'].head(10):
    n = n+1
    restaurantRating = data[data['cuisines'].str.contains(c, 
                                                      case = False, 
                                                      regex = True, 
                                                      na = False)][data['rate'] 
                                                                   != 'NEW'][data['rate'] 
                                                                             != '-']['rate']
    restaurantRating = restaurantRating.dropna(inplace = False)
    rating = []
    rlist = []
    for item in restaurantRating:
        rlist = float(item[0:3])
        rating.append(rlist)
    sns.distplot(a = rating, hist = False, rug = True, label= c)
    plt.title("Distribution plot of rating")
plt.show()


# From the above plot, we get the distribution graph of the ratings of the restaurant serving the top *selling* cuisines in Bangalore. We can draw the inference that Continental and Cafe have more rating than the Indian, North Indian and Chinese counterparts. Hence, it might be apparent, that though Indian, and CHinese foods are more famous than continental or cafe restaurants, the quality of food served by them exceeds the indian and Chinese restaurants.
# 
# *Conclusion 2:* **Bangalore has really good restaurants serving continental food and coffee**
