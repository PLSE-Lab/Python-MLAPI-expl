#!/usr/bin/env python
# coding: utf-8

# # Problem at hand
# Measure how many people are completing a series of steps like Add To Cart and Purchase to see how effectively a business is driving conversions.

# Data: https://www.kaggle.com/retailrocket/ecommerce-dataset/home
# 
# The data was collected over a span of 4.5 months by [RetailRocket](https://retailrocket.net/).

# ## Load packages

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Load relevant data

# In[ ]:


events = pd.read_csv('../input/events.csv')
events.sample(10)
#events.info()


# # Solution

# ## Null value presence?

# In[ ]:


events.isnull().sum()


# **Assumption**: All the null values in the `transactionid` column mean that no transaction occured with respect to that particular `itemid` - `visitorid` combination at a particular timestamp.

# This leads us to consider the observations where transactions happened i.e the observations with a transaction id logged. All we need is the number of entries which were added to cart and purchased.

# In[ ]:


x = list()
x = events['event'].value_counts()
#type(x)

prop_of_events = x/ sum(x)
prop_of_events


# Intuitively, the proportion of numbers makes sense with what happens on a day-to-day basis.

# For each item purchased, 
# 1. It could have been added to cart and then purchased
# 2. It could have been directly purchased. This happens rarely.
# 
# **Assumption**: To make a successful purchase, a visitor will go through adding the item to the cart.

# ## Percentage of purchases

# In[ ]:


perc_conv = x[2]*100/x[1]
perc_conv


# As the data was collected over a span of 4.5 months, **32.4%** of items which were added to the cart made a transaction/ purchase provided that the assumption above holds true. An item level calculation was done as the main goal is the number of transactions occurred after adding the items to the cart. And it is **0.81%** of all the data, simply put as ~**8 per 1000 sessions** made a purchase on the company's website (It could be repeated entry of the same visitor in any of the event category).

# # Extra study

# In[ ]:


#Define new data frame which doesn't have 'view' events
events_non_view = events[(events['event'] != "view")]
events_non_view.sample(5)


# ### Sample data frame for the visitorid: 1303838 (randomly chosen)

# In[ ]:


events_non_view[events_non_view['visitorid'] == 1303838]


# ![](http://)This is one such visitor who didn't make a transaction but added two different items to cart in the span of 4.5 months.

# ### Sample visitor who purchased an item two times by adding it to the cart two different times

# In[ ]:


events_non_view[(events_non_view["visitorid"] == 1210136) & (events_non_view["itemid"] == 253214)]


# In[ ]:


#An alternative way of checking addtocart and transaction logged for a user and a different item.
cond1 = (events_non_view["visitorid"] == 1210136)
cond2 = (events_non_view["itemid"] == 396732)

events_non_view[cond1 & cond2]


# In[ ]:


events.groupby('event').count()


# This is more like getting the counts per category but in a tabular format just like a correlation matrix. But what we want is the number of unique `visitorid`s per `event` type.

# In[ ]:


unique_visitors = events.groupby('event')['visitorid'].nunique()
unique_visitors


# To cross check, calculate the total number of unique visitorids:

# In[ ]:


len(events['visitorid'].unique())


# Considering the earlier [assumption](https://www.kaggle.com/akshayreddykotha/how-effectively-business-is-driving-conversions#Percentage-of-purchases), the proportions are:

# In[ ]:


perc_uniq_visitors = unique_visitors/ sum(unique_visitors)
print(perc_uniq_visitors)


# In[ ]:


perc_uniq_visitors[1]*100/perc_uniq_visitors[0]


# The above proportion gives the percentage of unique visitors who added to the cart and made a purchase relative to those who had just added an item to the cart. The number of absolute transactions remains same at **8 per 1000**. The calculation on a unique visitor basis helps us understand if the data is biased to some users who followed the series of steps adding to the item to the cart (`addtocart`), and made many of the transactions/ purchases (`transaction`). Clearly, that is not the case.

# In[ ]:


#mask = (events_non_view["visitorid"] == 1210136) & events_non_view["itemid"] == 396732)
#events_non_view.ix[mask, events_non_view]
#events_non_view[events_non_view['visitorid'] == 1210136] & events_non_view[events_non_view['itemid'] == 396732]

