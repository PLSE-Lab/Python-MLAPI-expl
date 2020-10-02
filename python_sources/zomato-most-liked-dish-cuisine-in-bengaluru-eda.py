#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Loading the Data File**

# In[ ]:


zomato = pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')


# **Displaying the head values for getting a intuition of the dataset**

# In[ ]:


zomato.head(5)


# **Data Cleaning**

# In[ ]:


#removing unnecsary features such as url,address and phone
del zomato['url']
del zomato['phone']
del zomato['address']


#  **Simplest Visualization**
# * Bar Chart 

# **Which location prodces more orders in Bangalore ?**

# In[ ]:


zomato['location'].value_counts().head(20).plot.bar()


# As plotted bar above we can clearly understand the pattern  BTM produces more orders. 

# In[ ]:


zomato['location'].value_counts().tail(20).plot.bar()


# The above plotted graph shows that the location at the tail have no or minimum orders in the datasets.We can conclude that as the distance increases from BTM layout area the order rate decreases.

# **Let's Check the rating **

# In[ ]:


zomato['rate'].value_counts().sort_index().plot.line()


# The above line chart indicates that between 3.3 rating to to 4.1 strecth are given to the order ,but chart fails to deliver the exact rating which would be useful.

# As infered from the above dataset we can conclude that due to '/5' present we cannot conclude the ratings correctly.Hence we remove it and try to infer the ratings more precisely

# In[ ]:


zomato['rate'] = zomato['rate'].str.extract('(\d\.\d)', expand=True)


# In[ ]:


zomato['rate'] = zomato['rate'].astype(float)


# **Let's check the aprrox cost **

# In[ ]:


zomato['approx_cost(for two people)'].unique()


# In[ ]:


#Let's clean it
zomato['approx_cost(for two people)'] = zomato['approx_cost(for two people)'].str.replace(',','')


# In[ ]:


zomato['approx_cost(for two people)'].fillna(0)


# **Let's Plot a graph for ratings**

# In[ ]:


zomato['rate'].value_counts().sort_index().plot.bar()


# As we remove it we can now clearly to infer that ratings given are in range 3.1 to 4.4 . From the bar graph above one thing we can conclude is the online ordering from zomato's listed restaurant's does not have that much bad impact on the customers as there are few negative ratings simply this means if we list our restaurant on zomato then definetly it can boost our sales.

# Let's Check the Votes 

# In[ ]:


zomato.votes.describe()


# There is a reasonable jump  from 198 to 16832 

# In[ ]:


zomato['votes'].plot.hist()


# We can infer that the data is skewed to the right.Hence we need to perform the transformation or we can just simply remove the outliers.

# Let's try some of the transformation of the skweness techniques.

# In[ ]:


def normalize(column):
    upper = column.max()
    lower = column.min()
    y = (column - lower)/(upper-lower)
    return y


# In[ ]:


helpful_votes = normalize(zomato.votes)
helpful_votes.describe()


# In[ ]:


helpful_votes.plot.hist()


# As we can note the normalization has just skewed the data as it was before but in range from 0 to 1

# In[ ]:


def sigmoid(x):
    e = np.exp(1)
    y = 1/(1+e**(-x))
    return y


# In[ ]:


sigmoid_votes = sigmoid(zomato.votes)
sigmoid_votes.describe()


# In[ ]:


sigmoid_votes.plot.hist()


# Sigmoid has spread the values from 0.5 to 1 as there are no negative values but the distribution in 25th quartile is within the 0.1 from the 100th percentile which is very far in the original dataset.

# In[ ]:


helpful_log = np.log(zomato.votes+1)


# In[ ]:


helpful_log.describe()


# It is an excellent technique as quartiles as reflective as orginal and we have also added +1 in order where the votes were 0 as in log transformation domain for log should be greater than 0.

# In[ ]:


helpful_log.plot.hist()


# In[ ]:


helpful_log_normalized = normalize(helpful_log)
helpful_log_normalized.describe()


# In[ ]:


helpful_log_normalized.plot.hist()


# A perfect transformation of the skew data by applying the log transformation and normalization together.

# **Let's solve some questions related to the dataset **

# In[ ]:


cost_dist=zomato[['rate','approx_cost(for two people)','online_order']].dropna()
cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))


# Let's Check Cost vs Rating

# In[ ]:


plt.figure(figsize=(10,7))
sns.scatterplot(x="rate",y='approx_cost(for two people)',hue='online_order',data=cost_dist)
plt.show()


# We can conclude that when order is placed through zomato then definitely the cost get's reduced as we can think of it as applied offer's and deals etc.

# **What is Cost of Distribution for two people ?**

# In[ ]:


plt.figure(figsize=(6,6))
sns.distplot(cost_dist['approx_cost(for two people)'])
plt.show()


# We can conclude that restaurants generally charge upto 1000(INR) for two people.

# **Which type of order has more votes ?**

# In[ ]:


votes_yes=zomato[zomato['online_order']=="Yes"]['votes']
trace0=go.Box(y=votes_yes,name="accepting online orders",
              marker = dict(
        color = 'rgb(214, 12, 140)',
    ))

votes_no=zomato[zomato['online_order']=="No"]['votes']
trace1=go.Box(y=votes_no,name="Not accepting online orders",
              marker = dict(
        color = 'rgb(0, 128, 128)',
    ))

layout = go.Layout(
    title = "Box Plots of votes",width=800,height=500
)

data=[trace0,trace1]
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# Online ordering will have high votes as it has the integrated feature of voting the restaurant.

# **Let's Look at which are the most liked dishes**

# In[ ]:


plt.figure(figsize=(7,7))
cuisines=zomato['dish_liked'].value_counts()[:10]
sns.barplot(cuisines,cuisines.index)
plt.xlabel('Count')
plt.title("Most liked dishes in Bangalore")


# Yes Biryani is the most liked dish and why it wouldn't be ? Bengaluru is a metro-city and is located in southern part of India too.

# Let's investigate it further

# In[ ]:


biryani_=zomato[zomato['dish_liked'].isin(['Biryani', 'Chicken Biryani']) ][zomato['online_order']=="Yes"]


# In[ ]:


biryani_['approx_cost(for two people)'] = biryani_['approx_cost(for two people)'].astype(int)


# **What should be the cost for highly rated biryani in Bengaluru ?**

# In[ ]:


plt.figure(figsize=(20,20))
biryani_.reset_index().plot.scatter(x = 'rate', y = 'approx_cost(for two people)')
plt.show()


# 200-800 INR is what it takes for good rated biryani.

# **Which type of restaurant's have the highly rated biryani ?**

# In[ ]:


plt.figure(figsize=(10,7))
sns.scatterplot(x="rate",y='rest_type',data=biryani_)
plt.show()


# Yes the restaurant with delivery,take-away and quick bites has quite good rating.

# **Let's Check which area in bangalore delivers the highly rated biryani ?**

# In[ ]:


biryani_r=biryani_[biryani_['rate']>=4]


# In[ ]:


plt.figure(figsize=(10,7))
sns.scatterplot(x="rate",y='location',data=biryani_r)
plt.show()


# Well Kumaraswamy Layout,Church Street,Malleshwaram is the best location to order from!!!

# **Which are the best cuisines to order from ?**

# In[ ]:


plt.figure(figsize=(7,7))
cuisines=zomato['cuisines'].value_counts()[:10]
sns.barplot(cuisines,cuisines.index)
plt.xlabel('Count')
plt.title("Most popular cuisines of Bangalore")


# As a result we can conclude that North Indian,South Indian and Chinese and Biryani cuisines are the most popular one's

# In[ ]:


cuisine_=zomato[zomato['cuisines'].isin(['North Indian', 'North Indian, Chinese','South Indian','Biryani','South Indian, North Indian, Chinese']) ][zomato['rate']>=4]


# **Which location have more cusines of North Indian,South Indian,Chinese and Biryani ?**

# In[ ]:


plt.figure(figsize=(7,7))
loc=cuisine_['location'].value_counts()[:10]
sns.barplot(loc,loc.index)
plt.xlabel('Count')
plt.title("Most popular locations serving Noth Indian,Chinese,South Indian and Biryani")


# Well Kormangala can be termed as most popular place in banglore which serves North Indian,Chinese,Biryani and South Indian Cuisines
