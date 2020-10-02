#!/usr/bin/env python
# coding: utf-8

# Hello everyone !! In this kernal session we will look into some basics of handling time series data.
# Time series data is nothing but a sequence of data points that measure the same variables over an ordered period of time. And it is also very powerful in many applications in reality. 
# Let's take a look at our Global Temperature variations over the years.
# ![](http://berkeleyearth.org/wp-content/uploads/2019/01/GlobalAverage_2018-1024x582.png)

# You can simply understand that we are moving towards a much hotter periods. Now we were able to say that due to the timestamped data in this graph. We call this as forecasting or in simple predicting the future.
# See, that's why time series data is so important. 

# Let's do a quick example.

# In[ ]:


import pandas as pd


# Let's get some real life data in. I will be using NETFLIX's stock values dataset for the recent quarter.
# You can always find stock value data through Google Finance and other sites and download these datasets. 
# I downloaded from [here](https://www.nasdaq.com/symbol/nflx/historical), and i have already included it in this kernal. Do check out similar datasets.

# When you run the below lines, you will see your dataset all set. But take a look at the additional values i have passed in the read_csv().
# * **parse_dates=\['date'\]** is used to convert the date column datatype to date. This is because normally the datatype of such data sets is read as String. So we need to parse into the date format to do operations.
# * **index_col='date'** This is a really cool method to assign the index of your dataset to your desired column. Since we are going to work with time/date here, setting it as index will make it easy to do operations.
# 
# > For you to understand better, try running these codes without passing these values, 
# like **netflix = pd.read_csv('../input/NetflixStocks.csv')** only and then check the dtype of date.
# 

# In[ ]:


netflix = pd.read_csv('../input/NetflixStocks.csv', parse_dates=['date'], index_col='date')
netflix.head(5)


# In[ ]:


type(netflix.index[0]) #Use this to check the type without assiging the index, then use type(netflix.date[0])


# Ok let's see how we can make use of having the date column as the index.
# Let's say you need to see the stock values for the 6th month only.

# In[ ]:


netflix['2019-06'] #As you will see it was a simple work of giving a partial string to get the data when you have date as the index


# In[ ]:


netflix['2019-06'].close.mean() #This will give you the average stock price for the month of June


# We can also check the monthly average value by using the resample(). You can play with the values passed to resample(). Do check out the documentation on the official pandas site or click **shift + tab** at the resample(). 

# In[ ]:


netflix.close.resample('M').mean() 


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
netflix.close.resample('M').mean().plot() #Plotting a graph will help us to understand the basic details.


# Here we can see the average stock prices have reduced drastically by the end of May but increased rapidly during the month of June. Such information is always vital in making future predictions leading to important decisions. But this is a very simple example.We will do a complex example in a different session.

# Okay let's see what we get when we plot a graph for all days in a month. Let's try June.

# In[ ]:


netflix['2019-06'].close.resample('D').mean().plot()


# As you can see there are broken gaps between the lines. This is because our dataset does not contain all days, i.e only Business days are taken in. So how can we overcome this. Let's checkout.

# We can use the asfreq() method for this purpose. But first we need to sort our index inorder for asfreq() to work.

# In[ ]:


netflix = netflix.sort_index().asfreq(freq='D', method='pad')


# In[ ]:


netflix['2019-06'].close.resample('D').mean().plot()


# Now when you plot the graph, you will see the gaps have been filled with the previous values. This helps us to have a better understanding on our data.

# Before we end this first part, we will look into one more issue we come across in real world datasets.
# Think of a datset that has been put together by various individuals, from different countries. Now the date format may vary from person top person. But in order to put them all together in one and work with the data you need to keep them all in a single format. How can we do that. Let's do simple one.
# Let's take a list of dates in various format. In order to get them to a single format we use the to_datetime().

# In[ ]:


dates = ['2019-10-07', 'Jul 7, 2019', '01/01/2019', '2019.01.01', '2019/07/10', '20170710']
pd.to_datetime(dates)


# As you can see, we have converted all the dates to a single format.
# Check out the documentation on what else you can do with this method.

# We will look into more on Time Series Data analysis in the next kernal session

# To learn about data visualizing techniques, try this kernal on ["Data Visualizing Techniques using Seaborn"](https://www.kaggle.com/saadzarook/data-visualization-techniques-using-seaborn)
