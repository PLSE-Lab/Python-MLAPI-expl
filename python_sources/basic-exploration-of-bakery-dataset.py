#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # data vizualisation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Many have already explored this dataset. Let's take a look anyways: it has about 21293 rows, corresponding to the number of items sold by the bakery; looks like the data is also very clean: no missing data. However, at the closer look we can see that number of items, namely 786, are listed as 'NONE'.

# In[ ]:


#Dataset overview
df = pd.read_csv('../input/BreadBasket_DMS.csv')
print(df.info())
print(df.head(5))
df[df['Item'] == 'NONE'].count()


# Let's clean it up a bit. Date in the existing form is somewhat cumbersome to work with; let's create separate columns for its parts - year, month, day, and hour of day. Minutes are probably not useful and we'll drop exact Time column. Also, let's remove the rows with 'NONE' items - not sure why does it appear? Cancelled purchase? Or some service sort of transaction? (This is entirely speculative)
# Number of different items ever sold in the bakery - 94.

# In[ ]:


df['Year'] = df.Date.apply(lambda x: x.split('-')[0])
df['Month'] = df.Date.apply(lambda x: x.split('-')[1])
df['Day'] = df.Date.apply(lambda x: x.split('-')[2])
df['Hour'] = df.Time.apply(lambda x: int(x.split(':')[0]))
df.drop(columns = 'Time', inplace = True)
df = df[df['Item'] != 'NONE']
unique_items = len(df['Item'].unique())
print('Unique items sold: ' + str(unique_items))


# This is pretty standard but here is the barplot of most selling items in the bakery. No surprses. Coffee wins by a large margin. Happy morning everyone!

# In[ ]:


sns.set(style = 'whitegrid')
sales = df['Item'].value_counts()
f = sales[:10].plot.bar(title = 'Top 10 sales')
f.legend(['Number of items sold'])


# Actually, morning? Let's see! How is coffee sold during the day? Well, coffee consumption is certainly shifted to the morning hours, but some are not afraid to drink coffee well after lunchtime.

# In[ ]:


coffee_sales = df[df['Item'] == 'Coffee']
coffee_times = coffee_sales['Hour'].value_counts().sort_index()
f = coffee_times.plot.bar(title = 'Coffee sales by hour')
f.set_xlabel('Time of day')


# What about other items? Let's look into time-dependent consumption of other frequently sold items in the bakery. Funny enough, tea consumption is shifted towards afternoon hours. Yes, you had your coffee in the morning, and too much may interfere with your sleep. Drink tea instead!
# Cookies and brownies are showing a bimodal distribution. I can see why... I can eat cookies all day! Have one with your coffee in the morning, another one with your tea in the afternoon.
# Unsurprisingly, sandwiches are for lunch. Also unsurprisingly, pastries are for breakfast. Medialunas are also for breakfast. I don't know if that's surprising or not. No idea what those are.

# In[ ]:


frequent_items = sales[1:10] #skip coffee
for item in frequent_items.index:
    plt.figure()
    curr_sales = df[df['Item'] == item]
    curr_times = curr_sales['Hour'].value_counts().sort_index()
    f = curr_times.plot.bar(title = (item + ' sales by hour'))
    f.set_xlabel('Time of day')


# Let's add a weekday to our dataset for each date and see how sales differ by the weekday. I guess there is not much information there. Saturdays are good days for business.

# In[ ]:


df['Day_of_week'] = pd.to_datetime(df['Date']).dt.weekday_name
sales_by_day = df['Day_of_week'].value_counts()
sales_by_day.plot.bar(title = 'Sales by day of week')


# Let's see what time we drink coffee on different days. Not surprising, Friday is a bit of a lazy day. Saturday is even better. Most coffee sales occur at 11 on Saturday and Sunday. On Sunday shop is also probably only open 9-5.

# In[ ]:


weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_coffee = df[df['Item'] == 'Coffee']
for day in weekdays:
    plt.figure()
    curr_sales = df_coffee[df_coffee['Day_of_week'] == day]
    curr_times = curr_sales['Hour'].value_counts().sort_index()
    curr_times.plot.bar(title = (day + ' coffee sales by hour'))


# Shop has only been open for about 6 months. How many transactions do we make each month? (Note, this is number of transactions, not items sold!)
# October 2016 bakery just opened, so let's ignore its little bar. November - it's popular! Probably people want to try it out. Then, December sales drop and sort of stay stable until March. Did people who try it out decide it was not that good? Or perhaps another shop opened nearby?
# A big drop in monthly sales from Nov to Dec can be explained by a large number of holidays in December. We'll look into those days in just a minute, but another way to see if number of sales dropped because of holidays or for other reasons is to look into average number of daily transactions for each month.

# In[ ]:


transactions_by_month = pd.DataFrame(df.groupby(by = ['Year', 'Month'])['Transaction'].nunique().rename('N transactions')).reset_index()
transactions_by_month['Date'] = transactions_by_month['Year'] + '-' + transactions_by_month['Month']
g = sns.barplot('Date', 'N transactions', data = transactions_by_month)
g.set_xticklabels(g.get_xticklabels(), rotation = 30)


# Hmmm. Looks like the daily number of transactions dropped in December too, so a bunch of holidays is probably not the only thing responsible for sales drop!

# In[ ]:


transactions_by_date = pd.DataFrame(df.groupby(by = ['Year', 'Month', 'Day'])['Transaction'].nunique().rename('Transactions a day')).reset_index()
transactions_by_date['Date'] = transactions_by_date['Year'] + '-' + transactions_by_date['Month']
g = sns.barplot('Date', 'Transactions a day', data = transactions_by_date)
g.set_xticklabels(g.get_xticklabels(), rotation = 30)


# Let's nevertheless see whethere there were any sales on holidays at all. Let's pick up the dates from Christmas Eve to New Year from the dataset and find number of transactions on those days too. Looks like people prepare for Christmas, buying sweets, or probably just meeting for coffee with friends and family. Then we are closed for 2 days, and then sales slowly recover into the New Year.

# In[ ]:


df_holidays = df[df['Month'] == '12']
df_holidays = df_holidays[df_holidays['Day'].isin(map(str, range(24, 32)))]
print(df_holidays.shape)
holiday_by_date = pd.DataFrame(df_holidays.groupby(by = ['Month', 'Day'])['Transaction'].nunique().rename('Transactions a day')).reset_index()
holiday_by_date['Date'] = holiday_by_date['Month'] + '-' + holiday_by_date['Day']
g = sns.barplot('Date', 'Transactions a day', data = holiday_by_date)
g.set_xticklabels(g.get_xticklabels(), rotation = 30)


# Thanks for reading!
