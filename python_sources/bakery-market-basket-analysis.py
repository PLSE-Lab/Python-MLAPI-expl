#!/usr/bin/env python
# coding: utf-8

# # **Market Basket Analysis for Beginner Data Science**
# ---
# 
# **Note: Hi Everyone, I'm Abdullah and I'm a new on Kaggle, This my detailed first article. **
# 
# In this tutorial, I'll show you how to analyze a data and how to Visualization a data, show to you. I hope you like it.

# <a class="anchor" id="0."></a>**Content:**
# 1. [Summary Of The Study](#1)
# 2. [Inclusion Of Required Libraries](#2)
# 3. [Inclusion Of The Data Set](#3)
# 4. [Contextual Evaluation Of The Data Set](#4)
# 5. [Technical Evaluation Of The Data Set](#5)
# 6. [Getting Information About the Data Set](#6)
# 7. [Data Set Column Listing](#7)
# 8. [Getting Statistical Information About The Data Set](#8)
# 9. [Check Whether The Value Is NONE](#9)
# 10. [List Of Sold Products](#10)
# 11. [Visualizations Of Sold Products With Pie Chart](#11)
# 12. [Visualizations Of Sold Products With Bar Chart](#12)
# 13. [Determination Of Most Sales Time Zones](#13)
# 14. [Determining The Best-selling Product In The Most-selling Time Frame](#14)
# 
# **Bonus**
# * How to use pandas: https://www.kaggle.com/abdullahsahin/step-by-step-pandas-tutorial-for-beginner

# <a class="anchor" id="1"></a>**1) Summary Of The Study**   ======>[Go to Content](#0.)
# 
# In this study, data analysis was performed using Matplotlib and Pandas libraries.
# 
# I'd like to make a brief statement before I go to case.
# 
# Why do we use the **Matplolib** library?
# 
# * Line plot is very good if you are using x axis.
# * Scatter is very good when correlating between two variables.
# * The histogram is very useful when you want to see the distribution of the numeric data.
# * This library has customization options.  For example; colors,labels,thickness of line, title, opacity, grid, figure, tick of axis and linestyle
# 
# Why Are We using the **Pandas** library?
# 
# * It has the ability to open different data sets.
# * It makes it easier for us.
# * Data filtering is easy.
# * Time-based data analysis is easy.
# * A speed-optimized library.

# <a class="anchor" id="2"></a>**2) Inclusion Of Required Libraries**   ======>[Go to Content](#0.)
# 
# In this tutorial we will use the **NumPy**, **Pandas** and **Matplotlip** libraries. I've included these libraries downstairs.
# 
# The print command prints the name of the database we are going to use on the screen.

# In[ ]:


# Inclusion Of Required Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a class="anchor" id="3"></a>**3) Inclusion Of The Data Set**   ======>[Go to Content](#0.)
# 
# In the code below, we assign the data set to a variable through **Pandas**.

# In[ ]:


data = pd.read_csv('../input/BreadBasket_DMS.csv')


# <a class="anchor" id="4"></a>**4) Contextual Evaluation Of The Data Set**   ======>[Go to Content](#0.)
# 
# The description of each data in the dataset is as follows.
# 
# * Date: Categorical variable that tells us the date of the transactions (YYYY-MM-DD format). The column includes dates from 30/10/2016 to 09/04/2017.
# * Time: Categorical variable that tells us the time of the transactions (HH:MM:SS format).
# * Transaction: Quantitative variable that allows us to differentiate the transactions. The rows that share the same value in this field belong to the same transaction, that's why the data set has less transactions than observations.
# * Item: Categorical variable with the products.

# <a class="anchor" id="5"></a>**5) Technical Evaluation Of The Data Set**   ======>[Go to Content](#0.)
# 
# *  RangeIndex: There are 21,293 data in the data set.
# *  There are 4 columns in the data set.
#     * Date Column:
#         * There are 21,293 data.
#         * There is no non-null data.
#         * The data type is objective.
#     * Time Column:
#         * There are 21,293 data.
#         * There is no non-null data.
#         * The data type is object.
#     * Transaction Column:
#         * There are 21,293 data.
#         * There is no non-null data.
#         * The data type is int64
#     * Item Column:
#         * There are 21,293 data.
#         * There is no non-null data.
#         * The data type is object.
#   * dtypes: There are a total of 3 int64 1 object type with a data type and in total.
#   * memory usage: Gives the total size of the data set.

# <a class="anchor" id="6"></a>**6) Getting Information About The Data Set**   ======>[Go to Content](#0.)

# In[ ]:


data.info()


# <a class="anchor" id="7"></a>**7) Data Set Column Listing**   ======>[Go to Content](#0.)
# 
# With the code found below, we first pulled the columns in our data set.

# In[ ]:


data.columns


# The code below brings us the data in the item column and does not repeat. There were 95 in total. So from here we can make a profit as if 95 products were sold.

# In[ ]:


print("List of Items sold at the Bakery:")
print("Total Items: ",len(data.Item.unique()))
print("-"*15)
for i in data.Item.unique():
    print(i)


# <a class="anchor" id="8"></a>**8) Getting Statistical Information About The Data Set**   ======>[Go to Content](#0.)

# data.describe() => The code only gives us some statistical information about columns with numeric values. This command does not provide the smallest detail.
# 
# If we write code as below, this Code provides a more comprehensive information. But in this code, it also tries to process columns that do not have numeric values. The column that will help us here is the transaction column. Let's examine the outputs.
# * count: It says how many data there are.
# * unique: It shows how many non-replicating data there are.
# * top: It specifies which of the most commonly found data.
# * freq: -
# * mean: Returns the average value.
# * std: standart deviation; returns the standard deviation value.
# * min: returns the minimum value.
# *25%: -
# *50%: -
# *75%: -
# * max: -

# In[ ]:


data.describe(include='all')


# <a class="anchor" id="9"></a>**9) Check Whether The Value Is NONE**   ======>[Go to Content](#0.)
# 
# We check the number of values in the data set, which are none in the item column.

# In[ ]:


len(data.loc[data["Item"] == "NONE",:])


# In the item column we bring only the first 10 of the data with no value.

# In[ ]:


data.loc[data["Item"] == "NONE",:].head(10)


# We have brought the last 10 of the values that are none in the item column in the data set.

# In[ ]:


data.loc[data["Item"] == "NONE",:].tail(10)


# I filtered item value none.

# In[ ]:


data = data[data.Item != 'NONE']
data.loc[data["Item"] == "NONE",:].tail(10)


# <a class="anchor" id="10"></a>**10) List Of Sold Products**   ======>[Go to Content](#0.)
# 
# Let's list the total number of products sold each time. We've only listed the first 15.

# In[ ]:


data["Item"].value_counts().head(15)


# <a class="anchor" id="11"></a>**11) Visualizations Of Sold Products With Pie Chart**   ======>[Go to Content](#0.)
# 
# Using the pie chart visualisation of the Matplotlib library, we have listed only 15 of the best-selling products.

# In[ ]:


# Pie Chart
plt.figure(1, figsize=(10,10))
data['Item'].value_counts().head(15).plot.pie(autopct="%1.1f%%")
plt.show()


# The best-selling products and the values of these products let us draw.

# In[ ]:


itemNames = data['Item'].value_counts().index
itemValues = data['Item'].value_counts().values


# <a class="anchor" id="12"></a>**12) Visualizations Of Sold Products With Bar Chart**   ======>[Go to Content](#0.)
# 
# We use the following structure when we want to show the first 15 of the best-selling products with bar chart in matplot.

# In[ ]:


plt.figure(figsize=(12,12))
plt.ylabel('Values', fontsize='medium')
plt.xlabel('Items', fontsize='medium')
plt.title('Top 20 Sell Bakery Items')
plt.bar(itemNames[:10],itemValues[:10], width = 0.7, color="blue",linewidth=0.4)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(12,12))
plt.style.use('fivethirtyeight')
ax.barh(itemNames[:5], itemValues[:5])
plt.show()


# <a class="anchor" id="13"></a>**13) Determination Of Most Sales Time Zones**   ======>[Go to Content](#0.)

# Which watches are important for companies and how much sales are made. I did this in the simplest way with the code below. It is discussed whether there is a logical method in software. I'm waiting for your suggestions in the comments section.
# 
# If I had to explain the code, I took advantage of the LoC function of the pandas library. This function performs the necessary filters in the data with the criteria specified.

# In[ ]:


firstMorning = data.loc[(data['Time']>='06:00:00')&(data['Time']<'09:00:00')]
secondMorning = data.loc[(data['Time']>='09:00:00')&(data['Time']<'12:00:00')]
firstAfternoon = data.loc[(data['Time']>='12:00:00')&(data['Time']<'15:00:00')]
secondAfternoon = data.loc[(data['Time']>='15:00:00')&(data['Time']<'18:00:00')]
night = data.loc[(data['Time']>='18:00:00')&(data['Time']<'21:00:00')]
hourlySales = {'firstMorning': len(firstMorning), 'secondMorning': len(secondMorning), 'firstAfternoon': len(firstAfternoon),'secondAfternoon': len(secondAfternoon),'night': len(night)}
print("This is night sales: ", hourlySales['night'])


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
ax.barh(range(len(hourlySales)), list(hourlySales.values()), align='center')
plt.show()


# <a class="anchor" id="14"></a>**14) Determining The Best-selling Product In The Most-selling Time Frame**   ======>[Go to Content](#0.)
# 
# Let's do a little deeper research. In the morning between 06-09 mesala sales, most of which we want to determine what is done to write the code will be as follows.

# In[ ]:


print(firstMorning['Item'].value_counts().head(15))


# In the morning, I showed only the top 5 of the best-selling products on bar plot.

# In[ ]:


# Bar Plot
plt.figure(figsize=(10,10))
plt.ylabel('Values', fontsize='medium')
plt.xlabel('Items', fontsize='medium')
plt.title('Top 20 Sell Bakery Items')
plt.bar(firstMorning['Item'][:5],firstMorning['Item'].value_counts()[:5], width = 0.7, color="blue",linewidth=0.4)
plt.show()


# In[ ]:


data['datetime'] = pd.to_datetime(data['Date']+" "+data['Time'])
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['weekday'] = data['datetime'].dt.weekday
data['hour'] = data['datetime'].dt.hour
data = data.drop(['Date'], axis=1)

data.head(5)


# In[ ]:


yearlyTable = data['year'].value_counts().plot(kind='bar',figsize=(10,5))
yearlyTable.set_xlabel("")
data['year'].value_counts().head()


# In[ ]:


data['monthlyTransaction'] = pd.to_datetime(data['datetime']).dt.to_period('M')
monthlyTransaction = data[['monthlyTransaction','Transaction']].groupby(['monthlyTransaction'], as_index=False).count().sort_values(by='monthlyTransaction')
monthlyTransaction.set_index('monthlyTransaction' ,inplace=True)

monthlyTable = monthlyTransaction.plot(kind='bar',figsize=(10,6))
monthlyTable.set_xlabel("")

monthlyTransaction


# In[ ]:


hourlyTransaction = data[['hour','Transaction']].groupby(['hour'], as_index=False).count()
hourlyTransaction.head(10)
hourlyTransaction.set_index('hour' ,inplace=True)

tableSort = hourlyTransaction.plot(kind='bar',figsize=(10,6))
tableSort.set_xlabel("")

hourlyTransaction


# In[ ]:


data['monthly'] = pd.to_datetime(data['datetime']).dt.to_period('M')
monthlyTransactionForItem = data[['monthly','Transaction', 'Item']].groupby(['monthly', 'Item'], as_index=False).count().sort_values(by='monthly')
monthlyTransactionForItem.set_index('monthly' ,inplace=True)

monthlyTransactionForItem.head(35)


# In[ ]:


cofeeSalesMonthly = monthlyTransactionForItem[monthlyTransactionForItem['Item']=='Coffee'].plot(kind='bar', figsize=(10,6))
cofeeSalesMonthly.set_xlabel("Coffee Sales Monthly")

plt.ylabel('Transaction', fontsize=16)
plt.xlabel('Month', fontsize=16)
plt.title("Monthly Coffee Sales", fontsize=16);


# In[ ]:


data['daily'] = pd.to_datetime(data['datetime']).dt.to_period('D')
dailyTransactionForItem = data[['daily','Transaction', 'Item']].groupby(['daily', 'Item'], as_index=False).count().sort_values(by='daily')
dailyTransactionForItem.set_index('daily' ,inplace=True)

dailyTransactionForItem.head(35)


# In[ ]:


data['hourly'] = pd.to_datetime(data['datetime']).dt.to_period('H')
hourlyTransactionForItem = data[['hourly','Transaction', 'Item']].groupby(['hourly', 'Item'], as_index=False).count().sort_values(by='hourly')
hourlyTransactionForItem.set_index('hourly' ,inplace=True)

hourlyTransactionForItem.head(35)


# **Note: Thank you for reading. Help me to improve myself. I hope it has been useful to you.**
