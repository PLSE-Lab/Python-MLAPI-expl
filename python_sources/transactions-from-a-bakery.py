#!/usr/bin/env python
# coding: utf-8

# ![The Bread Basket](https://storage.googleapis.com/kaggle-datasets-images/50231/92819/043118a1f4b815eb271992ea0f83987c/dataset-cover.jpg?t=2018-09-05-23-50-57=s2560 "The Bread Basket")
# 
# ## Context
# The data belongs to a bakery called **"The Bread Basket"**, located in the historic center of *Edinburgh*. 
# This bakery presents a refreshing offer of *Argentine* and *Spanish* products.
# 
# ## Content
# Data set containing 15,010 observations and more than 6,000 transactions from a bakery. The data set contains the following columns:
# 
# 1. ***Date:-***<br>
#  Categorical variable that tells us the date of the transactions __*(YYYY-MM-DD format)*__.<br>
#  The column includes dates from 30/10/2016 to 09/04/2017.
# 
# 2. ***Time:-***<br>
# Categorical variable that tells us the time of the transactions __*(HH:MM:SS format)*__.
# 
# 3. ***Transaction:-***<br>
# Quantitative variable that allows us to differentiate the transactions. The rows that share the same value in this field belong to the same transaction, that's why the data set has less transactions than observations.
# 
# 4. ***Item:-***<br>
# Categorical variable with the products.
# 
# ## Inspiration
# 1. Market Basket Analysis
# 
# 2. Apriori algorithm

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
from pandas_summary import DataFrameSummary
from IPython.display import display
from datetime import datetime

from fastai.imports import *
from fastai.structured import *
import seaborn as sns

sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=1.9,rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})              
plt.figure(figsize=(3.1, 3)) # Two column paper. Each column is about 3.15 inch wide.

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


trans = pd.read_csv("../input/BreadBasket_DMS.csv",parse_dates=['Date'])
display(trans.T)


# In[ ]:


display_all(trans.describe(include='all').T)


# In[ ]:


trans.info()


# In[ ]:


trans['Time'] =  pd.to_datetime(trans['Time'], format='%H:%M:%S')
trans.Item = trans.Item.astype('category')


# In[ ]:


trans.Item.value_counts()[:10]


# We can see we have `NONE` as item in the respective column. 
# 
# For now we will remove this item

# In[ ]:


trans = trans[trans.Item != 'NONE']


# In[ ]:


trans.Item.cat.categories


# In[ ]:


trans['Item_codes'] = trans.Item.cat.codes
display_all(trans.isnull().sum().sort_index()/len(trans))


# ## Data Processing:-

# In[ ]:


trans.head(10)


# In[ ]:


def add_datetime_features(df):
    # sleep: 12-5, 6-9: breakfast, 10-14: lunch, 14-17: dinner prep, 17-21: dinner, 21-23: deserts!
    df['Time'] = pd.DatetimeIndex(df['Time']).time
    hour = df['Time'].apply(lambda ts: ts.hour)
    df['Hour'],df['Time_Of_Day'] = hour,hour
    df['Time_Of_Day'].replace([i for i in range(0,6)], 'Sleep',inplace=True)
    df['Time_Of_Day'].replace([i for i in range(6,10)], 'Breakfast',inplace=True)
    df['Time_Of_Day'].replace([i for i in range(10,14)], 'Lunch',inplace=True)
    df['Time_Of_Day'].replace([i for i in range(14,17)], 'Dinner Prep',inplace=True)
    df['Time_Of_Day'].replace([i for i in range(17,21)], 'Dinner',inplace=True)
    df['Time_Of_Day'].replace([i for i in range(21,24)], 'Deserts',inplace=True)
    df.drop('Time',axis=1,inplace=True)  

    
    df['Season'] = pd.DatetimeIndex(df['Date']).month
    df['Season'].replace([1,2,12], 'Winter',inplace=True)
    df['Season'].replace([i for i in range(3,6)], 'Spring',inplace=True)
    df['Season'].replace([i for i in range(6,9)], 'Summer',inplace=True)
    df['Season'].replace([i for i in range(9,12)], 'Fall',inplace=True) 
    
    add_datepart(df, 'Date')
    
    return df


# In[ ]:


trans = add_datetime_features(trans)
trans.head(10)


# In[ ]:


trans.pivot_table(index='Season',columns='Item', aggfunc={'Item':'count'}).fillna(0)


# In[ ]:


trans.pivot_table(index='Time_Of_Day',columns='Item', aggfunc={'Item':'count'}).fillna(0)


# In[ ]:


trans.pivot_table(index='Year',columns='Item', aggfunc={'Item':'count'}).fillna(0)


# # Visualization
# 
# ## Top Sales Products:-

# In[ ]:


plt.figure(figsize=(20,10))
trans['Item'].value_counts()[:20].sort_values().plot.barh(title='Top 20 Sales',grid=True)


# ### **Insights**
# 
# From the above plot we can conclude that the mostly purchased products are :-
# 1. Coffee
# 2. Bread
# 3. Tea
# 4. Cake
# 5. Pastry

# ## Least Bought Products

# In[ ]:


plt.figure(figsize=(20,10))
trans['Item'].value_counts()[-20:-1].sort_values().plot.barh(title='Top 20 Least Sales',grid=True)


# ### **Insights**
# 
# From the above plot we can conclude that the least purchased products are :-
# 1. Polenta
# 2. Gift voucher
# 3. Bacon
# 4. Raw bars
# 5. Adjustment
# 

# ## Hourly Trend based on Activities of Day

# In[ ]:


df1=trans[['Transaction', 'Month', 'Year', 'Time_Of_Day','Dayofweek','Hour','Is_year_end','Is_year_start','Is_month_end','Is_month_start','Season']]
df1=df1.drop_duplicates()
plt.figure(figsize=(20,10))
sns.countplot(x='Hour',data=df1,hue='Time_Of_Day').set_title('General Transation Trend Throughout The Day',fontsize=25)


# ### **Insights**
# 
# From the above plot we can conclude that -
# +  Most of the transaction are made during **Lunch**
# + Also trancsaction during **Breakfast** and **Dinner Prep** also fairly significant.
# 
# 
# 

# ## Weekly Trend

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='Dayofweek',data=df1).set_title('Pattern of Transation Trend Throughout The Week',fontsize=25)


# ### **Insights**
# 
# From the above plot we can conclude that -
# +  Most of the transaction are made during **Saturday**
# + Also trancsaction during **Friday** and **Sunday** also fairly significant.

# ## Seasonality Trend

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='Season',data=df1).set_title('Pattern of Transation Trend During Different Season\'s',fontsize=25)


# ### **Insights**
# 
# From the above plot we can conclude that most of the transaction are made during **Winters**

# ## Yearly Trend

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='Year',data=df1,hue='Is_year_start').set_title('Transation Trend During Year Start',fontsize=25)


# ### **Insights**
# 
# From the above plot we can conclude that  the transactions are being affected by **Start of Year** 

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='Year',data=df1,hue='Is_year_end').set_title('Transation Trend During Year End',fontsize=25)


# ### **Insights**
# 
# From the above plot we can conclude that  the transactions are being affected by **End of Year** 

# ## Monthly Trend

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='Month',data=df1,hue='Is_month_start').set_title('Transation Trend During Month Start',fontsize=25)


# ### **Insights**
# 
# From the above plot we can conclude that  the transactions are being affected by **Start of Month** 

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='Month',data=df1,hue='Is_month_end').set_title('Transation Trend During Month End',fontsize=25)


# ### **Insights**
# 
# From the above plot we can conclude that  the transactions are being affected by **End of Month** 

# # To-Dos:- 
# + Find Association Rules
# + Find Apriori
