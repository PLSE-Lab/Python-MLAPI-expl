#!/usr/bin/env python
# coding: utf-8

# ![Pandas groupby](https://us.toluna.com/dpolls_images/2018/08/25/620535cf-e152-4bad-a02c-8372063b9fe1.jpg)

# Hello, everyone,
# 
# I just finished working with the **Groupby function by itself and along with .apply(), .filter(), .aggregate(), .transform()** and so on, and thought it might be usefull to some folks too.
# 
# Please, let me know what you think and how it can be made better. Any comments and suggestions are welcome!
# 
# Do not forget to **UPVOTE** my Notebook if you found it useful! 
# 
# Thank you, everyone)))

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', index_col = 'date', parse_dates = True )
df.head()


# In[ ]:


df.describe().T


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# Love this summary:

# In[ ]:


def summary(df):
    
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nas = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing = (df.isnull().sum() / df.shape[0]) * 100
    sk = df.skew()
    krt = df.kurt()
    
    print('Data shape:', df.shape)

    cols = ['Type', 'Total count', 'Null Values', 'Distinct Values', 'Missing Ratio', 'Unique Values', 'Skewness', 'Kurtosis']
    dtls = pd.concat([types, counts, nas, distincts, missing, uniques, sk, krt], axis=1, sort=False)
  
    dtls.columns = cols
    return dtls


# In[ ]:


details = summary(df)
details


# Setting up an Index

# In[ ]:


df.index.names = ['Date']
df.head()


# Dropping non-useful column:

# In[ ]:


df = df.drop(['date_block_num'], axis = 1)
df.head()


# In[ ]:


df.rename(columns={'shop_id':'Store ID', 'item_id':'Item ID', 'item_price':'Price', 'item_cnt_day':'Volume'}, inplace = True)
df.head()


# Creating new column Daily Revenue

# In[ ]:


df['Daily Revenue'] = df['Price']*df['Volume']
df.head()


# ## Groupby Method:

# Looking at the average price for each store:

# In[ ]:


df[['Store ID','Price']].groupby('Store ID').mean().head()


# ### Note: By specifying the column ['Daily Revenue'] before the aggregate function .sum(), you will significantly improve your speed.

# ### Note 2: To return a dataframe provide the column name as a list to the comumn selection by using double brackets like that: [['Volume']]

# In[ ]:


df.groupby('Store ID')[['Volume']].sum().head()


# ### This way, it can take a list of column names and perform an aggregation on all of the comumns based on Store ID and Store's Volume by using (['Name 1', 'Name 2'])

# In[ ]:


df_ml = df.groupby(['Store ID', 'Volume']).sum()
df_ml.head()


# ### Note 3: .xs can be used to select subsets of DF while working with MultiIndex tables like this:

# Tip: For a String use '39'

# In[ ]:


df_ml.xs(39, level = 'Store ID').head()


# ### groupby() while working with the Time Series.

# Daily Revenue summed up into Monthly Reevenue for every store using .sum() method:

# In[ ]:


df_monthly = df.reset_index().groupby(['Store ID', pd.Grouper(key='Date', freq = 'M')])[['Daily Revenue']].sum().rename(columns = {'Daily Revenue':'Monthly Revenue'})
df_monthly.head()


# ### .size() method counts rows in each group

# In[ ]:


df.reset_index().groupby(pd.Grouper(key='Date', freq='Q')).size()


# ### Note 4: using reset_index() method resets the index into columns

# Tip: by passing drop = True to the reset_insex(), your dff drops the columns that make up the MultiIndex and creates a new index with integer values.

# ### Total Daily Revenue:

# In[ ]:


df['Daily Revenue'].reset_index().groupby('Date', as_index = False).sum().rename(columns = {'Daily Revenue':'Total Daily Revenue'}).head()


# Another way to look at the data with groupby(). I sort it by Store ID here.

# In[ ]:


grouped = df.reset_index().groupby('Store ID')
grouped.head()


# It's supposed to be a DataFrameGroupBy:

# In[ ]:


type(grouped)


# Data filtered out only for store # 36

# In[ ]:


grouped.get_group(36).head()


# ## Size & pd.qcut() Method:

# Let's review: How to count rows in each group by using .size()

# In[ ]:


grouped.size().head()


# Cutting values into 3 equal buckets:

# In[ ]:


df.groupby(pd.qcut(x = df['Price'], q=3, labels=['Low', 'Medium','High'])).size()


# Alloccating Daily Revenue values into custom-sized buckets by specifying the bin boundaries:

# In[ ]:


df.groupby(pd.cut(df['Daily Revenue'], [0,500,1000,2500,5000,10000,50000, 100000, 175000, 250000, 500000, 750000, 1000000, 1250000])).size()


# Grouping by multiple columns:

# In[ ]:


df.groupby(['Store ID', 'Price']).size().head(10)


# In[ ]:


df_1 = df.loc[df['Store ID']==(59)]
df_1.groupby(['Volume','Price', 'Daily Revenue']).size().head(10)


# ## Aggregate Method:

# Aggregate by sum and mean:

# In[ ]:


df[['Price','Volume','Daily Revenue']].agg(['sum', 'mean'])


# Passing a dictionary to agg and specify operations for each column:

# In[ ]:


df.agg({'Price':['mean'], 'Volume':['sum','mean'], 'Daily Revenue':['sum','mean']})


# And now, using groupby and aggregate functions together, we get:

# In[ ]:


def my_agg(x): 
    names = { 
        'PriceMean': x['Price'].mean(),
        'VolumeMax': x['Volume'].max(), 
        'DailyRevMean': x['Daily Revenue'].mean(),
        'DailyRevMax': x['Daily Revenue'].max()
    }

    return pd.Series(names, index=[ key for key in names.keys()])

df.groupby('Store ID').apply(my_agg).head(10)


# Another, but not as useful way:

# In[ ]:


df.groupby('Store ID').agg({'Price':['mean'], 'Volume':['sum','mean'], 'Daily Revenue':['sum','mean']}).head()


# To make it horizontal, use transpose function at the end.

# In[ ]:


df.groupby('Store ID').agg({'Price':['mean'], 'Volume':['sum','mean'], 'Daily Revenue':['sum','mean']}).T


# ### The most frequently used functions:
# 

# * 'size' = counts the rows
# * 'sum' = summs up
# * 'max/min' = Maximum/Minimum
# * 'mean/median' = Mean/ Median
# * 'idxmax/idxmin' = Column's index of the max/min
# * pd.Series.nunique = Counts unique values

# ## Apply() Function

# First, let's take a look at a few examples how apply() function works:

# Note: By default, it applies the function along axis = 0 or for each column

# In[ ]:


df.apply(sum)


# By specifying axis = 1, it will apply the function for the rows

# ### So-called Groupby-Split-Apply-Combine chain mechanism

# ![Mechanism](https://pbs.twimg.com/media/CycthXVXgAAazkz?format=jpg&name=small)

# 1. First, *Groupby*, splits the data into groups by creating a groupby object or DataFrameGroupBy. 
# 2. Next, *Apply* method, applies a fuction. 
# 3. And the last step, *Combine*, combines all of the results in a single output.

# For example, applying a lambda function

# In[ ]:


df.groupby('Store ID').apply(lambda x:x.mean()).head()


# Here is another way that produces the same results

# In[ ]:


def df_mean(x):
    return x.mean()
df.groupby('Store ID').apply(df_mean).head()


# Apply() function can be also apply=ied to the Time Series:

# In[ ]:


df.reset_index().groupby(pd.Grouper(key = 'Date', freq = 'Q'))['Volume'].apply(sum)


# ## Transform function

# While, for example, aggregate function reduced DF, this function just transforms our DF. That is why it's normally used to assign to a new column

# In[ ]:


df['Volume %'] = df.groupby('Store ID')[['Volume']].transform(lambda x: x/sum(x)*100)
df.head()


# ## Filter Method

# Filtering does not change the data, but only selects a subset:

# In[ ]:


df.groupby('Store ID').filter(lambda x: x['Daily Revenue'].mean() > 1000).head()

