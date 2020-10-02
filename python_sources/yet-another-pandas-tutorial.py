#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc" style="margin-top: 1em;"><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Pandas" data-toc-modified-id="Pandas-0.1">Pandas</a></span><ul class="toc-item"><li><span><a href="#Creating-dataframes" data-toc-modified-id="Creating-dataframes-0.1.1">Creating dataframes</a></span></li><li><span><a href="#Utility-functions-for-quick-exploration" data-toc-modified-id="Utility-functions-for-quick-exploration-0.1.2"><strong>Utility functions for quick exploration</strong></a></span></li><li><span><a href="#Efficient-reading-of-data-in-pandas" data-toc-modified-id="Efficient-reading-of-data-in-pandas-0.1.3"><strong>Efficient reading of data in pandas</strong></a></span></li><li><span><a href="#Pandas-cleaning,-indexing-and-exploration" data-toc-modified-id="Pandas-cleaning,-indexing-and-exploration-0.1.4">Pandas cleaning, indexing and exploration</a></span><ul class="toc-item"><li><span><a href="#Series-vs-dataframe" data-toc-modified-id="Series-vs-dataframe-0.1.4.1"><strong>Series vs dataframe</strong></a></span></li><li><span><a href="#Explore" data-toc-modified-id="Explore-0.1.4.2"><strong>Explore</strong></a></span></li><li><span><a href="#Cleaning-and-processing" data-toc-modified-id="Cleaning-and-processing-0.1.4.3"><strong>Cleaning and processing</strong></a></span></li><li><span><a href="#Filtering-and-indexing" data-toc-modified-id="Filtering-and-indexing-0.1.4.4"><strong>Filtering and indexing</strong></a></span></li><li><span><a href="#Categorical-data" data-toc-modified-id="Categorical-data-0.1.4.5">Categorical data</a></span></li><li><span><a href="#Datetime-format" data-toc-modified-id="Datetime-format-0.1.4.6">Datetime format</a></span></li></ul></li><li><span><a href="#Pandas-manipulation" data-toc-modified-id="Pandas-manipulation-0.1.5">Pandas manipulation</a></span><ul class="toc-item"><li><span><a href="#Tidy-data" data-toc-modified-id="Tidy-data-0.1.5.1"><strong>Tidy data</strong></a></span></li><li><span><a href="#Adding-new-columns" data-toc-modified-id="Adding-new-columns-0.1.5.2">Adding new columns</a></span></li><li><span><a href="#Concatenating-dataframes" data-toc-modified-id="Concatenating-dataframes-0.1.5.3"><strong>Concatenating dataframes</strong></a></span></li><li><span><a href="#Setting-indexes" data-toc-modified-id="Setting-indexes-0.1.5.4">Setting indexes</a></span></li><li><span><a href="#Aggregation" data-toc-modified-id="Aggregation-0.1.5.5"><strong>Aggregation</strong></a></span></li><li><span><a href="#Merging-and-joins" data-toc-modified-id="Merging-and-joins-0.1.5.6">Merging and joins</a></span></li><li><span><a href="#Applying-function" data-toc-modified-id="Applying-function-0.1.5.7">Applying function</a></span></li><li><span><a href="#Dropping-columns" data-toc-modified-id="Dropping-columns-0.1.5.8">Dropping columns</a></span></li><li><span><a href="#Renaming-columns" data-toc-modified-id="Renaming-columns-0.1.5.9">Renaming columns</a></span></li><li><span><a href="#Exporting-pandas" data-toc-modified-id="Exporting-pandas-0.1.5.10">Exporting pandas</a></span></li></ul></li><li><span><a href="#Detailed-discussion-of-aggregation" data-toc-modified-id="Detailed-discussion-of-aggregation-0.1.6">Detailed discussion of aggregation</a></span></li></ul></li></ul></li></ul></div>

# **This notebook I have tried to put together all the common functionalities in pandas.**

# In[720]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd


# In[721]:


# !ln -s ~/data/ data #creating symlink


# I'm using pokemon dataset for this notebook. Link to the dataset: https://www.kaggle.com/shikhar1/complete-seaborn-tutorial-pokemon/data

# In[ ]:





# ## Pandas

# ### Creating dataframes

# In[722]:


data = pd.read_csv('../input/pokemon.csv')


# Creating series

# In[723]:


s = pd.Series([1, 3, 5, np.nan, 6, 8])


# In[724]:


s


# Dataframe from dictionaries

# In[725]:


d = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}


# In[726]:


d


# In[727]:


pd.DataFrame.from_dict(d)


# ### **Utility functions for quick exploration**

# Looking at the top few rows

# In[728]:


data.head(3)


# Looking at the last few rows

# In[729]:


data.tail(3)


# `info()` function outputs datatypes for each column, number of non-missing values and memory usage by the dataframe

# In[730]:


data.info(True)


# If you're just interested in the memory usage by column you can use `memory_usage()`. This way you can identify the columns which are taking the maximum memory

# In[731]:


data.memory_usage(deep=True)


# For looking at the datatypes: `dtypes`

# In[732]:


data.dtypes


# Pairwise Correlation plot between columns

# In[733]:


#correlation plot
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.5)


# In[734]:


data.columns  #listing the columns


# In[735]:


data.shape  #shape of the dataframe


# Finding missing values by column

# In[736]:


data.isnull().sum()


# Finding number of unique values per column

# In[737]:


data.nunique()


# In[738]:


data.sort_values(by='Attack', ascending=False)


# In[ ]:





# ### **Efficient reading of data in pandas**

# Many times we come across large datasets. For memory efficient loading of data we can use some tricks. This comes handy while doing quick experimentation with datasets

# For this I have taken data from this link : https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

# **Segue into magic commands**

# `%%timeit`: run the cell multiple times and then give mean time and std deviation

# `%%time`: run the cell one time and output the time taken

# `% vs %%`: % runs just the next line in the cell but %% is for the whole cell

# Data occupies ~800 MB on disk

# As the dataset is big I have commented out this section but I have shared the link for data. Download and then run this portion

# In[739]:


# %%timeit -r 3 #-r option to specify number of loops
# data1 = pd.read_csv('data/talking-data/test.csv')


# In[740]:


# %%time
# data1 = pd.read_csv('data/talking-data/test.csv')


# In[741]:


# %time
# data1.head()
# data1 = pd.read_csv(
#     'data/talking-data/test.csv')  #this won't be evaluated for runtime


# In[742]:


# data1.info(verbose=1)


# We can see the datatype used is `int64` and `object` which occupies a lot of memory. Our dataframe takes ~1000 MB on memory. Let's see if we actually require it

# Refer to this link for common datatypes and how much memory they store: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html

# In[743]:


# data1.memory_usage()


# In[744]:


# data1.head()


# For this we'll look at the minimum and maximum values for each column. We can use `describe()` for this. We don't have categorical data but in case of categorical columns they are generally stored as `object` which is inefficient. Explicitly specify them as `category` dtype while loading data

# In[745]:


# data1.describe()


# Based on the maximum values we can specify the dtypes which can accomodate the values and take minimum space. Defining dtypes beforehand greatly decreases the storage

# In[746]:


# %%time
# dtypes = {
#     'click_id': 'uint32',
#     'ip': 'uint32',
#     'app': 'uint16',
#     'device': 'uint16',
#     'os': 'uint16',
#     'channel': 'uint16'
# }
# data1 = pd.read_csv('data/talking-data/test.csv', dtype=dtypes)


# In[747]:


# data1.info(verbose=1)


# We can see that memory usage has decreased to ~430 MB which is more than 50% reduction

# **dask**

# Dask is used for parallelized operations on pandas dataframe. Here I'm showing how to use it for faster loading of data but it can be used for many more operations. In the background it stores the data in partitions

# In[748]:


# %%time
# df = dd.read_csv('data/talking-data/test.csv', dtype=dtypes)
# df = df.compute()  #.compute() converts dask dataframe back to pandas


# We can see that the wall time is 13s compared to 21s

# In[749]:


# df.info(verbose=1)


# **Back to pandas**

# You can choose which rows to load using `nrows`. This will load first 100 rows. This can be used to quickly insepct data

# In[750]:


# %%time
# data1 = pd.read_csv('data/talking-data/test.csv', dtype=dtypes, nrows=100)


# In[751]:


# data1.shape


# In[752]:


# data1.tail()


# You can skip rows also using `skiprows`

# In[753]:


# data1 = pd.read_csv(
#     'data/talking-data/test.csv', dtype=dtypes, nrows=100, skiprows=99)


# In[754]:


# data1.head()


# see the last and the first row is matching

# You can specify row number also for skipping

# In[755]:


# data1 = pd.read_csv(
#     'data/talking-data/test.csv',
#     dtype=dtypes,
#     nrows=100,
#     skiprows=range(1, 100))


# In[756]:


# data1.head()


# In[757]:


# data1 = pd.read_csv(
#     'data/talking-data/test.csv', dtype=dtypes, nrows=100, skiprows=[1, 3, 5])


# In[758]:


# data1.head()


# ### Pandas cleaning, indexing and exploration

# Back to pokemon data :)

# #### **Series vs dataframe**

# Series is the datastructure for a single column of a DataFrame, not only conceptually, but literally i.e. the data in a DataFrame is **actually stored in memory as a collection of Series.**: https://stackoverflow.com/a/26240208

# In[759]:


data.head()


# `[]` invokes series whereas `[[]]` returns a dataframe

# In[760]:


data['Name'][0:4]  #pandas series


# In[761]:


data.Name[0:4]  #we can also use .


# In[762]:


data[['Name']][0:4]  #pandas dataframe


# #### **Explore**

# `value_count()`:Quick count of obs for each level: especially useful for categorical data. This doesn't count NAs

# In[763]:


data['Type 1'].value_counts()  #can be called on a series and not a dataframe


# For NA inclusion

# In[764]:


# if there are nan values that also be counted
data['Type 1'].value_counts(dropna=False)


# Basic summary stats can be calles for numerical columns

# In[765]:


data.HP.mean()


# In[766]:


data.HP.max()


# In[767]:


data.HP.count()


# Basic plots are also there as methods for dataframe

# In[768]:


data.boxplot(column='Attack', by='Legendary')


# Transposing dataframe

# In[769]:


data.T


# This will just show numeric columns

# In[770]:


data.describe(include=['number'])


# Selecting columns by data type

# In[771]:


data.select_dtypes(include=['category'])


# #### **Cleaning and processing**

# In[772]:


data.isnull().sum()


# `type 1` has null values

# In[773]:


data.shape


# **Typecasting**

# In[774]:


data.dtypes


# In[775]:


data['Name'] = data['Name'].astype('category')


# In[776]:


data.dtypes


# **Missing values**

# NaN and None (in object arrays) are considered missing by the isnull and notnull

# For datetime64[ns] types, NaT represents missing values

# `dropna()`for removing rows containing missing values. `any` will remove rows with any NA and `all` requires the whole row to be NA

# In[777]:


data.dropna(axis=0, how='any').shape


# In[778]:


data.dropna(axis=0, how='all').shape


# `fillna()` for replacing NA values

# In[779]:


data['Type 1'].fillna('unknown').isnull().sum()


# **Replacing values**

# In[780]:


data.Name.replace('Bulbasaur', 'Bulba')[:4]


# Inplace transformations. Many of these functions have inplace argument. You can use that to make your code shorter. **But beware that you'll lose the original dataframe**

# In[781]:


data.dropna(axis=0, how='any', inplace=True)


# In[782]:


data.isnull().sum()


# **Duplicates**

# In[783]:


data.duplicated('Type 1')[:4]  #check for duplicates within a column


# In[784]:


data.Legendary.drop_duplicates(
    keep='last')  #remove duplicates and keep the last entry within duplicates


# `unique()` to get the unique values within a column

# In[785]:


data['Type 1'].unique()


# #### **Filtering and indexing**

# Different ways of indexing

# `loc`: using name and `iloc`: integer index is is used for indexing

# In[786]:


data.loc[1:4, ["HP"]]


# In[787]:


data.iloc[1, 1]


# In[788]:


data.loc[1:3, ["HP", "Attack"]]


# In[789]:


data[["HP", "Attack"]][:3]


# Conditional filtering

# In[790]:


data.loc[(data['Defense'] > 200) & (data['Attack'] > 100)]  #and


# In[791]:


data.loc[(data['Defense'] > 200) | (data['Attack'] > 100)]  #or


# In[792]:


my_pokemon = ['Volcanion', 'Bulbasaur']


# filtering based on a list

# In[793]:


data.loc[(data['Name'].isin(my_pokemon))]


# In[794]:


data.Attack.where(
    data.Attack > 49,
    49)  #filter the data greater than 49 and replace every other value to 49


# #### Categorical data

# In[795]:


data.dtypes


# For looking at different levels of the categorical variable

# In[796]:


data.Name.cat.categories


# check whether the categories are ordered or not

# In[797]:


data.Name.cat.ordered


# In[798]:


data.Attack.max()


# In[799]:


def create_bin(x):
    if x < 80:
        return 'low'
    elif x >= 80 and x < 120:
        return 'medium'
    else:
        return 'high'


# In[800]:


data['attack_level'] = data.Attack.apply(create_bin)


# In[801]:


data.head()


# In[802]:


data['attack_level'] = data.attack_level.astype('category')


# In[803]:


data.attack_level.cat.categories


# categorical column is stored as number in the background

# In[804]:


data.attack_level.cat.codes


# renaming the categories

# In[805]:


data.attack_level.cat.categories = ['H', 'L', 'M']


# In[806]:


data.attack_level.cat.categories


# In[807]:


data.head()


# In[808]:


data.attack_level.cat.reorder_categories = ['H', 'M', 'L']


# In[809]:


data.attack_level.cat.categories


# #### Datetime format

# `pd.datetime` is generally used to convert string to datetime format. There are many other kinds of date formats

# In[810]:


number = np.arange(0, 5)
date_list = [
    "1992-01-10", "1992-02-10", "1992-03-10", "1993-03-15", "1993-03-16"
]
df = pd.DataFrame({'a': pd.to_datetime(date_list), 'b': number})
df


# In the dt object of datetime column, we have functions to extract day,year, month, day of week,....

# In[811]:


df.a.dt.day


# In[812]:


df.a.dt.year


# ### Pandas manipulation

# #### **Tidy data**

# **Wide --> Long**

# In[813]:


data.head()


# In[814]:


# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(
    data,
    id_vars='Name',
    value_vars=['Attack', 'Defense'],
    var_name='skill-type',
    value_name='value')


# In[815]:


melted.head()


# **Long --> Wide**

# In[816]:


melted.pivot(index='Name', columns='skill-type', values='value').head()


# #### Adding new columns

# Both doing the same thing. Creating new columns based on existing columns

# In[817]:


data.assign(Nick=lambda x: 2 * x.Attack + 2 * x.Defense).head()


# In[818]:


data['Nick'] = data.apply(lambda x: x.Attack + x.Defense, axis=1)


# In[819]:


data.head()


# Adding columns from a new datastructure

# In[820]:


data['random'] = np.random.randint(0, 100, data.shape[0])


# In[821]:


data.head()


# #### **Concatenating dataframes**

# Vertical concatenation

# In[822]:


data1 = data.head()
data2 = data.tail()
data1.append(data2)


# In[823]:


pd.concat([data1, data2], axis=0, ignore_index=True)  #stack by rows


# Horizontal concatenation

# In[824]:


data1 = data['Attack'].head()
data2 = data['Defense'].head()
conc_data_col = pd.concat(
    [data1, data2], axis=1)  # axis = 0 : stack by columns
conc_data_col


# #### Setting indexes

# Single index

# In[825]:


data.set_index('Type 1', inplace=True)


# In[826]:


data.head()


# multi-level index

# In[827]:


data1 = data.set_index(["Type 2", "Legendary"])


# #### **Aggregation**

# * Splitting the data into groups based on some criteria
# * Applying a function to each group independently
# * Combining the results into a data structure

# In[828]:


data.head(4)


# NA groups in GroupBy are automatically excluded.

# In[829]:


data.groupby('Type 1').mean()  #it skipped categorical columns


# In[830]:


data.groupby(level=0).max()  #aggregating by index: Type1


# In[831]:


data1.groupby(level=1).mean()


# In[832]:


data1.groupby(level=1).agg({'Attack': lambda x: sum(x) / len(x)})


# removing the index

# In[833]:


data1.reset_index(inplace=True)


# In[834]:


data1.head()


# #### Merging and joins

# we can use merge for joining multiple tables

# In[835]:


np.random.choice(['Ash', 'Brock', 'TeamKat'], 5)


# In[836]:


pokemon_owner = pd.DataFrame({
    '#':
    data['#'],
    'Owner':
    np.random.choice(['Ash', 'Brock', 'TeamKat'], data.shape[0])
})


# In[837]:


pokemon_owner.head()


# In[838]:


pd.merge(data, pokemon_owner, how='left', on='#').head()


# #### Applying function

# In[839]:


data.Attack.apply(lambda x: 2 * x).head()


# #### Dropping columns

# In[840]:


data.drop(['Name'], axis=1)


# #### Renaming columns

# In[841]:


data.rename(columns={'Name': 'Pokemon_name', 'Type 2': 'Type2'}).head()


# making columns names lower case

# In[842]:


data.rename(columns=str.lower).head()


# Converting to numpy

# In[843]:


data.values


# #### Exporting pandas

# In[844]:


# data.to_csv('../output/xyz.csv', index=False)


# For big dataframes best way to backup is feather format as reading from it is a lot faster

# In[845]:


# data.to_feather('../outxyz')


# ### Detailed discussion of aggregation

# * Splitting the data into groups based on some criteria
# * Applying a function to each group independently
# * Combining the results into a data structure
# 

# In[846]:


data = pd.read_csv('../input/pokemon.csv')


# In[847]:


data.head()


# grouping the data

# In[848]:


group = data.groupby(['Type 1', 'Type 2'])


# In[849]:


group


# In[850]:


group.last()


# In[851]:


data.head()


# In[852]:


data['Type 2'].groupby(data['Type 1']).count()


# In[853]:


group = data.groupby(['Type 1', 'Type 2'])


# getting data for a specific group

# In[854]:


group.get_group(('Water', 'Dark'))


# We can use `aggregate()` to perform aggregation on groups

# In[855]:


group.aggregate(np.sum)


# You can apply `reset_index()` to bring index column into df

# In[856]:


group.aggregate(np.sum).reset_index()


# multiple aggregation

# In[857]:


group.aggregate([np.sum, np.mean]).reset_index()


# `size()` can be called on group to get the count in each group

# In[858]:


group.size()


# Group wise descriptive stats

# In[859]:


group.describe()


# Next up will be tutorial on numpy, regex and basics of python

# In[ ]:




