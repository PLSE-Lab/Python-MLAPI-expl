#!/usr/bin/env python
# coding: utf-8

# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# ---
# 

# # Creating Reading and Writing

# ## Relevant Resources
# * **[Creating, Reading and Writing Reference](https://www.kaggle.com/residentmario/creating-reading-and-writing-reference)** - Tutorial 
# * [General Pandas Cheat Sheet](https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf)
# * [IO tools](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html) - Pandas Documentation
# * [This Notebook on Kaggle](https://www.kaggle.com/mahendrabishnoi2/04-pandas)

# ## Creating Data

# In[ ]:


# import pandas
import pandas as pd

# creating a DataFrame
pd.DataFrame({'Yes': [50, 31], 'No': [101, 2]})


# In[ ]:


# another example of creating a dataframe
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland']})


# We are using the `pd.DataFrame` constructor to generate these `DataFrame` objects. The syntax for declaring a new one is a dictionary whose keys are the column names (Bob and Sue in this example), and whose values are a list of entries. This is the standard way of constructing a new `DataFrame`.
# 
# This way of creating `DataFrame` uses ascending values from 0 as index. If we want to set index ourselves we can use following way.

# In[ ]:


pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
              index = ['Product A', 'Product B'])


# A `Series` is a sequence of data values. If a `DataFrame` is a table, a `Series` is a list. And in fact you can create one with nothing more than a list:

# In[ ]:


# creating a pandas series
pd.Series([1, 2, 3, 4, 5])


# In[ ]:


# we can think of a Series as a column of a DataFrame.
# we can assign index values to Series in same way as pandas DataFrame
pd.Series([10, 20, 30], index=['2015 sales', '2016 sales', '2017 sales'], name='Product A')


# `Series` and the `DataFrame` are intimately related. It's helpful to think of a `DataFrame` as actually being just a bunch of `Series` "glue together".

# ## Reading common file formats
# 
# - CSV

# In[ ]:


import os
os.listdir("../input/188-million-us-wildfires")


# In[ ]:


# reading a csv file and storing it in a variable
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")


# In[ ]:


# we can use the 'shape' attribute to check size of dataset
wine_reviews.shape


# In[ ]:


# To show first five rows of data, use 'head()' method
wine_reviews.head()


# We can notice that this data has built in index, so we can use that while reading dataset from csv file. Here's how to do that:

# In[ ]:


wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()


# - SQL Files
# 
# SQL databases are where most of the data on the web ultimately gets stored.
# 
# Connecting to a SQL database requires a connector, something that will handle siphoning data from the database.
# 
# `pandas` won't do this for you automatically because there are many, many different types of SQL databases out there, each with its own connector. So for a SQLite database, we would need to first do the following (using the  `sqlite3` library that comes with Python):

# In[ ]:


import sqlite3
conn = sqlite3.connect("../input/188-million-us-wildfires/FPA_FOD_20170508.sqlite")


# The other thing you need to do is write a SQL statement. Internally, SQL databases all operate very differently. Externally, however, they all provide the same API, the "Structured Query Language" (or...SQL...for short).
# 
# For the purposes of analysis however we can usually just think of a SQL database as a set of tables with names, and SQL as a minor inconvenience in getting that data out of said tables.
# 
# To get the data out of `SQLite` and into `pandas`:

# In[ ]:


fires = pd.read_sql_query("SELECT * FROM fires", conn)


# In[ ]:


fires.head()


# ## Writing common file formats
# 
# - To CSV

# In[ ]:


wine_reviews.head().to_csv("wine_reviews.csv")


# - To SQL database
# 
# To output to a SQL database, supply the name of the table in the database we want to throw the data into, and a connector:

# In[ ]:


conn = sqlite3.connect("fires.sqlite")
fires.head(10).to_sql("fires", conn)


# Exercises of this tutorial are solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-creating-reading-and-writing)

#  

# # Indexing, Selecting & Assigning
# 
# ## Relevant Resources
# * **[Quickstart to indexing and selecting data](https://www.kaggle.com/residentmario/indexing-and-selecting-data/)** 
# * [Indexing and Selecting Data](https://pandas.pydata.org/pandas-docs/stable/indexing.html) section of pandas documentation
# * [Tutorial Link](https://www.kaggle.com/residentmario/indexing-selecting-assigning)

# In[ ]:


import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)


# ## Naive Accessors

# In[ ]:


reviews


# In[ ]:


# access 'country' property (or column) of 'reviews' 
reviews.country


# In[ ]:


# Another way to do above operation
# when a column name contains space, we have to use this method
reviews['country']


# In[ ]:


# To access first row of country column
reviews['country'][0]


# ## Index-based selection
# `pandas` has its own accessor operators, `loc` and `iloc`.
# 
# `pandas` indexing works in one of two paradigms. The first is index-based selection: selecting data based on its numerical position in the data. `iloc` follows this paradigm.
# 
# Both `loc` and `iloc` are row-first, column-second. This is the opposite of what we do in native Python, which is column-first, row-second.

# In[ ]:


# returns first row
reviews.iloc[0]


# In[ ]:


# returns first column (country) (all rows due to ':')
reviews.iloc[:, 0]


# In[ ]:


# retruns first 3 rows of first column
reviews.iloc[:3, 0]


# In[ ]:


# we can pass a list of indices of rows/columns to select
reviews.iloc[[0, 1, 2, 3], 0]


# In[ ]:


# We can also pass negative numbers as we do in Python
reviews.iloc[-5:]


# ## Label-based selection
# The second paradigm for attribute selection is the one followed by the `loc` operator: label-based selection. In this paradigm it's the data index value, not its position, which matters.

# In[ ]:


# To select first entry in country column
reviews.loc[0, 'country']


# In[ ]:


# select columns by name using 'loc'
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]


# **Note: `iloc` uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. So 0:10 will select entries 0,...,9. `loc`, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10.**

# ## Manipulating the index

# In[ ]:


# 'set_index' to the 'title' field
reviews.set_index('title')


# ## Conditional Selection
# 
# To find out better than average wines produced in Italy

# In[ ]:


# 1. Find out whether wine is produced in Italy
reviews.country == 'Italy'


# In[ ]:


# 2. Now select all wines produced in Italy
reviews.loc[reviews.country == 'Italy'] #reviews[reviews.country == 'Italy']


# In[ ]:


# Add one more condition for points to find better than average wines produced in Italy
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]  # use | for 'OR' condition


# `pandas` comes with a few pre-built conditional selectors, two of which we will highlight here. The first is `isin`. `isin` is lets you select data whose value "is in" a list of values. For example, here's how we can use it to select wines only from Italy or France:

# In[ ]:


reviews.loc[reviews.country.isin(['Italy', 'France'])]


# The second is `isnull` (and its companion `notnull`). These methods let you highlight values which are or are not empty (`NaN`). For example, to filter out wines lacking a price tag in the dataset, here's what we would do:

# In[ ]:


reviews.loc[reviews.price.notnull()]


# ## Assigning Data

# In[ ]:


reviews['critic'] = 'everyone'
reviews.critic


# In[ ]:


# using iterable for assigning
reviews['index_backwards'] = range(len(reviews), 0, -1)
reviews['index_backwards']


# Exercises of this tutorial are solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-indexing-selecting-assigning)

#  

# # Summary Functions and Maps
# 
# ## Relevant Resources
# - [Essential Basic Functionality](https://pandas.pydata.org/pandas-docs/stable/basics.html) - From Pandas Docs
# - [Tutorial](https://www.kaggle.com/residentmario/summary-functions-and-maps)

# In[ ]:


import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()


# ## Summary Functions
# - `describe`

# In[ ]:


reviews.describe()
# this method generates stats for numerical data only


# In[ ]:


reviews.taster_name.describe()          
# when 'describe' method is applied to string data


# - `mean`

# In[ ]:


# Find out a particular statistic of a DataFrame or Series
# For eg. find the average of points/rating given to wines
reviews.points.mean()


# - `unique` - To see a list of unique items in Series or DataFrame

# In[ ]:


reviews.taster_name.unique()


# - `value_counts()` - To see a list of all unique values and their count

# In[ ]:


reviews.taster_name.value_counts()


# ## Maps
# A "map" is a term, borrowed from mathematics, for a function that takes one set of values and "maps" them to another set of values. In data science we often have a need for creating new representations from existing data, or for transforming data from the format it is in now to the format that we want it to be in later.
# 
# There are two mapping method that you will use often. [`Series.map`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html) is the first, and slightly simpler one. For example, suppose that we wanted to remean the scores the wines recieved to 0. We can do this as follows:

# In[ ]:


review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)


# [`DataFrame.apply`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html) is the equivalent method if we want to transform a whole DataFrame by calling a custom method on each row.

# In[ ]:


def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')


# If we had called `reviews.apply` with `axis='index'`, then instead of passing a function to transform each row, we would need to give a function to transform each column.
# 
# Note that `Series.map` and `DataFrame.apply` return new, transformed Series and DataFrames, respectively. They don't modify the original data they're called on. If we look at the first row of `reviews`, we can see that it still has its original `points` value.

# In[ ]:


reviews.head(1)


# In[ ]:


# Another way (also faster one) to remean points
review_points_mean = reviews.points.mean()
reviews.points - review_points_mean


# Here there are lot of values on LHS of `-` and single value on RHS. `pandas` automatically substract single value from every value on LHS. **Broadcasting**.
# 
# It works equally well if we have equal number of values on both sides. Also for data types other than `int` and `float`. 

# In[ ]:


# Combining data from two string columns. concatenation
reviews.country + ' - ' + reviews.region_1


# These operators are faster than the `map` or `apply` because they uses speed ups built into `pandas`. All of the standard Python operators (`>`, `<`, `==`, and so on) work in this manner but they are not as flexible as `map` and `apply`. We can apply conditional logic etc. using `map` and `apply`.

# Exercises of this tutorial are solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-summary-functions-and-maps)

#  

# # Grouping and Sorting
# Maps allow us to transform data in a `DataFrame` or `Series` one value at a time for an entire column. However, often we want to group our data, and then do something specific to the group the data is in. We do this with the `groupby` operation.
# 
# ## Relevant Resources
# - [**Grouping Reference and Examples**](https://www.kaggle.com/residentmario/grouping-and-sorting-reference) - Tutorial 
# - [Groupby: split-apply-combine](https://pandas.pydata.org/pandas-docs/stable/groupby.html) - Grouping
# - [Advanced Indexing](https://pandas.pydata.org/pandas-docs/stable/advanced.html) - Multi Indexing
# - [Adavanced basic functionality](https://pandas.pydata.org/pandas-docs/stable/basics.html#sorting) - Sorting

# In[ ]:


import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)


# Maps allow us to transform data in a `DataFrame` or `Series` one value at a time for an entire column. However, often we want to group our data, and then do something specific to the group the data is in. To do this, we can use the `groupby` operation.
# 
# For example, one function we've been using heavily thus far is the `value_counts` function. We can replicate what `value_counts` does using `groupby` by doing the following:

# In[ ]:


reviews.groupby('points').points.count()


# `groupby` in pandas works in a similar way as `GROUP BY` in sql. Example:

# In[ ]:


reviews.groupby('points').count()


# `value_counts` is just a shortcut to this `groupby` operation. We can use any of the summary functions we've used before with this data. For example, to get the cheapest wine in each point value category, we can do the following:

# In[ ]:


reviews.groupby('points').price.min()


# We can think of each group we generate as being a slice of our `DataFrame` containing only data with values that match. This `DataFrame` is accessible to us directly using the `apply` method, and we can then manipulate the data in any way we see fit. For example, here's one way of selecting the name of the first wine reviewed from each winery in the dataset:

# In[ ]:


reviews.groupby('winery').apply(lambda df: df.title.iloc[0])


# For even more fine-grained control, we can also group by more than one column. For an example, here's how we would pick out the best wine by country and province:

# In[ ]:


# reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.argmax()])
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])


# Another groupby method worth mentioning is `agg`, which lets us run a bunch of different functions on our `DataFrame` simultaneously. For example, we can generate a simple statistical summary of the dataset as follows:

# In[ ]:


reviews.groupby('country').price.agg([len, min, max])


# ## Multi-indexes
# A multi-index differs from a regular index in that it has multiple levels. For example:

# In[ ]:


countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed


# In[ ]:


mi = _.index
type(mi)


# Multi-indices have several methods for dealing with their tiered structure which are absent for single-level indices. They also require two levels of labels to retrieve a value, an operation that looks something like this. Dealing with multi-index output is a common "gotcha" for users new to `pandas`.
# 
# However, in general the `MultiIndex` method you will use most often is the one for converting back to a regular index, the reset_index method:

# In[ ]:


countries_reviewed.reset_index()


# ## Sorting
# when outputting the result of a `groupby`, the order of the rows is dependent on the values in the index, not in the data.
# 
# To get data in the order want it in we can sort it ourselves. The `sort_values` method is handy for this.

# In[ ]:


countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')


# In[ ]:


# Descending sort
countries_reviewed.sort_values(by='len', ascending=False)


# In[ ]:


# sort by index
countries_reviewed.sort_index()


# In[ ]:


# sort by more than one column at a time
countries_reviewed.sort_values(by=['country', 'len'])


# Exercises of this tutorial are solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-grouping-and-sorting)

#  

# # Data Types and Missing Data
# 
# ## Relevant Resources
# - [Data Types and Missing Data Reference](https://www.kaggle.com/residentmario/data-types-and-missing-data-reference) - Tutorial
# - [Intro to data structures](https://pandas.pydata.org/pandas-docs/stable/dsintro.html)
# - [Working with missing data](https://pandas.pydata.org/pandas-docs/stable/missing_data.html)

# In[ ]:


import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)


# ## Data Types
# 
# The data type for a column in a `DataFrame` or a `Series` is known as the `dtype`.
# 
# We can use the `dtype` property to grab the type of a specific column:

# In[ ]:


reviews.price.dtype


# In[ ]:


# data types of all columns in a DataFrame
reviews.dtypes


# One peculiarity to keep in mind (and on display very clearly here) is that columns consisting entirely of strings do not get their own type; they are instead given the `object` type.
# 
# It's possible to convert a column of one type into another wherever such a conversion makes sense by using the `astype` function. For example, we may transform the `points` column from its existing `int64` data type into a `float64` data type:

# In[ ]:


reviews.points.astype('float64')


# In[ ]:


# data type of index of "Series" or "DataFrame"
reviews.index.dtype


# ## Missing Data
# Entries missing values are given the value `NaN`, short for "Not a Number". For technical reasons these `NaN` values are always of the `float64` dtype.
# 
# `pandas` provides some methods specific to missing data. To select `NaN` entreis we can use `pd.isnull` (or its companion `pd.notnull`).

# In[ ]:


reviews[reviews.country.isnull()]


# Replacing missing values is a common operation.  `pandas` provides a really handy method for this problem: `fillna`. `fillna` provides a few different strategies for mitigating such data. For example, we can simply replace each `NaN` with an `"Unknown"`:

# In[ ]:


reviews.region_2.fillna("Unknown")


# `fillna` supports a few strategies for imputing missing values explained in [the official function documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html).

# Alternatively, we may have a non-null value that we would like to replace. For example, suppose that since this dataset was published, reviewer Kerin O'Keefe has changed her Twitter handle from `@kerinokeefe` to `@kerino`. One way to reflect this in the dataset is using the `replace` method:

# In[ ]:


reviews.taster_twitter_handle.replace('@kerinokeefe', '@kerino')


# Exercises of this tutorial are solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-data-types-and-missing-values)

#  

# # Renaming and Combining Values
# 
# ## Relevant Resources
# - [Renaming and Combining Values Reference](https://www.kaggle.com/residentmario/renaming-and-combining-reference) - Tutorial
# - [Essential Basic Functionality](https://pandas.pydata.org/pandas-docs/stable/basics.html#renaming-mapping-labels) - Renaming
# - [Merge, join, concatenate](https://pandas.pydata.org/pandas-docs/stable/merging.html) - Combining

# In[ ]:


import pandas as pd
pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews


# ## Renaming
# - `rename` - lets us rename index names and/or column names. 
# 
# For example, to change the points column in our dataset to score, we would do:

# In[ ]:


reviews.rename(columns={'points': 'score'})


# `rename` lets us rename index or column values by specifying a index or column keyword parameter.

# In[ ]:


reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})


# Both the row index and the column index can have their own name attribute. The complimentary `rename_axis` method may be used to change these names. For example:

# In[ ]:


reviews.rename_axis('fields', axis='columns').rename_axis('wines', axis='rows')

#reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')    # Also correct


# ## Combining
# 
# When performing operations on a dataset we will sometimes need to combine different `DataFrame` and/or `Series` in non-trivial ways. `pandas` has three core methods for doing this. In order of increasing complexity, these are `concat`, `join`, and `merge`. Most of what `merge` can do can also be done more simply with `join`.
# 
# The simplest combining method is `concat`. This function works just like the `list.concat` method in core Python: given a list of elements, it will smush those elements together along an axis.
# 
# This is useful when we have data in different `DataFrame` or `Series` objects but having the same fields (columns). One example: the [YouTube Videos](https://www.kaggle.com/datasnaek/youtube-new) dataset, which splits the data up based on country of origin (e.g. Canada and the UK, in this example). If we want to study multiple countries simultaneously, we can use `concat` to smush them together:

# In[ ]:


canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])


# `pd.DataFrame.join` lets us combine different DataFrame objects which have an index in common. For example, to pull down videos that happened to be trending on the same day in both Canada and the UK, we could do the following:

# In[ ]:


left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')


# Exercises of this tutorial solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-renaming-and-combining)
