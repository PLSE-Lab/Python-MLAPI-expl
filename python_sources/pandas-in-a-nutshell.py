#!/usr/bin/env python
# coding: utf-8

# # Acknowledgements
# > Most information in this notebook are based on Kaggle micro-courses available in [Kaggle Learn](https://www.kaggle.com/learn/overview).

# In[ ]:


import pandas as pd


# # Data Creation
# There are two core objects in pandas: the **DataFrame** and the **Series**.

# ### DataFrame
# A DataFrame is a table. It contains an array of individual entries, each of which has a certain value. Each **entry** corresponds to a **row** (or record) and a **column** (or field).

# - **pd.DataFrame():** is used to generate DataFrame objects. The syntax for declaring a new one is a dictionary whose keys are the **column labels**, and whose values are a **list of entries**. The dictionary-list constructor assigns an ascending count from 0 for the **row labels**.

# In[ ]:


pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})


# The list of row labels used in a DataFrame is known as an **Index**. We can assign values to it by using an `index` parameter.

# In[ ]:


reviews = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']}, index=['Product A', 'Product B'])
reviews


# - **rename_axis():** set the name of the axis for the index (row labels) or columns.

# In[ ]:


reviews.rename_axis("Products", axis='rows').rename_axis("Reviewers", axis='columns')


# ### Series
# A Series is a sequence of data values. If a DataFrame is a table, a Series is a list.
# 
# - A Series is, in essence, a single column of a DataFrame. So you can assign row labels to the Series using an `index` parameter. 
# - A Series does not have a column name, it only has one overall `name`.

# In[ ]:


pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')


# - **to_csv():** write object to a Comma-Separated Values (CSV) file.

# In[ ]:


reviews.to_csv("reviews.csv")


# # Data Reading
# - **pd.read_csv():** load a CSV file, a data, into a DataFrame.

# In[ ]:


wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
wine_reviews.head()


# To make pandas use a column for the index (instead of creating a new one from scratch), we can specify an index_col ``index_col``.

# In[ ]:


wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()


# - **rename():** *DataFrame* -> alter axes labels. *Series* -> alter Series index labels or name.

# In[ ]:


wine_reviews = wine_reviews.rename(columns={'region_1': 'region', 'region_2': 'locale'})
wine_reviews


# # Indexing
# These are the two ways of selecting a specific Series out of a DataFrame in native Python.

# ### Attribute selection
# We can access the property of an object by accessing it as an **attribute**. Columns in a pandas DataFrame work in much the same way.

# In[ ]:


print(wine_reviews.country)
wine_reviews.country[0]


# ### Indexing operator []
# If we have a Python dictionary, we can access its values using the indexing operator **[]**. We can do the same with columns in a DataFrame.

# In[ ]:


print(wine_reviews['country'])
wine_reviews['country'][:5]


# However, pandas has its own accessor operators, loc and iloc. Both of them are row-first, column-second. 

# ### Index-based selection
# Selecting data based on its numerical position in the data. **iloc** follows this paradigm.

# In[ ]:


wine_reviews.head()


# In[ ]:


wine_reviews.iloc[0]


# In[ ]:


wine_reviews.iloc[:3,0]


# In[ ]:


wine_reviews.iloc[[0, 1, 2], [0,1]]


# In[ ]:


wine_reviews.country.iloc[1]


# ### Label-based selection
# The second paradigm for attribute selection is the one followed by the **loc** operator: label-based selection. In this paradigm, it's the data index value, not its position, which matters.

# In[ ]:


wine_reviews.head()


# In[ ]:


wine_reviews.loc[0, 'country']


# In[ ]:


wine_reviews.loc[[0,1,10,100],['country', 'province', 'region', 'locale']]


# In[ ]:


wine_reviews.loc[[1, 2, 3, 5, 8]]


# **Both loc and iloc are row-first, column-second. This is the opposite of what we do in native Python, which is column-first, row-second.**

# ### Conditional selection
# Conditional selection produce a Series of True/False booleans. This result can then be used inside of loc to select the relevant data.

# In[ ]:


wine_reviews[(wine_reviews.country == 'Brazil')]


# - **isin():** built-in conditional selector lets you select data whose value "is in" a list of values.

# In[ ]:


wine_reviews.loc[(wine_reviews.country.isin(['Australia', 'New Zealand'])) & (wine_reviews.points >= 95)]


# - **isnull(), notnull():** built-in conditional selectors let you highlight missing values which are NA/ non-NA. NA: None, NaN, NaT.

# In[ ]:


wine_reviews.loc[wine_reviews.price.isnull()]


# # Index Manipulation
# - **set_index():** *DataFrame* -> set the DataFrame index using existing columns.

# In[ ]:


wine_reviews.set_index("title")


# - **reset_index():** *DataFrame* -> reset the index, or a level of it. Reset the index of the DataFrame, and use the default one instead. If the DataFrame has a MultiIndex, this method can remove one or more levels. We can use the `drop` parameter to avoid the old index being added as a column.

# In[ ]:


wine_reviews.reset_index(drop=True)


# # Data Assignment 
# You can assign with a constant value:

# In[ ]:


wine_reviews['critic'] = 'everyone'
wine_reviews['critic']


# Or with an iterable of values:

# In[ ]:


wine_reviews['index_backwards'] = range(len(wine_reviews), 0, -1)
wine_reviews


# # Data Exploration 
# ### Summary functions
# - **describe():** generate a high-level summary of the data input. It is type-aware, meaning that its output changes based on the data type of the input.

# In[ ]:


wine_reviews.points.describe()


# In[ ]:


wine_reviews.taster_name.describe()


# - **head(), tail():** return the first/ last `n` rows. `n` is set to 5 by default.

# In[ ]:


wine_reviews.head()


# - **columns:** *DataFrame* -> return DataFrame column labels.

# In[ ]:


wine_reviews.columns


# - **mean():** return the mean of the values for the requested axis. 
# 

# In[ ]:


wine_reviews.price.mean()


# - **median():** return the median of the values for the requested axis.

# In[ ]:


wine_reviews.price.median()


# - **unique():** *Series* -> return unique values of Series object.

# In[ ]:


wine_reviews.taster_name.unique()


# - **nunique():** *DataFrame* -> Return Series with number of distinct observations. Can ignore NaN values. *Series* -> return number of unique elements in the object.

# In[ ]:


wine_reviews.nunique()


# In[ ]:


wine_reviews.taster_name.nunique()


# - **value_conuts():** *Series* -> return a Series containing counts of unique values.

# In[ ]:


wine_reviews.country.value_counts()


# - **count():** *DataFrame* -> count non-NA (`None`, `NaN`, `NaT`) cells for each column or row. *Series* -> return number of non-NA in the Series.

# In[ ]:


wine_reviews.count()


# - **min()/max():** return the minimum/ minimum of the values for the requested axis.

# In[ ]:


wine_reviews.price.min()


# - **idxmax():** *DataFrame* -> return index of first occurrence of maximum over requested axis. *Series* -> return the row label of the maximum value.

# In[ ]:


bargain_idx = (wine_reviews.points / wine_reviews.price).idxmax()
wine_reviews.loc[bargain_idx, 'title']


# ### Dtypes
# - **dtype:** *Series* -> return the dtype object of the underlying data, it could be used to get the type of a specific column. 

# In[ ]:


wine_reviews.points.dtype


# In[ ]:


wine_reviews.price.dtype


# Columns consisting entirely of strings or timestamps do not get their own type; they are instead given the **object** type.

# In[ ]:


wine_reviews.country.dtype


# - **dtypes:** DataFrame -> return the dtypes in the DataFrame.

# In[ ]:


wine_reviews.dtypes


# - **select_dtypes():** *DataFrame* -> return a subset of the DataFrame's columns based on the column dtypes.

# In[ ]:


wine_reviews.select_dtypes(include='object')


# In[ ]:


wine_reviews.select_dtypes(exclude='object')


# - **astype():** cast a pandas object to a specified dtype `dtype`. 

# In[ ]:


wine_reviews.points.astype('float64')


# ### Groupwise analysis
# - **groupby():** group DataFrame/ Series using a mapper or by a Series of columns.

# In[ ]:


wine_reviews.groupby('country').price.max()


# In[ ]:


wine_reviews.groupby('points').price.min()


# In[ ]:


wine_reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])


# - **agg():** aggregate using one or more operations over the specified axis.

# In[ ]:


wine_reviews.groupby(['country']).points.agg([len, min, max])


# ### Sorting
# - **sort_values()**: *DataFrame* -> sort by the values along either axis. *Series* -> sort by the values. Defaults to an ascending sort.

# In[ ]:


countries_reviewed = wine_reviews.groupby(['country', 'province']).points.agg([max])
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='max', ascending=False)


# It's possible to sort by more than one column at a time.

# In[ ]:


countries_reviewed.sort_values(by=['country', 'max'])


# - **sort_index():** *Dataframe* -> sort object by labels (along an axis). *Series* -> sort by the values.

# In[ ]:


countries_reviewed.sort_index()


# # Data Manipulation 
# ### Mapping
# **Map** is a term, borrowed from mathematics, for a function that takes one set of values and "maps" them to another set of values. In data science we often have a need for creating new representations from existing data, or for transforming data from the format it is in now to the format that we want it to be in later.
# 
# - **map():** Series -> a mapping method that map values of Series according to input correspondence. The function passed to map() should expect a single value from the Series and return a transformed version of that value. map() returns a **new** Series where all the values have been transformed by the function.

# In[ ]:


mean = wine_reviews.price.mean()
centered_price = wine_reviews.price.map(lambda p: p - mean)
centered_price


# In[ ]:


n_trop = wine_reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = wine_reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])
descriptor_counts


# - **apply():** *DataFrame* -> a mapping method that apply a function along an axis of the DataFrame by calling this custom function on each row or column. *Series* -> invoke function on values of Series. apply() returns a **new** DataFrame.

# In[ ]:


def reduced_price(row):
    row.price = row.price - 1
    return row

new_price = wine_reviews.apply(reduced_price, axis='columns')
new_price


# In[ ]:


def starring(row):
    if row.country == 'Canada' or row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1
star_ratings = wine_reviews.apply(starring, axis='columns')
wine_reviews['rating'] = star_ratings
wine_reviews


# Pandas provides many common mapping operations as built-ins, all of the standard Python operators (>, <, ==, +, -).

# In[ ]:


wine_reviews['country - province'] = wine_reviews.country + " - " + wine_reviews.province
wine_reviews


# In[ ]:


wine_reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])


# ### Missing Data
# - **isnull(), notnull():** detect NA or non-NA values. NA: None, NaN, NaT.

# In[ ]:


wine_reviews[wine_reviews.country.isnull()]


# - **fillna():** fill NA/NaN values using the specified method.

# In[ ]:


wine_reviews.locale.fillna("Unknown")


# - **Backfill strategy:** a strategy in which each missing value is filled with the first non-null value that appears sometime after the given record in the database.

# - **replace():** is handy for replacing missing data which is given some kind of sentinel value in the dataset: things like "Unknown", "Undisclosed", "Invalid", and so on.

# In[ ]:


wine_reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")


# - **drop():** *DataFrame* -> drop specified labels from rows or columns. *Series* -> return Series with specified index labels removed.

# In[ ]:


wine_reviews.drop(["index_backwards"], axis=1, inplace=True)
wine_reviews


# - **dropna():** *DataFrame* -> Remove missing values. *Series* -> return a new Series with missing values removed.

# In[ ]:


wine_reviews.dropna(axis=0, subset=['price'], inplace=True)
wine_reviews


# ### Combining
# To combine different DataFrames and/or Series in non-trivial ways:

# In[ ]:


canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")


# - **concat():** concatenate pandas objects along a particular axis with optional set logic
#     along the other axes. For instance, this is useful when we have data in different DataFrame or Series objects but having the **same fields** (columns).

# In[ ]:


pd.concat([canadian_youtube, british_youtube])


# - **join():** *DataFrame* -> join columns of another DataFrame, either on **index** or on a **key column**. The `lsuffix` and `rsuffix` parameters are necessary in case the data has the same column labels in both datasets.

# In[ ]:


canadian_youtube.set_index(['title', 'trending_date']).join(british_youtube.set_index(['title', 'trending_date']), lsuffix='_CAN', rsuffix='_UK')

