#!/usr/bin/env python
# coding: utf-8

# ## Pandas Refresher

# #### IMQAV
# 
# * Ingest
#     
#     * Import large volume of data rapidly
#     * eg. Tools like Kafka 
# * Model
#     
#     * Data Storage Techniques
#     * Eg. Mysql, MongoDb
# * Query
#     
#     * Query and Manipulate Data
# * Analyze
# 
#     * All ML data science Stat techniques go under here
# * Visualize
# 
#   * Transform data in insightful format and generate reports
#   
#   
#   [Pandas Cheetsheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)
# 

# ## Creating structure and importing Data

# In[ ]:


import pandas as pd
import numpy as np
import os

os.listdir('../input')


# Pandas Data frames are basically tables, with following three components
# * Index
# * Columns
# * Data
# 
# This is a two dimensional Structure.
# 
# Each Dataframe column can also be represented as a pandas series.
# Pandas Series is a 1D structure which includes
# * Index
# * Data
# 

# In[ ]:


# List of Lists (Row Wise)
data = [[30, 21]]
fruits = pd.DataFrame(data, columns=['Apples', 'Bananas'])

# Make individual column wise lists, then zip and add
# apples = [30]
# bananas = [21]
# fruits = pd.DataFrame(list(zip(apples, bananas)), columns = ['Apples', 'Bananas'])

# From A Dictionary
# fruits = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
fruits


# In[ ]:


# My chosen approach (With index)

# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
index = ['2017 Sales', '2018 Sales']
columns = ['Apples', 'Bananas']
apples = [35, 41]
bananas = [21, 34]
fruit_sales = pd.DataFrame(list(zip(apples, bananas)), columns=columns, index=index)

fruit_sales


# In[ ]:


# Each column of a dataframe is basically a series.

applesSeries = pd.Series(fruit_sales['Apples'])
applesSeries


# In[ ]:


# Creating a Series

index = ['Flour', 'Milk', 'Eggs', 'Spam']
data = ['4 cups', '1 cup', '2 large', '1 can']
ingredients = pd.Series(data, index = index, name = 'Dinner')

ingredients


# ## Imorting Data in Pandas
# 
# [Pandas Read CSV Datacamp](https://www.datacamp.com/community/tutorials/pandas-read-csv)
# 
# 

# In[ ]:


housing = pd.read_csv('../input/california-housing-prices/housing.csv', index_col = None, skiprows = 0)
housing.head()


# ## Saving Data

# In[ ]:


animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals
animals.to_csv('cows_and_goats.csv')


# ## Indexing Selecting and Assigning Data
# 

# In[ ]:


wineReviews = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv', index_col = 0)
wineReviews.head()

# Selecting a column
desc = wineReviews['description']
desc
print(type(desc))


# In[ ]:


# Accessing a specific entry

# wineReviews.description.iloc[0]

# wineReviews['description'][0]

# My chosen way

wineReviews.description[0]


# iloc stands for index location. It can be directly used for returning a row from the dataframe. Consider that columns exist as a property of dataframe so we can access them like df.columnName but same doesn't hold true for index values.

# In[ ]:


# return first row of data

# iloc return data based on postions so it doesn't 
# take anything apart from integers as input. It can take ranges though.

# iloc basically returns both columns as well as rows as function of no. of
# positions.

wineReviews.iloc[:4, :4]

# loc on the other hands returns rows and columns based on their labels.
# please note that loc includes the label at right side of range. iloc doesnt.

wineReviews.loc[:4, :'points']



# In[ ]:


# Return first few rows

# wineReviews.description.head(10)

# iloc also takes in range of data
wineReviews.description.iloc[:10]

# interesting things can be done with loc to select specific rows/columns
# Note that loc can take a boolean mask unlike iloc.
wineReviews.loc[wineReviews['country']=='USA', 'country'] = 'US'

# loc can take list of values as well
sample_reviews = wineReviews.loc[[1,2,3,5,8]]
sample_reviews


# In[ ]:


# loc since it takes a boolean mask as input, can be used for multiple things

# eg. Select all the wines made in italy

wineReviews.loc[wineReviews.country == 'Italy']


# In[ ]:


top_oceania_wines = wineReviews.loc[(wineReviews.points >=95) & (wineReviews.country.isin(['Australia', 'New Zealand']))]
top_oceania_wines


# ## Summary Functions
# 
# [Aggregation and Grouping in Pandas](https://data36.com/pandas-tutorial-2-aggregation-and-grouping/)

# In[ ]:


# Finding Median

#Dataframe.Median takes (axis={0:index, 1: columns})
# if used with a series object, it returns a single scalar value

#median over index and columns

housing.median(axis = 0)


# In[ ]:


# median over columns
housing.median(axis=1).head()


# In[ ]:


# Finding for a particular column
housing['total_rooms'].median()


# In[ ]:


# Finding unique values
#df.unqiue returns a numpy array with unique values.

housing['housing_median_age'].unique()


# In[ ]:


# Value Counts method to see how many times each value appears

housing['housing_median_age'].value_counts().head()


# In[ ]:


# Center the data

(housing.total_bedrooms - housing.total_bedrooms.mean()).head()


# In[ ]:


# Find median income of the household with maximum total bedrooms
# use of idxmax

housing.loc[housing.total_bedrooms.idxmax(), 'median_income']


# ## Maps 
# 
#  Map function is used on a series to do something on each value and replace it with new value
# 
#  Series.map takes either a dict, series ( with index same as column of caller series), function
# Python lambda function is especially helpful here.
# 
# [Python lambda](https://www.w3schools.com/python/python_lambda.asp)
# 
# [Pandas Series Map](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html)
# 
# 

# In[ ]:


import math
housing.median_income.map(lambda x: math.floor(x)).head()

# apply works similarly, it does not take dictionary or series as an argument 
# and works for dataframe as well. 
# Apply can take more complex functions as input

# df.apply takes function and axis as arguments. axis = 0 is across rows ( in index direction)


# ## Grouping and Sorting
# 
# [Grouping and Sorting Reference](https://www.kaggle.com/residentmario/grouping-and-sorting-reference)
# 
# Groupby as I already know divides the data in groups. A point worth noting in pandas is that each of these groups is a dataframe and can be accessed/manipulated by using apply.
# 

# In[ ]:


# Just groupby returns a groupby object, whose columns can be accessed and any
# aggregation functions can be used
housing.groupby('housing_median_age')


# In[ ]:


housing.groupby('housing_median_age').households.sum().head()


# In[ ]:


# apply can be directly used on groupby object to manipulate individual dataframes

# Know first total_rooms value in each group

housing.groupby('housing_median_age').apply(lambda df: df.total_rooms.iloc[0]).head()

# Groupby can also be used on multiple columns. This results in creation of a 
# multi-index


# In[ ]:


# agg function lets you apply multiple aggregation functions at the same time
# along with the groupby

housing.groupby('housing_median_age').households.agg(['count','sum','min','max']).head()


# In[ ]:


# sort_values function let's us sort the values of dataframe. by default
# the values returned are sorted by index when you do in operation on pandas

housing.sort_values(by = 'housing_median_age').head()

# This defaults to ascending sort. In case we want descending, put 

housing.sort_values(by = 'housing_median_age', ascending = False).head()


# In[ ]:


# to sort by index, sort_index is used

housing.sort_index().head()

# We can also sort by multiple columns, it first sorts by first columns, 
# then sorts by second column when encountered same values for first column.

housing.sort_values(by=['housing_median_age', 'total_rooms']).head()


# In[ ]:


# To simple return a size of each group , we do not need

# housing.groupby('housing_median_age').some_column.count()

# we can just do

housing.groupby('housing_median_age').size().head()


# ## Data Types and Missing Data
# 
# [Kaggle Reference](https://www.kaggle.com/residentmario/data-types-and-missing-data-reference)

# In[ ]:


# Find dtype of every column

housing.dtypes

# Find for some column

housing.longitude.dtype

# Convert dtype

housing.longitude.astype('int32').head()


# In[ ]:


# Missing values in pandas are represented as NaNs, all their dtype is always 
# float64

# Find out mssing values

# housing.loc[housing.households.isnull()]

# fill Nan Values

housing.total_rooms.fillna(0).head()

# replace values ( useful for mssing values where value is not NaN)

housing.longitude.replace('unknown', 'someValue').head()


# ## Renaming and Combining Data
# 
# [Essential Basic Functionality](https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html)
# 
# [Merge Join and Concat](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html)
# 
# [Kaggle Reference](https://www.kaggle.com/residentmario/renaming-and-combining-reference)

# In[ ]:


# rename columns

housing.head()

housing.rename(columns = {'longitude': 'long', 'latitude': 'lat'}).head()

# we can similarly use dictionary to rename index as well

housing.rename(index = {0: 'first'}).head()


# In[ ]:


# both row index and column index have their names apparently
# these names can be renamed

housing.rename_axis('fields', axis = 'columns').rename_axis('rows', axis = 'rows').head()


# In[ ]:


# Combining the dataframes

# Concat method joins two dataframes having same columns. It basically adds the 
# rows of both the dataframes


print(len(housing))

print(len(pd.concat([housing, housing])))


# In[ ]:


# Join method joins two dataframes with same indices. In case their column 
# names are same, we need to add lsuffix and rsuffix

housing.join(housing, lsuffix = '_left', rsuffix = '_right').head()


# In[ ]:


# Merge is the third command which works exactly like sql join

# result = pd.merge(user_usage,
#                  user_device[['use_id', 'platform', 'device']],
#                  on='use_id', 
#                  how='left')

# Pandas default join is inner join

# Note that Join and merge both behave in similar way. Join can also take an 
# attribute 'on' and behave similarly. But without this attribute, join functions 
# joins columns based on index. This is not the case with merge.


# In[ ]:




