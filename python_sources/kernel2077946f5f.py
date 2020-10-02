#!/usr/bin/env python
# coding: utf-8

# This notebook will help you with some of the basics of working with Pandas DataFrames. The first cell just imports a Kaggle dataset and trims it down. We only need a few columns for this demonstration.

# In[ ]:


import pandas as pd
df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
columns = ['Neighborhood', 'HouseStyle', 'OverallQual', 'OverallCond', 'SalePrice']
df = df[columns]
print(df.shape)
print(df.head())


# You can also use conditional arrays or lists to subset the rows of a DataFrame. It is worth noting that this always returns a DataFrame or Series even if there is only one True conditional.

# In[ ]:


df_expensive = df[df.SalePrice >= 150000]
print(df_expensive.shape)
print(df_expensive.head())


# Yet another way to index DataFrames is with the functions loc and iloc. Misunderstanding how these work will lead to confusion down the road. Sorting without reseting the index or renaming the index will cause loc to behave unexpectedly or throw a key error. 

# In[ ]:


# loc uses NAMES to index the DataFrame, iloc uses the integer location regardless of the name
# if the names are numeric, numeric types can be used. Conditional arrays are also valid.
# syntax is df.loc[row_names, col_names] and df.iloc[row_nums, col_nums]
print(df_expensive.loc[1,'Neighborhood'])
df_expensive.rename(index={1:'one'}, inplace=True)
try:
    df_expensive.loc[1,'Neighborhood']
except KeyError:
    print('Error was thrown, numbers in loc don\'t always behave!')
    print('Using iloc')
    print(df_expensive.iloc[1,0])
df_expensive.reset_index(drop=True, inplace=True)


# That covers the basics of slicing DataFrames. Personally I avoid confusion by using df[column_names].iloc[row_nums] or df[conditional][column_names].iloc[row_nums] when I want to combine column names with integer row indices. Next is renaming columns. Renaming is done with the rename method (as seen above). To rename columns use df.rename(columns={old_key:new_key}). 

# In[ ]:


df_expensive.rename(columns={'Neighborhood':'Hood'}, inplace=True)
print(df_expensive.head())


# So far we have created df_expensive by subsetting df, reset it's indices, and renamed the 'Neighborhood' column to 'Hood'. One neat feature of Pandas is that we can chain these operations together, making it more clear exactly what manipulations happened to df.

# In[ ]:


# parenthesis allow the carriage returns
df_expensive_ = (
    df[df.SalePrice >= 150000]
    .reset_index(drop=True)
    .rename(columns={'Neighborhood':'Hood'})
)
print(df_expensive_.head())


# How can we find the average home price in a neighborhood? Groupby will separate the data into groups and aggragate functions can be performed on each group individually. 

# In[ ]:


avg_price = (
    df[['Neighborhood', 'SalePrice']]
    .groupby(['Neighborhood'], as_index=False).mean()
)
print(avg_price.head())


# Now suppose we want to know how many of each style house were sold in each neighborhood... groupby can help here too

# In[ ]:


hood_count = (
    df[['Neighborhood', 'HouseStyle', 'SalePrice']]
    .groupby(['Neighborhood', 'HouseStyle'])
    .count()
)
print(hood_count.head(10))


# Now we're interested in finding the least expensive houses. sort_values can help.

# In[ ]:


df_sorted = df.sort_values(by=['SalePrice'])
print(df_sorted.head())
# sort descending
df_sorted = df.sort_values(by=['SalePrice'], ascending=False)
print(df_sorted.head())


# Now let's say that we need to put two new columns on df, NumHood = # of houses in the neighborhood and NumStyle = # of houses with that style. We can do that with merge. The merge method works like a join does with SQL. The basic strategy is to use groupby to make dataframes with the key-value pairs we want and then to merge them to df.

# In[ ]:


num_hood = df[['Neighborhood', 'HouseStyle']].groupby(['Neighborhood'], as_index=False).count().rename(columns={'HouseStyle':'NumHood'})
num_style = df[['Neighborhood', 'HouseStyle']].groupby(['HouseStyle'], as_index=False).count().rename(columns={'Neighborhood':'NumStyle'})
df_merged = df.merge(num_hood, on=['Neighborhood'], how='left').merge(num_style, on=['HouseStyle'], how='left')
print(df_merged.head())

