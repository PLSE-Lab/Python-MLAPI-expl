#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Dataset
# Load the cereal csv file into a DataFrame named `cereals`

# In[ ]:


cereals = pd.read_csv('/kaggle/input/80-cereals/cereal.csv')


# Using `head(n)`, we can display the first 10 rows to get a sense of what the dataset looks like.
# * `head()` would display 5 rows by default

# In[ ]:


cereals.head(10)


# ### **Further steps to clean data:** ###
# **1. Select necessary data for analysis**
# 
# As we need all cereals, but not all of the features, we can utilize multiple column selection by indexing with **a list** of column names:
# > `cereals[[column_name1,column_name2]]`
# 
# Note: to make sure the changes we make persist in the dataset, we need to save it into the `cereals` data frame

# In[ ]:


cereals = cereals[['name', 'mfr', 'calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'rating']]


# **2. Rename column names for readability purposes**
# 
# The `rename()` method can rename column names by passing in a dictionary with the `key:value` as `old_name:new_name`
# * To specify that we are renaming column names, we need to use `columns=` as the parameter

# In[ ]:


cereals = cereals.rename(columns={'mfr': 'manufacturer', 'protein': 'protein (g)', 'fat':'fat (g)', 
                          'sodium':'sodium (mg)', 'fiber': 'fiber (g)', 'carbo':'carb. (g)', 'sugars': 'sugar (g)'})


# Note: to check if the renaming of columns was successful, we can use the `columns` data attribute and the `tolist()` method for easier readability, instead of displaying individual cereal data which is not needed in this case.
# * `columns`: outputs the column names of the data frame
# * `tolist()`: converts an object into a list

# In[ ]:


cereals.columns.tolist()


# **3. Replace manufacturer abbreviations with full version for a more readable analysis**

# In[ ]:


# Call replace() only on the manufacturer column and save it back to manufacturer column for data persistence
cereals['manufacturer'] = cereals['manufacturer']                        .replace({'A':'American Home Food Products', 'G':'General Mills', 'K':'Kelloggs', 
                                 'N':'Nabisco', 'P':'Post', 'Q':'Quaker Oats', 'R':'Ralston Purina'})


# Note: we can use `\` to signify a line break so we can separate the code into multiple lines for easier readability

# **4. Set the 'name' column as the index for readability purposes**

# In[ ]:


cereals = cereals.set_index('name')


# Display the first 5 rows to see if the changes made were successful

# In[ ]:


cereals.head()


# # 1. Cereal Manufacturer Market Share #

# a. Display a pie plot of manufacturers to represent how "big" each are in terms of number of cereals produced and sold in a typical supermarket.

# In[ ]:


# 1. Select the 'manufacturer' column
# 2. Apply value_counts() to get the group size (number of cereals within each manufacturer group)
# 3. Plot the pie
cereals.manufacturer        .value_counts()        .plot.pie()


# * The pie plot shows what many may have assumed, which is that Kelloggs and General Mills are the top two manufacturers. Meanwhile, as American Home Food Products is a pretty unknown company and, thus understandably the smallest.
# 
# b. How many and what are the cereal(s) American Home Food Products manufacture?

# In[ ]:


# 1. Conditional selection: selects the row in the 'manufacturer' column that contains 'American Home Food Products'
# 2. shape[0]: Count the number of rows 
cereals[cereals.manufacturer == 'American Home Food Products']        .shape[0]


# Note: `shape` outputs both the number of rows and cells, `shape[1]` outputs the number of cells

# In[ ]:


# 1. Same conditional selection as above
# 2. Reset index so we can get the name by selecting the column (step 3b)
# 3. Since we know that there is only one row (cereal), to get the name of the cereal:
    # a. Select the first row using iloc[i], i = row index number
    # b. Select the name column using loc[column_name]
cereals[cereals.manufacturer == 'American Home Food Products']        .reset_index()        .iloc[0].loc['name']


# Note: without saving the data frame that resets the index, the `cereals` data frame is still indexed by name

# # 2. Ratings vs Nutrition #

# a. What are the top 10 ranked cereals in descending order?

# In[ ]:


# 1. Select the 'rating' column
# 2. Sort the values in descending order (default= ascending)
# 3. Display the top 10 rows using iloc[:] subslicing 
cereals.rating        .sort_values(ascending=False)        .iloc[:10]


# * It can be assumed that the ratings were based on how healthy the cereal. However, we don't know what are the factors that makes a cereal 'healthy'. 
# 
# b. Compare the nutritional statistics of the top 5 and bottom 5 cereals ranked in descending order using a horizontal bar plot

# In[ ]:


# Top 5 cereals (first would be highest ranked)
# 1. Sort the data in ascending order of ratings because the .barh() plots the rows in reverse order
# 2. Select the 'fiber', 'sugar', 'carb' columns
# 3. Select the top 5 with tail(n) (displays the bottom n rows) since the data is in ascending order
cereals.sort_values('rating')        [['fiber (g)', 'sugar (g)', 'carb. (g)']]        .tail(5)        .plot.barh()


# In[ ]:


# Bottom 10 cereals (last would be the lowest ranked)
cereals.sort_values('rating')[['fiber (g)', 'sugar (g)', 'carb. (g)']].head().plot.barh()


# * Given the two horizontal bar plots above, it may be assumed that sugar is the biggest deciding factor or at least correlated to how a cereal is rated. The more sugar the cereal contains, the lower the rating. 
# 
# c. Using a scatter plot, display and examine the relationship between sugar (x-axis) and rating (y-axis)
# > `scatter(x='column_name1', y='column_name2')`

# In[ ]:


cereals.plot.scatter(x='sugar (g)', y='rating')


# The plot shows that there is a cereal that has a sugar content below 0(g) but it is rated at around 50.
# 
# d. Display the cereal's other nutritional facts 

# In[ ]:


# Select the cereal (row) with the lowest sugar content
cereals[ cereals['sugar (g)'] == cereals['sugar (g)'].min() ]


# e. Compare to the highest rated cereal to find out why the cereal above is given a low rating despite low sugar content

# In[ ]:


# Select the cereal (row) with the highest rating
cereals[ cereals.rating == cereals.rating.max() ]


# * The only significant difference that could cause the drop in rating would be the lack of fiber in Quaker Oatmeal.
# * Given this observation, is fiber another factor that decides a cereal's rating?

# f. Using a scatter plot, display and examine the relationship between fiber (x-axis) and rating (y-axis)

# In[ ]:


cereals.plot.scatter(x='fiber (g)',y='rating')


# We can conclude from the two scatterplots above that more sugar and less fiber is correlated to a lower rating, while less sugar and more fiber is correlated to a higher rating.

# # 3. Manufacturer Rating

# a. What is the rank of manufacturers in terms of average cereal ratings, in descending order?

# In[ ]:


# 1. Split by manufacturer
# 2. Select the 'rating' column
# 3. Get the average 
# 4. Sort the values in descending order
cereals.groupby('manufacturer')    .rating    .mean()    .sort_values(ascending=False)


# b. Check to see if the output above coincides with the top 10 ranked cereals by also displaying the manufacturer of those cereals
# > `sort_values()` takes in another argument (the column to be sorted) if it is called on more than one column 
# 

# In[ ]:


# 1. Select the 'manufacturer' and rating' columns
# 2. Sort the rating in descending order 
# 3. Display the top 10 rows 
cereals[['manufacturer','rating']]        .sort_values('rating',ascending=False)        .iloc[:10]


# c. Are the ranking of average ratings driven by outliers? (Display in descending order)
# 
# Note: if our conclusion was right that sugar is correlated to cereal rating, the output will also provide insight if a manufacturer offers a diverse range of cereals (healthy and non-healthy)

# In[ ]:


# 1. Split by manufacturer
# 2. Select the 'rating' column
# 3. Get the standard deviation
# 4. Sort the values in descending order
cereals.groupby('manufacturer')        .rating        .std()        .sort_values(ascending=False)


# d. Find the standard deviation of sugar for each manufacturer and display in descending order to confirm that sugar content is correlated to rating

# In[ ]:


# 1. Split by manufacturer
# 2. Select the 'sugar' column
# 3. Get the standard deviation
# 4. Sort the values in descending order
cereals.groupby('manufacturer')        ['sugar (g)']        .std()        .sort_values(ascending=False)

