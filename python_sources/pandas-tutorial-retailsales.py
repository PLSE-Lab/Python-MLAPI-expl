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


# This is a guide introducing potential ways of using pandas to interpret sales dataset. First of all, let's import the dataframe we are using by
# 
# **pd.read_csv(' ')**
# 
# the string in the brackets should be the route to access your file

# In[ ]:


sales = pd.read_csv('/kaggle/input/retail-business-sales-20172019/business.retailsales.csv',index_col=0)


# # Display

# To see the dataset on python, we can use 
# 
# **.head(*n*)**
# 
# to display the top *n* rows of data, default *n*=5

# In[ ]:


sales.head()


# We can also use 
# 
# **.tail(*n*)**
# 
# to display the bottom *n* rows of data, default *n* = 5

# In[ ]:


sales.tail(10)


# # Data Attributes

# When we try to analyze this dataset, how do I check what information is provided?
# We can use
# 
#  **.columns**
# 
#  to display the name(index) of each column

# In[ ]:


sales.columns


# To quickly validate the comprehensiveness of this dataset, we can use
# 
# **.shape**
# 
# to know how many rows (number of samples) and columns (categories of information) are included.

# In[ ]:


sales.shape


# # Select

# To focus on only one aspect of the data, we can use
# 
# ** *DataFrame name*['*column name*']**
# 
# to select a specific column
# 

# In[ ]:


sales['Returns']


# What if we only want to focus on one product?
# We can display data with specific aspects by using
# 
# **.loc[]**
# 
# where in the sqaure brackets could be the name of the row(s) you want to see

# In[ ]:


sales.loc['Basket']


# To view the return of all baskets sold, we can combine the two codes above and display

# In[ ]:


sales['Returns'].loc['Basket']


# or

# In[ ]:


sales.loc['Basket']['Returns']


# To display multiple columns, we can combine them as a list

# In[ ]:


sales[ ['Gross Sales', 'Returns']].loc['Basket']


# # Summarize

# Now, to get a quick summary of the dataset, we can use function
# 
# **.sum()**

# In[ ]:


sales.sum()


# This can also be tailored to specific products by combining **.loc[]** and **.sum()** 

# In[ ]:


sales.loc['Basket'].sum()


# Similarily, we can get the average data by using
# 
# **.mean()**

# In[ ]:


sales.loc['Basket'].mean()


# # Sort

# Now I want to know which are the sales that generated most net profits.
# We can sort the data by using
# 
# **.sort_values()**
# 
# The default order of this function is ascending. To reverse it, we can add
# 
# **ascending = False** 
# ![](http://)to the bracket.
# 
# 

# In[ ]:


sales['Total Net Sales'].sort_values(ascending = False).head(10)


# For clarity, we can also write a long code like this

# In[ ]:


#select category 'Total Net Sales'
#sort values descendingly
#display the first 10 rows
sales['Total Net Sales']    .sort_values(ascending = False)    .head(10)


# 

# Alternatively, if I only want to know the one sale with the max net profits, we can use
# 
# **.max()** 
# 
# to access the max among our data

# In[ ]:


sales['Total Net Sales'].max()


# and
# 
# **.idxmax()**
# to access the product(index) associated with that value

# In[ ]:


sales['Total Net Sales'].idxmax()


# # Split-apply-combine

# When we look at a whole book of sales data, categorizing them can be very helpful. With pandas, we can *split* our data with specific criteria, and *apply* or *combine* different calculations on these data. We can split the dataset by
# 
# **.groupby(' ')**
# 
# in which the string should be the name of category you want to group by.

# In[ ]:


sales.groupby('Product Type').sum()


# In[ ]:


#split the dataframe by category 'Product Type' 
#and select column name 'Returns'
#generate the sum of each category
#sort values ascendingly

sales.groupby('Product Type')['Returns']    .sum()    .sort_values()


# To count how many items are in the groups, we can use the function
# 
# **.size()**

# In[ ]:


sales.groupby('Product Type').size()


# # Plots

# Finally, we can draw graphs with these data by using 
# 
# **.plot()**

# In[ ]:


#split data by category 'product type' 
#calculate size of each group
#plot the data
sales.groupby('Product Type').size().plot()


# we can draw many different types of plots with pandas, for example:

# In[ ]:


sales.groupby('Product Type').size().plot.pie()


# In[ ]:


#Split the data by 'Product Type' and select column 'Total Net Sales'
#calculate the sum of each group
#sort data in ascending order
#make a horizontal bargraph 
sales.groupby('Product Type')['Total Net Sales']    .sum()    .sort_values()    .plot.barh()


# # Analysis

# Here are some questions we can answer from analyzing this dataset with pandas.

# 1. What are the top 3 popular product categories and how mant of those items were sold?

# In[ ]:


#Split the data and select column 
#calculate the sum of each group
#sort data in ascending order
#display the first three rows
sales.groupby('Product Type')['Net Quantity']    .sum()    .sort_values(ascending=False)    .head(3)


# 2. What is the average discount given for each type of products?

# In[ ]:


#split dataframe by categories
#calculate discount over net quantity for each group
#round to two decimal places
#sort values ascendingly
a = sales.groupby('Product Type').sum()

(a['Discounts']/a['Net Quantity'])    .round(2)    .sort_values()


# 3. Which type of product have the highest average price? How about the lowest?

# In[ ]:


#split dataframe by category and sum up values
#caculate average price
#get the index of the max price value
a = sales.groupby('Product Type').sum()
b = (a['Gross Sales']/a['Net Quantity'])
b.idxmax()


# In[ ]:


b.idxmin()

