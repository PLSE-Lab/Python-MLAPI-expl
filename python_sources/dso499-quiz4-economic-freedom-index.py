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


# # **0. Introduction**
# This dataset includes 183 countries and data that describe their economics. It can be used by many individuals or organizations. Some examples include real estate investors who are looking for great countries to invest, United Nations who are investigating the reasons behind low GDP growth rate, or a university student who wants to take a look at the macroeconomic situations of certain countries.
# 
# In this pandas tutorial, you will see many practical cases that utilize this dataset and learn how to analyze the data using pandas simulataneously.

# # **1. Display Data**
# First of all, we need to use panda to read the csv file that stores the dataset into a dataframe, and then display the first 5 rows of that dataframe.
# * read_csv(): a method in panda module that reads a comma-separated values (csv) file into DataFrame.
# * head(): Return the first n rows. It is useful for quickly testing if your object has the right type of data in it.
#     - paramter: an integer, default value = 5, if there is no parameter in head(), that means displaying first 5 rows
# - Dataframe: a class object for representing tables, usually created from csv files

# In[ ]:


economics = pd.read_csv('/kaggle/input/the-economic-freedom-index/economic_freedom_index2019_data.csv',index_col = 'Country',encoding='ISO-8859-1')
economics.head()
# economics is a dataframe created from csv file
# index_col is setting the dataframe's index to "Country" Column


# As you can see in the above table, the economics dataframe shows every country's economic freedom index and its ranking in 2019. It also shows the related economic indices like Property rights, judicial effectiveness, GDP, etc.
# 
# The column 2019 score is the economic freedom index, which is to measure the degree of economic freedom in the world's nations. More info about the index can be found [here](http://https://en.wikipedia.org/wiki/Index_of_Economic_Freedom).

# # **2. Data Attributes**
# * .columns: get the column labels of the DataFrame.
# * .shape: Return a tuple representing the dimensionality of the DataFrame. 

# What are the columns of the dataframe economics?

# In[ ]:


economics.columns.to_list()
#to_list() change the output to list format


# As we can see, there are some redundancies in this dataset, which will be fixed latter. For example, The columns "Country Name","WEBNAME" and "Country" mean the same things. 

# In[ ]:


economics.shape


# The result means that there are 186 rows and 33 columns, meaning that the dataset includes 186 countries'data and there are 33 features of each country.

# # **3. Select Columns**
# 
# This dataframe has some redundancies in it as mentioned before, and we are not interested in analyzing all of its columns, so we need to select certain columns to prevent redundancies and select only the columns that we are most interested in.

# In[ ]:


economics = economics[['Region','World Rank','Region Rank','2019 Score',
                        'Population (Millions)','GDP Growth Rate (%)'
                       ,'Unemployment (%)','Inflation (%)']]
#We can use a list of column names to select columns of a dataframe
economics.head()


# Now, the updated dataframe as above does not have any redundancies and we assume that it only includes the columns that we are interested in.

# # **4. Select Cells**
# 
# Practice Case 1): a scholar from Brazil might be interested in the unemployment rate of Brazil in 2019. To help this scholar find out the number, we need to select cell using .loc[]
# * .loc[]: Access a group of rows and columns by label(s)

# In[ ]:


economics.loc['Brazil','Unemployment (%)']
# first argument is the name of the row (index name), and the second argument is the column name
# By using both row and column names, we can locate the cell and return the cell value


# # **5. Conditional Selectiion**
# 
# Practice Case 1): Tom, an real estate investor, wants to see every country's name that has GDP growth rate greater or equal to 8.0 because he believes real estate returns are proportional to GDP growth rate.
# 
# - .index: returns the index of selected rows. In this case, the index column is the country column.

# In[ ]:


# Step 1: select all the rows with GDP grotwh rate larger than or equal to 8.0
# Step 2: return the index of these rows (country names)
# Step 3: change the results in to list format
economics[economics['GDP Growth Rate (%)']>=8.0].index.to_list()


# # **6. Sort**
# Practice Case: an investor wants to invest his/her assets into a country that has high economic freedom. So he wants to take a look at the top 5 countries that has the highest 2019 score of economic freedom index before making a decision. He also wants other economics data for these 5 countries as well. 
# 
# We can use sort_values() to help him/her get the result.
# 
# * sort_values(): Sort by the values along either axis, which means that we can sort by the values in any columns
#     * ascending: use boolean to clarify whether to sort ascending vs. descending.
# * iloc[]: integer-location based indexing for selection by position    
#     * The first argument refers to the slection of rows, the second argument refers to columns    

# In[ ]:


# Step 1: sort the table by using values in 2019 Score in descending order
# Step 2: use iloc[] to select the top 5 rows only
economics.sort_values('2019 Score',ascending=False).iloc[:5]


# # **7. Split-Apply-Combine**
# Practice Case 1): a macroeconomics scholar wants to check the average values of 2019 Score and GDP Growth rates for each region.
# 
# * groupby(): The method can split rows into several categories, and the method produces a groupby object. It can be used to group large amounts of data and compute operations on these groups.
# * mean(): a method used to summarize values of certain rows, returns the mean value of the rows' values.

# In[ ]:


# Step 1: Split by Region
# Step 2: Select the relevant columns ('2019 Score' and 'GDP Growth Rate (%)')
# Step 2: use mean() to calculate average values of each economic indices
economics.groupby('Region')[['2019 Score','GDP Growth Rate (%)']].mean()


# Practice Case 2): A non profit organization (which dedicates to help reviving low-performing economics) wants to know the country that has the lowest economic freedom index in each region for further investigation.
# 
# * idxmin():Return index of first occurrence of minimum over requested axis. In this case, index is country.

# In[ ]:


# Step 1: Split by Region
# Step 2: Select the column '2019 Score'
# Step 2: use idxmin() to find out the country that has minimum score in each region
economics.groupby('Region')['2019 Score'].idxmin()


# According to the result, this NPO might want to start investigating one of these 5 countries

# Practice Case 3): United Nations want to see which region has the largest standard deviation in Economic Freedom Index.
# 
# * std(): calculates the standard deviation of selected rows

# In[ ]:


economics.groupby('Region')['2019 Score'].std()


# # **8. Frequency Counts**
# Practice Case: a college student is curious about how many countries in each region are included in this dataset to explore if the dataset is inclusive enough.
# 
# * groupby.size(): return the group sizes/counts

# In[ ]:


economics.groupby('Region').size()


# According to Wikipedia,there are 35 countries in Americas,48 countries in APAC, 44 countries in Europe,17 countries in Middle East and North Africa, and 48 countries in Sub-Saharan Africa. Hence, we can see that except for APAC, all the other regions include more than 90% of their countries.

# # **9. Plot**
# Practice Case: a scholar want to use a horizontal bar chart to show each region's average 2019 Score in descending order
# 
# * plot.barh(): Make a horizontal bar plot.

# In[ ]:


# Step 1: Split by Region
# Step 2: Select columns '2019 Column'
# Step 3: Summarize by mean()
# Step 4: Sort by values, note that horizontal plot plots in reverse order, so we need to sort in ascending order to make plot in descending order
# Step 5: plot the horizontal bar plot
economics.groupby('Region')['2019 Score'].mean().sort_values().plot.barh()


# From the horizontal bar plot above, we can conclude that according to the dataset, in 2019, Europe enjoys the highest economic freedom level in average.

# # **10. Words from Writer**
# There are still thousands of ways to practially use this dataset. Please refer to pandas documentation for more methods and data attributes. Meanwhile, feel free to use this notebook to explore on your own!
# 
