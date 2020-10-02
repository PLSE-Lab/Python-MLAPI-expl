#!/usr/bin/env python
# coding: utf-8

# # Foreign Exchange rate: A Pandas Tutorial
# 
# ## What are Pandas?
# In the physical world, pandas are a fuzzy black and white animal native to China. However, for programmers, they more frequently refer to a software library for Python that make manipulating and analyzing data easier. 
# 
# We can import this library using the following syntax:

# In[ ]:


import pandas as pd 


# ## What can Pandas do?
# 
# Just like the fluffy animals, the pandas we are referring to can take in a lot of input at a time. Included in pandas are DataFrames, which are class objects that represents data in tables. This data is often read in from one or more excel files.
# 
# To take a look at how to use a DataFrame in a practical way, we will ask and answer some questions about foreign exchange rates between the US and other countries. For the following examples, we will use datasets available in 2 csv files called "Foreign_Exchange_Rates" and "currency_exchange_rate".
# 
# We can import the datasets and save them as DataFrames named `rates` and `ex_rates`.

# In[ ]:


rates = pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')
ex_rates = pd.read_csv('/kaggle/input/currency-excahnge-rate/currency_exchange_rate.csv')


# ## Display
# 
# Now that we have saved our DataFrames, we may want to preview them without loading the whole thing. This is especially helpful when the datasets are particularly large.To do so, we can use `.head()` function.
# - Note that you can specify within the () the number of rows to display, otherwise the default of 5 rows will display

# In[ ]:


rates.head()


# In[ ]:


ex_rates.head(3)


# If we wanted to see the bottom few rows, we can use the `.tail()` function. 
# - Like the `.head` function, you can specify the number of rows within the () otherwise the default of bottom five rows will show

# In[ ]:


rates.tail()


# What can we see about these DataFrames? How are they similar and how are they different? Keep these two DataFrames separate within the following exercises.

# ## Data Attributes
# 
# Some of the differences you observed are differences in data attributes (or characteristics). Here we will take a look at: 1) Finding the size of the dataset, and 2) Finding the column names
# 
# 1. The number of rows and columns our `rates` DataFrame has is different that the number of rows and columns our `ex_rates` DataFrame has. We can get these dimensions using `.shape`.
# 
# The ouput is given as two numbers with the first number as the number of rows and the second as the number of columns.

# In[ ]:


rates.shape


# In[ ]:


ex_rates.shape


# 
# 2. The `rates` DataFrame is organized in columns of different countries. We can see these countries/ column labels using `.columns` 

# In[ ]:


rates.columns


# ## Selecting Data
# 
# Now that we know what our DataSets contain, we might want to select specific data in one of them. We can do this in various ways. Here will will take a look at: 1) using the index number and 2) with conditional selection.

# 1. The `.iloc[]` method takes in one or more indeces numbers

# In[ ]:


ex_rates.iloc[5:7]


# 2. Conditional selection allows us to specify conditions for what we want to look up or select
# 
# For example, supposed we wanted to ask the question: How did 9/11 affect exchange rates?
# 
# To answer this questions, we might want to look up the exchange rates in 2001 to see how that may compare to our expectations.

# In[ ]:


ex_rates[ ex_rates.TIME == 2001]


# ## Summarizing Data
# 
# Pandas also includes various summary statistics such as:
# - `.sum()` to calculate the sum of the rows
# - `.mean()` to calucate the average of the rows
# - `.median()` to calculate the median of the rows
# - `.std()` to calcuate the standard deviation of the rows
# 
# 
# Say I want to know the average exchange rate between the US and Australia between 1950-2017. I can use conditional selection and the `.mean()` summary function together to do so.
# 
# In one line, we can:
# 
# 1) select the rows in which the country is AUS
# 
# 2) specify the column we want summarized in Value
# 
# 3) summarize the data to find the average of the values

# In[ ]:


ex_rates[ ex_rates.LOCATION =='AUS'].Value.mean()


# ## Sorting Data
# 
# Maybe we are also interested in Japan's change in power on the global stage between 1950-2017. One way we can see the change in power is through the yen either appreciating or depreciating in value. Specifically, we may want to look at the periods in time when the yen was most valuable to see if there were certain national or international events that occurred that year. 
# 
# To do so, we can sort the values of our table so that the lowest rates (aka highest value of the yen compared to the US dollar) displays at the top. We can do this using the method `.sort_values()`.
# 
# As before, we do so in one line and two steps by:
# 
# 1) selecting the rows with the the location as JPN
# 
# 2) sorting the rates in ascending order

# In[ ]:


ex_rates[ ex_rates.LOCATION =='JPN'].sort_values('Value', ascending = True)


# ## Split-apply-combine
# 
# One powerful tool of Pandas is that it allows us to easily group data and display some sort of summary statistic. It does so in 3 steps: 
# 
# 1) Splits the data by column
# 
# 2) Applies a summary statistic
# 
# 3) Combines the data into a new dataset
# 
# This is made possible using the `.groupby()` method.
# 
# 
# For example, we can group by country to find the average exchange rate between 1950 - 2017.

# In[ ]:


ex_rates.groupby('LOCATION').Value.mean()


# ## Frequency counts
# 
# Another useful method is finding the size of each group. Say we want to count the number of years with data recorded between 1950-2017 for each country's exchange rate . We can do this using `.groubpy.size()`.
# 

# In[ ]:


ex_rates.groupby('LOCATION').size()


# ## Displaying with plots
# 
# One handy tool included with pandas is the `.plots()` function. There are many different plots for you to choose from (such as bar, line, and scattergram) depending on what data you have and how you want to display it. If we want to quickly visualize the changes in Japanese exchange rates, a line plot may be the best way to view it. We can do this using `.plot.line()`.
# - you can optionanally specify x and y values in your plot

# In[ ]:


ex_rates[ ex_rates.LOCATION == 'AUS'].plot.line(x='TIME', y = 'Value')

