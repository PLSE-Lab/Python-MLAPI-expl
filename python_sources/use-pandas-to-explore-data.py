#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Welcome to this kernel. I'm super excited to make this tutorial in which we will walk through the Pandas library, one of my favourites for data analysis. Data exploration gets easy when we know which tools to use and when we know the capabilities the tools offer. 
# 
# I assume you have some prior knowledge of python such as data types (list, dictionnaries... ).
# 
# In this Kernel I will do a complete data exploration of several datasets. 
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/4/45/Pandas_logo.png)
# 
# This notebook is easy to understand and completely beginer friendly. If you already know some Pandas, you can use this kernel as a reference notebook in case you need it.

# # What is Pandas?
# *Python Data Analysis Library*
# Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

# **1. Create our own dataset using pandas**

# There are several ways of creating a dataset using Pandas. If you want to create from scratch a dataset for analysis because you have the raw data and instead of using a spreadsheet, you want to write it directly in Pandas you can use it.

# In[ ]:


# We'll start by importing the pandas library
import pandas as pd
# Since we'll be generating numbers, we'll need to import the numpy library
import numpy as np


# In[ ]:


countries = pd.DataFrame({"Country":['France', 'Germany', 'Spain', 'Belgium',
                                  'Russia'], "GDP(2017)":[2500, 3600, 1300, 500, 1600],})
countries


# WHat we just did is using a dictionary in which we have a key that is the column name and the values are the row of each columns. 
# 
# We can alternatively do this using the `DataFrame` arguments

# In[ ]:


pd.DataFrame(np.random.normal(2.5, 1, (10,3)), 
             columns= ["1st column", "2nd column", "3rd Column"])


# **2. Read dataset from different file formats**
# 
# The pandas read function lets us read data from a large variety of data. We can read from Excel spreadsheets, CSV ...

# CSV files are one of the most popular dataset format. The Pandas read_csv doesn't only read in csv files but also tabular separated values, we just need to specify it in the sep argument. 

# In[ ]:


dataset = pd.read_csv("../input/fortune1000.csv")


# In[ ]:


dataset.head(3)
# The head and tail methods give us the first and last n elements in our datasets, n=5 by default


#  **3. Explore that dataset**

# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()


# What is very interesting with pandas is that sometimes  the outpout it gives are own pandas series or dataframes. We can take advantages of that to further explore the data.

# In[ ]:


# let's for example check the type of this output
type(dataset.describe())


# In result we have the `pandas.core.frame.DataFrame`. What's really interesting is that all the attributes and methods of a dataframe can apply. Keep that in mind whenever you explore a Pandas object.

# In[ ]:


dataset.describe()['Profits']


# Now it gives us a pandas Series with statistical moments as index. All pandas Series attributes and methods apply on that Series

# In[ ]:


dataset.describe()['Profits'].count()


# Let's now say we want to see each unique values are in a column.

# In[ ]:


dataset['Sector'].unique()


# In[ ]:


dataset['Sector'].nunique()


# The `unique()` and `nunique()` methods give us the unique values and their total number

# Since our dataset has 1000 rows, we may wonder wich unique value occurs the most. To do that we can use the `value_count()` which applies on a Pandas Series, and since it's a Series we can also apply the `head` or `tail`methods to display the first or last n rows.

# In[ ]:


dataset['Sector'].value_counts().head()


# In our dataset we have a Rank column, instead of having it as a column, we can define it as the index of the dataset

# In[ ]:


dataset.set_index("Rank", inplace = True)
dataset.head(3)


# We need to specify to set the **inplace** argument to true to commit the change we just do.

# **4. Working with the numerical columns**

# In[ ]:


dataset["Revenue"].head()


# When we select a numerical column we have a Pandas Series; we can then use some methods to analyse it. 
# The `describe()` methods already gives us meaningful information on the numerical colums, so we're going to use some other useful methods to quickly explore the data

# In[ ]:


dataset["Profits"].nlargest(3)


# In[ ]:


dataset["Profits"].nsmallest(4)


# Not only these methods give the n largest/smallest profits, they also give us the the index which we set to be the rank

# **4. Summarizing the data within groups**

# Let's now say, instead of having the descriptive statistics for all the dataset, we want to summarize the data based on the categories we have in another column of the data.

# In[ ]:


dataset.groupby("Sector").describe().head()


# In[ ]:


type(dataset.groupby("Sector").describe())


# Remember this output is itself a Pandas DataFrame.

# In[ ]:


# We can grab the single column we want
dataset.groupby("Sector").describe()["Profits"].head()


# We have a more flexible way to analyse the data using the `agg()`method on a grouped dataset. 

# In[ ]:


dataset.groupby("Sector").agg('mean').head()


# We can use the aggregation on different columns

# In[ ]:


dataset.groupby("Sector").agg({"Profits":['min','max'],"Revenue":['mean','median']}).head()


# **5. Apply method**

# The `.apply()` method helps us apply a function on every row or column of the dataset.
# 
# Let's say we want to grab the State from the **Location** column in order to explore the companies by States

# In[ ]:


dataset["Location"].head()


# The city is separated by a comma, so we can use string methods to grab some part of this column.

# In[ ]:


dataset["Location"].apply(lambda loc:loc.split(',')[1]).head()


# Let me break down this code, and introduce the concept of the `.apply()` method. 
# 
# I applied a lambda function on the Location column. The loc keyword is just a variable that is defined in the scope of the lambda function : for each row in the Location column, take the row, split it by ',' and take the second element (python indexing starts with 0).

# If you are not familiar with the idea of using a lambda function on a text column, you can directly use the python built-in `.split()` string method. In Pandas you need to specify the `str`prefix before using this method on the column. 
# 
# The `str`coerces any data type into a Pandas text Series

# In[ ]:


dataset["Location"].str.strip().str.split(',').str.get(1).head()


# Now, let's add a State column to the dataset using the method I just present.

# In[ ]:


dataset["State"] = dataset["Location"].apply(lambda loc:loc.split(',')[1])


# In[ ]:


dataset.head()


# How many unique states are represented in the dataset.

# In[ ]:


dataset["State"].nunique()


# Let's aggregate the data by State

# In[ ]:


dataset.groupby("State").agg({'Profits': ['min', 'max', 'mean']}).head()


# Let's use again the `.apply()`method but this time with our own custom function

# Let's say we want to rate the Profit of each company as follow: 
# 
# **Negative** is the profit is negative
# 
# **Average** it is in the range of 0 to 3500m
# 
# **High** if above 3500 m
# 

# In[ ]:


def rate_profit(profit):
    if profit <0:
        return "Negative"
    elif (profit >0) & (profit <=3500):
        return "Average"
    else:
        return "High"


# In[ ]:


rate_profit(200)


# Let's now create a column named Rating

# In[ ]:


dataset["Rating"] = dataset["Profits"].apply(rate_profit)


# In[ ]:


# Check the output
dataset.head(3)


# In[ ]:


dataset.groupby("Rating").agg('count')["Company"]


# **6. The `map()` method**

# Let's now talk about the `.map()`method
# 
# The `.map()`methods works like the `apply()`but the difference is that for every value within a column it returns a predefined value. 
# 
# For example, we can map the Ratings we just created to something else. It wors better in doing

# In[ ]:


dataset["Rating"].map({'High':'AAA', 'Average':'BAA', 'Negative':'BBB'}).head()


# **7. Filtering the dataset**

# In[ ]:


dataset.head(3)


# **a.  Filter with conditions**

# Filtering a dataset with conditions is often called masking. The mask is just the condition we pass in the dataset to subset it. 
# To subset with many conditions, it is better to use **&** instead of AND and **|** instead of OR to avoid python logical issues.

# In[ ]:


dataset[dataset["Rating"] == "High"].head()


# We can also select companies which have a profit between 1500 and 3000

# In[ ]:


dataset[dataset["Profits"].between(1500,3000)].head()


# We can also combine these conditions to filter the data on many columns values.

# In[ ]:


dataset[(dataset["State"].str.contains('CA'))& (dataset["Rating"] =="High")].head()


# In[ ]:


dataset[(dataset["Sector"] == "Health Care") &(dataset["Profits"]>3000)].head()


# **8. Pandas built-in visualizations**

# We can visualize our dataset pandas visualization methods based on matplotlib.

# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12,6]


# In[ ]:


# Histogram of the Revenues
dataset["Revenue"].plot(kind = 'hist', bins = 100)


# We can also draw a countplot of a categorical column. Just think of it as if instead of outputting a DataFrame or Series, you decide to output a plot.

# In[ ]:


dataset["State"].value_counts().plot(kind = 'bar')


# We can also plot statistical visualisation plot

# In[ ]:


np.log(dataset[dataset["Profits"]>0]['Profits']).plot(kind = 'box')
plt.title("Logarithm of the Profit")
plt.ylabel("log")


# If we grab a single Pandas Series we need to use the `.plot(kind)` method. But on a DataFrame, we can directly select the plot we want to use.

# In[ ]:


dataset.hist(by = "Rating", column= "Revenue", bins = 50)
plt.show()


# In[ ]:


dataset.plot.scatter("Revenue", "Profits")


# **9. Prepare your data for a Machine Learning Algorithm**

# When we want to use a Machine Learning Algorithm, we need to prepare the dataset to have the required shape for the models. The scikit-learn library comes with lots of preprocessing tools to make the data ready but Pandas also offers excellent ways to prepare your data.

# **a. Dummy variables**

# Suppose we want to include the Rating colum in our model. Since it's a categorical feature, we need to encode it to make it numeric than create dummy variables from it. But Pandas has an easiest way of doing so with the `pd.get_dummies()`function.

# In[ ]:


model_data = pd.get_dummies(dataset, columns= ["Rating"], drop_first= True)
model_data.head(3)


# This function automatically adds two dummy columns to our dataset, by specifying **`drop_first = True`**, it deletes one of the column to eliminate multicolinearity in the model. It makes it very easy !

# **b. Dealing with missing values**

# In[ ]:


np.sum(model_data.isna())


# In[ ]:


# We can also find the percentage
(np.sum(model_data.isna())/len(model_data.index))*100


# We have some missing values in this dataset, with Pandas it is easy to impute values

# In[ ]:


# Locate where are the missing values
model_data[model_data.isna().any(axis = 1)].head()


# In[ ]:


missing_index = model_data[model_data.isna().any(axis = 1)].index


# We can fill missing values with the `.fillna()`method. 
# 
# If we have an idea of what the missing values can be then we need only to specify value = to the value we guess to be. 
# 
# Else we can use the `method` arguments. 
# 
# `ffill`: replaces the the missing value with the previous value in the dataset

# In[ ]:


model_data.fillna(method='ffill').head()


# One of the best way we can use to impute missing values is the interpolate methods.

# In[ ]:


model_data.interpolate(inplace=True)


# In[ ]:


model_data.iloc[list(missing_index),].head()


# In[ ]:




