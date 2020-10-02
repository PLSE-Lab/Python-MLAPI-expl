#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Pandas is one of the most popular Python libraries for Data Science and Analytics. It helps you to manage two-dimensional data tables in Python. 

# In[ ]:


import pandas as pd


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "/kaggle/input/"]).decode("utf8"))


# In[ ]:


print(check_output(["ls", "/kaggle/input/loandata"]).decode("utf8"))


# ## DAY 2
# 
# # Recap
# 
# Let's try to load the data (loan data), get information about it

# In[ ]:


df = pd.read_csv("/kaggle/input/loandata/loanData.csv")


# In[ ]:


df.describe()


# <b>Describe a column and get more info</b>

# In[ ]:


df["annual_inc"].describe()


# In[ ]:


df[["annual_inc"]].info()


# <b>How to get list of columns? </b>

# In[ ]:


df.columns


# <b> How do we access third column (nth column)?</b>

# In[ ]:


df.iloc[:, 3]


# In[ ]:


df["final_d"]


# <b>What if I want columns from 4th to 6th?</b>

# In[ ]:


df.iloc[:, 3:6]


# <b>How about (nth ro[](http://)w)?</b>

# In[ ]:


df.iloc[4]


# <b>How about row 3:6?</b>

# In[ ]:


df.iloc[2:5]


# <b>Let's try a combination of selecting rows and columns</b>

# In[ ]:


df.iloc[2:5, 2:5]


# In[ ]:


df.iloc[:, 2].name


# In[ ]:


df.columns.get_loc("issue_d")


# <b>What if we want to drop duplicates? </b>

# 1. Drop duplicates on the entire set

# In[ ]:


df.count()


# In[ ]:


df.drop_duplicates()


# In[ ]:


df.drop_duplicates().count()


# Drop duplicates based on one column

# In[ ]:


df.drop_duplicates(subset ="interest_rate").count()


# In[ ]:


df.count()


# In[ ]:


df.drop_duplicates(subset ="interest_rate", inplace = True)


# In[ ]:


df.count()


# In[ ]:


df_copy = pd.read_csv("/kaggle/input/loandata/loanData.csv")


# In[ ]:


df_copy.head()


# In[ ]:


df_copy["id"].count()


# In[ ]:


df_copy.drop_duplicates(subset="interest_rate", keep=False).count()


# # Sorting

# Let's say that we want to sort the dataframe in increasing order for the scores of the losing team.
# 
# 1. Sort in ascending and descending order

# In[ ]:


df.sort_values("annual_inc").head()


# In[ ]:


df.sort_values("annual_inc", ascending = False).head()


# 2. Sort by multiple columns

# In[ ]:


df.sort_values(["home_ownership", "annual_inc"])


# # Filtering Rows Conditionally

# Now, let's say we want to find all of the rows that satisy a particular condition. For example, I want to find all the elements where interest rate is greater than 15.2.

# In[ ]:


df_int = df[df["interest_rate"] > 15.2]


# This also works if you have multiple conditions. I want to find all the elements where interest rate is greater than 15.2 and total payment is greater than 20000

# In[ ]:


df_conditioned = df[(df["interest_rate"] > 15.2) & (df["total_pymnt"] > 20000)]


# In[ ]:


df_conditioned.count()


# In[ ]:


df_conditioned.head()


# # Grouping

# Another important function in Pandas is **groupby()**. This is a function that allows you to group entries by certain attributes (e.g Grouping entries by Wteam number) and then perform operations on them. 

# In[ ]:


df_copy.groupby("home_ownership").size()


# In[ ]:


type(df_copy.groupby("home_ownership"))


# In[ ]:


df_copy.groupby("home_ownership").groups["MORTGAGE"]


# In[ ]:


df_copy.groupby("home_ownership")["interest_rate"].max().head()


# In[ ]:


df_copy.groupby("home_ownership")["interest_rate"].mean().head()


# Aggregation withing groupby

# In[ ]:


df.groupby("home_ownership").agg({"interest_rate": ["min", "max", "mean", "median"],
                                 "annual_inc":["min", "max", "mean", "median"]})


# In[ ]:


df_copy.groupby("home_ownership").size()


# In[ ]:


df_copy.groupby(["home_ownership", "income_category"]).size()


# ## Merging two datasets

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Data Cleaning

# One of the big jobs of doing well in Kaggle competitions is that of data cleaning. A lot of times, the CSV file you're given (especially like in the Titanic dataset), you'll have a lot of missing values in the dataset, which you have to identify. The following **isnull** function will figure out if there are any missing values in the dataframe, and will then sum up the total for each column. In this case, we have a pretty clean dataset.

# In[ ]:


df.isnull().sum()


# If you do end up having missing values in your datasets, be sure to get familiar with these two functions. 
# * **dropna()** - This function allows you to drop all(or some) of the rows that have missing values. 
# * **fillna()** - This function allows you replace the rows that have missing values with the value that you pass in.

# In[ ]:


df.dropna()
df.fillna(0)


# # Visualizing Data

# An interesting way of displaying Dataframes is through matplotlib. 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ax = df['interest_rate'].plot.hist(bins=20)
ax.set_xlabel('Interest Rate Histogram')


# In[ ]:


## 2 columns
df.plot(y='recoveries', x='total_pymnt')


# # Other Useful Functions

# * **drop()** - This function removes the column or row that you pass in (You also have the specify the axis). 
# * **agg()** - The aggregate function lets you compute summary statistics about each group
# * **apply()** - Lets you apply a specific function to any/all elements in a Dataframe or Series
# * **get_dummies()** - Helpful for turning categorical data into one hot vectors.
# * **drop_duplicates()** - Lets you remove identical rows

# # Lots of Other Great Resources

# Pandas has been around for a while and there are a lot of other good resources if you're still interested on getting the most out of this library. 
# * http://pandas.pydata.org/pandas-docs/stable/10min.html
# * https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python
# * http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/
# * https://www.dataquest.io/blog/pandas-python-tutorial/
# * https://drive.google.com/file/d/0ByIrJAE4KMTtTUtiVExiUGVkRkE/view
# * https://www.youtube.com/playlist?list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y
