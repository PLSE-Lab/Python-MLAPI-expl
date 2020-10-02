#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Pandas is one of the most popular Python libraries for Data Science and Analytics. It helps you to manage two-dimensional data tables in Python. 

# In[ ]:


import pandas as pd


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "/kaggle/input/"]).decode("utf8"))


# # Core Components of Pandas

# 1. Series: 
# Essentially a single column
# 2. DataFrame:
# Muti-dimensional table made up of series
# 
# 
# Let's try to create our dataframe using dict, define a dict for purchases, with keys: apples and oranges. 
# 

# In[ ]:


data = {
    'apples': [3, 2, 0, 1], 
    'oranges': [0, 3, 7, 2]
}


# In[ ]:


purchases = pd.DataFrame(data)


# In[ ]:


purchases


# Let's try to add two new columns, bananas and the purchase ID

# In[ ]:


purchases["bananas"] = [4, 5, 6, 7]
purchases["purchase_id"] = [1, 2, 3, 4]


# In[ ]:


purchases.head()


# # Loading in Data

# Pandas can read data from a lot of different formats, let's try to read the data from a csv format. 
# We can also read data from excel format (pd.read_excel)

# Just think of it as a table for now. 

# In[ ]:


df = pd.read_csv("/kaggle/input/loandata/loanData.csv")


# In[ ]:


#df = pd.read_csv('/kaggle/input/seasoncompactresults/RegularSeasonCompactResults.csv')


# # The Basics

# Now that we have our dataframe in our variable df, let's look at what it contains. We can use the function **head()** to see the first couple rows of the dataframe (or the function **tail()** to see the last few rows).

# In[ ]:


df.head(10)


# In[ ]:


df.tail()


# We can see the dimensions of the dataframe using the the **shape** attribute

# In[ ]:


df.shape


# ![](http://)We can also extract all the column names as a list, by using the **columns** attribute and can extract the rows with the **index** attribute

# In[ ]:


df.index.tolist()


# In[ ]:


df.columns.tolist()


# In order to get a better idea of the type of data that we are dealing with, we can call the **describe()** function to see statistics like mean, min, etc about each column of the dataset. 

# In[ ]:


df.describe()


# In[ ]:


df.info()


# Okay, so now let's looking at information that we want to extract from the dataframe. Let's say I wanted to know the max value of a certain column. The function **max()** will show you the maximum values of all columns

# In[ ]:


df.max()


# 1. Then, if you'd like to specifically get the max value for a particular column, you pass in the name of the column using the bracket indexing operator

# In[ ]:


df['emp_length_int'].max()


# Let's try to find the mean installment paid

# In[ ]:


df['installment'].mean()


# But what if we want to see the whole record with maximum employee length? We can call the **argmax()** function to identify the row index

# In[ ]:


df['installment'].idxmax()


# One of the most useful functions that you can call on certain columns in a dataframe is the **value_counts()** function. It shows how many times each item appears in the column. This particular command shows the number of games in each season

# In[ ]:


df['purpose'].value_counts()


# # Acessing Values

# Then, in order to get attributes about the game, we need to use the **iloc[]** function. Iloc is definitely one of the more important functions. The main idea is that you want to use it whenever you have the integer index of a certain row that you want to access. As per Pandas documentation, iloc is an "integer-location based indexing for selection by position."

# In[ ]:


df['loan_amount'].max()


# In[ ]:


max_loan = df.iloc[[df['loan_amount'].idxmax()]]
max_loan[["loan_amount"]]
len(max_loan)


# When you see data displayed in the above format, you're dealing with a Pandas **Series** object, not a dataframe object.

# In[ ]:


type(df.iloc[[df['loan_amount'].idxmax()]])


# In[ ]:


type(df.iloc[[df['loan_amount'].idxmax()]])


# The other really important function in Pandas is the **loc** function. Contrary to iloc, which is an integer based indexing, loc is a "Purely label-location based indexer for selection by label". Since all the games are ordered from 0 to 145288, iloc and loc are going to be pretty interchangable in this type of dataset

# In[ ]:


df.iloc[:3]


# In[ ]:


df.loc[:3]


# Notice the slight difference in that iloc is exclusive of the second number, while loc is inclusive. 

# Below is an example of how you can use loc to acheive the same task as we did previously with iloc

# In[ ]:


max_loan = df.loc[[df['loan_amount'].idxmax(axis = 0)]]
max_loan["loan_amount"]


# A faster version uses the **at()** function. At() is really useful wheneever you know the row label and the column label of the particular value that you want to get. 

# In[ ]:


df.at[df['loan_amount'].idxmax(), 'loan_amount']


# If you'd like to see more discussion on how loc and iloc are different, check out this great Stack Overflow post: http://stackoverflow.com/questions/31593201/pandas-iloc-vs-ix-vs-loc-explanation. Just remember that **iloc looks at position** and **loc looks at labels**. Loc becomes very important when your row labels aren't integers. 

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


df[["year"]].describe()


# In[ ]:


df[["year"]].info()


# <b>How to get list of columns? </b>

# In[ ]:


df.columns


# <b> How do we access third column (nth column)?</b>

# In[ ]:


df.iloc[:, 5]


# <b>What if I want columns from 4th to 6th?</b>

# In[ ]:


df.iloc[ :,4:7]


# <b>How about third row (nth row)?</b>

# In[ ]:


df.iloc[3]


# <b>How about row 3:6?</b>

# In[ ]:


df.iloc[3:7]


# <b>Let's try a combination of selecting rows and columns</b>

# In[ ]:


df.iloc[3:7, 3:7]


# In[ ]:


df.iloc[:,3].name


# In[ ]:


df.columns.get_loc("final_d")


# <b>What if we want to drop duplicates? </b>

# 1. Drop duplicates on the entire set

# In[ ]:


df.drop_duplicates()["id"].count()


# Drop duplicates based on one column

# In[ ]:


df.drop_duplicates(subset ="interest_rate")["id"].count()


# Let's check the count of our dataset now

# In[ ]:


df.drop_duplicates()["id"].count()


# In[ ]:


df.count()


# Any idea why? 

# In[ ]:


# dropping ALL duplicte values 
df.drop_duplicates(subset ="interest_rate", 
                     keep = "first", inplace=True)


# In[ ]:


df.count()


# # Sorting

# Let's say that we want to sort the dataframe in increasing order for the scores of the losing team.
# 
# 1. Sort in ascending and descending order

# In[ ]:


df.sort_values('loan_amount').head()


# In[ ]:


df.sort_values('loan_amount', ascending=False)


# 2. Sort by multiple columns

# In[ ]:


df.sort_values(['home_ownership', 'annual_inc'])


# # Filtering Rows Conditionally

# Now, let's say we want to find all of the rows that satisy a particular condition. For example, I want to find all the elements where interest rate is greater than 15.2.

# In[ ]:


df[df['interest_rate'] > 15.2]


# This also works if you have multiple conditions. I want to find all the elements where interest rate is greater than 15.2 and total payment is greater than 20000

# In[ ]:


df[(df['interest_rate'] > 15.2) & (df['total_pymnt'] > 20000)]


# # Grouping

# Another important function in Pandas is **groupby()**. This is a function that allows you to group entries by certain attributes (e.g Grouping entries by Wteam number) and then perform operations on them. 

# In[ ]:


df.groupby('home_ownership').size()


# In[ ]:


type(df.groupby('home_ownership'))


# In[ ]:


df.groupby('home_ownership').groups.keys()


# In[ ]:


df.groupby('home_ownership').groups["MORTGAGE"]


# Aggregation withing groupby

# In[ ]:


df.groupby('home_ownership')['interest_rate'].mean().head()


# In[ ]:


df.groupby('home_ownership')['interest_rate'].max().head()


# Multiple aggregations withing same groupby

# In[ ]:


df.groupby('home_ownership').agg({"interest_rate": ["min", "max", "mean"],
                                 "annual_inc": ["min", "max", "mean"],
                                 "installment": ["min", "max", "mean"] })


# Grouping by multiple columns

# In[ ]:


df.groupby(['home_ownership','income_category']).size()


# ## Merging two datasets

# In[ ]:


user_device = pd.read_csv("/kaggle/input/user-stats/user_device.csv")
user_usage = pd.read_csv("/kaggle/input/user-stats/user_usage.csv")


# In[ ]:


user_device.head()


# In[ ]:


user_device.head(10)


# In[ ]:


result1 = pd.merge(user_usage,
                user_device[['use_id', 'platform', 'device']],
                 on='use_id')


# In[ ]:


result1.count()


# In[ ]:


result2 = pd.merge(user_usage,
                 user_device[['use_id', 'platform', 'device']],
                 on='use_id')


# In[ ]:


result2.count()


# In[ ]:


result3 = pd.merge(user_usage,
                 user_device[['use_id', 'platform', 'device']],
                 on='use_id', how = 'left')


# In[ ]:


result3.count()


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
