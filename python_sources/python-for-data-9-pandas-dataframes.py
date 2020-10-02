#!/usr/bin/env python
# coding: utf-8

# # Python for Data 9: Pandas DataFrames
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# In the [last lesson](https://www.kaggle.com/hamelg/python-for-data-8-numpy-arrays), we learned that Numpy's ndarrays well-suited for performing math operations on one and two-dimensional arrays of numeric values, but they fall short when it comes to dealing with heterogeneous data sets. To store data from an external source like an excel workbook or database, we need a data structure that can hold different data types. It is also desirable to be able to refer to rows and columns in the data by custom labels rather than numbered indexes.
# 
# The pandas library offers data structures designed with this in mind: the series and the DataFrame. Series are 1-dimensional labeled arrays similar to numpy's ndarrays, while DataFrames are labeled 2-dimensional structures, that essentially function as spreadsheet tables.

# ## Pandas Series

# Before we get into DataFrames, we'll take a brief detour to explore pandas series. Series are very similar to ndarrays: the main difference between them is that with series, you can provide custom index labels and then operations you perform on series automatically align the data based on the labels.
# 
# To create a new series, first load the numpy and pandas libraries:

# In[ ]:


import numpy as np
import pandas as pd    


# *Note: It is common practice to import pandas with the shorthand "pd".*
# 
# Define a new series by passing a collection of homogeneous data like ndarray or list, along with a list of associated indexes to pd.Series():

# In[ ]:


my_series = pd.Series( data = [2,3,5,4],             # Data
                       index= ['a', 'b', 'c', 'd'])  # Indexes

my_series


# You can also create a series from a dictionary, in which case the dictionary keys act as the labels and the values act as the data:

# In[ ]:


my_dict = {"x": 2, "a": 5, "b": 4, "c": 8}

my_series2 = pd.Series(my_dict)

my_series2 


# Similar to a dictionary, you can access items in a series by the labels:

# In[ ]:


my_series["a"]


# Numeric indexing also works:

# In[ ]:


my_series[0]


# If you take a slice of a series, you get both the values and the labels contained in the slice:

# In[ ]:


my_series[1:3]


# As mentioned earlier, operations performed on two series align by label:

# In[ ]:


my_series + my_series


# If you perform an operation with two series that have different labels, the unmatched labels will return a value of NaN (not a number.).

# In[ ]:


my_series + my_series2


# Other than labeling, series behave much like numpy's ndarrays. A series is even a valid argument to many of the numpy array functions we covered last time:

# In[ ]:


np.mean(my_series)        # numpy array functions generally work on series


# ## DataFrame Creation and Indexing

# A DataFrame is a 2D table with labeled columns that can each hold different types of data. DataFrames are essentially a Python implementation of the types of tables you'd see in an Excel workbook or SQL database. DataFrames are the defacto standard data structure for working with tabular data in Python; we'll be using them a lot throughout the remainder of this guide.
# 
# You can create a DataFrame out a variety of data sources like dictionaries, 2D numpy arrays and series using the pd.DataFrame() function. Dictionaries provide an intuitive way to create DataFrames: when passed to pd.DataFrame() a dictionary's keys become column labels and the values become the columns themselves:

# In[ ]:


# Create a dictionary with some different data types as values

my_dict = {"name" : ["Joe","Bob","Frans"],
           "age" : np.array([10,15,20]),
           "weight" : (75,123,239),
           "height" : pd.Series([4.5, 5, 6.1], 
                                index=["Joe","Bob","Frans"]),
           "siblings" : 1,
           "gender" : "M"}

df = pd.DataFrame(my_dict)   # Convert the dict to DataFrame

df                           # Show the DataFrame


# Notice that values in the dictionary you use to make a DataFrame can be a variety of sequence objects, including lists, ndarrays, tuples and series. If you pass in singular values like a single number or string, that value is duplicated for every row in the DataFrame (in this case gender is set to "M" for all records and siblings is set to 1.).
# 
# Also note that in the DataFrame above, the rows were automatically given indexes that align with the indexes of the series we passed in for the "height" column. If we did not use a series with index labels to create our DataFrame, it would be given numeric row index labels by default:

# In[ ]:


my_dict2 = {"name" : ["Joe","Bob","Frans"],
           "age" : np.array([10,15,20]),
           "weight" : (75,123,239),
           "height" :[4.5, 5, 6.1],
           "siblings" : 1,
           "gender" : "M"}

df2 = pd.DataFrame(my_dict2)   # Convert the dict to DataFrame

df2                            # Show the DataFrame


# You can provide custom row labels when creating a DataFrame by adding the index argument:

# In[ ]:


df2 = pd.DataFrame(my_dict2,
                   index = my_dict["name"] )

df2


# A DataFrame behaves like a dictionary of Series objects that each have the same length and indexes. This means we can get, add and delete columns in a DataFrame the same way we would when dealing with a dictionary:

# In[ ]:


# Get a column by name

df2["weight"]


# Alternatively, you can get a column by label using "dot" notation:

# In[ ]:


df2.weight


# In[ ]:


# Delete a column

del df2['name']


# In[ ]:


# Add a new column

df2["IQ"] = [130, 105, 115]

df2


# Inserting a single value into a DataFrame causes it to populate across all the rows:

# In[ ]:


df2["Married"] = False

df2


# When inserting a Series into a DataFrame, rows are matched by index. Unmatched rows will be filled with NaN:

# In[ ]:



df2["College"] = pd.Series(["Harvard"],
                           index=["Frans"])

df2


# You can select both rows or columns by label with df.loc[row, column]:

# In[ ]:


df2.loc["Joe"]          # Select row "Joe"


# In[ ]:


df2.loc["Joe","IQ"]     # Select row "Joe" and column "IQ"


# In[ ]:


df2.loc["Joe":"Bob" , "IQ":"College"]   # Slice by label


# Select rows or columns by numeric index with df.iloc[row, column]:

# In[ ]:


df2.iloc[0]          # Get row 0


# In[ ]:


df2.iloc[0, 5]       # Get row 0, column 5


# In[ ]:


df2.iloc[0:2, 5:8]   # Slice by numeric row and column index


# You can also select rows by passing in a sequence boolean(True/False) values. Rows where the corresponding boolean is True are returned:

# In[ ]:


boolean_index = [False, True, True]  

df2[boolean_index] 


# This sort of logical True/False indexing is useful for subsetting data when combined with logical operations. For example, say we wanted to get a subset of our DataFrame with all persons who are over 12 years old. We can do it with boolean indexing:

# In[ ]:


# Create a boolean sequence with a logical comparison
boolean_index = df2["age"] > 12

# Use the index to get the rows where age > 12
df2[boolean_index]


# You can do this sort of indexing all in one operation without assigning the boolean sequence to a variable:

# In[ ]:


df2[ df2["age"] > 12 ]


# Learn more about indexing DataFrames [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html).

# ## Exploring DataFrames

# Exploring data is an important first step in most data analyses. DataFrames come with a variety of functions to help you explore and summarize the data they contain.
# 
# First, let's load in data set to explore: the Titanic Disaster training data. (We will cover reading and writing data in more detail in the next lesson.).

# In[ ]:


titanic_train = pd.read_csv("../input/train.csv")

type(titanic_train)


# Notice that Titanic data is loaded as a DataFrame. We can check the dimensions and size of a DataFrame with df.shape:

# In[ ]:


titanic_train.shape      # Check dimensions


# The output shows that Titanic trainin data has 891 rows and 12 columns.
# 
# We can check the first n rows of the data with the df.head() function:

# In[ ]:


titanic_train.head(6)    # Check the first 6 rows


# Similarly, we can check the last few rows with df.tail():

# In[ ]:


titanic_train.tail(6)   # Check the last 6 rows


# With large data sets, head() and tail() are useful to get a sense of what the data looks like without printing hundreds or thousands of rows to the screen. Since each row specifies a different person, lets try setting the row indexes equal to the passenger's name. You can access and assign new row indexes with df.index:

# In[ ]:


titanic_train.index = titanic_train["Name"]  # Set index to name
del titanic_train["Name"]                    # Delete name column

print(titanic_train.index[0:10])             # Print new indexes


# You can access the column labels with df.columns:

# In[ ]:


titanic_train.columns


# Use the df.describe() command to get a quick statistical summary of your data set. The summary includes the mean, median, min, max and a few key percentiles for numeric columns:

# In[ ]:


titanic_train.describe()    # Summarize the first 6 columns


# Since the columns of a DataFrame are series and series are closely related to numpy's arrays, many ndarray functions work on DataFrames, operating on each column of the DataFrame:

# In[ ]:


np.mean(titanic_train,
        axis=0)          # Get the mean of each numeric column


# To get an overview of the overall structure of a DataFrame, use the df.info() function:

# In[ ]:


titanic_train.info()


# ## Wrap Up

# Pandas DataFrames are the workhorse data structure for data analysis in Python. They provide an intuitive structure that mirrors the sorts of data tables we're using to seeing in spreadsheet programs and indexing functionality that follows the same pattern as other Python data structures. This brief introduction only scratches the surface; DataFrames offer a host of other indexing options and functions, many of which we will see in future lessons.

# ## Next Lesson: [Python for Data 10: Reading and Writing Data](https://www.kaggle.com/hamelg/python-for-data-10-reading-and-writing-data)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
