#!/usr/bin/env python
# coding: utf-8

# # Python Pandas for Beginners 
# 
# ### Priyaranjan Mohanty

# **What is Pandas ?**
# 
# Pandas is a software library written for the Python programming language for data manipulation and analysis. 
# In particular, it offers data structures and operations for manipulating numerical tables and time series. 
# It is free software released under the three-clause BSD license.
# 
# The name is derived from the term "panel data", an econometrics term for data sets that include observations over multiple time periods for the same individuals.

# **Pandas support 2 types of Data Structures :- **
# 
#     a) Dataframe
#     b) Series 
# 
# **Dataframe -**
# pandas DataFrame (in a Jupyter Notebook) appears to be nothing more than an ordinary table of data consisting of rows and columns. Hiding beneath the surface are the three components--the index, columns, and data (also known as values)
# 

# So as to understand the Dataframe better , lets load Movies data into a Dataframe and explore the methods and attributes avilable with Dataframe .

# ### Import the Pandas package with pd as alias 

# In[ ]:


import pandas as pd


# #### Use the read_csv function to read in the movie dataset
# 
#  read_csv is an important pandas function to read csv files and do operations on it.

# In[ ]:


# read a CSV file into Python and return a DataFrame as output
Movie_DF = pd.read_csv('../input/movie.csv')


# #### Let's check the dataype returned by read_csv() and confirm that indeed the output is of type Dataframe

# In[ ]:


# Check the Data type of the output returned by pd.read_csv()

print(type(Movie_DF))


# #### Now , that we have read the CSV file into a Pandas Dataframe - 
#        a) We check the data for first 5 observations using head() 
#        b) we can also check the data for last 5 observations using tail()

# In[ ]:


# Show the content of first 5 observations in a Dataframe

Movie_DF.head()


# In[ ]:


# Show the content of last 5 observations in a Dataframe

Movie_DF.tail()


# You will observe the following in the above Dataframe content displayed -
#     
#     a) Observation : Each Row of Data is called an Observation 
#     b) INDEX : Provides a label to the Rows of a Dataframe 
#     c) Columns : Each column also called Attribute / Feature contains a homogenous set of data 
#     d) Column Name : Each column is being uniquely identified through a Column Name
# 
# 
# Pandas uses NaN (not a number) to represent missing values. 

# ### Do you Know -
# 
# You can control how many observations to be displayed as part of head() or tail() 

# In[ ]:


# Show the content of first 3 observations in a Dataframe

Movie_DF.head(3)


# ### Accessing the 3 individual components of a DataFrame
# 
#     the index, 
#     columns, 
#     and data
#     
# 
# Each of these components is itself a Python object with its own unique attributes and methods.

# In[ ]:


# Fetch all the Index values from the Dataframe.

Movie_index = Movie_DF.index

print("The values in the index column is : " , Movie_index)

print('\n')

print("The Data Type of the Index is     : " , type(Movie_index))


# In[ ]:


# Fetch all the Column Names from the Dataframe.

Movie_Col_Names = Movie_DF.columns

print("The Column Names are  : " , Movie_Col_Names)

print('\n')

print("The Data Type of the Column Names is     : " , type(Movie_Col_Names))


# In[ ]:


# Fetch all the Values from the Dataframe.

Movie_Values = Movie_DF.values

print("The Values in the Dataframe are  : " , Movie_Values)

print('\n')

print("The Data Type of the Values is     : " , type(Movie_Values))


# **How to check the Structure of a DataFrame **
# 
# Structure of a Dataframe can be enquired / checked using info() method of a Dataframe object

# In[ ]:


# Display the structure of a Dataframe

Movie_DF.info()


# In[ ]:


# Just display the Data types of the columns of a Dataframe

Movie_DF.dtypes


# **We can extract the Count of each DataTypes in a Datframe using get_dtype_counts()**

# In[ ]:


# Display the cummulative Data Type Counts in a Dataframe

Movie_DF.get_dtype_counts()


# **Series**
# 
# Time to explore the Series object in Pandas .
# 
# A Series is a single column of data from a DataFrame. 
# It is a single dimension of data, composed of just an index and the data.

# In[ ]:


# Extract the values in the column 'director_name' from the Movies DataFrame

Movie_DF['director_name']


# In[ ]:


# Extract the values in the column 'director_name' from the Movies DataFrame usinf dot approach
# Note - This approach is not encouraged as it will not work when a column name has blank space

Movie_DF.director_name


# **Now , lets check the Data Type of the values extracted from a single column of a Dataframe **

# In[ ]:


# Display the Data type of a Column extracted from a DataFrame

print(type(Movie_DF['director_name']))


# **Explore the Methods available with SERIES **
# 
# let's select two Series with different data types from the Moview Dataframe. 
# The director_name column contains strings, formally an object data type, and the column actor_1_facebook_likes contains numerical data, formally float64:

# In[ ]:


# Create 2 Series from the Movies Dataframe.

director = Movie_DF['director_name']
actor_1_fb_likes = Movie_DF['actor_1_facebook_likes']

print(type(director))
print(type(actor_1_fb_likes))


# **head() with SERIES object **
# 
# head() method is available with SERIES Dataframe , using which we can fetch specific number of elements from the top ( default is 5 elements )

# In[ ]:


print(director.head())

print('\n')  # Print an Empty Line 

print(actor_1_fb_likes.head(3))


# **value_counts() method for SERIES with String / Object data type**
# 
# This method returns the count of Unique Values in a Series .

# In[ ]:


# Display the Count of Unique values in a SERIES 

print(director.value_counts())


# **using describe() method to extract Statistical summary of a Series of Numerical datatype**
# 

# In[ ]:


# Display the Statistical Summary of a Series of Numerical Data Type

print(actor_1_fb_likes.describe())


# **describe() method shows different outputs in case of Categorical datatype **

# In[ ]:


# Display the Summary of a Series of Categorical Data Type

print(director.describe())


# **Checking whether any value in SERIES is a Missing Value or NaN**
# 
# We can identify if a specific value in a Series is missing by using isnull() method

# In[ ]:


# Check if any value in a Series is Missing 

director.isnull()


# Once , we have identified whether there are Missing Values in a Series or Not , we can address the Missing Values in 2 ways -
# 
#     a) Replace the Missing Values with some Constant or by output of an Expression 
#     b) Drop those elements with Missing values from the Series 

# In[ ]:


# Replace the Missing Values with 0 

director_flled = director.fillna(0)

print("Count of Non Missing Values before applying fillna() :" ,director.count())
print("Count of Non Missing Values after applying fillna()  :" ,director_flled.count())


# In[ ]:


# Replace the Missing Values with 0 

director_NA_Dropped = director.dropna()

print("Count of elements before applying dropna() :" ,director.size)
print("Count of elements after applying dropna()  :" ,director_NA_Dropped.size)


# **Applying 'Operations' on a SERIES **
# 
# Operations applied on a SERIES is Vectorized - meaning an operation applied on a Series , gets applied on each element of the Series.

# In[ ]:


# Applying multiplaction operator on Series containing String data type 
# Resulting in each Value being concatenated to itself

director * 2


# In[ ]:


# Applying Division operator with the Series containing Numeric Data 
# Resulting in each value being divided by 100

actor_1_fb_likes / 100


# ### **Chaining Series methods together**
# 
# In Python, every variable is an object, and all objects have attributes and methods that refer to or return more objects. The sequential invocation of methods using the dot notation is referred to as method chaining. Pandas is a library that lends itself well to method chaining, as many Series and DataFrame methods return more Series and DataFrames, upon which more methods can be called. 

# In[ ]:


# Example of Method Chaining 

director.value_counts().head(3)


# A common way to count the number of missing values is to chain the sum method after isnull

# In[ ]:


# Find the total number of Missing values in a SERIES 

actor_1_fb_likes.isnull().sum()


# ## Index of a DataFrame 
# 
# The index of a DataFrame provides a label for each of the rows. 
# If no index is explicitly provided upon DataFrame creation, then by default, a RangeIndex is created with labels as integers from 0 to n-1, where n is the number of rows.
# 
# There are 2 ways , we can have an explicit Index Label -
#           
#       a) Defining one of the Column as Index during the read_csv() call 
#       b) Setting the Index to a column after read_csv() 

# In[ ]:


# Specify the Index Column during the read_csv() step 

Movie_DF2 = pd.read_csv('../input/movie.csv', 
                        index_col='movie_title')

print(Movie_DF2.head())


# In[ ]:


# Change the Index Column after the read_csv() step 

Movie_DF3 = Movie_DF.set_index('movie_title')

print(Movie_DF3.head())


# Conversely, it is possible to turn the index into a column with the reset_index method. This will make movie_title a column again and revert the index back to a RangeIndex. reset_index always puts the column as the very first one in the DataFrame, so the columns may not be in their original order:

# In[ ]:


Movie_DF4 = Movie_DF3.reset_index()

Movie_DF4.head()


# ### **Adding or Dropping Columns from a DataFrame**

# **Adding new column to a DataFrame**

# In[ ]:


# Count & Display the number of Columns in a Dataframe

Movie_DF.columns.size


# In[ ]:


# Add a New column to a DataFrame

Movie_DF['Tot_FB_Actors_Likes'] = Movie_DF.actor_1_facebook_likes + Movie_DF.actor_2_facebook_likes + Movie_DF.actor_3_facebook_likes


# In[ ]:


# Count & Display the number of Columns in a Dataframe after adding a new column

Movie_DF.columns.size


# In[ ]:


# Drop the new column from the Dataframe

Movie_DF.drop('Tot_FB_Actors_Likes',axis='columns' , inplace=True)


# In[ ]:


# Count & Display the number of Columns in a Dataframe after dropping the new column

Movie_DF.columns.size


# **DataFrame Operations **
# 
# In the earlier section of this script , We had extracted single column from a DataFrame resulting in a SERIES . But , now , lets extract multiple columns from the DataFrame .
# 
# 
# 

# In[ ]:


# Creata a DataFrame which is consisting of subset of the columns from the Original dataFrame

Movie_actor_director = Movie_DF[['actor_1_name', 'actor_2_name','actor_3_name', 'director_name']]

Movie_actor_director.head()


# **Use the select_dtypes method to select only the integer columns:**

# In[ ]:




Movie_DF.get_dtype_counts()


# In[ ]:


# Extract only those columns which are of Integer Data Type

Movie_DF.select_dtypes(include=['int']).head()


# In[ ]:


# Extract only those columns which are of Numeric Data Type

Movie_DF.select_dtypes(include=['number']).head()


# **Extracting Columns where the column name contains specific string **
# 
# Lets extract those columns only which has 'facebook' in the column name

# In[ ]:


# Extract only those columns which has 'facebook' in the Column name

Movie_DF.filter(like='facebook').head()


# ### Methods & Attributes used for DataFrame Objects 

# **Attributes**
# 
#     a) shape : returns the number of rows & columns in a DatFrame
#     b) size  : Returns the total numbers of elements in a Dataframe

# In[ ]:


# get the shape & size of a DataFrame

print(Movie_DF.shape)

print(Movie_DF.size)


# Methods available with DataFrame object -
# 
#     a) count()
#     b) describe()

# In[ ]:


# count method is used to find the number of non-missing values for each column.

print(Movie_DF.count())


# In[ ]:


# Find the Statistical Summary of each column in the DataFrame

print(Movie_DF.describe())


# In[ ]:




