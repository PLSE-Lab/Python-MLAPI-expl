#!/usr/bin/env python
# coding: utf-8

# # Pandas tutorial : Day 1 
# Here's what we are going to do today:
# 
# * [What is pandas?](#1)
# * [Get our enviornment setup](#2)
# * [Pandas Data stucture](#3)
# * [Import data](#4)
# * [Exporting data](#5)
# * [Creating test Dataframe](#6)

# ## What is pandas?<a id='1'></a>
# **pandas** is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
# 
# **pandas** is a NumFOCUS sponsored project. This will help ensure the success of development of pandas as a world-class open-source project, and makes it possible to donate to the project.

# ## Get our environment setup<a id='2'></a>

# In[ ]:


# importing useful libraries
import pandas as pd # data processing
import numpy as np
import os


# ## Pandas Data Structure<a id='3'></a>
# Pandas has two types of data-structures.
# 1. Series
# 1. DataFrame

# ### Series 
# 1D labeled array. It can accommodate any type of data in it.

# In[ ]:


mySeries = pd.Series([3, -5, 7, 4], index = ['a', 'b', 'c', 'd'])
print(mySeries)
print(type(mySeries))


# ### DataFrame
# 2D data structure. It contains rows and columns

# In[ ]:


data = {'Country' : ['Belgium', 'India', 'Brazil'],
       'Capital' : ['Brussels', 'New Delhi', 'Brassilia'],
       'Population' : [12345,  123456, 98745]}

df = pd.DataFrame(data, columns = ['Country', 'Capital', 'Population'])
print(df)
print(type(data))
print(type(df))


# ## Import Data<a id='4'></a>
# Data scientists are expected to build high-performing machine learning models, but the starting point is getting the data into the Python environment. Only after importing the data can the data scientist clean, wrangle, visualize, and build predictive models on it.
# 
# In this guide, you'll learn the techniques to import data into Python. 
# 
# ### Import CSV files
# It is important to note that a singlebackslash does not work when specifying the file path. You need to either change it to forward slash or add one more backslash like below
# * import pandas as pd
# * mydata= pd.read_csv("C:\\Users\\Deepanshu\\Documents\\file1.csv")
# 
# ### Import File from URL
# You don't need to perform additional steps to fetch data from URL. Simply put URL in read_csv() function (applicable only for CSV files stored in URL).
# * mydata = pd.read_csv("http://winterolympicsmedals.com/medals.csv")
# 
# 
# ### Read Text File
# We can use read_table() function to pull data from text file. We can also use read_csv() with sep= "\t" to read data from tab-separated file.
# * mydata = pd.read_table("C:\\Users\\jasprit\\Desktop\\example2.txt")
# * mydata = pd.read_csv("C:\\Users\\jasprit\\Desktop\\example2.txt", sep ="\t")
# 
# ### Read Excel File
# The read_excel() function can be used to import excel data into Python.
# * mydata = pd.read_excel(" https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls ", sheetname="Data 1", skiprows=2)
# 
# If you do not specify name of sheet in sheetname= option, it would take by default first sheet.

# ## Exporting Data<a id='5'></a>
# This is used to save the output/dataframe in the format you want.
# * df.to_csv(filename) -> Writes to a CSV file
# * df.to_excel(filename) -> Writes on an Excel file
# * df.to_sql(table_name, connection_object) -> Writes to a SQL table
# * df.to_json(filename) -> Writes to a file in JSON format
# * df.to_html(filename) -> Saves as an HTML table
# * df.to_clipboard() -> Writes to the clipboard

# ## Creating test Dataframe<a id='6'></a>

# In[ ]:


# Let's make a dataframe of 5 columns and 20 rows
pd.DataFrame(np.random.rand(20, 5))


# Hurey!!! We are done with some basics of pandas, now we will move to summaring data.
