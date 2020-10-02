#!/usr/bin/env python
# coding: utf-8

# # Basics of Jupyter
# 
# Jupyter is a wonderful online tool for doing data science with Python.  The Kaggle installation of Jupyter already comes with some remarkable add-on libraries of exploring and visualizing data.  

# ## Two kinds of blocks
# You can use markdown to build nicely formatted text. You can also write code.+

# In[ ]:


#upper = int(input("upper limit? "))
upper = 5
for i in range(upper):
    print(i)


# ## Importing your libraries
# Almost every Python-based data analysis project will begin with three libraries:
# * numpy adds basic linear algebra functionality (matrices and vectors) Mostly this is hidden from the programmer
# * pandas is a dataframe layer.  It massively simplifies working with data sets in a way that makes analysis easier
# * matplotlib is a graphing library very similar to matlab.  It provides high-powerd graphing features
# Note that some of these imports are built into the default Kaggle page, but you may need to add your own imports

# In[ ]:


#loading in standard libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting library


# ## Viewing the files
# When you open a kernal on kaggle, you will usually already have some data files loaded into your workspace. You can also upload your own data files.  Of course, you might want to know exactly which files you have access to, so this code is in every default kaggle page.  It gives a list of all the known data files.

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print("files available: ")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ## Working with a basic numpy data set
# Before working with files, you can create your own data set.  Any 2D data structure in Python can be a basis for a numpy data frame. In this case we create a list of tuples.
# Once you create a dataframe, simply name it to see the frame printed on the page in a nice format

# In[ ]:


#create a data file to play with
names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']
births = [968, 155, 77, 578, 973]

BabyDataSet = list(zip(names, births))
print(BabyDataSet)

df = pd.DataFrame(BabyDataSet, columns = ("Names", "Count"))
df


# ## Adding a basic plot
# 
# Pandas uses the matplotlib library to give easy access to multiple types of plots.  Note that a data set usually uses the line number as an index, so if you want the graph to look better, you'll need to indicate which column is meant to be the index.

# In[ ]:


df = df.set_index("Names")
df.plot(kind = "bar")


# ## Making a sqlite connection
# Sometimes datasets are in an SQL file.  Of course, if you already know SQL, you are well on your way.  You can use the ordinary Python mechanism to make a connection to an SQL database in your file system:

# In[ ]:


#read from sqlite database
import sqlite3
conn = sqlite3.connect("../input/database.sqlite")


# ## Get table names
# Of course, if you did not create the database yourself, you need to do some analysis to determine which tables are there. You can actually make a query against the database to find this information.

# In[ ]:


#get table names
c = conn.cursor()
result = c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
for row in result:
    print (row[0])


# ## Get column names for a table
# Likewise, once you are in a table, you'll need to know the column names.  There are a couple of ways to do this, but this is the easiest:

# In[ ]:


# get column names for indicators table
df = pd.read_sql_query("SELECT * FROM Indicators LIMIT 1", conn)
print(df.columns.values)


# ## View the dataframe
# Use the pd.read_sql_query() method to read the results into a dataframe. You can then print the dataframe out to see its values.

# In[ ]:


df = pd.read_sql_query("SELECT * FROM Indicators LIMIT 20", conn)
df


# ## Open SQLite query in pandas
# Once you have a rough view of the structure of the data, you'll no doubt want to refine the data to get specific information. You can use any SQL query to get exactly the dataframe you want.

# In[ ]:


#opening file in pandas
query = """
SELECT
  Year as 'Year',
  Value as 'Military exports'
FROM 
  indicators
WHERE
  indicatorCode= 'MS.MIL.XPRT.KD'
AND 
  CountryCode = "ARB"
"""
df = pd.read_sql_query(query, conn)
df


# ## plot military exports
# Now that we have a more structured dataframe, we can plot it out to see what it might be telling us.

# In[ ]:


df = df.set_index("Year")
df.plot(figsize = (12,8), kind = "bar")


# ## Data can also come from a csv file
# Many datasets are delivered in csv files.  They are more portable than sqlite, but rather than using SQL syntax to divine information, you'll use functions of the pandas dataframe to discern meaning.  
# The loc method takes a list containing two values.  The first is a slice of indices, and the second is a list containing the required column names.

# In[ ]:


df = pd.read_csv("../input/Country.csv")
#select name of element 10
print(df.loc[10,['LongName']])


# select countrycode and shortname from first ten elements
countryName =  df.loc[0:10,['CountryCode', 'ShortName']]
countryName


# ## More Interesting queries
# You can apply a boolean expression to extract subsets of a dataframe.  The head() method just returns the first few values so you don't have to look at all the data when you're just getting a feel for it.

# In[ ]:


df = pd.read_csv("../input/Indicators.csv")

#get the results from Afghanistan
afg = df[df.CountryCode == "AFG"]
afg.head()


# ## Subset of a subset
# It would be tempting to continue whittling down the data by making successively smaller dataframes, but this causes an indexing error (though it does return the expected results.)

# In[ ]:


#get members of afg dataset from 1960
afg60 = afg[df.Year == 1960]
afg60


# In[ ]:


#remove index issue by combining with boolean operators
#get all population values for Afghanistan
afgPop = df[(df.CountryCode == "AFG") & (df.IndicatorCode == "SP.POP.TOTL")]
afgPop


# ## Filter columns and plot the data
# The afgPop dataframe is great, but it includes a lot of extraneous data.  All we really want is the year and value.  You can whittle the contents down with a nested list, then set an index and draw a plot.

# In[ ]:


afgPop = afgPop[["Year", "Value"]]
afgPop = afgPop.set_index("Year")
afgPop.plot()


# 
