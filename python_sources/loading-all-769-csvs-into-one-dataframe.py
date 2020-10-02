#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import glob
import os


# I got a little overwhlemed with the number of CSV files in this dataset. This kernel is going to show you how to load all the CSV files into one dataframe. The tricky part is that the filenames are the stock tickers, so I will also addadding that as a new column.

# In[ ]:


# See how many files there are in the directory. 
# "!" commands are called "magic commands" and allow you to use bash
file_dir = '../input/kospi'
get_ipython().system('ls $file_dir')


# Holy crap there are a lot of files...

# In[ ]:


# Number of files we are dealing with
get_ipython().system('ls $file_dir | wc -l')


# In[ ]:


# Get a python list of csv files
files = glob.glob(os.path.join(file_dir, "*.csv"))


# In[ ]:


# Look at a few to see how we can merge them
df1 = pd.read_csv(files[0])
df2 = pd.read_csv(files[1])
df3 = pd.read_csv(files[2])

print(df1.head(), "\n")
print(df2.head(), "\n")
print(df3.head(), "\n")


# These files have the same columns so it seems reasonable to concatenate everything into one dataframe. However, I want to keep track of the file names because that's the only reference to the stock tickers. 
# <br><BR>
# 
# - First, create a list of dataframes with the filenames in a "stock_ticker" column 
# - Then concatenate them all into one **big ass dataframe**

# In[ ]:


# Make a list of dataframes while adding a stick_ticker column
dataframes = [pd.read_csv(file).assign(stock_ticker=os.path.basename(file).strip(".csv")) for file in files]
# Concatenate all the dataframes into one
df = pd.concat(dataframes, ignore_index=True)


# In[ ]:


df.head()


# In[ ]:


df.shape

