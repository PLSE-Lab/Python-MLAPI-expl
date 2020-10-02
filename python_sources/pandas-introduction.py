#!/usr/bin/env python
# coding: utf-8

# # Abstract:
# Python Data Analytics will help you tackle the world of data acquisition and analysis using the power of the Python language. At the heart of this book lies the coverage of pandas, an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.This Kernel is to shows the strength of the Python programming language when applied to processing, managing and retrieving information. Inside, you will see how intuitive and flexible it is to discover and communicate meaningful patterns of data using Python scripts, reporting systems, and data export. This Kernel examines how to go about obtaining, processing, storing, managing and analyzing data using the Python programming language.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Loading data

# In[ ]:


#Check what version of Pandas we are using
pd.__version__


# In[ ]:


# load data to a variable called df
'''
the data file is tsv which means a tab delimited file
we are going to use read_csv to load the data and will mention the delimiter as '\t' for tabspace.
'''
df = pd.read_csv('../input/gapminder.tsv', delimiter='\t')


# In[ ]:


# view the first 5 rows
df.head()


# In[ ]:


# what is this df object?
type(df)


# In[ ]:


# get num rows and columns
df.shape


# In[ ]:


# look at the data - summarized information
df.info()


# # Subsetting data
# ## Columns

# In[ ]:


# subset a single column
country_df = df['country']
country_df


# In[ ]:


# the output above is same as
df.country


# In[ ]:


#you can check the top 5 values of one column too...
df.country.head()


# In[ ]:


# subset multiple columns
df[['country', 'continent', 'year']].head()


# In[ ]:


# delete columns
# this will drop a column in-place
df_new = df.copy()
df_new


# In[ ]:


del df_new['country']  # del df_new.country won't work
df_new


# In[ ]:


# this won't unless you use the inplace parameter
df_new.drop('continent', axis=1)


# In[ ]:


#continent is still in...
df_new.head()


# In[ ]:


# df is unchanged
df.head()


# **Very Important**
# - We have copied the df to df_new using .copy() method
# - if we say df_new = df this is actually passing the object and changes on df_new will impact on df

# # Passing df to df_new

# In[ ]:


#display df
df.head()


# In[ ]:


#pass df to df_new
df_new = df
df_new.head()


# In[ ]:


len(df),len(df_new)


# In[ ]:


del df_new['country']


# In[ ]:


df_new.head()


# In[ ]:


df.head()


# In[ ]:


df.columns,df_new.columns


# # Rows

# In[ ]:


df = pd.read_csv('../input/gapminder.tsv', delimiter='\t')


# In[ ]:


# first row
df.loc[0]


# In[ ]:


# 100th row
df.loc[99]


# In[ ]:


# this will fail
df.loc[-1]


# # iloc

# In[ ]:


df.iloc[0]


# In[ ]:


df.iloc[99]


# In[ ]:


#this will work
df.iloc[-1]


# # ix
# .... to be continued

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# **References**
# 1. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1145460.svg)](https://doi.org/10.5281/zenodo.1145460)
# Swaroop Kallakuri. (2018, January 10). ksjpswaroop/Learn2Code: Learn2Code in Python 3 (Version V1). Zenodo. http://doi.org/10.5281/zenodo.1145460

# In[ ]:




