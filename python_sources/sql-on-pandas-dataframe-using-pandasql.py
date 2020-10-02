#!/usr/bin/env python
# coding: utf-8

# In this kernel i am going to explain about a Python package **"pandasql"**.  This python package allow you to write SQL querry on Pandas DataFrame. Sometime we do not know that, what will be appropriate api to perform some explicit tasks using Pandas on DataFrames. The person who knows SQL they can apply SQL querries to perform even very complex task using SQL. 

# Using pandasql package we are going to perform following tasks.
# - Column Selection 
# - Data Filtering 
# - Data aggregation 
# - Data Joining 

# **Important Component of package pandasql**
# 
# - load_births  : Load data set births
# - load_meat : Load DataSet meat
# - sqldf : A function which can be use to run SQL on pandas dataframes.

# In[ ]:


## Importing pandasql 
import pandasql as psql


# In[ ]:


##  Birth Data Set 
birth = psql.load_births()
birth.head()


# In[ ]:


##  Meat Data Set 
meat = psql.load_meat()
meat.head()


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# To run SQL on DataFrame, I am using Toy data set. But, first of all let me explain about sqldf() method.
# 
# ### Method sqldf()
# - Run sql on dataframes.
# - I am just going to discuss about its one input argument that is sql querry as String.

# In[ ]:


toydf = pd.read_csv('/kaggle/input/toy-dataset/toy_dataset.csv',header=0)


# In[ ]:


toydf.head()


# In[ ]:


toydf.City.unique()


# In[ ]:


toydf.Age.describe()


# In[ ]:


toydf.Income.describe()


# In[ ]:


toydf.Illness.unique()


# ### Data Selection

# In[ ]:


sdf =  psql.sqldf("select Age, Income from toydf")
sdf.head()


# In[ ]:


sdfc =  psql.sqldf("select Age, Income, City from toydf")
sdfc.head()


# In[ ]:


sdfc1 =  psql.sqldf("select Age, Income, City from toydf limit 5")
sdfc1.head()


# ### Data Filtering : Wee need all the Rows where City is Dallas

# In[ ]:


dallasDf = psql.sqldf("select * from toydf where City ='Dallas'")


# In[ ]:


dallasDf.head()


# In[ ]:


dallasDf1 = psql.sqldf("select * from toydf where City ='Dallas' limit 5")


# In[ ]:


dallasDf1


# In[ ]:


dallasDf1 = psql.sqldf("select * from toydf where City ='Dallas' and Age > 30 limit 5")


# In[ ]:


dallasDf1


# ### Data Aggregation 

# In[ ]:


cdf = psql.sqldf("select count(*), City from toydf group by City")


# In[ ]:


cdf.head()


# In[ ]:


## This query will return average age City wise
cdf1 = psql.sqldf("select avg(Age), City from toydf group by City")
cdf1.head()


# In[ ]:


## This query will return average age and Income grouped on City and Gender
cdf2 = psql.sqldf("select avg(Age),avg(Income), City, Gender from toydf group by City, Gender")
cdf2.head()


# Hope you have enjoyed this kernel. If enjoy do upvote it.

# # Inner Join

# In[ ]:


importDf = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv",header=0)
exportDf = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv",header=0)


# In[ ]:


importDf.head()


# In[ ]:


exportDf.head()


# In[ ]:


importDf.HSCode.describe()


# In[ ]:


exportDf1 = psql.sqldf("select * from exportDf where HSCode <30")
importDf1 = psql.sqldf("select * from importDf where HSCode <30")


# In[ ]:


joinedData = psql.sqldf("select i.HSCode, i.value as importVal,e.value as exportVal, i.year from  importDf1 i inner join exportDf1 e on i.HSCode = e.HSCode; ")


# In[ ]:


joinedData.head(500)

