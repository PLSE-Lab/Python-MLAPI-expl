#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/world-happiness/2019.csv")
df.head()


# In[ ]:


df.tail()


# In[ ]:


df.columns


# In[ ]:


df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
df.columns


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


# For example lets look frequency of pokemom types
# print(df['country_of_region'].value_counts(dropna =False))  # if there are nan values that also be counted
#in this dataset every columns has unique value. For this reason value_counts function doesn't do anything here.


# In[ ]:


df.describe()


# In[ ]:


t = df.score.mean()
df["score_level"] = ["high" if i > t else "low" for i in df.score]
df.boxplot(column='gdp_per_capita',by = 'score_level')


# In[ ]:


# Firstly I create new data from pokemons data to explain melt nore easily.
data_new = df.head()
data_new


# TIDY DATA
# We tidy data with melt(). Describing melt is confusing. Therefore lets make example to understand it.

# In[ ]:


# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'country_or_region', value_vars= ['score','gdp_per_capita'])
melted


# In[ ]:


# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'country_or_region', columns = 'variable',values='value')


# PIVOTING DATA
# Reverse of melting.

# In[ ]:


# Firstly lets create 2 data frame
df1 = df.head()
df2= df.tail()
conc_df_row = pd.concat([df1,df2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_df_row


# CONCATENATING DATA
# We can concatenate two dataframe.

# In[ ]:


df1 = df['score'].head()
df2= df['gdp_per_capita'].head()
conc_df_col = pd.concat([df1,df2],axis =1) # axis = 0 : adds dataframes in row
conc_df_col


# In[ ]:


df.dtypes


# 
# DATA TYPES
# There are 5 basic data types: object(string),booleab, integer, float and categorical.
# We can make conversion data types like from str to categorical or from int to float
# Why is category important:
# 
# make dataframe smaller in memory
# can be utilized for anlaysis especially for sklearn.

# In[ ]:


# lets convert object(str) to categorical and int to float.
df['score'] = df['score'].astype('category')
df['gdp_per_capita'] = df['overall_rank'].astype('float')
df.dtypes


# In[ ]:


# Lets look at does data have nan value
# As you can see there are 156 entries. There is no null object.
df.info()


# MISSING DATA and TESTING WITH ASSERT
# If we encounter with missing data, what we can do:
# 
# leave as is
# drop them with dropna()
# fill missing value with fillna()
# fill missing values with test statistics like mean
# Assert statement: check that you can turn on or turn off when you are done with your testing of the program.

# In[ ]:


df["score"].value_counts(dropna =False)


# In[ ]:


# droping nan values
df1=df   # also we will use data to fill missing value so I assign it to data1 variable
df1["score"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# If we have NaN values in this dataset this code will work.  


# In[ ]:


#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true


# In[ ]:


# In order to run all code, we need to make this line comment
assert 1==2 # return error because it is false


# In[ ]:


assert  df['score'].notnull().all() # returns nothing because we don't have nan values


# In[ ]:


# # With assert statement we can check a lot of thing. For example
# assert data.columns[1] == 'Name'
# assert data.Speed.dtypes == np.int


# This kernel prepared as a homework with the help of DATAI TEAM Data Science Tutorial documents.
# https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners
# 
