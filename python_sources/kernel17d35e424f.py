#!/usr/bin/env python
# coding: utf-8

# 1.  Reading Data

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


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# * Reading csv file

# In[ ]:


myfile=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
myfile


# In[ ]:


myfile.shape


# In[ ]:


myfile.dtypes


# Reviews, Price are given as object data types .They should be converted to numeric.

# In[ ]:


myfile.describe()


# In[ ]:


myfile.boxplot()


# * I see an outlier here

# In[ ]:


myfile.hist()


# In[ ]:


myfile.info()


# There are some null values in myfile

# * Data cleaning

# In[ ]:


myfile[myfile.Rating>5]


# * This is an outlier

# In[ ]:


myfile.drop([10472],inplace=True)
myfile[10470:10476]


# In[ ]:


myfile.hist()


# * It is right skewed. so NA should be replaced by median

# * Data manipulation

# In[ ]:


myfile.Rating=myfile.Rating.fillna(myfile.Rating.median())


# In[ ]:


myfile.isnull().sum()


# Table shows that there are 0 nulls in Rating.which means nulls ar4 replaced by median 
# And, There are nulls in type,Current Ver, Android Ver
# As they are object types, nulls in these columns shpould be replaced by mode

# In[ ]:


myfile.Type= myfile.Type.fillna(myfile.Type.mode().values[0])
myfile['Current Ver']= myfile['Current Ver'].fillna(myfile['Current Ver'].mode().values[0])
myfile['Android Ver']= myfile['Current Ver'].fillna(myfile['Android Ver'].mode().values[0])


# In[ ]:


myfile.isnull().sum()


# null values in 'Current ver','android Ver' are replaced by their respective modes

# In[ ]:


myfile.dtypes


# In[ ]:


myfile.Reviews= pd.to_numeric(myfile.Reviews,errors='coerce')


# Reviews column is converted from object tyoe to numeric

# In[ ]:


myfile.Installs=myfile.Installs.apply(lambda x: str(x).replace('+','') if '+' in str(x) else str(x))
myfile.Installs=myfile.Installs.apply(lambda x: str(x).replace(',','') if ',' in str(x) else str(x))
myfile.Installs=pd.to_numeric(myfile.Installs)


# alphanumeric charecters are removed from installs column

# In[ ]:


myfile.Price=myfile.Price.apply(lambda x: str(x).replace('$','') if '$' in str(x) else str(x))
myfile.Price=myfile.Price.astype(float)


# alphanumeric charecters are removed from price column

# In[ ]:


myfile.dtypes


# In[ ]:


myfile.describe()


# * Data visualization

# In[ ]:


grp_myfile=myfile.groupby('Category')
a=grp_myfile.Rating.mean()
b=grp_myfile.Reviews.mean()
c=grp_myfile.Price.mean()
print(a)
print(b)
print(c)




# In[ ]:


plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.plot(a)


# In[ ]:


plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.plot(b)


# In[ ]:


plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.plot(c)

