#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pandas as pd
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
data.head()


# In[ ]:


data.tail() # shows you the last 5 rows


# In[ ]:


data.columns #gives column names of features


# In[ ]:


data.shape #gives number of rows and columns in a tuble


# In[ ]:


data.info() #data types and memory usage


# In[ ]:


print(data["parental level of education"].value_counts(dropna=False))
#lets look at frequency of parental level of education


# In[ ]:


data.describe()
#to find median,upper quantiel, lower quantile,average


# **VISUAL EXPLORATORY**

# In[ ]:


data.boxplot(column="math score")
plt.show()
# to see median, upper quantile,lower quantile and outliers on plot


# **Tidy Data**

# In[ ]:


data_new=data.head()
data_new


# In[ ]:


melted=pd.melt(frame=data_new,id_vars="parental level of education",value_vars=["math score","writing score"])
melted
#id_vars= what we dont want to melt
#value_vars= what we want to melt


# In[ ]:


data1=data.head()
data2=data.tail()
conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row
# to concatenate two dataframe axis=0: adds dataframe in row


# In[ ]:


data1=data["math score"].head()
data2=data["reading score"].head()
conc_data_col=pd.concat([data1,data2],axis=1)
conc_data_col


# In[ ]:


data.dtypes


# *to convert object to categoriacal,int or float*

# In[ ]:


data["gender"]=data["gender"].astype("category")
data["math score"]=data["math score"].astype("float")


# Now, we will see the differences

# In[ ]:


data.dtypes


# **MISSING DATA AND TESTING WITH ASSERT**

# In[ ]:


#lets look if there is any  NaN value
data.info()


# In my case, there is no NaN values but if you want to drop NaN values in your list. Here is steps how to drop them

# data1=data
# data1["gender"].dropna(inpace=True)
# assert data[gender"].notnull(),all() # returns nothing because we droped NaN values.
