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


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("/kaggle/input/montcoalert/911.csv")
df.head()


# In[ ]:


df.info()


# # What are the top five zip codes for 911 calls?

# In[ ]:


df['zip'].value_counts().head(5)


# # What are the top five townships for 911 calls?

# In[ ]:


df['twp'].value_counts().head(5)


# # Take a look at the title column, how many unique title codes are there?

# In[ ]:


df["title"].nunique()


# # Creating new features

# ## ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.**

# ### *For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. *

# In[ ]:


df["Reason"] = df["title"].apply(lambda title: title.split(':')[0])
df['Reason']


# # ** What is the most common Reason for a 911 call based off of this new column? **

# In[ ]:


df['Reason'].value_counts().head(3)


# # ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[ ]:


sns.countplot(x="Reason", data=df, orient = 'h', palette = "husl")


# #  Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column?

# In[ ]:


type(df['timeStamp'].iloc[0])


# In[ ]:




