#!/usr/bin/env python
# coding: utf-8

# **Exploratory Analysis of Nobel Laureates Dataset**
# 
# My first Kaggle analysis

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


# In[ ]:


import seaborn.apionly as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


nobeldf = pd.read_csv("../input/archive.csv", parse_dates = ["Birth Date", "Death Date"])


# In[ ]:


nobeldf.info()


# The "Birth Date" didn't get converted to datetime because of "nan"s (str).
# 
# Missing data: "Motivation", "Birth City", "Birth Country", "Sex", "Organization Name", "Organization City", "Organization Country", "Death Date" (but probably some are still living), "Death City", "Death Country"

# In[ ]:


nobeldf.head()


# Number of individuals vs. organization laureates:

# In[ ]:


nobeldf["Laureate Type"].value_counts()


# **How many categories and how are they distributed?**

# In[ ]:


nobeldf.Category.value_counts()


# In[ ]:


sns.countplot(y="Category", data=nobeldf,
              order=nobeldf.Category.value_counts().index,
              palette='GnBu_d')
sns.despine();


# **Males vs. Females**

# In[ ]:


sns.countplot(y="Sex", data=nobeldf, palette='GnBu_d')
sns.despine();


# **Awards by Country**

# In[ ]:


sns.countplot(y="Organization Country", 
              data=nobeldf,
              order=nobeldf["Organization Country"].value_counts().index,
              palette='Blues_r')
sns.set(rc={"figure.figsize": (10, 12)})


# **Age of Nobel Laureates at the Time of Award**
# 
# --> Calculate the age using "Year" and "Birth Date". However, "Birth Date" column is still str and have some missing values.

# In[ ]:


nobeldf["Birth Date"].value_counts().head()


# In[ ]:


nobeldf["Birth Date"] = nobeldf["Birth Date"].replace(to_replace="nan", value=0)


# Create new column "Birth Year":

# In[ ]:


nobeldf["Birth Year"] = nobeldf["Birth Date"].str[0:4]


# In[ ]:


nobeldf["Birth Year"].head()


# In[ ]:


nobeldf["Birth Year"] = pd.to_numeric(nobeldf["Birth Year"])


# In[ ]:


nobeldf["Birth Year"].head()


# New column "Age":

# In[ ]:


nobeldf["Age"] = nobeldf["Year"] - nobeldf["Birth Year"]


# Statistics of ages of Nobel Laureates:

# In[ ]:


nobeldf.Age.describe()


# Distribution of ages:

# In[ ]:


sns.distplot(nobeldf.Age.dropna(), bins=35)
sns.set(rc={"figure.figsize": (10, 5)})


# ## In conclusion, you are most likely to be awarded a Nobel if you are:
# 
# 1. a male
# 2. in the area of medicine
# 3. living in the US
# 4. around 60-65 years old

# In[ ]:




