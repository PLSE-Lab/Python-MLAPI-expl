#!/usr/bin/env python
# coding: utf-8

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


# # Import necessary modules

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing the dataset

# In[ ]:


school = pd.read_csv('../input/Public_Schools.csv')
school.head(12)


# In[ ]:


# shape of the dataset
school.shape


# # Inspecting the data

# In[ ]:


school.info()


# In[ ]:


school.describe(include='all')


# In[ ]:


# How many schools in 'East Boston' city
east = school[school['CITY'] == 'East Boston']['SCH_NAME'].count()
east


# In[ ]:


# How many schools in 'Boston' city
boston = school[school['CITY'] == 'Boston']['SCH_NAME'].count()
boston


# In[ ]:


# Top 5 schools in 'Boston'
top_5 = school[school['CITY'] == 'Boston']['SCH_NAME'].head()
top_5


# # Visualizing the data

# In[ ]:


sns.pairplot(school)


# In[ ]:


plt.figure(figsize=(10, 8))
sns.boxplot(x='CITY', y='SCH_ID', data=school)
plt.tight_layout()


# In[ ]:




