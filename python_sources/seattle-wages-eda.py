#!/usr/bin/env python
# coding: utf-8

# # EDA For Seattle Wage Dataset
# Work in progress

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 10)  # Show more columns

from matplotlib import pyplot as plt  # plotting
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Overall Wage Data

# In[ ]:


wages = pd.read_csv('../input/city-of-seattle-wage-data.csv')
wages.head()


# In[ ]:


# Average rate by department
wages.groupby('Department').mean()


# In[ ]:


f, ax = plt.subplots(figsize=(7, 10))
sns.boxplot(x='Hourly Rate ', y='Department', data=wages)
plt.title('Distribution of Hourly Rates');


# In[ ]:


# Taking a look at just how many entries are present for each department
wages['Department'].value_counts()


# It seems like there's a number of departments with very few entries. It may not be meaningful to extract trends from them, so I'll exclude departments with fewer than 50 data points.

# In[ ]:


# Convert to dictionary so that I can make a list comprehension easily.
departments = dict(wages['Department'].value_counts())
# Add a department to large_dept if the size of the department is over 50
large_dept = [department for department in departments.keys() if departments[department] > 50]


# In[ ]:


# construct a mask using nested list comprehension
mask = [True if department in large_dept else False for department in wages['Department']]
filtered_wages = wages[mask]
# Filter based upon mask; success
filtered_wages['Department'].value_counts()


# In[ ]:


# Redo box plot
f, ax = plt.subplots(figsize=(7, 7))
sns.boxplot(x='Hourly Rate ', y='Department', data=filtered_wages)
plt.title('Distribution of Hourly Rates for Departments >50 Employees');


# Some of these distributions seem particularly skewed, with a lot of outliers, such as City Light and Parks.

# In[ ]:


light = filtered_wages[filtered_wages['Department'] == 'City Light']
sns.distplot(light['Hourly Rate '])
plt.title('Histogram of City Light Department Wages')
plt.ylabel('Distribution');


# In[ ]:


# Let's see who the top and bottom earners are
print('Top Five Earners in Light Department')
print(light[['Job Title', 'Hourly Rate ']].sort_values('Hourly Rate ', ascending=False).head())
print('\n\nBottom Five Earners in Light Department')
print(light[['Job Title', 'Hourly Rate ']].sort_values('Hourly Rate ', ascending=True).head())


# In[ ]:


# How about the Parks Department?
parks = filtered_wages[filtered_wages['Department'] == 'Parks Department']
sns.distplot(parks['Hourly Rate '])
plt.title('Histogram of Parks Department Wages')
plt.ylabel('Distribution');


# It seems these skewed departments have a very large base of interns and other near-minimum wage earners, and a handful of high executives.

# In[ ]:


# In contrast, the IT department has a much more normal distribution
it = filtered_wages[filtered_wages['Department'] == 'Seattle Information Technology']
sns.distplot(it['Hourly Rate '])
plt.title('Histogram of IT Department Wages')
plt.ylabel('Distribution');


# In[ ]:




