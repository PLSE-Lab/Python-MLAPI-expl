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


#     Let's get a quick look at the organization of the data

# In[ ]:


df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
df.head()


# Select US data from the remainder

# In[ ]:


df.rename(columns = {'suicides/100k pop': 'per100k', 'country-year': 'country_year'}, inplace = True)
df


# In[ ]:


country_number = df.country.nunique()
print(country_number)
df_US = df[(df.country == 'United States')]
df_US


# How to select data for a year

# In[ ]:


df_US[(df_US.year == 1995)]


# Want a list of total suicides and total population by year for US

# In[ ]:


total_suicides = np.sum(df_US[(df_US.year == 1995)])
total_suicides


# Apparently variable names can't start with a number

# In[ ]:


suicides_1995 = df_US[(df_US.year == 1995)]
suicides_1995


# In[ ]:


total_suicide_1995 = suicides_1995['suicides_no']
total_suicide_1995


# In[ ]:


np.sum(total_suicide_1995)


# compile a table of total suicides and year for US

# In[ ]:


total_US = df_US.groupby('year').suicides_no.sum()
total_US


# The total number of suicides appears to have grown by about 50% let's look at percapita

# In[ ]:


df_US_100k = df_US.groupby('year').per100k.sum()
df_US_100k


# Accumulate aggregates by country and year for this database

# In[ ]:


df_country_year = df.groupby(['country', 'year']).agg({'suicides_no':sum, 'per100k':sum})
df_country_year

