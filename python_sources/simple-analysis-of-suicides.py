#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('figure', figsize = (15, 10))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Reading the dataset file
df = pd.read_csv("../input/master.csv")


# In[ ]:


# Displaying the first 5 rows of the DataFrame
df.head()


# In[ ]:


# Data type in each column
df.dtypes


# In[ ]:


# Summary of information in all columns
df.describe().round(2)


# In[ ]:


# Number of rows and columns in the DataFrame
print('Number of rows:', df.shape[0])
print('Number of columns:', df.shape[1])


# In[ ]:


# Verify the country that the highest GDP per Capita
df[df['gdp_per_capita ($)'] == 126352.000000]


# In[ ]:


print("Luxembourg has the highest GDP per capita")


# <h2>Analysing the country with the highest GDP per capita - Luxembourg</h2>

# In[ ]:


df_luxembourg = df.query("country == 'Luxembourg'")


# In[ ]:


df_luxembourg.head(10)


# <h2>Number of suicides per year in Luxembourg</h2>

# In[ ]:


year_sum_luxembourg = df_luxembourg.groupby('year').sum()['suicides_no'].sort_values(ascending=False).reset_index()
figure = sns.barplot(x = 'year', y = 'suicides_no', data = year_sum_luxembourg, palette="BuGn_r", order=year_sum_luxembourg['year'])
figure.set_title('Number of suicides per year in Luxembourg', {'fontsize': 22})
figure.set_xlabel('Year', {'fontsize': 18})
figure.set_ylabel('Total', {'fontsize': 18})
plt.rcParams["xtick.labelsize"] = 10
plt.xticks(rotation= 90)


# <h2>Analysing the total of suicides per country (1985 - 2016)</h2>

# In[ ]:


new_df = pd.DataFrame(df.groupby('country').sum()['suicides_no'].sort_values(ascending=False).reset_index())
analysing_total = new_df.head(10)


# In[ ]:


figure = sns.barplot(x = 'country', y = 'suicides_no', data = analysing_total, palette="GnBu_d")
figure.set_title('Total of the suicides between 1985-2016', {'fontsize': 22})
figure.set_xlabel('Country', {'fontsize': 18})
figure.set_ylabel('Total', {'fontsize': 18})
plt.rcParams["xtick.labelsize"] = 3
plt.xticks(rotation= 90)


# <h2>Analysing of suicies in New Zealand and Australia (1985 - 2016)</h2>

# In[ ]:


countries_oceania = ['New Zealand', 'Australia']
df_ne = df[df['country'].isin(countries_oceania)]
ax = df_ne.groupby(['country', 'year'])['suicides/100k pop'].sum().unstack('country').plot(figsize=(10, 10))
ax.set_title('Suicides in Oceania', fontsize=20)
ax.legend(fontsize=15)
ax.set_xlabel('Year', fontsize=20)
ax.set_ylabel('Suicides Number', fontsize=20)
ax


#  <h2>Suicides per 100k population</h2>

# In[ ]:


countries_oceania = ['New Zealand', 'Australia']
for country in countries_oceania:
    grouped = df[df['country'] == country].groupby(['year', 'age'])['suicides/100k pop'].sum().unstack('age')
    grouped.plot(figsize=(10, 10),
               title='Suicides per 100k population by age in ' + country,
               legend=True)


# In[ ]:


new_zealand_analysis = df[df['country'] == 'New Zealand'].groupby(['year', 'age'])['suicides/100k pop'].sum().unstack('age')


# In[ ]:


new_zealand_analysis.plot(figsize=(10, 10),
               title='Suicides per 100k population by age in New Zealand',
               legend=True)


# Here's a [BBC](https://www.bbc.com/news/world-asia-40284130) article talking a bit more than it may be behind the high number of suicides in New Zealand.

# In[ ]:




