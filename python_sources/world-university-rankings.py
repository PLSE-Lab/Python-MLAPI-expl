#!/usr/bin/env python
# coding: utf-8

# ##### Importing important libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:


sns.set_style('darkgrid')


# ##### Read csv files and display them

# In[ ]:


times_df = pd.read_csv('/kaggle/input/world-university-rankings/timesData.csv' , thousands = ",")


# In[ ]:


shanghai_df = pd.read_csv('/kaggle/input/world-university-rankings/shanghaiData.csv')


# In[ ]:


times_df.head()


# In[ ]:


times_df.describe()


# In[ ]:


shanghai_df.head()


# In[ ]:


shanghai_df.describe()


# ##### Extracting info about both df

# In[ ]:


def extract_info(input_df,name):
    df  = input_df.copy()
    info_df = pd.DataFrame({'nb_rows':df.shape[0], 'nb_columns': df.shape[1], 'name': name}, index = range(1))
    return info_df


# In[ ]:


all_info = pd.concat([times_df.pipe(extract_info,'times'), shanghai_df.pipe(extract_info,'shanghai')])


# In[ ]:


all_info


# Creating a function to clean data before our analysis

# In[ ]:


def clean_world_rank(input_df):
    df = input_df.copy()
    df.world_rank = df.world_rank.str.split('-').str[0].str.split('=').str[0]
    return df


# Combining both df on the basis of common columns

# In[ ]:


common_col = set(shanghai_df.columns) & set(times_df.columns)


# In[ ]:


list(common_col)


# In[ ]:


def filter_year(input_df,years):
    df = input_df.copy()
    return df.query('year in {}'.format(list(years)))


# In[ ]:


common_years = set(times_df.year) & set(shanghai_df.year)


# In[ ]:


clean_times_df = times_df.loc[:,common_col].pipe(filter_year,common_years).pipe(clean_world_rank).assign(name='times')


# In[ ]:


clean_shanghai_df = shanghai_df.loc[:,common_col].pipe(filter_year,common_years).pipe(clean_world_rank).assign(name='shanghai')


# In[ ]:


ranking_df = pd.concat([clean_times_df,clean_shanghai_df])


# In[ ]:


ranking_df


# ###### Further analysis reveals that a lot of entries in 'total_score' are missing so it's better to drop these rows

# In[ ]:


pd.isnull(ranking_df.total_score).sum()/len(ranking_df)


# A lot of data is missing in total_score so it's better to drop this entry.

# In[ ]:


ranking_df.drop('total_score', axis = 1, inplace = True)


# In[ ]:


ranking_df


# In[ ]:


ranking_df.info()


# In[ ]:


ranking_df.info(memory_usage = 'deep')


# In[ ]:


# Cast `world_rank` as type `int16`
ranking_df.world_rank = ranking_df.world_rank.astype('int16')

# Cast `unversity_name` as type `category`
ranking_df.university_name = ranking_df.university_name.astype('category')

# Cast `name` as type `category`
ranking_df.name = ranking_df.name.astype('category')


# In[ ]:


ranking_df.info(memory_usage='deep')


# ##### It is observed that same university has two different names in our datasets Massachusetts Institute of Technology (MIT) & Massachusetts Institute of Technology

# In[ ]:


print(ranking_df.query("university_name == 'Massachusetts Institute of Technology (MIT)'"))


# In[ ]:


ranking_df.loc[lambda df: df.university_name == 'Massachusetts Institute of Technology (MIT)', 'university_name'] = 'Massachusetts Institute of Technology'


# In[ ]:


print(ranking_df.query("university_name == 'Massachusetts Institute of Technology'"))


# # Using Groupby to find top 5 universities (yearwise)

# This line will form a df of all those universities that have been in top5 ranking atleast once

# In[ ]:


top5_df = ranking_df.loc[lambda df : df.world_rank.isin(range(1,6)) , :]
top5_df.head()


# This function will help to find similarity between both times_df and shanghai_df

# In[ ]:


def compute_set_similarity(df):
    pivoted = df.pivot(values = 'world_rank', columns = 'name', index = 'university_name').dropna()
    set_similarity = 100 * len(set(pivoted['shanghai'].index) & set(pivoted['times'].index))/5
    return set_similarity


# Grouping yearwise and finding similarity

# In[ ]:


grouped_df = top5_df.groupby('year')


# In[ ]:


set_similarity_df = pd.DataFrame({'set_similarity' : grouped_df.apply(compute_set_similarity)}).reset_index()


# In[ ]:


set_similarity_df


# # Visualizing data using Matplotlib

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


shanghai_df.plot.scatter('total_score', 'alumni', c='year', colormap='viridis')
plt.show()


# Larger lengths of names are not preffered during visualizations

# In[ ]:


times_df.country = times_df.country.replace('United States of America', 'USA').replace('United Kingdom', 'UK')


# Finding no of entries from each country

# In[ ]:


count_df = times_df['country'].value_counts()[:10]


# In[ ]:


count_df


# In[ ]:


count_df = count_df.reset_index()
count_df


# Rename the columns

# In[ ]:


count_df.columns = ['country', 'count']


# In[ ]:


count_df


# In[ ]:


sns.barplot(x = 'country', y = 'count', data = count_df)
sns.despine()

plt.show()

