#!/usr/bin/env python
# coding: utf-8

# ![FIFA](https://statics.sportskeeda.com/wp-content/uploads/2014/04/fifa-logo-design-history-and-evolution-wkuq7omm-2161994.jpg)
# 
# 
# **Lets get started by  loading  all the neccesary datasets and Libraries.**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print("All Packages Loaded Successfully")


# **FIFA Ranking Data Set**
# 
# 
# *First Step*
# 1. Check the data.
# 2. Study the data and choose the columns for exploratory data analysis.
# 3. Check for amibigious names in each column.

# In[ ]:


sample = pd.read_csv('../input/fifa-international-soccer-mens-ranking-1993now/fifa_ranking.csv')


# In[ ]:


sample


# In[ ]:



sample.loc[sample['country_full'] == 'IR Iran']


# **Things To Note:**
# 
# 1. Load the Table in DataFrame, the data is inside : /input/fifa-international-soccer-mens-ranking-1993now/
# 2. In Pandas ,Indexing helps in the following way:
#     * Identifies data (i.e. provides metadata) using known indicators, important for analysis, visualization, and interactive console display.
#     * Enables automatic and explicit data alignment.
#     * Allows intuitive getting and setting of subsets of the data set.
# 3. [Weighted Points](https://www.wikihow.com/Calculate-Weighted-Average)
# 4. Convert the rank_date into a time stamp format

# In[ ]:


table_positions = pd.read_csv('../input/fifa-international-soccer-mens-ranking-1993now/fifa_ranking.csv')
table_positions = table_positions.loc[:,['rank', 'country_full', 'country_abrv', 'cur_year_avg_weighted', 'rank_date', 
                           'two_year_ago_weighted', 'three_year_ago_weighted']]
table_positions.country_full.replace("^IR Iran*", "Iran", regex=True, inplace=True)
table_positions['weighted_points'] =  table_positions['cur_year_avg_weighted'] + table_positions['two_year_ago_weighted'] + table_positions['three_year_ago_weighted']
table_positions['rank_date'] = pd.to_datetime(table_positions['rank_date'])


# In[ ]:


table_positions.head()


# **International Football result Data Set**
# 
# *First Steps*
# 1. Check the data.
# 2. Study the data and choose the columns for exploratory data analysis.
# 3. Check for amibigious names in each column.

# In[ ]:


matches = pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")


# **Things To Note:**
# 
# 1. Load the Table in DataFrame, the data is inside : /input/international-football-results-from-1872-to-2017/
# 2. In Pandas ,Indexing helps in the following way:
#     * Identifies data (i.e. provides metadata) using known indicators, important for analysis, visualization, and interactive console display.
#     * Enables automatic and explicit data alignment.
#     * Allows intuitive getting and setting of subsets of the data set.
# 3. Convert the rank_date into a time stamp format
# 4. Change the ambigious names in order to maintain uniformity in the 

# In[ ]:


matches =  matches.replace({'Germany DR': 'Germany', 'China': 'China PR'})
matches['date'] = pd.to_datetime(matches['date'])
matches.head()


# **World Cup 2018 Data Set**
# **Things To Note:**
# 
# 1. Load the Table in DataFrame, the data is inside : /input/fifa-worldcup-2018-dataset/
# 2. In Pandas ,Indexing helps in the following way:
#     * Identifies data (i.e. provides metadata) using known indicators, important for analysis, visualization, and interactive console display.
#     * Enables automatic and explicit data alignment.
#     * Allows intuitive getting and setting of subsets of the data set.
# 3. Drop all the NA values in order to keep the data uniform 
# 4. Change the ambigious names in order to maintain uniformity in the 
# 
# 

# In[ ]:


world_cup_data = pd.read_csv("../input/fifa-worldcup-2018-dataset/World Cup 2018 Dataset.csv")


# In[ ]:


world_cup_data


# In[ ]:




