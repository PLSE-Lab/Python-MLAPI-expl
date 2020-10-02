#!/usr/bin/env python
# coding: utf-8

# Motivation
# ----------
# The goal of this notebook is to determine answers to several questions such as:
# 
#  - Which countries have the highest-ranked universities on average? 
#  - In which countries are citizens most likely accepted to a job after graduating from university?
#  - Does the happiness index of a country affect the quality of its
#    education, or is it the other way around? In what way?

# In[68]:


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


# Import World University Ranking data set
educ_data_set = pd.read_csv("../input/world-university-rankings/cwurData.csv")


# In[ ]:


# Change all instances of 'USA' to 'United States'
educ_data_set.country[educ_data_set.country=='USA'] = 'United States'


# World University Ranking
# ------------------------
# Below are the World Happiness Index rankings per country sorted by rank.

# In[ ]:


# Display World University Ranking data set
educ_data_set


# In[ ]:


# Import and display World Happiness Index data set
happy_data_set = pd.read_csv("../input/world-happiness/2015.csv")
happy_data_set = happy_data_set.rename(columns={'Country': 'country_h', 'Region': 'region', 'Happiness Rank':'world_rank', 'Happiness Score':'score', 'Standard Error':'std_error', 'Economy (GDP per Capita)':'economy', 'Family':'family', 'Health (Life Expectancy)':'health', 'Freedom':'freedom', 'Trust (Government Corruption)':'trust', 'Generosity':'generosity', 'Dystopia Residual':'dystopia_residual'})
happy_data_set


# Countries sorted in ascending order by average university ranking
# ---------------------------------------------------
# Looking at the data frame below, Singapore appears to be the country with the highest-ranked universities when averaged, with next being Israel and Switzerland.

# In[ ]:


# Compute for the average of the world ranking per country
average_educ_rank = pd.DataFrame((educ_data_set.groupby('country'))['world_rank'].mean())
average_educ_rank = average_educ_rank.sort_values(by=['world_rank'])

average_educ_rank


# Countries sorted in ascending order by alumni employment ranks on average
# ---------------------------
# Seen below are the average alumni employment ranks of universities per country.

# In[ ]:


average_emp_rank = pd.DataFrame((educ_data_set.groupby('country'))['alumni_employment'].mean())
average_emp_rank = average_emp_rank.sort_values(by=['alumni_employment'])
average_emp_rank


# Correlation between columns for the World University Ranking
# ------------------------------------------------------------

# In[ ]:


# Compute Pearson's correlation for World University Ranking
correlation_table = educ_data_set.corr(method='pearson', min_periods=1)
correlation_table


# In[ ]:


import seaborn as sns
sns.heatmap(correlation_table, 
            xticklabels=correlation_table.columns.values,
            yticklabels=correlation_table.columns.values)


# World Happiness Index
# --------------------------------------------
# Below are the World Happiness Index rankings per country sorted by rank.

# In[ ]:


# Display World Happiness Index rankings
happy_rank = pd.DataFrame((happy_data_set['country_h']),(happy_data_set['world_rank']))
happy_rank


# Problems Encountered in Computing the Correlation Between World University Rankings and World Happiness Index
# ----------
# The two data sets cannot be easily joined for the following reasons
#  - Inconsistency of the naming of countries (ex.: USA in University Rankings, United States in World Happiness Index)
#  - Other countries in World Happiness Index are not listed in the World University Ranking
# 
# Noticeable pattern
# ----------
# Despite the problems stated above, it is noticeable that countries who are at the top of the World University Ranking are at the top of the World Happiness ranks as well.

# In[ ]:


# NEED TO PROVE RELATIONSHIP BETWEEN UNI RANKS AND HAPPINESS INDEX
# How to combine the two?

happy_rank.join(average_educ_rank,how='left', on='country')


# Conclusion
# ----------
# (Answer to highest-ranking countries based on per-country university ranking average and alumni employment ranks is found above)
# 
#  - The factors that contribute most to world ranking are the amount of publications, influence rating, amount of citations, and broad impact
#  - Attaining education in a country with high standards for education
#    contributes to higher employment rates, which is a factor that
#    contributes to higher ranks in the World Happiness Index.
