#!/usr/bin/env python
# coding: utf-8

# # Quiz 4 Part 1: Traffic Accidents in the U.S.
# 
# This submission explores traffic accidents across 49 states within the United States over the past 3 years (from February 2016 to March 2019).
# 
# > *Questions to Explore:*
# > * Can we determine a correlation between accidents and variables such as population, time of the year, or weather?
# > * After data analysis, how might we determine the most common causes of accidents and implement preventative measures?

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


# ## ** 1. Import the CSV data into a DataFrame and then preview the first 5 rows of data.**
# > *Methodology:*
# * (1) Preview: .head()

# In[ ]:


accidents = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_May19.csv')
accidents.head()


# ## 2. How many accidents occurred on average per year (rounded to the nearest integer)?
# 
# > *Methodology*: 
# * (2) Basic information: .shape

# In[ ]:


round(accidents.shape[0] / 3)


# ## **3. Find the number of accidents per states and display the top 10 states with the most accidents using a bar graph (in descending order).**
# 
# > *Methodology*: 
# * (3) Selection: .iloc
# * (6) Split-apply-combine: .groupby()
# * (7) Frequency count: .size()
# * (8) Plots: .plot()
# * (9) Sorting: .sort_values()

# In[ ]:


accidents.groupby('State')         .size()         .iloc[:10]         .sort_values(ascending=False)         .plot.bar()


# ## 4. What role might weather conditions play in the cause of accidents? Create a pie graph of the top 5 weather conditions with the most amount of accidents.
# > *Methodology:*
# * (3) Selection: .iloc
# * (6) Split-apply-combine: .groupby()
# * (7) Frequency count: .size()
# * (8) Plots: .plot()
# * (9) Sorting: .sort_values()

# In[ ]:


accidents.groupby('Weather_Condition')         .size()         .sort_values(ascending = False)         .iloc[:5]         .plot.pie()


# ## 5. Are there specific points in the year in which more accidents occur? Find the top 10 days and times in which the most accidents occur.
# > *Methodology:*
# * (3) Selection: .iloc
# * (6) Split-apply-combine: .groupby()
# * (7) Frequency count: .size()
# * (9) Sorting: .sort_values()

# In[ ]:


accidents.groupby('Start_Time')         .size()         .sort_values(ascending = False)         .iloc[:10]


# ## **6. In CA, what is the average severity of an accident (rounded to 2 decimal points)?**
# > *Methodology:*
# * (4) Conditional selection: .query()
# * (5) Summary statistics: .mean()

# In[ ]:


round(accidents.query('State == "CA"').Severity.mean(), 2)


# ## 7. Which county in CA experiences the most amount of accidents? The least? Output both these counties' names and their corresponding number of accidents. 
# 
# > *Methodology:*
# * (4) Conditional selection: .query()
# * (6) Split-apply-combine: .groupby()
# * (7) Frequency count: .size()
# * (10) Max and min: .max(), .min(), .idxmax(), .idxmin()

# In[ ]:


filtered = accidents.query('State == "CA"').groupby('County').size()
print(filtered.idxmax() + ': ' + str(filtered.max()))
print(filtered.idxmin() + ': ' + str(filtered.min()))

