#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## 1. Preview
# ------------------------
# *1.1 Load the dataset, display the first 8 rows and change columns name to make it easier to process.*

# In[ ]:


countries = pd.read_csv('/kaggle/input/undata-country-profiles/country_profile_variables.csv',usecols = [0,1,2,3,4,5,6,7,8])
countries.columns= countries.columns.str.replace('(','',regex=True)
countries.columns= countries.columns.str.replace(')','',regex=True)
countries.columns= countries.columns.str.replace(',','',regex=True)
countries.columns= countries.columns.str.replace('%','',regex=True)
countries.columns= countries.columns.str.replace('.','',regex=True)
countries.columns= countries.columns.str.replace('$','D',regex=True)
countries.columns= countries.columns.str.replace(' ','_',regex=True)
countries.head(8)


# *1.2 Show the last 10 rows*

# In[ ]:


countries.tail(10)


# ## 2. Basic Information
# ---------------
# *2.1 How many rows are in the dataset?*

# In[ ]:


countries.shape[0]


# *2.2 How many columns are in the dataset?*

# In[ ]:


len(countries.columns)


# *2.3 Set the index into 'Country' and show the first 5 rows*

# In[ ]:


countries = countries.set_index('country')
countries.head()


# ## 3. Selection
# --------
# *3.1 How many population are in Indonesia?* 

# In[ ]:


countries.loc['Indonesia','Population_in_thousands_2017']


# *3.2 Show the last 10 Country with their Region Only*

# In[ ]:


countries.Region.iloc[-10:]


# ## 4. Conditional Selection
# -----------
# *4.1 Show all countries with 'South-easternAsia' Region*

# In[ ]:


countries[countries.Region == 'South-easternAsia']


# *4.2 How many countries has more than 100 Million population?*

# In[ ]:


countries.query('Population_in_thousands_2017 >= 100000').shape[0]


# ## 5. Summary Statistics
# -----------
# *5.1 Display the Top 10 Region with most population*

# In[ ]:


countries.groupby('Region')['Population_in_thousands_2017'].sum().sort_values(ascending=False).head(10)


# *5.2 What is the average sex ratio in the World?* 
# > (Sex Ratio is measured as the number of male births for every 100 female births)

# In[ ]:


countries['Sex_ratio_m_per_100_f_2017'].mean()


# *5.3 Which Country has most women population?*

# In[ ]:


countries['Sex_ratio_m_per_100_f_2017'].idxmin()


# # 6. Plots
# ------------
# *6.1 Create a pie chart that display the percentage of Region by Area*

# In[ ]:


countries.groupby('Region')['Surface_area_km2'].size().plot.pie()


# *6.2 Create a horizontal bar chart that displays the average GDP Per Capita in every country and display the top 20*

# In[ ]:


countries.GDP_per_capita_current_USD.sort_values().tail(20).plot.barh()


# *6.3 Create a vertical bar that displays the average sex ration by region and show the legend*

# In[ ]:


countries.groupby('Region')['Sex_ratio_m_per_100_f_2017'].mean().plot.bar(legend=True)

