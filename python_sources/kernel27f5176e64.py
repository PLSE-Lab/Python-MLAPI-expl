#!/usr/bin/env python
# coding: utf-8

# # COVID-19 India -- Data Visualization

# ## Data
# 
# The data used here is obtained from the Kaggle dataset https://www.kaggle.com/sudalairajkumar/covid19-in-india . You can download it from this website and use it in your project. 

# # Data Visualization
# 
# ## Importing necessary packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## Reading the dataset

# In[ ]:


df_complete=pd.read_csv("../input/covid19-in-india/covid_19_india.csv")


# In[ ]:


df_complete.head()


# In[ ]:


df_complete.dropna(axis=0)


# In[ ]:


df_selective=pd.DataFrame(df_complete, columns= ['State/UnionTerritory','Confirmed'])


# In[ ]:


df_selective.head()


# ### Visualization of confirmed cases in India with respect to States/UnionTerritory in bar graph

# In[ ]:


df_confirmed=df_selective.groupby('State/UnionTerritory',as_index=False)['Confirmed'].sum()


# In[ ]:


df_confirmed.head()


# In[ ]:


df_confirmed


# In[ ]:


plt.figure(figsize=(10,6))
# title
plt.title("Covid-19 number of confirmed cases w.r.t state/union territory")
# Bar chart
sns.barplot(x=df_confirmed['State/UnionTerritory'], y=df_confirmed['Confirmed'])
plt.xticks(rotation=90)
plt.ylabel("Confirmed Cases Count")


# ### Visualization of cured cases in India with respect to States/UnionTerritory in bar graph

# In[ ]:


df_cured=df_complete.groupby('State/UnionTerritory',as_index=False)['Cured'].sum()


# In[ ]:


df_cured


# In[ ]:


plt.figure(figsize=(10,6))
# title
plt.title("Covid-19 Number of cured cases w.r.t state/union territory")
# Bar chart
sns.barplot(x=df_cured['State/UnionTerritory'], y=df_cured['Cured'])
plt.xticks(rotation=90)
plt.ylabel("Cured Cases Count")


# ### Visualization of death cases in India with respect to States/UnionTerritory in bar graph

# In[ ]:


df_death=df_complete.groupby('State/UnionTerritory',as_index=False)['Deaths'].sum()


# In[ ]:


df_death


# In[ ]:


plt.figure(figsize=(10,6))
# title
plt.title("Covid-19 Number of deaths w.r.t state/union territory")
# Bar chart
sns.barplot(x=df_death['State/UnionTerritory'], y=df_death['Deaths'])
plt.xticks(rotation=90)
plt.ylabel("Number of deaths")


# # Conclusion

# This analysis is done to represent the number of confirmed COVID19 cases in India with respect to the states and Union Territories. It also represents the number of cured cases in each state and the number of deaths in each state.
# 
# From this visualization, it can be easily found which state has the highest death rate, confirmed cases rate and cure rate. For now, it is clearly seen that Maharashtra has the highest number of confirmed cases. It has the highest number of cured cases but also the highest number of deaths.

# ## Thank you :)
