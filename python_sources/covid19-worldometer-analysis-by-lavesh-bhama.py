#!/usr/bin/env python
# coding: utf-8

# # First hands on with kaggle's COVID19 dataset and Python's visualization libraries.
# 
# ## Author: Lavesh Bhama

# In[ ]:


# Call libraries
get_ipython().run_line_magic('reset', '-f')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns


# In[ ]:


#  Display multiple outputs from a jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


# Go to folder containing data file
#os.chdir("C:\\Users\\LB\\Desktop\\Python\\Data Analytics Training - FORE School\\uncover\\worldometer\\worldometer")


# In[ ]:


# Read CSV file 
data = pd.read_csv("../input/uncover/UNCOVER/worldometer/worldometer-confirmed-cases-and-deaths-by-country-territory-or-conveyance.csv")


# In[ ]:


# Check data types of attributes and Some more dataset related information
data.dtypes
pd.options.display.max_columns = 100
data.head(3)
data.info()   
data.describe()
data.shape             #(213, 10)
data.columns.values
len(data)              #213


# In[ ]:


# removing two unwanted rows, first one at index 0 and last at index 212, having data where country is 'World' and 'Total'

dt = data[1:212].copy()
dt.shape 


# In[ ]:


# Checking the removal of dersired rows at top and bottom
dt.head(3)
dt.tail(3)


# In[ ]:


# sorting the data
dt.sort_values(by='total_cases', ascending=False).head()
dt.head()


# In[ ]:


# considering top 10 affected countries only where total deaths are above avergae deaths for the cleaned data.
major_fatalities=dt[dt.total_deaths>np.mean(dt.total_deaths)].head(10)
major_fatalities
major_fatalities.shape #(10, 10)
major_fatalities.country.value_counts().sum() # 10
major_fatalities.describe()


# In[ ]:


# Bar plot for the top 10 worst affected countries in terms of total cases and respective deaths.
fig=plt.figure(figsize=(10,5))
plt.bar(major_fatalities.country,major_fatalities.total_cases,label='total cases', color = 'r')
plt.bar(major_fatalities.country,major_fatalities.total_deaths,label='total_deaths', color = 'b')
plt.xlabel('country')
plt.ylabel('total cases and total deaths')
plt.title('Bar Graph')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# Visually evident from the pie char, the number of deaths is huge in Italy, USA and Spain among others.

fig=plt.figure(figsize=(7,7))
plt.pie(major_fatalities.total_deaths, labels = major_fatalities.country, wedgeprops=dict(width=0.5), shadow = True)


# In[ ]:


# Box plot for top 10 worst affected countries case-wise.
sns.boxplot(y = 'total_cases', data = major_fatalities)


# In[ ]:


# Single scatte plot, plotting total cases on x and total deaths along y exis.
sns.scatterplot(x='total_cases' ,y = 'total_deaths', data = major_fatalities)


# In[ ]:


# joint plot (KDE), plotting total cases on x and total deaths along y exis.
sns.jointplot(x='total_cases' ,y = 'total_deaths', data = major_fatalities, kind='kde')


# In[ ]:


# pair plots
sns.pairplot(major_fatalities[['total_cases','total_deaths','total_recovered']])

