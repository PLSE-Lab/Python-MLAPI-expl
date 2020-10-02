#!/usr/bin/env python
# coding: utf-8

# # Novel Coronavirus(COVID-19) 
# According to [WHO](https://www.who.int/health-topics/coronavirus#tab=tab_1) Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  Older people and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illnesses.
# 
# ## COVID-19 in Bangladesh
# - First Confirmed Case in Banglades: 8 Mar  
# - First Death Cases in Bangladesh: 18 Mar 
# 
# ## Questions 
# - Starting date of outbreak in Bangladesh 
# - No. of cumulative confirmed cases 
# - No. of cumulative deaths cases 
# - No. of cumulative recovered cases 
# - No. of active cases 
# - Visualization of cases over time 
# 
# ## Data Repository
# - Kaggle: https://www.kaggle.com/jhossain/covid19-in-bangladesh
# - Github: https://github.com/datasticslab/COVID-19_Data
# - Google Sheet: https://docs.google.com/spreadsheets/d/1XcjNvPcQBiWAyjlshj43EuKKNrzwPEpS5umkeHxCPZY/edit?usp=sharing
# 
# ### Acknowledgements
# - IEDCR - https://www.iedcr.gov.bd/index.php/component/content/article/73-ncov-2019
# - Coronainfo - http://www.corona.gov.bd/
# - Bangladesh Govt. Dashboard- http://103.247.238.81/webportal/pages/covid19.php
# **Note: Currently the region is not included in this dataset. I can't find that type of variable in the available data sources**
# 
# ## Useful Kernels 
# - https://www.kaggle.com/imdevskp/covid-19-analysis-visualization-comparisons
# - https://www.kaggle.com/therealcyberlord/coronavirus-covid-19-visualization-prediction
# - https://www.kaggle.com/therealcyberlord/coronavirus-covid-19-visualization-prediction

# ## Libraries 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # static data visulization 
import matplotlib.dates as mdates # datetime format 
import seaborn as sns # data visualization 
from datetime import datetime, timedelta
import plotly.express as px 

# Color pallate 
cnf = '#393e46' # confirmed - grey
dth = '#ff2e63' # death - red
rec = '#21bf73' # recovered - cyan
act = '#fe9801' # active case - yellow


# ## Dataset 

# In[ ]:



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Read and store dataset into dataframe called df 
df = pd.read_csv("/kaggle/input/covid19-in-bangladesh/COVID-19_in_bd.csv")


# ## Data Exploring 

# In[ ]:


# Examine first few rows 
df.head()


# In[ ]:


# Check shape of dataset 
df.shape 


# There are 18 rows and 4 columns.

# In[ ]:


# listing columns or variables 
df.columns


# In[ ]:


# Numerical summary 
df.describe()


# In[ ]:


# Basic info about dataset 
df.info()


# ## Data Cleaning 

# In[ ]:


# Check missing values 
df.isnull().sum()


# Looks good! there is no missing value in this dataset.

# In[ ]:


# Find starting date of outbreak in Bangladesh 
start = df['Date'].min()
print(f"Starting date of COVID-19 outbreak in Bangladesh is {start}")


# In[ ]:


# Cumulative confirm cases 
confirmed = df['Confirmed'].max()
# Cumulative deaths cases 
deaths = df['Deaths'].max()
# Cumulative recovered cases 
recovered = df['Recovered'].max()
# Active cases = confirmed - deaths - recovered 
active = df['Confirmed'].max() - df['Deaths'].max() - df['Recovered'].max()
# Printing the result 
print(f"Cumulative Confirmed Cases = {confirmed}")
print(f"Cumulative Deaths Cases = {deaths}")
print(f"Cumulative Recovered Cases = {recovered}")
print(f"Active Cases = {active}")


# In[ ]:


# Create a new column for active case 
df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']


# In[ ]:


# Now take a look at dataset 
df.head()


# ## Confirmed, Deaths and Recovered Cases Over Time: Line Plot

# In[ ]:


# Grouping cases by date 
temp = df.groupby('Date')['Confirmed', 'Recovered', 'Deaths', 'Active'].sum().reset_index() 
# Unpivoting 
temp = temp.melt(id_vars='Date',value_vars = ['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Case', value_name='Count') 

# Visualization
fig = px.line(temp, x='Date', y='Count', color='Case', color_discrete_sequence=[cnf, rec, dth, act], template='ggplot2') 
fig.update_layout(title="COVID-19 Cases Over Time in Bangladesh")
fig.show()


# ## Confirmed, Deaths and Recovered Cases Over Time: Area Plot

# In[ ]:


# Grouping cases by date 
temp = df.groupby('Date')['Confirmed', 'Recovered', 'Deaths', 'Active'].sum().reset_index() 
# Unpivoting 
temp = temp.melt(id_vars='Date',value_vars = ['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Case', value_name='Count') 

# Visualization
fig = px.area(temp, x='Date', y='Count', color='Case', color_discrete_sequence=[cnf, rec, dth, act], template='ggplot2') 
fig.update_layout(title="COVID-19 Cases Over Time in Bangladesh", xaxis_rangeslider_visible=True)
fig.show()


# > On going...Please share your thoughts and advice. **Happy Coding!**
