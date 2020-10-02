#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
from pandas_profiling import ProfileReport
from plotly.offline import iplot
get_ipython().system('pip install joypy')
import joypy
from sklearn.cluster import KMeans

plt.rcParams['figure.figsize'] = 8, 5
plt.style.use("fivethirtyeight")

data = pd.read_csv('../input/world-bank-youth-unemployment/API_ILO_country_YU.csv')


# # <h1 style="font-size:40px">Description of Data</h1>

# <img src="https://www.investopedia.com/thmb/G-f1etpNlgfR4UkDjN3E8pmgcrA=/735x0/unemployment-5bfc344bc9e77c00519c4b43.jpg">

# Unemployment, according to the Organisation for Economic Co-operation and Development (OECD), is persons above a specified age (usually above 15) not being in paid employment or self-employment but currently available for work during the reference period.
# 
# Unemployment is measured by the unemployment rate as the number of people who are unemployed as a percentage of the labour force (the total number of people employed added to those unemployed).

# In[ ]:


# describing the data

data.describe(include='all')


# In[ ]:


# Covariance

data.cov()


# In[ ]:


# correlation

data.corr()


# In[ ]:


sns.heatmap(data.corr())
plt.show()


# # <h1 style="font-size:40px">Report of Data</h1>

# In[ ]:


report = ProfileReport(data)


# In[ ]:


report


# # <h1 style="font-size:40px">Checking null and duplicate values</h1>

# In[ ]:


#checking for null values

data.isnull().sum()


# In[ ]:


#dropping duplicates

data = data.drop_duplicates()


# # <h1 style="font-size:40px">Distribution of the Variables</h1>

# In[ ]:


px.box(data.drop(['Country Name','Country Code'], axis=1))


# # <h1 style="font-size:40px">Asian Countries</h1>

# In[ ]:


asian_countries = ['India', 'China', 'Sri Lanka','Japan','Bangladesh']

df = data[data['Country Name'].isin(asian_countries)].reset_index(drop=True)

plt.figure(figsize=(10,7))
for i in range(df.shape[0]):
    lst = df.iloc[i].tolist()[2:]
    plt.plot([0,1,2,3,4], lst, label=df['Country Name'][i])
    
plt.legend()
plt.show()


# # <h1 style="font-size:40px">African Countries</h1>

# In[ ]:


african_countries = ['Nigeria', 'Kenya', 'Ghana','Ethiopia','Tanzania']

df = data[data['Country Name'].isin(african_countries)].reset_index(drop=True)

plt.figure(figsize=(10,7))
for i in range(df.shape[0]):
    lst = df.iloc[i].tolist()[2:]
    plt.plot([0,1,2,3,4], lst, label=df['Country Name'][i])
    
plt.legend()
plt.show()


# # <h1 style="font-size:40px">North American Countries</h1>

# In[ ]:


north_american_countries = ['United States', 'Canada', 'Panama','Mexico','Cuba']

df = data[data['Country Name'].isin(north_american_countries)].reset_index(drop=True)

plt.figure(figsize=(10,7))
for i in range(df.shape[0]):
    lst = df.iloc[i].tolist()[2:]
    plt.plot([0,1,2,3,4], lst, label=df['Country Name'][i])
    
plt.legend()
plt.show()


# # <h1 style="font-size:40px">US states unemployment data</h1>

# In[ ]:


data = pd.read_csv('../input/unemployment-by-county-us/output.csv')
pd.options.plotting.backend = 'plotly'


# ## Which county have highest unemployment rate? ---> San Juan County

# In[ ]:


df = data.loc[:,['County', 'Rate']]
df['maxrating'] = df.groupby('County')['Rate'].transform('max')
df = df.drop('Rate', axis=1).drop_duplicates().sort_values('maxrating', ascending=False).head(6)

df.plot(x='County', y='maxrating', kind='bar', color='maxrating')


# ### County unemployment rates per year

# In[ ]:


df = data.loc[:,['Year', 'County', 'Rate']]
df['meanrating'] = df.groupby([df.Year, df.County])['Rate'].transform('mean')
df = df.drop('Rate', axis=1).drop_duplicates().sort_values('meanrating', ascending=False)
df = df[df['County'].isin(['San Juan County','Starr County','Sioux County','Presidio County','Maverick County'])]
df = df.sort_values('Year')

fig=px.bar(df,x='County', y="meanrating", animation_frame="Year", 
           animation_group="County", color="County", hover_name="County", range_y=[0,45])
fig.show()


# ### Which state has highest Unemployment rate? ---> Colorado

# In[ ]:


df = data.loc[:,['State', 'Rate']]
df['maxrating'] = df.groupby('State')['Rate'].transform('max')
df = df.drop('Rate', axis=1).drop_duplicates().sort_values('maxrating', ascending=False).head(6)

df.plot(x='State', y='maxrating', kind='bar', color='maxrating')


# ### State unemployment rates per year

# In[ ]:


df = data.loc[:,['Year', 'State', 'Rate']]
df['meanrating'] = df.groupby([df.Year, df.State])['Rate'].transform('mean')
df = df.drop('Rate', axis=1).drop_duplicates().sort_values('meanrating', ascending=False)
df = df[df['State'].isin(['Colorado','Texas','North Dakota','Arizona','Michigan'])]
df = df.sort_values('Year')

fig=px.bar(df,x='State', y="meanrating", animation_frame="Year", 
           animation_group="State", color="State", hover_name="State", range_y = [0,15])
fig.show()


# ### Which year saw the highest unemployment? ---> 1992

# In[ ]:


df = data.loc[:,['Year', 'Rate']]
df['maxrating'] = df.groupby('Year')['Rate'].transform('max')
df = df.drop('Rate', axis=1).drop_duplicates().sort_values('maxrating', ascending=False).head(6)

df.plot(x='Year', y='maxrating', kind='bar', color='maxrating')


# In[ ]:




