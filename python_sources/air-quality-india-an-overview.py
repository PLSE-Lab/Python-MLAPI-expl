#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (8, 6)

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Reading the Data
air = r'../input/data.csv'
df = pd.read_csv(air, engine='python')
df.dtypes


# In[ ]:


df.head()


# In[ ]:


df.rename(columns={'stn_code': 'station_code', 'location': 'city', 'type': 'area_type', 'location_monitoring_station': 'monitoring_station'}, inplace=True)
df.head()


# In[ ]:


df[["state","city"]].describe(include=['O'])


# In[ ]:


df[['so2','state']].groupby(["state"]).count().sort_values(by='so2',ascending=False).plot.bar()
plt.show()


# In[ ]:


df[['no2','state']].groupby(["state"]).count().sort_values(by='no2',ascending=False).plot.bar()
plt.show()


# In[ ]:


df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d') # date parse
df['year'] = df['date'].dt.year # year
df['year'] = df['year'].fillna(0.0).astype(int)
df = df[(df['year']>0)]
df['year'].head()


# In[ ]:


var = "so2"

#Top 5 States for SO2 Level
temp_df = df[[var,'year','state']].groupby(["year"]).count().reset_index().sort_values(by='year',ascending=False)
topstate = df[[var,'state']].groupby(["state"]).count().sort_values(by='so2',ascending=False).index [:5]
state_col = ["green","red","yellow","orange","purple"]

# Plotting the Curves
fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('{} Observation Count by Year for '.format(var), fontsize=20,fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Observation Count')
plt.plot(temp_df.year,temp_df["so2"],marker='o', linestyle='--', color='black', label='Square')
for (i,col) in zip(topstate, state_col):
    state_df= df[df.state==i][[var,'year','state']].groupby(["year"])    .count().reset_index().sort_values(by='year',ascending=False)
    plt.plot(state_df.year,state_df[var],marker='o', linestyle='--', color=col, label='Square')
plt.legend(topstate.insert(0, "All") , loc=2,fontsize='large')
plt.show()


# In[ ]:


# Create Heatmap Pivot with State as Row, Year as Col, So2 as Value
var = "so2"
f, ax = plt.subplots(figsize=(12,12))
ax.set_title('{} by state and year'.format(var))
sns.heatmap(df.pivot_table(var, index='state',
                columns=['year'],aggfunc='mean',margins=True),
                annot=True, linewidths=.5, ax=ax,cbar_kws={'label': 'Annual Average'})


# In[ ]:


var = "no2"

#Top 5 States for NO2 Level
temp_df = df[[var,'year','state']].groupby(["year"]).count().reset_index().sort_values(by='year',ascending=False)
topstate = df[[var,'state']].groupby(["state"]).count().sort_values(by='no2',ascending=False).index [:5]
state_col = ["green","red","yellow","orange","purple"]

# Plotting the Curves
fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('{} Observation Count by Year for '.format(var), fontsize=20,fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Observation Count')
plt.plot(temp_df.year,temp_df["no2"],marker='o', linestyle='--', color='black', label='Square')
for (i,col) in zip(topstate, state_col):
    state_df= df[df.state==i][[var,'year','state']].groupby(["year"])    .count().reset_index().sort_values(by='year',ascending=False)
    plt.plot(state_df.year,state_df[var],marker='o', linestyle='--', color=col, label='Square')
plt.legend(topstate.insert(0, "All") , loc=2,fontsize='large')
plt.show()


# In[ ]:


# Create Heatmap Pivot with State as Row, Year as Col, So2 as Value
var = "no2"
f, ax = plt.subplots(figsize=(12,12))
ax.set_title('{} by state and year'.format(var))
sns.heatmap(df.pivot_table(var, index='state',
                columns=['year'],aggfunc='mean',margins=True),

            annot=True, linewidths=.5, ax=ax,cbar_kws={'label': 'Annual Average'})

