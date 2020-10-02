#!/usr/bin/env python
# coding: utf-8

# # Analysis of Crime in Oakland of the datasets 2011-2016

# ## Background information

# A small project to find details of crimes in Oakland, CA, over the datasets that span from 2011-2016. 
# 
# As a bit of background information, there are some names of crimes within the dataset that may not be clear to people outside of the United States (such as myself), so it is useful to research the terminology within the datasets. For e.g. definitions for '*priority 1 & 2*' crimes are: 
# 
# *Priority 1 crime is said to be an urgent crime, for e.g. lights and sirens authorised, armed robbery, officer down etc.*
# 
# *Priority 2 crime is said to be of less urgency, for e.g. lights and sirens authorised, but follow basic traffic rules.*
# 
# Whilst evaluating the data I had also found points where I have diregarded for being unnecessary. These were things such as - while analysing priority crimes, a *prirority 0* had appeared in only 3 datasets of which had a count of less than 10 crimes. Where as *priority 1 & 2* had a count of a much larger number for e.g. 25000.
# 
# A big point to make - the data set for 2016 is inconclusive and the last entry is in the middle of the year.

# ### Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Timestamp
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression


# #### Import files

# In[ ]:


df_2011 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2011.csv', parse_dates=['Create Time', 'Closed Time'])
df_2012 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2012.csv', parse_dates=['Create Time', 'Closed Time'])
df_2013 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2013.csv', parse_dates=['Create Time', 'Closed Time'])
df_2014 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2014.csv', parse_dates=['Create Time', 'Closed Time'])
df_2015 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2015.csv', parse_dates=['Create Time', 'Closed Time'])
df_2016 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2016.csv', parse_dates=['Create Time', 'Closed Time'])


# In[ ]:


list_dfs = [df_2011, df_2012, df_2013, df_2014, df_2015, df_2016]


# ### First few rows of data for all data sets.

# In[ ]:


def shapes():
    x = 0
    for i in list_dfs:
        print(f"Shape of dataset for {x+2011} is {i.shape}")
        x+=1
shapes()


# In[ ]:


df_2011.head()


# In[ ]:


df_2012.head()


# In[ ]:


df_2013.head()


# In[ ]:


df_2014.head()


# In[ ]:


df_2015.head()


# In[ ]:


df_2016.head()


# I have decided to focus on the Priority column within all datasets, and compare with other columns for analysis.

# > ### Priority Analysis

# Amount of Priority crimes for all years observed:

# In[ ]:


# Code to show count of priority crimes per year.
a = 0
for i in list_dfs:
    print(i[i['Priority']!=0].groupby(['Priority']).size().reset_index(name=str(f'Count in {a + 2011}')))
    a += 1
    print(' ')


# In[ ]:


# Bar charts for comparing priority type crimes
df = pd.DataFrame([
    [1, 36699, 41926, 43171, 42773, 42418, 24555],
    [2, 143314, 145504, 144859, 144707, 150162, 86272]
],
columns=['Priority']+[f'Count in {x}' for x in range(2011,2017)]
)

df.plot.bar(x='Priority', subplots=True, layout=(2,3), figsize=(15, 7))


# In[ ]:


pri1_2011 = 36699
pri2_2011 = 143314
total_2011 = pri1_2011 + pri2_2011
print(f"Priority 1 crimes amounted to {round((pri1_2011/total_2011)*100, 3)}%, priority 2 crimes amounted to {round((pri2_2011/total_2011)*100, 3)}% in 2011.")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
pri1_2012 = 41926
pri2_2012 = 145504
total_2012 = pri1_2012 + pri2_2012
print(f"Priority 1 crimes amounted to {round((pri1_2012/total_2012)*100, 3)}%, priority 2 crimes amounted to {round((pri2_2012/total_2012)*100, 3)}% in 2012.")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
pri1_2013 = 43171
pri2_2013 = 144859
total_2013 = pri1_2013 + pri2_2013
print(f"Priority 1 crimes amounted to {round((pri1_2013/total_2013)*100, 3)}%, priority 2 crimes amounted to {round((pri2_2013/total_2013)*100, 3)}% in 2013.")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
pri1_2014 = 42773
pri2_2014 = 144707
total_2014 = pri1_2014 + pri2_2014
print(f"Priority 1 crimes amounted to {round((pri1_2014/total_2014)*100, 3)}% priority 2 crimes amounted to {round((pri2_2014/total_2014)*100, 3)}% in 2014.")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
pri1_2015 = 42418
pri2_2015 = 150162
total_2015 = pri1_2015 + pri2_2015
print(f"Priority 1 crimes amounted to {round((pri1_2015/total_2015)*100, 3)}%, priority 2 crimes amounted to {round((pri2_2015/total_2015)*100, 3)}% in 2015.")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
pri1_2016 = 24555
pri2_2016 = 86272
total_2016 = pri1_2016 + pri2_2016
print(f"Priority 1 crimes amounted to {round((pri1_2016/total_2016)*100, 3)}% and priority 2 crimes amounted to {round((pri2_2016/total_2016)*100, 3)}%, for the first half of 2016.")
print("-----------------------------------------------------------------------------------------------------------------------------------------")


# Crime seems to be at a stable rate throughout the datasets. The margin of difference in percentage is only slight throughout the 6 years observed.

# ### Area ID analysis.

# In[ ]:


# Mean Priority count per Area/Location/Beat
def areaid_groupby():
    for i in list_dfs:
        print(i[i['Priority']!=0].groupby(['Area Id', 'Priority']).size())
        print(' ')
areaid_groupby()


# In[ ]:


fig, axes= plt.subplots(2, 3)
for i, d in enumerate(list_dfs):
    ax = axes.flatten()[i]
    dplot = d[['Area Id', 'Priority']].pivot_table(index='Area Id', columns=['Priority'], aggfunc=len)
    dplot = (dplot.assign(total=lambda x: x.sum(axis=1))
                  .sort_values('total', ascending=False)
                  .head(10)
                  .drop('total', axis=1))
    dplot.plot.bar(ax=ax, figsize=(15, 7), stacked=True)
    ax.set_title(f"Plot of Priority 1 and 2 crimes within Area Id for {i+2011}")
    plt.tight_layout()


# The Area Id's for each dataset have not been consistent with their category. To see the amount of crimes for each year split by priority, check below:

# Summing the amount of Priority 1 and 2 crimes per dataset we can see that there is an increase in both crimes.

# ### Beat Analysis

# In[ ]:


# Value count for beats displayed by priority 
for i in list_dfs:
    print(i[i['Priority']!=0].groupby(['Beat', 'Priority']).size())
    print(' ')


# In[ ]:


fig, axes = plt.subplots(2, 3)
for i, d in enumerate(list_dfs):
    ax = axes.flatten()[i]
    dplot = d[['Beat', 'Priority']].pivot_table(index='Beat', columns=['Priority'], aggfunc=len)
    dplot = (dplot.assign(total=lambda x: x.sum(axis=1))
                  .sort_values('total', ascending=False)
                  .head(10)
                  .drop('total', axis=1))
    dplot.plot.bar(ax=ax, figsize=(15, 7), stacked=True)
    ax.set_title(f"Top 10 Beats for {i+ 2011}")
    plt.tight_layout()


# ### Incident type description (Incident type id) analysis

# In[ ]:


# Top 20 most popular crimes across the data sets
df1 = df_2011['Incident Type Description'].value_counts()[:10]
df2 = df_2012['Incident Type Description'].value_counts()[:10]
df3 = df_2013['Incident Type Description'].value_counts()[:10]
df4 = df_2014['Incident Type Description'].value_counts()[:10]
df5 = df_2015['Incident Type Description'].value_counts()[:10]
df6 = df_2016['Incident Type Description'].value_counts()[:10]
list_df = [df1, df2, df3, df4, df5, df6]
fig, axes = plt.subplots(2, 3)
for d, i in zip(list_df, range(6)):
    ax=axes.ravel()[i];
    ax.set_title(f"Top 20 crimes in {i+2011}")
    d.plot.barh(ax=ax, figsize=(15, 7))
    plt.tight_layout()


# In[ ]:


fig, axes = plt.subplots(2, 3)
for i, d in enumerate(list_dfs):
    ax = axes.flatten()[i]
    dplot = d[['Incident Type Id', 'Priority']].pivot_table(index='Incident Type Id', columns='Priority',aggfunc=len)
    dplot = (dplot.assign(total=lambda x: x.sum(axis=1))
                  .sort_values('total', ascending=False)
                  .head(10)
                  .drop('total', axis=1))
    dplot.plot.barh(ax=ax, figsize=(15, 7), stacked=True)
    ax.set_title(f"Plot of Top 10 Incidents in {i+2011}")
    plt.tight_layout()


# Two graphs to show the 'Indcident Type Decription' as well as it's Id. The first graph shows that 'Alarm Ringer' is by far the most reported crime, however in graph 2 we can see that only a small percentage of that is *priority 1*. All through the 6 datasets we can see that 'Battery/242' is the highest reported *priority 1* crime.

# ### Time analysis

# In[ ]:


# Total amount of pripority crimes per month
pri_count_list = [df_2011.groupby(['Priority', df_2011['Create Time'].dt.to_period('m')]).Priority.count(),
                  df_2012.groupby(['Priority', df_2012['Create Time'].dt.to_period('m')]).Priority.count(),
                  df_2013.groupby(['Priority', df_2013['Create Time'].dt.to_period('m')]).Priority.count(),
                  df_2014.groupby(['Priority', df_2014['Create Time'].dt.to_period('m')]).Priority.count(),
                  df_2015.groupby(['Priority', df_2015['Create Time'].dt.to_period('m')]).Priority.count(),
                  df_2016.groupby(['Priority', df_2016['Create Time'].dt.to_period('m')]).Priority.count()]
fig, axes = plt.subplots(2, 3)
for d, ax in zip(pri_count_list, axes.ravel()):
    plot_df1 = d.unstack('Priority').loc[:, 1]
    plot_df2 = d.unstack('Priority').loc[:, 2]
    plot_df1.index = pd.PeriodIndex(plot_df1.index.tolist(), freq='m')
    plot_df2.index = pd.PeriodIndex(plot_df2.index.tolist(), freq='m')
    plt.suptitle('Visualisation of priorities by the year')
    plot_df1.plot(ax=ax, legend=True, figsize=(15, 7))
    plot_df2.plot(ax=ax, legend=True, figsize=(15, 7))


# The visualisation shows us that within each year, Priority 2 crimes seem to peak around July/August time. Apart from in 2014 where there seemed to be a drop.The plot for 2016 shows an inconclusive graph since the dataset was only a 7 month long time span.

# In[ ]:


count = 2011
x = []
for i in list_dfs:
    i['Difference in hours'] = i['Closed Time'] - i['Create Time']
    i['Difference in hours'] = i['Difference in hours']/np.timedelta64(1, 'h')
    mean_hours = round(i['Difference in hours'].mean(), 3)
    x.append(mean_hours)
    print(f"Difference in hours for {count} is {mean_hours} with a reported {i.shape[0]} crimes.")
    count += 1


# In[ ]:




