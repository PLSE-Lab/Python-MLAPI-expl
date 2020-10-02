#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("darkgrid")


# In[ ]:


df = pd.read_csv("../input/road-accidents-in-india/only_road_accidents_data3.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.isna().any()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


# Road Accidents over the years (counts)

df_over_years = df.loc[:, ["YEAR", "Total"]]
df_over_years = df_over_years.groupby("YEAR").sum()['Total']

#ploting graph

plt.figure(figsize=(20,7))
sns.lineplot(data=df_over_years, color="coral", marker="o")
plt.xlabel("Year")
plt.ylabel("Number of Road Accidents ")
plt.title("Number of Road Accidents by Year")
plt.xticks(df_over_years.index)


# In[ ]:


# Road Accidents over the years (% Change over the year)

new_df_over_years = df_over_years.to_frame()
new_df_over_years = new_df_over_years.pct_change().round(2)

#ploting graph

plt.figure(figsize=(18,9))
new_df_over_years.plot(kind='barh')
plt.xlabel("YEAR")
plt.ylabel("% change of Road Accidents ")
plt.title("% change in Road Accidents Over the Year")
plt.xticks(rotation=45)


# In[ ]:


# Number of Road Accidents by State

df_state_over_years = df.loc[:, ["Total", "STATE/UT"]]
df_state_over_years = df_state_over_years.groupby("STATE/UT").sum()['Total']
df_state_over_years = df_state_over_years.sort_values(ascending=False)

#ploting graph

plt.figure(figsize=(20,7))
df_state_over_years.plot(kind='bar')
plt.xlabel("STATE")
plt.ylabel("Number of Road Accidents ")
plt.title("Number of Road Accidents by State")
plt.xticks(rotation=90)


# In[ ]:


# Road Accidents by YEAR and STATE

df_year_vs_state = df.loc[:, ['STATE/UT', 'YEAR', 'Total']]
df_year_vs_state = df_year_vs_state.groupby(['STATE/UT', 'YEAR'], as_index=False).sum()

df_year_vs_state = pd.pivot_table(df_year_vs_state,index=["STATE/UT"],values=["Total"], columns=["YEAR"])

plt.figure(figsize=(20,9))
sns.heatmap(data=df_year_vs_state, cmap="YlGnBu", annot=True, linewidths=.5, fmt=".1f")
plt.xlabel("YEAR")
plt.ylabel("STATE")
plt.xticks(rotation=45,ha='right')
plt.yticks(rotation=45,ha='right')
plt.title("Road Accidents by YEAR and STATE")


# In[ ]:


# Road Accidents by YEAR and Day Hours

df_time = df.loc[:,["0-3 hrs. (Night)", "3-6 hrs. (Night)", "6-9 hrs (Day)", "9-12 hrs (Day)",                    "12-15 hrs (Day)", "15-18 hrs (Day)", "18-21 hrs (Night)", "21-24 hrs (Night)", "YEAR"]]

df_time = df_time.groupby('YEAR').sum()

plt.figure(figsize=(20,6))
sns.heatmap(data = df_time, cmap="YlGnBu", annot=True, linewidths=.5, fmt="0.1f")
plt.xlabel("Day Hours")
plt.ylabel("YEAR")
plt.xticks(rotation=45,ha='right')
plt.title("Road Accidents by YEAR and Day Hours")


# In[ ]:


# Road Accidents by STATE and Day Hours

df_time_state = df.loc[:,["0-3 hrs. (Night)", "3-6 hrs. (Night)", "6-9 hrs (Day)", "9-12 hrs (Day)",                    "12-15 hrs (Day)", "15-18 hrs (Day)", "18-21 hrs (Night)", "21-24 hrs (Night)", "STATE/UT"]]

df_time_state = df_time_state.groupby('STATE/UT').sum()

plt.figure(figsize=(20,10))
sns.heatmap(data = df_time_state, cmap="YlGnBu", annot=True, linewidths=.5, fmt="0.1f")
plt.xlabel("Day Hours")
plt.ylabel("STATE")
plt.xticks(rotation=45,ha='right')
plt.title("Road Accidents by STATE and Day Hours")


# In[ ]:


# Number of Road Accidents by top 5 State

def top_n_state_over_year(top, year):
    df_state = df.loc[:, ['YEAR', 'STATE/UT', 'Total']]
    df_state = df_state[df_state['YEAR'] == year].groupby(['STATE/UT'], as_index=False).sum()    .sort_values(by=['YEAR', 'Total'], ascending=[True,False]).head(top)
    plt.figure(figsize=(10,5))
    sns.barplot(x="STATE/UT", y="Total", data=df_state)
    plt.title(f"Top {top} States (Road Accidents in Year {year})")
    plt.xlabel("STATE")
    plt.ylabel("Number of Road Accidents")
    plt.xticks(rotation=90)
    return;


# In[ ]:


top_n_state_over_year(5, 2001)


# In[ ]:


top_n_state_over_year(5, 2002)


# In[ ]:


top_n_state_over_year(5, 2003)


# In[ ]:


top_n_state_over_year(5, 2004)


# In[ ]:


top_n_state_over_year(5, 2005)


# In[ ]:


top_n_state_over_year(5, 2006)


# In[ ]:


top_n_state_over_year(5, 2007)


# In[ ]:


top_n_state_over_year(5, 2008)


# In[ ]:


top_n_state_over_year(5, 2009)


# In[ ]:


top_n_state_over_year(5, 2010)


# In[ ]:


top_n_state_over_year(5, 2011)


# In[ ]:


top_n_state_over_year(5, 2012)


# In[ ]:


top_n_state_over_year(5, 2013)


# In[ ]:


top_n_state_over_year(5, 2014)


# In[ ]:




