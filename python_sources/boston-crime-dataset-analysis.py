#!/usr/bin/env python
# coding: utf-8

# ### About The Dataset
# Crime incident reports are provided by Boston Police Department (BPD) to document the initial details surrounding an incident to which BPD officers respond. This is a dataset containing records from the new crime incident report system, which includes a reduced set of fields focused on capturing the type of incident as well as when and where it occurred.
# 
# ### Tasks
# 
# * What types of crimes are most common?
# * Where are different types of crimes most likely to occur?
# * Does the frequency of crimes change over the day? Week? Year?
# 

# In[ ]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys, os, csv


# In[ ]:


pd.set_option("display.max_columns", 500)
dataset = pd.read_csv("../input/crimes-in-boston/crime.csv", encoding='latin-1')
dataset.shape


# In[ ]:


dataset.info()


# In[ ]:


dataset.head()


# ### Data Cleaning

# In[ ]:


dataset.columns


# In[ ]:


# Rename columns
rename = {
    'INCIDENT_NUMBER': 'incident_num',
    'OFFENSE_CODE':'code',
    'OFFENSE_CODE_GROUP':'code_group',
    'OFFENSE_DESCRIPTION':'description',
    'DISTRICT':'district',
    'REPORTING_AREA':'area',
    'SHOOTING':'shooting',
    'OCCURRED_ON_DATE':'date',
    'YEAR':'year',
    'MONTH':'month',
    'DAY_OF_WEEK':'day',
    'HOUR':'hour',
    'UCR_PART':'ucr_part',
    'STREET':'street',
    'Lat':'lat',
    'Long':'long',
    'Location':'location',
}

dataset.rename(columns=rename, inplace=True)


# In[ ]:


dataset['code_group'].value_counts()


# In[ ]:


dataset['ucr_part'].value_counts()


# In[ ]:


dataset['year'].value_counts()


# In[ ]:


dataset['shooting'].isnull().sum()


# In[ ]:


shooting = dataset['shooting'].copy()
shooting.fillna('N', inplace=True)
dataset['shooting'] = shooting

dataset['shooting'].head()


# In[ ]:


ucr_part = dataset['ucr_part'].copy()
ucr_part.replace(to_replace='Other', value='Part Four', inplace=True)
dataset['ucr_part'] = ucr_part


# In[ ]:


dataset['ucr_part'].value_counts()


# In[ ]:


dataset['ucr_part'].isnull().sum()


# In[ ]:


dataset[dataset['ucr_part'].isnull()]['code_group'].value_counts()


# In[ ]:


code_group = dataset['code_group'].copy()
code_group.replace(to_replace="INVESTIGATE PERSON", value="Investigate Person", inplace=True)
dataset['code_group'] = code_group


# In[ ]:


dataset.loc[(dataset['code_group'] == 'Investigate Person') & (dataset['ucr_part'].isnull()), 'ucr_part']= "Part Three"


# In[ ]:


dataset['ucr_part'].isnull().sum()


# In[ ]:


dataset.dropna(subset=['ucr_part'], inplace=True)


# In[ ]:


dataset['code_group'].value_counts().head()


# In[ ]:


order = dataset['code_group'].value_counts().head().index
order


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(data=dataset, x='code_group', hue='district', order=order)


# ### some observation
# * District B2 has worst driver with most accident
# * District D4 has the worst theft

# In[ ]:


data2017 = dataset[dataset['year']==2017].groupby(['month','district']).count()
data2017.head()


# In[ ]:


plt.figure(figsize=(12,12))
sns.lineplot(data=data2017.reset_index(), x='month', y='code', hue='district')


# In[ ]:


day_num_name = {'Monday':'1','Tuesday':'2','Wednesday':'3','Thursday':'4','Friday':'5','Saturday':'6','Sunday':'7',}
dataset['day_num'] = dataset['day'].map(day_num_name)


# In[ ]:


data_day_hour = dataset[dataset['year']==2017].groupby(['day_num','hour']).count()['code'].unstack()
plt.figure(figsize=(8,8))
sns.heatmap(data=data_day_hour, cmap='viridis', yticklabels=['Monday','Tuesday','Wednesday','Thursday','Friday','S'])


# * From `1am-7am` is the lowest incident rate and `4pm-6pm` is the highest.
# * From `3am - 7pm` is the lowest incident rate and doesn't have the same sharp peak at `5pm` like the weekdays.
# * Overall it looks like incidents are more prevalent during the week than during the weekend. 

# In[ ]:


df_day_hour_part1 = dataset[(dataset['year'] == 2017) & (dataset['code_group'] == 'Larceny')].groupby(['day_num','hour']).count()['code'].unstack()
plt.figure(figsize=(10,10))
sns.heatmap(data = df_day_hour_part1, cmap='viridis', yticklabels=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])


# * As we can see, theft usually happens in the middle of the day when people are working.
# * The middle of the week seems to hold the highest peak.

# In[ ]:


dfpart1 = dataset[(dataset['year'] == 2017) & (dataset['ucr_part'] == 'Part One')].groupby(['code_group','shooting']).count().reset_index().sort_values('code',ascending=False)
dfpart2 = dataset[(dataset['year'] == 2017) & (dataset['ucr_part'] == 'Part One') & (dataset['shooting'] == 'Y')].groupby(['code_group','shooting']).count().reset_index().sort_values('code',ascending=False)


# In[ ]:


order1 = dataset[dataset['ucr_part'] == 'Part One']['code_group'].value_counts().head()
order1


# In[ ]:


order1 = dataset[dataset['ucr_part'] == 'Part One']['code_group'].value_counts().head(5).index
plt.figure(figsize=(12,8))
sns.countplot(data = dataset, x='code_group',hue='district', order = order1)


# In[ ]:


order2 = dataset[dataset['ucr_part'] == 'Part Two']['code_group'].value_counts().head()
order2


# In[ ]:


order2 = dataset[dataset['ucr_part'] == 'Part Two']['code_group'].value_counts().head(5).index
plt.figure(figsize=(12,8))
sns.countplot(data = dataset, x='code_group',hue='district', order = order2)


# In[ ]:


order3 = dataset[dataset['ucr_part'] == 'Part Three']['code_group'].value_counts().head().index
plt.figure(figsize=(12,8))
sns.countplot(data = dataset, x='code_group',hue='district', order = order3)


# In[ ]:


plt.figure(figsize=(16,8))
plt.tight_layout()
sns.set_color_codes("pastel")
ax = sns.barplot(y="code", x="code_group", data=dfpart1, hue='shooting')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")


# #### What types of crimes are most common?
# * Most common incident - Motor Vehicle Response
# * Most common crimes - Larceny and Larceny from Motor Vehicle
# 
# #### Where are different types of crimes most likely to occur? 
# * UCR 1 (worst crimes) happen mostly in D4

# In[ ]:




