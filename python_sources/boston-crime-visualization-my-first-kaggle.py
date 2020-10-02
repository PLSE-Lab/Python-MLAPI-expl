#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# On this Boston crime sheet we'll answer the following questions...
# What types of crimes are most common? 
# Where are different types of crimes most likely to occur? 
# Does the frequency of crimes change over the day? Week? Year?


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime


# In[ ]:


df = pd.read_csv('../input/crime.csv', encoding='latin-1')


# In[ ]:


df.head(10)


# In[ ]:


# We see lots of work to be done to clean the data
# We won't need the incident numbers
df.drop('INCIDENT_NUMBER',axis=1,inplace=True)


# In[ ]:


# Make all the columns lowercase to make it easier to wrangle data
rename = {'OFFENSE_CODE': 'code',
          'OFFENSE_CODE_GROUP':'code_group', 
          'OFFENSE_DESCRIPTION': 'description', 
          'DISTRICT': 'district',
          'REPORTING_AREA': 'area',
          'SHOOTING': 'shooting',
          'OCCURRED_ON_DATE': 'date',
          'YEAR': 'year',
          'MONTH': 'month',
          'DAY_OF_WEEK': 'day',
          'HOUR': 'hour',
          'UCR_PART': 'ucr',
          'STREET': 'street',
          'Lat': 'lat',
          'Long': 'long',
          'Location': 'location'}
df.rename(columns=rename,inplace=True)


# In[ ]:


# ucr is not consistent naming convention.  It seems like it deals with the severity of the crime with Part 1 being more severe.
df['ucr'].unique()


# In[ ]:


# Add N for shooting No
df['shooting'].fillna('N',inplace=True)


# In[ ]:


#  We will change 'Other' to Part 4 and take a look at those that are NaN
df['ucr'].replace(to_replace='Other', value='Part Four',inplace=True)


# In[ ]:


df[df['ucr'].isnull()]['code_group'].unique()


# In[ ]:


# looking at the null values of ucr, we can see Investigate Person is present in Part 3.  
# None of the other values are present in the other ucr values.  
# Without knowing their actual ucr we'll have to drop them
df['code_group'].replace(to_replace='INVESTIGATE PERSON', value='Investigate Person',inplace=True)


# In[ ]:


df.loc[(df['code_group'] == 'Investigate Person') & (df['ucr'].isnull()), 'ucr']= 'Part Three'


# In[ ]:


# Droppig ucr's with NaN values
df.dropna(subset=['ucr'], inplace=True,axis=0)


# In[ ]:


# First visualization we'll see how the top 5 incidents pair against other districts 
order = df['code_group'].value_counts().head(5).index
plt.figure(figsize=(12,8))
sns.countplot(data = df, x='code_group',hue='district', order = order)


# In[ ]:


# Looks like District B2 has the worst drivers with the most accidents!   
# District D4 by far has the worst theft.  We can probably assume they will have the worst crimes as well.   

df2017 = df[df['year'] == 2017].groupby(['month','district']).count()


# In[ ]:


plt.figure(figsize=(12,12))
sns.lineplot(data = df2017.reset_index(), x='month', y='code',hue='district')


# In[ ]:


# District B2 has the highest incident rate month to month while district A15 has the least.


# In[ ]:


day_num_name = {'Monday':'1','Tuesday':'2','Wednesday':'3','Thursday':'4','Friday':'5','Saturday':'6','Sunday':'7',}
df['day_num'] = df['day'].map(day_num_name)


plt.figure(figsize=(8,8))
df_day_hour = df[df['year'] == 2017].groupby(['day_num','hour']).count()['code'].unstack()
sns.heatmap(data = df_day_hour, cmap='viridis', yticklabels=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])


# In[ ]:


# Weekdays - From 1am-7am is the lowest incident rate and 4pm-6pm is the highest.
# Weekends -  From 3am - 7pm is the lowest incident rate and doesn't have the same sharp peak at 5pm like the weekdays.
# Overall it looks like incidents are more prevalent during the week than during the weekend.  


# In[ ]:


# I am also curious about what time the larceny crime takes place during the day.  Larceny often happens during lunch.  Let's explore
plt.figure(figsize=(8,8))
df_day_hour_part1 = df[(df['year'] == 2017) & (df['code_group'] == 'Larceny')].groupby(['day_num','hour']).count()['code'].unstack()
sns.heatmap(data = df_day_hour_part1, cmap='viridis', yticklabels=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])


# In[ ]:


# As we can see, theft usually happens in the middle of the day when people are working.   
# The middle of the week seems to hold the highest peak.


# In[ ]:


dfpart1 = df[(df['year'] == 2017) & (df['ucr'] == 'Part One')].groupby(['code_group','shooting']).count().reset_index().sort_values('code',ascending=False)
dfpart2 = df[(df['year'] == 2017) & (df['ucr'] == 'Part One') & (df['shooting'] == 'Y')].groupby(['code_group','shooting']).count().reset_index().sort_values('code',ascending=False)


# In[ ]:


order1 = df[df['ucr'] == 'Part One']['code_group'].value_counts().head(5).index
plt.figure(figsize=(12,8))
sns.countplot(data = df, x='code_group',hue='district', order = order1)

# District D4 far surpasses other districts in theft.  B2 takes the lead in other UCR 1 crimes


# In[ ]:


order2 = df[df['ucr'] == 'Part Two']['code_group'].value_counts().head(5).index
plt.figure(figsize=(12,8))
sns.countplot(data = df, x='code_group',hue='district', order = order2)

# B2 seems to lead in all the highest occuring incidents in UCR 2 with district C11 behind it.


# In[ ]:


order3 = df[df['ucr'] == 'Part Three']['code_group'].value_counts().head(5).index
plt.figure(figsize=(12,8))
sns.countplot(data = df, x='code_group',hue='district', order = order3)

# District B2 leads in most categories in UCR 3


# In[ ]:





# In[ ]:


plt.figure(figsize=(16,8))
plt.tight_layout()
sns.set_color_codes("pastel")
ax = sns.barplot(y="code", x="code_group", data=dfpart1, hue='shooting')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")


#  Shootings are less prevalent then I originally anticipated but homicide seems to be the only crime in which
#  Shootings outnumber non-shootings.  
# 
# In Summary
# 
#  What types of crimes are most common? 
#     # Most common incident - Motor Vehicle Response
#     # Most common crimes - Larceny and Larceny from Motor Vehicle
#  Where are different types of crimes most likely to occur? 
#      UCR 1 (worst crimes) happen mostly in D4.   Other incidents and crimes most likely will occur in B2
#  Does the frequency of crimes change over the day? Week? Year?
#      Yes the frequency of crimes changes in the day.   The least amount of incidents happen late night to early morning
#      Most incidents occur from lunch time to 5pm.   Weekends have a noticible drop in activity.
#      Monthly the incidents seem to peak from about June-August.  Maybe because heat and tensions are high?
#     
#  District A15 looks like it has the least amount of activity.   If there were population counts on this dataset we 
#  could look at the crime rates per capita.   

# In[ ]:




