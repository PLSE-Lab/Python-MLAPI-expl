#!/usr/bin/env python
# coding: utf-8

# # [Beginner]
# # [Learning exercise]
# 
# #### The goal is to explore the data summer olympic medals

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Reading data

# In[ ]:


df = pd.read_csv('../input/summer-olympics-medals/Summer-Olympic-medals-1976-to-2008.csv', encoding='ISO-8859-1')


# In[ ]:


print(df.shape)
df.info()
df.head()


# In[ ]:


#rename columns into lower case
df.columns = df.columns.str.lower()
df.columns


# Mostly categorical data with year as numeric data. Explore some data

# #### Explore year data

# In[ ]:



df['year'].unique()


# In[ ]:


df['year'].value_counts(dropna = False) # there are 117 missing years


# In[ ]:


df[df['year'].isnull()].head() 


# All other rows are also null. Drop null values

# In[ ]:


df.dropna(inplace = True)


# In[ ]:


print(df.shape)
df.info()


# #### Exploring sport, discipline, and event columns

# In[ ]:


events = df.loc[:,'sport':'event']
events.head()


# In[ ]:


#here are all the sports of olympics. Most events are aquatics. 
events['sport'].value_counts()


# #### Explore why aquatics has the most number of events

# In[ ]:


events[events['sport'] == 'Aquatics'].loc[:,'discipline'].value_counts() 
#three types of aquatics discipline


# In[ ]:


events.groupby('sport')['discipline'].value_counts()


# Explore athletics

# In[ ]:


events.loc[events['sport'] == 'Athletics','event'].value_counts()


# There are many events under atheltics.

# #### Explore Countries

# In[ ]:


df['country'].unique().size

#there are 127 countries


# In[ ]:


#most number of countries in the dataset
df['country'].value_counts().head(10)


# #### Explore medals

# In[ ]:


df['medal'].unique()
#only three types of medals


# In[ ]:


df['athlete'].value_counts().head(10)
#michael phelps is the most common athlete in the dataset


# ## Analyses
# 
# explore and gain some insights from the data

# In[ ]:


df.head()


# #### Top 10 countries in terms of medal( gold, silver, bronze)

# In[ ]:


def get_top(df, col = 'event', n = 5):
    return df.sort_values(by = col)[-n:]


# In[ ]:


medal_country = df.groupby(['medal','country'], as_index = False)['event'].count()


# In[ ]:


medal_group = medal_country.groupby('medal', as_index = False).apply(get_top).sort_values(by = ['medal','event'], ascending = False)
medal_group = medal_group.reset_index().drop(['level_0', 'level_1'], axis = 1)


# In[ ]:


fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot()
sns.barplot(x = 'medal', y ='event', hue = 'country', data = medal_group, palette = 'Spectral', ax = ax, order = ['Gold','Silver','Bronze'])
plt.title('Top 5 countries by medal')


# It seems that USA has the highest number of medals. Next is soviet union

# ## Check USA

# In[ ]:


usa = df[df['country'] == 'United States']


# In[ ]:


usa.head()


# In[ ]:


plt.style.use('ggplot')
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot()
top5_usa_sport = usa.groupby('sport')['medal'].count().sort_values(ascending = False)
top5_usa_sport.head(10).plot(kind = 'bar', ax = ax)
ax.set_title('Top 10 highest medal earning sport in USA')


# #### Check the trend of medal earnings by countries

# In[ ]:


country_medal_count = df.groupby(['year','country'])['medal'].count().reset_index()
country_medal_count.sort_values(by = 'year')


# In[ ]:


fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot()
sns.lineplot( x = 'year', y = 'medal', hue = 'country', data = country_medal_count[country_medal_count['medal'] > 50])
plt.legend(ncol = 3, frameon = False, title = '')
plt.title('Medal count trend of countries with medals greater than 50')


# Soviet union stopped participating. China (which I cant see from the graph) has increased medal counts significantly. Some countries keep participating even though they cant improve their winnings

# Let's take a look at men basketball data

# In[ ]:


c1 = (df['sport'] == 'Basketball')
c2 = (df['gender'] == 'Men')
basketball = df[c1&c2]
print(basketball.shape)
basketball.head()


# In[ ]:


basketball_group = basketball.groupby(['year','country','medal'], as_index = False)['event'].count()
order = ['Gold','Silver','Bronze']
g = sns.FacetGrid(data = basketball_group, col = 'medal', height = 6, sharex = False, col_order = order)
g.map(sns.countplot, 'country' )
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Basketball winnings by medal', fontsize = 15)


# #### Check disciplines that are not won by US ever

# In[ ]:


won_by_us = df.loc[df['country'] == 'United States', 'discipline'].values


# In[ ]:


plt.figure(figsize = (10,6))
df.loc[~df['discipline'].isin(won_by_us), 'discipline'].value_counts().plot(kind = 'bar')
plt.title('Disciplines not won by US Ever')


# #### Men vs women

# In[ ]:


plt.figure(figsize = (10,6))
df['gender'].value_counts().plot(kind = 'pie', autopct='%1.0f%%', pctdistance=.5, labeldistance=1.1)
plt.title('distribution of men and women in summer olympics');


# Are there more men than women? Or is it because team sports have more men?

# In[ ]:


plt.figure(figsize = (10,6))
df.loc[:,['event','gender']].drop_duplicates().loc[:,'gender'].value_counts().plot(kind = 'pie', autopct = '%1.0f%%', pctdistance = .5)
plt.title('distribution of men and women on unique events')


# Even on unique event categories, there are more men than women

# In[ ]:


order = ['Men','Women']
plt.figure(figsize = (10,6))
sns.countplot(x = 'year', data = df, hue = 'gender', hue_order = order)
plt.title('count of male vs female participants across years');


# #### Athletes who played more than 1 event

# In[ ]:


athlete_group = df.groupby(['year','athlete','sport'])['event'].count()
more_than_4 = athlete_group[athlete_group > 4]


# In[ ]:


more_than_4 = more_than_4.reset_index()


# In[ ]:


more_than_4.head()


# In[ ]:


g = sns.FacetGrid(data = more_than_4, col = 'sport', height = 7, sharex = False)
g.map(sns.barplot, 'athlete', 'event', ci = False)
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Athletes with more than 4 events', fontsize = 15)


# Only athletes from gymnastics and aquatics have more than 4 events

# In[ ]:




