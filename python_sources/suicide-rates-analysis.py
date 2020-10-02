#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from matplotlib import pyplot as plt, patches as mpatches, cm
import re
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/master.csv', thousands=',')


# In[ ]:


df.rename(columns={' gdp_for_year ($) ': 'gdp_for_year ($)'}, inplace=True)


# In[ ]:


df.loc[:, 'age'] = df['age'].map(lambda x: x.replace('5-14 years', '05-14 years')) # adjust values to sort by age


# * Suicide rate comparison by age in Korea versus globally representing countries
# * The result is that suicides of 75+ years in Korea are increasing while ones in other countries are decreasing.

# In[ ]:


countries = ['Argentina', 'Australia', 'Canada', 'Israel', 'Republic of Korea', 'Japan', 'Slovenia', 'United States']
for country in countries:
    grouped = df[df['country'] == country].groupby(['year', 'age'])['suicides/100k pop'].sum().unstack('age')
    grouped.plot(figsize=(10, 10),
               title='Suicides per 100k population by age in ' + country,
               legend=True)


# * Further compare all in one for 75+ years

# In[ ]:


age = '75+ years'
countries = ['Argentina', 'Australia', 'Canada', 'Israel', 'Republic of Korea', 'Japan', 'Slovenia', 'United States']
df_75_global = df[df['age'] == age].groupby('year')['suicides/100k pop'].mean()
df_75_countries = df[(df['age'] == age) & df['country'].isin(countries)]
df_75_countries = df_75_countries.groupby(['year', 'country'])['suicides/100k pop'].sum().unstack('country')
df_75 = pd.concat([df_75_countries, df[df['age'] == age].groupby('year')['suicides/100k pop'].mean()], axis=1)
df_75.rename(columns={'suicides/100k pop': 'Global'}, inplace=True)
df_75.plot(figsize=(15, 15),
          title=age + ' for countries',
          legend=True,
          fontsize=20)
plt.show()


# * Top 10 groups of suicides per 100,000 population for groups of over 100,000 population

# In[ ]:


df_100k = df[df['population'] >= 10 ** 5]
df_100k.sort_values(by='suicides/100k pop', ascending=False)[:10]


# In[ ]:


north_europe = ['Sweden', 'Denmark', 'Norway', 'Finland', 'Republic of Korea']
df_ne = df[df['country'].isin(north_europe)]
ax = df_ne.groupby(['country', 'year'])['suicides/100k pop'].sum().unstack('country').plot(figsize=(10, 10))
ax.set_title('Suicides Number Comparison for Korea and North Europe Countries', fontsize=20)
ax.legend(fontsize=15)
ax.set_xlabel('Year', fontsize=20)
ax.set_ylabel('Suicides per 100k population', fontsize=20, color='red')
plt.show()


# In[ ]:


ax = df.groupby(['sex', 'year'])['suicides_no'].sum().unstack('sex').plot(figsize=(10, 10))
ax.set_title('Global suicides by gender', fontsize=30)
ax.set_ylabel('Suicides number', fontsize=20)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0, fontsize=20)
ax.set_xlabel('Year', color='Blue', fontsize=20, weight='heavy')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(15)
plt.show()


# In[ ]:


countries = ['Republic of Korea', 'Japan', 'Sweden', 'Hungary', 'Lithuania', 'United States']
df_adolescence = df[df['age'] == '15-24 years']
df_adolescence = df_adolescence[df['country'].isin(countries)]
ax = df_adolescence.groupby(['country', 'year'])['suicides/100k pop'].sum().unstack('country').plot(figsize=(10, 10))
ax.set_title('Teenagers Suicides for several countries', fontsize=25, family='sans-serif')
ax.set_xlabel('Year', fontsize=20, color='blue')
ax.set_ylabel('Suicides per 100k population', fontsize=20, color='red')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0, fontsize=15)
plt.show()


# * correlation between gdp and normalized suicides

# In[ ]:


df['suicides/100k pop'].corr(df['gdp_per_capita ($)'])


# In[ ]:


df['gdp_for_year ($)'].corr(df['suicides/100k pop'])


# # Draw scatter plot to see correlation between suicides and gdp per capita

# In[ ]:


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.scatter(df['suicides/100k pop'], df['gdp_per_capita ($)'])
ax.set_xlabel('suicides per 100k population', fontsize=20, color='red')
ax.set_ylabel('gdp_per_capita ($)', fontsize=20)
ax.set_title('Scatter plot for correlation between gdp and suicides', fontsize=25, weight='heavy')
plt.show()


# In[ ]:


df['age_num'] = df['age'].map(lambda x: x[:2]).map(int)


# In[ ]:


df['age_num'].corr(df['suicides/100k pop'])


# In[ ]:


countries = ['Republic of Korea', 'Japan', 'United States', 'Mexico', 'Argentian', 'Sweden']
ages = Series(df['age'].unique()).sort_values()
ages.index = np.arange(len(ages))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
for age, ax in zip(ages, axes):
    fig = plt.figure(figsize=(20, 20))
    ax.add_subplot(111)
    for country in countries:
            


# In[ ]:




