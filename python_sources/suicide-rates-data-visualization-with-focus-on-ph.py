#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# Create general analysis of suicide rates
# 
# Since this is my first official project, this is guided by some kaggle projects.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir('../input/suicide-rates-overview-1985-to-2016'))

# Any results you write to the current directory are saved as output


# #### Data input

# In[ ]:


df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
df.head()


# #### Explore data

# In[ ]:


df.info()


# Complete data except HDI

# In[ ]:


df.isnull().any()


# In[ ]:


df['age'].value_counts()

#age is already age group


# In[ ]:


df['generation'].unique()
#6 types of generation


# In[ ]:


df.loc[df['generation'] == 'Millenials','age'].value_counts()
#group per generation


# #### Question 1
# 
# Check the countries

# In[ ]:


df['country'].unique()


# In[ ]:


#side note because I saw the philippines
plt.style.use('ggplot')
plt.figure(figsize = (10,6))
df.loc[df['country'] == 'Philippines'].groupby('year')['suicides_no'].sum().plot(kind = 'bar')
plt.title('Yearly suicide count in the Ph')


# Check the data by country

# In[ ]:


fig = plt.figure(figsize = (10,25))
ax = fig.add_subplot()
ax = sns.countplot(y = 'country', data = df)
ax.set(title = 'data by country', xlabel = 'count of data')


# In[ ]:


fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot()
df['sex'].value_counts().plot(kind = 'pie', title = 'suicide by gender', use_index = False, ax = ax)
fig.patch.set_facecolor('white')


# Correlation between variables

# In[ ]:


plt.figure(figsize=(10,7))
cor = sns.heatmap(data = df.corr(), annot = True, cmap = 'RdBu')


# High positive correlation
# - HDI and GDP per capita
# - suicide_no and population  (of course)

# #### Question 2
# 
# which age group suicides the most

# In[ ]:


df.head()


# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize=(10,7))
sns.barplot(data = df, x = 'sex', y = 'suicides_no', ci = False, hue = 'age')
plt.ylabel('suicide count')


# From the plot, it seems that group 35-54 suicides the most

# #### Question 3
# 
# Which generation suicides the most

# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize=(10,7))
sns.barplot(data = df, x = 'generation', y = 'suicides_no', ci = False)
plt.ylabel('suicide count')


# Boomers lol

# #### Question 4:
# 
# create lineplot to see suicides numbers according to year with age group

# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(10,7))
sns.lineplot(x = 'year', y = 'suicides_no', hue = 'age', data = df)


# #### Question 5: 
# 
# Lineplot for male and female. Suicide count dropped in 2015?

# In[ ]:


plt.figure(figsize=(10,7))
sns.lineplot(x = 'year', y = 'suicides_no', hue = 'sex', data = df)


# #### GDP per capita and suicide count

# In[ ]:


gdp_suicide = df.loc[:, ['suicides_no', 'gdp_per_capita ($)']]
gdp_suicide.head()


# In[ ]:


gdp_suicide.rename(columns = {'gdp_per_capita ($)' : 'gdp_per_capita'}, inplace = True)
gdp_suicide


# In[ ]:


gdp_mean = gdp_suicide['gdp_per_capita'].mean()
gdp_std = gdp_suicide['gdp_per_capita'].std()

gdp_mean, gdp_std


# Remove outliers

# In[ ]:


no_outliers = gdp_suicide[gdp_suicide['gdp_per_capita'].apply(lambda x: (x-gdp_mean)/gdp_std < 3)]


# In[ ]:


no_outliers.head()


# In[ ]:


plt.figure(figsize = (10,7))
sns.scatterplot(x = 'gdp_per_capita', y = 'suicides_no', data = no_outliers)


# Low suicide count between 20000 usd to 40000 usd gdp per capita. Highest suicide count at low gdp per capita

# ## Philippines

# In[ ]:


ph = df[df['country'] == 'Philippines']
ph.head()


# In[ ]:


plt.figure(figsize = (10,7))
sns.lineplot(x = 'year', y = 'suicides/100k pop', data = ph, hue = 'sex')


# Male suicide in the philippines significantly increased over the years

# #### Suicide rate by age group

# In[ ]:


plt.figure(figsize = (10,7))
sns.barplot(x = 'sex', y = 'suicides_no', data = ph, hue = 'age', ci = False)


# Compared to global data where age group 35-54 has the highest suicide count, the age group 15-24 has the highest suicide count in the philippines

# #### Compare Philippines to other countries

# In[ ]:


country_df = df.set_index('country')
grouped = country_df.groupby('country').mean()
grouped.head()


# In[ ]:


plt.figure(figsize = (10,7))

top20 = grouped.sort_values(by = 'suicides/100k pop', ascending = False).loc[:,'suicides/100k pop'].head(20)
top20.plot(kind = 'bar')


# In[ ]:


ph_suicide100k = grouped.loc['Philippines','suicides/100k pop']
ph_suicide100k = pd.Series(ph_suicide100k, name = 'Philippines', index = ['Philippines'])

plt.figure(figsize = (10,7))
top20.append(ph_suicide100k).plot(kind = 'bar')
plt.title('Top20 countries with highest suicide rate and PH')


# Philippines has low suicide rate

# #### Trend of Philippine GDP 
# 
# I want to check whether the PH GDP is increasing where as GDP per capita is not

# In[ ]:


ph.columns = ph.columns.str.replace(' ','')
ph['gdp_for_year($)'] = ph['gdp_for_year($)'].str.replace(',','').astype('float')


# In[ ]:


fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ph.groupby('year')['gdp_per_capita($)'].mean().plot(ax = ax)
ph.groupby('year')['gdp_for_year($)'].mean().plot(ax = ax2)
ax.set_title('Philippine GDP per capita')
ax2.set_title('Philippine GDP')
ax.locator_params(axis='x', nbins=15)
ax2.locator_params(axis='x', nbins=15)


# Both are increasing
