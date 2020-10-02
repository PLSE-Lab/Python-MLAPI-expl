#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This is a kernel starter code demonstrating how to read in the data and begin exploring. Also contain some analitics of winners Winter Olympic Games in biathlon.  If you're inspired to dig deeper, click the blue "Fork Notebook" button at the top of this kernel to begin editing.
# 
# ### Data
# This is complete dataset of Winter Olympic medalists in Biathlon from Grenoble 1960 to Pyeongchang 2018. I scraped the data from [oympic.org](https://www.olympic.org/)
# 
# The dataset is really small and contains only 253 rows and 10 columns. Each row corresponds to an individual athlete or team winning in biathlon Olympic event. All team events represent a country and not including members of teams. It means that we can count for concrete athlete only individual winning medals. You can find a description of each column in a relative place for it. 
# 
# The are some features for the dataset:
# 
# * the dataset does not include 1924 military patrol medal's;
# * 2 silver medals and no bronze were awarded at 2010 men's individual distance;
# * 2 silver medals at the 2014 Olympics were stripped and they are not redistributed (not included).
# 
# Dataset was created at 17 September 2018.

# ## Imports
# 
# I am using a typical data science stack: `numpy`, `pandas`, `sklearn`, `matplotlib`. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for plotting
import matplotlib.pyplot as plt # for plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# ##  Read in Data 
# 
# There is not some specifics in dataset. Just read it.

# In[ ]:


data = pd.read_csv('../input/olympic_biathlon.csv')
print('Biathlon data: \nRows: {}\nCols: {}'.format(data.shape[0], data.shape[1]))


# In[ ]:


data.head()


# In[ ]:


list(data.columns)


# From the output below we can see that dataset hasn't a lot missing values. Only column **Country** had. 
# 
# Columns **Discipline URL** and **Game URL** are not important for the next analysis. 
# 
# `data.info()`  and `data.describe()` give for us some information about data that can make some intuition and first impression about.

# In[ ]:


data.info()


# In[ ]:


data.describe(include='all')


# From deeper analysis of column **Country** we can see that all missing values related only for Relay Competitions and can be replaced by winners.

# In[ ]:


data[data['Country'].isna()]


# In[ ]:


data['Country'] = np.where(data['Country'].isna(), data['Winner'], data['Country'])


# We can see that now we haven't  missing values in dataset.

# In[ ]:


data.info()


# The next step is not so important, because I applied it for column that I will remove for analysis, but it will give understanding for newcommers in pandas  how concatenate some text with existing column. We can see how column **Game URL** was change.

# In[ ]:


data['Game URL'] = 'https://www.olympic.org' + data['Game URL']
data.head()


# I removed some not important columns for my analysis.

# In[ ]:


data.drop(['Discipline URL','Game URL'], inplace=True, axis=1)
data.head()


# Let's go deeper to column **Discipline**. We can see all disciplines in biathlon through history. Some of them have the same meaning but different representation because rules, distance length and a number of biathletes in relay races were changed through the years. I am trying to create unification column with real disciplines.  
# 
# Whereas dataset specifics I create the dictionary manually for that tasks. New column **Discipline** represents real disciplines in biathlon.

# In[ ]:


data['Discipline'].value_counts()


# In[ ]:


disciplines = {'4x7.5km Relay': 'Relay', 
               '4x6km Relay':'Relay', 
               '2x6km Women + 2x7.5km Men Relay':'Relay', 
               '3x7.5km Relay':'Relay',
               '20km Individual':'Individual',
               '15km Individual':'Individual',
               '10km Sprint':'Sprint',
               '7.5km Sprint':'Sprint',
               '12.5km Pursuit':'Pursuit',
               '10km Pursuit':'Pursuit',
               '15km Mass Start':'Mass Start',
               '12.5km Mass Start':'Mass Start'
              }


# In[ ]:


data['Discipline'] = data['Discipline'].map(disciplines)


# In[ ]:


data['Discipline'].value_counts()


# ## Medal Distribution

# For plots building, I used a combination of `matplotlib` and  `seaborn` libraries. The first distribution is built with using `barplot` and shows all medal winners by country. 
# 
# We can see that TOP 3 biathlon nations are Germany, Norway, and France.

# In[ ]:


sns.set(style='whitegrid')
country = data['Country'].value_counts()[:]
plt.figure(figsize=(12,12))
ax = sns.barplot(x=country.values, y=country.index)
plt.title('All Olympic Winners in Biathlon by Country', fontsize=18)
plt.xlabel('Number of Medals', fontsize=18)
plt.ylabel('Country', fontsize=18)

for patch, value in zip(ax.patches, country.values[:]):
#   print(patch.get_y(), ' ' ,patch.get_height(), ' ', patch.get_x(), ' ', patch.get_width(), ' ', value)
  ax.text(patch.get_width()+0.5,
          patch.get_y()+0.65,
          value,
          ha="left",
          fontsize=16) 


# The next two distributions are built with using `countplot` and shows all medal winners by country and by gender. The result of plots the same but I used different approaches.
# 
# We can see that TOP 3 is changed. Norway won most medals in men  and Germany in women.

# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(12,12))
ax = sns.countplot(y='Country',data=data[data['Gender']=='Men'], order = data[data['Gender']=='Men']['Country'].value_counts().index)
plt.title('All Men Olympic Winners in Biathlon by Country', fontsize=18)
plt.xlabel('Number of Medals', fontsize=18)
plt.ylabel('Country', fontsize=18)

for patch, value in zip(ax.patches, data[data['Gender']=='Men']['Country'].value_counts()):
  ax.text(patch.get_width()+0.5,
          patch.get_y()+0.65,
          value,
          ha="left",
          fontsize=16) 


# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(12,12))
ax = sns.countplot(y='Country',data=data[data['Gender']=='Women'], order = data[data['Gender']=='Women']['Country'].value_counts().index)
plt.title('All Women Olympic Winners in Biathlon by Country', fontsize=18)
plt.xlabel('Number of Medals', fontsize=18)
plt.ylabel('Country', fontsize=18)

for patch, value in zip(ax.patches, data[data['Gender']=='Women']['Country'].value_counts().values[:]):
  ax.text(patch.get_width()+0.5,
          patch.get_y()+0.65,
          value,
          ha="left",
          fontsize=16) 


# If you interested in individual persons (relay races not included) you can find a top biathletes. The list below consist of biathletes that won more than two medals for a career. I am not surprised! A lot of respects to **Ole Einar BJOERNDALEN ** , **Martin FOURCADE**, **Anastazia KUZMINA  **,  and every athletes!

# In[ ]:


data[(data['Discipline']!='Relay')]['Winner'].value_counts()[data[(data['Discipline']!='Relay')]['Winner'].value_counts().values>=3]


# In[ ]:


sns.set(style='whitegrid')
biathletes = data[(data['Discipline']!='Relay')]['Winner'].value_counts()[data[(data['Discipline']!='Relay')]['Winner'].value_counts().values>=3]
plt.figure(figsize=(12,12))
ax = sns.barplot(x=biathletes.values, y=biathletes.index)
plt.title('All Olympic Winners in Biathlon by Athlete', fontsize=18)
plt.xlabel('Number of Medals', fontsize=18)
plt.ylabel('Athlete', fontsize=18)

for patch, value in zip(ax.patches, biathletes.values[:]):
#   print(patch.get_y(), ' ' ,patch.get_height(), ' ', patch.get_x(), ' ', patch.get_width(), ' ', value)
  ax.text(patch.get_width()+0.5,
          patch.get_y()+0.65,
          value,
          ha="left",
          fontsize=16) 


# And also the list of biathletes that won GOLD medals for a career more than one. A list of TOP 3 athletes still the same.

# In[ ]:


data[(data['Discipline']!='Relay')&(data['Medal']=='G')]['Winner'].value_counts()[data[(data['Discipline']!='Relay')&(data['Medal']=='G')]['Winner'].value_counts().values>=2]


# In[ ]:


sns.set(style='whitegrid')
biathletes = data[(data['Discipline']!='Relay')&(data['Medal']=='G')]['Winner'].value_counts()[data[(data['Discipline']!='Relay')&(data['Medal']=='G')]['Winner'].value_counts().values>=2]
plt.figure(figsize=(12,12))
ax = sns.barplot(x=biathletes.values, y=biathletes.index)
plt.title('All GOLD Olympic Winners in Biathlon by Athlete', fontsize=18)
plt.xlabel('Number of Medals', fontsize=18)
plt.ylabel('Athlete', fontsize=18)

for patch, value in zip(ax.patches, biathletes.values[:]):
#   print(patch.get_y(), ' ' ,patch.get_height(), ' ', patch.get_x(), ' ', patch.get_width(), ' ', value)
  ax.text(patch.get_width()+0.5,
          patch.get_y()+0.65,
          value,
          ha="left",
          fontsize=16) 


# If you wanted to group data in any way you can make something like below and continue the analysis.

# In[ ]:


grouped_data = data.groupby(['Country','Discipline'])['Gender'].count().sort_index().reset_index()
grouped_data


# Below heatmap of dataset for Countries and Disciplines with a counting of number medals.

# In[ ]:


grouped_data = grouped_data.pivot('Country','Discipline','Gender')
grouped_data


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(grouped_data, cmap='RdYlGn', annot=True, linewidths=.5)


# I also create medals distribution by year for TOP winners in men (Norway) and women (Germany).

# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(8,8))
filter_germany=(data['Country']=='GER')&(data['Gender']=='Women')
ax = sns.countplot(x='Year', data=data[filter_germany])#, order = data[filter_germany]['Year'].value_counts().index)
plt.title('All Germany Women Olympic Winners in Biathlon by Year', fontsize=18)
plt.ylabel('Number of Medals', fontsize=18)
plt.xlabel('Women Germany Team', fontsize=18)


# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(8,8))
filter_norway=(data['Country']=='NOR')&(data['Gender']=='Men')
ax = sns.countplot(x='Year', data=data[filter_norway])
plt.title('All Norway Men Olympic Winners in Biathlon by Year', fontsize=18)
plt.ylabel('Number of Medals', fontsize=18)
plt.xlabel('Men Norway Team', fontsize=18)


# ## Conclusions
# There are still lot opened questions for this dataset but they don't include to list of the dataset review.
# 
# I can give some additional list of questions that help received a new analytics about.
# 
# 1. How medals were distributed between Relay and Individual Races?
# 1. How different type medals are distributed between countries?
# 1. How distribution was changed for the last 5 or 7 years when all disciplines were included and athletes stay more professional than before?
# 
# I would like to write the article about it not only from data analysis point of view but like a fan and expert in biathlon on [Medium](https://medium.com/)
# 
# Welcome feedback and constructive criticism. I can be reached on comments. 

# In[ ]:




