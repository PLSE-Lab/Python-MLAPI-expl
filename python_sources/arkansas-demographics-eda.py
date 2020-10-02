#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of Akansas Demographics
# 
# Author: Ben Thornton
# Date created: 1/8/2018
# Data: https://www.kaggle.com/muonneutrino/us-census-demographic-data/data
# Summay: The purpose of this notebook is to explore figures from the 2015 U.S. census, specifically the demographics within the state of Arkansas.

# # Imports

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np


# # Load Data

# In[5]:


df = pd.read_csv('../input/acs2015_county_data.csv', index_col=0)
df.head()


# For this analysis, we'll focus specifically on the state of Arkansas.

# In[6]:


df_ark = df.groupby('State').get_group('Arkansas')
df_ark.head()


# County i.d.'s

# In[7]:


print(df_ark.index.unique)


# In[8]:


print(df_ark['County'].count())


# There are 75 counties in the state of Arkansas.

# Check for missing data

# In[10]:


total = df_ark.isnull().sum().sort_values(ascending=False)
percent = (df_ark.isnull().sum()/df_ark.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_data


# Awesome! No missing data.

# In[11]:


df_ark.describe()


# In[12]:


df_ark.columns


# Visualizations

# Total population of Arkansas

# In[14]:


pop = df_ark['TotalPop'].sum()
print('Total Population: ', pop)


# In[15]:


count_pop = pd.concat([df_ark['County'], df_ark['TotalPop'], df_ark['Men'], 
                       df_ark['Women']], axis=1, keys=['County', 'Population',
                                                       'Men', 'Women'])
count_pop = count_pop.sort_values(by='Population', ascending=True)
count_pop.head()


# Top 5 largest counties in Arkansas are located in Little Rock (Pulaski & Benton), Fayetteville (Washington), Fort Smith (Sebastian) and Conway (Faulkner)

# Population size by county

# In[17]:


data = count_pop
fig, ax = plt.subplots(figsize=(16,8))
fig = sns.barplot(x='County', y='Population', data=data)
fig.axis(ymin=0, ymax=400000)
plt.xticks(rotation=90)


# State population distribution among counties.

# In[18]:


sns.distplot(df_ark['TotalPop'])


# In[20]:


print("Skewness: %f" % df_ark['TotalPop'].skew())
print("Kurtosis: %f" % df_ark['TotalPop'].kurt())


# Men/Women Proportions

# In[21]:


labels = 'Men', 'Women'
sizes = [df_ark['Men'].sum(), df_ark['Women'].sum()]
explode = (0, 0.1)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.show


# Men/Women Proportions by County

# In[22]:


melt = pd.melt(count_pop, id_vars='County', value_vars=('Men', 'Women'), 
               var_name='Sex', value_name='Population')
melt.head()


# In[23]:


data = melt
gender = 'Men', 'Women'
fig = sns.factorplot(x='County', y='Population' , hue='Sex', data=data, kind='bar', aspect=.9, size=14)
plt.xticks(rotation=90)


# Race Demographics

# In[24]:


race = pd.concat([df_ark['County'], df_ark['Hispanic'], df_ark['White'],
                  df_ark['Black'], df_ark['Native'], df_ark['Asian'], 
                  df_ark['Pacific']],
                 axis=1, keys=['County', 'Hispanic', 'White', 'Black',
                              'Native', 'Asain'])

race.tail(10)


# In[25]:


melt2 = pd.melt(race, id_vars='County', value_vars=('Hispanic', 'White', 'Black', 'Native', 'Asain'), var_name='Race', value_name='Percent')
melt2.head()


# In[26]:


data = melt2
race = 'Hispanic', 'White', 'Black', 'Native', 'Asain'
fig = sns.factorplot(x='County', y='Percent' , hue='Race', data=data, kind='bar', aspect=1.5, size=11)
plt.xticks(rotation=90)


# Income distribution by county

# In[27]:


sns.distplot(df_ark['Income'])


# In[28]:


print("Skewness: %f" % df_ark['Income'].skew())
print("Kurtosis: %f" % df_ark['Income'].kurt())


# Percentage of citizens and non-citizens

# In[29]:


num_citizens = df_ark['Citizen'].sum()

non_citizens = pop - num_citizens

#total population is saved under "pop"

labels = 'Citizens', 'Non-Citizens'
sizes = [num_citizens, non_citizens]
explode = (.1, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.show()


# What about Washington County alone

# In[30]:


Wash = pd.DataFrame(df_ark.loc[5143])
Wash


# In[53]:


labels = 'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific'
sizes = [Wash.get_value('Hispanic', 5143), Wash.get_value('White', 5143),
         Wash.get_value('Black', 5143), Wash.get_value('Native', 5143),
         Wash.get_value('Asian', 5143), Wash.get_value('Pacific', 5143)]
f, ax = plt.subplots(figsize=(10,6))
explode = (0, 0.1, 0, 0, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.show()


# What types of jobs do people work in?

# In[33]:


labels = 'Professional', 'Service', 'Office', 'Construction', 'Production'
sizes = [Wash.get_value('Professional', 5143), Wash.get_value('Service', 5143),
         Wash.get_value('Office', 5143), Wash.get_value('Construction', 5143),
         Wash.get_value('Production', 5143)]
explode = (0.1, 0, 0, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.show()


# Employed & Unemployed

# In[34]:


#Washington County labor force is estimated at 147548
#https://www.census.gov/quickfacts/fact/table/washingtoncountyarkansas,AR,US/PST045216

num_unemployed = 0.062 * 147548
num_unemployed = round(num_unemployed)

labels = 'Employed', 'Unemployed'
sizes = [Wash.get_value('Employed', 5143), num_unemployed]
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.show()


# What sectors to people work in?

# In[35]:


labels = 'Public Work', 'Private Work', 'Self-Employed', 'Family Work'
sizes = [Wash.get_value('PublicWork', 5143), Wash.get_value('PrivateWork', 5143),
         Wash.get_value('SelfEmployed', 5143), Wash.get_value('FamilyWork', 5143)]
explode = (0, 0.1, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.show()


# # Feature correlations

# In[36]:


#correlation matrix

corrmat = df_ark.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True)


# For whatever reason, the colors on this plot are inverted.

# Lets focus in on income, poverty, and employment, & unemployment.

# In[42]:


k = 15
cols = corrmat.nlargest(k, 'Income')['Income'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df_ark[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size':8}, yticklabels=cols.values,
                xticklabels=cols.values)
plt.show()


# Features that are included in the scatterplot matrix have a correlation coefficient greater than 0.30.

# In[43]:


#Income linear relationships

sns.set()
cols = ['Income', 'IncomePerCap', 'Employed', 'Men', 'TotalPop', 'Citizen',
       'Women', 'PrivateWork', 'Professional', 'Office', 'Asian', 'Drive']
sns.pairplot(df_ark[cols], size=2.5)
plt.show()


# Some of the scatterplot matricies are a bit hard to see, it is possible to right-click and view those in another browser window.

# In[45]:


#Poverty

k = 7
cols = corrmat.nlargest(k, 'Poverty')['Poverty'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df_ark[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size':10}, yticklabels=cols.values,
                xticklabels=cols.values)
plt.show()


# In[46]:


#poverty liner relationships

sns.set()
cols = ['Poverty', 'ChildPoverty', 'Unemployment', 'Black', 'PublicWork', 'OtherTransp', 'Service']
sns.pairplot(df_ark[cols], size=2.5)
plt.show()


# In[48]:


# Employment

k = 15
cols = corrmat.nlargest(k, 'Employed')['Employed'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df_ark[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size':8}, yticklabels=cols.values,
                xticklabels=cols.values)
plt.show()


# In[49]:


#employment linear relationships

sns.set()
cols = ['Employed', 'TotalPop', 'Men', 'Women', 'Citizen', 'IncomePerCap',
       'Professional', 'Income', 'Asian', 'Office', 'PrivateWork', 'Pacific']
sns.pairplot(df_ark[cols], size=2.5)
plt.show()


# In[50]:


#unemployment

k = 7
cols = corrmat.nlargest(k, 'Unemployment')['Unemployment'].index
cm = np.corrcoef(df_ark[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size':10}, yticklabels=cols.values,
                xticklabels=cols.values)
plt.show()


# In[51]:


#unemployment linear relationships

sns.set()
cols = ['Unemployment', 'Black', 'Poverty', 'PublicWork', 'ChildPoverty', 'OtherTransp',
       'Service']
sns.pairplot(df_ark[cols], size=2.5)
plt.show()


# In[ ]:




