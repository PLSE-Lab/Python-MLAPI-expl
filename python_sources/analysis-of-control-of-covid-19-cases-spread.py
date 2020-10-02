#!/usr/bin/env python
# coding: utf-8

# This notebook is to explore the efficacy of testing

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Datasets pertaining to testing - one from World Bank and 4 from Our World in Data

# In[ ]:


file0 = '/kaggle/input/uncover/UNCOVER/world_bank/total-covid-19-tests-performed-by-country.csv'
file1 = '/kaggle/input/uncover/UNCOVER/our_world_in_data/per-million-people-tests-conducted-vs-total-confirmed-cases-of-covid-19.csv'
file2 = '/kaggle/input/uncover/UNCOVER/our_world_in_data/total-covid-19-tests-performed-per-million-people.csv'
file3 = '/kaggle/input/uncover/UNCOVER/our_world_in_data/tests-conducted-vs-total-confirmed-cases-of-covid-19.csv'
file4 = '/kaggle/input/uncover/UNCOVER/our_world_in_data/total-covid-19-tests-performed-by-country.csv'

df = pd.read_csv(file0)
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)


# df and df4 have the same data from different sources, df from World Bank and df4 from Our World in Data

# In[ ]:


df.head(3)


# In[ ]:


df4.head(3)


# Delete NaN rows

# In[ ]:


df = df.dropna()


# In[ ]:


fig, ax = plt.subplots(figsize=(20,6))
df[['entity', 'total_covid_19_tests']].groupby(by='entity').sum().sort_values(by='total_covid_19_tests', 
        ascending=False).head(20).plot(kind='bar', ax=ax)
plt.title('Covid Testing by Country - source:' + file0)
plt.xlabel('Country')
plt.ylabel('Number of tests performed')


# South Korea has conducted the most number of tests. Is the significant testing a key factor in controlling the pandemic?

# In[ ]:


df4.head(3)


# In[ ]:


fig, ax = plt.subplots(figsize=(20,6))

df4[['entity', 'total_covid_19_tests']].groupby('entity').sum().sort_values(by='total_covid_19_tests', 
                        ascending=False).head(20).plot(kind='bar', ax=ax)

plt.title('Covid Testing by Country - source:' + file4)
plt.xlabel('Country')
plt.ylabel('Number of tests performed')


# South Korea and China - Guangdong have conducted the most tests. For the most part, both datasets show the same entities in the top 20. 

# In[ ]:


df1.head(3)


# Remove rows with Null codes

# In[ ]:


f1 = df1['code'].isna()
df1 = df1.loc[~f1, :]


# In[ ]:


fig, ax = plt.subplots(figsize=(20,4))

cols = ['entity', 'total_covid_19_tests_per_million_people', 
     'total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']

data = df1[cols].groupby('entity').sum()
data.loc[data[cols[1]]>0,:].sort_values(by=cols[2], ascending=False).head(20).plot(kind='bar', ax=ax)

plt.title('Covid Testing by Country - source:' + file1)
plt.xlabel('Country')
plt.ylabel('Number of tests performed and cases per million people')


# This dataset will be skewed towards the countries that have a low population as the variables are per million of population. This shows that there are many small countries such as Iceland, Norway, South Korea that have performed aggressive testing, hence their testing per million is greater than their cases per million. Although one must ask whether Bahrain is overtesting?

# In[ ]:


df2.head(3)


# In[ ]:


f1 = df2['code'].isna()
df2.loc[f1, :]


# In[ ]:


fig, ax = plt.subplots(figsize=(20,4))
cols = ['entity', 'total_covid_19_tests_per_million_people']
df2[cols].groupby('entity').sum().sort_values(by=cols[1], ascending=False).head(20).plot(kind='bar', ax=ax)

plt.title('Covid Testing by Country - source:' + file2)
plt.xlabel('Country')
plt.ylabel('Number of tests performed per million people')


# In[ ]:


df3.head(3)


# In[ ]:


f1 = df3['code'].isna()
df3 = df3.loc[~f1, :]


# In[ ]:


cols = ['entity', 'total_covid_19_tests','total_confirmed_cases_of_covid_19_cases']

fig,ax = plt.subplots(figsize=(20,4))
data = df3[cols].groupby('entity').sum()
data.loc[data['total_covid_19_tests']>0,:].sort_values(by=cols[2], 
                                                      ascending=False).head(30).plot(kind='bar', ax=ax)

plt.title('Covid Testing by Country - source:' + file3)
plt.xlabel('Country')
plt.ylabel('Number of tests and cases')

