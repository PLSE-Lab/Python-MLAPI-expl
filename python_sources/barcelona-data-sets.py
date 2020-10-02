#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


population=pd.read_csv('../input/barcelona-data-sets/population.csv')
population


# In[ ]:


population.info()


# In[ ]:


population.describe()


# In[ ]:


population.isna().any().sum()


# In[ ]:


pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


sns.pairplot(population)


# # **population over years 2013-2017**

# In[ ]:


years=population.Year.unique()
population_year={}
for year in years:
    population_year[year]=population.Number.loc[population.Year==year].sum()
   
print('population = {}'.format(population_year))
    


# In[ ]:


plt.figure(figsize=(10,6))
plt.title('Population in Barcelona from 2013 to 2017')

plt.plot(list(population_year.keys()), list(population_year.values()))

plt.show()


# In[ ]:


years=population.Year.unique()
population_year={}
for year in years:
    population_year[year]=population.Number.loc[population.Year==year].sum()


population_year=pd.Series(population_year) 
print(population_year)


plt.figure(figsize=(10,6))
plt.title('Population in Barcelona from 2013 to 2017')
sns.barplot(x=population_year.index, y=population_year.values)    


# In[ ]:


population_in_2017=population.Number.loc[population.Year==2017].sum()
print('population in 2017= {}'.format(population_in_2017))
population_in_2016=population.Number.loc[population.Year==2016].sum()
print('population in 2016= {}'.format(population_in_2016))

population_in_2015=population.Number.loc[population.Year==2015].sum()
print('population in 2015= {}'.format(population_in_2015))

population_in_2014=population.Number.loc[population.Year==2014].sum()
print('population in 2014= {}'.format(population_in_2014))

population_in_2013=population.Number.loc[population.Year==2013].sum()
print('population in 2013= {}'.format(population_in_2013))


# In[ ]:


population.groupby(['Year', 'Gender'])['Number'].sum().unstack().plot.bar(title='Population per year and gender')
fig=plt.gcf()
fig.set_size_inches(18,6)


# In[ ]:


population_by_District_Name=population.groupby(['Year', 'District.Name'])['Number'].sum()
print(population_by_District_Name)
population_by_District_Name.unstack().plot.bar(title='Population per year and District')
fig=plt.gcf()
fig.set_size_inches(18,6)


# In[ ]:


population_by_District_Name_2017=population.loc[population.Year==2017].groupby(['District.Name' ])['Number'].sum()
print(population_by_District_Name_2017)
population_by_District_Name_2017.sort_values().plot.bar(title='Population in 2017 by District')
fig=plt.gcf()
fig.set_size_inches(18,6)


# In[ ]:


population_by_District_Name=population_by_District_Name.reset_index()
population_by_District_Name.sort_values(by=['Number', 'District.Name'])


# #  overview
# 
# * During the five years, there has been no significant change in the population of the city
# * Women are more numerous than men
# * The most populated district is  Eixample and the most densely populated district is Les Corts

# In[ ]:


population.Age.unique()


# In[ ]:


population.Age.dtype


# In[ ]:


for year in population.Year.unique():
    population_age=population.loc[population.Year==year].groupby(['Age','Year'])['Number'].sum().unstack().plot.bar(title='Population per year and ages')
    fig=plt.gcf()
    fig.set_size_inches(18,6)
    population_age


# In[ ]:


age=population.Age.unique()
population_age={}
for ag in age:
    population_age[ag]=population.Number.loc[population.Age==ag].sum()
   
print('population = {}'.format(population_age))


# #  no.of childern

# In[ ]:


def get_child_No(year_no):
    n0_chil=population.Number.loc[(population.Age<'20-24')&(population.Year==year_no)].sum()
    return n0_chil


# In[ ]:


print(get_child_No(2017))


# In[ ]:


def get_Elderly_No(year_no):
    n0_Elderly=population.Number.loc[(population.Age>'55-59')&(population.Year==year_no)].sum()
    return n0_Elderly


# In[ ]:


get_Elderly_No(2017)


# In[ ]:


n0_chilf=population.Number.loc[(population.Age<'20-24')& (population.Gender=="Female")&(population.Year==2017)].sum()
n0_chilf


# In[ ]:


n0_chilm=population.Number.loc[(population.Age<'20-24')& (population.Gender=="Male")&(population.Year==2017)].sum()
n0_chilm


# In[ ]:


percentage_of_child_in2017=(get_child_No(2017)/population_in_2017)*100
print('percentage of child in 2017 = {}  %'.format(percentage_of_child_in2017))


# In[ ]:


percentage_of_Elderly_in2017=(get_Elderly_No(2017)/population_in_2017)*100
print('percentage of Elderly in 2017 = {}  %'.format(percentage_of_Elderly_in2017))


# In[ ]:


Elderly_Nom=population.Number.loc[(population.Age>'55-59')&(population.Gender=="Male")&(population.Year==2017)].sum()
Elderly_Nom


# In[ ]:


Elderly_Nof=population.Number.loc[(population.Age>'55-59')&(population.Gender=="Female")&(population.Year==2017)].sum()
Elderly_Nof


# In[ ]:


for year in population.Year.unique():
    population_age=population.loc[population.Year==year].groupby(['Age','Year','Gender'])['Number'].sum().unstack().plot.bar(title='Population per year and ages')
    fig=plt.gcf()
    fig.set_size_inches(18,6)
    population_age


# In[ ]:


birth_df=pd.read_csv('../input/barcelona-data-sets/births.csv')
birth_df.info()


# In[ ]:


birth_df.describe()


# In[ ]:


birth_df


# In[ ]:


birth_df.groupby(['Year' ,'Gender'])['Number'].sum().unstack().plot.bar()
fig=plt.gcf()
fig.set_size_inches(18,6)


# In[ ]:


years=birth_df.Year.unique()
birth_year={}
for year in years:
    birth_year[year]=birth_df.Number.loc[birth_df.Year==year].sum()
   
print('population = {}'.format(birth_year))


# In[ ]:


plt.figure(figsize=(10,6))
plt.title('Birth in Barcelona from 2013 to 2017')
plt.plot(list(birth_year.keys()),list(birth_year.values()))


# In[ ]:


death_df=pd.read_csv('../input/barcelona-data-sets/deaths.csv')
death_df.head()


# In[ ]:


years=death_df.Year.unique()
birth_year={}
for year in years:
    birth_year[year]=death_df.Number.loc[death_df.Year==year].sum()
   
print('population = {}'.format(birth_year))
plt.figure(figsize=(10,6))
plt.title('Death in Barcelona from 2013 to 2017')
plt.plot(list(birth_year.keys()),list(birth_year.values()))


# In[ ]:


plt.figure(figsize=(20,10))
sns.scatterplot(x=death_df['Age'], y=death_df['Number'],hue=death_df['Year'])


# In[ ]:


death_df.groupby(['Year' ,'Age'])['Number'].sum().unstack().plot.bar()
fig=plt.gcf()
fig.set_size_inches(18,6)


# In[ ]:


death_df.loc[death_df.Year==2017].groupby(['Year' ,'Age'])['Number'].sum().unstack().plot.bar()
fig=plt.gcf()
fig.set_size_inches(10,10)


# In[ ]:


r={}

for age in death_df.Age.unique():
    r[age]=death_df.Number.loc[(death_df.Age==age)&(death_df.Year==2017)].sum()
r


# In[ ]:


plt.figure(figsize=(15,6))
plt.title('Death in Barcelona in 2017')
plt.plot(list(r.keys()),list(r.values()))


# In[ ]:


death=pd.DataFrame(death_df.groupby(['Age'])['Number'].sum())
death

