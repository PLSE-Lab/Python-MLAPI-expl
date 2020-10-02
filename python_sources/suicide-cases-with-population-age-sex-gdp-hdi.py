#!/usr/bin/env python
# coding: utf-8

# __Context__: HealthStats provides key health, nutrition and population statistics gathered from a variety of international sources. Themes include population dynamics, nutrition, reproductive health, health financing, medical resources and usage, immunization, infectious diseases, HIV/AIDS, DALY, population projections and lending. HealthStats also includes health, nutrition and population statistics by wealth quintiles.
# 
# 
# __Content__: This dataset includes 345 indicators, such as immunization rates, malnutrition prevalence, and vitamin A supplementation rates across 263 countries around the world. Data was collected on a yearly basis from 1960-2016.
# 
# __data__: 
# 1. country	
# 2. year	
# 3. sex	
# 4. age	
# 5. suicides_no	
# 6. population	
# 7. suicides/100k pop	
# 8. country-year	
# 9. HDI for year	
# 10. gdp_for_year
# 11. gdp_per_capita	
# 12. generation

# #### Insight of this kernal

# 1. Russia, United States, Japan, France and Ukraine are having highest number od suicidal cases
# 2. The people at age of 35-54 years suicide most. However, in the suicide per 100k people, the people at ages over 75 suicide most.
# 3. Men are more suicidal than women.
# 4. As the GDP per Capita increases the rate of suicides decreases
# 5. Boomers generation have highest number of suicides followed by Silent, Generation X
# 6. Man in age group of 35-54 are more suicidal
# 7. Female are less suicial in every generation.

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import norm
from scipy import stats
from sklearn import preprocessing

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read dataset

# In[ ]:


#data = pd.read_csv('/media/vishwadeepg/New Volume/Work/0. Gauty/Kernal/suicide_rates_overview/master.csv')
data = pd.read_csv('../input/master.csv')


# ### Variables Indetifications

# In[ ]:


#Size of datasets
print("Size of dataset  (Rows, Columns): ",data.shape)


# In[ ]:


# data snapshot
data.head()


# In[ ]:


data.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides_avg', 'country_year', 
                'HDI_for_year', 'gdp_for_year', 'gdp_per_capita', 'generation']


# In[ ]:


#general information of data
print("Data info: ",data.info())


# In[ ]:


print("Datatypes of dataset are:")
print(data.dtypes.value_counts())


# In[ ]:


data.describe()


# ### Missing Values

# In[ ]:


def missing_check(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 
    #print("Missing check:",missing_data )
    return missing_data


# In[ ]:


missing_check(data)


# # Univariate Analysis

# ### country

# In[ ]:


descending_order = data['country'].value_counts().sort_values(ascending=True).index
figure = plt.figure(figsize=(15,30))
ax = sns.countplot(y=data['country'], data=data, order=descending_order)


# ### sex

# In[ ]:


ax = sns.countplot(x=data['sex'], data=data)


# ### Year

# In[ ]:


descending_order = data['year'].value_counts().sort_values(ascending=True).index
figure = plt.figure(figsize=(10,15))
ax = sns.countplot(y=data['year'], data=data, order=descending_order)


# ### Age

# In[ ]:


figure = plt.figure(figsize=(10,8))
ax = sns.countplot(y=data['age'], data=data)


# ### population

# In[ ]:


data.population.hist(figsize=[10,5],bins=50, xlabelsize=10, ylabelsize=10)


# ### Avg. Suicides

# In[ ]:


data.suicides_avg.hist(figsize=[10,5],bins=50, xlabelsize=10, ylabelsize=10)


# ### GDP per Capita

# In[ ]:


data.gdp_per_capita.hist(figsize=[10,5],bins=50, xlabelsize=10, ylabelsize=10)


# ### gdp_for_year

# In[ ]:


#data.gdp_for_year.hist(figsize=[10,5], xlabelsize=10, ylabelsize=10)


# ### HDI_for_year

# In[ ]:


data.HDI_for_year = data.HDI_for_year.fillna(0)
data.HDI_for_year.hist(figsize=[10,5],bins=50, xlabelsize=10, ylabelsize=10)


# ### generation

# In[ ]:


figure = plt.figure(figsize=(10,8))
ax = sns.countplot(y=data['generation'], data=data)


# # Bivariate Analysis

# ### Number of Suicides by Country: top 30

# In[ ]:


figure = plt.figure(figsize=(15,8))
data.groupby(by=['country'])['suicides_no'].sum().reset_index().sort_values(['suicides_no'],
                    ascending=True).tail(30).plot(x='country',y='suicides_no',kind='bar', figsize=(15,8))


# ### Number of Suicides by Country: bottom 30

# In[ ]:


figure = plt.figure(figsize=(10,15))
data.groupby(by=['country'])['suicides_no'].sum().reset_index().sort_values(['suicides_no'],
                    ascending=True).head(30).plot(x='country',y='suicides_no',kind='bar', figsize=(15,8))


# ## Insight

# ### Number of Suicides by years

# In[ ]:


figure = plt.figure(figsize=(15,8))
plt.scatter(data.year, data.suicides_no, color='g')
plt.xlabel('year')
plt.ylabel('Number of suicides')
plt.show()


# ### Number of Suicides by Age

# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.barplot(x='age', y='suicides_no', data=data);


# ### Number of Suicides by SEX

# In[ ]:


plt.figure(figsize=(8,10))
ax = sns.barplot(x="sex", y="suicides_no", data=data)


# ### Number of Suicides by population

# In[ ]:


figure = plt.figure(figsize=(20,10))
ax = sns.regplot(x=data['population'],y='suicides_no', data=data, color='m')


# ### Number of Suicides vs Avg Number of Suicides 

# In[ ]:


figure = plt.figure(figsize=(20,10))
ax = sns.regplot(x=data['suicides_avg'],y='suicides_no', data=data, color='r')


# ### Number of Suicides by gdp_for_year

# In[ ]:


"""
figure = plt.figure(figsize=(20,10))
plt.scatter(data.gdp_for_year, data.suicides_no, color='r')
plt.xlabel('suicides_avg')
plt.ylabel('Number of suicides')
plt.show()
"""


# ### Number of Suicides by gdp_per_capita

# In[ ]:


figure = plt.figure(figsize=(20,10))
ax = sns.regplot(x=data['gdp_per_capita'],y='suicides_no', data=data)


# ### Number of Suicides by HDI_for_year

# In[ ]:


df = data[data['HDI_for_year']>0]
figure = plt.figure(figsize=(20,10))
ax = sns.regplot(x=df['HDI_for_year'],y='suicides_no', data=df)


# ### suicides/100k_pop and GDP per capita

# In[ ]:


figure = plt.figure(figsize=(20,10))
data_scaled = data.loc[:,['gdp_per_capita','suicides_avg']]
data_scaled = (data_scaled - data_scaled.mean()) / data_scaled.std()
sns.scatterplot(data=data_scaled,x='gdp_per_capita',y='suicides_avg', color='b')


# ### Correlation between suicides/100k_pop and HDI

# In[ ]:


figure = plt.figure(figsize=(20,10))
data_scaled = df.loc[:,['HDI_for_year','suicides_avg']]
data_scaled = (data_scaled - data_scaled.mean()) / data_scaled.std()
sns.scatterplot(data=data_scaled,x='HDI_for_year',y='suicides_avg',color='g')


# ### Number of Suicides by generation

# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.barplot(x='generation', y='suicides_no', data=data);


# ### Number of suicides by sex and age

# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(data=data,x='sex',y='suicides_no',hue='age')


# ### Number of suicides by Sex and year: year < 2004

# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(data=data[data['year']<2004],x='sex',y='suicides_no',hue='year')


# ### Number of suicides by Sex and year: year > 2004

# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(data=data[data['year']>2004],x='sex',y='suicides_no',hue='year')


# ### Number of suicides by sex and generation

# In[ ]:


plt.figure(figsize=(20,10))
ax = sns.barplot(x="generation", y="suicides_no", hue="sex", data=data)


# ### Number of suicides by Sex and year: year < 2004

# ##### year < 2004

# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(data=data[data['year']<2004],x='year',y='suicides_no',hue='age')


# #### year > 2003

# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(data=data[data['year']>2003],x='year',y='suicides_no',hue='age')


# In[ ]:


plt.figure(figsize=(20,10))
ax = sns.barplot(x="generation", y="suicides_no", hue="age", data=data)


# In[ ]:


data.head()


# In[ ]:


le = preprocessing.LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['generation'] = le.fit_transform(data['generation'])
data['age'] = le.fit_transform(data['age'])
data['country'] = le.fit_transform(data['country'])


# In[ ]:


df1 = data[['country', 'sex', 'age',  'population','suicides_avg', 'HDI_for_year', 'gdp_for_year',
            'gdp_per_capita', 'generation', 'suicides_no']]


# ### Correlation Analysis

# In[ ]:


data_corr = df1.corr()['suicides_no'][:-1] # -1 because the latest row is Target
golden_features_list = data_corr.sort_values(ascending=False)
golden_features_list


# In[ ]:


corr = df1.corr() 
plt.figure(figsize=(12, 10))

sns.heatmap(corr, 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# ### Thank You

# In[ ]:




