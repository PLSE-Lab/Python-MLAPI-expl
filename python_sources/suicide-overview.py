#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('../input/master.csv')


# Hi!
# 
# This is my first Kernel in Kaggle. I hope this info its gonna be usefull for you.
# If you have some advices, dont hestiate to tell me, everything will be welcome.

# In[ ]:


#Lets see how the date looks like
df.head()


# In[ ]:


df.shape


# In[ ]:



df['country'].unique()


# In[ ]:



df['country'].nunique()


# There is a lot of countries in the dataset. We are gonna make first a general analyse before the data exploring in order to see how the data is distributed. 
# Then we should try to classify the date for a better analyse

# In[ ]:


df.dtypes


# In[ ]:


#I may use the gdp_for_year  so i want to convert this column in floats
df[' gdp_for_year ($) '] = df[' gdp_for_year ($) '].apply(lambda x: x.replace(',','')).astype(float)


# In[ ]:


df.dtypes


# In[ ]:


#I would like to know if there is some NaN values
df.isnull().any()


# In[ ]:


df['HDI for year'].isna().sum()


# In[ ]:


fig = plt.figure(figsize=(20,2))
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'ocean')


# In[ ]:


#As we see there is a plenity of them so it is a little bit useless in our analyse, Maybe later we are gonna use that to see in which countries there is more NaN values
df = df.drop(columns = 'HDI for year')


# In[ ]:


plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, linewidths = 0.3)
plt.title('Correlation of the dataset', size=16)
plt.show()


# We see the biggest correlation between suicides_no and population, population and gdp_for_year. This is something what we dont need to use now

# In[ ]:


#Let's see how the suicides per gender are distributed
dfSex=df.groupby(["sex"])["suicides_no"].sum().reset_index()
sns.barplot(x="sex", y="suicides_no", data=dfSex, palette="Blues_d")
plt.show()


# In[ ]:


#Let's see how the suicides per age are distributed
dfAge = df.groupby(['age'])['suicides_no'].sum().reset_index()
dfAge = dfAge.sort_values(by='suicides_no',ascending=False)
plt.subplots(figsize=(10,6))
sns.barplot(x='age', y='suicides_no', data=dfAge, palette = 'Blues_d')


# In[ ]:


#Let's see how the suicides per generation are distributed
dfGeneration = df.groupby(['generation'])['suicides_no'].sum().reset_index()
dfGeneration = dfGeneration.sort_values(by='suicides_no',ascending=False)
plt.subplots(figsize=(10,6))
sns.barplot(x='generation', y='suicides_no', data=dfGeneration, palette = 'Blues_d')


# So, we see that there are more man who have committed suicide.
# People between 35-54 committed more suicide and they should be from the Boomers Generation.
# 
# Let's see how it looks by country

# In[ ]:


dfCountry = df.groupby(['country'])['suicides_no'].sum().reset_index()
dfCountry = dfCountry.sort_values('suicides_no',ascending=False)
dfCountry = dfCountry.head(10)

plt.subplots(figsize=(15,6))
sns.barplot(x='country', y='suicides_no', data=dfCountry, palette = 'Blues_d')


# As wee see the countries that commit the most suicides are Russio, USA and Japan. It would be interesting to analys this data per year

# In[ ]:


array = ['Russian Federation', 'United States', 'Japan', 'France', 'Ukraine', 'Germany', 'Republic of Korea', 'Brazil', 'Poland', 'United Kingdom']
dfPeriod = df.loc[df['country'].isin(array)]
dfPeriod = dfPeriod.groupby(['country', 'year'])['suicides_no'].sum().unstack('country').plot(figsize=(20, 7))
dfPeriod.set_title('Top suicide countries', size=15, fontweight='bold')


# In[ ]:


dfSexPeriod =df.groupby(['sex', 'year'])['suicides_no'].sum().unstack('sex').plot(figsize=(20, 7))
dfSexPeriod.set_title('Suicide per Sex', size=15, fontweight='bold')


# In[ ]:


dfAgePeriod =df.groupby(['age', 'year'])['suicides_no'].sum().unstack('age').plot(figsize=(20, 10))
dfAgePeriod.set_title('Suicide per Age', size=15, fontweight='bold')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




