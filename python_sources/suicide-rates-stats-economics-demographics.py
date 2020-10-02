#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/master.csv")
data.head()


# # Checking null values in the dataframe

# In[ ]:


data.isnull().sum()


# In[ ]:


data.info()


# # Summary of the dataset

# In[ ]:


data.describe()


# ### 1. Dropping "HDI for year" because 19456 values are Null values
# ### 2. Dropping "country-year" to avoid redundancy of data

# In[ ]:


data.drop(["HDI for year",'country-year'],inplace=True,axis=1)


# In[ ]:


data.head(3)


# # Population of different countries

# In[ ]:


sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(8, 20))
ax = sns.barplot(data.population.groupby(data.country).count(),data.population.groupby(data.country).count().index)
plt.show()


# # Frequency of different generations

# In[ ]:


f,ax = plt.subplots(1,1,figsize=(15,4))
ax = sns.countplot(data.generation,palette='rainbow')
plt.show()


# # Number of suicides in different age groups

# ### Age category with maximum suicide number is 35-54 years

# In[ ]:


f,ax = plt.subplots(1,1,figsize=(15,4))
ax = sns.barplot(x = data.age.sort_values(),y = 'suicides_no',hue='sex',data=data,palette='rainbow')
plt.show()


# # Suicides each year since 2000

# ### There are least cases of suicides in year 2016 since the year 2000

# In[ ]:


f,ax = plt.subplots(1,1,figsize=(15,4))
ax = sns.barplot(x = data[data.year > 2000]['year'],y = 'suicides_no',data=data,palette='rainbow')
plt.show()


# # Distribution of suicides/100k pop

# In[ ]:


f,ax = plt.subplots(1,1,figsize=(15,4))
ax = sns.kdeplot(data['suicides/100k pop'])
plt.show()


# # Top 10 Countries with maximum number of suicides/100k pop

# In[ ]:


data_suicide_mean = data['suicides/100k pop'].groupby(data.country).mean().sort_values(ascending=False)
f,ax = plt.subplots(1,1,figsize=(15,4))
ax = sns.barplot(data_suicide_mean.head(10).index,data_suicide_mean.head(10),palette='coolwarm')


# # Change in number of suicides each year

# In[ ]:


data_time = data['suicides_no'].groupby(data.year).count()
data_time.plot(figsize=(20,10), linewidth=2, fontsize=15,color='purple')
plt.xlabel('Year', fontsize=15)
plt.ylabel('No of suicides',fontsize=15)
plt.show()


# # Change in gdp_per_capita per year

# In[ ]:


data_gdp = (data['gdp_per_capita ($)'].groupby(data.year)).sum()
data_gdp.plot(figsize=(20,10), linewidth=2, fontsize=15,color='red')
plt.xlabel('Year', fontsize=15)
plt.ylabel(' Total gdp_per_capita ($)',fontsize=15)
plt.show()


# # Top 10 countries with maximum number of suicides since 1985

# In[ ]:


data_suicide = data['suicides_no'].groupby(data.country).sum().sort_values(ascending=False)
f,ax = plt.subplots(1,1,figsize=(6,15))
ax = sns.barplot(data_suicide.head(10),data_suicide.head(10).index,palette='coolwarm')


# # Top 10 countries with least number of suicides since 1985

# In[ ]:


data_suicide = data['suicides_no'].groupby(data.country).sum().sort_values(ascending=False)
f,ax = plt.subplots(1,1,figsize=(6,15))
ax = sns.barplot(data_suicide.tail(10),data_suicide.tail(10).index,palette='coolwarm')


# # gdp_per_capita Vs suicides/100k pop

# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))
ax = sns.regplot(x='gdp_per_capita ($)', y='suicides/100k pop',data=data)
plt.show()


# In[ ]:



from mpl_toolkits.mplot3d import axes3d
f, ax = plt.subplots(figsize=(12, 4))
ax = f.add_subplot(111, projection='3d')
ax.scatter(data['gdp_per_capita ($)'], data.year, data['suicides/100k pop'], alpha=0.2, c="blue", edgecolors='none', s=30, label="people") 
plt.title('gdp_per_capita ($), year, suicides/100k pop')
plt.legend(loc=1)
plt.show()


# In[ ]:


f,ax = plt.subplots(1,1,figsize=(10,10))
ax = sns.heatmap(data.corr(),annot=True)
plt.show()


# # END

# In[ ]:




