#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# <hr>
# Greetings everyone! In this notebook,we will study the dataset of crimes in Uk from year 2003 - 2018. This data is from [Recorded Crime Data at the Police Force Area Level dataset](https://www.kaggle.com/r3w0p4/recorded-crime-data-at-police-force-area-level) dataset.
# 

# ## **Why is dataset?**
# <hr>
# I choose this dataset because of its simplicity. And there is only data file and there are only few variable and observation.

# ## **What are about this dataset?**
# <hr>
# Recorded crime for the Police Force Areas of England and Wales. The data are rolling 12-month totals, with points at the end of each financial year between year ending March 2003 to March 2007 and at the end of each quarter from June 2007.
# The data are a single .csv file with comma-separated data. It has the following attributes:
# <font color=blue>
# *  12 months ending: the end of the financial year.
# *  PFA: the Police Force Area.
# *  Region: the region that the criminal offence took place.
# *  Offence: the name of the criminal offence.
# *  Rolling year total number of offences: the number of occurrences of a given offence in the last year.
# 

# <hr>
# ### **Table of Content**
# <font color = green>
# 1.  [Overview](#1)
# 1.  [Import libraries and reading Data](#2)
# 1.  [Data Analysis](#3)
# 1.  [Linear Model](#4)
# 1.  [Conclusion](#5)

# ## <span id='1'> </span> **1. Overview**

# ### Columns
# 1. Index
# 1. 12 month ending : 12 month data
# 1. PFA : Police Force Area 
# 1. Region : Particular Region of crime
# 1. Offence : crime name
# 1. Rolling year total number of offences

# ## <span id='2'></span> 2.Import libraries and reading Data 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


dataset = pd.read_csv('../input/rec-crime-pfa.csv')


# ## <span id='3'></span> 3.Data Analysis

# In[ ]:


dataset.head()


# In[ ]:


dataset.tail()


# In[ ]:


print('There are '+ str(dataset.shape[0])+ ' rows and '+ str(dataset.shape[1]) +' columns')


# In[ ]:


dataset.info()


# In[ ]:


data = {'unique_values' : dataset.nunique(),
        'na_values' : dataset.isna().sum(),
        'data_types' : dataset.dtypes}
pd.DataFrame(data)


# Convert column type to datatime

# In[ ]:


dataset['12 months ending'] = pd.to_datetime(dataset['12 months ending'])

#Get year, month, day
dataset['Year'] = dataset['12 months ending'].dt.year
dataset['Month'] = dataset['12 months ending'].dt.month
dataset['Day'] = dataset['12 months ending'].dt.day


# In[ ]:


dataset.head()


# Drop [12 months ending] column

# In[ ]:


dataset.drop(['12 months ending'], inplace=True, axis=1)
dataset.head()


# In[ ]:


dataset.rename(inplace=True, columns={'PFA':'pfa', 'Region':'region', 'Offence':'offence', 'Rolling year total number of offences':'total', 'Year':'year', 'Month':'month', 'Day':'day'})


# In[ ]:


dataset.head()


# Unique pfa, region, and offence

# In[ ]:


# Making data more simple
dataset.loc[dataset['offence'] == 'Domestic burglary', 'offence'] = 'Burglary'
dataset.loc[dataset['offence'] == 'Non-domestic burglary', 'offence'] = 'Burglary'
dataset.loc[dataset['offence'] == 'Non-residential burglary', 'offence'] = 'Burglary'
dataset.loc[dataset['offence'] == 'Residential burglary', 'offence'] = 'Burglary'

dataset.loc[dataset['offence'] == 'Bicycle theft', 'offence'] = 'Theft'
dataset.loc[dataset['offence'] == 'Shoplifting', 'offence'] = 'Theft'
dataset.loc[dataset['offence'] == 'Theft from the person', 'offence'] = 'Theft'
dataset.loc[dataset['offence'] == 'All other theft offences', 'offence'] = 'Theft'

dataset.loc[dataset['offence'] == 'Violence with injury', 'offence'] = 'Violence'
dataset.loc[dataset['offence'] == 'Violence without injury', 'offence'] = 'Violence'


# In[ ]:


{
    'unique_pfa': dataset['pfa'].unique(),
    'unique_region': dataset['region'].unique(), 
    'unique_offence': dataset['offence'].unique()
}


# In[ ]:


## Crime based on year
plt.figure(figsize=(15,6))
ax = sns.barplot(x='year', y='total', data=dataset)
plt.xticks(rotation=45,fontsize=10)
plt.show()


# In[ ]:


## Crime based on month 
plt.figure(figsize=(15,6))
ax = sns.barplot(x='month', y='total', data=dataset)
plt.show()


# In[ ]:


## Crime based on region
plt.figure(figsize=(16,5))
ax = sns.barplot(x='region',y='total', data=dataset)
plt.xticks(rotation=70)
plt.show()


# In[ ]:


dataset1 = dataset[dataset['year']>2014]
dataset2=dataset1.sort_values('total', ascending=False).head(10)
dataset2.sort_values('year')


# We can say that from year 2015 to 2018 offence CIFAS and Action Fraud is becoming popular. And they usually attack on the 31st or 30th day of month.
# 

# ### Region East EDA

# In[ ]:


east = dataset[dataset['region'] == 'East']
east.head(10)


# In[ ]:


# East Distribution based on year
sns.barplot(x='year', y='total', data=east)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#Most Popular offence in East
sns.set()
sns.jointplot(x='year',y='total',data=east)
plt.show()


# In[ ]:


#Popular crime in east
popular = east.groupby('offence')['total'].count()
popular.head()


# In[ ]:


#increase of offence
sns.lineplot(x='offence', y='total', data=dataset)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# rise in CIFAS and action fraud offence
offen= dataset[dataset['offence']=='CIFAS']
offen1= dataset[dataset['offence']=='Action Fraud']
sns.lineplot(x='year',y='total', data=offen)
sns.lineplot(x='year',y='total', data=offen1)
plt.xticks(rotation=90)
plt.legend('upper center')
plt.show()


# In[ ]:


label_weapon = 'Possession of weapons offences'
df_weapon = dataset.loc[dataset['offence'] == label_weapon]
labels_weapon_high = ['Metropolitan Police', 'Greater Manchester', 'West Midlands', 'West Yorkshire']
df_weapon_high = df_weapon.loc[df_weapon['pfa'].isin(labels_weapon_high)]

sns.lineplot(data=df_weapon_high, x='year', y='total', hue='pfa')
plt.show()


# ## TO be Continue!

# In[ ]:




