#!/usr/bin/env python
# coding: utf-8

# Investigatng the data and exploratory data analysis
# 
# First installing all the libraries that will use in our application. Installing the libraries in the first part because the algorithm we use later and the analysis we make more clearly will be done. Furthermore, investigating the data, presented some visualization and analyzed some features. Lets write it. Importing necessary packages and libraries.

# Now we are uploading our data set using the variable "corona" in the pandas library.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


corona=pd.read_csv("../input/covid19-italy-province/covid19_italy_province.csv")


# In[ ]:


#after the loading the data. Next step is to view/see the top 10 rows of the loaded data set

corona.head()


# In[ ]:


#last 10 rows of loaded data set

corona.tail(10)


# Describe function is a function that allos analysis between the numeric values contained in the the dataset. Using this function count, mean,std, min,max,25%,50%,75%.

# In[ ]:


corona.describe()


# checking statistic summary and it will show only for numeric data not categorical 
# 

# In[ ]:


#information about each var

corona.info()


# In[ ]:


#we will be listing the columns of all the data.
#we will check all columns

corona.columns


# In[ ]:


corona.sample(frac=0.01)


# In[ ]:


#sample: random rows in the dataset
#useful for future analysis
corona.sample(5)


# In[ ]:


#next, how many rows an columns are there in the loaded data set

corona.shape


# In[ ]:


# and, will check null on all the data and if there is any null, getting the sum of all the null data's

corona.isna().sum()


# In[ ]:


#Removing duplicates if any

corona.duplicated().sum()
corona.drop_duplicates(inplace =True)


# In[ ]:


#count all the region name

corona['RegionName'].value_counts()


# In[ ]:


#total positive cases happened daily
df=corona.groupby('Date')['TotalPositiveCases'].sum()
df=df.reset_index()
df=df.sort_values('Date', ascending= True)
df.head(60)


# In[ ]:


#total positive cases in the region

df=corona.groupby('RegionName')['TotalPositiveCases'].sum()
df=df.reset_index()
df=df.sort_values('RegionName', ascending= True)
df.head(60)


# In[ ]:


#total positive cases in Italy
corona['TotalPositiveCases'].sum()


# **Data Visualization**
# 

# In[ ]:


#checking the null values via graph , where you can find yellow color lines means that column contains null values.

sns.heatmap(corona.isnull(), yticklabels= False)


# The yellow lines from the above graph indicates that the column ProvinceAbbrevation has null values

# In[ ]:


#plotting graph which region has maximum

sns.countplot(y=corona['RegionName'],).set_title('Regions affected overall')


# In[ ]:


#which country has most affected with corona
sns.countplot(x='Country',data=corona,hue='Country')


# In[ ]:


#which region code has highest affected in the country
sns.countplot(y='RegionCode', data=corona, hue='Country')


# In[ ]:


#which regioncode has highest affected
sns.countplot(y='RegionCode', data=corona, hue='RegionCode')


# Total confirmed positive cases

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(y='RegionCode', data=corona, hue='Date')


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(y='Country', data=corona, hue='Date')


# In[ ]:


plt.figure(figsize=(8,8))
sns.countplot(y='Country', data=corona, hue='TotalPositiveCases')


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(y='Date', data=corona, hue='TotalPositiveCases')


# In[ ]:


plt.figure(figsize=(10,5))
Confirmed_positive_cases=corona['Date'].value_counts().sort_index()
Confirmed_positive_cases.cumsum().plot(legend='accumulated')
Confirmed_positive_cases.plot(kind='bar',color='orange',legend='daily',grid=True)


# In[ ]:


# cases confirmed before number 250 positive cases confirmed

plt.figure(figsize=(8,2))
df= Confirmed_positive_cases[:corona[corona['SNo']==250]['Date'].values[0]]
df.cumsum().plot(legend='accumulated')
df.plot(kind='bar',color='orange',legend='daily',grid=True)


# In[ ]:


# cases confirmed after number 250 positive cases confirmed
plt.figure(figsize=(10,5))
df= Confirmed_positive_cases[corona[corona['SNo']==250]['Date'].values[0]:]
df.cumsum().plot(legend='accumulated')
df.plot(kind='bar',color='orange',legend='daily',grid=True)

