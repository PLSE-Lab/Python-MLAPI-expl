#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Importing important Libraries..
# 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# **Importing dataset..**

# In[ ]:


data=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv")


# In[ ]:


data.head()


# **Features of data****

# In[ ]:


print(data.columns)


# **Shape of data..**

# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().values.any()


# **Checking Count of Missing Values of all features**

# In[ ]:


null=data.isnull().sum()
null


# In[ ]:


fig=plt.figure(figsize=(20,4))
null.plot(kind='bar',color='green')
plt.title('List of columns and there Missing Values Count')


# **Removing Features that are having above 60% of Null Values..**

# In[ ]:


percent_null_values=null/len(data)*100
percent_null_values


# In[ ]:


lis=percent_null_values
for i in range(len(lis)):
    if lis[i]>60:
        del data[lis.index[i]]


# In[ ]:


data.shape


# In[ ]:


fig1=plt.figure(figsize=(20,4))
data.isnull().sum().plot(kind='bar',color='red')
plt.title('List of Features after removing Missing Values above 60%')


# In[ ]:


print(data.iloc[:,0:10].info())
print(data.iloc[:,0:10].nunique())


# **Data Visualization..**

# In[ ]:


#Source column
data.Source.value_counts(dropna=False)


# In[ ]:


sns.countplot(data['Source'])


# From above data analysis we can conclude that the sources of accidents are mainly from MapQuest.. 

# In[ ]:


#TMC(Traffic Message Channel)
data.TMC.value_counts(dropna=False)


# In[ ]:


plt.figure(figsize=(10,4))
data.TMC.value_counts(dropna=False).plot(kind='bar',color='orange')


# From above count plot the description of event is more from channel 201 and also we can see that there are missing values NA..

# In[ ]:


#Severity 
data.Severity.value_counts(dropna=False)


# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(data['Severity'])


# From above plot we observed that the severity of the accident is less that is delay of traffic is less

# In[ ]:


print(data.Start_Time.head())
print(data.End_Time.head())


# Convert Start_Time and End_Time to datetypes

# In[ ]:


data['Start_Time']=pd.to_datetime(data['Start_Time'],errors='coerce')
data['Start_Time'].head()


# In[ ]:


data['End_Time']=pd.to_datetime(data['End_Time'],errors='coerce')
data['End_Time'].head()


# Extracting Year,Month,Day,Hour and Weekday from Start_time and End_time

# In[ ]:


data['SYear']=data['Start_Time'].dt.year
data['SMonth']=data['Start_Time'].dt.strftime('%b')
data['SDay']=data['Start_Time'].dt.day
data['SHour']=data['Start_Time'].dt.hour
data['SWeekday']=data['Start_Time'].dt.strftime('%a')


# In[ ]:


data['EYear']=data['End_Time'].dt.year
data['EMonth']=data['End_Time'].dt.strftime('%b')
data['EDay']=data['End_Time'].dt.day
data['EHour']=data['End_Time'].dt.hour
data['EWeekday']=data['End_Time'].dt.strftime('%a')


# Calculating time duration of accident in minutes and round them to nearest integer

# In[ ]:


td='Time_Duration(min)'
data[td]=round((data['End_Time']-data['Start_Time'])/np.timedelta64(1,'m'))
data[td].head(20)


# In[ ]:


data.info()


# Checking for negative time duration values..

# In[ ]:


data[td][data[td]<=0]


# Dropping Rows of negative time duration values 

# In[ ]:


neg_values=data[td]<=0

#set negative values with NAN
data[neg_values]=np.nan


# In[ ]:


data[neg_values]


# In[ ]:


data.dropna(subset=[td],axis=0,inplace=True)


# In[ ]:


data.shape


# In[ ]:


data['Time_Duration(min)'].nunique()


# In[ ]:


plt.figure(figsize=(16,8))
sns.boxplot(data['Time_Duration(min)'])


# From above Boxplot we see that there are many outliers..

# In[ ]:


data.Start_Lat.nunique()


# In[ ]:


##Start_Lat
plt.figure(figsize=(10,4))
plt.hist(data['Start_Lat'])


# In[ ]:


##Start_Lng
plt.figure(figsize=(10,4))
plt.hist(data['Start_Lng'])


# In[ ]:


#Distance in miles
data['Distance(mi)'].nunique()


# In[ ]:


plt.figure(figsize=(10,4))
plt.hist(data['Distance(mi)'])


# Above histogram tells that the length of the road extent affected by the accident mostly lies between 0 to 50 miles..

# In[ ]:


#Description
data['Description'].nunique()


# In[ ]:


data.isnull().sum()


# In[ ]:


#Street
data['Street'].nunique()


# In[ ]:


#Side
data['Side'].value_counts()


# In[ ]:


sns.countplot(data['Side'])


# Above countplot says that relative side of the street in address field is mostly Right(R)..

# In[ ]:


#City
data.City.nunique()


# In[ ]:


#Country
data['Country'].value_counts()


# In[ ]:


print(data['State'].nunique())
print(data['State'].value_counts())


# In[ ]:


plt.figure(figsize=(16,8))
sns.countplot(data['State'],palette='Set2')


# The State that was mostly in the Address Field is 'CA'. Therefore, the Number of accidents that took place in Califorina are high.

# In[ ]:


data.isnull().sum()


# In[ ]:


##Missing Values Imputation with Most frequently occuring values 
def fillna(col):
    col.fillna(col.value_counts().index[0],axis=0,inplace=True)
    return col
data=data.apply(lambda col: fillna(col))


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# Showing Correlation through Heat Map

# In[ ]:


plt.figure(figsize=(16,16))
sns.heatmap(data.corr(),annot=True,cmap='magma')

