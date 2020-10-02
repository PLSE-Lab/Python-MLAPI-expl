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


import matplotlib.pyplot as plt
import seaborn as sns


# The data used here is downloaded from https://aqicn.org/data-platform/register/ 
# There are plenty of data on AQI available corresponding to each city.
# I have used the AQI database of Manali which is a town in the outskirts of chennai, where plenty of factories and oil refineries are present.

# In[ ]:


df=pd.read_csv(r"/kaggle/input/aqi-of-manalichennai/manali.csv",parse_dates=True)


# # Cleaning the data
# 
# Like any data science project the first step is to take a good look at it and clean it.
# 

# In[ ]:


print(df.columns)


# Here the column names are havingg an additional space in their name.
# To avoid errors while using the columns name I am renaming the columns.

# In[ ]:


df.columns=['date','pm25','pm10','o3','no2','so2','co']
print(df.columns)
print(df.dtypes)


# The datatypes are supposted to be numerical, and float in this case, as te values are decimal.
# So I am converting it to the corresponding datatype.

# In[ ]:


df["o3"] =pd.to_numeric(df["o3"],errors="coerce")
df["no2"] =pd.to_numeric(df["no2"],errors="coerce")
df["so2"] =pd.to_numeric(df["so2"],errors="coerce")
df['co'] =pd.to_numeric(df["co"],errors="coerce")
df['pm25'] =pd.to_numeric(df["pm25"],errors="coerce")
df['pm10'] =pd.to_numeric(df["pm10"],errors="coerce")

print(df.dtypes)


# Running a quick statistics on the data we have.

# In[ ]:


df.describe()


# Now coming to the most important part of Data cleaning
# # Handling of missing values
# First I will find the number of missing values

# In[ ]:


df.isnull().sum()


# Here I check the variability and uniquness of a specific column to understand the data.

# In[ ]:


print(df.pm10.value_counts())


# In[ ]:


import missingno as msno


# # Visualising missing values
# I am importing this library to understand and analyse the missing values more easily, by visualising it
# 

# In[ ]:


msno.matrix(df)


# The index is numerical by default, I am changing the numerical index into DateTimeIndex as our data set is time series.

# In[ ]:


df.set_index("date")


# Sorting the index in ascending order.

# In[ ]:


df.sort_index(axis = 0,ascending=True,inplace=True)


# Visualising the missing data as a graph

# In[ ]:


msno.bar(df)


# In the column  pm10 and o3 more than 80% of the data is missing.
# We can try imputing the missing values with possible values, but since we have only less than 20% data available in these columns, its better to drop these columns.

# In[ ]:


df.columns


# Dropping the columns (axis=1) and making the changes in the original dataframe(inplace=True)

# In[ ]:


df.drop(df.columns[[2,3]],axis=1,inplace=True)


# In[ ]:


df.columns


# 

# In[ ]:


plt.figure(figsize=(16,8))
msno.bar(df)


# Now we have only less null values

# In[ ]:


df.fillna(method="ffill",inplace=True)


# I am using the method ffill to fill the missing values
# Ffill uses the previous known value and fills the next missing value.
# I prefer this method to other imputations because mostly AQI values from one day to the next dont have much difference.

# In[ ]:


df.isnull().isnull().sum()


# Now we dont have any null values.

# In[ ]:


df['date']=pd.to_datetime(df['date'])


# The datetime column is to be changed to datetime format for using it efficiently

# In[ ]:


df['date'].min()


# In[ ]:


df['date'].max()


# # DATA Visualisation

# In[ ]:


import seaborn as sns


# I am plotting the concentration of the gaseous elements present in the air, in a time series x axis.

# In[ ]:


fig, axes = plt.subplots(figsize=(20,6),nrows=4)
df.reset_index().plot(x="date",y="pm25",ax=axes[0],color='coral')
df.reset_index().plot(x="date",y="co",ax=axes[1],color='goldenrod')
df.reset_index().plot (x="date",y="no2",ax=axes[2],color='lightsteelblue')
df.reset_index().plot( x="date",y="so2",ax=axes[3],color='yellowgreen')


# Though we couldnt say a lot from this, we can see the similarities in pattern of so2 and no2.This can be because of a single source emmission, such as Ammonia fertiliser manufacturing plant in manali.

# In[ ]:


df.dtypes


# After resetting the index for plotting,Im again setting datetimeindex .Its to be noted that when we have datetime index we cant set x=Column name, We have to use only df.index if the column is index column.

# In[ ]:


df.set_index('date',inplace=True)
df.index


# For having a clear analysis I am breaking down the date into months, days, years respectively.

# In[ ]:


df['Year'] = df.index.year
df['Month'] = df.index.month
df['Weekday Name'] =df.index.day_name


# In[ ]:


df["Day"]=df.index.day


# In[ ]:


df['Weekday Name'] =df.index.day_name()
df['Month Name'] = df.index.month_name()


# In[ ]:


df.tail(15)


# df.loc is an useful function to locate a specific set of data in a dataframe based on the condtion provided.
# Here I provided te condition that the datetimeindex = 2017-11-14

# In[ ]:


df.loc['2017-11-14']


# In[ ]:


cols_plot = ['pm25', 'no2', 'so2','co']
axes = df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('AQI Reading')


# Scatterplot is more efficient in studying value based datasets .Here we can see some spikes in 2016 and 2017.At some period like this, due to geographical and industrial reasons the AQI became very poor.

# Heres a monthwise analysis.
# The so2 and no2 are mostly under safe limit.But the pm25 and co has been found in critically risk levels.

# In[ ]:


fig, axes = plt.subplots(figsize=(16,25),nrows=4)
sns.barplot(x=df["Month Name"],y=df['so2'],ax=axes[0],color='coral')
sns.barplot(x=df["Month Name"],y=df['pm25'],ax=axes[1],color='yellowgreen').axhline(y=120)
sns.barplot(x=df["Month Name"],y=df['co'],ax=axes[2],color='lightsteelblue').axhline(y=10)
sns.barplot(x=df["Month Name"],y=df['no2'],ax=axes[3],color='goldenrod')


# The blue line in co and pm25 plot is to mark the safety level, and it can be noticed that in August, co level usually crosses the limits.
# Also the pm25 reaches its peak on january and slowly settles down in the subsequent months

# In[ ]:



pv1 = pd.pivot_table(df, index=df.index, columns=df.index.year,
                    values='pm25', aggfunc='sum')


pv1.plot(figsize=(25,3))


# # Effect of Lockdown-2020
# During the months of April,May and June, the pollution level has significantly reduced due to shutting down of factories and absence of automobile emmissions.Here we can visualise the decline till may and it starts slowly rising again by last part of may, as the lockdown was lifted step by step.

# In[ ]:


df.sample(5)


# In order to clearly visualise the difference, I am taking the data of March- June of all the years and putting it in a new dataframe to analyse and study it further.

# In[ ]:


df.dtypes


# In[ ]:


d2=df.loc[((df.Month >=4) &( df.Month<=6)),["pm25","co","so2","no2","Year","Day"]]


# In[ ]:


d2.sample(5)


# Plotting the effect of lockdown on the concentration of the four gaseous elements respectively.

# In[ ]:


fig, axes = plt.subplots(figsize=(16,25),nrows=4)
sns.lineplot(x=d2.Day, y="co", hue='Year', data=d2,palette=['green','orange','brown','dodgerblue','red',"yellowgreen","black"],ci=None,ax=axes[0])
sns.lineplot(x=d2.Day, y="so2", hue='Year', data=d2,palette=['green','orange','brown','dodgerblue','red',"yellowgreen","black"],ci=None,ax=axes[1])
sns.lineplot(x=d2.Day, y="pm25", hue='Year', data=d2,palette=['green','orange','brown','dodgerblue','red',"yellowgreen","black"],ci=None,ax=axes[2])
sns.lineplot(x=d2.Day, y="no2", hue='Year', data=d2,palette=['green','orange','brown','dodgerblue','red',"yellowgreen","black"],ci=None,ax=axes[3])


# **The blackline is 2020, I have taken the data of march-june and taken the mean value over 30 days and plotted it.**
# We can easily understand and study the difference in the AQI, made by a lockdown order.
# This can be further studied as to fight against climate change by understanding the effect of man made emmissions on the AQI.
# 
# The pm25 data for 2014 is not available and hence the imputed value line is plain.
# 

# # Heatmap
# It is the best way to visualise the 2D data.
# First I am making a pivot table to understand the changes in AQI with respect to time.

# In[ ]:


pvmp = pd.pivot_table(df, index=df.Year, columns=df["Month"],
                    values='pm25', aggfunc='mean')
print(pvmp)
plt.figure(figsize=(11,9))
sns.heatmap(data=pvmp,fmt=".1f",annot=True,cmap = sns.cm.rocket_r)


# Here you can clearly see that during 4,5 (April and May) the pm2.5 emmission has decreased significantly.

# 

# In[ ]:





# In[ ]:


plt.figure(figsize=(16,8))
sns.regplot(x=df.index,y=df["pm25"])


# I have made a regression plot to see the seasonal pattern of pm2.5 emmission.
# We can clearly see a sine-wave pattern here, which suggests a pattern in production of emmissions possibly indicating climatic influences like wind season , rainy season or maybe the production target and schedule of industries making the emmissions.
# 

# 

# In[ ]:





# In[ ]:




