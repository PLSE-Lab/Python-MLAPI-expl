#!/usr/bin/env python
# coding: utf-8

# In[32]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df=pd.read_csv('../input/Mass Shootings Dataset.csv',encoding='cp1252',parse_dates=["Date"])
df.shape


# In[33]:


df.info()


# As we see the counts of Summary ,Race, Gender,Latitude and Gratitude is missing for some rows
# Before any exploration , we need to take care of those missing values

# In[34]:


df.Gender.replace(['M', 'M/F'], ['Male', 'Male/Female'], inplace=True)
print(df.groupby('Gender').count())


# Here we see no. of Male is 346, hence Most likely the Gender missing value will be fill using 'Male'

# In[35]:


df['Gender'].fillna(value='Male',inplace=True)
df.info()


# Now we will be looking for mor columns for filling missing value
# Let's see into Race Columns

# In[36]:


df_total=df.groupby("Race").sum()
df_total.drop(df_total.columns[[0, 4, 5]], axis=1,inplace=True) 
print(df_total.head(20))


# Let's see total attack per year, total fatalities and total injured

# In[37]:


z=df['Date'].dt.year
df_tot=pd.DataFrame({'Year':z,'Fatalities':df['Fatalities'],'Injured':df['Injured'],"total_attack":np.ones(len(z)).astype(np.int32)})
#print(df_tot['Year'].value_counts())
#print("\nAs You will see most attack happened in year 2016\n")
print(df_tot.groupby('Year').sum())


# Maximum Attack - 76 attacks in 2016,  
# Maximum People Injured -573 people in 2017,   
# Maximum People Fatalities -255 people in 2015
