#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("/kaggle/input/raw_data.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df = df.drop(['Source_1','Source_2','Source_3','State Patient Number','Estimated Onset Date','Status Change Date'],axis=1)
df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# # Univariate Analysis and Bivariate Analysis

# In[ ]:


df.describe()


# In[ ]:


df['Age Bracket'].plot.hist(bins=50)


# Age bracket in the range of 20-40 are most affected, then the once who are old

# In[ ]:


df['Date Announced'].value_counts().plot(kind='bar')


# We can clearly see that the number of cases went on MULTIPLYING from 2-Feb to 27-Mar

# In[ ]:


(df['Gender'].value_counts()/len(df['Gender'])*100).plot(kind='bar')


# Almost double in number, males are affected than the number of women affected by the virus

# In[ ]:


df['Detected State'].value_counts()


# In[ ]:


df['Detected State'].value_counts().plot(kind='bar')


# Maharashtra is the most affected state with 196 patients, followed by kerala with 182

# In[ ]:


## Lets look for which districts are most affected in Maharashtra

temp_df=df.loc[df['Detected State']=='Maharashtra']
temp_df


# In[ ]:


temp_df['Notes'].value_counts()


# There is no clear information about how the virus affected so effectively in Maharashtra

# In[ ]:


temp_df['Detected District'].value_counts().plot(kind='bar')


# Mumbai District of Maharashtra is the most affected

# In[ ]:


## Lets look for which city are most affected in Maharashtra

temp_df['Detected City'].value_counts().plot(kind='bar')


# Mumbai City area is the most affected in maharashtra

# In[ ]:


## Lets look for which districts are most affected in Karnataka

temp2_df=df.loc[df['Detected State']=='Karnataka']
temp2_df


# In[ ]:


temp2_df['Detected District'].value_counts().plot(kind='bar')


# In[ ]:


## Current Status of the patients

df['Current Status'].value_counts()


# Totally 18 are dead

# In[ ]:


df['Current Status'].value_counts().plot(kind='bar')


# In[ ]:


df['Nationality'].value_counts()


# In[ ]:


df['Nationality'].value_counts().plot.bar()


# In[ ]:


df['Notes'].value_counts()


# In[ ]:


temp3_df = df.loc[(df['Nationality']=='India') & ((df['Notes']=='Travelled from Dubai') | (df['Notes']=='Travelled from UK'))]
temp3_df


# In[ ]:


temp3_df.shape


# 40 Indians travelled from Dubai or UK who are infected

# In[ ]:


temp4_df = df.loc[df['Nationality']=='India']
temp4_df['Notes'].value_counts()


# Here, you can see that the virus infected large number of patients who travelled from abroad, also there are 242 reasons why Indians got affected

# In[ ]:


temp4_df['Current Status'].value_counts()


# In[ ]:


temp4_df['Current Status'].value_counts().plot.bar()


# Large number of Indian patients are Hospitalized, few are already recovered and 9 are dead

# In[ ]:


temp5_df = df.loc[df['Current Status']=='Deceased']
temp5_df['Age Bracket'].value_counts().plot.bar()


# Older People in the age group above 60 are the highest to be Deceased

# In[ ]:


temp6_df = df.loc[df['Current Status']=='Recovered']
temp6_df['Age Bracket'].value_counts().plot.bar()


# Young people are more likely to recover from the virus

# In[ ]:


temp7_df= df[['Age Bracket','Notes','Current Status']]
temp7_df


# In[ ]:


males = df[df['Gender']=='M']
females = df[df['Gender']=='F']

m=males['Current Status'].value_counts()/len(males)*100
f=females['Current Status'].value_counts()/len(females)*100
m, f


# In[ ]:


m.plot(kind='bar')


# In[ ]:


f.plot(kind='bar')


#  Male percentage of patients those are Deceased are 4% whereas females are 2%. Whereas, 90.9% and 93.82% are Hospitalized, 
# also almost same percentile of Males and Females have recovered.

# # Data Manipulation

# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


df.isnull().sum()


# In[ ]:


tem = ['Date Announced','Detected City','Gender','Detected District','Detected State','Current Status','Nationality']

for i in tem:
    print('--------------********-------------')
    print(df[i].value_counts())


# The Dataset is well maintained and there are no mistakes in the datset relating to spelling etc. Therefore there is no need to replace any attributes in any of the columns

# Since there is no clear information on Details of how the virus was infected and other data about the patients, it would not be a good idea to drop the rows or columns since there will be loss of data
