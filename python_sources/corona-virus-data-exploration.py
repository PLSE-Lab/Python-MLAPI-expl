#!/usr/bin/env python
# coding: utf-8

# # Latest Corona Virus Data Visulization

# ![](https://www.sciencemag.org/sites/default/files/styles/article_main_large/public/images/sn-hepatitis.jpg?itok=HRawWYZy)

# ## Introduction
# The 2019-nCoV is a contagious coronavirus that hailed from Wuhan, China. This new strain of virus has striked fear in many countries as cities are quarantined and hospitals are overcrowded. This dataset will help us understand how 2019-nCoV is spread aroud the world.

# ## Data Loading

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Data Visulizations
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


file = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")
file = file.drop(['Unnamed: 0'], axis = 1) 
file.info()


# - **Province/State ** - City of virus suspected cases.
# - **Country** - Country of virus suspected cases.
# - **Date last updated	** - Date of update of patient infected
# - **Confirmed** - Confirmation by doctors that this patient is infected with deadly virus
# - **Suspected** - Number of cases registered
# - **Recovered** - Recovery of the patient
# - **Deaths** - Death of the patient due to corna virus.
# 

# Some Staticals calculations on dataset

# In[ ]:


round(file.describe())


# In[ ]:


# first few record of the dataset
file.head(10)


# Ok, now that we have a glimpse of the data, let's explore them.

# ## Data Explorations & Visulizations

# ### Relationship Between Confirmend,Suspected,Recovered and Death by Contry and States

# In[ ]:


plt.figure(figsize=(20,6))
sns.pairplot(file, size=3.5);


# In[ ]:


plt.figure(figsize=(20,6))
sns.pairplot(file,hue='Country' ,size=3.5);


# In[ ]:


plt.figure(figsize=(20,6))
sns.pairplot(file,hue='Province/State' ,size=3.5);


# ### Country and State wise Explorations

# In[ ]:


data= pd.DataFrame(file.groupby(['Country'])['Confirmed','Suspected','Recovered','Deaths'].agg('sum')).reset_index()
data.head(19)


# In[ ]:


data= pd.DataFrame(file.groupby(['Country'])['Confirmed','Suspected','Recovered','Deaths'].agg('sum')).reset_index()

data.sort_values(by=['Confirmed'], inplace=True,ascending=False)

plt.figure(figsize=(12,6))

#  title
plt.title("Number of Patients Confirmed Infected by Corona Virus, by Country")

# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country
sns.barplot(y=data['Country'],x=data['Confirmed'],orient='h')

# Add label for vertical axis
plt.ylabel("Number of Confirmed Patients")


# In[ ]:


data.sort_values(by=['Suspected'], inplace=True,ascending=False)

plt.figure(figsize=(12,6))

#  title
plt.title("Number of Patients Suspected Infected by Corona Virus, by Country")

# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country
sns.barplot(y=data['Country'],x=data['Suspected'],orient='h')

# Add label for vertical axis
plt.ylabel("Number of Suspected Patients")


# In[ ]:


data.sort_values(by=['Recovered'], inplace=True,ascending=False)

plt.figure(figsize=(12,6))

#  title
plt.title("Number of Patients Recovered from by Corona Virus, by Country")

# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country
sns.barplot(y=data['Country'],x=data['Recovered'],orient='h')

# Add label for vertical axis
plt.ylabel("Number of Recovered Patients")


# In[ ]:


data.sort_values(by=['Deaths'], inplace=True,ascending=False)

plt.figure(figsize=(12,6))

#  title
plt.title("Number of Patients Died by Corona Virus, by Country")

# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country
sns.barplot(y=data['Country'],x=data['Deaths'],orient='h')

# Add label for vertical axis
plt.ylabel("Number of Deaths")


# As we got the insight that china and some other countries near by china have many cases.

# ## Sates of China

# In[ ]:


china= file[file['Country'] == 'Mainland China']
china_data= pd.DataFrame(china.groupby(['Province/State'])['Confirmed','Suspected','Recovered','Deaths'].agg('sum')).reset_index()
china_data.head(35)


# In[ ]:


china_data.sort_values(by=['Confirmed'], inplace=True,ascending=False)

plt.figure(figsize=(25,10))

#  title
plt.title("Number of Patients Confirmed Infected by Corona Virus, by States")

# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country
sns.barplot(x=china_data['Province/State'],y=china_data['Confirmed'],orient='v')


# Add label for vertical axis
plt.ylabel("Number of Confirmed Patients")


# In[ ]:


china_data.sort_values(by=['Deaths'], inplace=True,ascending=False)

plt.figure(figsize=(25,10))

#  title
plt.title("Number of Patients Died by Corona Virus, by States")

# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country
sns.barplot(x=china_data['Province/State'],y=china_data['Deaths'],orient='v')


# Add label for vertical axis
plt.ylabel("Number of Deaths")


# This is not end I am still exploring data
