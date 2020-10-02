#!/usr/bin/env python
# coding: utf-8

# In[20]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[22]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# #### Do Upvote if you like it
# 
# ## Loading Data

# In[24]:



resources=pd.read_csv('../input/Resources.csv')
donors=pd.read_csv('../input/Donors.csv')
schools=pd.read_csv('../input/Schools.csv')
teachers=pd.read_csv('../input/Teachers.csv')
projects=pd.read_csv('../input/Projects.csv')
donations=pd.read_csv('../input/Donations.csv')


# ### A peak into the Data

# In[26]:


resources.head()


# In[28]:


donors.head()


# In[ ]:


schools.head()


# In[ ]:


teachers.head()


# In[ ]:


projects.head()


# In[ ]:


donations.head()


# ### Looking at the shape of data

# In[ ]:


donations.shape


# In[ ]:


teachers.shape


# In[ ]:


schools.shape


# In[ ]:


donors.shape


# In[ ]:


projects.shape


# In[ ]:


resources.shape


# ### Looking at the type of data

# In[ ]:


donations.describe()


# In[ ]:


donations.isnull().sum().apply(lambda x: x/donations.shape[0])


# In[ ]:


donors.describe()


# In[ ]:


donors.isnull().sum().apply(lambda x: x/donors.shape[0])


# In[ ]:


donors[['Donor City','Donor Zip']].nunique()


# I wouldn't go into the hasstle of immputing the missing values in <i>`Donor Zip` </i> by looking at the <i>`Donor State`</i> and instead remove the missing values.

# In[ ]:


resources.describe()


# In[ ]:


resources.dtypes


# In[ ]:


resources['Resource Vendor Name'].nunique()


# In[ ]:


resources.isnull().sum().apply(lambda x: x/resources.shape[0])


# In[ ]:


schools.describe()


# In[ ]:


schools.isnull().sum().apply(lambda x: x/schools.shape[0])


# `School Percentage Free Lunch` can be imputed with the **mean or median**

# In[ ]:


projects['Teacher Project Posted Sequence'].describe()


# In[ ]:


projects['Teacher Project Posted Sequence'].nunique()


# In[ ]:


projects.isnull().sum().apply(lambda x: x/projects.shape[0])


# In[ ]:


teachers.describe()


# In[ ]:


teachers['Teacher Prefix'].unique()


# In[ ]:


teachers.isnull().sum().apply(lambda x: x/teachers.shape[0])


# So, a lot of data points are missing(in number) but when compared to the whole dataset, the percentage is very less.<br>
# So, it's better to remove the missing data in some cases and impute them in some cases.<br>
# <br>
# **Mean or Median** for continuous data.<br>
# **Mode** for categorical data.<br>
# 
# ### Now about Handling missing data

#  Since, the number of `Donation ID` missing the same as `Donation Amount` in `Donations.csv`, so need to impute them.<br>
#  Same in case of `Donors.csv`, I won't impute the `Donor City` in terms of **mode** as it would corrupt the data.
#  
#  Creating a New table combining Donors and Donation.
#  
# `School Percentage Free Lunch` has mean greater than median, hence it is right skewed.

# In[ ]:


donors_donations = donations.merge(donors, on='Donor ID', how='inner')


# In[ ]:


donors_donations.head()


# In[ ]:


donations.dropna(axis=0,how='any',inplace=True)
donors.dropna(axis=0,how='any',inplace=True)
resources.dropna(axis=0,how='any',inplace=True)
projects.dropna(axis=0,how='any',inplace=True)
schools.dropna(axis=0,how='any',inplace=True)
teachers.dropna(axis=0,how='any',inplace=True)


# Now, all the missing data has been removed. <br>
# Now, I will start with some **visulizations.**

# In[ ]:


df = donors.groupby("Donor State")['Donor City'].count().to_frame().reset_index()
X = df['Donor State'].tolist()
Y = df['Donor City'].apply(float).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Donor State' : Z[-5:][::-1], 'Count of Donor Cities' : sorted(Y)[-5:][::-1] })
sns.barplot(x="Donor State",y="Count of Donor Cities",data=data)


# In[ ]:


sns.countplot(x='School Metro Type',data = schools)


# In[ ]:


donors_state_amount=donors_donations.groupby('Donor State')['Donation Amount'].sum().reset_index()
donors_state_amount['Donation Amount']=donors_state_amount['Donation Amount'].apply(lambda x: format(x, 'f'))

df = donors_state_amount[['Donor State','Donation Amount']]
X = df['Donor State'].tolist()
Y = df['Donation Amount'].apply(float).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Donor State' : Z[-5:][::-1], 'Total Donation Amount' : sorted(Y)[-5:][::-1] })
sns.barplot(x="Donor State",y="Total Donation Amount",data=data)


# In[ ]:


data = donors_donations["Donor State"].value_counts().head(25)
plt.figure(figsize=(20,10))
sns.barplot(data=da,x='Donor State')


# In[ ]:


state_count = data.to_frame(name="number_of_projects").reset_index()
state_count = state_count.rename(columns= {'index': 'Donor State'})
# merging states with projects and amount funded
donor_state_amount_project = state_count.merge(donors_state_amount, on='Donor State', how='inner')

val = [x/y for x, y in zip(donor_state_amount_project['Donation Amount'].apply(float).tolist(),donor_state_amount_project['number_of_projects'].tolist())]
state_average_funding = pd.DataFrame({'Donor State':donor_state_amount_project['Donor State'][-5:][::-1],'Average Funding':val[-5:][::-1]})
sns.barplot(x="Donor State",y="Average Funding",data=state_average_funding)


# In[ ]:


schools.groupby('School Metro Type')['School Percentage Free Lunch'].describe()


# * Rural Schools Average of free Lunch 55.87 % with Deviation of 21%.
# * Suburban Schools Average of free Lunch 49.33% with Deviation of 27%.
# * Town Schools Average of free Lunch 58.33% with Deviation of 19.65%.
# * Urban Schools Average of free Lunch 68.33% with Deviation of 24%.
# * Unknown Schools Average of free Lunch 62.34% with Deviation of 22%.
# 

# In[ ]:


projects_resources = projects.merge(resources, on='Project ID', how='inner')
projects_resources.head()


# In[ ]:


sns.barplot(projects_resources['Resource Vendor Name'])


# In[ ]:


project_title = projects_resources['Project Title'].value_counts()[:5]
sns.barplot(data=project_title).set_title('Unique project title')


# In[ ]:


school_project = schools.merge(projects, on='School ID', how='inner')
project_open_close=school_project[['Project Resource Category','Project Posted Date','Project Fully Funded Date']]
project_open_close['Project Posted Date'] = pd.to_datetime(project_open_close['Project Posted Date'])
project_open_close['Project Fully Funded Date'] = pd.to_datetime(project_open_close['Project Fully Funded Date'])

time_gap = []
for i in range(school_project['School ID'].count()):
    if school_project['Project Current Status'][i] =='Fully Funded':
        time_gap.append(abs(project_open_close['Project Fully Funded Date'][i]-project_open_close['Project Posted Date'][i]).days)
    else:
        time_gap.append(-1)

project_open_close['Time Duration(days)'] = time_gap
project_open_close.head()


# In[ ]:


project_open_close_resource=project_open_close.groupby('Project Resource Category')['Time Duration(days)'].mean().reset_index()
df = project_open_close_resource[['Project Resource Category','Time Duration(days)']]
X = df['Project Resource Category'].tolist()
Y = df['Time Duration(days)'].apply(int).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Project Resource Category' : Z[0:5], 'Total Time Duration(days)' : sorted(Y)[0:5] })
sns.barplot(x="Total Time Duration(days)",y="Project Resource Category",data=data)


# In[ ]:





# In[ ]:





# In[ ]:




