#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# ## Load Packages

# In[6]:


import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
from IPython import display
#collection of machine learning algorithms
import sklearn
import time

#charting tools
import matplotlib.pyplot as plt
import seaborn as sns

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ## Exploratory Analysis

# In[7]:


## Load Data
df_donations = pd.read_csv('../input/Donations.csv')
print(df_donations.columns)
print('-'*25)
df_donors = pd.read_csv('../input/Donors.csv')
print(df_donors.columns)
#Donations and Donors can be merged on Donor ID.
print('-'*50)
df_projects = pd.read_csv('../input/Projects.csv')
#project Data looks strange. 
df_resources = pd.read_csv('../input/Resources.csv')
print(df_resources.columns)
print('-'*25)
df_schools = pd.read_csv('../input/Schools.csv')
print(df_schools.columns)
print('-'*25)
df_teachers = pd.read_csv('../input/Teachers.csv')
print(df_teachers.columns)
print('-'*25)


# In[9]:


#merge two dataset on Donor ID.
set1 = pd.merge(df_donations, df_donors, on = 'Donor ID')
print(set1.info())
#print missing data
print(pd.isnull(set1).sum())


# Donor City and Donor Zip data are missing for some. If we find this to be important, we'll come back to it later. 

# ### General information on donation amount. 

# In[10]:


print('Min amount is: $', round(set1['Donation Amount'].min(),2))
print('Max amount is: $', round(set1['Donation Amount'].max(),2))
print('Median amount is: $', round(set1['Donation Amount'].median(),2))
print('Mean amount is: $', round(set1['Donation Amount'].mean(),2))


# Median gives us a better idea on how much a typical doner would give. $60,000 is clearly an outlier and it would significantly affect the mean. 

# In[11]:


chart1 = set1[df_donations['Donation Amount']>0]['Donation Amount']
sns.distplot(chart1, bins=40, kde=False);


# The histogram above suggests that this dataset is not normally distributed; it has a longtail on the right. We can filter the dataset by amounts of less than 100 USD to see the shape of the data. 

# In[12]:


chart2 = set1[df_donations['Donation Amount']<100]['Donation Amount']
sns.distplot(chart2, bins=40, kde=False);


# We get a better idea of what the donations when we remove the longtail on the right. There are some clusters around 20 and 50.

# ### Top donors: Frequency vs Total Amount

# In[13]:


set1['Donor ID'].value_counts()[0:10]


# These are the top 10 donors by frequency. Top 3 donors have donated 18035, 14565, 10515 times respectively. That's quite a lot.

# In[19]:


set1['Donation Amount'].groupby(df_donations['Donor ID']).sum().sort_values(ascending=False)[0:10]


# This is the top 10 donors in gross amount. While I am not surprised that ID "a0e1d358aa17745ff3d3f4e4909356f3" is on the top 10 frequency list, that's where the overlap of the ID's stop. We should investigate this more. 

# In[18]:


by_freq = set1['Donor ID'].value_counts().to_dict()
#dictionary of frequency data
by_value = set1['Donation Amount'].groupby(df_donations['Donor ID']).sum().to_dict()
#dictionary of sum of donation amount
#then build two columns "Frequency", "Total Amount", and "Average Amount"
set1['Frequency'] = set1['Donor ID'].map(by_freq)
set1['Total Amount'] = set1['Donor ID'].map(by_value)
set1['Average Amount'] = set1['Total Amount'] / set1['Frequency']


# In[21]:


#Top 10 Donors in Frequency, Total Amount, and Average Amount
#need to re-write code that prints top 10 Donor ID's by Frequency, and Total Amount. 
#The table needs to have Donor ID, Frequency, Total Amount, and Average Amount


# * There is only 1 donor (a0e1d358aa17745ff3d3f4e4909356f3) who is both on the top 10 list in frequency and total value.
# * Top 20 has 1 donor
# * Top 30 has 4 donors
# * Top 40 has 4 donors
# * Top 50 has 5 donors
# * Top 60 has 9 donors
# * Top 70 has 10 donors
# * Top 100 has 14 donors
# * Top 1000 has 287 donors
# 
# It also appears that top donors have an average donation amount of less than 10 dollars, except one outlier (a0e1d358aa17745ff3d3f4e4909356f3).

# ### Donations by State

# In[27]:


tmp = set1['Donor State'].value_counts()
df1 = pd.DataFrame({'State': tmp.index,'Number of donations': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,15))
s = sns.barplot(ax = ax1, x = 'Number of donations', y="State",data=df1)


# Total Donations by State

# In[23]:


tmp = set1.groupby('Donor State')['Donation Amount'].sum().sort_values(ascending = False)
df1 = pd.DataFrame({'State': tmp.index,'Total sum of donations in 10M': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,15))
s = sns.barplot(ax = ax1, x = 'Total sum of donations in 10M', y="State",data=df1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




