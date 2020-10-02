#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


resource = pd.read_csv('../input/Resources.csv')
school = pd.read_csv('../input/Schools.csv')
donor = pd.read_csv('../input/Donors.csv')
donation = pd.read_csv('../input/Donations.csv')
teacher = pd.read_csv('../input/Teachers.csv')
project = pd.read_csv('../input/Projects.csv')


# In[3]:


resource.info()


# In[4]:


school.info()


# In[5]:


donor.info()


# In[6]:


donation.info()


# In[7]:


teacher.info()


# In[8]:


project.info()


# In[9]:


donation_amount = donation.groupby('Project ID')['Donation Amount']


# ## Distribution of project's donation amount
# I use log function to draw dist plot.I can't get any infomation when I use amount directly.
# * Donation amount obey the normal distribution
# * Most of projects' donation amount are around 2^8(256) dollars

# In[10]:


sns.distplot(np.log2(donation_amount.sum()))
plt.xlabel('log value of donation amount')


# ## Distribution of donation count

# In[18]:


sns.distplot(np.log2(donation_amount.count()),kde=False)
plt.xlabel('donation count')


# More projects have low donation count

# ## Distribution of per donation amount

# In[12]:


donation_donar = donation.groupby('Donor ID')['Donation Amount']


# In[17]:


sns.distplot(np.log2(donation_donar.count()),kde=False)
plt.xlabel('donation count per donor')


# In[52]:


donation_count_per_donor = donation_donar.count().sort_values(ascending=False).reset_index().rename(index=str, columns={'Donation Amount':'Donation Count'})
gt_100 = donation_count_per_donor[donation_count_per_donor['Donation Count']>100]
gt_100_dornor = pd.merge(gt_100,donor,how='left',on=['Donor ID'])
gt_100_dornor.head(5)


# In[51]:


plt.figure(figsize=(10,12))
plt.title('distribution of state donor gt 100')
sns.countplot(y=gt_100_dornor['Donor State'])


# In California,New York, lllinois,North Carolina,Texas and Florida, dornors make more donation. 

# In[49]:


plt.title('distribution of teacher')
sns.countplot(y=gt_100_dornor['Donor Is Teacher'])


# Teacher may like make more donation

# # Project

# In[59]:


prjoct_donation_count = donation.groupby('Project ID')['Donation ID'].count().sort_values(ascending=False).reset_index().rename(index=str, columns={'Donation ID':'Donation Count'})


# In[63]:


gt_50_prjoct_donation_count = prjoct_donation_count[prjoct_donation_count['Donation Count']>=50]
gt_50_project = pd.merge(gt_50_prjoct_donation_count,project,how='left',on=['Project ID'])


# ## Distribution of project resource category

# In[62]:


sns.countplot(y=project['Project Resource Category'])


# In[64]:


sns.countplot(y=gt_50_project['Project Resource Category'])


# In[65]:


sns.countplot(y=project['Project Grade Level Category'])


# In[66]:


sns.countplot(y=gt_50_project['Project Grade Level Category'])


# In[67]:


gt_50_project.head(5)


# In[ ]:


fig,axes = plt.subplots(2,1,)
sns.distplot(np.log2(project['Project Cost'].dropna()),ax=axes[0])
sns.distplot(np.log2(gt_50_project['Project Cost'].dropna()),ax=axes[1])


# In[ ]:




