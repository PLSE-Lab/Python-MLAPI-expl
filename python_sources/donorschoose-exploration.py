#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)


# In[3]:


donations.info()


# In[4]:


donors.info()


# In[5]:


schools.info()


# In[6]:


teachers.info()


# In[7]:


projects.info()


# In[8]:


resources.info()


# Lets explore donations a bit

# In[9]:


donations.head()


# In[10]:


# number of rows and columns in donations dataset
donations.shape


# In[11]:


# unique number of project ids in donation dataset
donations['Project ID'].nunique()


# In[12]:


# unique number of donation ids, donor ids in donation dataset
donations['Donation ID'].nunique(), donations['Donor ID'].nunique()


# In[13]:


donations.describe()


# The mean of the donation amount is quite higher than median.  The distribution of donation amount has a large right tail (positive skewness)

# In[14]:


# skewness of donation amount and donor cart sequence
donations['Donation Amount'].skew(), donations['Donor Cart Sequence'].skew()


# In[15]:


#distribution of optional donation
donations['Donation Included Optional Donation'].value_counts(normalize = True)


# In[16]:


# what is the average donation when the optional donation is no or yes
grouped = donations.groupby('Donation Included Optional Donation')['Donation Amount'].mean().reset_index()
grouped


# Surprisingly, when the optional donation is not present the donation amount is high

# In[17]:


donations['Donation Received Date'] = pd.to_datetime(donations['Donation Received Date'])


# In[18]:


donations['Donation Received Date wd'] = donations['Donation Received Date'].dt.weekday_name
plt.rcParams['figure.figsize'] = [15, 4]
sns.barplot(x = 'Donation Received Date wd', y = 'Donation Amount', data = donations)


# Expectedly saturdays and sundays normally have a low amount donation amount

# In[19]:


donations['Donation Received Date month'] = donations['Donation Received Date'].dt.month
plt.rcParams['figure.figsize'] = [15, 4]
sns.barplot(x = 'Donation Received Date month', y = 'Donation Amount', data = donations)


# The donations is normally higher around the christmas month

# In[20]:


donations['Donation Received Date year'] = donations['Donation Received Date'].dt.year
sns.barplot(x = 'Donation Received Date year', y = 'Donation Amount', data = donations)


# In[21]:


# lets check the counts of the year
donations['Donation Received Date year'].value_counts()


# There is not much difference in the average donation amount yearwise. 2012 is a bit higher because its count is less

# In[22]:


donors.head()


# In[23]:


# how many cities of donor cities and states of donor dataset are there
donors['Donor City'].nunique(), donors['Donor State'].nunique()


# In[24]:


# counts of whether a donor is a teacher or not
donors['Donor Is Teacher'].value_counts(normalize = True)


# In[28]:


grouped = donations.groupby('Donor ID')['Donation Amount'].sum().reset_index()
donors = pd.merge(donors, grouped, on = 'Donor ID', how = 'inner')
donors.head(2)


# In[34]:


# which state has the highest median donation amount
grouped = donors.groupby('Donor State')['Donation Amount'].median().reset_index()
grouped = grouped.sort_values(by = 'Donation Amount', ascending = False)
print (grouped.iloc[0])

# which state has the lowest median donation amount
print (grouped.iloc[-1])


# In[35]:


donors[donors['Donor State'] == 'Idaho'].head(2)


# In[36]:


# which city has the highest median donation amount
grouped = donors.groupby('Donor City')['Donation Amount'].median().reset_index()
grouped = grouped.sort_values(by = 'Donation Amount', ascending = False)
print (grouped.iloc[0])

# which city has the lowest median donation amount
print (grouped.iloc[-1])


# In[37]:


# which state has the highest variation in donation amount
grouped = donors.groupby('Donor State')['Donation Amount'].std().reset_index()
grouped = grouped.sort_values(by = 'Donation Amount', ascending = False)
print (grouped.iloc[0])

# which state has the lowest variation in donation amount
print (grouped.iloc[-1])


# New york has the highest variation in donation amount. We have already seen earlier that the mean of the donation amount is quite higher than the median of the donation amount. So New york will most likely have the highest average donation amount

# In[39]:


# which state has the highest average donation amount
grouped = donors.groupby('Donor State')['Donation Amount'].mean().reset_index()
grouped = grouped.sort_values(by = 'Donation Amount', ascending = False)
print (grouped.iloc[0])


# In[38]:


# which city has the highest variation in donation amount
grouped = donors.groupby('Donor City')['Donation Amount'].std().reset_index()
grouped = grouped.sort_values(by = 'Donation Amount', ascending = False)
print (grouped.iloc[0])

# which city has the lowest variation in donation amount
print (grouped.iloc[-1])


# In[42]:


# how many average unique donors are there from a state
donors.shape[0]/donors['Donor State'].nunique()


# In[43]:


# 90000 average donors are normally there from a state
donors[donors['Donor State'] == 'New York']['Donor ID'].nunique()


# In[45]:


# plot distribution of donors from a state
grouped = donors.groupby('Donor State').size().reset_index()
grouped.columns = ['Donor State', 'Donor_count']
sns.distplot(grouped['Donor_count'])


# In[46]:


grouped.describe()


# In[47]:


# how many unique states are there?
grouped['Donor State'].nunique()


# In[48]:


# how many states have donor count more than the average donor count
grouped[grouped['Donor_count']>grouped['Donor_count'].mean()].shape[0]


# In[50]:


# which state has the highest number of donors?
print(grouped[grouped['Donor_count']==grouped['Donor_count'].max()])
# which state has the lowest number of donors?
print(grouped[grouped['Donor_count']==grouped['Donor_count'].min()])


# One thing Wyoming has the lowest variation in donation amount as well as donor count

# ## More to  come. Stay Tuned !!!

# In[ ]:




