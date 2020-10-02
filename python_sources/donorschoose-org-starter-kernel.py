#!/usr/bin/env python
# coding: utf-8

# # DonorsChoose.org Starter Kernel
# 
# ---
# 
# Howdy, here's a little starter kernel to help you get started with this Data Science for Good Event!  In this kernel I am going to do a quick merge as just one example of how you can work with this data! 
# 

# In[1]:


import pandas as pd


# In[2]:


# List files
get_ipython().system('ls ../input')


# In[3]:


donors = pd.read_csv('../input/Donors.csv', low_memory=False)
donations = pd.read_csv('../input/Donations.csv')


# In[4]:


# Merge donation data with donor data 
df = donations.merge(donors, on='Donor ID', how='inner')
df.head()


# In[8]:


donation_count = pd.DataFrame()
donation_count['counts'] = df.groupby('Donor ID')['Donation ID'].count()


# In[11]:


donation_count.describe()


# That's all from me, now it's up to you to help DonorsChoose.org find donors to fund these projects!
# 

# In[ ]:




