#!/usr/bin/env python
# coding: utf-8

# India Rainfall Analysis

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[2]:


data = pd.read_csv("../input/rainfall in india 1901-2015.csv",sep=",")
data.info()


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.hist(figsize=(12,12));


# In[6]:


data.groupby("YEAR").sum()['ANNUAL'].plot(figsize=(12,8));


# In[7]:


data.columns


# In[8]:


data[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("YEAR").sum().plot(figsize=(13,8));


# In[9]:


data[['YEAR','Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']].groupby("YEAR").sum().plot(figsize=(13,8));


# Span of Jun-Sep has maximum rainfall records.

# In[10]:


data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("SUBDIVISION").mean().plot.barh(stacked=True,figsize=(13,8));


# Graph shows top 3 subdivisions having high average rainfall:
# 1. ARUNACHAL PRADESH
# 2. COASTAL KARNATAKA
# 3. KOKAN & GOA

# In[11]:


data[['SUBDIVISION', 'Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']].groupby("SUBDIVISION").sum().plot.barh(stacked=True,figsize=(16,8));


# In[12]:


data[['Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']].plot(kind="kde",figsize=(13,8));


# In[13]:


data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("SUBDIVISION").max().plot.barh(figsize=(16,8));


# Graph shows Max rainfall for specific month registered in ARUNACHAL PRADESH.

# **District wise details**

# In[14]:


district = pd.read_csv("../input/district wise rainfall normal.csv",sep=",")
district.info()


# In[19]:


district[['DISTRICT', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("DISTRICT").mean()[:40].plot.barh(stacked=True,figsize=(13,8));


# In[20]:


district[['DISTRICT', 'Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']].groupby("DISTRICT").sum()[:40].plot.barh(stacked=True,figsize=(16,8));


# **Madhya Pradesh Data**

# In[23]:


mp_data = district[district['STATE_UT_NAME'] == 'MADHYA PRADESH']


# In[24]:


mp_data[['DISTRICT', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("DISTRICT").mean()[:40].plot.barh(stacked=True,figsize=(13,8));


# In[25]:


mp_data[['DISTRICT', 'Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']].groupby("DISTRICT").sum()[:40].plot.barh(stacked=True,figsize=(16,8));

