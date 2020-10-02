#!/usr/bin/env python
# coding: utf-8

# In this notebook we'll try to analyse the data of SF Restaurant Scores - LIVES Standard as a continued updating data(updated daily) This notebook is a work for Dashboarding with scheduled notebooks. This is the first draft(without cleaning data)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#what the data looks like
data  = pd.read_csv("../input/restaurant-scores-lives-standard.csv")
data.head(10)


# In[ ]:


data['inspection_type'].value_counts()


# Let's  see do a light EDA

# In[ ]:


plt.hist(data['inspection_type'].value_counts())
plt.show()


# In[ ]:


print(data['risk_category'].value_counts())


# In[ ]:


data["business_postal_code"].value_counts()


# We need to clean the data. But before let's see some plots

# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df = data[["business_postal_code",'risk_category']]
sns.countplot(x='risk_category',data=df)


# There is two variables we need to track:
# 1/ the location either by the longitude-latitude or by the business postal code. The location give us an idea if there is any regions where the inspections are often high risk
# 
# 2/ The nature of inspection cared by the variable inspection_type: We need to know if there is any correlation between the unexpected inspections and the risk_category

# In[ ]:


data['inspection_type'].value_counts()


# In[ ]:


sns.countplot(x="inspection_type", hue = "risk_category", data=data)


# In[ ]:


simple_inspection_type = {'inspection_type':{'Routine - Unscheduled':'Unexpected',
                                             'Reinspection/Followup':'Expected',
                                             'Complaint':'Expected',
                                             'New Ownership':'Expected',
                                             'New Construction':'Expected',
                                             'Non-inspection site visit':'Expected',
                                             'Structural Inspection':'Expected',
                                             'New Ownership - Followup':'Expected',
                                             'Complaint Reinspection/Followup':'Expected',
                                             'Foodborne Illness Investigation':'Expected',
                                             'Routine - Scheduled':'Expected',
                                             'Special Event':'Expected',
                                             'Administrative or Document Review':'Expected',
                                             'Home Environmental Assessment':'Expected',
                                             'Community Health Assessment':'Expected'
                                             }}

data_copy = data.copy()
data_copy.replace(simple_inspection_type,inplace = True)
sns.countplot(x="inspection_type", hue = "risk_category", data=data_copy)


# First conclusion: Maybe my classification isn't that true. We need to dig more in each inspection type to know for sure what unexpected visits are and what are not. 
# But the plot is that clear that we have more results with the Unexpected visits
# 

# Let's see if we can have an idea of a pattern related to the postal code so to not use here longitude-latitude but regions defined by postal codes.
# It will not be very accurate to say that a region is more at risk than another if we don't compare it with the number of restaurant per surface and the density of population

# In[ ]:


data["business_postal_code"].unique()


# In[ ]:


len(data["business_postal_code"].unique())


# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.countplot(x="business_postal_code", hue = "risk_category", data=data_copy)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# Here we can see as a first draft that the zones: 94103, 94110,94133, 94109 represent the bigget risk. 

# In[ ]:




