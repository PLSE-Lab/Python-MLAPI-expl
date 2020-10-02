#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns


# ## Read the inspections data

# In[ ]:


df=pd.read_csv("../input/restaurant-and-market-health-inspections.csv")
df.head(3).T


# In[ ]:


df.shape


# ## Analyze Sore and Grade distribution
# 
# * Score distribution

# In[ ]:


# sore distribution
df['score'] = df['score'].astype(int)
ax = df["score"].plot(kind = "hist",title = "Score Distribution",color = "forestgreen",figsize=(10,7),alpha=0.5)
ax.set_xlabel("sore value")


# * Grade dirstribution
# 
#     first, we need to find the abnormal value of grade

# In[ ]:


# find the abnormal value of grade
df[~df['grade'].isin(['A','B','C'])]


# In[ ]:


# deal with the abnormal value
df.loc[49492,['grade']] = 'C'


# In[ ]:


# list grade distirbution
grade_disribution = df.groupby('grade').size()
pd.DataFrame({'Count':grade_disribution.values},index = grade_disribution.index)


# ## Risk Analysis

# In[ ]:


temp = df.groupby('pe_description').size()
description_distribution = pd.DataFrame({'Count':temp.values},index = temp.index)
description_distribution


# In[ ]:


description_distribution.plot(kind='barh',color = 'g',figsize=(10,7))


# Majority of restaurants and markets have very high risks

# In[ ]:


def sub_risk(x):
    return x.split(' ')[-2]
    
df['risk'] = df['pe_description'].astype(str).apply(sub_risk)
temp =  df.groupby('risk').size()
risk_distribution = pd.DataFrame({'Count':temp.values},index = temp.index)


ax = risk_distribution['Count'].plot(kind="pie", legend=True,autopct='%.2f', figsize=(6, 6))
ax.set_title("Risk Distribution")


# Analyze the correlation between score, grade and risk

# In[ ]:


def risk2value(x):
    if x == 'LOW':
        return 10
    elif x == 'MODERATE':
        return 5
    else:
        return 0
def grade2value(x):
    if x == 'A':
        return 10
    elif x == 'B':
        return 5
    else:
        return 0
df['risk_v'] = df['risk'].apply(risk2value)
df['grade_v'] = df['grade'].apply(grade2value)
df2 = df.loc[:,['score','grade_v','risk_v']]
corr = df2.corr()
corr = (corr)
sns.heatmap(corr,xticklabels = corr.columns.values, yticklabels = corr.columns.values, cmap = "Purples",center = 0)


# ## Analyze Facilities and Owners of these Restaurants and Markets

# In[ ]:


# list top 20 facilities with most restaurants or markets
facility_distirbution = df.groupby(['facility_id','facility_name']).size()
top20_facility = facility_distirbution.sort_values(ascending=False).head(20)
pd.DataFrame({'Count':top20_facility.values},index = top20_facility.index)


# In[ ]:


# list top 20 owners with most restaurants or markets
owner_distirbution = df.groupby(['owner_id','owner_name']).size()
top20_owner = owner_distirbution.sort_values(ascending=False).head(20)
pd.DataFrame({'Count':top20_owner.values},index = top20_owner.index)


# ## Read the violations data

# In[ ]:


df2=pd.read_csv("../input/restaurant-and-market-health-violations.csv")
df2.head(3).T


# ## Common Health Code Violations
# 
# 
# 3 most violated codes:
#     #44. Floors, walls and ceilings: properly built, maintained in good repair and clean	30289
#     #33. Nonfood-contact surfaces clean and in good repair	28426
#     #35. Equipment/Utensils - approved; installed; clean; good repair, capacity

# In[ ]:


violation_description = df2.groupby('violation_description').size()
pd.DataFrame({'Count':violation_description.values},index = violation_description.index).sort_values(by = 'Count',ascending=False)

