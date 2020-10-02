#!/usr/bin/env python
# coding: utf-8

# I am new to data science and this is my first notebook on kaggle. If there are any mistakes or room for improvement please point them out. If you found this notebook useful consider upvoting it. 
# 
# Thank You

# Importing required libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Loading Case.csv dataset

# In[ ]:


data = pd.read_csv("../input/coronavirusdataset/Case.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# Under the infection_case column we can obtain what are the places the patients contaminated the disease

# In[ ]:


data.infection_case.unique()#returns only unique values


# Since there are several common places like Hospitals, Churches, Centers, or  Shelter/Nursing homes, I decided to group them under a single value.

# In[ ]:


def group(col):
        if "Hospital" in col:
            return "Hospital"
        elif "Church" in col:
            return "Church"
        elif "Center" in col:
            return "Center"
        elif "Shelter" in col or "Nursing Home" in col:
            return "Shelter/Nursing Home"
        else:
            return col


# In[ ]:


data['place'] = data.infection_case.apply(group)


# In[ ]:


data


# In[ ]:


data.place.unique()


# In[ ]:


data.isnull().sum()


# In[ ]:





# In[ ]:


fig, ax = plt.subplots(figsize=(15,8))
chart = sns.countplot(data.place)
chart.set(ylabel="Cases")
chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


# From the above graph:
#             * The maximum no of contaminations occured within churches.
#             * Then follows patients arriving from overseas and contacting with a patient

# In[ ]:


fig, ax = plt.subplots(figsize=(6,8))
sns.countplot(data.group)


# From the above graph:
#             * About 55% of the patient was infected as groups

# In[ ]:


fig, ax = plt.subplots(figsize=(15,8))
chart = sns.countplot(data.city)
chart.set(ylabel="Cases")
chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


# From the above graph:
#             * We cannot get an accurate information about cases per city since most of the cities are not provided.
#             * From the given cities we can realize that all of them have similar amount of cases.

# The graph below visualizes the no of case in each province

# In[ ]:


fig, ax = plt.subplots(figsize=(10,8))
chart = sns.countplot(data.province)
chart.set(ylabel="Cases")
chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
chart = sns.countplot(data.province, hue=data.place=='Church')
ax.legend(labels=['Other', 'Church'])
chart.set(ylabel="Cases")
chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
chart = sns.countplot(data.province, hue=data.place=='overseas inflow')
ax.legend(labels=['Other', 'Overseas inflow'])
chart.set(ylabel="Cases")
chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
chart = sns.countplot(data.province, hue=data.place=='Hospital')
ax.legend(labels=['Other', 'Hospital'])
chart.set(ylabel="Cases")
chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
chart = sns.countplot(data.province, hue=data.place=='contact with patient')
ax.legend(labels=['Other', 'Hospital'])
chart.set(ylabel="Cases")
chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

