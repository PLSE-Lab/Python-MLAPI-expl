#!/usr/bin/env python
# coding: utf-8

# # EDA: 15 Years of Power Outage

# * Title: 15 Years Of Power Outages
# 
# * Objective: Predict attrition of your valuable employees
# 
# * Kaggle link: https://www.kaggle.com/autunno/15-years-of-power-outages
# 
# * Inspired by: http://insideenergy.org/2014/08/18/data-explore-15-years-of-power-outages/#comment-3862651149
# 
# This notebook aims to make an EDA (Exploratory Data Analysis) on 15 years of outage to find out the features' relations and to prepare the ground for a Machine Learning model to predict the cause.

# # How  this notebook is organized

# 1. [Data pre-processing](#1.-Data-pre-processing)
# 2. [Data analysis](#2.-Data-analysis)

# # 1. Data pre-processing

# We start by importing all the libraries we're going to use:

# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# We now need to import our data:

# In[8]:


dataset = pd.read_csv('../input/Grid_Disruption_00_14_standardized - Grid_Disruption_00_14_standardized.csv')
dataset.head()


# In[3]:


print("Number of entries: " + str(len(dataset.index)))


# A quick peek at the data shows us that empty values are defined as "Unknown", which means we should treat them as NULL. To decide what to do with each value, we must first analyze how many empty values each column has:

# ### Dealing with empty values

# We can see many columns have "Unknown", which needs to be cleaned. We need to have special care with our numerical columns. Year is pretty likely pretty clean, I expect that "Number of Customers Affected" is more troublesome:

# In[4]:


len(pd.to_numeric(dataset['Year'], 'coerce').dropna().astype(int))


# "Year" seems to be perfectly filled, we don't need to worry about it.

# In[5]:


len(pd.to_numeric(dataset['Demand Loss (MW)'], 'coerce').dropna().astype(int))


# Over 700 rows are not numeric on 'Demand Loss (MW)'. It's quite a lot of missing values (almost 50%), we'll have to decide if we want to keep it or not.

# In[6]:


len(pd.to_numeric(dataset['Number of Customers Affected'], 'coerce').dropna().astype(int))


# As we can see above, we have too many non numerical rows for this column (only 222 are correctly filled), it may be best to simply drop it. Let's take a quick look at 'Demand Loss (MW)' first:

# In[7]:


print('Demand Loss (MW)')
dataset.iloc[:, 9]


# Besides the usual 'None', 'NaN' and 'Unknown', there are some range values in place (separated by '-'). In other circumstances, I would just remove this column, but since we're likely dropping 'Number of Customers Affected', let's simply remove the empty values:

# In[8]:


dataset = dataset.iloc[pd.to_numeric(dataset['Demand Loss (MW)'], 'coerce').dropna().astype(int).index, :]
print(len(dataset.index))


# Let's take a quick look at 'Number of Customers Affected', to make sure it isn't anything we can fix:

# In[9]:


print('Number of Customers Affected')
dataset.iloc[:, 10]


# As we suspected, this data is not in good shape to be used. Besides the usual culprits ("NaN", "Unknown", "None"), we also have some strange choices, such as using "Approx. " and " - " to indicate a possible range of values. With that in mind, let's proceed with the plan to drop it:

# In[10]:


dataset = dataset[dataset.columns.difference(['Number of Customers Affected'])]


# With that done, we can continue with our data pre-processing and replace 'Unknown' with None on all other columns, so that we can have a better idea of how many empty values we have:

# In[11]:


for column in dataset.columns:
    dataset[column].replace('Unknown', None, inplace=True)


# In[12]:


dataset.isnull().any()


# Many columns have empty values, lets now check how bad it is:

# In[13]:


print("Total number of rows: " + str(len(dataset.index)))
print("Number of empty values:")
for column in dataset.columns:
    print(" * " + column + ": " + str(dataset[column].isnull().sum()))


# We now have very few columns left with 'None' values, we can just remove these rows.

# In[14]:


dataset = dataset.dropna()


# Despite "Event Description" being a very interesting column, we don't need it, since it tells the same story as "Tags" (which is simplified):

# In[15]:


dataset = dataset[dataset.columns.difference(['Event Description'])]


# We can now check if our data is properly cleaned:

# In[37]:



print("Total number of rows: " + str(len(dataset.index)))
print("Number of empty values:")
for column in dataset.columns:
    print(" * " + column + ": " + str(dataset[column].isnull().sum()))


# It is! The new size of our dataset is 797 (down from 1652). We lost quite a lot of data (a little bit over 50%), but it should be enough for our analysis. Before proceeding, let's take a last quick look at it:

# In[38]:


dataset.head()


# # 2. Data analysis

# With our dataset properly cleaned, we can now take a look and see how it's distributed (and how the columns relate to each other). A few interesting plots comes to mind:
#  * Year and Tags
#  * Demand Loss (MW) and Year
#  * Count of causes (Tags)
#  * Count of occurrences per Year

# It's a little too soon for feature engineering, but we should do a little more work on the Tags column before we proceed if we want to be able to do any meaningfull analysis. Since there are many different values, it could be a good idea to rebrand all severe weather columns with just 'severe weather':

# In[60]:


dataset.loc[dataset['Tags'].str.contains('severe weather', case=False), 'Tags'] = 'severe weather'


# We can now start the plots:

# In[39]:


dim = (12, 30)
fig, ax = plt.subplots(figsize=dim)
sns.swarmplot(x="Year", y="Tags", ax=ax, data=dataset)


# In[59]:


dim = (40, 10)
fig, ax = plt.subplots(figsize=dim)
demand_plot = sns.lvplot(x="Demand Loss (MW)", y="Year", ax=ax, data=dataset)

for item in demand_plot.get_xticklabels():
    item.set_rotation(45)


# In[42]:


dim = (30, 10)
fig, ax = plt.subplots(figsize=dim)
tag_plot = sns.countplot(x="Tags", ax=ax, data=dataset)

for item in tag_plot.get_xticklabels():
    item.set_rotation(45)


# In[35]:


dim = (20, 10)
fig, ax = plt.subplots(figsize=dim)
sns.countplot(x="Year", ax=ax, data=dataset)


# From the plots we made, we can tell a few things about power outages:
#  * Outages due to severe weather are common almost every year, and is by far the biggest cause in outages.
#  * We had a pretty bad year at 2011, but it eventually got better.
#  * As times goes by, demand loss increases, even if we have less outages (which is expected, since there are more people on the grid)
#  
# Since there are so many severe weather occurrences, it may not be worth to keep the Tags column; we can't make any meaningful prediction with it, since we don't have the number of times that severe weather doesn't cause an outage. 
#  
# With that in mind, what is left for us to analyze? Well, we have a few options:
#  * Build a regression model to predict demand loss.
#  * Build a regression to predict the duration of an outage. 
#  * Build a classifier to predict what part of the day (e.g. morning, afternoon, night, etc) or month of the year outages are more likely to occur.
#  * Find out which respondents solve outages faster.
#  * Only use data which has number of customers affected known, and find the relation between this data and the other features.
