#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

figsize = (16,8)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # SMPD Arrest Data Analysis
# ---
# 
# ## **Things I can answer with arrest data:**
# 
# * How many arrests on average does SMPD have each year?
# * Has that number gone up or down over the years? 
# * How many black people does SMPD arrest each year? 
# * How many white people does SMPD arrest each year?
# * How many other races does SMPD arrest each year? 
# * How many men does SMPD arrest each year?
# * How many women does SMPD arrest each year?
# * How many other does SMPD arrest each year?
# * What is the average age of arrest over the past three years?
# 
# ## **For after I get incident reports:**
# 
#  * How many citations turn into arrests? 
# 
# ## **For after we get conviction data:**
# 
# * How many convictions do arrests from SMPD lead to each year? 
# 
# ## **For deeper analysis:**
# 
# * How does that compare to other races? 
# * How does that compare to other districts of the same size?
# * What is the individual black arrest rate per officer? 
# * Do some offers arrest a certain race at an alarming rate?
# * What is an alarming rate?
# * Why is this information not more accesible to the public? 
#  
# 
# ## **Data integrity questions:**
# 
# * No hispanic race designation? 
# * Is SMPD including hispanics in their white totals? If so-- why?
# * Are officers putting in their own badge numbers? 
# * Why are there blank arrest reasons
# * Why are there some arrests with no officer names
# * What is the difference between an A and C in the arrest number 
# * How do you know if these are arrests of the same person multiple times? 
# 

# ****
# # About the data... 
# ---
# 
# SMPD Publishes arrest lists daily (See: [SMPD Media Arrest List](https://www.sanmarcostx.gov/DocumentCenter/View/5913/SMPD-Media-Arrest-List--06-23-2020-PDF?bidId= "Data Goldmine!") & [SMPD Incident List](https://www.sanmarcostx.gov/DocumentCenter/View/5875/SMPD-Incident-Blotter-06-29-2020-pdf "This one is not as detailed")) I submitted an open records request for these going back three full years (2017, 2018, 2019) & will be using this for my dataset. 
# 
# Fields from these reports:
# 
# Incident List Fields | Media Arrest List Fields| Conviction Data
# ---| ---| ---
# Incident #| Incident #|.|
# Time Reported| Name|.|
# Activity| Arrest Date|.|
# Disposition| Officer|.|
# Location| Age|.|
# . | Sex|.|
# . | Race|.|
# . | Officer|.| 
# . | Charge|.|

# I have not received incident reports yet but have requested them. I have no idea how to even get conviction data. 
# 
# So first we will look at the cleaned arrest lists CSV. The original data was some kind of report export, Daniel standardized it using a formula & regular expression in google sheets to remove blank space and standardize officer (because it looks like officers can type in their own badge/lastname so it was a mess.)  

# # Read my file, Panda! 

# In[ ]:


# Now that my data is pretty, I am telling Panda to read it & calling it 'arrest_data' from this point further. I don't actually know if this needs to be here or not... but I'm gonna keep it. 

arrest_data = pd.read_csv('../input/smpd-data/arrest list - total.csv')


# In[ ]:


# These are the columns (fields) available on arrest_data. An index I suppose: 

arrest_data.columns


# Next I'll calculate some simple totals using value.counts:

# ## Helper functions

# In[ ]:


# Compute statistics regarding the relative quanties of arrests, warnings, and citations (work in progress)

def compute_stats(data):
    n_arrests = len(data)

    return(pd.Series(data = {
        '# of arrests': n_arrests
    }))


# In[ ]:


# Attempting to calculate percent of arrests by race

def prcnt(x, y):
    if not x and not y:
       print('x = 0%\ny = 0%')
    elif x < 0 or y < 0:
       print("can't be negative!")
    else:
       total = 100 / (x + y)
       x *= total
       y *= total
       print('x = {}%\ny = {}%'.format(x, y))
        
        
arrest_data['Officer_Last_Name'].value_counts(prcnt)


# ## Total arrests 2017 - 2019

# In[ ]:


#I have data for 3 full years of arrests. How many total arrests for that period were there?

arrest_data['Incident_Nr'].count()


# In[ ]:


#Naming the variable? does this need to be done?

total_arrests = arrest_data['Incident_Nr'].count()


# ## Arrests by officer

# In[ ]:


# Now I am pulling the 'Officer' field my arrest_data & using the value.count function to see how many arrests each officer has made. 
# The square brackets call the list of officers and then counts the values. 

arrest_data['Officer'].value_counts()


# ## Arrests by gender

# In[ ]:


# same as above but for gender

arrest_data['Gender'].value_counts()


# In[ ]:


# Create chart of arrests by gender using compute_stats

arrest_data.groupby('Gender').apply(compute_stats)


# There is an estimated 64,776 people in San Marcos, Texas as of 2019. 
# 51.3% of the population is female, or about 34,000 people. 
# 
# Over three years, 2,404 females were arrested. Or about 30% of total arrests. 
# There was an average of 800 female arrests per year. 
# 
# 
# 
# 

# ## Arrests by race

# In[ ]:


# race 

arrest_data['Race'].value_counts()


# In[ ]:


# Create chart of arrests by race using compute_stats

arrest_data.groupby('Race').apply(compute_stats)


# Race breakdown in San Marcos, Texas:
# 
# * Black or African American alone, 6%
# * American Indian and Alaska Native alone, 0%
# * Asian alone, 2%
# * Native Hawaiian and Other Pacific Islander alone, 0%
# * Two or More Races, 3%
# * Hispanic or Latino, 40%
# * White alone, not Hispanic or Latino, 48%

# ## Arrests by age

# In[ ]:


# age

arrest_data['Age'].value_counts()


# In[ ]:


# Create chart of # of arrests by age using compute_stats (asc by age)

arrest_data.groupby('Age').apply(compute_stats)


# ## Arrest Frequency by Age & Race

# In[ ]:


# Not accurate yet. Do not use. 

fig, ax = plt.subplots()
ax.set_xlim(14,83)
for race in arrest_data['Race'].unique():
    s = arrest_data[arrest_data['Race'] == race]['Age']
    s.plot.kde(ax=ax, label=race)
ax.legend()


# ## Arrest Frequency by Race & Officer

# In[ ]:


# Not accurate yet. Do not use. 

arrest_data.groupby(['Officer_Last_Name','Race']).apply(prcnt)

