#!/usr/bin/env python
# coding: utf-8

# This is my first attempt at data visualization.  I am following the tutorials provided by kaggle found [here](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-1?utm_medium=email&utm_source=intercom&utm_campaign=dashboarding-event).

# In[ ]:


import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 20]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('../input/daily-inmates-in-custody.csv')


# In[ ]:


#clean data by removing rows containing NaN  
data = data[pd.notnull(data['RACE'])]


# In[ ]:


data.info()


# I decided to begin by examining the age distribution.

# In[ ]:


plt.figure(figsize=(20,7))
plt.hist(pd.to_numeric(data['AGE']), facecolor='black', bins=100)
plt.title("Distribution of Ages")
plt.xlabel("Age of Inmates")
plt.ylabel("Count")
plt.show()


# ****I was curious to see what time of the day were the arrests being made.****

# In[ ]:


data['Date_reviewed'] = pd.to_datetime(data['ADMITTED_DT'])

plt.figure(figsize=(20,7))
plt.hist(data['Date_reviewed'].dt.hour,
             alpha=.8, bins=100)
plt.title('Arrests Made Every Hour')
plt.xlabel('hour of day')
plt.ylabel('number of records')
plt.show()


# **I also wanted to know if there a corelation between the race of the datainee and the hour of the day in which the crime was comitted.**

# In[ ]:


plt.figure(figsize=(20,7))
sns.countplot(x=data['Date_reviewed'].dt.hour, hue=data['RACE'], palette="Set2")
plt.title('Relationship of Race/Hour')
plt.xlabel('Hour of Day')
plt.ylabel("Number of Records")
plt.show()

