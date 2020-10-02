#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#  LTFS receives a lot of requests for its various finance offerings that include housing loan, two-wheeler loan, real estate financing and micro loans. The number of applications received is something that varies a lot with season. Going through these applications is a manual process and is tedious. Accurately forecasting the number of cases received can help with resource and manpower management resulting into quick response on applications and more efficient processing.
# 
# You have been appointed with the task of forecasting daily cases for next 3 months for 2 different business segments at the country level keeping in consideration the following major Indian festivals (inclusive but not exhaustive list): Diwali, Dussehra, Ganesh Chaturthi, Navratri, Holi etc. (You are free to use any publicly available open source external datasets). Some other examples could be:
# 
# Weather
# Macroeconomic variables
#  Note that the external dataset must belong to a reliable source.

# Data Dictionary
# The train data has been provided in the following way:
# 
# For business segment 1, historical data has been made available at branch ID level
# For business segment 2, historical data has been made available at State level.
#  
# 
# Train File
# Variable	Definition
# application_date	Date of application
# segment	Business Segment (1/2)
# branch_id	Anonymised id for branch at which application was received
# state	State in which application was received (Karnataka, MP etc.)
# zone	Zone of state in which application was received (Central, East etc.)
# case_count	(Target) Number of cases/applications received
#  
# 
# **Test File
#  Forecasting needs to be done at country level for the dates provided in test set for each segment.
#  
# Variable	Definition
# id	Unique id for each sample in test set
# application_date	Date of application
# segment	Business Segment (1/2)**

# ### Regression task

# In[ ]:


data = pd.read_csv('/kaggle/input/train_fwYjLYX.csv')
test = pd.read_csv('/kaggle/input/test_1eLl9Yf.csv')


# In[ ]:


data.head()


# In[ ]:


test.head()


# In[ ]:


data['application_date'] = pd.to_datetime(data['application_date'])
data['year'] = data['application_date'].dt.year
data['month'] = data['application_date'].dt.month
data['day_of_year'] = data['application_date'].dt.dayofyear
data['weekday'] = data['application_date'].dt.weekday


# In[ ]:


data.head()


# ## We have data from dates 2017-04-01 to 2019-07-23

# In[ ]:


ax = sns.countplot(x = 'year', data = data)
plt.title('count of data points given in respective years')
plt.xlabel('years')
plt.ylabel('frequence')
for p in ax.patches:
        ax.annotate('%{:.1f}'.format(p.get_height()/data.shape[0]*100), (p.get_x() + 0.1, p.get_height()))
plt.show()


# ## Most of the data given is of year 2018

# ## Analysing Count cases datewise

# In[ ]:


f, axes = plt.subplots(2,2)
f.set_size_inches(20,20)

sns.barplot(y="case_count", x= "year", data=data,  orient='v' , ax=axes[0,0],)
sns.barplot(y="case_count", x= "month", data=data,  orient='v' , ax=axes[0,1])
sns.barplot(y = 'case_count', x='weekday', data = data, orient = 'v' , ax = axes[1,0])
sns.barplot(y = 'case_count', x='day_of_year', data = data, orient = 'v' , ax = axes[1,1])
plt.show()


# # Inferences
# ## 1. Value of case_count is increasing every year.
# ## 2. Some trend also observed in months, with 4, 5 having comparitevly less value
# ## 3. Weekday 6 has less values i.e could be a holiday
# ## 4. In the "day_of_year" column definite trend is observed. Middle of months having higher values -of case_count

# # Analysis segment wise

# In[ ]:


ax = sns.barplot(x = ['1', '2'], y = [list(data['segment'].value_counts())[0], list(data['segment'].value_counts())[1]])
plt.title('% of segment')
plt.xlabel('segment')
plt.ylabel('frequency')
for p in ax.patches:
    ax.annotate('%{:.1f}'.format(p.get_height()/data.shape[0]*100), (p.get_x() + 0.1, p.get_height()))
plt.show()


# ## 83.2 % data has segment "1" and only 16.8% has segment 2

# In[ ]:


ax = sns.barplot(y="case_count", x= "segment", data=data,  orient='v')
for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x() + 0.1, p.get_height()))
plt.show()


# ## segment 1 has average value of 32.7 case_count while segment 2 has 942.3

# # Analysis branch id

# In[ ]:


data['branch_id'].nunique()


# ## We have a total of 83 branch_id with each branch id only belonging to particular state

# In[ ]:


f = plt.figure()
f.set_size_inches(25,10)
sns.barplot(y="case_count", x= "branch_id", data=data,  orient='v' )
plt.show()


# In[ ]:


# branch_id having highest count_case 
data.groupby('branch_id')['case_count'].mean().sort_values(ascending = False).iloc[0:5]


# In[ ]:


data.groupby('branch_id')['case_count'].mean().sort_values().iloc[0:2]


# ## branch_id 269 and 263 have no applications

# In[ ]:


# plotting trends


# In[ ]:


data_new = pd.read_csv('/kaggle/input/train_fwYjLYX.csv')
data_new.index = pd.to_datetime(data_new.application_date)

monthly = data_new.resample('M').mean()
weekly = data_new.resample('W').mean()
dayly = data_new.resample('D').mean()

fig, axs = plt.subplots(3,1) 
#hourly.Count.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0]) 
dayly.case_count.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[0]) 
weekly.case_count.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[1]) 
monthly.case_count.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[2]) 
plt.show()


# ## Inference
# ## 1. on daily case_count plot season is observed 
# ## 2. This season is more prominant on monthly weekly
# ## 3. In the end i.e july 2019 there is sudden increase in case_count

# # Analysis state

# In[ ]:


states = data['state'].unique()


# In[ ]:


f = plt.figure()
f.set_size_inches(25,10)
g = sns.barplot(y="case_count", x= "state", data=data,  orient='v' , palette= sns.color_palette("muted"), )

plt.title('Count_Case state wise')
plt.xlabel('state')
plt.ylabel('caount_case')

plt.show()


# ## Tamil nadu has highest applications

# In[ ]:


branch_state = []
for state in states:
    branch_state.append(data[data['state'] == state]['branch_id'].nunique())


# In[ ]:


f = plt.figure()
f.set_size_inches(25,10)
g = sns.barplot(y= branch_state, x= states)

plt.title('branch_id state wise')
plt.xlabel('state')
plt.ylabel(' Number of branch_id')

plt.show()


# ## Each state has unique set of branche_id (i.e branches)
# ## Maharashtra has highest number of branches follwed by west bengal and Gujarat

# # Zone Analysis

# In[ ]:


data['zone'].unique()


# In[ ]:


for state in states:
    print(state, "occurs in ", data[data['state'] == state]['zone'].nunique(), "zones")


# ## Thus Orissa occurs in 2 zones rest occur in 1 zone

# # Other data sets can also be merged, I have also attached the test data you can have a look !
# # Suggestions welcomed
# # Please upvote
# # THANKYOU 

# In[ ]:




