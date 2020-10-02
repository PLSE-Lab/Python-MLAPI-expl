#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#bring in the hard drive dataset
df_train = pd.read_csv('../input/harddrive.csv')


# In[3]:


#Check out the columns
df_train.columns


# In[4]:


#descriptive statistics summary
df_train.describe()


# In[5]:


#Look into failure distributions
df_train['failure'].value_counts()


# In[6]:


#Box Plot of capacity_bytes and failure
plt.figure(figsize=(10,4))
sns.boxplot(x=df_train['failure'])

plt.figure(figsize=(10,4))
sns.boxplot(x=df_train['capacity_bytes'])


# In[7]:


#Obtain the failed hard rive disks into dataframe
hard_drive_failure_data = df_train[df_train['failure'] == 1]
hard_drive_failure_data.head()


# In[8]:


#Checkout the features and non-nulls
hard_drive_failure_data.info()


# In[9]:


#Find the lifetime of failed hard drives
lifetimes = []
for index, row in hard_drive_failure_data.iterrows():
    start_date = hard_drive_failure_data.iloc[0]['date']
    #print(start_date)
    end_date = row['date']
    #print(end_date)

    time_difference = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    lifetimes.append(time_difference.days)
    
hard_drive_failure_data['lifetime'] = lifetimes
hard_drive_failure_data.head()


# In[10]:


#Fliter failed hard drives that have lifetime larger than 0
filtered_hard_drive_failure_data = hard_drive_failure_data[hard_drive_failure_data['lifetime'] > 0]
filtered_hard_drive_failure_data.info()


# In[11]:


#Distribution of the lifetime
sns.distplot(filtered_hard_drive_failure_data['lifetime']); 


# In[12]:


#Box Plot of the lifetime
plt.figure(figsize=(10,4))
sns.boxplot(x=filtered_hard_drive_failure_data['lifetime'])


# In[20]:


filtered_hard_drive_failure_data.to_csv('hard_drive_with_lifetime.csv', index = False)


# In[13]:


#Save the dataframe with lifetime as csv file
#hard_drive_with_lifetime = filtered_hard_drive_failure_data.to_csv(index=False)

