#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Importing necessary Libraries

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(rc={'figure.figsize':(16,8)})


# ### Loading the dataset 
#  * leads.csv file

# ## Table 1: Leads

# In[ ]:


leads = pd.read_csv('/kaggle/input/telecaller-operations-dataset/leads.csv', encoding='latin1')
leads


# ### Feature Selection or Handling
# 
# 1. Since leads id is uniquely identify each row, So we can drop this feature 
# 2. Since userId is used to refer internal database entry, This can also be dropped.
# 3. Similarly Name is not important.
# 4. Since phone number is all same in this data, or all different in the original data, It can be dropped.

# In[ ]:


leads['userId'].nunique()


# In[ ]:


leads1 = leads.drop(['id', 'userId', 'name', 'phoneNumber'], axis=1)
leads1


# ### Handle Datatype and Missing Values
# 
# 1. find how many missing values are there in city, state, source and isExternal
# 2. convert createdAt and receivedAt feature to datetime datatype
# 3. convert boolean isExternal into 0 and 1 integer datatype

# In[ ]:


leads1.dtypes


# In[ ]:


leads1['createdAt'] = pd.to_datetime(leads1['createdAt'])
leads1['receivedAt'] = pd.to_datetime(leads1['receivedAt'])
leads1.isExternal = leads1.isExternal.astype(int)
leads1.dtypes


# ### Insights
#  * We have to take <b>city</b> name as well while creating data because around <b>7.5 % </b> data doesn't have city mensioned
#  * Overall We are doing good in <b>Hyderabad, Bangalore, Chennai, Delhi/NCR, Mumbai, Delhi, Mysore</b>
#  * But We also need to improve in other cities as well

# In[ ]:


# leads1['city'].isnull().sum() ## 738 ## 7.44%
# leads1['city'].nunique() ## 61
leads1['city'].value_counts().head(10)


# ### Insights
#  * We have to take <b>state</b> name as well while creating data because around <b>14 %</b> data doesn't have state mensioned
#  * Overall We are doing good in <b>Telangana, Karnataka, Tamilnadu, Delhi, Maharashta</b>
#  * But We also need to improve in other states as well

# In[ ]:


# leads1['state'].isnull().sum() ## 1380 ## 13.91%
# leads1['state'].nunique() ## 20
leads1['state'].value_counts().head(10)


# ### Insights
#  * <b>Referrals, API, WhatsApp OptIn Form, Facebook</b> are the main sources of leads

# In[ ]:


# leads1['source'].isnull().sum() ## 2
# leads1['source'].nunique() ##108
leads1['source'].value_counts().head(20)


# ### Insights 1
# 
# * We have <b>13.58 %</b> External leads. This needs to be minimised.
# * We can also have a look from which cities or states these leads coming from, So that We can improve in those cities

# In[ ]:


# leads1['isExternal'].isnull().sum() ## 0
# leads1['isExternal'].nunique() ##2
leads1['isExternal'].value_counts(normalize=True)*100


# In[ ]:


leads2 = leads1[['city', 'isExternal', 'source']].groupby(['city','isExternal']).count()
leads2 = leads2.reset_index()
leads2 = leads2.pivot(index='city', columns='isExternal', values='source')
del leads2.columns.name
leads2 = leads2.reset_index()
leads2.columns=['city', 'LeadsByJobAssist', 'LeadsByExternal']


# ### Insights 2 Citywise Performance
# 
#  * We have are doing good in <b>Hyderabad, Bengalore, Chennai, Delhi/NCR, Mumbai and Delhi</b>

# In[ ]:


leads20 = leads2.sort_values(by='LeadsByJobAssist', ascending=False)
leads20.reset_index(inplace=True)
# leads20.to_csv('leads20.csv', index=False)
leads20.head(10)


# ### Insights 3 Citywise Improvements
# 
#  * We need to improve in <b>Bangalore and Mysore</b>

# In[ ]:


leads21 = leads2.sort_values(by='LeadsByExternal', ascending=False)
leads21.reset_index(inplace=True)
# leads21.to_csv('leads21.csv', index=False)
leads21.head(10)


# In[ ]:


leads3 = leads1[['state', 'isExternal', 'source']].groupby(['state','isExternal']).count()
leads3 = leads3.reset_index()
leads3 = leads3.pivot(index='state', columns='isExternal', values='source')
del leads3.columns.name
leads3 = leads3.reset_index()
leads3.columns=['state', 'LeadsByJobAssist', 'LeadsByExternal']


# ### Insights 4 Statewise Performance
# 
#  * We are doing good in <b>Telangana, Karnataka, Tamilnadu, Delhi, Maharashtra</b>

# In[ ]:


leads30 = leads3.sort_values(by='LeadsByJobAssist', ascending=False)
# leads30.to_csv('leads30.csv', index=False)
leads30.head(10)


# ### Insights 5 Statewise Improvements
# 
#  * We need to improve in <b>Karnataka</b>

# In[ ]:


leads31 = leads3.sort_values(by='LeadsByExternal', ascending=False)
# leads31.to_csv('leads31.csv', index=False)
leads31.head(10)


# ### Insights 6 Duration between when lead was received by us and the time of creation of the table enry
# 
#  * Maximun <b>317 days</b>
#  * Minimum <b>27 days</b>
#  * Average <b>162 days</b>
#  * We need to <b>minimise this duration</b>, i.e, we must get the entries in table as early we receive the leads as possible

# In[ ]:


leads1['duration'] = pd.to_timedelta(leads1['createdAt'] - leads1['receivedAt'])
leads1['duration'] = leads1['duration'].dt.days
print(leads1['duration'].max())
print(leads1['duration'].min())
print(leads1['duration'].mean())
leads1


# In[ ]:


leads1.drop(['duration'], axis=1, inplace=True)
leads11 = leads1.copy()


# ### Insights 7 When did entries created in the table
# 
#  * on 26 Dec 2019 between 17:00 to 19:00

# In[ ]:


leads11['createdAt_day'] = leads11['createdAt'].dt.day
leads11['createdAt_day'].value_counts()


# In[ ]:


leads11['createdAt_hour'] = leads11['createdAt'].dt.hour
leads11['createdAt_hour'].value_counts()


# ### Insights 8 When the leads was received by us
# 
#  * Dates are between <b>12 feb 2019 to 29 Nov 2019</b>
#  * Majority of people received on <b>2019-07-25, 2019-05-01, 2019-04-01, 2019-11-01, 2019-08-01, 2019-07-30, 2019-06-09, 2019-06-08, 2019-08-02, 2019-11-03</b> respectively

# In[ ]:


leads11['receivedAt_date'] = leads11['receivedAt'].dt.date
leads11['receivedAt_date'].value_counts().head(10)


# In[ ]:


leads11['receivedAt_date'].min()


# In[ ]:


leads11['receivedAt_date'].max()


# In[ ]:


leads11['receivedAt_dt_doy'] = pd.to_datetime(leads11['receivedAt'])
leads11['receivedAt_dt_doy'] = leads11['receivedAt_dt_doy'].dt.dayofyear
# leads11['receivedAt_dt_doy'].min() ## 43
# leads11['receivedAt_dt_doy'].max() ## 333


# In[ ]:


leads110 = leads11[leads11['isExternal']==0]
leads111 = leads11[leads11['isExternal']==1]


# In[ ]:


df_time110 = leads110[['receivedAt_dt_doy', 'isExternal']].groupby('receivedAt_dt_doy').count()
df_time110.reset_index(inplace=True)


# In[ ]:


df_time111 = leads111[['receivedAt_dt_doy', 'isExternal']].groupby('receivedAt_dt_doy').count()
df_time111.reset_index(inplace=True)


# In[ ]:


df_time11 = leads11[['receivedAt_dt_doy', 'isExternal']].groupby('receivedAt_dt_doy').count()
df_time11.reset_index(inplace=True)


# ### Insights 9 Number of leads via job assist is increasing over time
# 
#  * Assuming maximum number of leads via <b>job assist</b> in a day is 150

# In[ ]:


df_time110_1 = df_time110[df_time110['isExternal']<151]
sns.lmplot(y='isExternal', x='receivedAt_dt_doy', data=df_time110_1)


# ### Insights 10 Number of leads via external source is also increasing over time
# 
#  * Assuming maximum number of leads via <b>external source</b> in a day is 75

# In[ ]:


df_time111_1 = df_time111[df_time111['isExternal']<75]
sns.lmplot(y='isExternal', x='receivedAt_dt_doy', data=df_time111_1)


# ### Insights 11 Number of overall leads via external source as well as job assist is also increasing over time
# 
#  * Assuming maximum number of leads via <b>external source and job assist</b> in a day is 200

# In[ ]:


df_time11_1 = df_time11[df_time11['isExternal']<201]
sns.lmplot(y='isExternal', x='receivedAt_dt_doy', data=df_time11_1)


# ## Table 2: Telecallers

# In[ ]:


telecallers = pd.read_csv('/kaggle/input/telecaller-operations-dataset/telecallers.csv', encoding='latin1')
telecallers


# ### Insights 12 
# 1. We have 5 Telecallers in our Company, Those are <b>Amila, Sheeba, Islam, Rakshith, and Manasa</b>
# 2. All the entries are created in Telecallers Table at the same time i.e, 2019-12-25 at 5:50 AM

# * <b>We can delete phoneNumber and createdAt from Telecallers table as it is not required as of now</b>

# In[ ]:


telecallers = telecallers.drop(['phoneNumber', 'createdAt'], axis=1)
telecallers.columns = ['telecallerId', 'telecallerName']
telecallers


# ## Table 3: Lead_Calls

# In[ ]:


lead_calls = pd.read_csv('/kaggle/input/telecaller-operations-dataset/lead_calls.csv', encoding='latin1')
lead_calls


# ### Feature Selection or Handling
# 
# 1. Since lead_calls <b>id</b> is uniquely identify each row, So we can drop this feature 
# 2.  <b>telecallerId</b> of Lead_calls table is being replaced by <b>telecallerName</b> of Telecallers Table, Because we have only few telecallers which can be identified by their name also

# In[ ]:


lead_calls['id'].nunique()


# In[ ]:


lead_calls = pd.merge(telecallers, lead_calls, on='telecallerId', how='inner')
lead_calls = lead_calls.drop(['id', 'telecallerId'], axis=1)
lead_calls


# ### Insights 13
# * Calls made by telecallers are <b>not equally distributed</b>
# * <b>Sheela has called maximum 8770 times</b>, After that <b>Amila 3206 times</b>, and <b>few calls were from Rakshith 359</b>
# * There is <b>no call made from Islam and Manasa</b>, So We can ask them as well to make calls to the leads 

# In[ ]:


lead_calls['telecallerName'].value_counts()


# In[ ]:


lead_calls['createdAt'] = pd.to_datetime(lead_calls['createdAt'])
lead_calls['calledAt'] = pd.to_datetime(lead_calls['calledAt'])


# ### Insights 14 When did entries created in the table
# 
#  * on 26 Dec 2019 between 17:00 to 19:00

# In[ ]:


lead_calls['createdAt_dt_date'] = lead_calls['createdAt'].dt.date
lead_calls['createdAt_dt_date'].value_counts()


# In[ ]:


lead_calls['createdAt_dt_hour'] = lead_calls['createdAt'].dt.hour
lead_calls['createdAt_dt_hour'].value_counts()


# In[ ]:


lead_calls.drop(['createdAt_dt_date', 'createdAt_dt_hour', 'createdAt'], axis=1, inplace=True)


# ### Insights 15 time of the call
# 
#  * Dates are between <b>9 July 2019 to 30 Aug 2020</b>
#  * Majority of people received on <b>2020-05-01, 2020-04-01, 2020-06-01, 2019-10-10, 2019-08-06, 2019-07-30, 2019-08-13, 2019-08-20, 2019-07-22, 2019-07-19</b> respectively

# In[ ]:


lead_calls['calledAt_dt_date'] = lead_calls['calledAt'].dt.date
lead_calls['calledAt_dt_date'].value_counts().head(10)


# In[ ]:


lead_calls.drop(['calledAt'], axis=1, inplace=True)


# In[ ]:


lead_calls['calledAt_dt_date'].min()


# In[ ]:


import datetime
lead_calls[lead_calls['calledAt_dt_date']>datetime.date(2016, 8, 26)]['calledAt_dt_date'].min()


# In[ ]:


lead_calls['calledAt_dt_date'].max()


# ### Insights 16 Number of leads called how many times
# * telecallers called only <b>once to 8316</b> different leads <b>83.86 % leads</b>
# * telecallers called only <b>twice to 969</b> different leads <b>9.77 % leads</b>
# * telecallers called only <b>thrice to 534</b> different leads <b>5.38 % leads</b>
# * telecallers called only <b>4 times to 58</b> different leads <b>0.58 % leads</b>

# In[ ]:


# lead_calls['leadId'].value_counts().value_counts()
lead_calls['leadId'].value_counts().value_counts(normalize=True)*100


# In[ ]:


lead_calls

