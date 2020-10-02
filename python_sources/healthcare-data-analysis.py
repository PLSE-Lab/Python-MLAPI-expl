#!/usr/bin/env python
# coding: utf-8

# A software product is used to free up capacity in operating rooms and create a much more transparent and surgeon-centric process for measuring Operating Room (OR) utilization. The goal of this analysis is to showcase the effectiveness of the marketplace provided by this product. We have the data for the 'Exchange' transactions that happened this product for an existing customer[](http://). 

# **EXECUTIVE SUMMARY**
# 
# **Goal**
# 
# *To increase Surgeon OR Access **via Marketplace for OR time.*
# 
# **Need**
# 
# *Over 10,000 unutilized hours every 100 days with Block Scheduling.*
# 
# **Our Results (88 days)**
# 
# *250+  Transfers processed (2000+ hours)*
# 
# *1100+ Requests fulfilled (4500+ hours)*
# 
# *1150+ Releases approved (9000+ hours)*
# 
# **Key Advantages**
# 
# *>85% Requests fulfilled*
# 
# *>50% of utilization of released time*
# 
# *Average Request processing time < 10 hours*

# In[ ]:


#Import the required libraries
import os
import random
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Connect to database and extract the exchange_transactions table 
os.chdir("/kaggle/input/healthcare-or-transactions-data")
sqlite_file = 'Healthcare OR Transactions Data.db'
conn = sqlite3.connect(sqlite_file)
extr = pd.read_sql_query("SELECT * FROM exchange_transactions", conn)


# In[ ]:


print("Dimensions of the exchange transactions table are: {}".format(extr.shape))


# In[ ]:


extr.describe()


# From the description we can note the following points:
# * We have a total of 6965 but only 6935 unique transaction ids. Some transactions have been repeated in the dataset.
# * Some transactions have a parent transaction.
# * A parent transaction may be the parent of 1 or more transactions.
# * There are 275 schedulers in the system.
# * There are 357 surgeons in the system.
# 
# We will proceed from here by deleting the repeated/duplicate transactions, converting the columns containning date and time to datetype in Python and add a column corresponding to the duration of time blocked scheduling time for the transaction.

# In[ ]:


#Drop duplicate rows
extr.drop_duplicates(keep='first', inplace=True)


# In[ ]:


#Convert relevant columns to datetime type and add a column for duration of each transaction
extr['created_datetime'] = pd.to_datetime(extr['created_datetime'])
extr['snapshot_date'] = pd.to_datetime(extr['snapshot_date'])
extr['start_time'] = pd.to_datetime(extr['start_time'])
extr['end_time'] = pd.to_datetime(extr['end_time'])
extr['duration'] = (extr['end_time'] - extr['start_time']).dt.total_seconds()/3600


# In[ ]:


#Calculate number of days of operation of iQueue
max(extr['snapshot_date']) - min(extr['snapshot_date'])


# In[ ]:


extr.shape[0] #Check the number of rows in the data after removing duplicates


# In[ ]:


extr.head() #Look at the top few rows of data


# Here we notice that some transactions do not have a parent transaction. We will filter such 'fresh' transactions and look at the actions performed for them followed by looking at the count of locations and rooms.

# In[ ]:


#Filter the transactions that do not have a parent transaction
fresh_tr = extr[extr['parent_transaction_id'].isnull()]
fresh_tr['action'].value_counts()


# In[ ]:


fresh_tr['location'].value_counts() #Count of locations


# In[ ]:


fresh_tr['room_name'].value_counts() #Count of rooms (top and bottom few)


# It is evident that the total of 2772 'fresh' transactions are either of type REQUEST, RELEASE or TRANSFER.
# 
# Let us look at the duration of each type of transactions by plotting histograms.

# In[ ]:


requests_data = fresh_tr[fresh_tr['action']=='REQUEST']
req_dur = requests_data['duration']
release_data = fresh_tr[fresh_tr['action']=='RELEASE']
rel_dur = release_data['duration']

bins = np.linspace(0, 12, 30)
fig, ax = plt.subplots(1, 2)
ax[0].hist(req_dur, bins, alpha=0.5, color = 'black')
ax[0].set_ylabel('Count')
ax[0].set_xlabel('Request duration')
ax[0].set_ylim(0, 250)
ax[1].hist(rel_dur, bins, alpha=0.5, color = 'blue')
ax[1].set_ylim(0, 250)
ax[1].set_xlabel('Release duration')
plt.show()


# In[ ]:


transfer_data = fresh_tr[fresh_tr['action']=='TRANSFER']
plt.hist(transfer_data['duration'], color = 'green')
plt.xlabel('Transfer duration')
plt.ylabel('Count')
plt.show()


# Next, we will filter the transactions having a parent transaction and look at the actions performed in them.

# In[ ]:


notnullpar_tr = extr[extr['parent_transaction_id'].notnull()]
notnullpar_tr['action'].value_counts()


# It is evident that the follow up transactions are of one of the types: MARK_UPDATED, APPROVE_REQUEST, APPROVE_TRANSFER, DENY_REQUEST, DENY_RELEASE, DENY_TRANSFER.
# 
# Next, we will filter some columns from the fresh transactions table, some from table having parent transactions and perform a left join on the two tables. This will give us details of the downstream transactions taking place on every parent transaction.

# In[ ]:


fresh = fresh_tr[['transaction_id', 'created_datetime', 'action', 'duration']]
notnull = notnullpar_tr[['parent_transaction_id', 'created_datetime', 'action']]
fresh_leftjoin_notnull = fresh.merge(notnull, left_on='transaction_id', right_on='parent_transaction_id', how='left')
fresh_leftjoin_notnull.sort_values('transaction_id').head()


# Now we can look at the downstream transactions for each type of parent transaction by filtering data from this table along with the processing time for each!

# **1) TRANSFER**

# In[ ]:


#Filter transactions having Transfer and Mark updated actions
transfer_upd = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='TRANSFER') & (fresh_leftjoin_notnull['action_y']=='MARK_UPDATED')]
transfer_upd.shape[0]


# In[ ]:


#Filter transactions having Transfer and Approve transfer actions
transfer_app = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='TRANSFER') & (fresh_leftjoin_notnull['action_y']=='APPROVE_TRANSFER')]
transfer_app.shape[0]


# In[ ]:


#Filter transactions having Transfer and Deny transfer actions
transfer_deny = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='TRANSFER') & (fresh_leftjoin_notnull['action_y']=='DENY_TRANSFER')]
transfer_deny.shape[0]


# We note here that from the 263 TRANSFER transactions, 256 were APPROVED and MARKED UPDATED in that order. The remaining 7 transfer transactions were denied. 97.3% percent of TRANSFER transactions were APPROVED.

# In[ ]:


transfer_app['duration'].sum()


# In[ ]:


transfer_upd.head()


# In[ ]:


transfer_updated = transfer_upd.copy()
transfer_updated.loc[:,'process_time'] = pd.Series((pd.to_datetime(transfer_updated['created_datetime_y']) - pd.to_datetime(transfer_updated['created_datetime_x'])).dt.total_seconds()/3600)
plt.hist(transfer_updated['process_time'])
plt.xlabel('Transfer processing time')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


transfer_updated['process_time'].mean()


# Over 2085 hours of time was successfully transferred during the 88 days of operation of our product. On an average, we were able to transfer 27.3 hours of time only through TRANSFERS with an average processing time of 7.37 hours with only a few outliers which take a few days to process.

# **2) RELEASE**

# In[ ]:


#Filter transactions having Release and Mark updated actions
release_upd = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='RELEASE') & (fresh_leftjoin_notnull['action_y']=='MARK_UPDATED')]
release_upd.shape[0]


# In[ ]:


#Filter transactions having Release and Deny release actions
release_den = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='RELEASE') & (fresh_leftjoin_notnull['action_y']=='DENY_RELEASE')]
release_den.shape[0]


# We see that from the 1200 RELEASE transactions, 1178 were MARKED UPDATED. The remaining 23 transfer transactions were denied. 99% percent of RELEASE transactions were APPROVED (directly MARKED UPDATED in this case).

# In[ ]:


release_upd['duration'].sum()


# In[ ]:


release_updated = release_upd.copy()
release_updated.loc[:,'process_time'] = pd.Series((pd.to_datetime(release_updated['created_datetime_y']) - pd.to_datetime(release_updated['created_datetime_x'])).dt.total_seconds()/3600)
plt.hist(release_updated['process_time'])
plt.xlabel('Release processing time')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


release_updated['process_time'].mean()


# About 9434 hours of time was released during the 88 days of operation of our product. On an average, 107 hours of time was RELEASED per day. The average processing time to process a release was 7.15 hours.

# **3) REQUEST**

# In[ ]:


#Filter transactions having Request and Mark updated actions
req_upd = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='REQUEST') & (fresh_leftjoin_notnull['action_y']=='MARK_UPDATED')]
req_upd.shape[0]


# In[ ]:


#Filter transactions having Request and Approve transfer actions
request_app = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='REQUEST') & (fresh_leftjoin_notnull['action_y']=='APPROVE_REQUEST')]
request_app.shape[0]


# In[ ]:


#Filter transactions having Request and Deny transfer actions
request_deny = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='REQUEST') & (fresh_leftjoin_notnull['action_y']=='DENY_REQUEST')]
request_deny.shape[0]


# We note here that from the 1309 REQUEST transactions, 1134 were APPROVED and MARKED UPDATED in that order. The remaining 175 REQUEST transactions were denied. 86.6% percent of REQUESTS were APPROVED.

# In[ ]:


request_app['duration'].sum()


# In[ ]:


request_approved = request_app.copy()
request_approved.loc[:,'process_time'] = pd.Series((pd.to_datetime(request_approved['created_datetime_y']) - pd.to_datetime(request_approved['created_datetime_x'])).dt.total_seconds()/3600)
plt.hist(request_approved['process_time'])
plt.xlabel('Request processing time')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


request_approved['process_time'].mean()


# About 4668 hours of time was released during the 88 days of operation of our product. On an average, 53 hours of time was REQUESTED per day. 

# **Conclusions:**
# * 11500 hours of total time was available for transactions in 88 days.
# * 6700 hours of time transferred via the transactions.
# * Our product is effective in helping reduce unutilized time by more than half.
# * The average processing time for a request is 10.2 hours, it is even lesser (7 hours) for release and transfer.
