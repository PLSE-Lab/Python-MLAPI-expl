#!/usr/bin/env python
# coding: utf-8

# # Quick initial look and analysis

# In[35]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 100)


# #### Load the data
# * kaggle doesn't display the raw data file correctly

# In[36]:


df = pd.read_csv('../input/kds_full_monitoring.csv',parse_dates=["to_timestamp"],infer_datetime_format=True,low_memory=False)
print(df.shape)
df.head()


# * Check for unary columns: 

# In[37]:


df.nunique()


# # Create target column
# * We could look at visitors per country/university/origin , or other forms of analyses (e.g. predicting user attributes). 
#     * We'll start with something simple - predicting total users per day. 
#         * We'll need to see if this covers multiple websites/portal.

# In[38]:


df_entries = df.set_index("to_timestamp").resample("D")["user_id"].count()


# In[39]:


print("total days:",df_entries.shape[0])
print("Days with visitors: :",(df_entries>0).sum())


# * Drop the unary column(s) from our primary data

# In[40]:


df.drop(["user_id","domain_id"],axis=1,inplace=True)


# In[41]:


df.head()


# # Alternate target (Session level): Predict if user is a bot

# In[42]:


df.bot.describe()


# In[43]:


# Note that we'll need to remove the org_name, if we want a model that can identify "stealth" bots!

df[df.bot==True].head()


# # Export data

# In[44]:


df_entries.to_csv("kds_Daily_webtraffic.csv")


# In[45]:


df.sample(frac=0.15).to_csv("kds_sample_webtraffic.csv.gz",compression="gzip")


# In[ ]:




