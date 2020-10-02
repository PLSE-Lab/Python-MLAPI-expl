#!/usr/bin/env python
# coding: utf-8

# In[15]:



#Import necessary packages
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[7]:


train_spl = pd.read_csv("../input/train_sample.csv")
train_spl.info()


# 1. Overall information on sample training dataset:
#             *  100,000 records, 8 atrrtibutes in total
#             *  labels: 0 for not download the app; 1 for download the app
# 2.  Two subset specifically for records which download the app (dld_train_spl) and not download the app(no_dld_train_spl)

# In[8]:


#Create subset for download apps and NOT download apps
dld_train_spl = train_spl[train_spl['is_attributed']==1]
dld_train_spl.info()

no_dld_train_spl = train_spl[train_spl['is_attributed']==0]
no_dld_train_spl.info()


# * Among 100,000 records in the train sample dataset:
#             251 records download apps
#             99,749 records NOT download apps

# #### Take a look of ip address for those not download app.

# In[29]:


ip_no_dld = no_dld_train_spl["ip"].value_counts()
ip_no_dld[:10]


# In[27]:


#Plot the ip addresses which are not download apps
sns.countplot(x = "ip",  data = no_dld_train_spl,             order = ip_no_dld[:10].index).set(            xlabel = "ip address not download the app")


# * From the ip address amount, top 10 ip addresses clicked the ad but not download. they are: 5348, 5314, 73487, 73516, 53454, 114276, 26995, 95766, 17149, 100275
# * For the top 10 ip addresses which not download the app, find them from the dataset which download apps.

# In[30]:


#Extract the top 10 ip addresses(not download the app) from the download dataset
ip_in_dld = dld_train_spl.loc[dld_train_spl['ip'].isin(ip_no_dld[:10].index)]
ip_in_dld.shape

#Group by ip
groupby_ip_dld = ip_in_dld.groupby("ip").size()
groupby_ip_dld[:10]


# *  Then we can take a look of the download rate(download amount/total click amount) for each top 10 ip addresses.

# In[31]:


#In train sample set, extract the ip addresses whcih not download the app
#In this way, we can find the total click amount
ip_no_dld_in_all = train_spl.loc[train_spl['ip'].isin(ip_no_dld[:10].index)]
ip_no_dld_in_all.shape

#Group by ip
groupby_no_dld_ip=ip_no_dld_in_all.groupby("ip").size()
groupby_no_dld_ip

#Plot the total click amount by ip addresses
sns.countplot(x = 'ip', data = ip_no_dld_in_all,              order = ip_no_dld[:10].index).set(            xlabel = "ip address click amount")


# In[32]:


#Merge the download times and click times based on same ip addresses
click_dld_ = pd.merge(groupby_no_dld_ip.reset_index(),ip_no_dld[:10].reset_index(),                      left_on = "ip", right_on = "index").iloc[:,[0,1,3]]
click_dld_.columns = ['ip','click times', 'not download times']
click_dld_['download times'] = click_dld_['click times'] - click_dld_['not download times']

#Calculate the download rate
click_dld_['download rate'] = click_dld_['download times']/ click_dld_['click times']
click_dld_

#Average download rate for the 10 ip addresses
print("Average download rate is: " , click_dld_["download rate"].mean())


# * For those top 10 ip addresses(not download the app), they have the highest click amount, but only have extremely less download amounts.
# * In other words, for instance, for ip "5314", this ip address has the click times of 640, but  download only 3 times,  it means the download rate is only 3/640 = 0.004687

# #### Take a look of ip address for those download app.
# In order to have a better contrast of the click times and download times, we can use the download ip address as a contrast. 

# In[43]:


#Top 10 ip addresses which download the apps
ip_dld = dld_train_spl["ip"].value_counts()
ip_dld[:10]

ip_dld_in_all = train_spl.loc[train_spl['ip'].isin(ip_dld[:10].index)]

sns.countplot(x = 'ip', data = ip_dld_in_all,              order = ip_dld[:10].index).set(            xlabel = "ip address click amount")


# Ip addresses "5314" has a overwhelming amount click times compared with other ip addresses, but extremely less download times, so it'll affect the whole download rate for other ip addresses. So as ip address "100275". These two "spetical" ip address should be outlier.
# 
# Therefore, the following analysis will not take these two ip addresses into account

# In[50]:


#In train sample set, extract the ip addresses whcih download the app
ip_dld_in_all = train_spl.loc[train_spl['ip'].isin(ip_dld[2:12].index)]

#Group by ip
groupby_dld_ip=ip_dld_in_all.groupby("ip").size()
groupby_dld_ip

#Plot the total click amount by ip addresses
sns.countplot(x = 'ip', data = ip_dld_in_all,              order = ip_dld[2:12].index).set(            xlabel = "ip address click amount")


# In[51]:


#Merge the download times and click times based on same ip addresses
click_dld = pd.merge(groupby_dld_ip.reset_index(),ip_dld[2:12].reset_index(),                      left_on = "ip", right_on = "index").iloc[:,[0,1,3]]
click_dld.columns = ['ip','click times', 'download times']

#Calculate the download rate
click_dld['download rate'] = click_dld['download times']/ click_dld['click times']
click_dld

#Average download rate for the 10 ip addresses
print("Average download rate is: " , click_dld["download rate"].mean())


# The  download rate for an average in the 10 ip addresses(download the app) is 0.5683. 
# 
# In general,  each ip address usually click no more than 5 times, download only 1 time.

# **Summary:**
# 
# Based on the download rate comparing with download and NOT download the app, we assume a hypothesis that: if all the clicks are not fradulent, then we could expect a positive linear relationship between the click times and download times. 
# 
# In other words, if an ip adress has a huge click amount but  download amount is extremely sparse, it's more likely to be frau**dulent than a genuine one.

# In[ ]:




