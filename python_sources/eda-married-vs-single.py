#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import h2o
h2o.init()


# In[ ]:


data=h2o.import_file("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")
data.head()


# In[ ]:


data.dim


# In[ ]:


data.columns


# In[ ]:


data["default.payment.next.month"].table()


# MARRIAGE: Marital status (1=married, 2=single, 3=others)

# In[ ]:


data["MARRIAGE"].hist()


# There is around the same amount of people who are married as to single -> let's compare their repayment habits 

# In[ ]:


data["MARRIAGE"].table()


# ANALYSIS OF MARRIED GROUP REPAYMENT BEHAVIOR: 

# In[ ]:


married = data[data["MARRIAGE"]==1]
married


# let's look at defaults within married people:

# In[ ]:


married["default.payment.next.month"].table()


# the status of no default has higher frequency-> let's observe the behavior and look for correlation

# look at married people who have paid duly in April, 2005 (the earliest month recorded)

# In[ ]:


early_paying_married = married[married["PAY_6"]==-1]
early_paying_married


# married people who pay the first month -> do they show similar behavior in future (month of September = the last month of observation) -> Yes, they do because the histogram shows below the highest frequecy of "pay duly" which means they paid their credit in the last month observed.

# In[ ]:


early_paying_married["PAY_0"].hist()


# In[ ]:


early_paying_married["PAY_0"].table()


# looking below:
# married people who pay duly the first month show to have low frequency of defaults compared no default status

# In[ ]:


early_paying_married["default.payment.next.month"].table()


# ANALYSIS OF SINGLE GROUP REPAYMENT BEHAVIOR:

# In[ ]:


single = data[data["MARRIAGE"]==2]
single


# look at single who have paid duly in September, 2005 (the earliest month recorded)

# In[ ]:


early_paying_single = single[single["PAY_6"]==-1]
early_paying_single


# single people who pay the first month -> do they show similar behavior in future (month of September = the last month of observation) -> Yes, they do because the histogram shows below the highest frequecy of "pay duly" which means they paid their credit in the last month observed.

# In[ ]:


early_paying_single["PAY_0"].hist()


# In[ ]:


early_paying_single["PAY_0"].table()


# looking below: single people who pay duly the first month show to have low frequency of defaults compared no default status

# In[ ]:


early_paying_single["default.payment.next.month"].table()


# CONCLUSION: within both groups we see that if they "pay duly" the first month of observation, they most likley will "pay duly" the last month of observation. Furthermore, they also have less defaults compared to defaults.

# ANALYSIS OF MARRIED GROUP NON-REPAYMENT BEHAVIOR: 

# In[ ]:


married["PAY_6"].table()


# DATA PROBLEM: what do -2 and 0 mean
# 
# * no existence of 1 = payment delay for one month in data for married people
# 
# found in discussion:
# 
# Answering from knowledge of the industry, not with information directly from the dataset creator: I presume that the values from -2 to 0 more precisely mean the following:
# 
# -2 = Balance paid in full and no transactions this period (we may refer to this credit card account as having been 'inactive' this period)
# 
# -1 = Balance paid in full, but account has a positive balance at end of period due to recent transactions for which payment has not yet come due
# 
# 0 = Customer paid the minimum due amount, but not the entire balance. I.e., the customer paid enough for their account to remain in good standing, but did revolve a balance

# APPLY this to married people analysis: will they not pay late the last month if they have delayed payment the first month of observation

# In[ ]:


late_paying_married = married[married["PAY_6"]==2]
late_paying_married


# In[ ]:


late_paying_married["PAY_0"].hist()


# In[ ]:


late_paying_married["PAY_0"].table()


# looking below: married people who have the payment status 2 (payment delay for two month)in the first month of observation show to have higher frequency of defaults compared no defaults.

# In[ ]:


late_paying_married["default.payment.next.month"].table()


# CONCLUSION: repayment statuses of married people who paid 2 monthes late the 1st month of observation are likely to have repayment statuses 0,1,2 
# 
# 0 = Customer paid the minimum due amount, but not the entire balance. I.e., the customer paid enough for their account to remain in good standing, but did revolve a balance
# 
# 1=payment delay for one month 
# 
# 2=payment delay for two months

# UNCLEAR CONCLUSION: so we will look at histograms of other month's payment status within this married group 

# In[ ]:


late_paying_married["PAY_2"].hist()


# In[ ]:


late_paying_married["PAY_2"].table()


# MORE CLEAR CONCLUSION: Here, we see in the month before the last month of observation a lot of the late paying students congregated around payment statuses of 0 and 2.
# 
# 0 = Customer paid the minimum due amount, but not the entire balance. I.e., the customer paid enough for their account to remain in good standing, but did revolve a balance
# 
# 2=payment delay for two months
# 
# -> SAME CONCLUSION AS LATE_PAYING_STUDENTS

# ANALYSIS OF SINGLE GROUP NON-REPAYMENT BEHAVIOR:

# In[ ]:


single["PAY_6"].table()


# will single people who have delayed payments in the first month show to have same behvior in the last month of observation
# * payment status 2 seems to be the most frequent payment status that confirms a delayed payment so I will observe that payment status 

# In[ ]:


late_paying_single = single[single["PAY_6"]==2]
late_paying_single


# In[ ]:


late_paying_single["PAY_0"].hist()


# In[ ]:


late_paying_single["PAY_0"].table()


# looking below: single people who have the payment status 2 (payment delay for two month)in the first month of observation show to have almost the same frequency of defaults and no defaults

# In[ ]:


late_paying_single["default.payment.next.month"].table()


# 

# CONCLUSION: repayment statuses of single people who have the payment status of 2 the 1st month of observation are likely to have repayment statuses 0,1,2 
# * same conclusion as the late paying married group
