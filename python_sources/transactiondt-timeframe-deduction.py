#!/usr/bin/env python
# coding: utf-8

# # <font color='orange'>Introduction</font>
# **TransactionDT** is a column represents date and time of each transaction. Problem is, the values don't traditionally start from 1970/1/1 as usual but rather some random point in time. Figuring out when the start time is can help tremendously if one wants to use seasonality in their analyses and modeling.
# 
# Let's find out

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')

from IPython.display import display
import pandas as pd
pd.options.display.max_columns = None
from IPython.display import display, HTML



# In[ ]:


import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns


# In[ ]:


train_trans = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')


# In[ ]:


test_trans = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')


# # <font color='orange'>Overview</font>

# In[ ]:


plt.figure(figsize=(10,5))
_ = plt.hist(train_trans['TransactionDT'], bins=100), plt.hist(test_trans['TransactionDT'], bins=100)
plt.legend(['train','test'])


# <font color='orange'>*Notice the minimum value of TransactionDT is 86400 that happens to be the number of seconds of 1 day so we can assume that the unit of the column is second*</font>
# 
# * **TransactionDT** data of train and test set are from different distribution as the graphs don't overlap.
# * The data spreads across 396 days which is about 13 months
# * Looking at the peaks at both ends, they're likely to be associated with a festive season, Black Friday or Christmas maybe

# # Year

# ### Let's see which devices made the first transactions 

# In[ ]:


temp = train_trans.merge(train_id,on='TransactionID',how='inner')


# In[ ]:


temp.groupby('DeviceInfo').agg({'TransactionDT':'min'}).sort_values('TransactionDT').head(20)


# <font color='orange'>A quick check shows that the Samsung Galaxy S8 ***(SAMSUNG SM-G892A Build/NRD90M)*** was released in the US the earliest on 21st April 2017 (Wikipedia). Since there's transactions made by this phone the 2nd day of the data's time, this data apparently can only be as old as ***April of 2017***</font>

# # Date

# In[ ]:


np.max(test_trans['TransactionDT'])/86400 - np.min(train_trans['TransactionDT'])/86400


# <font color='white'>The data spreads over 395 days which is about 13 months. Also there're 2 peaks at the ends of the period. 
# 1. And I heard Black Friday is the peak shopping season in the US, no? <br/><br/> Let's find out!</font>

# <font color='white'>Looks like it's the case. Google Trends shows high traffic associated with Shopping related keywords during Black Friday season.</font>
# 
# https://trends.google.com/trends/explore?cat=18&date=2017-04-01%202018-12-31&geo=US

# In[ ]:


trends = pd.read_csv('../input/google-trends-shopping-data/multiTimeline.csv')


# In[ ]:


plt.figure(figsize=(20,10))
plt.xticks(rotation=65)
_ = plt.plot('Week','Traffic',data=trends)


# ### How about plotting the TransactionDT day wise?

# In[ ]:


plt.figure(figsize=(20,10))
train, test = plt.hist(np.ceil(train_trans['TransactionDT']/86400), bins=182), plt.hist(np.ceil(test_trans['TransactionDT']/86400), bins=182)


# 1. <font color='navy'>**Assuming we're right that the peaks are associated with the Black Friday season. 
#     Let's see what are the dates we're looking at**</font>

# In[ ]:


train[1][:182][train[0]> 6000]


# Since 2017's Black Friday was 24th November that happens to be tally with the numbers above. Let us just assume that the data begins on November 1st of 2017, Let's have a look at the 2nd peak in the test data

# In[ ]:


test_peaks = test[1][:182][test[0]> 5000]


# In[ ]:


test_peaks


# In[ ]:


[datetime.date(2017,11,1) + datetime.timedelta(days=x) for x in test_peaks.tolist()]


# <font color='navy'>**23rd November 2018 is Black Friday, and 26th November 2018 is Cyber Monday, how cool is that**</font>

# # Conclusion

# *Based on the patterns found within the data, together with insights provided by Google Trends, the data's start date is likely 2017/11/1*
