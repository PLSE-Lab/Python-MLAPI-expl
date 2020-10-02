#!/usr/bin/env python
# coding: utf-8

# Well we don't have time intervals in our training data set to compare. But **Demo Account** has some information.
# If we compare in users report `2017-09-01 to 2017-10-15` (train) with `2017-12-01 to 2018-01-30` (test) the same intervals but last year. 
# 
# It is just a simple report I've downloaded from Google Analytics. Test period contains only 3185 users with revenue, so it is a one file with 5000 records after after sorting by 
# revenue.
# 
# **Update:**   fix error in metric, now real revenue is multiplied to million as in dataset. Added proportion of time periods in calculation.
# 
# #### Sample data:
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
df = pd.read_csv('../input/gmsga2017/Analytics 3 Raw Data View User Explorer 20171201-20180131 20170501-20171015.csv',skiprows=5)
df.head(3).T


# In[ ]:


df['Revenue'] = pd.to_numeric(df.Revenue.str.replace(',','').str[3:])
df['Range'] = df['Date Range'].replace({'01-Dec-2017 - 31-Jan-2018':'Test', '01-May-2017 - 15-Oct-2017':'Train'})
df=df[['Sessions','Revenue','Range','Client ID']].set_index(['Range','Client ID']).unstack('Range')
df.columns = ['_'.join(t) for t in df.columns]
df.sample(3).T


# In[ ]:


train_user_revenue_on_test = df[(df.Revenue_Test>0)&(df.Sessions_Train>0)].Revenue_Test
train_user_revenue_on_test.agg(['sum','count','mean','min','max'])


# In[ ]:


total_abs_error = np.log1p(train_user_revenue_on_test*10**6).sum()
print ('total abs error',total_abs_error)
total_square_error = (np.log1p(train_user_revenue_on_test*10**6)**2).sum()
print ('total square error',total_square_error)
dates = pd.to_datetime([
    '2 August 2017',#old test start
    '30 April 2018',#old test end
    '1 May 2018',#new test start
    '15 October 2018']) #new test end
prop = (dates[3] - dates[2])/(dates[1] - dates[0])
print('new test length as proportion of old one',prop) 


# In[ ]:


df_sub = pd.read_csv('../input/ga-customer-revenue-prediction/sample_submission.csv')
print('total users', df_sub.shape[0], 'RMSE on all zero:',np.sqrt(total_square_error/(df_sub.shape[0]*prop)))


# I'm not sure about how many users will be on new submissions, but you can see that only 0.58%  of users still active after 3 months.  

# In[ ]:


pd.DataFrame([[4.85,1.60,0.58]], index = ['User Retention'], columns =['1 Month','2 Month','3 Month'])


# So all zero submissions looks like a strong baseline.

# 