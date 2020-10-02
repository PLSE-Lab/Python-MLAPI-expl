#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from scipy import optimize
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# This is my first experience with submission on Kaggle. So, my apoligies if something does not work well here with this  first attempt :).
# 
# This is a simple way for calculating and projecting LTV ( Life Time Value) for free-to-play platform/game, using a small portion of historical data.
# After reading and analyzing the data sample where we can see that our data contains: useer ID, game installation date, day of payment and amount of payment by user on a payment day.
# 
# The objective here is to:
# 
# a) Perform a data exploratory analysis
# 
# b) Identify the best function for LTV projection. Here we can go with explicit choice of the function to fit (like I did) or use some alternative approach with preliminary feature engineering.
# 
# c) Project values for LTV90 and LTV180 where LTV{N} forecast of LTV value for the day N
# 
# d) Considering that our player's lifetime is 180 days, we can calculate the coefficients K1, K3, K7, K30 and K180, where K{N} is the ratio between revenue by day N and LTV180, identifyig the bending point at which cumulative revenue generation slows down and we either need to attract more users or stimulate existing users to make additional purchases.

# In[ ]:


data = pd.read_csv('../input/sq_data.csv')
print (data.shape)
data.head()


# In[ ]:


data['install_date'] = pd.to_datetime(data['install_date'],dayfirst = True)
data['pay_date'] = pd.to_datetime(data['pay_date'], dayfirst = True)
data = data.sort_values('pay_date')
def get_cum_sum(date):
    return data['sum'].where(data['pay_date']<=date).sum()

data['cum_sum'] = data['pay_date'].map(lambda x: get_cum_sum(x))
def get_users_utd(date):
    return data['user'].where(data['install_date'] <=date).count()

data['users_n_utd'] = data['pay_date'].map(lambda x: get_users_utd(x))
data['ltv'] = data['cum_sum']/data['users_n_utd'].astype(float)
data['day'] = pd.to_timedelta(data['pay_date'] - data['install_date'].min()).dt.days + 1
data['day'] = data['day'].astype(int)
data.head(10)


# In[ ]:


plt.scatter(data['day'], data['ltv'], label ='Days')
plt.scatter(data['users_n_utd'], data['ltv'], color='r', label='Players')
plt.scatter(data['cum_sum'], data['ltv'], color='g', label='Revenue')
plt.scatter(data['sum'], data['ltv'], color='b', label='Daily sales')
plt.xlabel('Days, Players, Revenue, Daily sales')
plt.ylabel('LTV')
plt.legend()


# In[ ]:


sns.pairplot(data)


# In[ ]:


ltv_data = data[['day','users_n_utd', 'cum_sum', 'ltv']]
X = ltv_data[['day', 'users_n_utd','cum_sum']]
y = ltv_data['ltv']


# LTV in our case simply depends on two variables - cumulative sales and number of users - all that by days.
# So, here is an approach:
# LTV by day has somewhat like a log-fnction shape and we can predict it via optimizing the fit of the LTV value to the log-like function curve. This can be done with 2 assumptions: number of users and cumulative sales are reaching saturation, which is somewhat true for the users once we look at the users over days plot and not so evident for the daily and thus cumulative sales. However, we can assume that the normal behaviour of the user on the platform shows that after certain amount of days user does not pay anymore or churns. Then we can easily apply here the LTV prediction using optimize.curve_fit method.

# In[ ]:


plt.scatter(data['day'], data['ltv'])
plt.xlabel('Days')
plt.ylabel('LTV')


# In[ ]:


X_l = X['day'].values 
Y_l = y.values

coefs_l, cov = optimize.curve_fit(lambda t,a,b: a+b*np.log(t),  X_l,  Y_l)

print (coefs_l)

def ltv_func(param):
    result = coefs_l[0] + coefs_l[1]*np.log(param)
    return result
    

ltv_90_180 = ltv_func([90., 180.])
ltv90 = round(ltv_90_180[0],2)
ltv180 = round(ltv_90_180[1],2)

print ("LTV by day 90: " + str(ltv90))
print ("LTV by day 180: " + str(ltv180))


# In[ ]:


days = np.hstack([X_l, [90, 180]])
plt.scatter(X_l,y.values)
plt.plot(days,ltv_func(days.reshape(-1, 1)))
plt.xlabel('Days')
plt.ylabel('LTV')
plt.legend(['LTV forecast', 'LTV'])


# In[ ]:


k_days = [1, 3, 7, 30]
ks= []
for k in k_days:
    k_revenue = ltv_data['cum_sum'].loc[ltv_data['day'] == k].values[0] 
    coeff = ltv180/k_revenue
    ks.append(coeff)
    print ("K" + str(k) + ": " + str(round(coeff, 2)))
k180 = ltv180/(ltv180 * ltv_data['users_n_utd'].max())


# In[ ]:


plt.title('Cumulative sales elbow-curve.')
plt.plot(k_days,ks)
plt.xlabel('Days')
plt.ylabel('K')


# Analyzin the chart above we can see that probably around days 3-7 we need to simulate our users with additional offers pushing them to perform more purchases on the platform.
# 
