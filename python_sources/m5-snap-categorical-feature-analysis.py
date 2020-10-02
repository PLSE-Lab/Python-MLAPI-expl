#!/usr/bin/env python
# coding: utf-8

# # What is SNAP?
# `snap_CA`, `snap_TX` and `snap_WI` are three categorical features (binary) in `calendar.csv`. From the official pdf provided by the organizers:
# 
# ![](https://i.imgur.com/o3gEk13.png)
# More info here: https://www.fns.usda.gov/snap/supplemental-nutrition-assistance-program
# 
# I did a little analysis on them.

# # Imports

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# # Analyze SNAP 

# In[ ]:


df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
df.head()


# In[ ]:


# The pattern in these plots is beatiful <3
plt.figure(figsize=(15,8))
df.snap_CA.plot()
plt.title('SNAP-CA')
plt.show()
plt.figure(figsize=(15,8))
df.snap_TX.plot()
plt.title('SNAP-TX')
plt.show()
plt.figure(figsize=(15,8))
df.snap_WI.plot()
plt.title('SNAP-WI')
plt.show()


# In[ ]:


#snap_CA
print("Years", df[df.snap_CA==1].date.apply(lambda x:x.split('-')[0]).unique())
print("Months", df[df.snap_CA==1].date.apply(lambda x:x.split('-')[1]).unique())
print("Days", df[df.snap_CA==1].date.apply(lambda x:x.split('-')[2]).unique())


# In[ ]:


#snap_TX
print("Years", df[df.snap_TX==1].date.apply(lambda x:x.split('-')[0]).unique())
print("Months", df[df.snap_TX==1].date.apply(lambda x:x.split('-')[1]).unique())
print("Days", df[df.snap_TX==1].date.apply(lambda x:x.split('-')[2]).unique())


# In[ ]:


#snap_WI
print("Years", df[df.snap_WI==1].date.apply(lambda x:x.split('-')[0]).unique())
print("Months", df[df.snap_WI==1].date.apply(lambda x:x.split('-')[1]).unique())
print("Days", df[df.snap_WI==1].date.apply(lambda x:x.split('-')[2]).unique())


# # Conclusion
# SNAP is provided every year, every month on a set of dates. These dates are different in all three states but is fixed forever for a particular state.  
# 
# 
# I am thinking of engineering special features like "SNAP increase" that will measure the difference between normal days (average) and snap days (average) sales for each item, this will help identifying items for which the sale is heavily dependent on snap. Many similar features can be created and might prove very useful.
# 
# Also, [JohnM](https://www.kaggle.com/jpmiller) has shared some external [data](https://www.kaggle.com/jpmiller/publicassistance) and a[kernel](https://www.kaggle.com/jpmiller/improving-m5-forecasts-with-snap-data) relating to SNAP that might be very helpful for this competition. I will carefully look into it and try to do some feature engineering.
