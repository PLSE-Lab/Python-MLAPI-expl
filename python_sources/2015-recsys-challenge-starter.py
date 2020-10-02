#!/usr/bin/env python
# coding: utf-8

# # 2015 RecSys Challenge  Solution

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
import os
os.chdir('../input/yoochoose-data')
print(os.listdir())
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-white')


# ## Data Wrangling
# 
# *  doesn't this (following cell and setting as 1/0) need to be filtered by session and item and mark once per item if purchased or not (just clicks) ? 

# In[ ]:


# set status of buys as 1
buys_raw=pd.read_csv('yoochoose-buys.dat',names=['sessionID','ts','itemID','price','cnt'])
print("buys train shape",buys_raw.shape)
buys_raw['status']=1

# set status of clicks as 0
clicks_raw=pd.read_csv('yoochoose-clicks.dat',names=['sessionID','ts','itemID','cat'])
print("clicks_raw train shape",clicks_raw.shape)
clicks_raw['status']=0

# concat two kinds of data set,sorted by sessionID and itemID
union=pd.concat([clicks_raw, buys_raw], ignore_index=True).sort_values(by=['sessionID','itemID'])
print("union shape",union.shape)
union.head()


# In[ ]:


# ts to datetime
union['ts']=pd.to_datetime(union.ts,infer_datetime_format=True)


# In[ ]:


union.head(12345).groupby(["itemID","sessionID"])["status"].nunique().max()


# In[ ]:


# replace NaNs of cat by preceding values in the same item group 
union['cat']=union['cat'].fillna(method='ffill')

### ts to datetime
# union['ts']=pd.to_datetime(union.ts,infer_datetime_format=True)
union['hour']=union.ts.dt.hour
union['weekday']=union['ts'].dt.dayofweek.astype(int)+1


# In[ ]:


union.head()


# ## Exploratory Data Analysis 

# ## Overview

# In[ ]:


buyID_num=buys_raw.sessionID.nunique() #
buyEvents=buys_raw.shape[0]
clickID_num=clicks_raw.sessionID.nunique()
itemClicks=clicks_raw.shape[0]
BC_ratio=buyID_num/clickID_num


# In[ ]:


print(r'''buyID_num:{} 
buyEvents:{} 
clickID_num:{} 
itemClicks:{} 
buy ratio of sessions :{}
buy ratio of clicks:{}'''
.format(buyID_num,buyEvents,clickID_num,itemClicks,BC_ratio,buyEvents/clickID_num))


# In[ ]:


buyEvents/clickID_num


# # Statistics
# 

# ## Buy Ratio Averaged for Time

# In[ ]:


# Buy ratio averaged for hour
hour_info=union.groupby(['hour','status'])['sessionID'].nunique().reset_index(name='count')
hour_click=hour_info[hour_info['status']==0]
hour_buy=hour_info[hour_info['status']==1]
hour_info=pd.merge(hour_click,hour_buy,on='hour')
hour_info['ratio']=hour_info['count_y']/hour_info['count_x']


# In[ ]:


# Buy ratio averaged for weekday
weekday_info=union.groupby(['weekday','status'])['sessionID'].nunique().reset_index(name='count')
weekday_click=weekday_info[weekday_info['status']==0]
weekday_buy=weekday_info[weekday_info['status']==1]
weekday_info=pd.merge(weekday_click,weekday_buy,on='weekday')
weekday_info['ratio']=weekday_info.count_y/weekday_info.count_x


# In[ ]:


fig = plt.figure(figsize=(15,6))
fig.suptitle(' Buy Ratio Averaged for Time', fontsize=20)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(hour_info['hour'],hour_info['ratio'],color='b')
ax1.bar(hour_info['hour'],hour_info['ratio'],width=0.1,color='b')


ax1.set_xlabel('$hour$', fontsize=17)
ax1.set_ylabel('$buy/click ratio$', fontsize=17)
ax2.scatter(weekday_info['weekday'],weekday_info['ratio'],color='b')
ax2.bar(weekday_info['weekday'],weekday_info['ratio'],width=0.03,color='b')
ax2.set_xlabel('$weekday$', fontsize=17)
ax2.set_ylabel('$buy/click ratio$', fontsize=17)
plt.show()


# ## Buy Ratio Averaged for Category

# In[ ]:


def cat_classfier(value):
    if value=='S':
        return '13'
    elif len(value)<=2:
        return value
    else:
        return '14'


# In[ ]:


union['cat']=union.cat.astype(str).apply(cat_classfier).astype(int)
cat_info=union.groupby(['cat','status'])['sessionID'].nunique().reset_index(name='count').sort_values(by='cat')
cat_click=cat_info[cat_info['status']==0]
cat_buy=cat_info[cat_info['status']==1]
cat_info=pd.merge(cat_click,cat_buy,on='cat')
cat_info['ratio']=cat_info.count_y/cat_info.count_x


# In[ ]:


fig = plt.figure(figsize=(15,6))
fig.suptitle(' Buy Ratio Averaged for Category', fontsize=20)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


fig = plt.figure(figsize=(8,4))
# plt.plot(cat_info.count_y)
ax1.scatter(cat_info.cat,cat_info.ratio,color='b')
ax1.bar(cat_info.cat,cat_info.ratio,width=0.07,color='b')
ax1.set_xticks(range(0,15))
ax1.set_ylim(0.02)
ax1.set_ylabel('$buy/click ratio$',fontsize=16)
ax1.set_xlabel('$Category$',fontsize=16)

ax2.plot(cat_info.count_x,color='b')
ax2.set_yscale('log')
ax2.set_ylabel('$clicks$',fontsize=16)
ax2.set_xticks(range(0,15))
ax2.set_xlabel('$Category$',fontsize=16)

plt.show()


# ## Buy Ratio Averaged for Category Number

# In[ ]:


# clear unused variables
import gc
del hour_info,weekday_info,union_raw,buys_raw,clicks_raw
gc.collect()


# In[ ]:


cat=union[['sessionID','cat','status']]
cat_num=cat.groupby('sessionID')['cat'].nunique().reset_index(name='cat_num')
session_status=cat.groupby('sessionID')['status'].max().reset_index(name='status')
cat_info=pd.merge(cat_num,session_status,on='sessionID')


# In[ ]:


cat_buys=cat_info.groupby('cat_num')['status'].sum().reset_index(name='buys')
cat_nums=cat_info.groupby('cat_num')['status'].count().reset_index(name='session_nums')
cat_aggr=pd.merge(cat_buys,cat_nums,on='cat_num')
cat_aggr['ratio']=cat_aggr.buys/cat_aggr.session_nums


# In[ ]:


cat_aggr


# In[ ]:


fig = plt.figure(figsize=(15,6))
fig.suptitle(' Buy Ratio Averaged for Category Number', fontsize=20)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


fig = plt.figure(figsize=(8,4))
ax1.scatter(cat_aggr.cat_num,cat_aggr.ratio,color='b')
ax1.bar(cat_aggr.cat_num,cat_aggr.ratio,color='b',width=0.07)
ax1.set_ylim(0.02)
ax1.set_ylabel('$Buy\,Ratio$',fontsize=16)
ax1.set_xlabel('$Category \,Number$',fontsize=16)


ax2.plot(cat_aggr.cat_num,cat_aggr.session_nums,color='b')
ax2.set_yscale('log')
ax2.set_ylabel('$Session\,Number$',fontsize=16)
ax2.set_xlabel('$Category\,Number$',fontsize=16)

plt.show()


# ## Buy Ratio Averaged for Session Length

# In[ ]:


session_length=union.groupby('sessionID')['status'].count().reset_index(name='length')
session_length=pd.merge(session_status,session_length,on='sessionID')


# In[ ]:


len_buys=session_length.groupby('length')['status'].sum().reset_index(name='buys')
len_num=session_length.groupby('length')['status'].count().reset_index(name='len_num')
len_aggr=pd.merge(len_buys,len_num,on='length')


# In[ ]:


len_aggr['ratio']=len_aggr.buys/len_aggr.len_num


# In[ ]:


fig = plt.figure(figsize=(15,6))
fig.suptitle(' Buy Ratio Averaged for Session Length', fontsize=20)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


fig = plt.figure(figsize=(8,4))
ax1.scatter(len_aggr.length,len_aggr.ratio,color='b')
ax1.set_ylabel('$Buy\,Ratio$',fontsize=16)
ax1.set_xlabel('$Session\, Length$',fontsize=16)
ax1.set_xscale('log')

ax2.plot(len_aggr.length,len_aggr.len_num,color='b')
ax2.set_yscale('log')
#ax2.set_xscale('log')
ax2.set_ylabel('$Session\,Number$',fontsize=16)
ax2.set_xlabel('$Session\, Length$',fontsize=16)


plt.show()


# ## Buy Ratio Averaged for Dwell Time

# In[ ]:


dwell_max=union.groupby('sessionID')['ts'].max().reset_index(name='max')
dwell_min=union.groupby('sessionID')['ts'].min().reset_index(name='min')

session_dwell=pd.merge(dwell_max,dwell_min,on='sessionID')
session_dwell['dwell']=(session_dwell['max']-session_dwell['min']).astype('timedelta64[m]')
session_dwell.head()
pd.merge(session_dwell,sessionID_status,on='sessionID')


# In[ ]:


session_dwell=pd.merge(session_dwell,session_status,on='sessionID')


# In[ ]:


session_dwell.head()


# In[ ]:


dwell_buys=session_dwell.groupby('dwell')['status'].sum().reset_index(name='buys')
dwell_num=session_dwell.groupby('dwell')['status'].count().reset_index(name='dwell_num')
dwell_aggr=pd.merge(dwell_buys,dwell_num,on='dwell')
dwell_aggr['ratio']=dwell_aggr.buys/dwell_aggr.dwell_num


# In[ ]:


fig = plt.figure(figsize=(15,6))
fig.suptitle(' Buy Ratio Averaged for Dwell Time', fontsize=20)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


fig = plt.figure(figsize=(8,4))
ax1.scatter(dwell_aggr.dwell,dwell_aggr.ratio,color='b')
ax1.set_ylabel('$Buy\,Ratio$',fontsize=16)
ax1.set_xlabel('$Dwell\, Time$',fontsize=16)
# ax1.set_xscale('log')
ax1.set_xlim(0,200)


ax2.plot(dwell_aggr.dwell,dwell_aggr.dwell_num,color='b')
ax2.set_yscale('log')
#ax2.set_xscale('log')
ax2.set_ylabel('$Count$',fontsize=16)
ax2.set_xlabel('$Dwell\, Time$',fontsize=16)
ax2.set_xlim(0,200)

plt.show()


# In[ ]:




