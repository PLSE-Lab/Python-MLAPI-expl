#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import ensemble, neighbors, linear_model, metrics, preprocessing
from datetime import datetime
import glob, re
import time, datetime
from datetime import timedelta

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

# from JdPaletto & the1owl1
# JdPaletto - https://www.kaggle.com/jdpaletto/surprised-yet-part2-lb-0-503?scriptVersionId=1867420
# the1owl1 - https://www.kaggle.com/the1owl/surprise-me
start1 =time.time()
data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])# bring air id to hpd reserve data

print('Training data....',data['tra'].shape)
print('Unique store id in training data',len(data['tra']['air_store_id'].unique()))
print('Id data....',data['id'].shape)
print('Air store data....',data['as'].shape,'& unique-',data['as']['air_store_id'].unique().shape)
print('Hpg store data....',data['hs'].shape,'& unique-',data['hs']['hpg_store_id'].unique().shape)
print('Air reserve data....',data['ar'].shape,'& unique-',data['ar']['air_store_id'].unique().shape)
print('Hpg reserve data....',data['hr'].shape,'& unique-',data['hr']['air_store_id'].unique().shape)
      
#converting datetime to date for reservation data
for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    
    #calculate reserve time difference and summarizing ar,hr to date
    data[df]['reserve_datetime_diff'] = data[df].apply(
        lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[[
        'reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'})

#breaking down dates on training data & summarizing 
data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.weekday_name
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

#extracting store id and date info from test data
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.weekday_name
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

#extract unique stores based on test data and populate dow 1 to 6
unique_stores = data['tes']['air_store_id'].unique()#extract unique stores id from test data
#populating unique stores to dow
store_7days = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) 
                    for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
store_sum = pd.DataFrame({'air_store_id': unique_stores})

# mapping train data dow to stores(test data) - min, mean, median, max, count 
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)[
    'visitors'].sum().rename(columns={'visitors':'total_visitors'})
store_7days = pd.merge(store_7days, tmp, how='left', on=['air_store_id','dow']) 
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)[
    'visitors'].mean().rename(columns={'visitors':'mean_visitors'})
store_7days = pd.merge(store_7days, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)[
    'visitors'].median().rename(columns={'visitors':'median_visitors'})
store_7days = pd.merge(store_7days, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)[
    'visitors'].max().rename(columns={'visitors':'max_visitors'})
store_7days = pd.merge(store_7days, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)[
    'visitors'].count().rename(columns={'visitors':'count_observations'})
store_7days = pd.merge(store_7days, tmp, how='left', on=['air_store_id','dow']) 
# map stores(test) to store genre and location detail
store_7days = pd.merge(store_7days, data['as'], how='left', on=['air_store_id']) 
# Encoding categories Air _genre and air area
#lbl = preprocessing.LabelEncoder()
#stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
#stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])# to drop categorical for algo use later

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
#data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 
train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

col = [c for c in train if c not in ['id', 'air_store_id','visit_date','visitors']]
train = train.fillna(0) #change to one for algo training
test = test.fillna(0)
#df=df.rename(columns = {'two':'new_name'})
train=train.rename(columns = {'reserve_datetime_diff_x':'reserve_datetime_diff_air','reserve_visitors_x':'reserve_visitors_air',
                             'reserve_datetime_diff_y':'reserve_datetime_diff_hpg','reserve_visitors_y':'reserve_visitors_hpg'})
train['v_no_reservation']=train['visitors']-train['reserve_visitors_air']-train['reserve_visitors_hpg']
print(round(time.time()-start1,4))


# In[ ]:


store_sum


# In[ ]:


data['as'].head()


# In[ ]:


data['as']['air_genre_name'].value_counts()


# In[ ]:


data['hs'].head()


# In[ ]:


data['hs']['hpg_genre_name'].value_counts()


# In[ ]:


data['hr'].head()


# In[ ]:





# In[ ]:


test1=pd.merge(data['tra'],data['id'], how='left', on='air_store_id')
test1['hpg_store_id'].notnull().sum()
#test1.head()


# In[ ]:


import folium
from folium.plugins import MarkerCluster

location =store_7days.groupby(['air_store_id','air_genre_name'])['latitude','longitude'].mean().reset_index()


lbl = preprocessing.LabelEncoder()
store1['air_genre_name'] = lbl.fit_transform(store1['air_genre_name'])

data = np.array([location['latitude'] ,location['longitude'],location['air_genre_name']]).T
map1 = folium.Map(location=[39, 139.6917], 
                        tiles = "Stamentoner",# width=1000, height=500,
                        zoom_start = 5)
marker_cluster = MarkerCluster(data).add_to(map1)
map1


# In[ ]:


stored2.head()


# In[ ]:


import folium
import os
from folium import plugins
N = 100

data = np.array(
    [
        np.random.uniform(low=35, high=60, size=N),  # Random latitudes in Europe.
        np.random.uniform(low=-12, high=30, size=N),  # Random longitudes in Europe.
        range(N),  # Popups texts are simple numbers.
    ]
).T

m = folium.Map([45, 3], zoom_start=4)

plugins.MarkerCluster(data).add_to(m)

#m.save(os.path.join('results', 'Plugins_1.html'))

m


# In[ ]:





# In[ ]:





# 

# Visualising holidays in Japan

# In[ ]:


print('Total visitors - ',train['visitors'].sum())
print('Total reservation air - ',train['reserve_visitors_air'].sum())
print('Total reservation hpg - ',train['reserve_visitors_hpg'].sum())


# In[ ]:


f,ax = plt.subplots(1,1,figsize=(15,1))
x=data['hol']['visit_date']
y=data['hol']['holiday_flg']
plt.plot(x,y, color='m')
plt.show()


# In[ ]:


#Visitor each day
f,ax = plt.subplots(1,1,figsize=(15,4.5))
plt1 = train.groupby(['visit_date'], as_index=False).agg({'visitors': np.sum})
plt1=plt1.set_index('visit_date')
plt1.plot(color='c', kind='area', ax=ax)
plt.ylabel("Sum of Visitors")
plt.title("Visitor each day")


# Visitors based on days of the week for the last 1 year

# In[ ]:


max_date=max(train['visit_date'])
one_year = datetime.timedelta(days=364)
year_ago= max_date - one_year
train2=train[train['visit_date']>year_ago]
pvt=pd.pivot_table(train2, index=['dow'], columns='month',values='visitors',aggfunc=[np.mean],fill_value=0)
pvt2=pd.pivot_table(train2, index=['dow'], columns='month',values='visitors',aggfunc=[np.median],fill_value=0)
pvt3=pd.pivot_table(train2, index=['dow'], columns='month',values='visitors',aggfunc=[np.max],fill_value=0)
pvt4=pd.pivot_table(train2, index=['dow'], columns='month',values='visitors',aggfunc=[np.sum],fill_value=0)
f, ax=plt.subplots(2,2, figsize=(15,8))
sns.heatmap(pvt, ax=ax[0,0],cmap='cool')
sns.heatmap(pvt2, ax=ax[0,1],cmap='cool')
sns.heatmap(pvt3, ax=ax[1,0],cmap='cool')
sns.heatmap(pvt4, ax=ax[1,1],cmap='cool')
ax[0,0].set_title('Mean Visitors')
ax[0,1].set_title('Median Visitors')
ax[1,0].set_xlabel('Max Visitors', fontsize=13)
ax[1,1].set_xlabel('Total Visitors', fontsize=13)
#plt.xlabel("Month")


# In[ ]:


plt1 = train.groupby(['visit_date'], as_index=False).agg({'visitors': np.sum})


# In[ ]:


plt.style.use('fivethirtyeight')
plt1=train['visitors'].value_counts().reset_index().sort_index()
fig,ax = plt.subplots()
ax.bar(plt1['index'] ,plt1['visitors'],color='darkturquoise')
fig.set_size_inches(12,4, forward=True)
ax.set_xlim(0, 150)
ax.set_title("PAX Frequency")
ax.set_ylabel('Counts')
ax.set_xlabel('Number of PAX per visit')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn import ensemble, neighbors, linear_model, metrics, preprocessing
from datetime import datetime
import glob, re
import time, datetime
from datetime import timedelta
data = {
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv')}


# In[ ]:


'''data['ar']['visit_datetime'] = pd.to_datetime(data['ar']['visit_datetime'])
data['ar']['reserve_datetime'] = pd.to_datetime(data['ar']['reserve_datetime'])
data['ar']['visit_hour']=data['ar']['visit_datetime'].apply(lambda x: x.time().hour)
data['ar']['dow_visit'] = data['ar']['visit_datetime'].dt.dayofweek
data['ar']['month'] = data['ar']['visit_datetime'].dt.month
data['ar']['reserve_day'] = data['ar'].apply(
    lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)'''

data['hr']['visit_datetime'] = pd.to_datetime(data['hr']['visit_datetime'])
data['hr']['reserve_datetime'] = pd.to_datetime(data['hr']['reserve_datetime'])
data['hr']['visit_hour']=data['hr']['visit_datetime'].apply(lambda x: x.time().hour)
data['hr']['dow_visit'] = data['hr']['visit_datetime'].dt.dayofweek
data['hr']['month'] = data['hr']['visit_datetime'].dt.month
data['hr']['reserve_day'] = data['hr'].apply(
    lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)

data['hr'].head()


# In[ ]:





# In[ ]:




