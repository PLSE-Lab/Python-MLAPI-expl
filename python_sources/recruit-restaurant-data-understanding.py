#!/usr/bin/env python
# coding: utf-8

# In this kernel I'd like to share my understanding about the dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# As far a I understand, the task is about predicting the visitors int he Air restaurants in a period which spans from the last  week of April and May 2017. We can train models using the air_reserve data which are like this:

# In[ ]:


air = pd.read_csv('../input/air_reserve.csv')
air.head()


# and  provides informtion about  Air restaurant reservations in time. As we can see by means of tail, data are up to 2017-05-31. So they also contain the test period.

# In[ ]:


air.tail()


# We can infact analyse the air_visit_data file and verify it is up to th 22 of April which is when the train period ends and test period starts.

# In[ ]:


airvisit = pd.read_csv('../input/air_visit_data.csv')
airvisit['visit_date'] = pd.to_datetime(airvisit['visit_date']).dt.date
airvisit.tail()


# Now let's check the air_store_id out.

# In[ ]:


len(air['air_store_id'].unique())


# In[ ]:


len(airvisit['air_store_id'].unique())


# It is clear the reservation datase does not provide information for all the air restaurants we have to manage. But how can we make prediction for the missing ids? We still have another file to inspect: air_store_info. Let's check it out...

# In[ ]:


airinfo = pd.read_csv('../input/air_store_info.csv')
airinfo.head()


# In[ ]:


len(airinfo['air_store_id'].unique())


# It looks like it contains generic information about all the restaurants.

# ## First conclusions

# I think a possible approach is to use the air store info to create generic representations of the 829 restaurants, use them to train a model against the air reservation dataset and use the generic model to make predictions for the restaurants in the test set. 

# ## Clustering Restaurants

# In[ ]:


len(airinfo['air_genre_name'].unique())


# We have 14 genres of restaurants so we can classify them according to this information. We can use Kmeans to cluster restaurants according to their latitude and longitude data into 10 different clusters.

# In[ ]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=10, random_state=0).fit(airinfo[['longitude','latitude']])
airinfo['cluster'] = kmeans.predict(airinfo[['longitude','latitude']])


# In[ ]:


from mpl_toolkits.basemap import Basemap
m = Basemap(projection='aeqd',width=2000000,height=2000000, lat_0=37.5, lon_0=138.2)
cx = [c[0] for c in kmeans.cluster_centers_]
cy = [c[1] for c in kmeans.cluster_centers_]
cm = plt.get_cmap('gist_rainbow')
colors = [cm(2.*i/10) for i in range(10)]
colored = [colors[k] for k in airinfo['cluster']]
f,axa = plt.subplots(1,1,figsize=(15,16))
m.drawcoastlines()
m.fillcontinents(color='lightgray',lake_color='aqua',zorder=1)
m.scatter(airinfo.longitude.values,airinfo.latitude.values,color=colored,s=20,alpha=1,zorder=999,latlon=True)
m.scatter(cx,cy,color='Black',s=50,alpha=1,latlon=True,zorder=9999)
plt.setp(axa.get_yticklabels(), visible=True)
plt.annotate('Fukuoka', xy=(0.04, 0.32), xycoords='axes fraction',fontsize=20)
plt.annotate('Shikoku', xy=(0.25, 0.25), xycoords='axes fraction',fontsize=20)
plt.annotate('Hiroshima', xy=(0.2, 0.36), xycoords='axes fraction',fontsize=20)
plt.annotate('Osaka', xy=(0.40, 0.30), xycoords='axes fraction',fontsize=20)

plt.annotate('Tokyo', xy=(0.60, 0.4), xycoords='axes fraction',fontsize=20)
plt.annotate('Shizoku', xy=(0.50, 0.32), xycoords='axes fraction',fontsize=20)

for i in range(len(cx)):
    xpt,ypt = m(cx[i],cy[i])
    plt.annotate(i, (xpt+500,ypt+500),zorder=99999,fontsize=16)
plt.show()


# In[ ]:


airgenres = airinfo['air_genre_name'].unique()
def genre2num(genre):
    return np.where(airgenres == genre)[0][0]
def num2genre(num):
    return airgenres[num]


# In[ ]:


gencodes = [genre2num(genre) for genre in airinfo['air_genre_name']]
airinfo['air_genre_name']=gencodes


# In[ ]:


airinfo.head()


# In[ ]:


final = pd.merge(airvisit,airinfo).drop(['latitude','longitude'],axis=1)
final.head()


# In[ ]:


dates = pd.read_csv('../input/date_info.csv')
vdt = pd.to_datetime(final.visit_date)
final['vd']=vdt.dt.date
final['yday']=vdt.dt.dayofyear
final['wday']=vdt.dt.dayofweek
final = final.drop(['vd'],axis=1)
dts = pd.to_datetime(dates.calendar_date)

days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
dates['calendar_date'] = pd.to_datetime(dates['calendar_date']).dt.date
dates['dw'] = [days.index(dw) for dw in dates.day_of_week]
final = pd.merge(final,dates,left_on='visit_date',right_on='calendar_date')
dates.head()


# In[ ]:


final.head()


# ## Creating Training and Test data frames

# In[ ]:


traindf = final.copy()
traindf = traindf.drop(['air_area_name','wday','air_store_id','visit_date','day_of_week','calendar_date'],axis=1)
traindf.head()


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


reg = GradientBoostingRegressor(n_estimators=100)
scores = cross_val_score(reg, traindf.drop(['visitors'],axis=1), traindf['visitors'])
scores


# In[ ]:


reg.fit(traindf.drop(['visitors'],axis=1), traindf['visitors'])


# In[ ]:


import re
sub = pd.read_csv('../input/sample_submission.csv')
ids = [re.split(r'_\d\d\d\d-',sub['id'][i])[0] for i in range(len(sub))]
ids = pd.Series(ids).unique()
len(ids)*39


# In[ ]:


base = pd.to_datetime("2017-04-23")
date_list = pd.date_range(base,periods = 39)
k = 1 
datedf = pd.DataFrame({"key":k,"date":date_list})


# In[ ]:


k = 1 
ids = pd.DataFrame({"key":k,"air_store_id":ids})


# In[ ]:


testdf = pd.merge(ids,datedf,on="key")
testdf['date'] = pd.to_datetime(testdf['date']).dt.date
testdf['yday'] = pd.to_datetime(testdf['date']).dt.dayofyear
finalt = pd.merge(testdf,airinfo).drop(['air_area_name','latitude','longitude'],axis=1)
finalt = pd.merge(finalt,dates,left_on='date',right_on='calendar_date').drop(['day_of_week','calendar_date'],axis=1)
finalt = finalt.drop(['date','key'],axis=1)
finalt.head()


# In[ ]:


cols = ['air_store_id','air_genre_name','cluster','yday','holiday_flg','dw']
finalt = finalt[cols]


# In[ ]:


pred = reg.predict(finalt.drop('air_store_id',axis=1))
len(pred)


# In[ ]:


ids = []
for r in range(len(testdf)):
    ids.append(testdf.iloc[r][0]+'_'+str(testdf.iloc[r][2]))


# In[ ]:


rdf = pd.DataFrame({'id':ids,'visitors':pred})


# In[ ]:


rdf.to_csv('out.csv',header=True,index=False)


# In[ ]:





# In[ ]:




