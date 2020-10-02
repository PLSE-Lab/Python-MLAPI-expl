#!/usr/bin/env python
# coding: utf-8

# # New York City Taxi Data Exploration

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (6, 4)
get_ipython().run_line_magic('matplotlib', 'inline')

color = sns.color_palette()
import warnings
warnings.filterwarnings('ignore') 


# ##  Load Files

# In[ ]:


train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
sample = pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission.csv')


# ### File check...!!

# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head(2)


# In[ ]:


test.head(2)


# In[ ]:


train.columns


# In[ ]:


test.columns


# Of course, No `dropoff_datetime` and `trip_duration` for test set.

# In[ ]:


train_pickup = pd.read_csv('../input/manhattan-or-not/train_pickup_save.csv')
train_dropoff = pd.read_csv('../input/manhattan-or-not/train_dropoff_save.csv')

test_pickup = pd.read_csv('../input/manhattan-or-not/test_pickup_save.csv')
test_dropoff = pd.read_csv('../input/manhattan-or-not/test_dropoff_save.csv')


# ### Manhattan or Not?

# In[ ]:


fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(121)
ax1.scatter(train.pickup_longitude[train_pickup.pickup_manhattan=='Manhattan'],
            train.pickup_latitude[train_pickup.pickup_manhattan=='Manhattan'],
            s=1,alpha=0.1,color='red')
ax1.scatter(train.pickup_longitude[train_pickup.pickup_manhattan=='Non-Manhattan'],
            train.pickup_latitude[train_pickup.pickup_manhattan=='Non-Manhattan'],
            s=1,alpha=0.1,color='blue')

plt.ylim([40.60,41.00])
plt.xlim([-74.15,-73.70])
plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)
plt.title('Pickup Location (Train)',fontsize=18)

ax1 = fig.add_subplot(122)
ax1.scatter(train.dropoff_longitude[train_dropoff.dropoff_manhattan=='Manhattan'],
            train.dropoff_latitude[train_dropoff.dropoff_manhattan=='Manhattan'],
            s=1,alpha=0.1,color='red')
ax1.scatter(train.dropoff_longitude[train_dropoff.dropoff_manhattan=='Non-Manhattan'],
            train.dropoff_latitude[train_dropoff.dropoff_manhattan=='Non-Manhattan'],
            s=1,alpha=0.1,color='blue')

plt.ylim([40.60,41.00])
plt.xlim([-74.15,-73.70])
plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)
plt.title('Dropoff Location (Train)',fontsize=18)


# In[ ]:


fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(121)
ax1.scatter(test.pickup_longitude[test_pickup.pickup_manhattan=='Manhattan'],
            test.pickup_latitude[test_pickup.pickup_manhattan=='Manhattan'],
            s=1,alpha=0.1,color='red')
ax1.scatter(test.pickup_longitude[test_pickup.pickup_manhattan=='Non-Manhattan'],
            test.pickup_latitude[test_pickup.pickup_manhattan=='Non-Manhattan'],
            s=1,alpha=0.1,color='blue')

plt.ylim([40.60,41.00])
plt.xlim([-74.15,-73.70])
plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)
plt.title('Pickup Location',fontsize=18)

ax1 = fig.add_subplot(122)
ax1.scatter(test.dropoff_longitude[test_dropoff.dropoff_manhattan=='Manhattan'],
            test.dropoff_latitude[test_dropoff.dropoff_manhattan=='Manhattan'],
            s=1,alpha=0.1,color='red')
ax1.scatter(test.dropoff_longitude[test_dropoff.dropoff_manhattan=='Non-Manhattan'],
            test.dropoff_latitude[test_dropoff.dropoff_manhattan=='Non-Manhattan'],
            s=1,alpha=0.1,color='blue')

plt.ylim([40.60,41.00])
plt.xlim([-74.15,-73.70])
plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)
plt.title('Dropoff Location',fontsize=18)


# ## Correlation Function

# In[ ]:


numtrain = train.select_dtypes(include=[np.number])
corr = numtrain.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr,vmax=1,square=True,annot=True)


# The highest correlation coefficient is between pickup_longitude and dropoff_longitude. This is higher than that of between pickup_langitude and dropoff_latitude, 0.49 because Manhattan is elongated in the North-South direction. This means that there is more variation in the latitude range and less variation of the longitude. 
# 
# Otherwise, most of coefficient is really very low related to `trip_duration`, which means we either have to find more sources of data or be creative in our feature engineering.

# ### miles

# In[ ]:


train['new_pickup'] = [(a,b) for a, b in zip(train.pickup_latitude,train.pickup_longitude)]
train['new_dropoff'] = [(a,b) for a, b in zip(train.dropoff_latitude,train.dropoff_longitude)]

from geopy.distance import great_circle
train['miles'] = [great_circle(a,b).miles for a, b in zip(train.new_pickup,train.new_dropoff)]

test['new_pickup'] = [(a,b) for a, b in zip(test.pickup_latitude,test.pickup_longitude)]
test['new_dropoff'] = [(a,b) for a, b in zip(test.dropoff_latitude,test.dropoff_longitude)]

test['miles'] = [great_circle(a,b).miles for a, b in zip(test.new_pickup,test.new_dropoff)]


# Obviously, `miles` are not real distance because taxies cannot drive the shortest distance between two points. However, here I just simplify the driving distance as the distance between two locations.

# In[ ]:


train["velocity"] = train.miles/(train.trip_duration/(60*60))


# In[ ]:


train = train[((train.trip_duration < 10000) & (train.trip_duration >10))]


# In[ ]:


sns.jointplot('miles','trip_duration',data=train[:10000],s=10,alpha=0.5,color='green')


# It is obvious that `miles` is correlated with `trip_duration`, but it seems to be skewed. So, I can plot with log scale.

# In[ ]:


#plt.figure(figsize=(50,30))
sns.jointplot(np.log10(train["miles"][:10000]+1),np.log10(train["trip_duration"][:10000]+1),s=10,alpha=0.5,color='green')
plt.xlabel('log (mile)')
plt.ylabel('log (trip_duration)')


# ### Pickup Location & Dropoff Location

# In[ ]:


fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(121)
ax1.scatter(train.pickup_longitude,train.pickup_latitude,s=1,alpha=0.1)
plt.ylim([40.60,41.00])
plt.xlim([-74.15,-73.70])
plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)
plt.title('Pickup Location',fontsize=18)
ax2 = fig.add_subplot(122)
ax2.scatter(train.dropoff_longitude,train.dropoff_latitude,s=1,color='green',alpha=0.1)
plt.ylim([40.60,41.00])
plt.xlim([-74.15,-73.70])
plt.title('Dropoff Location',fontsize=18)
plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)


# I can check some of the facts:
# 
# * Most of pickup location is from Manhattan.
# * Some of pickup location is from either Brookline or Airport (JFK or LGA)
# * Dropoff Location is more distributed in not only Manhattan but Brookline or Quincy, etc.

# ## DateTime

# In[ ]:


train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['pickup_hour'] = train.pickup_datetime.dt.hour
train['pickup_week'] = train.pickup_datetime.dt.weekday
train['pickup_month'] = train.pickup_datetime.dt.month
train['pickup_day'] = train.pickup_datetime.dt.day

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
test['pickup_hour'] = test.pickup_datetime.dt.hour
test['pickup_week'] = test.pickup_datetime.dt.weekday
test['pickup_month'] = test.pickup_datetime.dt.month
test['pickup_day'] = test.pickup_datetime.dt.day


# In[ ]:


#import calendar
#list(calendar.day_name)
#train['pickup_week'] = train['pickup_week'].apply(lambda x: calendar.day_name[x])
#test['pickup_week'] = test['pickup_week'].apply(lambda x: calendar.day_name[x])


# In[ ]:


train['pickup_hour'] = train['pickup_hour'].apply(int)
test['pickup_hour'] = test['pickup_hour'].apply(int)

train['pickup_day'] = train['pickup_day'].apply(int)
test['pickup_day'] = test['pickup_day'].apply(int)

train['pickup_month'] = train['pickup_month'].apply(int)
test['pickup_month'] = test['pickup_month'].apply(int)


# In[ ]:


plt.figure(figsize=(8,6))
sns.pointplot(x='pickup_hour',y='miles',data=train,kind='point',hue='pickup_week',hue_order=list(calendar.day_name))
plt.xlabel('pickup_hour',fontsize=16)
plt.ylabel('mean(Miles)',fontsize=16)


# In[ ]:


plt.figure(figsize=(8,6))
sns.pointplot(x='pickup_hour',y='trip_duration',data=train,hue='pickup_week',hue_order=list(calendar.day_name))
plt.xlabel('pickup_hour',fontsize=16)
plt.ylabel('mean(trip_duration)',fontsize=16)


# In[ ]:


plt.figure(figsize=(8,6))
sns.pointplot(x='pickup_hour',y='miles',data=train,kind='point',hue='pickup_month')
plt.xlabel('pickup_hour',fontsize=16)
plt.ylabel('mean(miles)',fontsize=16)


# In[ ]:


plt.figure(figsize=(8,6))
sns.pointplot(x='pickup_hour',y='trip_duration',data=train,hue='pickup_month')
plt.xlabel('pickup_hour',fontsize=16)
plt.ylabel('mean(trip_duration)',fontsize=16)


# It is interesting that the relation with `trip_duration` and `pickup_hour` seems to change with different months in the day time (9am-7pm). 

# ### why are there higher pick during the warmer season daytime? Is that because of more Taxi ride or?

# In[ ]:


weekcount = train.groupby(['pickup_hour','pickup_week'],as_index=False).count()[['pickup_hour','pickup_week','id']]
monthcount = train.groupby(['pickup_hour','pickup_month'],as_index=False).count()[['pickup_hour','pickup_month','id']]


# In[ ]:


plt.figure(figsize=(8,6))
sns.pointplot(x='pickup_hour',y='id',data=weekcount,hue='pickup_week',hue_order=list(calendar.day_name))
plt.xlabel('pickup_hour',fontsize=16)
plt.ylabel('Count',fontsize=16)


# In[ ]:


plt.figure(figsize=(8,6))
sns.pointplot(x='pickup_hour',y='id',data=monthcount,hue='pickup_month')
plt.xlabel('pickup_hour',fontsize=16)
plt.ylabel('Count',fontsize=16)


# No obvious trend showing an increase in the number of rides as you go from January to June, which means the increased trip duration in later months is not due to an increases in the number of rides.  However, there could be more traffice jam due to many other cars in later months. In conclusion, there are either many more local people around or many tourists in New York City during the warmer seasons, especially day time.

# ### store and fwd flag

# In[ ]:


pd.DataFrame(train.groupby(['store_and_fwd_flag']).count()['id']).plot(kind='bar',color='blue')
plt.ylabel('Number')


# ### vendor_id

# In[ ]:


pd.DataFrame(train.groupby(['vendor_id']).count()['id']).plot(kind='bar',color='blue')
plt.ylabel('Number')


# ### passenger_count

# In[ ]:


pd.DataFrame(train.groupby(['passenger_count']).count()['id']).plot(kind='bar',color='blue')
plt.ylabel('Number')


# In[ ]:


plt.figure(figsize=(14,6))
sns.barplot(x='pickup_hour',y='trip_duration',data=train,hue='store_and_fwd_flag')
plt.xlabel('pickup_hour',fontsize=16)
plt.ylabel('mean(trip_duration)',fontsize=16)


# In[ ]:


plt.figure(figsize=(14,6))
sns.barplot(x='pickup_hour',y='trip_duration',data=train,hue='vendor_id')
plt.xlabel('pickup_hour',fontsize=16)
plt.ylabel('mean(trip_duration)',fontsize=16)


# Here, I just want to simplify `passenger_count` as either one or more than one person. So,

# In[ ]:


#train['passenger_count'] = train.passenger_count.apply(lambda x:2 if x>=2 else 1)
#test['passenger_count'] = test.passenger_count.apply(lambda x:2 if x>=2 else 1)


# In[ ]:


pd.DataFrame(train['passenger_count'].value_counts()).plot(kind='bar')


# In[ ]:


plt.figure(figsize=(8,6))
sns.pointplot(x='pickup_hour',y='trip_duration',data=train,hue='passenger_count')
plt.xlabel('pickup_hour',fontsize=16)
plt.ylabel('mean(trip_duration)',fontsize=16)


# Here, passenger_count = 2 means that the number of passenger is more than 1.

# In[ ]:


train["pickup_manhattan"]=train_pickup["pickup_manhattan"].copy()
train["dropoff_manhattan"]=train_dropoff["dropoff_manhattan"].copy()

test["pickup_manhattan"]=test_pickup["pickup_manhattan"].copy()
test["dropoff_manhattan"]=test_dropoff["dropoff_manhattan"].copy()


# In[ ]:


train['pickup_manhattan'] = train.pickup_manhattan.apply(lambda x:1 if x=="Manhattan" else 0)
test['pickup_manhattan'] = test.pickup_manhattan.apply(lambda x:1 if x=="Manhattan" else 0)


# In[ ]:


train['dropoff_manhattan'] = train.pickup_manhattan.apply(lambda x:1 if x=="Manhattan" else 0)
test['dropoff_manhattan'] = test.pickup_manhattan.apply(lambda x:1 if x=="Manhattan" else 0)


# In[ ]:


train['store_and_fwd_flag'] = train.pickup_manhattan.apply(lambda x:1 if x=="Y" else 0)
test['store_and_fwd_flag'] = test.pickup_manhattan.apply(lambda x:1 if x=="Y" else 0)


# In[ ]:


from sklearn.cluster import KMeans
train_pickup_loc = pd.DataFrame()
train_pickup_loc['longitude'] = train['pickup_longitude']
train_pickup_loc['latitude'] = train['pickup_latitude']

test_pickup_loc = pd.DataFrame()
test_pickup_loc['longitude'] = test['pickup_longitude']
test_pickup_loc['latitude'] = test['pickup_latitude']

train_dropoff_loc = pd.DataFrame()
train_dropoff_loc['longitude'] = train['dropoff_longitude']
train_dropoff_loc['latitude'] = train['dropoff_latitude']

test_dropoff_loc = pd.DataFrame()
test_dropoff_loc['longitude'] = test['dropoff_longitude']
test_dropoff_loc['latitude'] = test['dropoff_latitude']

both_loc = pd.concat([train_pickup_loc,test_pickup_loc,train_dropoff_loc,test_dropoff_loc],axis=0)
kmeans = KMeans(n_clusters=15, random_state=0).fit(both_loc)


# In[ ]:


len_train = len(train)
len_test = len(test)


# In[ ]:


train['pickup_label'] = kmeans.labels_[:len_train]
test['pickup_label'] = kmeans.labels_[len_train:len_train+len_test]
train['dropoff_label'] = kmeans.labels_[len_train+len_test:len_train*2+len_test]
test['dropoff_label'] = kmeans.labels_[len_train*2+len_test:]


# ## Data Cleaning for Model

# In[ ]:


train_cleaning = train.drop(['id','pickup_datetime','dropoff_datetime', 'new_pickup','new_dropoff','velocity','trip_duration'],axis=1)
test_cleaning = test.drop(['id','pickup_datetime','new_pickup','new_dropoff'],axis=1)


# In[ ]:


print(train_cleaning.shape)
print(test_cleaning.shape)


# In[ ]:


str_columns = ['pickup_hour','pickup_month','vendor_id','passenger_count']
for i in str_columns:
    train_cleaning[i]=train_cleaning[i].apply(str)
    test_cleaning[i]=test_cleaning[i].apply(str)


# In[ ]:


numerical = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_cleaning[numerical] = scaler.fit_transform(train_cleaning[numerical])
test_cleaning[numerical] = scaler.fit_transform(test_cleaning[numerical])


# In[ ]:


final_train = pd.get_dummies(train_cleaning)
final_test = pd.get_dummies(test_cleaning)


# In[ ]:


X=final_train
y=train["trip_duration"]

