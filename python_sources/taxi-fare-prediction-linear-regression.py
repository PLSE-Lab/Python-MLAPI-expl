#!/usr/bin/env python
# coding: utf-8

# # Import Libraries and load dataset

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings  
warnings.filterwarnings('ignore')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


import datashader as ds


# In[ ]:


taxi = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv',nrows=50000)
taxi.head()


# # Check Statistical summary & null values

# In[ ]:


taxi.info()


# In[ ]:


taxi.describe()


# In[ ]:


taxi.isnull().sum()


# # Feature Engineering

# In[ ]:


taxi['pickup_datetime'] = pd.to_datetime(taxi['pickup_datetime'])


# In[ ]:


taxi['year'] = taxi['pickup_datetime'].dt.year


# In[ ]:


taxi['month'] = taxi['pickup_datetime'].dt.month


# In[ ]:


taxi['hour'] = taxi['pickup_datetime'].dt.hour


# In[ ]:


taxi['day_of_week'] = taxi['pickup_datetime'].dt.dayofweek


# # Data Cleaning

# ## Check & remove outliers

# In[ ]:


sns.boxplot(taxi['fare_amount'])


# In[ ]:


sns.boxplot(taxi['pickup_latitude'])


# In[ ]:


sns.boxplot(taxi['pickup_longitude'])


# In[ ]:


sns.boxplot(taxi['dropoff_latitude'])


# In[ ]:


sns.boxplot(taxi['dropoff_longitude'])


# In[ ]:


sns.boxplot(taxi['passenger_count'])


# In[ ]:


sns.boxplot(taxi['year'])


# In[ ]:


sns.boxplot(taxi['day_of_week'])


# In[ ]:


def iqr(col):
    q1,q3 = np.quantile(taxi[col],[0.25,0.75])
    iqR = q3 - q1
    ul = q3 + 1.5*iqR
    ll = q1 - 1.5*iqR
    taxi[col] = taxi[(taxi[col] <= ul) & (taxi[col] >= ll)][col]


# In[ ]:


#iqr('passenger_count')


# In[ ]:


iqr('dropoff_longitude')


# In[ ]:


iqr('dropoff_latitude')


# In[ ]:


iqr('pickup_longitude')


# In[ ]:


iqr('pickup_latitude')


# In[ ]:


iqr('fare_amount')


# ## Fill null values

# In[ ]:


taxi['fare_amount'].fillna(taxi['fare_amount'].mean(),inplace=True)


# In[ ]:


taxi['pickup_latitude'].fillna(method='ffill',inplace=True)


# In[ ]:


taxi['pickup_longitude'].fillna(method='ffill',inplace=True)


# In[ ]:


taxi['dropoff_latitude'].fillna(method='ffill',inplace=True)


# In[ ]:


taxi['dropoff_longitude'].fillna(method='ffill',inplace=True)


# In[ ]:


taxi.dropna(inplace=True)


# ## Calculate Distance using Haversine Formula

# In[ ]:


def calculateDistance(plat,dlat,plong,dlong):
    
    lat1,lat2,long1,long2 = map(np.radians,[plat,dlat,plong,dlong])
    diffLat = lat2 - lat1
    diffLong = long2 - long1
    r = 6371000
    
    a = (np.sin(diffLat/2))**2 + np.cos(lat1) * np.cos(lat2) * np.sin(diffLong/2)**2
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))
    diskm = (r*c)/1000
    return diskm


# In[ ]:


taxi['travel_distance(km)'] = calculateDistance(taxi.pickup_latitude,taxi.dropoff_latitude,taxi.pickup_longitude,taxi.dropoff_longitude)


# In[ ]:


taxi.drop(['pickup_datetime'],axis=1,inplace=True)


# In[ ]:


taxi.head()


# # Visualization

# In[ ]:


sns.countplot(taxi.passenger_count)

people generally prefer to travel alone than sharing.
# In[ ]:


sns.heatmap(taxi.corr(),annot=True)

fare has high correlation with travel distance.
# ## Plot the latitudes & longitudes

# In[ ]:


print(taxi.pickup_latitude.min(),taxi.pickup_latitude.max())
print(taxi.pickup_longitude.min(),taxi.pickup_longitude.max())
print(taxi.dropoff_latitude.min(),taxi.dropoff_latitude.max())
print(taxi.dropoff_longitude.min(),taxi.pickup_longitude.max())


# In[ ]:


def plotLocation(lat,long,colormap):
    x_range, y_range = ((40.686192, 40.818789),(-74.032472, -73.92981))
    cvs = ds.Canvas( x_range = x_range, y_range = y_range)
    agg = cvs.points(x=lat,y=long,source=taxi)
    img = ds.transfer_functions.shade(agg, cmap = colormap)
    return ds.transfer_functions.set_background(img,'black')


# In[ ]:


plotLocation('pickup_latitude','pickup_longitude',ds.colors.viridis)


# In[ ]:


plotLocation('dropoff_latitude','dropoff_longitude',ds.colors.Hot)


# # Linear Regression

# ## Split features & target variables

# In[ ]:


x = taxi.drop(['key','fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)
y = taxi['fare_amount']


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30)


# In[ ]:


xtrain.shape,ytrain.shape,xtest.shape,ytest.shape


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)


# ## Check accuracy

# In[ ]:


r2_score(ytest,ypred)


# In[ ]:


mean_squared_error(ytest,ypred)**0.5

