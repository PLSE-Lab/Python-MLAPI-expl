#!/usr/bin/env python
# coding: utf-8

# # Using the Manhattan distance to approximate distances in the Manhattan.
# Many of use are familiar with different distance metrics from some math courses years ago. I believe that I'm not the only one who figured out that [Manhattan distance](https://en.wiktionary.org/wiki/Manhattan_distance) might be suitable for this problem. In this notebook I share my approach and results with that method.
# 
# This approach should clearly be an improvement to the standard eucledian distance that can be calculated with [haversine](https://en.wikipedia.org/wiki/Haversine_formula) or [Vincenty](https://en.wikipedia.org/wiki/Vincenty%27s_formulae) (which are different on larger distances, but in a small area as in this problem don't really have a difference). In New York you simply can't take the shortest route but have to follow the streets!
# 
# Although this approach does not hold on longer distances in the city or in many different special cases. For that you need to use actual map data, which is something that I wanted to leave out from this Mathematical simplification.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as b
from haversine import haversine
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans
from geopy.distance import vincenty

import matplotlib.pyplot as plt
from math import cos, sin, radians, sqrt
import numpy as np


# In[ ]:


ds=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


# Lets use a smaller dataset firts to develop a better distance function
examples = ds.head(1000)
examples = examples.copy()


# In[ ]:


examples['eucledian'] = examples.apply(lambda x: haversine((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude'])),axis=1)


# ## Approximating haversine distance
# Before we are able to calculate manhatan distances, we first need a function to approximate the harversine or vincenty distance. The harversine formula can not really be easily used to calculate manhatan distances, so let's simplify it a little bit. Instead of taking into account that earth is a sphere, we can just calculate the amount one has to walk east (w) and north (h) to get from point A to point B.

# In[ ]:


def simple_eucledian(a, b):
    h = abs(a[0]-b[0])
    
    # Normalize east-west distance to the same scale as north-south
    w = abs((a[1]-b[1]) * cos(radians(a[0])))
    
    return sqrt(w**2+h**2)


# In[ ]:


examples['simple_eucledian']=examples.apply(lambda x: simple_eucledian((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude'])),axis=1)

ratio = examples['simple_eucledian']/examples['eucledian']
print(np.min(ratio), np.max(ratio), (np.max(ratio) - np.min(ratio)) / np.max(ratio))

plt.plot(ratio)
plt.show()


# Close enough (< 1%). We can use this to build the function to calculate Manhatan distance.
# 
# ## Manhatan distance
# 
# One thing to note is that streets on Manhatan are not exactly North-South and East-West -- they are in a degree of 29 degrees. The angle effects the distance metrics, so let's perform a rotation before calculating the final value.
# 
# http://www.charlespetzold.com/etc/AvenuesOfManhattan/
# 
# Let's also perform some unit tests to make sure that everyting works correctly. <u>Data Scientists need testing too!</u>

# In[ ]:


def get_rotation(alpha):
    alpha_r = radians(alpha)
    return [[cos(alpha_r), -sin(alpha_r)],[sin(alpha_r), cos(alpha_r)]]

def manhattan(a, b, rotation):
    h = a[0]-b[0]
    w = (a[1]-b[1]) * cos(radians(a[0]))   # Normalize east-west distance to the same scale as north-south
    x = np.dot(rotation,[w, h])
    return abs(x[0])+abs(x[1])


# In[ ]:


import unittest
from random import random

class TestStringMethods(unittest.TestCase):

    def test_no_rotation(self):
        r = get_rotation(0)
        self.assertAlmostEqual(manhattan((0, 0), (1, 0), r), 1, delta = 0.001)
        self.assertAlmostEqual(manhattan((0, 0), (0, 1), r), 1, delta = 0.001)
        self.assertAlmostEqual(manhattan((0, 0), (1, 1), r), 2, delta = 0.001)
        self.assertAlmostEqual(manhattan((1, 1), (1, 1), r), 0, delta = 0.001)
        self.assertAlmostEqual(manhattan((1, 1), (3, 3), r), 4, delta = 0.001)

    def test_90_deg_rotation(self):
        r = get_rotation(90)
        self.assertAlmostEqual(manhattan((0, 0), (1, 0), r), 1, delta = 0.001)
        self.assertAlmostEqual(manhattan((0, 0), (0, 1), r), 1, delta = 0.001)
        self.assertAlmostEqual(manhattan((0, 0), (1, 1), r), 2, delta = 0.001)
        self.assertAlmostEqual(manhattan((1, 1), (1, 1), r), 0, delta = 0.001)
        self.assertAlmostEqual(manhattan((1, 1), (3, 3), r), 4, delta = 0.001) 

    def test_45_rotation(self):
        r = get_rotation(45)
        self.assertAlmostEqual(manhattan((0, 0), (1, 1), r), sqrt(2), delta = 0.001)
        
    def test_random_rotation(self):
        alpha = random()*90
        r = get_rotation(alpha)
        self.assertAlmostEqual(manhattan((0, 0), (0, 1), r), sin(radians(alpha)) + cos(radians(alpha)), delta = 0.001)
        self.assertAlmostEqual(manhattan((0, 0), (1, 0), r), sin(radians(alpha)) + cos(radians(alpha)), delta = 0.001)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# ## Move to the big data set
# We now have our Manhatan distance function ready so we can move to the larger dataset. This part includes some standard feature engineering.

# In[ ]:


me = np.mean(ds['trip_duration'])
st = np.std(ds['trip_duration'])
ds = ds[ds['trip_duration'] <= me + 2*st]
ds = ds[ds['trip_duration'] >= me - 2*st]


# In[ ]:


ds = ds[ds['pickup_longitude'] <= -73.75]
ds = ds[ds['pickup_longitude'] >= -74.03]
ds = ds[ds['pickup_latitude'] <= 40.85]
ds = ds[ds['pickup_latitude'] >= 40.63]
ds = ds[ds['dropoff_longitude'] <= -73.75]
ds = ds[ds['dropoff_longitude'] >= -74.03]
ds = ds[ds['dropoff_latitude'] <= 40.85]
ds = ds[ds['dropoff_latitude'] >= 40.63]


# In[ ]:


coords = np.vstack((ds[['pickup_latitude', 'pickup_longitude']].values,
                    ds[['dropoff_latitude', 'dropoff_longitude']].values))


# In[ ]:


sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])


# In[ ]:


ds.loc[:,'pickup_id']=kmeans.predict(ds[['pickup_latitude', 'pickup_longitude']])
ds.loc[:,'dropoff_id']=kmeans.predict(ds[['dropoff_latitude', 'dropoff_longitude']])


# In[ ]:


test.loc[:,'pickup_id']=kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:,'dropoff_id']=kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])


# In[ ]:


ds['pickup_datetime']=pd.to_datetime(ds['pickup_datetime'])
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])


# In[ ]:


ds['pickup_weekday']=ds['pickup_datetime'].dt.weekday
ds['pickup_hour']=ds['pickup_datetime'].dt.hour
ds['pickup_month']=ds['pickup_datetime'].dt.month


# In[ ]:


test['pickup_weekday']=test['pickup_datetime'].dt.weekday
test['pickup_hour']=test['pickup_datetime'].dt.hour
test['pickup_month']=test['pickup_datetime'].dt.month


# In[ ]:


ds['isweekend']= ds.apply(lambda x : (x['pickup_weekday']==6 | x['pickup_weekday']==5),axis=1)
ds['isweekend']=ds['isweekend'].map({True: 1, False:0})
ds['store_and_fwd_flag']=ds['store_and_fwd_flag'].map({'N': 1, 'Y':0})


# In[ ]:


test['isweekend']= test.apply(lambda x : (x['pickup_weekday']==6 | x['pickup_weekday']==5),axis=1)
test['isweekend']=test['isweekend'].map({True: 1, False:0})
test['store_and_fwd_flag']=test['store_and_fwd_flag'].map({'N': 1, 'Y':0})


# ## Calculate distances features

# In[ ]:


ds['eucledian']=ds.apply(lambda x: haversine((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude'])),axis=1)
test['eucledian']=test.apply(lambda x: haversine((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude'])),axis=1)


# In[ ]:


# The roads are in a 29 degree angle to North-South axis
# http://www.charlespetzold.com/etc/AvenuesOfManhattan/

ds['manhattan']=ds.apply(lambda x: manhattan((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude']), get_rotation(29)),axis=1)
test['manhattan']=test.apply(lambda x: manhattan((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude']), get_rotation(29)),axis=1)


# ## Teach the model

# In[ ]:


feature_cols=['vendor_id','passenger_count','pickup_id','dropoff_id','pickup_latitude','dropoff_latitude','pickup_weekday','pickup_hour'
              ,'pickup_month','isweekend','store_and_fwd_flag' ,'eucledian', 'manhattan']
X=ds[feature_cols]
Y=ds['trip_duration']


# In[ ]:


X_train,X_val,Y_train,Y_val= b.train_test_split(X,Y,test_size=0.2, random_state=420)


# In[ ]:


rf=RandomForestRegressor(n_estimators=20,max_depth=50,min_samples_split=10)
rf.fit(X_train,Y_train)


# ## Validate the results

# In[ ]:


features=rf.feature_importances_
for i, weight in enumerate(features):
    print(feature_cols[i], "\t", weight)


# In[ ]:


Y_val_pred=rf.predict(X_val)


# In[ ]:


def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))


# In[ ]:


# When I started with this fork, this was the RMSLE
benchmark = 0.41319301815016068
RMSLE = rmsle(Y_val_pred,Y_val)
print("RMSLE", RMSLE)
print("Improvement to benchmark", (benchmark-RMSLE)/benchmark *100, "%")


# ## Fit the final model with all data

# In[ ]:


rf_full=RandomForestRegressor(n_estimators=20,max_depth=50,min_samples_split=10)
rf_full.fit(X,Y)
features=rf.feature_importances_
for i, weight in enumerate(features):
    print(feature_cols[i], "\t", weight)


# In[ ]:


y_test_pred=rf_full.predict(test[feature_cols])


# In[ ]:


output = pd.DataFrame()
output['id'] = test['id']
output['trip_duration'] = y_test_pred
output.to_csv('randomforest.csv', index=False)

