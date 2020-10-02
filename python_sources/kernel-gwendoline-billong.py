#!/usr/bin/env python
# coding: utf-8

# In[71]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from math import radians, cos, sin, asin, sqrt

import os
print(os.listdir("../input"))


# # Import the train dataset 

# In[72]:


pn = pd.read_csv('../input/train.csv', index_col='id') 
pn['pickup_datetime'] = pd.to_datetime(pn['pickup_datetime']) #i Convert the date  type as the  le date format 
pn['dropoff_datetime'] = pd.to_datetime(pn['dropoff_datetime']) #i Convert the date  type as the  le date format 
pn.head()


# In[73]:


pn = pn[pn.passenger_count >= 1 ]
pn = pn[pn.trip_duration <= 7200]
pn = pn[pn.trip_duration >= 200]


# In[74]:


pn['year_pickup'] = pn['pickup_datetime'].dt.year 
pn['month_pickup']=pn['pickup_datetime'].dt.month 
pn['day_pickup']=pn['pickup_datetime'].dt.day
pn['weekday_pickup'] = pn['pickup_datetime'].dt.weekday
pn['hour_pickup']=pn['pickup_datetime'].dt.hour


# In[75]:


pn.describe()


# In[76]:


selected_columns = ['passenger_count','pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                   'dropoff_latitude', 'year_pickup',
                   'month_pickup', 'day_pickup', 'hour_pickup','weekday_pickup', 
                   ]
X = pn[selected_columns]
y = pn['trip_duration']
X.shape, y.shape


# In[ ]:





# # Cross&Validation 

# In[77]:


cv = ShuffleSplit(1, test_size=0.01, train_size=0.02, random_state=0) 


# In[78]:


from sklearn.metrics import mean_squared_error
rf = RandomForestRegressor()
scores = cross_val_score(rf, X , y , cv=cv ,scoring='neg_mean_squared_log_error')
np.sqrt(- scores.mean())


# In[79]:


rf.fit (X,y)


# # Import the test dataset 
# 

# In[80]:


pn_test = pd.read_csv('../input/test.csv', index_col='id') 
pn_test['pickup_datetime'] = pd.to_datetime(pn_test['pickup_datetime']) #i Convert the date  type as the  le date format 
pn_test.head()


# In[81]:


pn_test['year_pickup'] = pn_test['pickup_datetime'].dt.year
pn_test['month_pickup']=pn_test['pickup_datetime'].dt.month 
pn_test['day_pickup']=pn_test['pickup_datetime'].dt.day
pn_test['weekday_pickup'] = pn_test['pickup_datetime'].dt.weekday
pn_test['hour_pickup']=pn_test['pickup_datetime'].dt.hour


# In[82]:


pn_test.head()


# In[83]:


X_test = pn_test[selected_columns]


# In[84]:


y_pred = rf.predict(X_test)
y_pred.mean()


# # Import the sample_submission dataset   

# In[85]:


sub = pd.read_csv('../input/sample_submission.csv')
sub.head()


# In[86]:


sub['trip_duration'] = y_pred 
sub.head()


# In[87]:



sub.describe()


# In[88]:


sub.to_csv('sub.csv', index=False)


# In[89]:


get_ipython().system('ls')


# In[ ]:




