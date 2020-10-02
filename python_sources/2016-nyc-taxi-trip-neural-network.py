#!/usr/bin/env python
# coding: utf-8

# # Import packages

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing

from math import radians, cos, sin, asin, sqrt

from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# # Change display format to display 18 decimal precision

# In[2]:


pd.options.display.float_format = "{:.18f}".format


# # Show files in Kernel Virtual Machine

# In[3]:


get_ipython().system('ls ../input')


# # Load train dataset

# In[4]:


df = pd.read_csv('../input/train.csv')
df.head()


# # Remove empty values

# In[5]:


df.dropna(inplace=True)


# In[ ]:


mean, std_deviation = np.mean(df['trip_duration']), np.std(df['trip_duration'])
df = df[df['trip_duration'] <= mean + 2 * std_deviation]
df = df[df['trip_duration'] >= mean - 2 * std_deviation]


# # Create shortest variables to large string labels

# In[6]:


plg, plt = 'pickup_longitude', 'pickup_latitude'
dlg, dlt = 'dropoff_longitude', 'dropoff_latitude'
pdt, ddt = 'pickup_datetime', 'dropoff_datetime'


# # Function to calculate distance from pickup to dropoff

# In[7]:


# https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def euclidian_distance(x):
    x1, y1 = np.float64(x[plg]), np.float64(x[plt])
    x2, y2 = np.float64(x[dlg]), np.float64(x[dlt])    
    return haversine(x1, y1, x2, y2)


# # Create column with calculated distance from pickup to dropoff

# In[8]:


get_ipython().run_cell_magic('time', '', "df['distance'] = df[[plg, plt, dlg, dlt]].apply(euclidian_distance, axis=1)\ndf.head()")


# # Convert string to datetime

# In[9]:


df[pdt] = df[pdt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df[ddt] = df[ddt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))


# # Create colums from pickuptime

# In[10]:


df['month'] = df[pdt].apply(lambda x : x.month)
df['weekDay'] = df[pdt].apply(lambda x : x.weekday())
df['dayMonth'] = df[pdt].apply(lambda x : x.day)
df['pickupTimeMinutes'] = df[pdt].apply(lambda x : x.hour * 60.0 + x.minute)
df.head()


# # Remove unecessary columns

# In[11]:


df.drop(['id', pdt, ddt, dlg, dlt, 'store_and_fwd_flag'], inplace=True, axis=1)
df.head()


# # Rearrange columns

# In[12]:


df = df[[plg, plt, 'distance', 'month', 'dayMonth', 'weekDay', 'pickupTimeMinutes', 'passenger_count', 'vendor_id', 'trip_duration']]
df.head()


# # Get train data

# In[13]:


X, y = df.iloc[:, :-1], df.iloc[:, -1]


# # Normalize input

# In[14]:


scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)


# # Kfold to split train and test data

# In[15]:


kf = KFold(n_splits=3, shuffle=True, random_state=4305)


# # Create Multilayer Perceptron models

# In[16]:


mlp = MLPRegressor()
param_grid = {'hidden_layer_sizes': [(9, 9), (8, 8), (7, 7), (6, 6), (5, 5), (4, 4), (3, 3), (9, 9, 9), (8, 8, 8), (7, 7, 7), (6, 6, 6), (5, 5, 5), (4, 4, 4), (3, 3, 3)],
              'activation': ['relu'],
              'solver': ['adam'],
              'nesterovs_momentum' : [True],
              'momentum' : [0.9, 0.8, 0.7, 0.6, 0.5],                          
              'learning_rate_init': [0.001, 0.01, 0.1, 0.3, 0.5, 1, 2],              
              'learning_rate' : ['adaptive'],
              'max_iter': [1000],
              'early_stopping': [False],
              'warm_start': [True]
             }


# # Create function to calculate error

# In[17]:


# https://www.kaggle.com/jpopham91/rmlse-vectorized
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.square(np.subtract(np.log1p(y_true), np.log1p(y_pred)))))


# # Create SearchCV

# In[23]:


search_cv = RandomizedSearchCV(mlp, param_grid, scoring=make_scorer(rmsle, greater_is_better=False),
                   cv=kf, verbose=3, pre_dispatch='2*n_jobs')


# # Train models

# In[ ]:


get_ipython().run_cell_magic('time', '', 'search_cv.fit(X, y)')


# # Show Top5 models

# In[ ]:


results = pd.DataFrame(search_cv.cv_results_).sort_values(by='mean_test_score', ascending=False)
results.head()


# #  Show Top1 model

# In[ ]:


search_cv.best_estimator_


# # Load test dataset

# In[ ]:


df_test = pd.read_csv('../input/test.csv')
df_test.head()


# # Prepocessing test dataset

# In[ ]:


df_test['distance'] = df_test[[plg, plt, dlg, dlt]].apply(euclidian_distance, axis=1)
df_test[pdt] = df_test[pdt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df_test['month'] = df_test[pdt].apply(lambda x : x.month)
df_test['weekDay'] = df_test[pdt].apply(lambda x : x.weekday())
df_test['dayMonth'] = df_test[pdt].apply(lambda x : x.day)
df_test['pickupTimeMinutes'] = df_test[pdt].apply(lambda x : x.hour * 60.0 + x.minute)
df_test.drop(['pickup_datetime', dlg, dlt, 'store_and_fwd_flag'], inplace=True, axis=1)
df_test = df_test[['id', plg, plt, 'distance', 'month', 'dayMonth', 'weekDay', 'pickupTimeMinutes', 'passenger_count', 'vendor_id']]
df_test.head()


# # Split test data

# In[ ]:


X_id, X_test = df_test.iloc[:, 0], df_test.iloc[:, 1:]
X_id.shape, X_test.shape


# # Normalize input

# In[ ]:


scaler = preprocessing.StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)


# # Predict outputs

# In[ ]:


y_pred = search_cv.best_estimator_.predict(X_test)


# # Generate output file

# In[ ]:


df_output = pd.DataFrame({'id' : X_id, 'trip_duration': y_pred})
df_output.to_csv('submission.csv', index=False)


# In[ ]:


pd.read_csv('submission.csv').head()

