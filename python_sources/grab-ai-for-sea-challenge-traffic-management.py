#!/usr/bin/env python
# coding: utf-8

# Import libraries
# pandas, numpy, xgboost, randomforest and sklearn tools

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import Geohash

import time

import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error


import seaborn as sns


# Get the data into the dataframe

# In[ ]:


df = pd.read_csv('../input/training.csv')


# Sample some of data; for exploration and earlier iteration

# In[ ]:


df = df.sample(10000)


# Function to round the latitude/longitude decimals points to match the geohash6 floating error and addressing dimensional overfitting.

# In[ ]:


def round3(x):
    return round(float(x)*10000)/10000


# Convert geohash6 to lat,lng. Represent spatial features

# In[ ]:


df['lat'] = df.apply(lambda x: round3(Geohash.decode_exactly(x['geohash6'])[0]), axis=1)

df['lng'] = df.apply(lambda x: round3(Geohash.decode_exactly(x['geohash6'])[1]), axis=1)


# Split timestamp into hour and minute
# 

# In[ ]:


df['hour'] = df.apply(lambda x: float(x['timestamp'].split(':')[0]), axis=1)

df['minute'] = df.apply(lambda x: float(x['timestamp'].split(':')[1]), axis= 1)


# Compute day of week, as the users behavior are affected by the day factor eg:weekeends, weekdays.

# In[ ]:


df['dow'] =  df.apply(lambda x: x['day']%7, axis =1)


# Some simple Analysis

# Get the dataset basic info

# In[ ]:


df.info()


# Features correlation to the target variable 'demand'

# In[ ]:


df.corr()['demand']


# In[ ]:


sns.lineplot(data=df, x='hour', y='demand')


# In[ ]:


sns.lineplot(data=df, x='dow', y='demand')


# In[ ]:


sns.lineplot(data=df, x='lat', y='demand')


# In[ ]:


sns.lineplot(data=df, x='lng', y='demand')


# In[ ]:


sns.scatterplot(data=df, x='lng', y='lat', hue='demand')


# we can see some high demand areas are clustered, thus we can say the demands are spatio dependant

# In[ ]:


sns.lineplot(data=df, x='minute', y='demand')


# Select only high correlation features, from the analysis we can see the minute features doest not vary much thus we remove it

# In[ ]:


selectedColumn = ['lat','lng','hour','dow']


# #Random Forest

# In[ ]:


clf = RandomForestRegressor(max_depth=25,  n_estimators=240)


# In[ ]:


dfTrain, dfTest = train_test_split(df,test_size=0.2)


# In[ ]:


clf.fit(X=dfTrain[selectedColumn],y=dfTrain['demand'])


# In[ ]:


dfTest['predict'] = clf.predict(X=dfTest[selectedColumn])


# In[ ]:


mean_squared_error(dfTest['demand'], dfTest['predict'])


# #XGBRegressor

# In[ ]:


xgb_reg = xgb.XGBRegressor(learning_rate=0.01,max_depth=25,n_estimators=240, tree_method='hist')


# In[ ]:


xgb_reg.fit(X=dfTrain[selectedColumn],y=dfTrain['demand'])


# In[ ]:


dfTest['predict_xgb'] = xgb_reg.predict(data=dfTest[selectedColumn])


# In[ ]:


mean_squared_error(dfTest['demand'], dfTest['predict_xgb'])


# In[ ]:


xgb.plot_importance(xgb_reg)


# In[ ]:


xgb.plot_importance(xgb_reg, importance_type='cover')


# In conclusion, both random forest and xgboost regressor work well on the dataset and the use cases. 
# 
# However, there are some variables like weather, calendar(holidays) that may help to further 

# Now, we repeat the above steps with full data

# In[ ]:


df = pd.read_csv('../input/training.csv')


# In[ ]:


df['lat'] = df.apply(lambda x: round3(Geohash.decode_exactly(x['geohash6'])[0]), axis=1)

df['lng'] = df.apply(lambda x: round3(Geohash.decode_exactly(x['geohash6'])[1]), axis=1)


# In[ ]:


df['hour'] = df.apply(lambda x: float(x['timestamp'].split(':')[0]), axis=1)

df['minute'] = df.apply(lambda x: float(x['timestamp'].split(':')[1]), axis= 1)


# In[ ]:


df['dow'] =  df.apply(lambda x: x['day']%7, axis =1)


# In[ ]:


dfTrain, dfTest = train_test_split(df,test_size=0.2)


# # XGBRegressor on GPU
# 
# we choose xgboost regressor over the sklearn random forest for full dataset, as xgb boost support GPU, and we need to do on GPU as our data is quite big.

# In[ ]:


xgb_reg = xgb.XGBRegressor(learning_rate=0.01,max_depth=25,n_estimators=240, tree_method='gpu_hist')


# In[ ]:


starttime = time.time()
xgb_reg.fit(X=dfTrain[selectedColumn],y=dfTrain['demand'])
print('Training Time (s): ', time.time() - starttime)


# In[ ]:


dfTest['predict_xgb'] = xgb_reg.predict(data=dfTest[selectedColumn])


# In[ ]:


mean_squared_error(dfTest['demand'], dfTest['predict_xgb'])


# Conclusion: XGBoost Regressor run well with whole data (80% training, 20% testing) and the MSE is very good 0.0039.
# 
# For future plan, we can add more data such as weather, local calendar etc.

# Below is the function to do prediction with the saved model.

# Save the model

# In[ ]:


xgb_reg.save_model('xgbReg.model')


# Predict Demand Function, consume 
# 1. Saved Model
# 2. geohash6
# 3. day
# 4. timestamp

# In[ ]:


def predictDemand(savedModel, geohash6='qp03wc', day=100, timestamp='00:00'):
    
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    import Geohash
    
    def round3(x):
        return round(float(x)*10000)/10000
    
    
    lat = round3(Geohash.decode_exactly(geohash6)[0])
    lng = round3(Geohash.decode_exactly(geohash6)[1])
    
    hour = float(timestamp.split(':')[0])
    
    dow = day%7
    
    dataX = pd.DataFrame({'lat': [lat], 'lng': [lng], 'hour': [hour], 'dow': [dow] })
    
    output = savedModel.predict(dataX)
    
    return output[0]
                 
                 


# Usage Example

# In[ ]:


import xgboost as xgb

# load model
savedModel = xgb.XGBRegressor()
savedModel.load_model('xgbReg.model')

predictDemand(savedModel,geohash6='qp09sy', day=39, timestamp='3:0' )


# In[ ]:




