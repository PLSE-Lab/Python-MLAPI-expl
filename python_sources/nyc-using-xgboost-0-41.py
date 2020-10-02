#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as b
from haversine import haversine
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans
from xgboost import XGBRegressor
import tensorflow as tf
from sklearn.metrics import mean_absolute_error


# In[ ]:



ds=pd.read_csv('../input/nyc-taxi-trip-duration/train.zip',compression='zip')
test=pd.read_csv('../input/nyc-taxi-trip-duration/test.zip',compression='zip')
holidays=pd.read_csv('../input/nyc2016holidays/NYC_2016Holidays.csv')
weather=pd.read_csv('../input/knycmetars2016/KNYC_Metars.csv')
nyc_weather=pd.read_csv('../input/weather-data-in-new-york-city-2016/weather_data_nyc_centralpark_2016(1).csv')


# In[ ]:


# nyc_weather['date'] = pd.to_datetime(nyc_weather['date'],format="%Y-%m-%d %H:%M:%S")


# In[ ]:



# nyc_weather.loc[nyc_weather['snow fall']=='T']=0.01
# nyc_weather['year']=nyc_weather['date'].dt.year
# nyc_weather['pickup_day']=nyc_weather['date'].dt.day
# nyc_weather['pickup_month']=nyc_weather['date'].dt.month
# nyc_weather['pickup_hour']=nyc_weather['date'].dt.hour
# nyc_weather=nyc_weather[nyc_weather['year']==2016][['pickup_day','pickup_month','pickup_hour','snow fall','snow depth','maximum temerature']]


# Removing Outlier from the train set which are more than 2 standard devations apart
# 

# In[ ]:


me = np.mean(ds['trip_duration'])
st = np.std(ds['trip_duration'])
ds = ds[ds['trip_duration'] <= me + 2*st]
ds = ds[ds['trip_duration'] >= me - 2*st]


# In[ ]:


ds['trip_duration_log']=(ds['trip_duration']+1).apply(np.log)


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
kmeans = MiniBatchKMeans(n_clusters=50, batch_size=10000).fit(coords[sample_ind])


# In[ ]:


ds['pickup_id']=kmeans.predict(ds[['pickup_latitude', 'pickup_longitude']])
ds['dropoff_id']=kmeans.predict(ds[['dropoff_latitude', 'dropoff_longitude']])


# In[ ]:


test['pickup_id']=kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test['dropoff_id']=kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])


# In[ ]:


ds['pickup_datetime']=pd.to_datetime(ds['pickup_datetime'])
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])


# In[ ]:


meanvisible=np.mean(weather['Visibility'])
weather.fillna(value=meanvisible,inplace=True)


# In[ ]:


weather['Time']=pd.to_datetime(weather['Time'])
weather['year']=weather['Time'].dt.year
weather['pickup_day']=weather['Time'].dt.day
weather['pickup_month']=weather['Time'].dt.month
weather['pickup_hour']=weather['Time'].dt.hour
weather=weather[weather['year']==2016][['pickup_day','pickup_month','pickup_hour','Temp.','Precip','Visibility']]


# In[ ]:


ds['distance']=ds.apply(lambda x: haversine((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude'])),axis=1)
test['distance']=test.apply(lambda x: haversine((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude'])),axis=1)


# In[ ]:


# from geopy.distance import vincenty
# ds['geo_distance']=ds.apply(lambda x: vincenty((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude'])).miles,axis=1)
# test['geo_distance']=test.apply(lambda x: vincenty((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude'])).miles,axis=1)


# In[ ]:


# from geopy.distance import great_circle
# ds['greatcircle_distance']=ds.apply(lambda x: great_circle((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude'])).miles,axis=1)
# test['greatcircle_distance']=test.apply(lambda x: great_circle((x['pickup_latitude'] ,x['pickup_longitude']),(x['dropoff_latitude'], x['dropoff_longitude'])).miles,axis=1)


# In[ ]:


ds['pickup_weekday']=ds['pickup_datetime'].dt.weekday
ds['pickup_hour']=ds['pickup_datetime'].dt.hour
ds['pickup_month']=ds['pickup_datetime'].dt.month
ds['pickup_day']=ds['pickup_datetime'].dt.day


# In[ ]:


test['pickup_weekday']=test['pickup_datetime'].dt.weekday
test['pickup_hour']=test['pickup_datetime'].dt.hour
test['pickup_month']=test['pickup_datetime'].dt.month
test['pickup_day']=ds['pickup_datetime'].dt.day


# In[ ]:


# ds=pd.merge(ds,nyc_weather, on = ['pickup_month', 'pickup_day', 'pickup_hour'], how = 'left')
# test=pd.merge(test,nyc_weather, on = ['pickup_month', 'pickup_day', 'pickup_hour'], how = 'left')


# In[ ]:


ds['isweekend']= ds.apply(lambda x : (x['pickup_weekday']==6 | x['pickup_weekday']==5),axis=1)
ds['isweekend']=ds['isweekend'].map({True: 1, False:0})
ds['store_and_fwd_flag']=ds['store_and_fwd_flag'].map({'N': 1, 'Y':0})


# In[ ]:


test['isweekend']= test.apply(lambda x : (x['pickup_weekday']==6 | x['pickup_weekday']==5),axis=1)
test['isweekend']=test['isweekend'].map({True: 1, False:0})
test['store_and_fwd_flag']=test['store_and_fwd_flag'].map({'N': 1, 'Y':0})


# In[ ]:


feature_cols=['vendor_id','passenger_count','pickup_id','dropoff_id','pickup_weekday','pickup_hour'
              ,'pickup_month','store_and_fwd_flag' ,'distance']
# ,'distance','greatcircle_distance','pickup_latitude','dropoff_latitude','geo_distance','Temp.','Precip','Visibility'
X=ds[feature_cols]
Y=ds['trip_duration_log']
test_features=test[feature_cols]


# In[ ]:


X_train,X_test,Y_train,Y_test= b.train_test_split(X,Y,test_size=0.2, random_state=420)
X_train,X_Val,Y_train,Y_Val= b.train_test_split(X_train,Y_train,test_size=0.1, random_state=420)


# In[ ]:


taxi_trip_model = XGBRegressor(n_estimators=200,learning_rate=0.05)
taxi_trip_model.fit(X_train,Y_train,early_stopping_rounds=10,
             eval_set=[(X_Val, Y_Val)])
pred_test = taxi_trip_model.predict(X_test, ntree_limit=taxi_trip_model.best_ntree_limit)


# In[ ]:


mae = mean_absolute_error(Y_test,pred_test)
print(mae)


# In[ ]:


predtest =  taxi_trip_model.predict(test_features,ntree_limit=taxi_trip_model.best_ntree_limit)


# In[ ]:


y_test =np.exp(predtest)-1


# In[ ]:


output = pd.DataFrame({'id': test['id'],
                       'trip_duration': y_test})
output.to_csv('submission_xgb_best.csv', index=False)


# Neural Network

# In[ ]:


# learning_rate = 0.01
# training_epochs = 15
# batch_size = 100
# display_step = 50
# n_hidden_1 = 256 # 1st layer number of neurons
# n_hidden_2 = 256 # 2nd layer number of neurons
# n_input = X_train.shape[0]
# n_out=1 #linear regression


# In[ ]:


# X = tf.placeholder("float", [None, n_input])
# Y = tf.placeholder("float", [None, n_out])
# weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_hidden_2, n_out]))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_out]))
# }


# In[ ]:


# def mlp(x):
#     # Hidden fully connected layer with 256 neurons
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     # Hidden fully connected layer with 256 neurons
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     # Output fully connected layer with a neuron for each class
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
# return out_layer


# In[ ]:


# data_tr  = xgb.DMatrix(X_train, label=Y_train)
# data_cv  = xgb.DMatrix(X_Val , label=Y_Val)
# evallist = [(data_tr, 'train'), (data_cv, 'valid')]


# In[ ]:


# parms = {'max_depth':8, #maximum depth of a tree
#          'objective':'reg:linear',
#          'eta'      :0.3,
#          'subsample':0.9,
#          'lambda '  :4, #L2 regularization term
#          'colsample_bytree ':0.7,
#          'colsample_bylevel':1,
#          'min_child_weight': 10,
#          'nthread'  :-1}  #number of cpu core to use

# model = xgb.train(parms, data_tr, num_boost_round=1000, evals = evallist,
#                   early_stopping_rounds=30, maximize=False, 
#                   verbose_eval=100)

# print('score = %1.5f, n_boost_round =%d.'%(model.best_score,model.best_iteration))


# In[ ]:


# data_test = xgb.DMatrix(test_features)
# ytest = model.predict(data_test)


# In[ ]:


# xgb.plot_importance(model)


# In[ ]:


# y_test = np.exp(ytest)-1


# In[ ]:


# output = pd.DataFrame()
# output['id'] = test['id']
# output['trip_duration'] = y_test
# output.to_csv('randomforest.csv', index=False)

