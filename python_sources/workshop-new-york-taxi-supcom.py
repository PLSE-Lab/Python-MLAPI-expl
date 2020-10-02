#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
get_ipython().run_line_magic('matplotlib', 'inline')
import gc
from sklearn.cross_validation import train_test_split

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train = pd.read_csv("../input/train.csv", nrows=2000000)
test= pd.read_csv("../input/test.csv")


# In[ ]:


gc.collect()


# In[ ]:


ts=time.time()
train=train[(train.fare_amount>0) & (train.fare_amount<200)]
train=train[(train.passenger_count>=0) &(train.passenger_count<=8)]
train= train[(train.pickup_longitude>=-74.5) & (train.pickup_longitude<=-72.8)]
train= train[(train.pickup_latitude>=40.5) & (train.pickup_latitude<=41.8)]
train= train[(train.dropoff_longitude>=-74.5) & (train.dropoff_longitude<=-72.8)]
train= train[(train.dropoff_latitude>=40.5) & (train.dropoff_latitude<=41.8)]
train = train.dropna(how = 'any', axis = 'rows')
time.time()-ts


# In[ ]:


ts=time.time()
train["key"]=pd.to_datetime(train.key,format="%Y-%m-%d %H:%M:%S")
train["year"]=train.key.dt.year
train["month"]=train.key.dt.month
train["day"]=train.key.dt.day
train["dayOfWeek"]=train.key.dt.dayofweek
train["hour"]=train.key.dt.hour
time.time()-ts


# In[ ]:


ts=time.time()
test["key"]=pd.to_datetime(test.key,format="%Y-%m-%d %H:%M:%S")
test["year"]=test.key.dt.year
test["month"]=test.key.dt.month
test["day"]=test.key.dt.day
test["dayOfWeek"]=test.key.dt.dayofweek
test["hour"]=test.key.dt.hour
time.time()-ts


# In[ ]:


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


# In[ ]:


ts=time.time()
train['distance']=distance(train.pickup_latitude,train.pickup_longitude,train.dropoff_latitude,train.dropoff_longitude)
time.time()-ts


# In[ ]:


ts=time.time()
test['distance']=distance(test.pickup_latitude,test.pickup_longitude,test.dropoff_latitude,test.dropoff_longitude)
time.time()-ts


# In[ ]:


ts=time.time()
train["diff_lat"]=abs(train.pickup_latitude-train.dropoff_latitude)
train["diff_long"]=abs(train.pickup_longitude-train.dropoff_longitude)
time.time()-ts


# In[ ]:


ts=time.time()
test["diff_lat"]=abs(test.pickup_latitude-test.dropoff_latitude)
test["diff_long"]=abs(test.pickup_longitude-test.dropoff_longitude)
time.time()-ts


# In[ ]:


nyc = ("pk_dist_nyc",40.7141667 ,-74.0063889 )
jfk = ("pk_dist_jfk",40.6441666667,-73.7822222222 )
ewr = ("pk_dist_ewr",40.69, -74.175 )
lgr = ("pk_dist_lgr",40.77, -73.87 )
centers=[nyc,jfk,ewr,lgr]

ts=time.time()
print("pickup columns")
for place in centers:
    train[place[0]]=distance(train.pickup_latitude,train.pickup_longitude,place[1],place[2])
    print(place[0]," is done")


nyc = ("drop_dist_nyc",40.7141667 ,-74.0063889 )
jfk = ("drop_dist_jfk",40.6441666667,-73.7822222222 )
ewr = ("drop_dist_ewr",40.69, -74.175 )
lgr = ("drop_dist_lgr",40.77, -73.87 )
centers=[nyc,jfk,ewr,lgr]

print("dropoff columns")
for place in centers:
    train[place[0]]=distance(train.dropoff_latitude,train.dropoff_longitude,place[1],place[2])
    print(place[0]," is done")
time.time()-ts


# In[ ]:


nyc = ("pk_dist_nyc",40.7141667 ,-74.0063889 )
jfk = ("pk_dist_jfk",40.6441666667,-73.7822222222 )
ewr = ("pk_dist_ewr",40.69, -74.175 )
lgr = ("pk_dist_lgr",40.77, -73.87 )
centers=[nyc,jfk,ewr,lgr]

ts=time.time()
print("pickup columns")
for place in centers:
    test[place[0]]=distance(test.pickup_latitude,test.pickup_longitude,place[1],place[2])
    print(place[0]," is done")


nyc = ("drop_dist_nyc",40.7141667 ,-74.0063889 )
jfk = ("drop_dist_jfk",40.6441666667,-73.7822222222 )
ewr = ("drop_dist_ewr",40.69, -74.175 )
lgr = ("drop_dist_lgr",40.77, -73.87 )
centers=[nyc,jfk,ewr,lgr]

print("dropoff columns")
for place in centers:
    test[place[0]]=distance(test.dropoff_latitude,test.dropoff_longitude,place[1],place[2])
    print(place[0]," is done")
time.time()-ts


# In[ ]:


cordonne= ['pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude']
limit=[-74.5,40.5,-74.5,40.5]


# In[ ]:


train.pickup_longitude=(train.pickup_longitude+74.5).abs()
train.pickup_latitude=(train.pickup_latitude-40.5).abs()
train.dropoff_longitude=(train.dropoff_longitude+74.5).abs()
train.dropoff_latitude=(train.dropoff_latitude-40.5).abs()

test.pickup_longitude=(test.pickup_longitude+74.5).abs()
test.pickup_latitude=(test.pickup_latitude-40.5).abs()
test.dropoff_longitude=(test.dropoff_longitude+74.5).abs()
test.dropoff_latitude=(test.dropoff_latitude-40.5).abs()


# In[ ]:


for col in cordonne:
    train[col+"_cos"]=np.cos(train[col])
    train[col+"_sin"]=np.sin(train[col])
    train[col+"_carre"]=train[col]*train[col]
for col in cordonne:
    test[col+"_cos"]=np.cos(test[col])
    test[col+"_sin"]=np.sin(test[col])
    test[col+"_carre"]=test[col]*test[col]


# In[ ]:


train["pickup_product"]=train.pickup_longitude*train.pickup_latitude
train["dropoff_product"]=train.dropoff_longitude*train.dropoff_latitude

test["pickup_product"]=test.pickup_longitude*test.pickup_latitude
test["dropoff_product"]=test.dropoff_longitude*test.dropoff_latitude


# In[ ]:


train["year"]=train["year"]-2009
test["year"]=test["year"]-2009


# In[ ]:


train["month"]=train["month"]+train["year"]*12
test["month"]=test["month"]+test["year"]*12


# In[ ]:


def min (a,b):
    return (a+b-(a-b).abs())/2.0
for data in[train,test]:
    data["dist_nyc"]=min(data.drop_dist_nyc, data.pk_dist_nyc)
    data["dist_jfk"]=min(data.drop_dist_jfk, data.pk_dist_jfk)
    data["dist_ewr"]=min(data.drop_dist_ewr, data.pk_dist_ewr)
    data["dist_lgr"]=min(data.drop_dist_lgr, data.pk_dist_lgr)


# In[ ]:


for data in[train,test]:
    data["distCat_nyc"]=np.nan
    data["distCat_nyc"][(data.dist_nyc>=0)&(data.dist_nyc<1)]=0
    data["distCat_nyc"][(data.dist_nyc>=1)&(data.dist_nyc<5)]=1
    data["distCat_nyc"][(data.dist_nyc>=5)&(data.dist_nyc<11)]=2
    data["distCat_nyc"][(data.dist_nyc>=11)&(data.dist_nyc<20)]=3
    data["distCat_nyc"][(data.dist_nyc>=20)&(data.dist_nyc<100)]=4
    data["distCat_nyc"][(data.dist_nyc>=100) & (data.distance!=0)]=5
    data["distCat_nyc"][data.distance==0]=6
for data in[train,test]:
    data["distCat_jfk"]=np.nan
    data["distCat_jfk"][(data.dist_jfk>=0)&(data.dist_jfk<1)]=0
    data["distCat_jfk"][(data.dist_jfk>=1)&(data.dist_jfk<12)]=1
    data["distCat_jfk"][(data.dist_jfk>=12)&(data.dist_jfk<18)]=2
    data["distCat_jfk"][(data.dist_jfk>=18)&(data.dist_jfk<23)]=3
    data["distCat_jfk"][(data.dist_jfk>=23)&(data.dist_jfk<28)]=4
    data["distCat_jfk"][(data.dist_jfk>=28)&(data.dist_jfk<30)]=5
    data["distCat_jfk"][(data.dist_jfk>=30)& (data.distance!=0)]=6
    data["distCat_jfk"][data.distance==0]=7
for data in[train,test]:
    data["distCat_ewr"]=np.nan
    data["distCat_ewr"][(data.dist_ewr>=0)&(data.dist_ewr<0.5)]=0
    data["distCat_ewr"][(data.dist_ewr>=0.5)&(data.dist_ewr<1)]=1
    data["distCat_ewr"][(data.dist_ewr>=1)&(data.dist_ewr<17)]=2
    data["distCat_ewr"][(data.dist_ewr>=17)&(data.dist_ewr<27)]=3
    data["distCat_ewr"][(data.dist_ewr>=27)&(data.dist_ewr<45)]=4
    data["distCat_ewr"][(data.dist_ewr>=45)& (data.distance!=0)]=5
    data["distCat_ewr"][data.distance==0]=6
for data in[train,test]:
    data["distCat_lgr"]=np.nan
    data["distCat_lgr"][(data.dist_lgr>=0)&(data.dist_lgr<1)]=0
    data["distCat_lgr"][(data.dist_lgr>=1)&(data.dist_lgr<7.5)]=1
    data["distCat_lgr"][(data.dist_lgr>=7.5)&(data.dist_lgr<10)]=2
    data["distCat_lgr"][(data.dist_lgr>=10)&(data.dist_lgr<16)]=3
    data["distCat_lgr"][(data.dist_lgr>=16)&(data.dist_lgr<20)]=4
    data["distCat_lgr"][(data.dist_lgr>=20)&(data.dist_lgr<45)]=5
    data["distCat_lgr"][(data.dist_lgr>=45)& (data.distance!=0)]=6
    data["distCat_lgr"][data.distance==0]=7
    
for data in [train,test]:
    data["distCat"]=np.nan
    data.distCat[data.distance==0]=0
    data.distCat[(data.distance>0)    & (data.distance<0.5)]=1
    data.distCat[(data.distance>=0.5) & (data.distance<1)]=2
    data.distCat[(data.distance>=1)   & (data.distance<1.5)]=3
    data.distCat[(data.distance>=1.5) & (data.distance<2)]=4
    data.distCat[(data.distance>=2) & (data.distance<2.5)]=5
    data.distCat[(data.distance>=2.5)   & (data.distance<12)]=np.around(data.distance)+2
    data.distCat[(data.distance>=12)  & (data.distance<14)]=16
    data.distCat[(data.distance>=14)]=20
    data.distCat[(data.distance>=55)  & (data.distance<65)]=18

def meanEnc(matrix,cols, newName):
    newName=newName+"_avg_fare"
    group = matrix.groupby(cols).agg({'fare_amount': ['mean']})
    group.columns = [ newName ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=cols, how='left')
    matrix[newName] = matrix[newName].astype(np.float16)
    return matrix


list_mean_end=[ [["passenger_count"],"pass"],
    [["year"],"year"],
    [["month"],"month"],
    [["day"],"day"],
    [["dayOfWeek"],"dow"],
    [["hour"],"hour"],
    [["distCat"],"dist"],
               
    [["month","year"],"year_month"],
    [["passenger_count","distCat"],"pass_dist"],
    [["passenger_count","hour"],"pass_hour"],
    [["year","distCat"],"year_dist"],
    [["month","day"],"month_day"],
    [["month","dayOfWeek"],"month_dow"],
    [["month","hour"],"month_hour"],
    [["day","hour"],"day_hour"],
    [["hour","distCat"],"hour_dist"], 
    [["year","month","day","distCat"],"time_dist"],
]

data=pd.concat([train,test])
s=0;
for group in list_mean_end:
    ts=time.time()
    data=meanEnc(data,group[0],group[1])
    print(group[0], " is done in: ",time.time()-ts)
    s+=time.time()-ts;
print("mean encoding is done in ", s)

train = data.loc[:train.shape[0]-1,:]
test = data.loc[train.shape[0]:,:]


# In[ ]:





# In[ ]:





# In[ ]:


# float_cols=['pickup_longitude',
#        'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
#        'distance', 'diff_lat', 'diff_long', 'pk_dist_nyc', 'pk_dist_jfk',
#        'pk_dist_ewr', 'pk_dist_lgr', 'drop_dist_nyc', 'drop_dist_jfk',
#        'drop_dist_ewr', 'drop_dist_lgr', 'pickup_longitude_cos',
#        'pickup_longitude_sin', 'pickup_longitude_carre', 'pickup_latitude_cos',
#        'pickup_latitude_sin', 'pickup_latitude_carre', 'dropoff_longitude_cos',
#        'dropoff_longitude_sin', 'dropoff_longitude_carre',
#        'dropoff_latitude_cos', 'dropoff_latitude_sin',
#        'dropoff_latitude_carre', 'pickup_product', 'dropoff_product']
# for col in float_cols:
#     train[col] = train[col].astype(np.float16)


# In[ ]:


gc.collect()


# In[ ]:


y     =train["fare_amount"]
X     =train
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.06, random_state =123 )


# In[ ]:


import lightgbm as lgb
gc.collect()


# In[ ]:


features=[ 'day', 'dayOfWeek', 'diff_lat', 'diff_long', 'distCat', 'distCat_ewr',
       'distCat_jfk', 'distCat_lgr', 'distCat_nyc', 'dist_ewr', 'dist_jfk',
       'dist_lgr', 'dist_nyc', 'distance', 'drop_dist_ewr', 'drop_dist_jfk',
       'drop_dist_lgr', 'drop_dist_nyc', 'dropoff_latitude',
       'dropoff_latitude_carre', 'dropoff_latitude_cos',
       'dropoff_latitude_sin', 'dropoff_longitude', 'dropoff_longitude_carre',
       'dropoff_longitude_cos', 'dropoff_longitude_sin', 'dropoff_product',
       'hour',  'month', 'passenger_count',
       'pickup_latitude', 'pickup_latitude_carre',
       'pickup_latitude_cos', 'pickup_latitude_sin', 'pickup_longitude',
       'pickup_longitude_carre', 'pickup_longitude_cos',
       'pickup_longitude_sin', 'pickup_product', 'pk_dist_ewr', 'pk_dist_jfk',
       'pk_dist_lgr', 'pk_dist_nyc', 'year', 'pass_avg_fare', 'year_avg_fare',
       'month_avg_fare', 'day_avg_fare', 'dow_avg_fare', 'hour_avg_fare',
       'dist_avg_fare', 'year_month_avg_fare', 'pass_dist_avg_fare',
       'pass_hour_avg_fare', 'year_dist_avg_fare', 'month_day_avg_fare',
       'month_dow_avg_fare', 'month_hour_avg_fare', 'day_hour_avg_fare',
       'hour_dist_avg_fare', 'time_dist_avg_fare']
features=['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'distance', 'diff_lat', 'diff_long', 'pk_dist_nyc', 'pk_dist_jfk',
       'pk_dist_ewr', 'pk_dist_lgr', 'drop_dist_nyc', 'drop_dist_jfk',
       'drop_dist_ewr', 'drop_dist_lgr', 'pickup_longitude_cos',
       'pickup_longitude_sin', 'pickup_longitude_carre', 'pickup_latitude_cos',
       'pickup_latitude_sin', 'pickup_latitude_carre', 'dropoff_longitude_cos',
       'dropoff_longitude_sin', 'dropoff_longitude_carre',
       'dropoff_latitude_cos', 'dropoff_latitude_sin',
       'dropoff_latitude_carre', 'pickup_product', 'dropoff_product']


# In[ ]:


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 256,  
    'max_depth': 10,  
    'min_child_samples': 200,   
    'subsample': 0.7,  
    'subsample_freq': 1,  
    'colsample_bytree': 0.7,  
    'min_child_weight': 0,  
    'subsample_for_bin': 200000,  
    'min_split_gain': 0,  
    'reg_alpha': 0,  
    'reg_lambda': 0,  
    'nthread': -1,
    'verbose': 5,
    'n_estimators':1000,
#     'scale_pos_weight'
    }
lgb_params = {
                'objective':'regression',
                'boosting_type':'gbdt',
                'metric':'rmse',
                'n_jobs':4,
                'learning_rate':0.1,
                'num_leaves': 2**5,
                'max_depth':-1,
                'colsample_bytree': 0.7,
                'subsample_freq':1,
                'subsample':0.7,
                'n_estimators':1000,
                'max_bin':256,
                'verbose':-1,
                'seed': 0,
                'early_stopping_rounds':100, 
            } 
model = lgb.LGBMRegressor(**params)
print("Training the model...")
# X_train,X_val,y_train,y_val
model.fit(
        X_train[features], y_train,
        eval_set=[(X_train[features], y_train),(X_val[features], y_val)],
        verbose=10,
        early_stopping_rounds=500)

# dtrain = lgb.Dataset(X_train[features].values, label=y_train.values,
#                       feature_name=features
#                       )
# dvalid = lgb.Dataset(X_val[features].values, label=y_val.values,
#                       feature_name=features
#                     )

# evals_results = {}



# lgb_model = lgb.train(params, 
#                  dtrain, 
#                  valid_sets=[dtrain, dvalid], 
#                  valid_names=['train','valid'], 
#                  evals_result=evals_results, 
#                  num_boost_round=10000,
#                  early_stopping_rounds=100,
#                  verbose_eval=100)


# In[ ]:


Y_test = lgb_model.predict(test[features])


# In[ ]:


sample=pd.read_csv("../input/sample_submission.csv")


# In[ ]:


sample.fare_amount=Y_test


# In[ ]:


sample.to_csv("lightgbm_model.csv",index=False)


# In[ ]:


Y_test = lgb_model.predict(X_val[features])
sub = pd.DataFrame({
    "key": X_val.key, 
    "fare_amount_lgb": Y_test
})
sub.to_csv("val_lgb.csv",index=False)
X_val["fare_amount"].to_csv("val_target",index=False)


# In[ ]:


import pickle 
pickle.dump(lgb_model, open("lgb_taxi.pickle.dat", "wb"))


# In[ ]:




