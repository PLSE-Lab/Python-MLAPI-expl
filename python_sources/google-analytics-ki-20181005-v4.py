#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize

JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
def load_df(csv_path='../input/train.csv'):

    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'})
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    return df


# In[ ]:


train = load_df("../input/train.csv")
score = load_df("../input/test.csv")

print(train.shape)
print(score.shape)

train_1=pd.DataFrame(train.drop_duplicates(('fullVisitorId','date','visitId'), keep='first'))
print(train_1.shape)
score_1=pd.DataFrame(score.drop_duplicates(('fullVisitorId','date','visitId'), keep='first'))
print(score_1.shape)
train_2=train_1.sort_values(['fullVisitorId','date','visitId'], ascending=[True,True,True])
score_2=score_1.sort_values(['fullVisitorId','date','visitId'], ascending=[True,True,True])


# In[ ]:


#deletebased on train
del train_2['socialEngagementType']
del train_2['device.browserSize']
del train_2['device.browserVersion']
del train_2['device.flashVersion']
del train_2['device.language']
del train_2['device.mobileDeviceBranding']
del train_2['device.mobileDeviceInfo']
del train_2['device.mobileDeviceMarketingName']
del train_2['device.mobileDeviceModel']
del train_2['device.mobileInputSelector']
del train_2['device.operatingSystemVersion']
del train_2['device.screenColors']
del train_2['device.screenResolution']
del train_2['geoNetwork.cityId']
del train_2['geoNetwork.latitude']
del train_2['geoNetwork.longitude']
del train_2['geoNetwork.networkLocation']
del train_2['totals.visits']
del train_2['trafficSource.adwordsClickInfo.criteriaParameters']
del train_2['trafficSource.campaignCode']


# In[ ]:


print(train_2.shape)


# In[ ]:


#deletebased on train
del score_2['socialEngagementType']
del score_2['device.browserSize']
del score_2['device.browserVersion']
del score_2['device.flashVersion']
del score_2['device.language']
del score_2['device.mobileDeviceBranding']
del score_2['device.mobileDeviceInfo']
del score_2['device.mobileDeviceMarketingName']
del score_2['device.mobileDeviceModel']
del score_2['device.mobileInputSelector']
del score_2['device.operatingSystemVersion']
del score_2['device.screenColors']
del score_2['device.screenResolution']
del score_2['geoNetwork.cityId']
del score_2['geoNetwork.latitude']
del score_2['geoNetwork.longitude']
del score_2['geoNetwork.networkLocation']
del score_2['totals.visits']
del score_2['trafficSource.adwordsClickInfo.criteriaParameters']
#del score_2['trafficSource.campaignCode']


# In[ ]:


score_2.shape


# In[ ]:


print(train_2['date'].min())
print(train_2['date'].max())
print(score_2['date'].min())
print(score_2['date'].max())


# In[ ]:


#USING TfidfVectorizer TO EXTRACT FEATURES FROM geoNetwork_networkDomain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
Tvect=TfidfVectorizer(ngram_range=(1,2),max_features=20000)
vect=Tvect.fit(train_2['geoNetwork.networkDomain'])
train_vect=vect.transform(train_2['geoNetwork.networkDomain'])
test_vect=vect.transform(score_2['geoNetwork.networkDomain'])

#DIMENSIONALITY REDUCTION ON EXTRACTED FEATURES
svd=TruncatedSVD(n_components=10)

#CREATING DATAFRAMES AFTER FEATURE EXTRACTION AND REDUCTION
vect_cols=['cr_var_'+str(x) for x in range(1,11)]
df_train_vect=pd.DataFrame(svd.fit_transform(train_vect),columns=vect_cols)
df_test_vect=pd.DataFrame(svd.fit_transform(test_vect),columns=vect_cols)

#VIEW OF EXTRACTED AND REDUCED FEATURES
print(train_vect.shape,test_vect.shape)
display(df_train_vect.head())
display(df_test_vect.head())
print('Shape of vector dataframes:',df_train_vect.shape,df_test_vect.shape)
train_2=pd.concat([train_2,df_train_vect],axis=1)
score_2=pd.concat([score_2,df_test_vect],axis=1)


# In[ ]:


train_2_dev = train_2.ix[train_2['date']>=20161100 ]
print(train_2_dev.shape)
train_2_val = train_2.ix[train_2['date']<20161100 ]
print(train_2_val.shape)
dev_y=pd.DataFrame(train_2_dev['totals.transactionRevenue'])
print(dev_y.shape)
val_y=pd.DataFrame(train_2_val['totals.transactionRevenue'])
print(val_y.shape)
del train_2_dev['totals.transactionRevenue']
del train_2_val['totals.transactionRevenue']
dev_x=train_2_dev
print(dev_x.shape)
val_x=train_2_val
print(val_x.shape)
scr_x=score_2.ix[score_2['date']>0]
print(scr_x.shape)


# In[ ]:


dev_x['visitNumber']=dev_x['visitNumber'].astype(float)
dev_x['totals.bounces']=dev_x['totals.bounces'].astype(float)
dev_x['totals.hits']=dev_x['totals.hits'].astype(float)
dev_x['totals.newVisits']=dev_x['totals.newVisits'].astype(float)
dev_x['totals.pageviews']=dev_x['totals.pageviews'].astype(float)
dev_y['totals.transactionRevenue']=dev_y['totals.transactionRevenue'].astype(float)

val_x['visitNumber']=val_x['visitNumber'].astype(float)
val_x['totals.bounces']=val_x['totals.bounces'].astype(float)
val_x['totals.hits']=val_x['totals.hits'].astype(float)
val_x['totals.newVisits']=val_x['totals.newVisits'].astype(float)
val_x['totals.pageviews']=val_x['totals.pageviews'].astype(float)
val_y['totals.transactionRevenue']=val_y['totals.transactionRevenue'].astype(float)

scr_x['visitNumber']=scr_x['visitNumber'].astype(float)
scr_x['totals.bounces']=scr_x['totals.bounces'].astype(float)
scr_x['totals.hits']=scr_x['totals.hits'].astype(float)
scr_x['totals.newVisits']=scr_x['totals.newVisits'].astype(float)
scr_x['totals.pageviews']=scr_x['totals.pageviews'].astype(float)

dev_x['var_date'] = pd.to_datetime(dev_x['visitStartTime'], unit='s')
dev_x['cr_var_day_of_week'] = dev_x['var_date'].dt.dayofweek
dev_x['cr_var_hour'] = dev_x['var_date'].dt.hour
dev_x['cr_var_day'] = dev_x['var_date'].dt.day
dev_x['cr_var_month'] = dev_x['var_date'].dt.month

val_x['var_date'] = pd.to_datetime(val_x['visitStartTime'], unit='s')
val_x['cr_var_day_of_week'] = val_x['var_date'].dt.dayofweek
val_x['cr_var_hour'] = val_x['var_date'].dt.hour
val_x['cr_var_day'] = val_x['var_date'].dt.day
val_x['cr_var_month'] = val_x['var_date'].dt.month

scr_x['var_date'] = pd.to_datetime(scr_x['visitStartTime'], unit='s')
scr_x['cr_var_day_of_week'] = scr_x['var_date'].dt.dayofweek
scr_x['cr_var_hour'] = scr_x['var_date'].dt.hour
scr_x['cr_var_day'] = scr_x['var_date'].dt.day
scr_x['cr_var_month'] = scr_x['var_date'].dt.month

dev_y.columns=['TARGET']
dev_y['TARGET']=dev_y['TARGET'].astype(float)

val_y.columns=['TARGET']
val_y['TARGET']=val_y['TARGET'].astype(float)


# In[ ]:


dev_x.head()


# In[ ]:


#VARIABLE SPLIT - START
def var_split (var_name,src_dataset,master):
    
    
    t=master[[var_name]]
    t[var_name]=t[var_name].fillna('kw_miss_val')
    t['counter_var']=1
    
    t1=t.groupby([var_name]).agg(['count'])
    t1=pd.DataFrame(t1)

    
    t1['kw_key']=t1.index
    t2=pd.DataFrame(t1.values)
    t2 = t2.ix[t2[0]>25000]
    var_val=t2[1].tolist()

    c=0
    for i in var_val:
        for j in src_dataset:
            j['cr_var'+var_name+'_'+str(c)]=np.where(j[var_name]==var_val[c],1,0)
        c=c+1
#VARIABLE SPLIT - STOP
dev_x_a = dev_x.ix[dev_x['visitNumber']>0]
val_x_a = val_x.ix[val_x['visitNumber']>0]
scr_x_a = scr_x.ix[scr_x['visitNumber']>0]
impacted_ds=[dev_x_a,val_x_a,scr_x_a]
var_split('channelGrouping',impacted_ds,dev_x) 
var_split('device.browser',impacted_ds,dev_x) 
var_split('device.deviceCategory',impacted_ds,dev_x) 
var_split('device.isMobile',impacted_ds,dev_x) 
var_split('device.operatingSystem',impacted_ds,dev_x) 
var_split('geoNetwork.city',impacted_ds,dev_x) 
var_split('geoNetwork.continent',impacted_ds,dev_x) 
var_split('geoNetwork.country',impacted_ds,dev_x) 
var_split('geoNetwork.metro',impacted_ds,dev_x) 
var_split('geoNetwork.networkDomain',impacted_ds,dev_x) 
var_split('geoNetwork.region',impacted_ds,dev_x) 
var_split('geoNetwork.subContinent',impacted_ds,dev_x) 
var_split('trafficSource.adwordsClickInfo.adNetworkType',impacted_ds,dev_x) 
var_split('trafficSource.adwordsClickInfo.gclId',impacted_ds,dev_x) 
var_split('trafficSource.adwordsClickInfo.isVideoAd',impacted_ds,dev_x) 
var_split('trafficSource.adwordsClickInfo.page',impacted_ds,dev_x) 
var_split('trafficSource.adwordsClickInfo.slot',impacted_ds,dev_x) 
var_split('trafficSource.campaign',impacted_ds,dev_x) 
var_split('trafficSource.isTrueDirect',impacted_ds,dev_x) 
var_split('trafficSource.keyword',impacted_ds,dev_x) 
var_split('trafficSource.medium',impacted_ds,dev_x) 
var_split('trafficSource.referralPath',impacted_ds,dev_x) 
var_split('trafficSource.source',impacted_ds,dev_x) 

print(dev_x.shape,dev_x_a.shape)
print(val_x.shape,val_x_a.shape)
print(scr_x.shape,scr_x_a.shape)


# In[ ]:


col_name=(dev_x_a.columns.tolist())
reslst = []
for i in col_name:
    if i[0] == 'c' and i[1] == 'r' and i[2] == '_' and i[3] == 'v' and i[4] == 'a' and i[5] == 'r':
        reslst.append(i)
added=['visitNumber','totals.bounces','totals.hits','totals.newVisits','totals.pageviews']
reslst=reslst+added
dev_x_1=dev_x_a[reslst]
val_x_1=val_x_a[reslst]
scr_x_1=scr_x_a[reslst]
print(dev_x_1.shape)
print(val_x_1.shape)
print(scr_x_1.shape)


# In[ ]:


dev_x_1.head()
#print(dev_x_1['cr_varchannelGrouping_0'].sum())


# In[ ]:


impacted_var=['visitNumber','totals.bounces','totals.hits','totals.newVisits','totals.pageviews']
impacted_dat=[dev_x_1,val_x_1,scr_x_1]
fill_anyval_with_anyval(impacted_var,impacted_dat,np.nan,0)

impacted_var=['TARGET']
impacted_dat=[dev_y,val_y]
fill_anyval_with_anyval(impacted_var,impacted_dat,np.nan,0)


# In[ ]:


dev_y['TARGET']=dev_y['TARGET'].apply(lambda x: np.log1p(x))
val_y['TARGET']=val_y['TARGET'].apply(lambda x: np.log1p(x))


# In[ ]:


dev_y['TARGET'].sum()+val_y['TARGET'].sum()


# In[ ]:


t=train[['totals.transactionRevenue']]
t['totals.transactionRevenue']=t['totals.transactionRevenue'].astype(float)
t['totals.transactionRevenue'].sum()


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

dev_x_2=dev_x_1.values
dev_x_2=sc.fit_transform(dev_x_2)
dev_x_2=pd.DataFrame(dev_x_2)

val_x_2=val_x_1.values
val_x_2=sc.transform(val_x_2)
val_x_2=pd.DataFrame(val_x_2)

scr_x_2=scr_x_1.values
scr_x_2=sc.transform(scr_x_2)
scr_x_2=pd.DataFrame(scr_x_2)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers  import SGD
from keras.optimizers  import Adam
from keras.optimizers  import RMSprop
np.random.seed(999)

ann1=Sequential()
ann1.add(Dense(units=5,input_dim=len(dev_x_2.columns),activation='relu',kernel_initializer='uniform'))
#ann1.add(Dense(units=3,activation='relu',kernel_initializer='uniform'))
#ann1.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))
ann1.add(Dense(units=1,activation='relu',kernel_initializer='uniform'))
ann1.compile(optimizer='adam',loss='mean_squared_error')
sgd = SGD(lr=0.5)
ann1.compile(optimizer=sgd,loss='mean_squared_logarithmic_error')
#adam = Adam(lr=0.5, beta_1=0.9, beta_2=0.999, decay=0.0)
#ann1.compile(optimizer=adam,loss='mean_squared_error')
#rmsprop = RMSprop(lr=0.5)
#ann1.compile(optimizer=rmsprop,loss='mean_squared_error')

#ann1.fit(data_select_dev_xx,data_select_dev_yy,epochs=1,batch_size=1)

# Fit the model
history = ann1.fit(dev_x_2,dev_y,epochs=10,batch_size=100000,validation_data=(val_x_2,val_y))
# summarize history for loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[ ]:


import xgboost as xgb
from xgboost import XGBRegressor
xgb_a = XGBRegressor(max_depth=3, n_estimators=100)
xgb_a.fit(dev_x_2,dev_y)


# In[ ]:


from catboost import CatBoostRegressor
cb = CatBoostRegressor(max_depth=5, n_estimators=100)
cb.fit(dev_x_2, dev_y)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 200, random_state = 0,min_samples_leaf=.01)
rf.fit(dev_x_2, dev_y)


# In[ ]:


from lightgbm.sklearn import LGBMRegressor
#model=LGBMRegressor(boosting_type='gbdt',num_leaves=31,max_depth=-1,learning_rate=0.01,n_estimators=1000,max_bin=255,subsample_for_bin=50000,objective=None,min_split_gain=0,min_child_weight=3,min_child_samples=10,subsample=1,subsample_freq=1,colsample_bytree=1,reg_alpha=0.1,reg_lambda=0,seed=17,silent=False,nthread=-1,n_jobs=-1)
model=LGBMRegressor(boosting_type='gbdt',num_leaves=31,max_depth=10,learning_rate=0.01,n_estimators=525,max_bin=255,subsample_for_bin=50000,objective=None,min_split_gain=0,min_child_weight=3,min_child_samples=10,subsample=1,subsample_freq=1,colsample_bytree=1,reg_alpha=0.1,reg_lambda=0,seed=17,silent=False,nthread=-1,n_jobs=-1)
model.fit(dev_x_2, dev_y)


# In[ ]:


def scoring(model,model_col,src_dataset):
    c=0
    for i in model:    
        t = pd.DataFrame(i.predict(src_dataset))     
        t=pd.DataFrame(t[model_col[c]])
        t.columns=[c]
        if c==0:
            t1=t
        else:
            t1 = pd.merge(t1,t,left_index=True, right_index=True)
        c=c+1
        
    t1=pd.DataFrame(t1.fillna(0))   
    t1['final_pred'] = t1.mean(axis=1)    
    t1=t1[['final_pred']]
    return t1

model_list=[model]
model_col_list=[0]

dev_y_pred=scoring(model_list,model_col_list,dev_x_2)
val_y_pred=scoring(model_list,model_col_list,val_x_2)
scr_y_pred=scoring(model_list,model_col_list,scr_x_2)


# In[ ]:


dev_x_2.columns


# In[ ]:


t_lt0 = dev_y_pred.ix[dev_y_pred['final_pred']<0 ]
t_lt0['final_pred']=0
t_ge0 = dev_y_pred.ix[dev_y_pred['final_pred']>=0 ]
t_ge0=t_ge0.append(t_lt0)
dev_y_pred=t_ge0

t_lt0 = val_y_pred.ix[val_y_pred['final_pred']<0 ]
t_lt0['final_pred']=0
t_ge0 = val_y_pred.ix[val_y_pred['final_pred']>=0 ]
t_ge0=t_ge0.append(t_lt0)
val_y_pred=t_ge0


# In[ ]:


def rmsle(predicted, real):
    sum=0.0
    for x in range(len(predicted)):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

print(rmsle(dev_y_pred.values, dev_y.values))
print(rmsle(val_y_pred.values, val_y.values))


# In[ ]:


#FUNCTION FOR CALCULATING RSME
from sklearn.metrics import mean_squared_error
def rsme(y,pred):
    return(mean_squared_error(y,pred)**0.5)

#acc=rsme(dev_y_pred.values,dev_y.values)
#print(acc)

print(rsme(dev_y_pred.values, dev_y.values))
print(rsme(val_y_pred.values, val_y.values))


# In[ ]:


display(scr_x_a.head())
display(scr_y_pred.head())
display(scr_x_a.shape)
display(scr_y_pred.shape)
final = pd.merge(scr_x_a,scr_y_pred,left_index=True, right_index=True)
final_1=final[['fullVisitorId','final_pred']]
final_1.columns=['fullVisitorId','PredictedLogRevenue']
display(final_1.head())
display(final_1.shape)


# In[ ]:


#GROUPING PREDICTED DATA ON fullVisitorId
final_2 = final_1.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
final_2.columns = ["fullVisitorId", "PredictedLogRevenue"]
display(final_2.shape)
display(final_2.head())


# In[ ]:


#READING SUMISSION FILE
submission=pd.read_csv('../input/sample_submission.csv')

#CREATING JOIN BETWEEN PREDICTED DATA WITH SUBMISSION FILE
submission=submission.join(final_2.set_index('fullVisitorId'),on='fullVisitorId',lsuffix='_sub')
submission.drop('PredictedLogRevenue_sub',axis=1,inplace=True)

#HANDLING NaN IN CASE OF MISSING fullVisitorId
submission.fillna(0,inplace=True)

display(submission.head())
display(submission.shape)


# In[ ]:


#SUBMITING FILE
submission.to_csv('LGBM_submission.csv',index=False)


# In[ ]:



total = train_2.isnull().sum().sort_values(ascending = False)
percent = (train_2.isnull().sum() / train_2.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_application_train_data)


# In[ ]:


train_2['totals.transactionRevenue']=train_2['totals.transactionRevenue'].replace([np.nan],9999999999)


# In[ ]:


train_2['totals.transactionRevenue'].astype(float)


# In[ ]:


nest=train_1.sort_values(['fullVisitorId','date','visitId'], ascending=[True,True,True])


# In[ ]:


nest1=nest.sort_values(['totals.transactionRevenue'], ascending=[True])
nest1=nest1['totals.transactionRevenue']


# In[ ]:


nest2=pd.DataFrame(nest1.values[0:100]).astype(float)
nest2.columns=['a']
#nest2.head()
print(nest2['a'].sum())


# In[ ]:





# In[ ]:


print(train_2['totals.transactionRevenue'])


# In[ ]:


x['totals.transactionRevenue']=train_1['totals.transactionRevenue']


# In[ ]:


print(x['totals.transactionRevenue'].sum())


# In[ ]:


s=pd.DataFrame(s.values)


# In[ ]:


print(type(train_2['totals.transactionRevenue'][15]))
print(type(train_2['totals.transactionRevenue'][16]))
print(type(train_2['totals.transactionRevenue'][17]))
print(type(train_2['totals.transactionRevenue'][18]))
print(type(train_2['totals.transactionRevenue'][19]))
print(type(train_2['totals.transactionRevenue'][20]))


# In[ ]:


s.head()


# In[ ]:


train_2.head()


# In[ ]:


df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= [2, 'A', 4], columns=['A', 'B', 'C'])
print(type(df))
print(df.head())
print(df['A'])
print(df['A'].sum())


# In[ ]:


x=pd.DataFrame(n1['tot'][18:20])
x.columns=['col']
print(x['col'].sum())


# In[ ]:


n1.head(100)


# In[ ]:


type(x)


# In[ ]:





# In[ ]:


train_2_0 = train_2.ix[train_2['totals.transactionRevenue']==9999999999]
train_2_1 = train_2.ix[train_2['totals.transactionRevenue']!=9999999999]
print(train_2_0.shape)
print(train_2_1.shape)
from sklearn.model_selection import train_test_split
train_2_0_dev,train_2_0_val=train_test_split(train_2_0,train_size=.6,random_state=0)
train_2_1_dev,train_2_1_val=train_test_split(train_2_1,train_size=.6,random_state=0)
print(train_2_0_dev.shape)
print(train_2_0_val.shape)
print(train_2_1_dev.shape)
print(train_2_1_val.shape)

train_2_dev=train_2_1_dev 
train_2_dev=train_2_dev.append(train_2_0_dev)
train_2_dev_x=train_2_dev
train_2_dev_y=pd.DataFrame(train_2_dev[['totals.transactionRevenue']])
train_2_dev_y.columns=['TARGET']
del train_2_dev_x['totals.transactionRevenue']
train_2_dev_y['TARGET']=train_2_dev_y['TARGET'].replace([9999999999],0)

train_2_val=train_2_1_val 
train_2_val=train_2_val.append(train_2_0_val)
train_2_val_x=train_2_val
train_2_val_y=pd.DataFrame(train_2_val[['totals.transactionRevenue']])
train_2_val_y.columns=['TARGET']
del train_2_val_x['totals.transactionRevenue']
train_2_val_y['TARGET']=train_2_val_y['TARGET'].replace([9999999999],0)

print(train_2_dev_x.shape)
print(train_2_dev_y.shape)

print(train_2_val_x.shape)
print(train_2_val_y.shape)


# In[ ]:


#fill any value with any value-start
def fill_anyval_with_anyval(var_list,slave_data,to_be_filled,filler):
    for i in var_list:
        for j in slave_data:
            j[i]=j[i].replace({to_be_filled:filler})
#fill any value with any value-stop

#VARIABLE SPLIT - START
def var_split (var_name,var_val,src_dataset):
    c=0
    for i in var_val:
        for j in src_dataset:
            j[var_name+'_'+str(c)]=np.where(j[var_name]==var_val[c],1,0)
        c=c+1
#VARIABLE SPLIT - STOP


# In[ ]:


impacted_ds=[train_2]
fill_anyval_with_anyval(train_2.columns,impacted_ds,np.nan,0)
fill_anyval_with_anyval(data_select_dev_x.columns,impacted_ds,np.inf,0)
fill_anyval_with_anyval(data_select_dev_x.columns,impacted_ds,-np.inf,0)

impacted_ds=[score_2]
fill_anyval_with_anyval(score_2.columns,impacted_ds,np.nan,0)
fill_anyval_with_anyval(data_select_dev_x.columns,impacted_ds,np.inf,0)
fill_anyval_with_anyval(data_select_dev_x.columns,impacted_ds,-np.inf,0)


# In[ ]:


print(train_2_val_y['TARGET'].value_counts(dropna=False))


# In[ ]:




