#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime
import lightgbm as lgb
import numpy as np


# In[ ]:


#add lat long!!!


# In[ ]:


lat_long = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")


# In[ ]:


lat_long = lat_long[['Province/State','Country/Region','Lat','Long']].drop_duplicates() 


# In[ ]:


lat_long.columns = ['Province_State', 'Country_Region', 'Lat', 'Long']


# In[ ]:


train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
sub = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")


# In[ ]:


print(len(train))
print(len(test))


# In[ ]:


train = pd.merge(train,lat_long, on=['Province_State','Country_Region'], how='left')
test = pd.merge(test,lat_long, on=['Province_State','Country_Region'],how='left')


# In[ ]:


print(len(train))
print(len(test))


# In[ ]:


train.head()


# In[ ]:


lb_date_pp = train['Date'].max()


# In[ ]:


lb_date_pp


# In[ ]:


lb_date = test['Date'].min()


# In[ ]:


lb_date


# In[ ]:


#train.loc[(train['Date']=='2020-03-24')&(train['Country_Region']=='France')&(train['Province_State']=='France'),'ConfirmedCases'] = 22654
#train.loc[(train['Date']=='2020-03-24')&(train['Country_Region']=='France')&(train['Province_State']=='France'),'Fatalities'] = 1000


# In[ ]:


#train.loc[(train['Date']=='2020-03-24')&(train['Country_Region']=='France')&(train['Province_State']=='France')]#,'ConfirmedCases']


# In[ ]:


train = train[train['Date']<lb_date].append(test[test['Date']>=lb_date])


# In[ ]:


import datetime 
train['Date'] = pd.to_datetime(train['Date'], format='%Y-%m-%d')


# In[ ]:


train['day_dist'] = train['Date']-train['Date'].min()


# In[ ]:


train['day_dist'] = train['day_dist'].dt.days


# In[ ]:


print(train['Date'].max())
#print(val['Date'].max())
print(test['Date'].min())
print(test['Date'].max())
#print(test['Date'].max()-test['Date'].min())


# In[ ]:


cat_cols = train.dtypes[train.dtypes=='object'].keys()
cat_cols


# In[ ]:


for cat_col in cat_cols:
    train[cat_col].fillna('no_value', inplace = True)


# In[ ]:


train['place'] = train['Province_State']+'_'+train['Country_Region']
#vcheck = train[(train['Date']>='2020-03-12')]


# In[ ]:


from sklearn import preprocessing


# In[ ]:


cat_cols = train.dtypes[train.dtypes=='object'].keys()
cat_cols


# In[ ]:


for cat_col in ['place']:
    #train[cat_col].fillna('no_value', inplace = True) #train[cat_col].value_counts().idxmax()
    le = preprocessing.LabelEncoder()
    le.fit(train[cat_col])
    train[cat_col]=le.transform(train[cat_col])


# In[ ]:


train.keys()


# In[ ]:


drop_cols = ['Id','ForecastId', 'ConfirmedCases','Date', 'Fatalities','day_dist', 'Province_State', 'Country_Region'] #,'day_dist','shift_22_ft','shift_23_ft','shift_24_ft','shift_25_ft','shift_26_ft']


# In[ ]:


#val = train[(train['Id']).isnull()==True]
#train = train[(train['Id']).isnull()==False]
val = train[(train['Date']>lb_date)&(train['Id'].isnull()==False)]
#test = train[(train['Date']>='2020-03-12')&(train['Id'].isnull()==True)]
#train = train[(train['Date']<'2020-03-22')&(train['Id'].isnull()==False)]


# In[ ]:


y_ft = train["Fatalities"]
y_val_ft = val["Fatalities"]



y_cc = train["ConfirmedCases"]
y_val_cc = val["ConfirmedCases"]

#train.drop(drop_cols, axis=1, inplace=True)
#test.drop(drop_cols, axis=1, inplace=True)
#val.drop(drop_cols, axis=1, inplace=True)


# In[ ]:


def rmsle (y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))


# In[ ]:


def mape (y_true, y_pred):
    return np.mean(np.abs(y_pred -y_true)*100/(y_true+1))


# In[ ]:


import numpy as np


# In[ ]:



params = {
    "objective": "regression",
    "boosting": 'gbdt', #"gbdt",
    "num_leaves": 1280,
    "learning_rate": 0.05,
    "feature_fraction": 0.9, # 0.9,
    "reg_lambda": 2,
    "metric": "rmse",
    'min_data_in_leaf':20
}


# In[ ]:


dates = test['Date'].unique()


# In[ ]:


dates_pp = dates[dates>lb_date_pp]
dates = dates[dates>=lb_date]


# In[ ]:


len(dates)


# In[ ]:


k=5
i = 0
fold_n = 0
for date in dates:

    fold_n = fold_n +1 
    i = i+1
    if i==1:
        nrounds = 200
    else:
        nrounds = 100
    print(i)
    print(nrounds)
    train['shift_1_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i)
    train['shift_2_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+1)
    train['shift_3_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+2)
    train['shift_4_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+3)
    train['shift_5_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+4)
    train['shift_7_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+6)
    train['shift_14_cc'] = train.groupby(['place'])['ConfirmedCases'].shift((i+6)*2)
    train['shift_56_cc'] = train.groupby(['place'])['ConfirmedCases'].shift((i+6)*8)
    #train = train.merge(train.groupby(['Country_Region'])['ConfirmedCases'].apply(pd.Series.autocorr, lag=1), on=['Country_Region'], suffixes = ['', '_autoc_1']) 
    #train = train.merge(train.groupby(['Country_Region'])['ConfirmedCases'].apply(pd.Series.autocorr, lag=10), on=['Country_Region'], suffixes = ['', '_autoc_10']) 
    #train = train.merge(train.groupby(['Country_Region'])['ConfirmedCases'].apply(pd.Series.autocorr, lag=2), on=['Country_Region'], suffixes = ['', '_autoc_2']) 
    #train = train.merge(train.groupby(['Country_Region'])['ConfirmedCases'].apply(pd.Series.autocorr, lag=5), on=['Country_Region'], suffixes = ['', '_autoc_5']) 
    print(nrounds)
    #train['shift_22_ft'] = train.groupby(['place'])['Fatalities'].shift(i)
    #train['shift_23_ft'] = train.groupby(['place'])['Fatalities'].shift(i+1)
    #train['shift_24_ft'] = train.groupby(['place'])['Fatalities'].shift(i+2)
    #train['shift_25_ft'] = train.groupby(['place'])['Fatalities'].shift(i+3)
    #train['shift_26_ft'] = train.groupby(['place'])['Fatalities'].shift(i+4)
    
    #train['shift_22_ft'] = train['shift_22_ft']*100/train['shift_1_cc']
    #train['shift_23_ft'] = train['shift_23_ft']*100/train['shift_2_cc']
    #train['shift_24_ft'] = train['shift_24_ft']*100/train['shift_3_cc']
    #train['shift_25_ft'] = train['shift_25_ft']*100/train['shift_4_cc']
    #train['shift_26_ft'] = train['shift_26_ft']*100/train['shift_5_cc']

    #train['diff_1_7_cc'] = (train['shift_1_cc']-train['shift_7_cc'])#/train['shift_1_cc']
    #train['diff_1_14_cc'] = (train['shift_1_cc']-train['shift_14_cc'])#/train['shift_1_cc']
    #train['diff_1_56_cc'] = (train['shift_1_cc']-train['shift_56_cc'])#/train['shift_1_cc']
    train['diff_23_24_cc'] = (train['shift_3_cc']-train['shift_2_cc'])#/train['shift_2_cc']
    train['diff_24_25_cc'] = (train['shift_5_cc']-train['shift_4_cc'])#/train['shift_4_cc']
    train['diff_22_24_cc'] = (train['shift_4_cc']-train['shift_1_cc'])#/train['shift_1_cc']
    train['diff_22_25_cc'] = (train['shift_5_cc']-train['shift_1_cc'])#/train['shift_1_cc']
    print(nrounds)
    #train['moving_avg_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].var().reset_index(0,drop=True)
    #train['moving_min_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].min().reset_index(0,drop=True)
    #train['moving_min_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].var().reset_index(0,drop=True)
    #train['moving_min_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].min().reset_index(0,drop=True)
    #train['moving_min_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].var().reset_index(0,drop=True)
    #train['moving_min_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].min().reset_index(0,drop=True)
    #train['moving_min_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].var().reset_index(0,drop=True)
    #train['moving_min_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].min().reset_index(0,drop=True)
    #train['moving_min_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].max().reset_index(0,drop=True)
    print("aoooo")
    #train['moving_avg_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].var().reset_index(0,drop=True)
    #train['moving_min_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].min().reset_index(0,drop=True)
    #train['moving_min_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].max().reset_index(0,drop=True)
    print("aoooo")
        
    val2 = train[train['Date']==date]
    train2 = train[(train['Date']<date)]
    y_cc = train2["ConfirmedCases"]
    #y_val_cc = val2["ConfirmedCases"]
    
    train2.drop(drop_cols, axis=1, inplace=True)
    val2.drop(drop_cols, axis=1, inplace=True)
    
    #np.log1p(y)
    #feature_importances = pd.DataFrame()
    #feature_importances['feature'] = train.keys()
    
    #score = 0       
    dtrain = lgb.Dataset(train2, label=y_cc)
    dvalid = lgb.Dataset(val2, label=y_val_cc)

    model = lgb.train(params, dtrain, nrounds, 
                            #valid_sets = [dtrain, dvalid],
                            categorical_feature = ['place'], #'Province/State', 'Country/Region'
                            verbose_eval=False)#, early_stopping_rounds=50)

    y_pred = model.predict(val2,num_iteration=nrounds)  #model.best_iteration
    #y_pred = np.expm1( y_pred)
    #vcheck.loc[vcheck['Date']==date,'cc_predict'] = y_pred
    test.loc[test['Date']==date,'ConfirmedCases'] = y_pred
    train.loc[train['Date']==date,'ConfirmedCases'] = y_pred
    #y_oof[valid_index] = y_pred

    #rmsle_score = rmsle(y_val_cc, y_pred)
    #mape_score = mape(y_val_cc, y_pred)
    #score += rmsle_score
    #print (f'fold: {date}, rmsle: {rmsle_score:.5f}' )
    #print (f'fold: {date}, mape: {mape_score:.5f}' )


# In[ ]:


test[test['Country_Region']=='Italy']


# In[ ]:


y_pred.mean()


# In[ ]:


i = 0
fold_n = 0
for date in dates:

    fold_n = fold_n +1 
    i = i+1
    if i==1:
        nrounds = 200
    else:
        nrounds = 100
    print(i)
    print(nrounds)
    i = i+1
    train['shift_1_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i)
    train['shift_2_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+1)
    train['shift_3_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+2)
    train['shift_4_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+3)
    train['shift_5_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+4)
    train['shift_7_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+6)
    train['shift_14_cc'] = train.groupby(['place'])['ConfirmedCases'].shift((i+6)*2)
    train['shift_56_cc'] = train.groupby(['place'])['ConfirmedCases'].shift((i+6)*8)
    
    train['shift_1_ft'] = train.groupby(['place'])['Fatalities'].shift(i)
    train['shift_2_ft'] = train.groupby(['place'])['Fatalities'].shift(i+1)
    train['shift_3_ft'] = train.groupby(['place'])['Fatalities'].shift(i+2)
    train['shift_4_ft'] = train.groupby(['place'])['Fatalities'].shift(i+3)
    train['shift_5_ft'] = train.groupby(['place'])['Fatalities'].shift(i+4)
    train['shift_7_ft'] = train.groupby(['place'])['Fatalities'].shift(i+6)
    train['shift_14_ft'] = train.groupby(['place'])['Fatalities'].shift((i+6)*2)
    train['shift_56_ft'] = train.groupby(['place'])['Fatalities'].shift((i+6)*8)
    

    

    train['diff_1_7_cc'] = (train['shift_1_cc']-train['shift_7_cc'])#/train['shift_1_cc']
    train['diff_1_14_cc'] = (train['shift_1_cc']-train['shift_14_cc'])#/train['shift_1_cc']
    train['diff_1_56_cc'] = (train['shift_1_cc']-train['shift_56_cc'])#/train['shift_1_cc']
    #train['diff_23_24_cc'] = (train['shift_3_cc']-train['shift_2_cc'])#/train['shift_2_cc']
    #train['diff_24_25_cc'] = (train['shift_5_cc']-train['shift_4_cc'])#/train['shift_4_cc']
    #train['diff_22_24_cc'] = (train['shift_4_cc']-train['shift_1_cc'])#/train['shift_1_cc']
    #train['diff_22_25_cc'] = (train['shift_5_cc']-train['shift_1_cc'])#/train['shift_1_cc']
    
    train['diff_1_7_ft'] = (train['shift_1_ft']-train['shift_7_ft'])#/train['shift_1_ft']
    train['diff_1_14_ft'] = (train['shift_1_ft']-train['shift_14_ft'])#/train['shift_1_ft']
    train['diff_1_56_ft'] = (train['shift_1_ft']-train['shift_56_ft'])#/train['shift_1_ft']
    train['diff_23_24_ft'] = (train['shift_3_ft']-train['shift_2_ft'])#/train['shift_2_ft']
    train['diff_24_25_ft'] = (train['shift_5_ft']-train['shift_4_ft'])#/train['shift_4_ft']
    train['diff_22_24_ft'] = (train['shift_4_ft']-train['shift_1_ft'])#/train['shift_1_ft']
    train['diff_22_25_ft'] = (train['shift_5_ft']-train['shift_1_ft'])#/train['shift_1_ft']
    
    #train['moving_avg_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].var().reset_index(0,drop=True)
    #train['moving_min_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].min().reset_index(0,drop=True)
    #train['moving_min_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].var().reset_index(0,drop=True)
    #train['moving_min_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].min().reset_index(0,drop=True)
    #train['moving_min_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].var().reset_index(0,drop=True)
    #train['moving_min_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].min().reset_index(0,drop=True)
    #train['moving_min_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].var().reset_index(0,drop=True)
    #train['moving_min_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].min().reset_index(0,drop=True)
    #train['moving_min_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].var().reset_index(0,drop=True)
    #train['moving_min_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].min().reset_index(0,drop=True)
    #train['moving_min_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].max().reset_index(0,drop=True)
    
    #train['moving_avg_22_ft'] = train.groupby(['place']).rolling(k)['shift_1_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_22_ft'] = train.groupby(['place']).rolling(k)['shift_1_ft'].var().reset_index(0,drop=True)
    #train['moving_min_22_ft'] = train.groupby(['place']).rolling(k)['shift_1_ft'].min().reset_index(0,drop=True)
    #train['moving_min_22_ft'] = train.groupby(['place']).rolling(k)['shift_1_ft'].max().reset_index(0,drop=True)

    #train['moving_avg_23_ft'] = train.groupby(['place']).rolling(k)['shift_2_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_23_ft'] = train.groupby(['place']).rolling(k)['shift_2_ft'].var().reset_index(0,drop=True)
    #train['moving_min_23_ft'] = train.groupby(['place']).rolling(k)['shift_2_ft'].min().reset_index(0,drop=True)
    #train['moving_min_23_ft'] = train.groupby(['place']).rolling(k)['shift_2_ft'].max().reset_index(0,drop=True)

    #train['moving_avg_24_ft'] = train.groupby(['place']).rolling(k)['shift_3_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_24_ft'] = train.groupby(['place']).rolling(k)['shift_3_ft'].var().reset_index(0,drop=True)
    #train['moving_min_24_ft'] = train.groupby(['place']).rolling(k)['shift_3_ft'].min().reset_index(0,drop=True)
    #train['moving_min_24_ft'] = train.groupby(['place']).rolling(k)['shift_3_ft'].max().reset_index(0,drop=True)
    
    #train['moving_avg_7_ft'] = train.groupby(['place']).rolling(k)['shift_7_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_7_ft'] = train.groupby(['place']).rolling(k)['shift_7_ft'].var().reset_index(0,drop=True)
    #train['moving_min_7_ft'] = train.groupby(['place']).rolling(k)['shift_7_ft'].min().reset_index(0,drop=True)
    #train['moving_min_7_ft'] = train.groupby(['place']).rolling(k)['shift_7_ft'].max().reset_index(0,drop=True)

    #train['moving_avg_25_ft'] = train.groupby(['place']).rolling(k)['shift_14_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_25_ft'] = train.groupby(['place']).rolling(k)['shift_14_ft'].var().reset_index(0,drop=True)
    #train['moving_min_25_ft'] = train.groupby(['place']).rolling(k)['shift_14_ft'].min().reset_index(0,drop=True)
    #train['moving_min_25_ft'] = train.groupby(['place']).rolling(k)['shift_14_ft'].max().reset_index(0,drop=True)

    #train['moving_avg_26_ft'] = train.groupby(['place']).rolling(k)['shift_56_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_26_ft'] = train.groupby(['place']).rolling(k)['shift_56_ft'].var().reset_index(0,drop=True)
    #train['moving_min_26_ft'] = train.groupby(['place']).rolling(k)['shift_56_ft'].min().reset_index(0,drop=True)
    #train['moving_min_26_ft'] = train.groupby(['place']).rolling(k)['shift_56_ft'].max().reset_index(0,drop=True)
 
        
    val2 = train[train['Date']==date]
    train2 = train[(train['Date']<date)]
    y_ft = train2["Fatalities"]
    #y_val_cc = val2["ConfirmedCases"]
    
    train2.drop(drop_cols, axis=1, inplace=True)
    val2.drop(drop_cols, axis=1, inplace=True)
    
    #np.log1p(y)
    #feature_importances = pd.DataFrame()
    #feature_importances['feature'] = train.keys()
    
    #score = 0       
    dtrain = lgb.Dataset(train2, label=y_ft)
    dvalid = lgb.Dataset(val2, label=y_val_ft)

    model = lgb.train(params, dtrain, nrounds, 
                            #valid_sets = [dtrain, dvalid],
                            categorical_feature = ['place'], #'Province/State', 'Country/Region'
                            verbose_eval=False)#, early_stopping_rounds=50)

    y_pred = model.predict(val2,num_iteration=nrounds)  #model.best_iteration
    #y_pred = np.expm1( y_pred)
    #vcheck.loc[vcheck['Date']==date,'cc_predict'] = y_pred
    test.loc[test['Date']==date,'Fatalities'] = y_pred
    train.loc[train['Date']==date,'Fatalities'] = y_pred
    #y_oof[valid_index] = y_pred

    #rmsle_score = rmsle(y_val_cc, y_pred)
    #mape_score = mape(y_val_cc, y_pred)
    #score += rmsle_score
    #print (f'fold: {date}, rmsle: {rmsle_score:.5f}' )
    #print (f'fold: {date}, mape: {mape_score:.5f}' )


# In[ ]:


train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
#test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
sub = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")

train = pd.merge(train,lat_long, on=['Province_State','Country_Region'], how='left')
#test = pd.merge(test,lat_long, on=['Province_State','Country_Region'],how='left')


# In[ ]:


lb_date_pp = train['Date'].max()


# In[ ]:


train = train.append(test[test['Date']>lb_date_pp])


# In[ ]:


import datetime 
train['Date'] = pd.to_datetime(train['Date'], format='%Y-%m-%d')
train['day_dist'] = train['Date']-train['Date'].min()
train['day_dist'] = train['day_dist'].dt.days


# In[ ]:


cat_cols = train.dtypes[train.dtypes=='object'].keys()
cat_cols

for cat_col in cat_cols:
    train[cat_col].fillna('no_value', inplace = True)
    
train['place'] = train['Province_State']+'_'+train['Country_Region']
#vcheck = train[(train['Date']>='2020-03-12')]


# In[ ]:


test.head()


# In[ ]:


from sklearn import preprocessing
cat_cols = train.dtypes[train.dtypes=='object'].keys()
cat_cols
for cat_col in ['place']:
    #train[cat_col].fillna('no_value', inplace = True) #train[cat_col].value_counts().idxmax()
    le = preprocessing.LabelEncoder()
    le.fit(train[cat_col])
    train[cat_col]=le.transform(train[cat_col])


# In[ ]:


dates = test['Date'].unique()
dates_pp = dates[dates>lb_date_pp]
dates = dates[dates>=lb_date]


# In[ ]:


dates_pp


# In[ ]:


k=5
i = 0
fold_n = 0
for date in dates_pp:

    fold_n = fold_n +1 
    i = i+1
    if i==1:
        nrounds = 200
    else:
        nrounds = 100
    print(i)
    print(nrounds)
    train['shift_1_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i)
    train['shift_2_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+1)
    train['shift_3_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+2)
    train['shift_4_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+3)
    train['shift_5_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+4)
    train['shift_7_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+6)
    train['shift_14_cc'] = train.groupby(['place'])['ConfirmedCases'].shift((i+6)*2)
    train['shift_56_cc'] = train.groupby(['place'])['ConfirmedCases'].shift((i+6)*8)
    #train = train.merge(train.groupby(['Country_Region'])['ConfirmedCases'].apply(pd.Series.autocorr, lag=1), on=['Country_Region'], suffixes = ['', '_autoc_1']) 
    #train = train.merge(train.groupby(['Country_Region'])['ConfirmedCases'].apply(pd.Series.autocorr, lag=10), on=['Country_Region'], suffixes = ['', '_autoc_10']) 
    #train = train.merge(train.groupby(['Country_Region'])['ConfirmedCases'].apply(pd.Series.autocorr, lag=2), on=['Country_Region'], suffixes = ['', '_autoc_2']) 
    #train = train.merge(train.groupby(['Country_Region'])['ConfirmedCases'].apply(pd.Series.autocorr, lag=5), on=['Country_Region'], suffixes = ['', '_autoc_5']) 
    print(nrounds)
    #train['shift_22_ft'] = train.groupby(['place'])['Fatalities'].shift(i)
    #train['shift_23_ft'] = train.groupby(['place'])['Fatalities'].shift(i+1)
    #train['shift_24_ft'] = train.groupby(['place'])['Fatalities'].shift(i+2)
    #train['shift_25_ft'] = train.groupby(['place'])['Fatalities'].shift(i+3)
    #train['shift_26_ft'] = train.groupby(['place'])['Fatalities'].shift(i+4)
    
    #train['shift_22_ft'] = train['shift_22_ft']*100/train['shift_1_cc']
    #train['shift_23_ft'] = train['shift_23_ft']*100/train['shift_2_cc']
    #train['shift_24_ft'] = train['shift_24_ft']*100/train['shift_3_cc']
    #train['shift_25_ft'] = train['shift_25_ft']*100/train['shift_4_cc']
    #train['shift_26_ft'] = train['shift_26_ft']*100/train['shift_5_cc']

    #train['diff_1_7_cc'] = (train['shift_1_cc']-train['shift_7_cc'])#/train['shift_1_cc']
    #train['diff_1_14_cc'] = (train['shift_1_cc']-train['shift_14_cc'])#/train['shift_1_cc']
    #train['diff_1_56_cc'] = (train['shift_1_cc']-train['shift_56_cc'])#/train['shift_1_cc']
    train['diff_23_24_cc'] = (train['shift_3_cc']-train['shift_2_cc'])#/train['shift_2_cc']
    train['diff_24_25_cc'] = (train['shift_5_cc']-train['shift_4_cc'])#/train['shift_4_cc']
    train['diff_22_24_cc'] = (train['shift_4_cc']-train['shift_1_cc'])#/train['shift_1_cc']
    train['diff_22_25_cc'] = (train['shift_5_cc']-train['shift_1_cc'])#/train['shift_1_cc']
    print(nrounds)
    #train['moving_avg_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].var().reset_index(0,drop=True)
    #train['moving_min_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].min().reset_index(0,drop=True)
    #train['moving_min_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].var().reset_index(0,drop=True)
    #train['moving_min_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].min().reset_index(0,drop=True)
    #train['moving_min_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].var().reset_index(0,drop=True)
    #train['moving_min_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].min().reset_index(0,drop=True)
    #train['moving_min_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].var().reset_index(0,drop=True)
    #train['moving_min_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].min().reset_index(0,drop=True)
    #train['moving_min_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].max().reset_index(0,drop=True)
    print("aoooo")
    #train['moving_avg_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].var().reset_index(0,drop=True)
    #train['moving_min_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].min().reset_index(0,drop=True)
    #train['moving_min_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].max().reset_index(0,drop=True)
    print("aoooo")
        
    val2 = train[train['Date']==date]
    train2 = train[(train['Date']<date)]
    y_cc = train2["ConfirmedCases"]
    #y_val_cc = val2["ConfirmedCases"]
    
    train2.drop(drop_cols, axis=1, inplace=True)
    val2.drop(drop_cols, axis=1, inplace=True)
    
    #np.log1p(y)
    #feature_importances = pd.DataFrame()
    #feature_importances['feature'] = train.keys()
    
    #score = 0       
    dtrain = lgb.Dataset(train2, label=y_cc)
    dvalid = lgb.Dataset(val2, label=y_val_cc)

    model = lgb.train(params, dtrain, nrounds, 
                            #valid_sets = [dtrain, dvalid],
                            categorical_feature = ['place'], #'Province/State', 'Country/Region'
                            verbose_eval=False)#, early_stopping_rounds=50)

    y_pred = model.predict(val2,num_iteration=nrounds)  #model.best_iteration
    #y_pred = np.expm1( y_pred)
    #vcheck.loc[vcheck['Date']==date,'cc_predict'] = y_pred
    test.loc[test['Date']==date,'ConfirmedCases'] = y_pred
    train.loc[train['Date']==date,'ConfirmedCases'] = y_pred
    #y_oof[valid_index] = y_pred

    #rmsle_score = rmsle(y_val_cc, y_pred)
    #mape_score = mape(y_val_cc, y_pred)
    #score += rmsle_score
    #print (f'fold: {date}, rmsle: {rmsle_score:.5f}' )
    #print (f'fold: {date}, mape: {mape_score:.5f}' )


# In[ ]:


test[test['Country_Region']=='Italy']


# In[ ]:


i = 0
fold_n = 0
for date in dates:

    fold_n = fold_n +1 
    i = i+1
    if i==1:
        nrounds = 200
    else:
        nrounds = 100
    print(i)
    print(nrounds)
    i = i+1
    train['shift_1_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i)
    train['shift_2_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+1)
    train['shift_3_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+2)
    train['shift_4_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+3)
    train['shift_5_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+4)
    train['shift_7_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+6)
    train['shift_14_cc'] = train.groupby(['place'])['ConfirmedCases'].shift((i+6)*2)
    train['shift_56_cc'] = train.groupby(['place'])['ConfirmedCases'].shift((i+6)*8)
    
    train['shift_1_ft'] = train.groupby(['place'])['Fatalities'].shift(i)
    train['shift_2_ft'] = train.groupby(['place'])['Fatalities'].shift(i+1)
    train['shift_3_ft'] = train.groupby(['place'])['Fatalities'].shift(i+2)
    train['shift_4_ft'] = train.groupby(['place'])['Fatalities'].shift(i+3)
    train['shift_5_ft'] = train.groupby(['place'])['Fatalities'].shift(i+4)
    train['shift_7_ft'] = train.groupby(['place'])['Fatalities'].shift(i+6)
    train['shift_14_ft'] = train.groupby(['place'])['Fatalities'].shift((i+6)*2)
    train['shift_56_ft'] = train.groupby(['place'])['Fatalities'].shift((i+6)*8)
    

    

    train['diff_1_7_cc'] = (train['shift_1_cc']-train['shift_7_cc'])#/train['shift_1_cc']
    train['diff_1_14_cc'] = (train['shift_1_cc']-train['shift_14_cc'])#/train['shift_1_cc']
    train['diff_1_56_cc'] = (train['shift_1_cc']-train['shift_56_cc'])#/train['shift_1_cc']
    #train['diff_23_24_cc'] = (train['shift_3_cc']-train['shift_2_cc'])#/train['shift_2_cc']
    #train['diff_24_25_cc'] = (train['shift_5_cc']-train['shift_4_cc'])#/train['shift_4_cc']
    #train['diff_22_24_cc'] = (train['shift_4_cc']-train['shift_1_cc'])#/train['shift_1_cc']
    #train['diff_22_25_cc'] = (train['shift_5_cc']-train['shift_1_cc'])#/train['shift_1_cc']
    
    train['diff_1_7_ft'] = (train['shift_1_ft']-train['shift_7_ft'])#/train['shift_1_ft']
    train['diff_1_14_ft'] = (train['shift_1_ft']-train['shift_14_ft'])#/train['shift_1_ft']
    train['diff_1_56_ft'] = (train['shift_1_ft']-train['shift_56_ft'])#/train['shift_1_ft']
    train['diff_23_24_ft'] = (train['shift_3_ft']-train['shift_2_ft'])#/train['shift_2_ft']
    train['diff_24_25_ft'] = (train['shift_5_ft']-train['shift_4_ft'])#/train['shift_4_ft']
    train['diff_22_24_ft'] = (train['shift_4_ft']-train['shift_1_ft'])#/train['shift_1_ft']
    train['diff_22_25_ft'] = (train['shift_5_ft']-train['shift_1_ft'])#/train['shift_1_ft']
    
    #train['moving_avg_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].var().reset_index(0,drop=True)
    #train['moving_min_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].min().reset_index(0,drop=True)
    #train['moving_min_22_cc'] = train.groupby(['place']).rolling(k)['shift_1_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].var().reset_index(0,drop=True)
    #train['moving_min_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].min().reset_index(0,drop=True)
    #train['moving_min_23_cc'] = train.groupby(['place']).rolling(k)['shift_2_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].var().reset_index(0,drop=True)
    #train['moving_min_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].min().reset_index(0,drop=True)
    #train['moving_min_24_cc'] = train.groupby(['place']).rolling(k)['shift_3_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].var().reset_index(0,drop=True)
    #train['moving_min_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].min().reset_index(0,drop=True)
    #train['moving_min_25_cc'] = train.groupby(['place']).rolling(k)['shift_14_cc'].max().reset_index(0,drop=True)

    #train['moving_avg_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].mean().reset_index(0,drop=True)
    #train['moving_var_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].var().reset_index(0,drop=True)
    #train['moving_min_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].min().reset_index(0,drop=True)
    #train['moving_min_26_cc'] = train.groupby(['place']).rolling(k)['shift_56_cc'].max().reset_index(0,drop=True)
    
    #train['moving_avg_22_ft'] = train.groupby(['place']).rolling(k)['shift_1_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_22_ft'] = train.groupby(['place']).rolling(k)['shift_1_ft'].var().reset_index(0,drop=True)
    #train['moving_min_22_ft'] = train.groupby(['place']).rolling(k)['shift_1_ft'].min().reset_index(0,drop=True)
    #train['moving_min_22_ft'] = train.groupby(['place']).rolling(k)['shift_1_ft'].max().reset_index(0,drop=True)

    #train['moving_avg_23_ft'] = train.groupby(['place']).rolling(k)['shift_2_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_23_ft'] = train.groupby(['place']).rolling(k)['shift_2_ft'].var().reset_index(0,drop=True)
    #train['moving_min_23_ft'] = train.groupby(['place']).rolling(k)['shift_2_ft'].min().reset_index(0,drop=True)
    #train['moving_min_23_ft'] = train.groupby(['place']).rolling(k)['shift_2_ft'].max().reset_index(0,drop=True)

    #train['moving_avg_24_ft'] = train.groupby(['place']).rolling(k)['shift_3_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_24_ft'] = train.groupby(['place']).rolling(k)['shift_3_ft'].var().reset_index(0,drop=True)
    #train['moving_min_24_ft'] = train.groupby(['place']).rolling(k)['shift_3_ft'].min().reset_index(0,drop=True)
    #train['moving_min_24_ft'] = train.groupby(['place']).rolling(k)['shift_3_ft'].max().reset_index(0,drop=True)
    
    #train['moving_avg_7_ft'] = train.groupby(['place']).rolling(k)['shift_7_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_7_ft'] = train.groupby(['place']).rolling(k)['shift_7_ft'].var().reset_index(0,drop=True)
    #train['moving_min_7_ft'] = train.groupby(['place']).rolling(k)['shift_7_ft'].min().reset_index(0,drop=True)
    #train['moving_min_7_ft'] = train.groupby(['place']).rolling(k)['shift_7_ft'].max().reset_index(0,drop=True)

    #train['moving_avg_25_ft'] = train.groupby(['place']).rolling(k)['shift_14_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_25_ft'] = train.groupby(['place']).rolling(k)['shift_14_ft'].var().reset_index(0,drop=True)
    #train['moving_min_25_ft'] = train.groupby(['place']).rolling(k)['shift_14_ft'].min().reset_index(0,drop=True)
    #train['moving_min_25_ft'] = train.groupby(['place']).rolling(k)['shift_14_ft'].max().reset_index(0,drop=True)

    #train['moving_avg_26_ft'] = train.groupby(['place']).rolling(k)['shift_56_ft'].mean().reset_index(0,drop=True)
    #train['moving_var_26_ft'] = train.groupby(['place']).rolling(k)['shift_56_ft'].var().reset_index(0,drop=True)
    #train['moving_min_26_ft'] = train.groupby(['place']).rolling(k)['shift_56_ft'].min().reset_index(0,drop=True)
    #train['moving_min_26_ft'] = train.groupby(['place']).rolling(k)['shift_56_ft'].max().reset_index(0,drop=True)
 
        
    val2 = train[train['Date']==date]
    train2 = train[(train['Date']<date)]
    y_ft = train2["Fatalities"]
    #y_val_cc = val2["ConfirmedCases"]
    
    train2.drop(drop_cols, axis=1, inplace=True)
    val2.drop(drop_cols, axis=1, inplace=True)
    
    #np.log1p(y)
    #feature_importances = pd.DataFrame()
    #feature_importances['feature'] = train.keys()
    
    #score = 0       
    dtrain = lgb.Dataset(train2, label=y_ft)
    dvalid = lgb.Dataset(val2, label=y_val_ft)

    model = lgb.train(params, dtrain, nrounds, 
                            #valid_sets = [dtrain, dvalid],
                            categorical_feature = ['place'], #'Province/State', 'Country/Region'
                            verbose_eval=False)#, early_stopping_rounds=50)

    y_pred = model.predict(val2,num_iteration=nrounds)  #model.best_iteration
    #y_pred = np.expm1( y_pred)
    #vcheck.loc[vcheck['Date']==date,'cc_predict'] = y_pred
    test.loc[test['Date']==date,'Fatalities'] = y_pred
    train.loc[train['Date']==date,'Fatalities'] = y_pred
    #y_oof[valid_index] = y_pred

    #rmsle_score = rmsle(y_val_cc, y_pred)
    #mape_score = mape(y_val_cc, y_pred)
    #score += rmsle_score
    #print (f'fold: {date}, rmsle: {rmsle_score:.5f}' )
    #print (f'fold: {date}, mape: {mape_score:.5f}' )


# In[ ]:


test[test['Country_Region']=='France']


# In[ ]:


print(len(test))


# In[ ]:


train_sub = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")


# In[ ]:


#train_sub.loc[(train_sub['Date']=='2020-03-24')&(train_sub['Country_Region']=='France')&(train_sub['Province_State']=='France'),'ConfirmedCases'] = 22654
#train_sub.loc[(train_sub['Date']=='2020-03-24')&(train_sub['Country_Region']=='France')&(train_sub['Province_State']=='France'),'Fatalities'] = 1000


# In[ ]:


test = pd.merge(test,train_sub[['Province_State','Country_Region','Date','ConfirmedCases','Fatalities']], on=['Province_State','Country_Region','Date'], how='left')


# In[ ]:


print(len(test))


# In[ ]:


test.head()


# In[ ]:


test.loc[test['ConfirmedCases_x'].isnull()==True]


# In[ ]:


test.loc[test['ConfirmedCases_x'].isnull()==True, 'ConfirmedCases_x'] = test.loc[test['ConfirmedCases_x'].isnull()==True, 'ConfirmedCases_y']


# In[ ]:


test.head()


# In[ ]:


test.loc[test['Fatalities_x'].isnull()==True, 'Fatalities_x'] = test.loc[test['Fatalities_x'].isnull()==True, 'Fatalities_y']


# In[ ]:


dates


# In[ ]:


last_amount_italy = test.loc[(test['Country_Region']=='Italy')&(test['Date']==lb_date_pp),'ConfirmedCases_x']
last_fat_italy = test.loc[(test['Country_Region']=='Italy')&(test['Date']==lb_date_pp),'Fatalities_x']


# In[ ]:


last_amount_ny = test.loc[(test['Country_Region']=='US')&(test['Province_State']=='New York')&(test['Date']==lb_date_pp),'ConfirmedCases_x']
last_fat_ny = test.loc[(test['Country_Region']=='US')&(test['Province_State']=='New York')&(test['Date']==lb_date_pp),'Fatalities_x']


# In[ ]:


last_amount_ny


# In[ ]:


last_amount_spain = test.loc[(test['Country_Region']=='Spain')&(test['Date']==lb_date_pp),'ConfirmedCases_x']
last_fat_spain = test.loc[(test['Country_Region']=='Spain')&(test['Date']==lb_date_pp),'Fatalities_x']


# In[ ]:


last_amount_germany = test.loc[(test['Country_Region']=='Germany')&(test['Date']==lb_date_pp),'ConfirmedCases_x']
last_fat_germany = test.loc[(test['Country_Region']=='Germany')&(test['Date']==lb_date_pp),'Fatalities_x']


# In[ ]:


print(last_amount_germany)
print(last_fat_germany)


# In[ ]:


i = 0
k = 35


# In[ ]:


i = 0
k = 35
#dates_pp = dates[dates>=lb_date_pp]
for date in dates_pp:
    k = k-1
    i = i + 1
    test.loc[(test['Country_Region']=='Italy')&(test['Date']==date),'ConfirmedCases_x'] =  last_amount_italy.values[0]+i*(5000-(80*i))
    test.loc[(test['Country_Region']=='Italy')&(test['Date']==date),'Fatalities_x'] =  last_fat_italy.values[0]+i*(800-(10*i))


# In[ ]:


i = 0
k = 35
#dates_pp = dates[dates>=lb_date_pp]
for date in dates_pp:
    k = k-1
    i = i + 1
    test.loc[(test['Country_Region']=='Spain')&(test['Date']==date),'ConfirmedCases_x'] =  last_amount_spain.values[0]+i*(5000-(80*i))
    test.loc[(test['Country_Region']=='Spain')&(test['Date']==date),'Fatalities_x'] =  last_fat_spain.values[0]+i*(800-(10*i))


# In[ ]:


i = 0
k = 35
#dates_pp = dates[dates>=lb_date_pp]
for date in dates_pp:
    k = k-1
    i = i + 1
    test.loc[(test['Country_Region']=='Germany')&(test['Date']==date),'ConfirmedCases_x'] =  last_amount_germany.values[0]+i*(5000-(80*i))
    test.loc[(test['Country_Region']=='Germany')&(test['Date']==date),'Fatalities_x'] =  last_fat_germany.values[0]+i*(800-(10*i))


# In[ ]:


i = 0
k = 35
for date in dates_pp:
    k = k-1
    i = i + 1
    test.loc[(test['Country_Region']=='US')&(test['Province_State']=='New York')&(test['Date']==date),'ConfirmedCases_x'] = 75833.0+i*(5000-(80*i))
    test.loc[(test['Country_Region']=='US')&(test['Province_State']=='New York')&(test['Date']==date),'Fatalities_x'] =  1550.0+i*(800-(10*i))


# In[ ]:


test.loc[(test['Country_Region']=='US')&(test['Province_State']=='New York')]


# In[ ]:


test.loc[(test['Country_Region']=='Italy')] #&(test['Date']==date),'ConfirmedCases_x' 


# In[ ]:


sub = test[['ForecastId', 'ConfirmedCases_x','Fatalities_x']]


# In[ ]:


sub.columns = ['ForecastId', 'ConfirmedCases', 'Fatalities']


# In[ ]:


sub.head()


# In[ ]:


sub.loc[sub['ConfirmedCases']<0, 'ConfirmedCases'] = 0


# In[ ]:


sub.loc[sub['Fatalities']<0, 'Fatalities'] = 0


# In[ ]:


sub['Fatalities'].describe()


# In[ ]:


sub.to_csv('submission.csv',index=False)

