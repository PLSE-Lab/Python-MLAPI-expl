#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime
import lightgbm as lgb


# In[ ]:


train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")
sub = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")


# In[ ]:


#test['cc_predict'] = 0


# In[ ]:


train.loc[(train['Date']=='2020-03-24')&(train['Country/Region']=='France')&(train['Province/State']=='France'),'ConfirmedCases'] = 22654
train.loc[(train['Date']=='2020-03-24')&(train['Country/Region']=='France')&(train['Province/State']=='France'),'Fatalities'] = 1000


# In[ ]:


train[(train['Date']=='2020-03-24')&(train['Country/Region']=='France')&(train['Province/State']=='France')]


# In[ ]:


train = train.append(test[test['Date']>'2020-03-24'])


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


train['place'] = train['Province/State']+'_'+train['Country/Region']
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


drop_cols = ['Id','ForecastId', 'ConfirmedCases','Date', 'Fatalities','day_dist', 'Province/State', 'Country/Region'] #,'day_dist','shift_22_ft','shift_23_ft','shift_24_ft','shift_25_ft','shift_26_ft']


# In[ ]:


#val = train[(train['Id']).isnull()==True]
#train = train[(train['Id']).isnull()==False]
val = train[(train['Date']>='2020-03-12')&(train['Id'].isnull()==False)]
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


dates = dates[dates>'2020-03-24']


# In[ ]:


len(dates)


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
    train['shift_1_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i)
    train['shift_2_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+1)
    train['shift_3_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+2)
    train['shift_4_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+3)
    train['shift_5_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+4)
        
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


test[test['Country/Region']=='Italy']


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
    
    train['shift_1_cc'] = train.groupby(['place'])['Fatalities'].shift(i)
    train['shift_2_cc'] = train.groupby(['place'])['Fatalities'].shift(i+1)
    train['shift_3_cc'] = train.groupby(['place'])['Fatalities'].shift(i+2)
    train['shift_4_cc'] = train.groupby(['place'])['Fatalities'].shift(i+3)
    train['shift_5_cc'] = train.groupby(['place'])['Fatalities'].shift(i+4)
        
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


test[test['Country/Region']=='Italy']


# In[ ]:


print(len(test))


# In[ ]:


train_sub = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")


# In[ ]:


train_sub.loc[(train_sub['Date']=='2020-03-24')&(train_sub['Country/Region']=='France')&(train_sub['Province/State']=='France'),'ConfirmedCases'] = 22654
train_sub.loc[(train_sub['Date']=='2020-03-24')&(train_sub['Country/Region']=='France')&(train_sub['Province/State']=='France'),'Fatalities'] = 1000


# In[ ]:


test = pd.merge(test,train_sub[['Province/State','Country/Region','Lat','Long','Date','ConfirmedCases','Fatalities']], on=['Province/State','Country/Region','Lat','Long','Date'], how='left')


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


last_amount = test.loc[(test['Country/Region']=='Italy')&(test['Date']=='2020-03-24'),'ConfirmedCases_x']


# In[ ]:


last_fat = test.loc[(test['Country/Region']=='Italy')&(test['Date']=='2020-03-24'),'Fatalities_x']


# In[ ]:


last_fat.values[0]


# In[ ]:


dates


# In[ ]:


len(dates)


# In[ ]:


30/29


# In[ ]:


i = 0
k = 35


# In[ ]:



for date in dates:
    k = k-1
    i = i + 1
    test.loc[(test['Country/Region']=='Italy')&(test['Date']==date),'ConfirmedCases_x'] =  last_amount.values[0]+i*(5000-(100*i))
    test.loc[(test['Country/Region']=='Italy')&(test['Date']==date),'Fatalities_x'] =  last_fat.values[0]+i*(800-(10*i))


# In[ ]:


test.loc[(test['Country/Region']=='Italy')] #&(test['Date']==date),'ConfirmedCases_x' 


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

