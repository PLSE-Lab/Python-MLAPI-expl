#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# Using Olivier's wonderful script as a base line I decided to evolve models using GP. 
# Note that I made it a logistic regression problem by scaling outputs to between zero and one before evolving the models.  (I don't like minimizing to zero if the score is less than zero!)
# 

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

# I don't like SettingWithCopyWarnings ...
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Get the extracted data

# In[ ]:


train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
train.shape, test.shape


# ### Define folding strategy

# In[ ]:


def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids


# ### Get session target

# In[ ]:


y_reg = train['totals.transactionRevenue'].fillna(0)
del train['totals.transactionRevenue']

if 'totals.transactionRevenue' in test.columns:
    del test['totals.transactionRevenue']


# ### Add date features
# 
# Only add the one I think can ganeralize

# In[ ]:


train.columns


# In[ ]:


for df in [train, test]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_hours'] = df['date'].dt.hour
    df['sess_date_dom'] = df['date'].dt.day


# ### Create features list

# In[ ]:


excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]


# ### Factorize categoricals

# In[ ]:


for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])


# ### Predict revenues at session level

# In[ ]:


folds = get_folds(df=train, n_splits=5)

train_features = [_f for _f in train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(train.shape[0])
sub_reg_preds = np.zeros(test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5


# ### Display feature importances

# In[ ]:


import warnings
warnings.simplefilter('ignore', FutureWarning)

importances['gain_log'] = np.log1p(importances['gain'])
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 12))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))


# ### Create user level predictions

# In[ ]:


train['predictions'] = np.expm1(oof_reg_preds)
test['predictions'] = sub_reg_preds


# In[ ]:


# Aggregate data at User level
trn_data = train[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Create a list of predictions for each Visitor\ntrn_pred_list = train[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\\\n    .apply(lambda df: list(df.predictions))\\\n    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})")


# In[ ]:


# Create a DataFrame with VisitorId as index
# trn_pred_list contains dict 
# so creating a dataframe from it will expand dict values into columns
trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
trn_feats = trn_all_predictions.columns
trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
full_data = pd.concat([trn_data, trn_all_predictions], axis=1)
del trn_data, trn_all_predictions
gc.collect()
full_data.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "sub_pred_list = test[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\\\n    .apply(lambda df: list(df.predictions))\\\n    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})")


# In[ ]:


sub_data = test[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()
sub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=sub_data.index)
for f in trn_feats:
    if f not in sub_all_predictions.columns:
        sub_all_predictions[f] = np.nan
sub_all_predictions['t_mean'] = np.log1p(sub_all_predictions[trn_feats].mean(axis=1))
sub_all_predictions['t_median'] = np.log1p(sub_all_predictions[trn_feats].median(axis=1))
sub_all_predictions['t_sum_log'] = np.log1p(sub_all_predictions[trn_feats]).sum(axis=1)
sub_all_predictions['t_sum_act'] = np.log1p(sub_all_predictions[trn_feats].fillna(0).sum(axis=1))
sub_all_predictions['t_nb_sess'] = sub_all_predictions[trn_feats].isnull().sum(axis=1)
sub_full_data = pd.concat([sub_data, sub_all_predictions], axis=1)
del sub_data, sub_all_predictions
gc.collect()
sub_full_data.shape


# ### Create target at Visitor level

# In[ ]:


train['target'] = y_reg
trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()


# In[ ]:


df = pd.concat([full_data,sub_full_data],sort=False)
trainsize = full_data.shape[0]
del full_data, sub_full_data
gc.collect()
df = df.reset_index(drop=False)
for c in df.columns[1:]:
    if((df[c].min()>=0)&(df[c].max()>=10)):
        df[c] = np.log1p(df[c])
    elif((df[c].min()<0)&((df[c].max()-df[c].min())>=10)):
        df.loc[df[c]!=0,c] = np.sign(df.loc[df[c]!=0,c])*np.log(np.abs(df.loc[df[c]!=0,c]))
from sklearn.preprocessing import StandardScaler
for c in df.columns[1:]:
    ss = StandardScaler()
    df.loc[~np.isfinite(df[c]),c] = np.nan
    df.loc[~df[c].isnull(),c] = ss.fit_transform(df.loc[~df[c].isnull(),c].values.reshape(-1,1))
df.fillna(-99999,inplace=True)
gp_trn_users = df[:trainsize].copy().set_index('fullVisitorId')
maxval = np.log1p(trn_user_target['target'].values).max()
gp_trn_users['target'] = np.log1p(trn_user_target['target'].values)
gp_trn_users.target /= maxval
gp_sub_users = df[trainsize:].copy().set_index('fullVisitorId')
newcols =  [x.replace('.','_') for x in gp_trn_users.columns]
gp_trn_users.columns = newcols
newcols =  [x.replace('.','_') for x in gp_sub_users.columns]
gp_sub_users.columns = newcols
del df
gc.collect()


# In[ ]:


maxval 


# In[ ]:


def Output(p):
    return 1.0/(1+np.exp(-p))

def GP1(data):
    v = pd.DataFrame()
    v["i0"] = 0.100000*np.tanh((((((np.tanh((((data["pred_4"]) + (data["pred_6"]))/2.0)) )) * ((12.07693195343017578)))) - ((10.0))))  
    v["i1"] = 0.100000*np.tanh(((np.where((np.tanh((((((((((10.0)) - (data["pred_5"]))) + (2.0))) - (data["pred_0"]))) * (3.0))) )<0, 3.0, ((-3.0) - ((10.0))) )) * ((((9.80441665649414062)) * 2.0))))  
    v["i2"] = 0.100000*np.tanh((((((np.tanh(((((-1.0) / 2.0)) + (data["pred_8"]))) )) + (np.maximum((((np.tanh(((((-1.0) / 2.0)) + (data["pred_8"]))) ))), ((-1.0)))))) * ((6.0))))  
    v["i3"] = 0.100000*np.tanh((((np.tanh(((-2.0) + (data["pred_3"]))) )) * ((((-1.0*(((((3.0) + (((((-3.0) - ((9.0)))) * 2.0)))/2.0))))) * 2.0))))  
    v["i4"] = 0.100000*np.tanh((((np.tanh(((data["pred_3"]) - ((2.44455146789550781)))) )) * ((10.55293273925781250))))  
    v["i5"] = 0.100000*np.tanh(((np.where((np.tanh((((data["pred_2"]) + (-3.0))/2.0)) )>0, -3.0, (9.0) )) * (((((((-3.0) - ((9.0)))) * 2.0)) * 2.0))))  
    v["i6"] = 0.100000*np.tanh((((((((np.tanh(((((((2.48090791702270508)) + (data["pred_4"]))/2.0)) + (-2.0))) )) * ((10.93176364898681641)))) * 2.0)) * 2.0))  
    v["i7"] = 0.100000*np.tanh((((((((np.tanh(((data["pred_1"]) - (2.0))) )) * 2.0)) / 2.0)) * ((9.0))))  
    v["i8"] = 0.100000*np.tanh((((((np.where(((-3.0) - ((9.55589389801025391))) < -99998, -1.0, 1.0 )) > ((np.tanh(((data["pred_0"]) - (-1.0))) )))*1.)) * (((-3.0) * 2.0))))  
    v["i9"] = 0.100000*np.tanh((((12.06137466430664062)) * ((np.tanh((((data["t_mean"]) + (((-3.0) + (-2.0))))/2.0)) ))))  
    v["i10"] = 0.100000*np.tanh(((-3.0) * ((((10.0)) * ((np.tanh((((8.0)) - (data["t_mean"]))) ))))))  
    v["i11"] = 0.100000*np.tanh((((((-1.0*(((((5.0)) * 2.0))))) / 2.0)) + (((np.where((np.tanh(((data["pred_2"]) - (3.0))) )<0, (np.tanh(((data["pred_2"]) - (3.0))) ), (6.70732259750366211) )) * 2.0))))  
    v["i12"] = 0.100000*np.tanh((((np.tanh(((((((-1.0*((3.0)))) * 2.0)) + (np.maximum(((-3.0)), ((data["t_mean"])))))/2.0)) )) * ((13.89055156707763672))))  
    v["i13"] = 0.100000*np.tanh((((7.0)) * (np.where(np.minimum((((5.90891504287719727))), (((((np.tanh(np.minimum(((((data["t_mean"]) - ((4.69426012039184570))))), (((((((-3.0) + (data["t_mean"]))) + (np.maximum(((data["pred_5"])), ((-3.0)))))/2.0))))) )) * ((np.tanh(np.minimum(((((data["t_mean"]) - ((4.69426012039184570))))), (((((((-3.0) + (data["t_mean"]))) + (np.maximum(((data["pred_5"])), ((-3.0)))))/2.0))))) ))))))>0, (np.tanh(np.minimum(((((data["t_mean"]) - ((4.69426012039184570))))), (((((((-3.0) + (data["t_mean"]))) + (np.maximum(((data["pred_5"])), ((-3.0)))))/2.0))))) ), (np.tanh(np.minimum(((((data["t_mean"]) - ((4.69426012039184570))))), (((((((-3.0) + (data["t_mean"]))) + (np.maximum(((data["pred_5"])), ((-3.0)))))/2.0))))) ) ))))  
    v["i14"] = 0.100000*np.tanh((((11.24391651153564453)) * ((((np.tanh(((((((data["t_sum_log"]) + ((((-2.0) + ((((-1.0) + (-2.0))/2.0)))/2.0)))/2.0)) + (((np.tanh((-2.0))) * 2.0)))/2.0)) )) * 2.0))))  
    v["i15"] = 0.100000*np.tanh(((np.where((7.0)>0, (np.tanh(((((((((data["t_sum_log"]) / 2.0)) < (data["pred_0"]))*1.)) + (((data["t_sum_log"]) - ((6.0)))))/2.0)) ), (((((np.tanh(((((((((data["t_sum_log"]) / 2.0)) < (data["pred_0"]))*1.)) + (((data["t_sum_log"]) - ((6.0)))))/2.0)) )) * ((((9.0)) * ((np.tanh(((((((((data["t_sum_log"]) / 2.0)) < (data["pred_0"]))*1.)) + (((data["t_sum_log"]) - ((6.0)))))/2.0)) )))))) * 2.0) )) * ((6.0))))  
    v["i16"] = 0.100000*np.tanh((((((np.tanh(np.tanh((((1.0) - ((-1.0*((0.0)))))))) )) - ((9.39388847351074219)))) * ((((((((np.tanh((-1.0*((np.maximum(((-3.0)), ((((data["t_sum_log"]) / 2.0)))))))) )) - (((-2.0) / 2.0)))) * 2.0)) * 2.0))))  
    v["i17"] = 0.100000*np.tanh(((((np.where((np.tanh((((-3.0) + (((data["t_sum_act"]) / 2.0)))/2.0)) )>0, (np.tanh((((-3.0) + (((data["t_sum_act"]) / 2.0)))/2.0)) ), (((((((np.tanh((((-3.0) + (((data["t_sum_act"]) / 2.0)))/2.0)) )) + ((np.tanh((((-3.0) + (((data["t_sum_act"]) / 2.0)))/2.0)) )))) * 2.0)) + ((1.0))) )) * 2.0)) * 2.0))  
    v["i18"] = 0.100000*np.tanh((((((((12.23419475555419922)) * ((np.tanh(((np.maximum(((((-3.0) + (data["pred_3"])))), ((-3.0)))) + (((data["t_sum_log"]) + (-3.0))))) )))) * 2.0)) - ((((((9.04281616210937500)) + ((9.0)))) * ((np.tanh(((np.maximum(((((-3.0) + (data["pred_3"])))), ((-3.0)))) + (((data["t_sum_log"]) + (-3.0))))) ))))))  
    v["i19"] = 0.100000*np.tanh((((((np.tanh(((data["t_sum_log"]) - (((np.maximum((((7.0))), ((data["t_sum_log"])))) / 2.0)))) )) * 2.0)) * 2.0))  
    v["i20"] = 0.100000*np.tanh((((np.tanh(np.minimum(((data["pred_5"])), ((data["totals_pageviews"])))) )) * ((13.97667598724365234))))  
    v["i21"] = 0.100000*np.tanh(((((np.tanh(np.maximum(((data["pred_4"])), ((((data["t_sum_log"]) + (-3.0)))))) )) + ((((((6.0)) * ((np.tanh(np.maximum(((data["pred_4"])), ((((data["t_sum_log"]) + (-3.0)))))) )))) - ((6.0)))))/2.0))  
    v["i22"] = 0.100000*np.tanh(((np.where((np.tanh((((data["pred_1"]) + (-1.0))/2.0)) )<0, 2.0, (np.tanh((((data["pred_1"]) < (-1.0))*1.)) ) )) * (((-3.0) * ((14.82431602478027344))))))  
    v["i23"] = 0.100000*np.tanh((((13.51132678985595703)) * ((np.tanh((((-3.0) + (data["t_sum_log"]))/2.0)) ))))  
    v["i24"] = 0.100000*np.tanh((-1.0*((((np.minimum(((3.0)), (((np.tanh((((data["t_sum_log"]) < ((9.0)))*1.)) ))))) * ((11.02354240417480469)))))))  
    v["i25"] = 0.100000*np.tanh((((((14.53642463684082031)) * (((np.where((14.62061119079589844)<0, (np.tanh(np.minimum(((data["t_sum_log"])), ((3.0)))) ), (np.tanh(np.minimum(((data["t_sum_log"])), ((3.0)))) ) )) - ((np.tanh((((3.0) < (data["pred_2"]))*1.)) )))))) - ((14.53642177581787109))))  
    v["i26"] = 0.100000*np.tanh((((np.tanh(((data["t_sum_log"]) + (-3.0))) )) * ((10.0))))  
    v["i27"] = 0.100000*np.tanh((((10.0)) * ((np.tanh((((-1.0) + ((((data["t_median"]) > ((5.0)))*1.)))/2.0)) ))))  
    v["i28"] = 0.100000*np.tanh(((np.where((np.tanh(((-2.0) + (data["t_sum_log"]))) )>0, 0.0, (-1.0*(((7.0)))) )) * 2.0))  
    v["i29"] = 0.100000*np.tanh(((((np.where(-3.0 < -99998, (np.tanh(((data["t_sum_act"]) + ((-1.0*(((((((data["t_median"]) + (-2.0))/2.0)) * (np.maximum(((data["pred_2"])), ((-2.0))))))))))) ), ((((-2.0) + ((((np.tanh(((data["t_sum_act"]) + ((-1.0*(((((((data["t_median"]) + (-2.0))/2.0)) * (np.maximum(((data["pred_2"])), ((-2.0))))))))))) )) * 2.0)))) * 2.0) )) * 2.0)) * 2.0))  
    v["i30"] = 0.100000*np.tanh(((np.where((8.40624523162841797)>0, (((np.tanh(np.minimum(((data["t_mean"])), ((((((((data["t_sum_act"]) + (-2.0))) * (data["t_sum_act"]))) + (-1.0)))))) )) * ((8.40624523162841797))), (((np.tanh(np.minimum(((data["t_mean"])), ((((((((data["t_sum_act"]) + (-2.0))) * (data["t_sum_act"]))) + (-1.0)))))) )) * ((8.40624141693115234))) )) - ((8.31408786773681641))))  
    v["i31"] = 0.100000*np.tanh((((((3.0)) * ((((np.where(3.0 < -99998, (9.90981674194335938), (np.tanh(((((data["t_sum_log"]) - ((((np.maximum(((data["t_mean"])), (((((data["t_mean"]) > (data["t_mean"]))*1.))))) + (data["visitNumber"]))/2.0)))) * 2.0)) ) )) + (-1.0))/2.0)))) * ((9.90981674194335938))))  
    v["i32"] = 0.100000*np.tanh((((((((np.tanh(np.minimum(((1.0)), ((np.minimum(((((((((data["t_sum_log"]) * ((1.95805239677429199)))) + (-2.0))) - ((2.0))))), ((data["geoNetwork_continent"]))))))) )) * 2.0)) - ((2.54177641868591309)))) + ((((np.tanh(np.minimum(((1.0)), ((np.minimum(((((((((data["t_sum_log"]) * ((1.95805239677429199)))) + (-2.0))) - ((2.0))))), ((data["geoNetwork_continent"]))))))) )) * ((2.54177641868591309))))))  
    v["i33"] = 0.100000*np.tanh((((((13.79329872131347656)) * ((((np.tanh((((data["pred_0"]) + (((((((data["t_sum_log"]) - (((data["t_median"]) * (data["visitNumber"]))))) + (data["t_sum_log"]))) - (data["visitNumber"]))))/2.0)) )) + (np.where((np.tanh((((data["pred_0"]) + (((((((data["t_sum_log"]) - (((data["t_median"]) * (data["visitNumber"]))))) + (data["t_sum_log"]))) - (data["visitNumber"]))))/2.0)) ) < -99998, (np.tanh(np.minimum(((data["pred_3"])), ((((((data["totals_pageviews"]) + ((((np.maximum(((data["totals_pageviews"])), ((-2.0)))) > (-2.0))*1.)))) + (data["t_mean"])))))) ), -1.0 )))))) * 2.0))  
    v["i34"] = 0.100000*np.tanh((-1.0*((((((-1.0) + ((np.tanh(((data["t_sum_act"]) + (-3.0))) )))) * ((-1.0*(((((((np.tanh(((data["t_sum_act"]) + (-3.0))) )) + ((np.tanh(((data["t_sum_act"]) + (-3.0))) )))) * (-3.0)))))))))))  
    v["i35"] = 0.100000*np.tanh(((np.minimum((((((np.tanh(((((data["pred_0"]) * 2.0)) - (data["pred_3"]))) )) * 2.0))), (((((((np.tanh(((((-1.0) + (((((data["t_sum_log"]) + (-2.0))) + (data["t_sum_log"]))))) * 2.0)) )) * 2.0)) - ((3.0))))))) + ((np.tanh(((((-1.0) + (((((data["t_sum_log"]) + (-2.0))) + (data["t_sum_log"]))))) * 2.0)) ))))  
    v["i36"] = 0.100000*np.tanh(((np.where(-1.0 < -99998, (np.tanh(((((((((((data["t_sum_act"]) * (data["t_sum_act"]))) + (-2.0))) * 2.0)) + (-1.0))) * 2.0)) ), (((((np.tanh(((((((((((data["t_sum_act"]) * (data["t_sum_act"]))) + (-2.0))) * 2.0)) + (-1.0))) * 2.0)) )) * 2.0)) + (-2.0)) )) * 2.0))  
    v["i37"] = 0.100000*np.tanh((-1.0*((np.where((np.tanh(((((data["t_mean"]) * (np.where(data["pred_4"]<0, data["t_sum_log"], data["t_sum_log"] )))) + (-3.0))) )>0, (((((np.tanh(np.maximum((((((data["t_nb_sess"]) + (data["t_sum_act"]))/2.0))), ((data["t_nb_sess"])))) )) - (1.0))) * ((7.0))), (8.39564132690429688) )))))  
    v["i38"] = 0.100000*np.tanh(((((((np.where((np.tanh(((data["t_sum_log"]) * (np.where(((data["t_sum_log"]) - (np.minimum(((data["visitNumber"])), ((data["t_sum_log"])))))>0, data["t_sum_log"], -1.0 )))) ) < -99998, (((np.tanh(((data["t_sum_log"]) * (np.where(((data["t_sum_log"]) - (np.minimum(((data["visitNumber"])), ((data["t_sum_log"])))))>0, data["t_sum_log"], -1.0 )))) )) * 2.0), (((np.tanh(((data["t_sum_log"]) * (np.where(((data["t_sum_log"]) - (np.minimum(((data["visitNumber"])), ((data["t_sum_log"])))))>0, data["t_sum_log"], -1.0 )))) )) * 2.0) )) - (2.0))) * 2.0)) * 2.0))  
    v["i39"] = 0.100000*np.tanh(((np.minimum(((((((((((-1.0) + ((np.tanh(((np.where(((data["pred_2"]) + (data["t_sum_act"])) < -99998, data["pred_5"], data["t_sum_act"] )) + (((data["t_sum_act"]) + (np.tanh((-3.0))))))) )))) + ((np.tanh((((data["pred_6"]) < (((((data["pred_6"]) * 2.0)) + (data["t_mean"]))))*1.)) )))) * 2.0)) * 2.0))), (((np.tanh(((np.where(((data["pred_2"]) + (data["t_sum_act"])) < -99998, data["pred_5"], data["t_sum_act"] )) + (((data["t_sum_act"]) + (np.tanh((-3.0))))))) ))))) * 2.0))  
    v["i40"] = 0.100000*np.tanh(((((-2.0) + (((np.where(-2.0<0, np.where((np.tanh(((((((data["t_sum_act"]) - (data["visitNumber"]))) * 2.0)) + (-1.0))) )>0, (np.tanh(((((((data["t_sum_act"]) - (data["visitNumber"]))) * 2.0)) + (-1.0))) ), -2.0 ), -2.0 )) * 2.0)))) * 2.0))  
    v["i41"] = 0.100000*np.tanh(np.minimum(((((((((np.tanh(((((data["t_sum_log"]) * ((-1.0*((data["totals_hits"])))))) + (((((data["t_mean"]) + (data["t_sum_log"]))) + (data["t_sum_log"]))))) )) - ((3.08904719352722168)))) + (2.0))/2.0))), ((((((3.0) * ((np.tanh(np.where(data["t_sum_act"] < -99998, np.minimum(((((data["t_mean"]) * 2.0))), ((np.maximum(((0.0)), ((data["t_sum_log"])))))), data["t_sum_log"] )) )))) - (3.0))))))  
    v["i42"] = 0.100000*np.tanh((((((-3.0) + ((((-2.0) + ((np.tanh(((((((((((data["t_sum_log"]) * (data["t_sum_act"]))) * (data["geoNetwork_continent"]))) * (data["t_sum_act"]))) * (data["t_sum_act"]))) - (1.0))) )))/2.0)))/2.0)) + ((((np.tanh(((((((((((data["t_sum_log"]) * (data["t_sum_act"]))) * (data["geoNetwork_continent"]))) * (data["t_sum_act"]))) * (data["t_sum_act"]))) - (1.0))) )) * 2.0))))  
    v["i43"] = 0.100000*np.tanh((((((((np.tanh(((((data["t_sum_log"]) * (((data["t_sum_log"]) * (((np.minimum(((data["t_sum_log"])), ((data["t_sum_log"])))) - (data["visitNumber"]))))))) + (data["t_sum_log"]))) )) + (-1.0))) * 2.0)) * 2.0))  
    v["i44"] = 0.100000*np.tanh(np.where((np.tanh(np.where(np.where(data["pred_1"]>0, data["pred_2"], data["t_sum_log"] ) < -99998, data["pred_2"], ((-1.0) + (data["t_sum_act"])) )) )<0, ((-2.0) * ((8.40580558776855469))), (((np.tanh(((((np.tanh(np.where(np.where(data["pred_1"]>0, data["pred_2"], data["t_sum_log"] ) < -99998, data["pred_2"], ((-1.0) + (data["t_sum_act"])) )) )) * ((4.0)))))) > ((np.tanh(np.where(np.where(data["pred_1"]>0, data["pred_2"], data["t_sum_log"] ) < -99998, data["pred_2"], ((-1.0) + (data["t_sum_act"])) )) )))*1.) ))  
    v["i45"] = 0.100000*np.tanh(((((((((-1.0) + ((np.tanh(np.where(((((((data["t_sum_log"]) * 2.0)) - (data["visitNumber"]))) - (data["visitNumber"]))>0, ((data["t_sum_log"]) * 2.0), data["pred_3"] )) )))) * 2.0)) * 2.0)) * 2.0))  
    v["i46"] = 0.100000*np.tanh(np.where((((np.tanh(np.where(data["visitNumber"]<0, np.where(data["t_mean"]>0, (((data["geoNetwork_continent"]) + (data["visitNumber"]))/2.0), data["pred_1"] ), (((data["pred_1"]) + (data["t_sum_log"]))/2.0) )) )) / 2.0)>0, (((((np.tanh(np.where(data["visitNumber"]<0, np.where(data["t_mean"]>0, (((data["geoNetwork_continent"]) + (data["visitNumber"]))/2.0), data["pred_1"] ), (((data["pred_1"]) + (data["t_sum_log"]))/2.0) )) )) / 2.0)) / 2.0), np.minimum(((((((-3.0) * 2.0)) * 2.0))), (((np.tanh(np.where(data["visitNumber"]<0, np.where(data["t_mean"]>0, (((data["geoNetwork_continent"]) + (data["visitNumber"]))/2.0), data["pred_1"] ), (((data["pred_1"]) + (data["t_sum_log"]))/2.0) )) )))) ))  
    v["i47"] = 0.100000*np.tanh((((7.32015085220336914)) * (((np.tanh((1.0))) * ((((np.tanh(((((((np.minimum(((data["t_sum_log"])), ((data["geoNetwork_metro"])))) * 2.0)) + (np.where(data["pred_0"]>0, data["pred_0"], data["t_sum_log"] )))) * 2.0)) )) + (-1.0)))))))  
    v["i48"] = 0.100000*np.tanh((((((((np.tanh(((np.tanh((data["t_sum_log"]))) + ((((data["t_sum_log"]) + (data["t_sum_log"]))/2.0)))) )) - (((((((0.0) < ((np.tanh(((np.tanh((data["t_sum_log"]))) + ((((data["t_sum_log"]) + (data["t_sum_log"]))/2.0)))) )))*1.)) > (((((np.tanh(((np.tanh((data["t_sum_log"]))) + ((((data["t_sum_log"]) + (data["t_sum_log"]))/2.0)))) )) + ((np.tanh(((np.tanh((data["t_sum_log"]))) + ((((data["t_sum_log"]) + (data["t_sum_log"]))/2.0)))) )))/2.0)))*1.)))) * 2.0)) * 2.0))  
    v["i49"] = 0.100000*np.tanh((((((np.where((np.tanh(np.where(np.where(((data["t_sum_act"]) + (data["t_sum_log"]))<0, data["pred_4"], (((data["totals_pageviews"]) > (data["totals_hits"]))*1.) )>0, data["totals_hits"], data["pred_4"] )) )>0, np.where((np.tanh(np.where(np.where(((data["t_sum_act"]) + (data["t_sum_log"]))<0, data["pred_4"], (((data["totals_pageviews"]) > (data["totals_hits"]))*1.) )>0, data["totals_hits"], data["pred_4"] )) )>0, (np.tanh(np.where(np.where(((data["t_sum_act"]) + (data["t_sum_log"]))<0, data["pred_4"], (((data["totals_pageviews"]) > (data["totals_hits"]))*1.) )>0, data["totals_hits"], data["pred_4"] )) ), -3.0 ), -3.0 )) + (np.tanh((-1.0))))/2.0)) * 2.0))  
    v["i50"] = 0.100000*np.tanh(np.minimum((((((np.tanh(((((data["t_sum_act"]) / 2.0)) + (((data["t_mean"]) - (((1.0) * 2.0)))))) )) * ((((11.44591140747070312)) + (1.0)))))), (((((1.0) > ((((1.0) > ((np.tanh(((((data["t_sum_act"]) / 2.0)) + (((data["t_mean"]) - (((1.0) * 2.0)))))) )))*1.)))*1.)))))  
    v["i51"] = 0.100000*np.tanh(np.where((np.tanh((((data["t_mean"]) + (np.where((((data["t_sum_log"]) + (data["pred_1"]))/2.0)<0, data["t_sum_log"], np.tanh((data["t_mean"])) )))/2.0)) )>0, ((((np.tanh((((data["t_mean"]) + (np.where((((data["t_sum_log"]) + (data["pred_1"]))/2.0)<0, data["t_sum_log"], np.tanh((data["t_mean"])) )))/2.0)) )) < (((((np.tanh((((data["t_mean"]) + (np.where((((data["t_sum_log"]) + (data["pred_1"]))/2.0)<0, data["t_sum_log"], np.tanh((data["t_mean"])) )))/2.0)) )) < (np.tanh(((3.0)))))*1.)))*1.), -3.0 ))  
    v["i52"] = 0.099756*np.tanh(np.where((np.tanh(np.where(np.minimum(((np.where(data["t_sum_act"] < -99998, data["geoNetwork_subContinent"], data["t_mean"] ))), ((((data["geoNetwork_subContinent"]) * 2.0))))<0, data["pred_3"], np.tanh((data["t_sum_act"])) )) )>0, ((np.maximum((((np.tanh(np.where(np.minimum(((np.where(data["t_sum_act"] < -99998, data["geoNetwork_subContinent"], data["t_mean"] ))), ((((data["geoNetwork_subContinent"]) * 2.0))))<0, data["pred_3"], np.tanh((data["t_sum_act"])) )) ))), (((np.tanh(np.where(np.minimum(((np.where(data["t_sum_act"] < -99998, data["geoNetwork_subContinent"], data["t_mean"] ))), ((((data["geoNetwork_subContinent"]) * 2.0))))<0, data["pred_3"], np.tanh((data["t_sum_act"])) )) ))))) - (np.tanh((2.0)))), ((-3.0) - (2.0)) ))  
    v["i53"] = 0.100000*np.tanh((((((((((((((np.tanh(((((((-1.0) + (((((((data["t_sum_log"]) * ((9.82299137115478516)))) - (3.0))) * 2.0)))) - (1.0))) * 2.0)) )) + ((((np.tanh(((((((-1.0) + (((((((data["t_sum_log"]) * ((9.82299137115478516)))) - (3.0))) * 2.0)))) - (1.0))) * 2.0)) )) - ((np.tanh((((data["totals_hits"]) + ((((((data["t_mean"]) + (data["t_sum_log"]))/2.0)) + (data["geoNetwork_metro"]))))/2.0)) )))))) - ((np.tanh(((((((-1.0) + (((((((data["t_sum_log"]) * ((9.82299137115478516)))) - (3.0))) * 2.0)))) - (1.0))) * 2.0)) )))) * 2.0)) * 2.0)) * 2.0)) * 2.0))  
    v["i54"] = 0.100000*np.tanh((((((((np.tanh(np.where(data["pred_4"]<0, ((data["t_median"]) + (np.where(data["pred_1"]<0, ((data["t_sum_act"]) * 2.0), data["totals_pageviews"] ))), data["totals_pageviews"] )) )) + (-1.0))) * 2.0)) * 2.0))  
    v["i55"] = 0.100000*np.tanh(((((((((np.tanh(np.minimum(((np.minimum(((((data["geoNetwork_metro"]) + (np.minimum(((((data["geoNetwork_metro"]) + (1.0)))), ((data["t_sum_act"]))))))), ((data["t_sum_act"]))))), ((data["t_mean"])))) )) < (0.0))*1.)) * (-3.0))) * 2.0))  
    v["i56"] = 0.099951*np.tanh(np.where((np.tanh(((np.tanh((data["pred_1"]))) + ((((((data["t_sum_log"]) + (((((data["t_median"]) * 2.0)) * (((data["pred_0"]) * 2.0)))))/2.0)) * 2.0)))) )<0, np.minimum(((np.minimum(((-3.0)), ((np.maximum(((0.0)), ((0.0)))))))), ((-3.0))), 0.0 ))  
    v["i57"] = 0.098095*np.tanh(((np.minimum((((((14.66910171508789062)) * ((((((14.66910171508789062)) * ((np.tanh((((-1.0) + (np.where(((data["totals_hits"]) + (-3.0))>0, data["geoNetwork_metro"], ((data["t_mean"]) + (((data["t_sum_act"]) * 2.0))) )))/2.0)) )))) * 2.0))))), ((np.minimum(((0.0)), ((0.0))))))) * 2.0))  
    v["i58"] = 0.100000*np.tanh(np.where((np.tanh(((((((np.minimum(((((data["totals_hits"]) - (data["totals_pageviews"])))), ((data["t_mean"])))) * 2.0)) * 2.0)) + (np.tanh((data["totals_hits"]))))) )>0, np.where((0.24389153718948364)>0, np.where((np.tanh(((((((np.minimum(((((data["totals_hits"]) - (data["totals_pageviews"])))), ((data["t_mean"])))) * 2.0)) * 2.0)) + (np.tanh((data["totals_hits"]))))) )>0, (0.24389153718948364), -3.0 ), (0.24389153718948364) ), -3.0 ))  
    v["i59"] = 0.051295*np.tanh(np.minimum((((np.tanh(np.where(np.where(((data["t_mean"]) * 2.0)>0, data["t_mean"], data["t_mean"] )>0, (((data["t_mean"]) > (data["t_sum_log"]))*1.), -3.0 )) ))), ((((3.0) * (np.where((np.tanh(np.where(np.where(((data["t_mean"]) * 2.0)>0, data["t_mean"], data["t_mean"] )>0, (((data["t_mean"]) > (data["t_sum_log"]))*1.), -3.0 )) )<0, -3.0, (np.tanh(np.where(np.where(((data["t_mean"]) * 2.0)>0, data["t_mean"], data["t_mean"] )>0, (((data["t_mean"]) > (data["t_sum_log"]))*1.), -3.0 )) ) )))))))  
    v["i60"] = 0.099853*np.tanh(((((-1.0) + ((np.tanh(((np.where(data["t_sum_log"]>0, (((np.where(data["pred_1"]<0, data["t_sum_act"], 2.0 )) > (data["pred_3"]))*1.), data["pred_3"] )) * 2.0)) )))) * 2.0))  
    v["i61"] = 0.099805*np.tanh((((3.0)) * (np.minimum((((np.tanh(((data["t_sum_log"]) + (np.minimum(((np.minimum(((((data["t_sum_log"]) - (data["visitNumber"])))), ((((data["t_mean"]) + (data["t_median"]))))))), ((data["t_median"])))))) ))), ((((3.0) * (np.minimum(((0.0)), (((np.tanh(((data["t_sum_log"]) + (np.minimum(((np.minimum(((((data["t_sum_log"]) - (data["visitNumber"])))), ((((data["t_mean"]) + (data["t_median"]))))))), ((data["t_median"])))))) ))))))))))))  
    v["i62"] = 0.083781*np.tanh((((((np.tanh(((data["totals_hits"]) + (((data["t_median"]) * (np.maximum(((np.minimum(((data["t_sum_act"])), ((data["t_sum_log"]))))), ((((data["t_sum_act"]) + (data["t_median"])))))))))) )) - (np.tanh((((2.0) + (2.0))))))) * (2.0)))  
    v["i63"] = 0.099951*np.tanh(np.where((np.tanh(np.minimum((((((data["t_mean"]) < (data["t_sum_act"]))*1.))), ((((((data["t_mean"]) + (data["t_sum_act"]))) + (data["t_sum_act"])))))) )<0, ((-3.0) * 2.0), np.minimum((((((np.tanh(np.minimum((((((data["t_mean"]) < (data["t_sum_act"]))*1.))), ((((((data["t_mean"]) + (data["t_sum_act"]))) + (data["t_sum_act"])))))) )) * 2.0))), ((np.tanh((np.minimum((((np.tanh(np.minimum((((((data["t_mean"]) < (data["t_sum_act"]))*1.))), ((((((data["t_mean"]) + (data["t_sum_act"]))) + (data["t_sum_act"])))))) ))), (((np.tanh(np.minimum((((((data["t_mean"]) < (data["t_sum_act"]))*1.))), ((((((data["t_mean"]) + (data["t_sum_act"]))) + (data["t_sum_act"])))))) ))))))))) ))  
    v["i64"] = 0.076014*np.tanh(np.minimum(((np.minimum((((np.tanh(((((-2.0) + (((data["t_sum_log"]) / 2.0)))) * ((12.65624237060546875)))) ))), (((((np.tanh(((((-2.0) + (((data["t_sum_log"]) / 2.0)))) * ((12.65624237060546875)))) )) + (-1.0))))))), (((np.tanh(((((-2.0) + (((data["t_sum_log"]) / 2.0)))) * ((12.65624237060546875)))) )))))  
    v["i65"] = 0.046165*np.tanh(np.minimum(((np.minimum((((np.tanh(((data["t_sum_act"]) - (data["t_sum_log"]))) ))), (((((0.0) > ((np.tanh(((data["t_sum_act"]) - (data["t_sum_log"]))) )))*1.)))))), ((0.0))))  
    v["i66"] = 0.100000*np.tanh(np.minimum((((((((np.tanh(((((data["t_sum_log"]) * 2.0)) + ((((np.where(np.minimum(((data["t_sum_log"])), ((data["visitNumber"])))<0, data["totals_hits"], data["pred_1"] )) + (-2.0))/2.0)))) )) * 2.0)) * 2.0))), ((((np.minimum((((0.06700874865055084))), (((np.tanh(((((data["t_sum_log"]) * 2.0)) + ((((np.where(np.minimum(((data["t_sum_log"])), ((data["visitNumber"])))<0, data["totals_hits"], data["pred_1"] )) + (-2.0))/2.0)))) ))))) * 2.0)))))  
    v["i67"] = 0.098828*np.tanh(np.minimum((((((0.0) > ((((0.0) > (0.0))*1.)))*1.))), (((np.tanh(((data["t_sum_log"]) * (2.0))) )))))  
    v["i68"] = 0.100000*np.tanh(((np.minimum((((((np.tanh(np.minimum(((((data["totals_pageviews"]) - (((((data["totals_pageviews"]) - (data["totals_hits"]))) * ((10.0))))))), ((((data["totals_pageviews"]) - (data["totals_hits"])))))) )) * 2.0))), (((((np.tanh(np.minimum(((((data["totals_pageviews"]) - (((((data["totals_pageviews"]) - (data["totals_hits"]))) * ((10.0))))))), ((((data["totals_pageviews"]) - (data["totals_hits"])))))) )) * 2.0))))) * 2.0))  
    v["i69"] = 0.049731*np.tanh(np.minimum((((0.04890204593539238))), ((((-2.0) * (np.maximum((((np.tanh((((((data["t_mean"]) + (-1.0))) > (data["t_sum_log"]))*1.)) ))), (((0.04890204593539238))))))))))  
    v["i70"] = 0.098925*np.tanh(np.minimum(((0.0)), (((np.tanh(((data["t_sum_log"]) - (1.0))) )))))  
    v["i71"] = 0.068100*np.tanh(((np.tanh((((np.tanh((np.tanh((((-3.0) + (-2.0))))))) + ((np.tanh(np.minimum((((((((data["t_sum_act"]) - (((2.0) * (np.tanh((-1.0))))))) < ((7.0)))*1.))), ((data["pred_2"])))) )))))) + (1.0)))  
    v["i72"] = 0.100000*np.tanh(np.minimum(((((np.minimum((((np.tanh((((7.90549468994140625)) + (((((((data["t_sum_log"]) * ((-1.0*((data["pred_2"])))))) + (data["t_mean"]))) + (data["t_mean"]))))) ))), (((np.tanh((((7.90549468994140625)) + (((((((data["t_sum_log"]) * ((-1.0*((data["pred_2"])))))) + (data["t_mean"]))) + (data["t_mean"]))))) ))))) + ((np.tanh((((7.90549468994140625)) + (((((((data["t_sum_log"]) * ((-1.0*((data["pred_2"])))))) + (data["t_mean"]))) + (data["t_mean"]))))) ))))), ((((((np.tanh((((7.90549468994140625)) + (((((((data["t_sum_log"]) * ((-1.0*((data["pred_2"])))))) + (data["t_mean"]))) + (data["t_mean"]))))) )) < (((((np.tanh((((7.90549468994140625)) + (((((((data["t_sum_log"]) * ((-1.0*((data["pred_2"])))))) + (data["t_mean"]))) + (data["t_mean"]))))) )) > ((0.0)))*1.)))*1.)))))  
    v["i73"] = 0.100000*np.tanh((((0.0) < ((np.tanh((((np.maximum(((data["t_mean"])), ((0.0)))) < (data["totals_pageviews"]))*1.)) )))*1.))  
    v["i74"] = 0.099707*np.tanh(np.minimum(((((np.minimum((((np.tanh(np.minimum(((((((((((((data["geoNetwork_continent"]) + (data["geoNetwork_metro"]))) * ((6.0)))) + (data["geoNetwork_metro"]))) * 2.0)) * 2.0))), ((data["t_median"])))) ))), ((0.0)))) * 2.0))), ((((np.minimum((((np.tanh(np.minimum(((((((((((((data["geoNetwork_continent"]) + (data["geoNetwork_metro"]))) * ((6.0)))) + (data["geoNetwork_metro"]))) * 2.0)) * 2.0))), ((data["t_median"])))) ))), ((0.0)))) * 2.0)))))  
    v["i75"] = 0.099951*np.tanh(((((((np.tanh((((data["pred_2"]) > (np.maximum((((5.28380632400512695))), ((data["pred_0"])))))*1.)) )) * 2.0)) > (0.0))*1.))  
    v["i76"] = 0.099951*np.tanh(np.minimum((((np.tanh(((data["trafficSource_referralPath"]) * (-2.0))) ))), ((0.0))))  
    v["i77"] = 0.100000*np.tanh(((((np.minimum(((np.minimum((((0.04203201457858086))), ((2.0))))), (((np.tanh(np.minimum(((data["totals_pageviews"])), (((((((data["t_mean"]) * ((-1.0*((data["trafficSource_referralPath"])))))) + (data["visitNumber"]))/2.0))))) ))))) * (2.0))) * 2.0))  
    v["i78"] = 0.099951*np.tanh((((np.minimum((((np.tanh(((data["t_nb_sess"]) + ((4.0)))) ))), (((np.tanh(np.maximum(((2.0)), ((data["totals_hits"])))) ))))) < (0.0))*1.))  
    v["i79"] = 0.099902*np.tanh(((((np.tanh((((((0.0) + (data["geoNetwork_continent"]))/2.0)) * ((4.63319158554077148)))) )) < (0.0))*1.))  
    v["i80"] = 0.099756*np.tanh(((((((np.tanh(((data["pred_2"]) * ((((data["pred_2"]) < (3.0))*1.)))) )) + ((((((((((((((-1.0) + ((np.tanh(((data["pred_2"]) * ((((data["pred_2"]) < (3.0))*1.)))) )))/2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)))/2.0)) / 2.0))  
    v["i81"] = 0.099463*np.tanh(((((((0.0) < ((np.tanh((((data["totals_hits"]) > ((4.0)))*1.)) )))*1.)) > ((np.tanh((((data["totals_hits"]) > ((4.0)))*1.)) )))*1.))  
    v["i82"] = 0.099756*np.tanh((((0.0) < ((np.tanh((((3.0) > (data["t_sum_log"]))*1.)) )))*1.))  
    v["i83"] = 0.099951*np.tanh(np.minimum(((((((np.tanh(np.minimum(((((((data["t_sum_act"]) - ((((data["totals_hits"]) + (data["totals_hits"]))/2.0)))) + (data["visitNumber"])))), ((((data["visitNumber"]) - (data["trafficSource_referralPath"])))))) )) < (np.minimum((((-1.0*(((((-1.0) < (((2.0) / 2.0)))*1.)))))), (((np.tanh(np.minimum(((((((data["t_sum_act"]) - ((((data["totals_hits"]) + (data["totals_hits"]))/2.0)))) + (data["visitNumber"])))), ((((data["visitNumber"]) - (data["trafficSource_referralPath"])))))) ))))))*1.))), (((np.tanh(np.minimum(((((((data["t_sum_act"]) - ((((data["totals_hits"]) + (data["totals_hits"]))/2.0)))) + (data["visitNumber"])))), ((((data["visitNumber"]) - (data["trafficSource_referralPath"])))))) )))))  
    v["i84"] = 0.100000*np.tanh(((np.minimum((((((((np.minimum((((9.0))), (((np.tanh(((((data["channelGrouping"]) + (((data["geoNetwork_continent"]) + (data["geoNetwork_metro"]))))) * (data["geoNetwork_metro"]))) ))))) > (2.0))*1.)) * ((np.tanh(((((data["channelGrouping"]) + (((data["geoNetwork_continent"]) + (data["geoNetwork_metro"]))))) * (data["geoNetwork_metro"]))) ))))), (((np.tanh(((((data["channelGrouping"]) + (((data["geoNetwork_continent"]) + (data["geoNetwork_metro"]))))) * (data["geoNetwork_metro"]))) ))))) * ((11.39656257629394531))))  
    v["i85"] = 0.099951*np.tanh(((((((np.minimum((((np.tanh(np.minimum(((((((((1.0) < (data["t_sum_log"]))*1.)) > (data["geoNetwork_metro"]))*1.))), ((np.minimum(((data["t_sum_log"])), (((((data["t_mean"]) + (data["totals_hits"]))/2.0)))))))) ))), (((0.03772259503602982))))) * 2.0)) * 2.0)) * 2.0))  
    v["i86"] = 0.099365*np.tanh(((((np.tanh((((-2.0) + (data["t_mean"]))/2.0)) )) < (0.0))*1.))  
    v["i87"] = 0.099853*np.tanh(np.minimum(((np.minimum(((0.0)), ((np.minimum(((0.0)), ((0.0)))))))), ((np.minimum((((np.tanh(((((-2.0) + (np.minimum((((((((13.26135158538818359)) * (data["geoNetwork_country"]))) - (data["geoNetwork_country"])))), ((data["totals_hits"])))))) * (data["trafficSource_source"]))) ))), ((0.0)))))))  
    v["i88"] = 0.099951*np.tanh(np.minimum((((((np.tanh(np.where(data["t_sum_act"]<0, data["t_sum_log"], np.tanh(((((np.maximum(((data["pred_1"])), (((5.75660467147827148))))) > (data["t_sum_log"]))*1.))) )) )) - ((0.17407540977001190))))), (((np.tanh(np.where(data["t_sum_act"]<0, data["t_sum_log"], np.tanh(((((np.maximum(((data["pred_1"])), (((5.75660467147827148))))) > (data["t_sum_log"]))*1.))) )) )))))  
    v["i89"] = 0.099902*np.tanh(((np.tanh(((4.0)))) + (np.tanh((((np.where((np.tanh(((data["t_sum_log"]) - (((np.maximum(((-2.0)), ((data["trafficSource_source"])))) + (data["pred_2"]))))) )<0, (13.98869228363037109), -1.0 )) + (-2.0)))))))  
    v["i90"] = 0.100000*np.tanh((-1.0*(((((((np.minimum(((0.0)), (((np.tanh(((np.maximum(((-1.0)), ((((np.minimum(((data["t_sum_log"])), ((data["t_sum_act"])))) * (data["channelGrouping"])))))) - (((data["t_sum_log"]) * (data["channelGrouping"]))))) ))))) * (2.0))) < ((0.0)))*1.)))))  
    v["i91"] = 0.066097*np.tanh(((((0.05042554065585136)) + ((((((((np.tanh(((((((data["t_median"]) - ((((data["geoNetwork_metro"]) + ((((data["t_mean"]) + (data["pred_2"]))/2.0)))/2.0)))) * (data["pred_1"]))) * 2.0)) )) + ((((np.tanh(((((((data["t_median"]) - ((((data["geoNetwork_metro"]) + ((((data["t_mean"]) + (data["pred_2"]))/2.0)))/2.0)))) * (data["pred_1"]))) * 2.0)) )) / 2.0)))/2.0)) + ((np.tanh(((((((data["t_median"]) - ((((data["geoNetwork_metro"]) + ((((data["t_mean"]) + (data["pred_2"]))/2.0)))/2.0)))) * (data["pred_1"]))) * 2.0)) )))/2.0)))/2.0))  
    v["i92"] = 0.100000*np.tanh((((((np.tanh(((((12.78093624114990234)) < (((data["pred_0"]) + (np.maximum(((data["channelGrouping"])), ((np.maximum(((data["totals_hits"])), ((data["channelGrouping"]))))))))))*1.)) )) + ((((np.tanh(((((12.78093624114990234)) < (((data["pred_0"]) + (np.maximum(((data["channelGrouping"])), ((np.maximum(((data["totals_hits"])), ((data["channelGrouping"]))))))))))*1.)) )) / 2.0)))) / 2.0))  
    v["i93"] = 0.100000*np.tanh((((6.0)) * ((((6.0)) * (np.minimum(((((((6.0)) < (np.minimum(((0.0)), ((0.0)))))*1.))), (((np.tanh(np.where(np.maximum(((((data["geoNetwork_continent"]) + (data["geoNetwork_metro"])))), ((-1.0)))>0, np.minimum(((data["t_sum_act"])), ((data["t_sum_log"]))), data["pred_2"] )) )))))))))  
    v["i94"] = 0.077088*np.tanh((((np.where((np.tanh(np.where(data["pred_4"] < -99998, ((((1.0) + (data["pred_4"]))) / 2.0), ((data["pred_2"]) - (data["trafficSource_source"])) )) )<0, (0.01666546240448952), (12.52340126037597656) )) + ((0.01666546240448952)))/2.0))  
    v["i95"] = 0.082413*np.tanh((((((np.tanh(((((data["channelGrouping"]) - (data["visitNumber"]))) * 2.0)) )) / 2.0)) / 2.0))  
    v["i96"] = 0.100000*np.tanh(np.maximum((((np.tanh(np.tanh((np.tanh((((((((data["geoNetwork_continent"]) + (((data["t_nb_sess"]) / 2.0)))/2.0)) < (-1.0))*1.)))))) ))), ((-3.0))))  
    v["i97"] = 0.099902*np.tanh(((((np.tanh((((4.26091861724853516)) - (data["totals_pageviews"]))) )) < (0.0))*1.))  
    v["i98"] = 0.100000*np.tanh((((0.0) > ((np.tanh(((data["t_nb_sess"]) + ((7.0)))) )))*1.))  
    v["i99"] = 0.046654*np.tanh(((np.minimum(((0.0)), (((np.tanh(((data["t_sum_act"]) + (((np.minimum((((((3.0) > (data["geoNetwork_metro"]))*1.))), ((data["totals_pageviews"])))) - (((data["t_mean"]) + (data["geoNetwork_subContinent"])))))))))))) * 2.0))

    return Output(v.sum(axis=1))

def GP2(data):
    v = pd.DataFrame()
    v["i0"] = 0.100000*np.tanh(((np.tanh((data["pred_4"]))) + (((((((((data["t_sum_act"]) - (((3.0) * 2.0)))) * 2.0)) + (-2.0))) * 2.0)))) 
    v["i1"] = 0.100000*np.tanh(((((((data["pred_4"]) + (((((np.maximum(((data["t_median"])), ((data["pred_3"])))) - ((4.92847061157226562)))) + (data["pred_2"]))))) * 2.0)) * 2.0)) 
    v["i2"] = 0.100000*np.tanh(((((-1.0) - (data["t_sum_log"]))) + (((((((data["t_sum_log"]) - ((13.55201435089111328)))) + (data["t_mean"]))) * ((13.55201435089111328)))))) 
    v["i3"] = 0.100000*np.tanh(((3.0) * (((data["t_sum_act"]) - (np.where(np.where(data["pred_2"] < -99998, data["pred_1"], 3.0 )>0, (7.0), (9.0) )))))) 
    v["i4"] = 0.100000*np.tanh(((data["t_mean"]) + ((((((((9.21076107025146484)) / 2.0)) - ((((9.21076107025146484)) * 2.0)))) + (np.maximum((((3.85717129707336426))), ((data["t_sum_log"])))))))) 
    v["i5"] = 0.100000*np.tanh(((((np.maximum(((((((data["pred_4"]) * 2.0)) + (data["pred_0"])))), ((data["pred_0"])))) - ((((12.73920822143554688)) - (data["geoNetwork_metro"]))))) * 2.0)) 
    v["i6"] = 0.100000*np.tanh(((((np.minimum(((((((data["t_median"]) * (((data["visitNumber"]) * 2.0)))) + (data["visitNumber"])))), ((data["pred_1"])))) * 2.0)) - ((9.0)))) 
    v["i7"] = 0.100000*np.tanh((((((6.33722066879272461)) * (((((((data["t_sum_log"]) - ((6.33722066879272461)))) + (((-3.0) * 2.0)))) + (data["t_mean"]))))) * 2.0)) 
    v["i8"] = 0.100000*np.tanh(((((np.maximum(((data["pred_4"])), ((data["pred_2"])))) + (data["pred_6"]))) - (((3.0) - (np.maximum(((data["pred_4"])), ((data["pred_1"])))))))) 
    v["i9"] = 0.100000*np.tanh(((((((((-2.0) + (data["t_sum_act"]))) - ((3.93038845062255859)))) * 2.0)) * 2.0)) 
    v["i10"] = 0.100000*np.tanh(((((((np.maximum(((data["pred_10"])), ((np.maximum(((data["pred_13"])), ((np.maximum(((data["pred_5"])), ((data["pred_3"])))))))))) - (2.0))) * 2.0)) * 2.0)) 
    v["i11"] = 0.100000*np.tanh(((((((((data["t_sum_log"]) - ((((9.0)) - (((data["t_mean"]) + (-2.0))))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i12"] = 0.100000*np.tanh(((((data["t_sum_log"]) - (((3.0) * 2.0)))) - (((((((((data["visitNumber"]) > (2.0))*1.)) * 2.0)) + (data["visitNumber"]))/2.0)))) 
    v["i13"] = 0.100000*np.tanh(((((((-2.0) + (np.where(data["pred_2"]>0, np.where(data["pred_3"]>0, data["pred_5"], data["pred_2"] ), data["pred_3"] )))) * 2.0)) * 2.0)) 
    v["i14"] = 0.100000*np.tanh(np.where(((data["t_sum_log"]) + (((data["t_mean"]) - ((9.50519275665283203)))))<0, ((-3.0) - ((9.50519275665283203))), 1.0 )) 
    v["i15"] = 0.100000*np.tanh(((((((((((data["pred_6"]) * 2.0)) * 2.0)) + (data["pred_1"]))) + (np.minimum(((data["pred_7"])), ((-3.0)))))) * 2.0)) 
    v["i16"] = 0.100000*np.tanh(((data["t_sum_act"]) + (((((((data["t_sum_act"]) * (2.0))) - (np.maximum(((3.0)), ((data["visitNumber"])))))) - ((11.07681083679199219)))))) 
    v["i17"] = 0.100000*np.tanh(((-3.0) + (((((-2.0) + (data["t_mean"]))) - (np.maximum(((0.0)), ((((data["visitNumber"]) * (data["visitNumber"])))))))))) 
    v["i18"] = 0.100000*np.tanh(((np.minimum((((((3.0)) - (((data["t_mean"]) - ((4.0))))))), ((((data["t_sum_log"]) - ((4.0))))))) * 2.0)) 
    v["i19"] = 0.100000*np.tanh(((((data["t_sum_act"]) - (np.maximum(((data["visitNumber"])), ((np.maximum(((data["t_mean"])), (((((7.68621540069580078)) - (data["t_mean"]))))))))))) * 2.0)) 
    v["i20"] = 0.100000*np.tanh(((-3.0) + (((np.where(((data["t_sum_log"]) - (data["visitNumber"]))>0, data["t_sum_log"], ((data["t_sum_log"]) / 2.0) )) + (-2.0))))) 
    v["i21"] = 0.100000*np.tanh((-1.0*((np.maximum((((((data["t_sum_act"]) > ((7.0)))*1.))), ((np.where((((data["t_sum_log"]) + (-3.0))/2.0)<0, (7.0), data["pred_6"] )))))))) 
    v["i22"] = 0.100000*np.tanh(np.minimum(((((((((((data["t_sum_log"]) - ((4.0)))) * ((13.82894325256347656)))) * (3.0))) * ((4.0))))), ((0.0)))) 
    v["i23"] = 0.100000*np.tanh(((((((np.minimum(((0.0)), ((((((((data["t_sum_act"]) * 2.0)) + ((-1.0*(((7.07149600982666016))))))) * 2.0))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i24"] = 0.100000*np.tanh(np.minimum(((((((data["t_sum_log"]) + (np.minimum(((data["t_sum_log"])), ((data["t_mean"])))))) - ((6.0))))), (((((data["t_sum_act"]) < ((6.0)))*1.))))) 
    v["i25"] = 0.100000*np.tanh(np.where((((-3.0) + (data["t_sum_act"]))/2.0)>0, (((((data["t_sum_log"]) - (3.0))) > (data["t_mean"]))*1.), (-1.0*(((9.0)))) )) 
    v["i26"] = 0.100000*np.tanh((((((((3.0) > (data["t_sum_act"]))*1.)) * (((-3.0) * 2.0)))) - ((((data["t_sum_act"]) < (((data["totals_hits"]) * 2.0)))*1.)))) 
    v["i27"] = 0.100000*np.tanh(((np.where(((data["t_sum_log"]) + (-3.0))>0, np.where(data["pred_8"]<0, (0.0), data["t_sum_log"] ), ((-3.0) * 2.0) )) * 2.0)) 
    v["i28"] = 0.100000*np.tanh(((((np.minimum(((data["t_median"])), ((((((data["t_sum_act"]) - (3.0))) * ((((3.0) > (data["t_median"]))*1.))))))) * 2.0)) * 2.0)) 
    v["i29"] = 0.100000*np.tanh(np.minimum((((((((((-3.0) + (data["t_mean"]))) * 2.0)) < (3.0))*1.))), ((((((data["t_sum_log"]) - (3.0))) * 2.0))))) 
    v["i30"] = 0.100000*np.tanh(np.where(((((data["t_sum_log"]) * 2.0)) - ((3.95256853103637695)))>0, (((data["t_nb_sess"]) < (((-3.0) * 2.0)))*1.), ((-3.0) * 2.0) )) 
    v["i31"] = 0.100000*np.tanh(np.minimum(((np.minimum(((0.0)), ((np.minimum(((data["pred_0"])), ((data["t_median"])))))))), ((((data["t_sum_log"]) - ((((5.0)) - (data["t_sum_log"])))))))) 
    v["i32"] = 0.100000*np.tanh(((((data["t_mean"]) - ((10.0)))) * (((((((data["t_median"]) > (data["pred_5"]))*1.)) > (((((data["t_mean"]) / 2.0)) / 2.0)))*1.)))) 
    v["i33"] = 0.100000*np.tanh(((((np.minimum((((((2.0) > (((data["t_sum_act"]) + (-2.0))))*1.))), ((((data["t_sum_log"]) + (-2.0)))))) * 2.0)) * 2.0)) 
    v["i34"] = 0.100000*np.tanh(((((((((((data["t_sum_log"]) * (data["t_sum_log"]))) + (-3.0))) * ((((3.0) > (data["t_sum_log"]))*1.)))) * 2.0)) * 2.0)) 
    v["i35"] = 0.100000*np.tanh(np.where((((-3.0) + (data["t_sum_log"]))/2.0)>0, 0.0, ((((data["t_sum_log"]) - (3.0))) * 2.0) )) 
    v["i36"] = 0.100000*np.tanh((((-1.0*(((8.26050090789794922))))) * ((((data["t_sum_act"]) < (np.maximum(((((data["t_median"]) - (data["totals_hits"])))), (((2.0))))))*1.)))) 
    v["i37"] = 0.100000*np.tanh(np.minimum(((((((((data["t_sum_log"]) - (data["visitNumber"]))) * 2.0)) - (((2.0) * 2.0))))), (((((data["totals_pageviews"]) > ((4.61927986145019531)))*1.))))) 
    v["i38"] = 0.100000*np.tanh(((np.where((((data["t_sum_log"]) < (1.0))*1.)>0, -3.0, (((-3.0) < ((-1.0*((data["t_sum_log"])))))*1.) )) * 2.0)) 
    v["i39"] = 0.100000*np.tanh((((-1.0*(((6.07970380783081055))))) * ((((data["t_sum_log"]) < (((((3.29153609275817871)) + (np.maximum(((data["visitNumber"])), ((data["totals_hits"])))))/2.0)))*1.)))) 
    v["i40"] = 0.100000*np.tanh(np.where(((((data["t_sum_act"]) / 2.0)) - (np.tanh((1.0))))<0, ((-3.0) * 2.0), (((data["t_median"]) < (0.0))*1.) )) 
    v["i41"] = 0.100000*np.tanh(((np.where(((((((data["pred_0"]) + (-3.0))/2.0)) + (data["t_sum_act"]))/2.0)<0, -3.0, np.minimum(((0.0)), ((data["t_median"]))) )) * 2.0)) 
    v["i42"] = 0.100000*np.tanh(np.where((((((data["t_sum_log"]) < (1.0))*1.)) * (data["pred_2"]))<0, data["pred_5"], (-1.0*(((((data["pred_2"]) > (data["totals_pageviews"]))*1.)))) )) 
    v["i43"] = 0.100000*np.tanh(((((((((((data["geoNetwork_continent"]) + (data["t_sum_act"]))/2.0)) < (np.maximum(((data["pred_5"])), ((1.0)))))*1.)) * 2.0)) * (((data["pred_3"]) * 2.0)))) 
    v["i44"] = 0.100000*np.tanh((((-1.0*(((7.0))))) * (np.where(data["t_mean"]<0, 2.0, ((((((1.0) > (data["t_sum_log"]))*1.)) > (data["t_sum_act"]))*1.) )))) 
    v["i45"] = 0.100000*np.tanh(np.where(data["t_mean"]<0, ((data["t_mean"]) - ((6.90487623214721680))), (((((data["totals_pageviews"]) > (data["totals_hits"]))*1.)) / 2.0) )) 
    v["i46"] = 0.100000*np.tanh(((data["pred_4"]) * (((((((data["pred_4"]) < (data["geoNetwork_continent"]))*1.)) > (((data["geoNetwork_continent"]) * ((((data["totals_hits"]) + (1.0))/2.0)))))*1.)))) 
    v["i47"] = 0.100000*np.tanh(((((np.where(((data["t_sum_act"]) + (np.tanh((-3.0))))>0, ((((2.22855496406555176)) > (data["t_mean"]))*1.), -2.0 )) * 2.0)) * 2.0)) 
    v["i48"] = 0.100000*np.tanh(np.where(data["t_mean"]>0, np.minimum(((0.0)), ((np.where(data["geoNetwork_subContinent"]>0, (((data["geoNetwork_subContinent"]) + (data["geoNetwork_metro"]))/2.0), -3.0 )))), -3.0 )) 
    v["i49"] = 0.099805*np.tanh(np.where(((np.minimum(((data["geoNetwork_continent"])), ((((data["t_mean"]) - (1.0)))))) * 2.0)>0, ((data["t_sum_act"]) - (data["t_mean"])), -3.0 )) 
    v["i50"] = 0.099756*np.tanh(np.minimum((((-1.0*((data["pred_3"]))))), ((((((-3.0) * 2.0)) * ((((1.0) > (((data["t_sum_act"]) * 2.0)))*1.))))))) 
    v["i51"] = 0.099951*np.tanh(((np.where(np.minimum(((data["geoNetwork_subContinent"])), ((np.minimum(((data["geoNetwork_continent"])), ((data["t_sum_act"]))))))<0, -3.0, np.maximum(((data["t_nb_sess"])), ((0.0))) )) * 2.0)) 
    v["i52"] = 0.098339*np.tanh(np.where(data["t_mean"]<0, -3.0, np.minimum((((((np.where(data["geoNetwork_metro"]<0, (3.0), data["totals_pageviews"] )) > (data["totals_pageviews"]))*1.))), ((data["t_mean"]))) )) 
    v["i53"] = 0.100000*np.tanh(np.minimum(((np.where(data["pred_3"]<0, data["t_mean"], data["geoNetwork_metro"] ))), ((np.where(data["t_mean"]>0, (((data["visitNumber"]) < (data["pred_3"]))*1.), data["pred_3"] ))))) 
    v["i54"] = 0.100000*np.tanh(np.where(np.minimum(((np.where(data["trafficSource_referralPath"]>0, data["pred_5"], data["t_mean"] ))), ((data["geoNetwork_country"])))>0, 0.0, ((-3.0) + (-3.0)) )) 
    v["i55"] = 0.099951*np.tanh((-1.0*(((((((data["t_sum_log"]) < (((np.maximum(((data["t_sum_act"])), ((data["geoNetwork_subContinent"])))) * (data["geoNetwork_subContinent"]))))*1.)) * ((7.03404140472412109))))))) 
    v["i56"] = 0.100000*np.tanh(np.where(data["t_sum_log"]<0, data["pred_3"], np.where(data["t_nb_sess"]<0, 0.0, (-1.0*(((((data["totals_pageviews"]) < (data["totals_hits"]))*1.)))) ) )) 
    v["i57"] = 0.100000*np.tanh(((((np.where(data["t_mean"]<0, data["pred_3"], (((data["pred_2"]) > (((data["t_nb_sess"]) + (data["t_sum_log"]))))*1.) )) * 2.0)) * 2.0)) 
    v["i58"] = 0.100000*np.tanh(((((np.where(data["t_sum_log"]<0, -2.0, (-1.0*(((((((data["t_sum_log"]) + (data["totals_pageviews"]))) < (data["visitNumber"]))*1.)))) )) * 2.0)) * 2.0)) 
    v["i59"] = 0.099805*np.tanh(np.where(data["t_mean"]<0, -3.0, (((data["t_sum_act"]) < ((((data["t_sum_log"]) + ((((data["t_mean"]) + ((4.0)))/2.0)))/2.0)))*1.) )) 
    v["i60"] = 0.099707*np.tanh(((np.where(data["t_sum_log"]>0, (-1.0*(((((data["geoNetwork_continent"]) < (np.tanh((1.0))))*1.)))), -3.0 )) * 2.0)) 
    v["i61"] = 0.100000*np.tanh(np.where(data["t_sum_log"]>0, ((((data["trafficSource_referralPath"]) * 2.0)) * (((data["device_browser"]) - (data["trafficSource_referralPath"])))), ((data["trafficSource_referralPath"]) - ((7.0))) )) 
    v["i62"] = 0.099756*np.tanh(np.where(data["t_sum_log"]<0, -3.0, ((np.where(data["t_median"]<0, data["geoNetwork_metro"], (((-3.0) > (data["geoNetwork_metro"]))*1.) )) / 2.0) )) 
    v["i63"] = 0.100000*np.tanh((-1.0*((((((((np.maximum(((data["totals_pageviews"])), ((((data["t_median"]) / 2.0))))) < (data["totals_hits"]))*1.)) > (np.tanh((data["t_mean"]))))*1.))))) 
    v["i64"] = 0.089253*np.tanh(np.where(data["t_sum_act"]<0, -3.0, ((data["totals_hits"]) * ((((data["t_median"]) < (np.where(data["pred_0"]>0, data["pred_2"], 2.0 )))*1.))) )) 
    v["i65"] = 0.100000*np.tanh(np.where(data["pred_0"]<0, ((data["pred_0"]) * 2.0), (-1.0*(((((3.0) > (((data["pred_0"]) + (((data["geoNetwork_metro"]) * 2.0)))))*1.)))) )) 
    v["i66"] = 0.100000*np.tanh((-1.0*((np.where(data["channelGrouping"]>0, np.where(data["geoNetwork_metro"]>0, 0.0, ((data["visitNumber"]) / 2.0) ), np.minimum(((0.0)), ((data["geoNetwork_metro"]))) ))))) 
    v["i67"] = 0.100000*np.tanh(np.where(data["t_mean"]>0, np.where(data["geoNetwork_continent"]>0, (((-1.0*(((((data["geoNetwork_subContinent"]) > (data["geoNetwork_continent"]))*1.))))) * 2.0), data["pred_1"] ), -3.0 )) 
    v["i68"] = 0.083537*np.tanh(np.where(data["t_mean"]<0, -3.0, (((((((data["geoNetwork_continent"]) * (data["pred_1"]))) > (((data["t_mean"]) / 2.0)))*1.)) * (data["pred_2"])) )) 
    v["i69"] = 0.071373*np.tanh(np.where(data["t_sum_act"]<0, ((-3.0) * 2.0), (((-3.0) > (((0.0) - (((data["geoNetwork_metro"]) * 2.0)))))*1.) )) 
    v["i70"] = 0.099853*np.tanh(np.where(data["geoNetwork_continent"]<0, data["totals_pageviews"], (-1.0*(((((data["pred_2"]) > (((np.where(data["geoNetwork_metro"]<0, data["pred_0"], data["t_median"] )) / 2.0)))*1.)))) )) 
    v["i71"] = 0.099951*np.tanh(np.minimum(((data["totals_pageviews"])), ((((data["t_sum_act"]) * ((((data["t_mean"]) < (np.where(data["pred_3"]<0, data["totals_pageviews"], data["visitNumber"] )))*1.))))))) 
    v["i72"] = 0.099902*np.tanh(np.where(np.minimum(((data["t_mean"])), ((data["t_sum_log"])))<0, data["pred_4"], (((-1.0*(((((data["geoNetwork_metro"]) > (((data["t_mean"]) / 2.0)))*1.))))) * 2.0) )) 
    v["i73"] = 0.100000*np.tanh(((((np.minimum(((np.minimum(((data["t_sum_act"])), (((((data["pred_0"]) < (data["t_median"]))*1.)))))), (((((data["totals_hits"]) < (data["trafficSource_referralPath"]))*1.))))) * 2.0)) * 2.0)) 
    v["i74"] = 0.100000*np.tanh(np.where(np.where(data["trafficSource_referralPath"]<0, data["geoNetwork_country"], (((data["trafficSource_referralPath"]) > (data["geoNetwork_country"]))*1.) )>0, np.minimum(((0.0)), ((data["geoNetwork_country"]))), data["pred_2"] ))
    return Output(v.sum(axis=1))


def GP3(data):
    v = pd.DataFrame()
    v["i0"] = 0.100000*np.tanh(((((((((((np.maximum(((data["t_sum_log"])), ((data["t_mean"])))) - ((8.52710533142089844)))) * 2.0)) + (np.tanh((data["t_mean"]))))) * 2.0)) * 2.0)) 
    v["i1"] = 0.100000*np.tanh(((((((((data["t_sum_act"]) + (((np.minimum(((data["pred_1"])), ((-2.0)))) - ((5.0)))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i2"] = 0.100000*np.tanh((((9.12465190887451172)) * ((((9.12465190887451172)) * (((((data["totals_hits"]) + (((((data["t_sum_log"]) - ((9.12465190887451172)))) * 2.0)))) * 2.0)))))) 
    v["i3"] = 0.100000*np.tanh(np.minimum(((((((data["t_sum_log"]) - ((((5.99947023391723633)) - (((data["t_mean"]) - ((8.0)))))))) * 2.0))), ((data["visitNumber"])))) 
    v["i4"] = 0.100000*np.tanh(((((np.minimum(((((data["t_sum_log"]) - (np.maximum((((7.0))), ((data["visitNumber"]))))))), ((data["visitNumber"])))) * 2.0)) * 2.0)) 
    v["i5"] = 0.100000*np.tanh(((((data["t_sum_act"]) * 2.0)) - (np.maximum(((data["t_sum_act"])), ((np.maximum((((((((data["t_sum_act"]) * 2.0)) + ((13.88008403778076172)))/2.0))), (((13.80931282043457031)))))))))) 
    v["i6"] = 0.100000*np.tanh(((np.where(((-3.0) + (data["pred_3"]))>0, data["totals_hits"], ((data["pred_0"]) + (((-3.0) - ((8.0))))) )) * 2.0)) 
    v["i7"] = 0.100000*np.tanh(((((data["pred_2"]) + (((((data["pred_4"]) + (-2.0))) - ((((data["pred_9"]) < (data["pred_2"]))*1.)))))) * 2.0)) 
    v["i8"] = 0.100000*np.tanh(((np.where(((((data["t_mean"]) + (-3.0))) + (-3.0))>0, (9.18304538726806641), data["pred_3"] )) + ((-1.0*(((7.48107624053955078))))))) 
    v["i9"] = 0.100000*np.tanh(((data["pred_2"]) - ((((((((3.65108466148376465)) - (data["pred_1"]))) * ((((3.65108466148376465)) - (data["pred_2"]))))) - (data["pred_1"]))))) 
    v["i10"] = 0.100000*np.tanh((((((11.89422702789306641)) * (((((data["t_sum_log"]) + (2.0))) - ((8.0)))))) + (np.maximum(((data["pred_0"])), ((2.0)))))) 
    v["i11"] = 0.100000*np.tanh(((np.where(data["pred_8"]>0, data["pred_5"], ((data["pred_6"]) + (np.minimum(((data["pred_7"])), ((((-3.0) + (data["pred_6"]))))))) )) * 2.0)) 
    v["i12"] = 0.100000*np.tanh(((data["t_sum_log"]) - (np.maximum(((np.maximum((((6.0))), (((4.0)))))), ((((((data["t_sum_log"]) - ((6.0)))) * ((6.0))))))))) 
    v["i13"] = 0.100000*np.tanh(((data["t_sum_log"]) - (np.where(((data["t_sum_log"]) + (((data["t_sum_log"]) - ((9.73407459259033203)))))<0, (11.21657562255859375), data["visitNumber"] )))) 
    v["i14"] = 0.100000*np.tanh(((((((((((((((data["pred_4"]) * 2.0)) - (3.0))) + (data["pred_5"]))) + (data["pred_5"]))) * 2.0)) * 2.0)) * 2.0)) 
    v["i15"] = 0.100000*np.tanh(((((data["t_sum_act"]) - ((5.0)))) * ((((4.0)) - (((((data["t_sum_act"]) - ((5.0)))) * 2.0)))))) 
    v["i16"] = 0.100000*np.tanh(((((((((np.where(((-3.0) + (data["visitNumber"]))<0, data["t_sum_act"], data["pred_7"] )) - ((4.87859916687011719)))) * 2.0)) * 2.0)) * 2.0)) 
    v["i17"] = 0.100000*np.tanh(((((((((((((((((data["t_mean"]) * 2.0)) - ((8.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) + (data["pred_4"]))) * 2.0)) 
    v["i18"] = 0.100000*np.tanh(((((((data["t_mean"]) + (((((data["t_sum_log"]) + (((((data["t_sum_log"]) - ((7.0)))) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) 
    v["i19"] = 0.100000*np.tanh(((((3.0) - (data["t_median"]))) + (((((data["pred_6"]) - (np.where(data["t_median"]<0, 3.0, (5.0) )))) * 2.0)))) 
    v["i20"] = 0.100000*np.tanh(((((np.where(((data["t_mean"]) - ((6.0)))>0, (4.0), data["t_sum_log"] )) - ((4.0)))) * ((8.0)))) 
    v["i21"] = 0.100000*np.tanh(((-3.0) + (np.where(((data["t_median"]) - ((4.61104393005371094)))<0, ((data["t_mean"]) - (3.0)), 3.0 )))) 
    v["i22"] = 0.100000*np.tanh(((np.where((((-3.0) + (data["t_sum_log"]))/2.0)>0, (((-3.0) + ((7.15739059448242188)))/2.0), (-1.0*(((7.15739059448242188)))) )) - (data["visitNumber"]))) 
    v["i23"] = 0.100000*np.tanh(((data["t_sum_act"]) - (((np.maximum(((data["totals_pageviews"])), ((np.maximum(((np.maximum(((data["pred_2"])), ((((data["visitNumber"]) / 2.0)))))), ((2.0))))))) * 2.0)))) 
    v["i24"] = 0.100000*np.tanh(np.where(((-3.0) + (data["t_sum_act"]))<0, ((data["t_sum_act"]) - ((9.0))), (((data["t_nb_sess"]) < (-3.0))*1.) )) 
    v["i25"] = 0.100000*np.tanh(((data["t_sum_act"]) - (np.maximum(((np.where((((-3.0) + (data["t_sum_act"]))/2.0)<0, (7.0), data["pred_3"] ))), ((data["t_sum_act"])))))) 
    v["i26"] = 0.100000*np.tanh(((((((data["t_sum_log"]) - (3.0))) * ((((((((data["t_mean"]) - (3.0))) * 2.0)) < (3.0))*1.)))) * 2.0)) 
    v["i27"] = 0.100000*np.tanh(np.minimum((((((data["t_sum_act"]) < ((5.0)))*1.))), ((np.minimum(((((data["t_sum_act"]) - ((4.0))))), (((((4.0)) - (data["pred_1"]))))))))) 
    v["i28"] = 0.100000*np.tanh(((((((data["t_sum_log"]) - ((13.52637577056884766)))) * ((((data["t_mean"]) < (3.0))*1.)))) * ((((data["t_sum_log"]) < (3.0))*1.)))) 
    v["i29"] = 0.100000*np.tanh((((((1.0) > (((-3.0) + (data["t_sum_act"]))))*1.)) * (((2.0) * (((-3.0) + (data["t_sum_log"]))))))) 
    v["i30"] = 0.100000*np.tanh((((((data["t_sum_log"]) < (((np.tanh((np.maximum(((data["totals_hits"])), ((3.0)))))) * 2.0)))*1.)) * (((data["t_sum_act"]) - ((7.80285549163818359)))))) 
    v["i31"] = 0.100000*np.tanh(np.where(((-2.0) + (data["t_sum_log"]))>0, (((((-2.0) + (data["t_sum_log"]))) > (data["t_sum_act"]))*1.), (-1.0*(((8.0)))) )) 
    v["i32"] = 0.100000*np.tanh(((np.where(((((data["t_sum_act"]) - (2.0))) * (data["pred_6"]))>0, -3.0, (((data["pred_6"]) > (-3.0))*1.) )) * 2.0)) 
    v["i33"] = 0.100000*np.tanh((((((2.0) > (data["t_sum_log"]))*1.)) * ((((((2.0) > (data["t_sum_act"]))*1.)) * (((((-3.0) * 2.0)) * 2.0)))))) 
    v["i34"] = 0.100000*np.tanh(np.minimum(((np.minimum(((((-3.0) + (np.where(data["t_mean"]>0, data["t_mean"], -3.0 ))))), ((0.0))))), ((data["t_median"])))) 
    v["i35"] = 0.100000*np.tanh(((np.minimum((((((data["t_sum_log"]) < (3.0))*1.))), ((np.minimum(((((((data["t_sum_act"]) + (-2.0))) * 2.0))), ((data["t_median"]))))))) * 2.0)) 
    v["i36"] = 0.100000*np.tanh(np.minimum(((0.0)), (((((data["geoNetwork_metro"]) + (((3.0) * ((((((((-2.0) + (data["t_sum_log"]))/2.0)) * 2.0)) * 2.0)))))/2.0))))) 
    v["i37"] = 0.100000*np.tanh(np.where(data["t_mean"]<0, (-1.0*(((14.43007755279541016)))), (((data["pred_3"]) > (np.where(data["pred_2"]<0, data["visitNumber"], data["t_mean"] )))*1.) )) 
    v["i38"] = 0.100000*np.tanh(np.minimum((((((((data["t_sum_log"]) + (-3.0))) < (data["pred_2"]))*1.))), ((((data["t_sum_log"]) + (((data["t_sum_log"]) + (-3.0)))))))) 
    v["i39"] = 0.100000*np.tanh(((np.minimum(((data["t_median"])), ((((data["t_sum_act"]) - (np.maximum(((np.maximum(((2.0)), ((data["totals_hits"]))))), ((data["t_sum_act"]))))))))) * 2.0)) 
    v["i40"] = 0.100000*np.tanh(((np.where(((data["t_sum_act"]) - (1.0))<0, -3.0, ((((((8.0)) - (data["t_sum_log"]))) < (-3.0))*1.) )) * 2.0)) 
    v["i41"] = 0.100000*np.tanh(np.where(np.where(data["geoNetwork_continent"]<0, data["pred_4"], ((data["t_sum_log"]) + (((data["t_sum_log"]) + (-3.0)))) )>0, 0.0, -3.0 )) 
    v["i42"] = 0.100000*np.tanh(((((data["pred_5"]) * 2.0)) * (np.where((((3.0) > (data["pred_4"]))*1.)>0, (((data["t_sum_log"]) < (1.0))*1.), data["pred_4"] )))) 
    v["i43"] = 0.100000*np.tanh((((((((np.tanh((((data["t_sum_log"]) * (data["t_sum_log"]))))) * (data["geoNetwork_continent"]))) < (((1.0) / 2.0)))*1.)) * (data["pred_5"]))) 
    v["i44"] = 0.100000*np.tanh(((np.where((((-1.0) + (data["t_mean"]))/2.0)>0, np.where(data["pred_5"]<0, 0.0, data["geoNetwork_metro"] ), data["pred_1"] )) * ((10.0)))) 
    v["i45"] = 0.100000*np.tanh(np.where(data["t_sum_log"]<0, -3.0, (((-2.0) + (np.where(((data["totals_hits"]) - (data["totals_pageviews"]))<0, 3.0, -3.0 )))/2.0) )) 
    v["i46"] = 0.100000*np.tanh(((np.where(((data["t_sum_act"]) + (np.tanh((-3.0))))>0, (((data["totals_hits"]) > ((4.0)))*1.), (-1.0*(((4.0)))) )) * 2.0)) 
    v["i47"] = 0.100000*np.tanh(((-3.0) + (np.where(data["geoNetwork_subContinent"]>0, np.where(data["t_mean"]>0, 3.0, -3.0 ), -3.0 )))) 
    v["i48"] = 0.100000*np.tanh(np.where(np.where(data["t_mean"]<0, data["pred_2"], data["t_sum_log"] )<0, -3.0, (((np.maximum(((data["pred_3"])), ((data["pred_2"])))) > (data["t_median"]))*1.) )) 
    v["i49"] = 0.100000*np.tanh(((-3.0) * ((((data["t_sum_log"]) < ((((data["visitNumber"]) + (np.maximum(((data["t_mean"])), ((((3.0) / 2.0))))))/2.0)))*1.)))) 
    v["i50"] = 0.100000*np.tanh(np.where(((((data["t_sum_act"]) * 2.0)) + (-1.0))<0, -3.0, (((3.0) > (((data["t_sum_act"]) + (-1.0))))*1.) )) 
    v["i51"] = 0.100000*np.tanh(np.where((((((data["t_sum_act"]) * 2.0)) + (-1.0))/2.0)<0, ((-1.0) - ((10.0))), (((data["geoNetwork_continent"]) + (-1.0))/2.0) )) 
    v["i52"] = 0.100000*np.tanh(((np.where((((data["t_sum_act"]) + (np.tanh((data["pred_1"]))))/2.0)<0, -3.0, (((data["t_sum_act"]) < (data["pred_1"]))*1.) )) * 2.0)) 
    v["i53"] = 0.100000*np.tanh(((np.minimum(((np.where(data["t_mean"]<0, -3.0, (((data["t_mean"]) < (data["t_sum_log"]))*1.) ))), (((((data["t_mean"]) < (data["totals_pageviews"]))*1.))))) * 2.0)) 
    v["i54"] = 0.100000*np.tanh(((np.minimum(((((((data["t_sum_act"]) * 2.0)) + (-3.0)))), (((((data["t_sum_act"]) < (((data["t_sum_log"]) + (-3.0))))*1.))))) * 2.0)) 
    v["i55"] = 0.100000*np.tanh((((((-1.0*((np.maximum(((data["trafficSource_referralPath"])), (((((data["t_sum_log"]) < ((-1.0*((((data["visitNumber"]) * (data["trafficSource_referralPath"])))))))*1.)))))))) * 2.0)) * 2.0)) 
    v["i56"] = 0.100000*np.tanh((((9.97595596313476562)) * ((-1.0*(((((0.0) > (((data["t_sum_act"]) + (np.where(data["visitNumber"]>0, data["pred_1"], data["visitNumber"] )))))*1.))))))) 
    v["i57"] = 0.100000*np.tanh(np.where(data["t_mean"]<0, data["pred_3"], (((((np.where(data["pred_2"]<0, data["t_mean"], ((data["pred_2"]) * 2.0) )) < (data["t_sum_act"]))*1.)) / 2.0) )) 
    v["i58"] = 0.099951*np.tanh((-1.0*((((((((-1.0*((data["pred_4"])))) + (data["t_sum_act"]))/2.0)) * ((((data["t_sum_act"]) < (((data["t_mean"]) * (data["pred_5"]))))*1.))))))) 
    v["i59"] = 0.099853*np.tanh(((np.minimum(((np.where(data["t_mean"]>0, (((data["t_sum_log"]) < (2.0))*1.), -3.0 ))), (((-1.0*((data["trafficSource_referralPath"]))))))) * 2.0)) 
    v["i60"] = 0.099511*np.tanh((((((((((data["visitNumber"]) - (-3.0))) / 2.0)) > (data["t_sum_log"]))*1.)) * (-3.0))) 
    v["i61"] = 0.100000*np.tanh(((np.where(data["t_sum_act"]<0, -3.0, (((((data["geoNetwork_subContinent"]) > (np.tanh((((data["geoNetwork_subContinent"]) * 2.0)))))*1.)) * (-3.0)) )) * 2.0)) 
    v["i62"] = 0.099951*np.tanh(np.where(data["t_mean"]<0, -3.0, (((-1.0*((((((-1.0*((np.tanh((np.tanh((data["t_sum_act"])))))))) > (data["geoNetwork_metro"]))*1.))))) * 2.0) )) 
    v["i63"] = 0.100000*np.tanh(np.where(((((data["totals_hits"]) * 2.0)) - (((data["t_median"]) / 2.0)))<0, ((data["pred_2"]) * 2.0), (((data["t_sum_log"]) > ((9.24615192413330078)))*1.) )) 
    v["i64"] = 0.100000*np.tanh(((np.where(data["t_sum_log"]<0, -3.0, np.minimum(((((data["totals_pageviews"]) - (data["totals_hits"])))), ((((data["totals_hits"]) - (data["totals_pageviews"]))))) )) * 2.0)) 
    v["i65"] = 0.099756*np.tanh(np.minimum(((data["totals_pageviews"])), ((np.minimum(((np.tanh((np.tanh((data["geoNetwork_subContinent"])))))), (((((data["trafficSource_referralPath"]) + ((((data["geoNetwork_continent"]) > (data["geoNetwork_subContinent"]))*1.)))/2.0)))))))) 
    v["i66"] = 0.099951*np.tanh(((np.where(data["t_sum_act"]<0, data["pred_1"], (((-1.0*(((((data["geoNetwork_metro"]) < ((-1.0*((np.tanh((data["geoNetwork_continent"])))))))*1.))))) * 2.0) )) * 2.0)) 
    v["i67"] = 0.099756*np.tanh(((data["visitNumber"]) * ((-1.0*(((((data["channelGrouping"]) > (np.minimum(((np.minimum(((data["trafficSource_referralPath"])), ((0.0))))), ((((data["visitNumber"]) * 2.0))))))*1.))))))) 
    v["i68"] = 0.100000*np.tanh(np.where(data["t_sum_act"]<0, ((((-1.0) * 2.0)) * 2.0), (((data["pred_4"]) > ((((((data["geoNetwork_metro"]) * 2.0)) + (data["pred_1"]))/2.0)))*1.) )) 
    v["i69"] = 0.099902*np.tanh(np.where(data["t_sum_log"]<0, data["pred_3"], (((((((data["t_mean"]) < (data["totals_hits"]))*1.)) * (data["totals_hits"]))) * (data["visitNumber"])) )) 
    v["i70"] = 0.100000*np.tanh((((((data["t_nb_sess"]) < ((-1.0*((((((((data["geoNetwork_metro"]) > (data["geoNetwork_continent"]))*1.)) + (data["t_sum_log"]))/2.0))))))*1.)) * (data["t_sum_log"]))) 
    v["i71"] = 0.099902*np.tanh(np.minimum((((((((data["totals_pageviews"]) < ((2.0)))*1.)) * (((data["t_median"]) * 2.0))))), ((((((2.0)) < (((data["t_median"]) * 2.0)))*1.))))) 
    v["i72"] = 0.099805*np.tanh(np.minimum(((data["t_sum_act"])), (((((((data["t_mean"]) < ((2.48734521865844727)))*1.)) * (np.where(data["t_mean"]<0, data["pred_3"], ((data["totals_pageviews"]) / 2.0) ))))))) 
    v["i73"] = 0.099902*np.tanh((-1.0*((np.where(data["pred_3"]<0, 0.0, np.maximum(((data["pred_2"])), (((((((data["totals_pageviews"]) > (data["pred_3"]))*1.)) * (data["pred_3"]))))) ))))) 
    v["i74"] = 0.100000*np.tanh(((((((np.where(data["t_mean"]>0, (((data["geoNetwork_continent"]) < ((((data["totals_hits"]) + (-3.0))/2.0)))*1.), -3.0 )) * 2.0)) * 2.0)) * 2.0)) 
    v["i75"] = 0.100000*np.tanh(np.where(data["geoNetwork_subContinent"]>0, (((((data["pred_0"]) > (((data["totals_hits"]) - (data["pred_3"]))))*1.)) * ((-1.0*((data["trafficSource_referralPath"]))))), data["pred_1"] ))
    return Output(v.sum(axis=1))

def GP4(data):
    v = pd.DataFrame()
    v["i0"] = 0.100000*np.tanh((((((((((((data["t_sum_act"]) + (data["t_sum_log"]))/2.0)) - ((7.28741741180419922)))) * 2.0)) * 2.0)) * 2.0)) 
    v["i1"] = 0.100000*np.tanh(((np.maximum(((-2.0)), ((data["pred_4"])))) + (((((((((data["t_mean"]) - (3.0))) - ((4.06806802749633789)))) * 2.0)) * 2.0)))) 
    v["i2"] = 0.100000*np.tanh(((((((((((((((((((data["t_sum_log"]) - ((8.19811630249023438)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i3"] = 0.100000*np.tanh(((np.minimum(((((data["t_sum_log"]) - (data["visitNumber"])))), ((((data["t_sum_log"]) - ((7.0))))))) + (((data["t_sum_act"]) - ((7.0)))))) 
    v["i4"] = 0.100000*np.tanh((((((((((((((((data["t_sum_act"]) + ((-1.0*(((7.0))))))/2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - (-3.0))) 
    v["i5"] = 0.100000*np.tanh(np.minimum(((((data["t_mean"]) + (((data["t_sum_log"]) - ((12.75276947021484375))))))), ((np.minimum(((data["t_median"])), ((((data["t_mean"]) + (data["pred_1"]))))))))) 
    v["i6"] = 0.100000*np.tanh(((((((((((-1.0) + (((((((data["t_sum_log"]) - ((9.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i7"] = 0.100000*np.tanh(((((((((data["t_sum_log"]) - ((7.04534244537353516)))) - (np.tanh(((((data["t_sum_log"]) + (data["pred_9"]))/2.0)))))) * 2.0)) * 2.0)) 
    v["i8"] = 0.100000*np.tanh(((((((((((data["pred_1"]) + (((data["pred_2"]) + (np.minimum(((-3.0)), ((data["pred_6"])))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i9"] = 0.100000*np.tanh((((((((((4.59890556335449219)) + (((((((((data["t_median"]) - ((8.0)))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) 
    v["i10"] = 0.100000*np.tanh(((((((np.maximum(((data["pred_3"])), ((np.maximum(((data["pred_4"])), ((((data["pred_1"]) + (data["pred_2"]))))))))) + (-3.0))) * 2.0)) * 2.0)) 
    v["i11"] = 0.100000*np.tanh(((((((((((((((data["t_sum_log"]) - ((((4.0)) * 2.0)))) * 2.0)) + (data["t_mean"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i12"] = 0.100000*np.tanh(((((((data["pred_10"]) + (((((data["t_mean"]) - (((data["visitNumber"]) / 2.0)))) * 2.0)))) + (data["pred_3"]))) * 2.0)) 
    v["i13"] = 0.100000*np.tanh(((((data["t_mean"]) + (np.minimum(((((((((data["t_sum_log"]) - ((6.0)))) * 2.0)) + (-3.0)))), ((-3.0)))))) * 2.0)) 
    v["i14"] = 0.100000*np.tanh(((((data["t_sum_act"]) + (((np.minimum(((-3.0)), (((((-1.0*((data["visitNumber"])))) * (data["visitNumber"])))))) + (-2.0))))) * 2.0)) 
    v["i15"] = 0.100000*np.tanh(np.minimum(((((data["pred_3"]) + (((data["t_mean"]) - ((4.77416133880615234))))))), (((((5.0)) + ((-1.0*((data["visitNumber"]))))))))) 
    v["i16"] = 0.100000*np.tanh(np.minimum(((((((data["t_sum_act"]) - ((4.75877285003662109)))) * 2.0))), (((((((((data["t_sum_act"]) - ((4.75877285003662109)))) * 2.0)) < ((4.75877285003662109)))*1.))))) 
    v["i17"] = 0.100000*np.tanh(np.minimum((((-1.0*(((((data["pred_2"]) > ((((6.0)) / 2.0)))*1.)))))), ((((data["t_mean"]) - ((6.0))))))) 
    v["i18"] = 0.100000*np.tanh(np.where(((data["t_sum_log"]) - ((4.0)))>0, ((((5.0)) > (data["t_mean"]))*1.), ((data["pred_8"]) - ((5.0))) )) 
    v["i19"] = 0.100000*np.tanh((-1.0*(((((((((((data["t_mean"]) < ((4.0)))*1.)) * 2.0)) * 2.0)) * 2.0))))) 
    v["i20"] = 0.100000*np.tanh(np.minimum((((((4.56055736541748047)) - (data["pred_1"])))), ((np.minimum(((((data["t_sum_log"]) - ((4.56055355072021484))))), (((((data["pred_4"]) > (data["pred_1"]))*1.)))))))) 
    v["i21"] = 0.100000*np.tanh(np.minimum((((((((((data["t_mean"]) * 2.0)) - ((9.0)))) < (3.0))*1.))), ((((((data["t_mean"]) * 2.0)) - ((8.0))))))) 
    v["i22"] = 0.100000*np.tanh(((np.minimum((((6.0))), ((np.where((6.0)>0, ((data["t_sum_log"]) * (((data["t_sum_log"]) / 2.0))), data["t_sum_log"] ))))) - ((6.0)))) 
    v["i23"] = 0.100000*np.tanh(np.minimum((((((data["t_sum_log"]) > ((10.95762443542480469)))*1.))), ((((-3.0) - ((((10.95762443542480469)) * ((((3.0)) - (data["t_sum_log"])))))))))) 
    v["i24"] = 0.100000*np.tanh(((((np.minimum(((0.0)), ((((((data["t_sum_act"]) + (((data["t_sum_act"]) - ((6.0)))))) - (data["visitNumber"])))))) * 2.0)) * 2.0)) 
    v["i25"] = 0.100000*np.tanh(((np.minimum(((data["t_mean"])), ((((np.minimum((((((data["t_mean"]) < ((4.22032451629638672)))*1.))), ((((-3.0) + (data["t_sum_log"])))))) * 2.0))))) * 2.0)) 
    v["i26"] = 0.100000*np.tanh(((np.where(((-3.0) + (data["t_sum_log"]))<0, ((((data["pred_2"]) + (-2.0))) + (-2.0)), 0.0 )) * 2.0)) 
    v["i27"] = 0.100000*np.tanh(np.minimum(((((((data["t_sum_log"]) - ((4.0)))) - (((data["visitNumber"]) / 2.0))))), (((((data["t_sum_act"]) < (data["visitNumber"]))*1.))))) 
    v["i28"] = 0.100000*np.tanh(np.minimum(((((((((data["t_sum_act"]) - (3.0))) * 2.0)) * 2.0))), ((np.minimum(((data["t_mean"])), ((((data["t_sum_act"]) - (data["t_mean"]))))))))) 
    v["i29"] = 0.100000*np.tanh(np.where(((-3.0) + (data["t_median"]))<0, ((data["t_sum_log"]) + (((((data["t_sum_log"]) * 2.0)) - ((8.0))))), 0.0 )) 
    v["i30"] = 0.100000*np.tanh(np.where(data["pred_1"]>0, data["t_nb_sess"], np.minimum(((((((data["pred_0"]) - ((8.80866241455078125)))) * 2.0))), ((data["geoNetwork_metro"]))) )) 
    v["i31"] = 0.100000*np.tanh(np.where((((2.0) < (data["t_sum_log"]))*1.)>0, (((((5.0)) - (data["totals_hits"]))) - (data["visitNumber"])), data["pred_5"] )) 
    v["i32"] = 0.100000*np.tanh(np.where((((2.0) < (((data["t_sum_act"]) / 2.0)))*1.)>0, 0.0, ((((-3.0) + (data["t_sum_act"]))) * 2.0) )) 
    v["i33"] = 0.100000*np.tanh(((np.where(np.where(((data["t_sum_act"]) - (2.0))>0, data["geoNetwork_continent"], data["pred_5"] )>0, (6.0), data["pred_1"] )) - ((6.0)))) 
    v["i34"] = 0.100000*np.tanh(np.minimum((((((data["pred_4"]) > (-3.0))*1.))), ((((((((((data["t_sum_act"]) * 2.0)) - (3.0))) * 2.0)) - (data["t_sum_act"])))))) 
    v["i35"] = 0.100000*np.tanh(np.minimum(((((((((data["t_sum_act"]) - (3.0))) * 2.0)) + (data["t_nb_sess"])))), (((((data["pred_5"]) > ((0.0)))*1.))))) 
    v["i36"] = 0.100000*np.tanh(((np.minimum(((((((-3.0) + (((data["t_sum_act"]) * 2.0)))) * ((((2.0) > (data["t_sum_act"]))*1.))))), ((data["t_median"])))) * 2.0)) 
    v["i37"] = 0.100000*np.tanh((((((((((data["t_sum_log"]) < (3.0))*1.)) * 2.0)) * ((((((data["t_sum_log"]) * 2.0)) + (-3.0))/2.0)))) * 2.0)) 
    v["i38"] = 0.100000*np.tanh(np.minimum(((data["totals_pageviews"])), (((((((((((1.0) > (data["t_median"]))*1.)) * (data["pred_2"]))) * (data["pred_3"]))) * (data["pred_5"])))))) 
    v["i39"] = 0.100000*np.tanh(((((((data["pred_5"]) * ((((2.0) > (data["t_sum_act"]))*1.)))) * ((11.94209194183349609)))) * ((((2.0) > (data["t_sum_log"]))*1.)))) 
    v["i40"] = 0.100000*np.tanh(np.minimum(((((((((data["pred_6"]) * 2.0)) * 2.0)) * ((((((data["t_sum_act"]) < (((1.0) * 2.0)))*1.)) * 2.0))))), ((data["t_median"])))) 
    v["i41"] = 0.100000*np.tanh(np.where(((((data["t_nb_sess"]) + (((data["t_sum_log"]) * 2.0)))) + (-3.0))<0, -3.0, (((-2.0) > (data["t_nb_sess"]))*1.) )) 
    v["i42"] = 0.100000*np.tanh(np.where(((data["t_sum_log"]) - (1.0))<0, ((data["t_sum_log"]) - ((12.49324989318847656))), ((data["geoNetwork_continent"]) - (np.tanh((data["totals_hits"])))) )) 
    v["i43"] = 0.100000*np.tanh(((np.minimum(((0.0)), ((((((((((data["t_sum_act"]) * 2.0)) - (2.0))) * 2.0)) * 2.0))))) * 2.0)) 
    v["i44"] = 0.100000*np.tanh(np.where(((data["t_sum_act"]) - (1.0))<0, ((-3.0) * 2.0), ((np.tanh(((-1.0*((data["trafficSource_referralPath"])))))) / 2.0) )) 
    v["i45"] = 0.100000*np.tanh(((-3.0) * (np.where(data["t_mean"]>0, ((((((data["t_sum_log"]) < ((9.68791198730468750)))*1.)) < (data["pred_5"]))*1.), (9.68791198730468750) )))) 
    v["i46"] = 0.099951*np.tanh(((((((((-1.0) + (np.minimum(((np.tanh((((data["totals_pageviews"]) * 2.0))))), ((data["t_sum_act"])))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i47"] = 0.100000*np.tanh((((((((-1.0*(((((((((data["t_sum_act"]) * 2.0)) + (-1.0))) < (data["pred_0"]))*1.))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i48"] = 0.099902*np.tanh(((data["pred_4"]) * (((((((((data["t_sum_act"]) - (data["visitNumber"]))) > (data["pred_4"]))*1.)) > (((data["t_mean"]) - (data["visitNumber"]))))*1.)))) 
    v["i49"] = 0.099756*np.tanh(((((-3.0) * 2.0)) * ((((np.tanh((data["t_sum_act"]))) < ((((np.minimum(((data["totals_pageviews"])), ((data["totals_hits"])))) < (data["totals_hits"]))*1.)))*1.)))) 
    v["i50"] = 0.099951*np.tanh(np.minimum(((np.where(data["t_mean"]>0, (((-3.0) < (data["pred_1"]))*1.), -3.0 ))), ((np.where(data["geoNetwork_subContinent"]>0, data["geoNetwork_subContinent"], data["pred_1"] ))))) 
    v["i51"] = 0.100000*np.tanh(np.where(data["t_mean"]<0, -3.0, (((np.maximum((((3.0))), ((((data["t_sum_log"]) - (data["totals_pageviews"])))))) > (data["t_mean"]))*1.) )) 
    v["i52"] = 0.100000*np.tanh((((((-1.0*(((((data["t_sum_log"]) < (((np.maximum(((((data["visitNumber"]) * 2.0))), ((data["t_sum_act"])))) / 2.0)))*1.))))) * 2.0)) * 2.0)) 
    v["i53"] = 0.100000*np.tanh(((np.where(np.minimum(((data["geoNetwork_subContinent"])), ((data["geoNetwork_subContinent"])))<0, data["pred_3"], (((-1.0*(((((data["geoNetwork_subContinent"]) > (data["t_sum_log"]))*1.))))) * 2.0) )) * 2.0)) 
    v["i54"] = 0.099658*np.tanh((-1.0*((np.where(data["t_sum_log"]<0, (7.95942497253417969), np.where(data["pred_3"]<0, (((data["t_sum_log"]) > ((7.95942497253417969)))*1.), data["pred_2"] ) ))))) 
    v["i55"] = 0.099951*np.tanh((-1.0*((np.where(data["t_mean"]<0, (6.22654199600219727), ((((((data["geoNetwork_metro"]) > (0.0))*1.)) > (((data["geoNetwork_continent"]) + (data["geoNetwork_metro"]))))*1.) ))))) 
    v["i56"] = 0.099951*np.tanh(((((((np.where(data["t_mean"]>0, ((data["visitNumber"]) * ((((data["totals_hits"]) > (data["t_mean"]))*1.))), -3.0 )) * 2.0)) * 2.0)) * 2.0)) 
    v["i57"] = 0.099951*np.tanh((((((-1.0*((np.where(data["t_mean"]>0, ((((((data["geoNetwork_continent"]) > (data["geoNetwork_subContinent"]))*1.)) < (data["geoNetwork_continent"]))*1.), (9.77862071990966797) ))))) * 2.0)) * 2.0)) 
    v["i58"] = 0.099609*np.tanh((((((-1.0*(((((np.tanh((data["geoNetwork_subContinent"]))) < ((((data["t_mean"]) < (((data["t_mean"]) * (data["geoNetwork_subContinent"]))))*1.)))*1.))))) * 2.0)) * 2.0)) 
    v["i59"] = 0.100000*np.tanh(((np.where(data["geoNetwork_country"]>0, ((np.where(data["t_sum_act"]>0, (((data["pred_3"]) < (data["pred_5"]))*1.), data["pred_5"] )) * 2.0), data["pred_3"] )) * 2.0)) 
    v["i60"] = 0.099951*np.tanh(np.where((((data["pred_0"]) < (((data["t_nb_sess"]) / 2.0)))*1.)>0, data["pred_4"], (((((data["pred_0"]) < (data["pred_4"]))*1.)) * (-3.0)) )) 
    v["i61"] = 0.100000*np.tanh(np.where(data["t_sum_log"]<0, -3.0, (((np.maximum(((data["pred_3"])), ((3.0)))) < (((data["totals_hits"]) + (np.tanh((-2.0))))))*1.) )) 
    v["i62"] = 0.100000*np.tanh(((((np.where(data["t_sum_log"]<0, data["pred_3"], (-1.0*(((((((data["trafficSource_referralPath"]) * (data["t_sum_log"]))) > (data["visitNumber"]))*1.)))) )) * 2.0)) * 2.0)) 
    v["i63"] = 0.100000*np.tanh(np.where(data["t_sum_act"]<0, ((data["totals_hits"]) + (-3.0)), (((data["t_median"]) < (((((data["totals_pageviews"]) + (-2.0))) * 2.0)))*1.) )) 
    v["i64"] = 0.095457*np.tanh(np.where(data["t_mean"]<0, (-1.0*(((9.0)))), (((data["t_mean"]) < (np.where(data["t_sum_act"]>0, (((9.0)) / 2.0), data["t_mean"] )))*1.) )) 
    v["i65"] = 0.100000*np.tanh(np.minimum(((data["t_median"])), (((-1.0*((np.where(data["t_sum_log"]<0, 3.0, (((data["geoNetwork_metro"]) > (((data["t_mean"]) / 2.0)))*1.) )))))))) 
    v["i66"] = 0.100000*np.tanh(np.where(((data["geoNetwork_continent"]) - (data["geoNetwork_subContinent"]))<0, -3.0, np.where(data["t_sum_act"]<0, -3.0, (((data["totals_pageviews"]) > (data["t_mean"]))*1.) ) )) 
    v["i67"] = 0.100000*np.tanh(((-3.0) * (((((((np.tanh((((((data["totals_pageviews"]) - (data["totals_hits"]))) * 2.0)))) > (data["geoNetwork_country"]))*1.)) > (data["geoNetwork_country"]))*1.)))) 
    v["i68"] = 0.099951*np.tanh(np.where((((data["pred_0"]) + (data["t_sum_act"]))/2.0)<0, -3.0, ((np.tanh(((((((data["pred_0"]) < (data["t_median"]))*1.)) / 2.0)))) / 2.0) )) 
    v["i69"] = 0.100000*np.tanh((((((data["t_sum_log"]) < (((np.maximum(((data["t_nb_sess"])), ((data["totals_hits"])))) * 2.0)))*1.)) * ((-1.0*(((((data["totals_pageviews"]) < (data["totals_hits"]))*1.))))))) 
    v["i70"] = 0.100000*np.tanh(((((np.where(data["geoNetwork_continent"]<0, data["visitNumber"], (((np.tanh((data["t_sum_act"]))) > ((((data["totals_hits"]) + (data["visitNumber"]))/2.0)))*1.) )) * 2.0)) * 2.0)) 
    v["i71"] = 0.099902*np.tanh(((-3.0) * ((((((data["geoNetwork_metro"]) / 2.0)) > (np.where(data["t_mean"]>0, ((np.tanh((data["totals_hits"]))) * 2.0), data["t_mean"] )))*1.)))) 
    v["i72"] = 0.100000*np.tanh(((((data["t_mean"]) * ((((data["t_nb_sess"]) < (((np.maximum(((((data["t_sum_log"]) / 2.0))), ((data["pred_2"])))) - (data["t_sum_log"]))))*1.)))) * 2.0)) 
    v["i73"] = 0.100000*np.tanh(np.where(data["pred_0"]<0, (((-1.0*(((((data["pred_3"]) > ((5.0)))*1.))))) * 2.0), (-1.0*(((((data["pred_0"]) < (data["visitNumber"]))*1.)))) )) 
    v["i74"] = 0.083586*np.tanh(np.minimum(((data["t_sum_act"])), (((((data["geoNetwork_metro"]) > (np.maximum(((((data["pred_1"]) / 2.0))), ((np.maximum((((1.40993750095367432))), ((data["visitNumber"]))))))))*1.)))))
    return Output(v.sum(axis=1))

def GP5(data):
    v = pd.DataFrame()
    v["i0"] = 0.100000*np.tanh(((((((data["t_sum_act"]) + (((data["t_sum_act"]) - ((9.0)))))) + (((((data["t_sum_log"]) - ((10.0)))) * 2.0)))) * 2.0)) 
    v["i1"] = 0.100000*np.tanh(((data["t_sum_act"]) - ((((6.94758796691894531)) - (((data["t_sum_log"]) + (((((-2.0) * 2.0)) * 2.0)))))))) 
    v["i2"] = 0.100000*np.tanh(((((((((data["t_sum_act"]) + (((3.0) - ((10.0)))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i3"] = 0.100000*np.tanh(((((np.tanh((np.tanh((data["pred_3"]))))) + (((((data["t_sum_act"]) * 2.0)) - ((13.52748107910156250)))))) * 2.0)) 
    v["i4"] = 0.100000*np.tanh(((((((data["t_sum_log"]) - (np.maximum((((10.0))), ((data["t_median"])))))) + (((data["t_sum_log"]) / 2.0)))) + (-3.0))) 
    v["i5"] = 0.100000*np.tanh(((((((((((data["t_mean"]) - ((10.0)))) * 2.0)) + (data["t_sum_log"]))) * 2.0)) * 2.0)) 
    v["i6"] = 0.100000*np.tanh(((np.where(np.where(data["pred_0"]>0, data["pred_2"], ((data["sess_date_dow"]) - (data["pred_0"])) )>0, data["pred_0"], data["t_sum_log"] )) - ((9.53606986999511719)))) 
    v["i7"] = 0.100000*np.tanh(np.minimum(((np.tanh((((data["t_median"]) * 2.0))))), ((((((3.0) * 2.0)) * (((data["t_sum_act"]) - ((6.01647520065307617))))))))) 
    v["i8"] = 0.100000*np.tanh(((((((((((np.minimum(((data["t_sum_log"])), (((11.64070796966552734))))) * 2.0)) - ((11.64070796966552734)))) * 2.0)) - (-1.0))) - (data["visitNumber"]))) 
    v["i9"] = 0.100000*np.tanh(((((np.where(data["t_median"]>0, data["t_sum_log"], ((((((((data["pred_2"]) / 2.0)) * 2.0)) * 2.0)) * 2.0) )) - ((10.57200813293457031)))) * 2.0)) 
    v["i10"] = 0.100000*np.tanh((((((7.02941036224365234)) - (data["t_sum_act"]))) * (((data["t_sum_act"]) - ((((((((7.02941036224365234)) - (data["t_sum_act"]))) * 2.0)) * 2.0)))))) 
    v["i11"] = 0.100000*np.tanh(np.minimum(((data["pred_1"])), (((((((6.0)) / 2.0)) * (((data["t_sum_log"]) - ((6.0))))))))) 
    v["i12"] = 0.100000*np.tanh(((((data["t_sum_act"]) - (np.maximum(((data["t_sum_act"])), ((((((-3.0) * (np.tanh(((1.58075487613677979)))))) * (-2.0)))))))) * 2.0)) 
    v["i13"] = 0.100000*np.tanh(((((data["t_sum_act"]) * 2.0)) - (np.maximum(((((data["visitNumber"]) * 2.0))), ((np.maximum(((((data["t_sum_act"]) * 2.0))), (((10.78281593322753906)))))))))) 
    v["i14"] = 0.100000*np.tanh(np.minimum((((((data["t_sum_act"]) < ((6.0)))*1.))), ((((((np.minimum(((data["sess_date_hours"])), ((((data["t_sum_act"]) - ((5.0))))))) * 2.0)) * 2.0))))) 
    v["i15"] = 0.100000*np.tanh(np.minimum((((((data["pred_1"]) < ((((10.0)) - (data["t_sum_log"]))))*1.))), ((((data["t_sum_log"]) - ((((10.0)) - (data["t_sum_log"])))))))) 
    v["i16"] = 0.100000*np.tanh(np.minimum(((0.0)), ((((data["t_sum_log"]) + (np.minimum(((((-1.0) - (data["visitNumber"])))), ((((data["t_mean"]) - ((9.99547958374023438)))))))))))) 
    v["i17"] = 0.100000*np.tanh(((((-3.0) * 2.0)) + (np.minimum(((np.maximum(((data["pred_0"])), ((data["t_sum_log"]))))), ((((-3.0) * (-2.0)))))))) 
    v["i18"] = 0.100000*np.tanh(((((((np.minimum(((((((5.0)) > (data["t_sum_act"]))*1.))), ((((data["t_sum_act"]) - ((4.49360561370849609))))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i19"] = 0.100000*np.tanh(((np.minimum(((((((((((data["t_mean"]) / 2.0)) + (data["t_sum_log"]))/2.0)) + (((data["t_sum_log"]) * 2.0)))/2.0))), (((6.0))))) - ((6.0)))) 
    v["i20"] = 0.100000*np.tanh(np.minimum(((0.0)), ((((np.where((((data["geoNetwork_networkDomain"]) + ((1.0)))/2.0)<0, 0.0, ((data["t_sum_act"]) * 2.0) )) - ((9.07516670227050781))))))) 
    v["i21"] = 0.100000*np.tanh(np.minimum(((((((data["t_mean"]) - ((3.69601702690124512)))) * ((7.17828702926635742))))), (((((3.0) > (((data["t_sum_act"]) - (3.0))))*1.))))) 
    v["i22"] = 0.100000*np.tanh(((np.minimum(((0.0)), ((((data["t_sum_log"]) + (((((data["t_sum_log"]) + (((-2.0) + (-3.0))))) * 2.0))))))) * 2.0)) 
    v["i23"] = 0.100000*np.tanh(np.minimum(((0.0)), ((np.minimum(((((data["geoNetwork_networkDomain"]) / 2.0))), (((((7.0)) * (((data["t_sum_act"]) - ((((7.0)) / 2.0)))))))))))) 
    v["i24"] = 0.100000*np.tanh(np.minimum(((0.0)), ((((((data["t_sum_log"]) - ((7.0)))) + ((((data["t_sum_log"]) + (data["t_sum_act"]))/2.0))))))) 
    v["i25"] = 0.100000*np.tanh(((np.minimum(((data["t_median"])), ((((((((np.minimum(((0.0)), ((((data["t_sum_log"]) + (-3.0)))))) * 2.0)) * 2.0)) * 2.0))))) * 2.0)) 
    v["i26"] = 0.100000*np.tanh(((np.minimum(((((((((((data["t_sum_act"]) + (-3.0))) * 2.0)) * 2.0)) * 2.0))), ((0.0)))) * ((9.0)))) 
    v["i27"] = 0.100000*np.tanh(((((data["pred_2"]) - ((5.99040746688842773)))) * ((((((np.maximum(((3.0)), ((((data["pred_2"]) * 2.0))))) > (data["t_sum_log"]))*1.)) * 2.0)))) 
    v["i28"] = 0.100000*np.tanh(np.minimum(((((data["sess_date_dow"]) / (-2.0)))), ((((((data["t_sum_log"]) * (data["t_sum_log"]))) - ((7.08554697036743164))))))) 
    v["i29"] = 0.100000*np.tanh(np.minimum(((((((((data["t_sum_log"]) * 2.0)) * 2.0)) - ((9.0))))), (((((((0.0) - (data["pred_0"]))) > (data["geoNetwork_networkDomain"]))*1.))))) 
    v["i30"] = 0.100000*np.tanh(((((((data["t_sum_log"]) + (-3.0))) * ((((2.0) > (((data["t_sum_act"]) / 2.0)))*1.)))) * 2.0)) 
    v["i31"] = 0.100000*np.tanh(np.minimum(((0.0)), ((((data["totals_pageviews"]) - (np.where((((data["t_sum_act"]) + (((-1.0) * 2.0)))/2.0)<0, (9.12072467803955078), data["totals_hits"] ))))))) 
    v["i32"] = 0.100000*np.tanh(np.minimum(((((((((((((((-2.0) + (data["t_sum_log"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0))), ((0.0)))) 
    v["i33"] = 0.100000*np.tanh(np.where((((-2.0) + (data["t_sum_log"]))/2.0)<0, ((data["t_sum_log"]) - ((10.0))), ((-1.0) / ((9.0))) )) 
    v["i34"] = 0.100000*np.tanh(((np.minimum(((data["t_median"])), ((((np.minimum((((((data["pred_2"]) > (data["t_median"]))*1.))), ((((data["t_sum_act"]) + (-2.0)))))) * 2.0))))) * 2.0)) 
    v["i35"] = 0.100000*np.tanh(((((np.where((((-3.0) + (data["t_sum_log"]))/2.0)<0, ((((data["t_sum_log"]) * 2.0)) + (-3.0)), 0.0 )) * 2.0)) * 2.0)) 
    v["i36"] = 0.100000*np.tanh(((np.where(((-3.0) + (data["t_sum_act"]))<0, -3.0, 0.0 )) * 2.0)) 
    v["i37"] = 0.100000*np.tanh(np.minimum((((((data["t_mean"]) < (3.0))*1.))), ((((((((((-3.0) + (((data["t_sum_act"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0))))) 
    v["i38"] = 0.100000*np.tanh(((((np.minimum(((data["pred_1"])), ((np.where(data["pred_2"]>0, data["t_mean"], -3.0 ))))) * 2.0)) * ((((data["t_mean"]) < (2.0))*1.)))) 
    v["i39"] = 0.100000*np.tanh(np.minimum((((0.0))), ((((data["t_median"]) - ((((5.18295526504516602)) - (((data["t_sum_log"]) * (((data["t_sum_act"]) + (data["t_sum_log"])))))))))))) 
    v["i40"] = 0.100000*np.tanh(np.minimum(((((data["totals_pageviews"]) - (((((2.0) / (data["t_mean"]))) / (data["t_mean"])))))), ((0.0)))) 
    v["i41"] = 0.100000*np.tanh(((((((-3.0) * ((((1.0) > (data["t_sum_act"]))*1.)))) * 2.0)) * 2.0)) 
    v["i42"] = 0.100000*np.tanh(((-3.0) * ((((6.85303115844726562)) * ((((data["t_sum_log"]) < (np.maximum(((data["visitNumber"])), ((np.tanh((2.0)))))))*1.)))))) 
    v["i43"] = 0.100000*np.tanh(np.minimum(((((data["t_sum_log"]) - (2.0)))), ((np.minimum(((((2.0) / (data["t_sum_log"])))), ((((data["t_sum_act"]) - (data["t_mean"]))))))))) 
    v["i44"] = 0.100000*np.tanh(np.minimum(((((((2.0) - (data["visitNumber"]))) / (data["t_sum_log"])))), ((((((data["t_sum_log"]) - (data["visitNumber"]))) - (2.0)))))) 
    v["i45"] = 0.099951*np.tanh(np.minimum(((np.minimum(((data["t_sum_act"])), ((((1.0) / (data["t_sum_act"]))))))), ((((data["sess_date_dom"]) + (np.tanh((np.tanh((data["t_sum_log"])))))))))) 
    v["i46"] = 0.100000*np.tanh(np.minimum(((np.minimum(((data["t_mean"])), (((((((3.0) / ((5.0)))) > (data["t_median"]))*1.)))))), (((((3.95915246009826660)) / (data["t_mean"])))))) 
    v["i47"] = 0.100000*np.tanh(np.minimum(((np.minimum(((((data["visitNumber"]) / (data["t_sum_act"])))), ((((data["pred_1"]) / (((data["t_sum_act"]) / (data["visitNumber"]))))))))), ((0.0)))) 
    v["i48"] = 0.100000*np.tanh(np.where(data["t_mean"]>0, ((-1.0) * (np.minimum((((((data["totals_hits"]) > (data["totals_pageviews"]))*1.))), ((np.tanh((data["totals_hits"]))))))), -3.0 )) 
    v["i49"] = 0.100000*np.tanh(np.minimum(((((3.0) / (((data["t_mean"]) * 2.0))))), ((((data["t_sum_act"]) + (((((data["t_sum_act"]) * 2.0)) - (3.0)))))))) 
    v["i50"] = 0.100000*np.tanh(np.where(((((data["t_nb_sess"]) + (data["t_sum_log"]))) + (data["t_sum_log"]))>0, (((-3.0) > (data["t_nb_sess"]))*1.), -3.0 )) 
    v["i51"] = 0.100000*np.tanh(((np.minimum((((((((data["pred_0"]) > (((data["t_sum_log"]) * 2.0)))*1.)) / (data["trafficSource_source"])))), ((((-1.0) + (data["t_sum_act"])))))) * 2.0)) 
    v["i52"] = 0.099902*np.tanh((((((((((data["totals_hits"]) * (data["t_sum_act"]))) < (1.0))*1.)) * (-3.0))) * 2.0)) 
    v["i53"] = 0.100000*np.tanh(np.minimum((((((((2.0) < (data["totals_pageviews"]))*1.)) + (data["t_median"])))), (((((((data["t_median"]) < (2.0))*1.)) / (data["t_mean"])))))) 
    v["i54"] = 0.099805*np.tanh((((((((data["t_sum_act"]) * 2.0)) < (data["pred_0"]))*1.)) * (((((-2.0) * 2.0)) * 2.0)))) 
    v["i55"] = 0.099902*np.tanh(np.minimum((((((data["pred_1"]) > (data["geoNetwork_city"]))*1.))), ((((((((5.19124889373779297)) > (data["t_mean"]))*1.)) / (np.tanh((data["t_mean"])))))))) 
    v["i56"] = 0.100000*np.tanh(np.where((((-1.0) + (((data["t_sum_log"]) * 2.0)))/2.0)>0, ((((data["geoNetwork_networkDomain"]) * (data["geoNetwork_networkDomain"]))) / (-2.0)), -3.0 )) 
    v["i57"] = 0.100000*np.tanh(((np.where(data["t_mean"]>0, (((data["trafficSource_source"]) > (np.tanh((np.tanh((np.tanh((data["totals_pageviews"]))))))))*1.), (7.56566238403320312) )) / (-2.0))) 
    v["i58"] = 0.099951*np.tanh(np.minimum(((((((3.0) * 2.0)) / (data["t_mean"])))), ((np.minimum(((data["t_median"])), (((((-1.0) > (data["sess_date_dom"]))*1.)))))))) 
    v["i59"] = 0.099951*np.tanh(np.minimum(((np.where(data["t_sum_log"]<0, -3.0, ((data["channelGrouping"]) * (np.tanh((data["t_nb_sess"])))) ))), ((((data["channelGrouping"]) * (data["channelGrouping"])))))) 
    v["i60"] = 0.093258*np.tanh(np.where(data["t_mean"]>0, np.minimum(((np.tanh((np.tanh((data["pred_0"])))))), ((((data["geoNetwork_networkDomain"]) / (data["visitNumber"]))))), -3.0 )) 
    v["i61"] = 0.099951*np.tanh(np.minimum(((((((((data["t_sum_log"]) * 2.0)) * 2.0)) - ((4.0))))), ((((((((data["visitNumber"]) + (data["pred_0"]))/2.0)) > ((9.99041843414306641)))*1.))))) 
    v["i62"] = 0.097460*np.tanh((((((data["t_mean"]) < (((-1.0) / (data["pred_1"]))))*1.)) / (((np.tanh((data["t_mean"]))) / 2.0)))) 
    v["i63"] = 0.099951*np.tanh(np.where(((data["geoNetwork_networkDomain"]) + (np.tanh((np.tanh((data["t_sum_log"]))))))>0, ((-1.0) + (np.tanh((data["totals_pageviews"])))), -3.0 )) 
    v["i64"] = 0.093747*np.tanh(((((((data["t_sum_act"]) + (((-3.0) * 2.0)))) * (((((((data["t_sum_log"]) > ((10.0)))*1.)) > (data["sess_date_hours"]))*1.)))) * 2.0)) 
    v["i65"] = 0.100000*np.tanh(np.minimum((((((data["sess_date_hours"]) < (((-2.0) * (data["visitNumber"]))))*1.))), ((((((data["t_sum_log"]) - (data["visitNumber"]))) / (data["geoNetwork_country"])))))) 
    v["i66"] = 0.100000*np.tanh((((((data["geoNetwork_continent"]) < ((((((((((data["trafficSource_referralPath"]) + (data["geoNetwork_subContinent"]))/2.0)) + (data["geoNetwork_subContinent"]))/2.0)) + (1.0))/2.0)))*1.)) * (-3.0))) 
    v["i67"] = 0.100000*np.tanh(np.where(data["t_mean"]>0, np.where(data["geoNetwork_country"]>0, (((((-2.0) / (data["geoNetwork_city"]))) > (data["totals_pageviews"]))*1.), -3.0 ), -3.0 )) 
    v["i68"] = 0.099853*np.tanh(np.minimum(((np.where(data["geoNetwork_country"]>0, np.where(data["t_median"]>0, (((data["t_sum_log"]) < (data["totals_pageviews"]))*1.), data["sess_date_hours"] ), -3.0 ))), ((data["geoNetwork_subContinent"])))) 
    v["i69"] = 0.100000*np.tanh(np.minimum(((0.0)), ((np.minimum(((data["totals_pageviews"])), (((((((((data["totals_pageviews"]) > (data["totals_hits"]))*1.)) + (data["trafficSource_source"]))) * 2.0)))))))) 
    v["i70"] = 0.100000*np.tanh(((np.minimum(((((data["geoNetwork_continent"]) - (data["geoNetwork_country"])))), ((((data["geoNetwork_country"]) - (np.where(data["t_median"]>0, data["geoNetwork_country"], data["trafficSource_source"] ))))))) * 2.0)) 
    v["i71"] = 0.100000*np.tanh(np.where(data["t_sum_log"]>0, np.where(data["geoNetwork_subContinent"]>0, (((data["t_sum_log"]) > ((11.69831085205078125)))*1.), ((((data["trafficSource_source"]) * 2.0)) * 2.0) ), -3.0 )) 
    v["i72"] = 0.091060*np.tanh(np.minimum((((((((data["sess_date_hours"]) < (((data["geoNetwork_networkDomain"]) - ((((-3.0) + (data["t_mean"]))/2.0)))))*1.)) / (data["geoNetwork_networkDomain"])))), ((data["totals_hits"])))) 
    v["i73"] = 0.099951*np.tanh(np.minimum(((((2.0) / (data["geoNetwork_subContinent"])))), (((((((((data["geoNetwork_country"]) > (data["geoNetwork_subContinent"]))*1.)) * (data["t_median"]))) * (data["geoNetwork_city"])))))) 
    v["i74"] = 0.099951*np.tanh(np.minimum((((((((data["t_sum_log"]) - (data["totals_pageviews"]))) > (data["t_mean"]))*1.))), ((((data["t_sum_log"]) * ((10.0))))))) 
    v["i75"] = 0.097557*np.tanh(np.minimum(((((data["t_sum_log"]) * 2.0))), (((((((data["t_median"]) + (((data["trafficSource_source"]) * 2.0)))) < ((((data["t_median"]) > (data["totals_pageviews"]))*1.)))*1.))))) 
    v["i76"] = 0.100000*np.tanh((((((((((((((4.0)) < (data["t_mean"]))*1.)) / 2.0)) / (data["trafficSource_source"]))) + (-1.0))/2.0)) / 2.0)) 
    v["i77"] = 0.099902*np.tanh(((np.minimum(((data["t_median"])), ((((np.minimum(((data["totals_pageviews"])), (((((data["totals_pageviews"]) > ((((8.65321922302246094)) / 2.0)))*1.))))) * ((8.65321922302246094))))))) * 2.0)) 
    v["i78"] = 0.099218*np.tanh(((np.where(data["t_median"]>0, data["channelGrouping"], np.where(data["channelGrouping"]>0, data["t_median"], data["pred_0"] ) )) * (np.minimum(((data["geoNetwork_country"])), ((data["geoNetwork_region"])))))) 
    v["i79"] = 0.099853*np.tanh((((np.where(data["t_median"]<0, 2.0, data["t_mean"] )) < (np.minimum(((data["t_sum_log"])), ((np.minimum(((data["t_sum_log"])), ((data["totals_hits"]))))))))*1.)) 
    v["i80"] = 0.099951*np.tanh(((np.where(data["t_sum_act"]>0, (((2.0) < (((((-2.0) / (data["trafficSource_source"]))) / 2.0)))*1.), 2.0 )) * (-3.0))) 
    v["i81"] = 0.058964*np.tanh((((data["geoNetwork_region"]) > ((((data["geoNetwork_networkDomain"]) < ((((((data["geoNetwork_country"]) * ((11.84922599792480469)))) < (data["t_sum_log"]))*1.)))*1.)))*1.)) 
    v["i82"] = 0.097655*np.tanh(np.minimum(((((data["geoNetwork_country"]) / (((-3.0) + (data["geoNetwork_networkDomain"])))))), ((((((-1.0) / 2.0)) / (data["pred_0"])))))) 
    v["i83"] = 0.080459*np.tanh(np.minimum((((((data["geoNetwork_city"]) < (data["t_median"]))*1.))), ((np.minimum(((data["t_median"])), ((np.minimum(((data["t_sum_act"])), (((((data["t_sum_act"]) > (data["t_mean"]))*1.))))))))))) 
    v["i84"] = 0.099951*np.tanh((((np.where(data["geoNetwork_city"]<0, data["sess_date_dom"], np.where(data["sess_date_dom"]<0, ((data["sess_date_dom"]) / (data["totals_hits"])), data["sess_date_hours"] ) )) < (-1.0))*1.)) 
    v["i85"] = 0.099316*np.tanh((((((data["totals_pageviews"]) < (data["totals_hits"]))*1.)) / (np.minimum(((np.where(data["channelGrouping"]>0, data["totals_hits"], -2.0 ))), ((-1.0)))))) 
    v["i86"] = 0.100000*np.tanh(((((((((((((data["geoNetwork_networkDomain"]) < (0.0))*1.)) + (data["geoNetwork_city"]))) > ((2.0)))*1.)) / (data["geoNetwork_city"]))) / (data["geoNetwork_city"]))) 
    v["i87"] = 0.067904*np.tanh(np.minimum(((((data["t_sum_act"]) - (data["t_mean"])))), (((((1.0)) - (data["geoNetwork_city"])))))) 
    v["i88"] = 0.099902*np.tanh((((((-1.0) + ((((((data["sess_date_hours"]) > (data["geoNetwork_country"]))*1.)) / (data["sess_date_hours"]))))/2.0)) / 2.0)) 
    v["i89"] = 0.099951*np.tanh((((((((np.minimum(((data["pred_0"])), (((((data["pred_0"]) > (data["t_median"]))*1.))))) < (((data["t_median"]) / (data["pred_0"]))))*1.)) / 2.0)) / 2.0)) 
    v["i90"] = 0.099951*np.tanh(((np.where(data["t_sum_act"]>0, (((data["geoNetwork_region"]) > (((np.minimum(((data["sess_date_dom"])), ((data["sess_date_hours"])))) + (data["t_sum_log"]))))*1.), -3.0 )) * 2.0)) 
    v["i91"] = 0.099902*np.tanh(((((np.minimum(((np.minimum(((data["totals_hits"])), (((((data["t_mean"]) < (data["totals_pageviews"]))*1.)))))), ((((data["totals_pageviews"]) - (data["totals_hits"])))))) * 2.0)) * 2.0)) 
    v["i92"] = 0.099951*np.tanh((((((data["t_median"]) > (np.where(data["geoNetwork_networkDomain"]<0, data["geoNetwork_networkDomain"], data["t_sum_act"] )))*1.)) / (((((-3.0) + (data["t_mean"]))) * 2.0)))) 
    v["i93"] = 0.099951*np.tanh((((((((((((((data["geoNetwork_networkDomain"]) / (data["pred_0"]))) + (data["sess_date_dom"]))) + (-1.0))) > (3.0))*1.)) * 2.0)) * 2.0)) 
    v["i94"] = 0.099853*np.tanh(((((((np.minimum(((data["sess_date_dow"])), ((((1.0) / (data["trafficSource_source"])))))) / 2.0)) / 2.0)) / 2.0)) 
    v["i95"] = 0.082169*np.tanh(np.tanh((((((data["sess_date_dow"]) - (data["geoNetwork_country"]))) / (((((data["sess_date_hours"]) * (data["sess_date_hours"]))) - (data["geoNetwork_country"]))))))) 
    v["i96"] = 0.099951*np.tanh((((((data["totals_hits"]) < (((3.0) / 2.0)))*1.)) * (-1.0))) 
    v["i97"] = 0.100000*np.tanh(np.where(data["t_sum_act"]<0, -1.0, ((((((np.minimum(((-1.0)), ((data["sess_date_dow"])))) / 2.0)) + (data["geoNetwork_country"]))) / (data["geoNetwork_networkDomain"])) )) 
    v["i98"] = 0.099951*np.tanh(np.where(data["t_sum_act"]<0, -3.0, ((((((6.35631465911865234)) - ((8.0)))) > (np.where(data["sess_date_dow"]<0, 0.0, data["sess_date_hours"] )))*1.) )) 
    v["i99"] = 0.099951*np.tanh(((np.minimum(((((((((((data["trafficSource_referralPath"]) + (2.0))/2.0)) > (data["totals_hits"]))*1.)) * (data["trafficSource_referralPath"])))), ((data["totals_hits"])))) * 2.0))

    return Output(v.sum(axis=1))
    
def GP6(data):
    v = pd.DataFrame()
    v["i0"] = 0.100000*np.tanh(((((data["t_mean"]) - ((9.82170104980468750)))) + (((data["t_mean"]) - (((((data["t_sum_log"]) - ((9.04127788543701172)))) * (-3.0))))))) 
    v["i1"] = 0.100000*np.tanh((((((14.27411556243896484)) * (((data["t_mean"]) + (((((data["t_sum_log"]) + (((((-3.0) * 2.0)) * 2.0)))) * 2.0)))))) * 2.0)) 
    v["i2"] = 0.100000*np.tanh(((((((((np.where(data["pred_1"] < -99998, data["pred_1"], ((data["t_mean"]) - ((7.67527198791503906))) )) * 2.0)) * 2.0)) * 2.0)) + (3.0))) 
    v["i3"] = 0.100000*np.tanh(((-3.0) + (((((((data["t_sum_log"]) + (((data["t_mean"]) - ((((7.0)) * 2.0)))))) * ((7.0)))) * 2.0)))) 
    v["i4"] = 0.100000*np.tanh(((((data["t_sum_log"]) - (np.maximum(((np.maximum((((6.96472883224487305))), ((((((data["visitNumber"]) * 2.0)) - (data["t_sum_log"]))))))), ((data["t_median"])))))) * 2.0)) 
    v["i5"] = 0.100000*np.tanh(((((((1.0) + (((((1.0) + (((((data["t_mean"]) - ((7.40952205657958984)))) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) 
    v["i6"] = 0.100000*np.tanh(((np.where(((data["t_sum_log"]) - ((9.06132984161376953)))>0, ((data["t_sum_log"]) - ((9.06132698059082031))), ((data["t_median"]) - ((9.06132698059082031))) )) * 2.0)) 
    v["i7"] = 0.100000*np.tanh(((np.minimum(((((((((data["t_sum_act"]) - (((3.0) * 2.0)))) * 2.0)) * 2.0))), (((((data["t_sum_act"]) > (data["t_mean"]))*1.))))) * 2.0)) 
    v["i8"] = 0.100000*np.tanh(np.minimum(((((((data["t_sum_act"]) - ((((12.94684314727783203)) / 2.0)))) * 2.0))), (((((data["pred_3"]) > (((((data["t_sum_act"]) / 2.0)) / 2.0)))*1.))))) 
    v["i9"] = 0.100000*np.tanh(((data["t_sum_log"]) + (((((((((((data["t_sum_log"]) + (((-3.0) * 2.0)))) * 2.0)) * 2.0)) - (data["visitNumber"]))) * 2.0)))) 
    v["i10"] = 0.100000*np.tanh(((((((((((((((-3.0) * 2.0)) + (data["sess_date_dom"]))) + (data["t_sum_log"]))) - ((5.83093786239624023)))) * 2.0)) * 2.0)) * 2.0)) 
    v["i11"] = 0.100000*np.tanh(((((-3.0) + (((data["t_sum_log"]) + (((((((data["t_sum_act"]) + (((-3.0) * 2.0)))) * 2.0)) * 2.0)))))) * 2.0)) 
    v["i12"] = 0.100000*np.tanh((((((10.66844654083251953)) * ((((((10.66844272613525391)) * (((data["t_sum_log"]) - ((10.66844272613525391)))))) - (data["t_sum_log"]))))) - ((10.66844272613525391)))) 
    v["i13"] = 0.100000*np.tanh(((((((data["t_sum_log"]) + (((data["t_sum_act"]) - (np.maximum(((np.maximum((((10.0))), ((data["pred_0"]))))), (((10.0))))))))) * 2.0)) * 2.0)) 
    v["i14"] = 0.100000*np.tanh(((-3.0) + (np.minimum(((np.minimum(((((data["t_mean"]) + (((data["t_sum_act"]) - ((9.0))))))), (((2.37444567680358887)))))), (((2.37444567680358887))))))) 
    v["i15"] = 0.100000*np.tanh((((((14.27473449707031250)) * (((np.where(data["sess_date_hours"]>0, ((data["t_sum_act"]) - ((3.87348985671997070))), -1.0 )) + (-1.0))))) * 2.0)) 
    v["i16"] = 0.100000*np.tanh(np.minimum(((data["geoNetwork_networkDomain"])), ((((np.maximum(((data["t_median"])), ((data["pred_0"])))) - (np.maximum((((((12.08827495574951172)) - (data["pred_0"])))), ((data["pred_0"]))))))))) 
    v["i17"] = 0.100000*np.tanh(np.minimum(((((data["t_sum_log"]) - ((((10.0)) / 2.0))))), ((((2.0) - (((data["t_sum_act"]) - ((((10.0)) / 2.0))))))))) 
    v["i18"] = 0.100000*np.tanh(((((((((np.minimum((((((data["t_sum_log"]) < ((5.0)))*1.))), ((((data["t_sum_log"]) - ((4.0))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i19"] = 0.100000*np.tanh(np.minimum((((((6.83367300033569336)) - (data["t_mean"])))), ((np.minimum(((data["pred_1"])), ((((data["t_mean"]) - ((((6.93451642990112305)) - (data["t_mean"]))))))))))) 
    v["i20"] = 0.100000*np.tanh(np.minimum((((((-1.0) + (2.0))/2.0))), ((((((((((((data["t_mean"]) - ((4.01557540893554688)))) * 2.0)) * 2.0)) * 2.0)) * 2.0))))) 
    v["i21"] = 0.100000*np.tanh(np.minimum(((0.0)), (((((((data["t_sum_log"]) * 2.0)) + (((data["t_sum_log"]) - ((((12.81262588500976562)) + (data["visitNumber"]))))))/2.0))))) 
    v["i22"] = 0.100000*np.tanh(((np.minimum(((((data["t_sum_act"]) - ((((2.0)) * 2.0))))), (((((((data["t_sum_act"]) - (3.0))) < ((2.0)))*1.))))) * 2.0)) 
    v["i23"] = 0.100000*np.tanh(np.minimum(((0.0)), ((((((((np.where(data["pred_0"]>0, data["pred_0"], data["t_mean"] )) - (3.0))) * 2.0)) * 2.0))))) 
    v["i24"] = 0.100000*np.tanh((((10.90211677551269531)) * ((((8.0)) * (((-3.0) + (np.minimum(((3.0)), ((np.minimum(((data["t_sum_act"])), ((data["t_sum_log"]))))))))))))) 
    v["i25"] = 0.100000*np.tanh(((data["t_sum_log"]) - (np.where(((-3.0) + ((((data["t_sum_log"]) + (data["t_mean"]))/2.0)))<0, (7.14634370803833008), data["t_sum_log"] )))) 
    v["i26"] = 0.100000*np.tanh(((((((np.where(((data["t_mean"]) - (3.0))>0, 3.0, data["t_sum_log"] )) - (3.0))) * 2.0)) * 2.0)) 
    v["i27"] = 0.100000*np.tanh((((((((np.minimum(((((((data["t_mean"]) * 2.0)) * 2.0))), ((data["t_sum_act"])))) < (3.0))*1.)) * (-3.0))) * ((6.24691629409790039)))) 
    v["i28"] = 0.100000*np.tanh((((((((((((data["t_sum_act"]) / 2.0)) < (2.0))*1.)) * (((((data["t_sum_act"]) * 2.0)) - ((5.52667522430419922)))))) * 2.0)) * 2.0)) 
    v["i29"] = 0.100000*np.tanh(np.minimum(((((np.minimum(((0.0)), (((((((((data["t_sum_log"]) - ((6.83591127395629883)))) + (data["t_sum_act"]))/2.0)) * 2.0))))) * 2.0))), ((data["t_median"])))) 
    v["i30"] = 0.100000*np.tanh(((np.where(np.where(((-2.0) + (data["t_sum_act"]))>0, data["t_mean"], -3.0 )>0, 0.0, -3.0 )) * 2.0)) 
    v["i31"] = 0.100000*np.tanh((((((data["t_sum_log"]) < (2.0))*1.)) * (((((-3.0) * 2.0)) * 2.0)))) 
    v["i32"] = 0.100000*np.tanh((((3.0) + (np.where((((data["t_sum_act"]) + (-2.0))/2.0)<0, ((-2.0) * ((9.0))), -3.0 )))/2.0)) 
    v["i33"] = 0.100000*np.tanh(np.minimum(((np.tanh((((((((data["trafficSource_source"]) < (data["pred_2"]))*1.)) > (data["sess_date_hours"]))*1.))))), ((((((-2.0) + (data["t_sum_log"]))) * 2.0))))) 
    v["i34"] = 0.100000*np.tanh(((np.minimum((((((data["t_median"]) > ((10.0)))*1.))), ((((np.minimum(((data["t_median"])), ((((data["t_sum_act"]) + (-2.0)))))) * 2.0))))) * 2.0)) 
    v["i35"] = 0.100000*np.tanh(((np.minimum(((((np.minimum(((((data["t_sum_act"]) - ((2.0))))), ((((((data["geoNetwork_country"]) * 2.0)) * 2.0))))) * 2.0))), ((0.0)))) * 2.0)) 
    v["i36"] = 0.100000*np.tanh(((((np.minimum((((((np.maximum(((data["t_median"])), ((data["t_median"])))) < (data["pred_2"]))*1.))), ((((data["t_sum_log"]) + (-2.0)))))) * 2.0)) * 2.0)) 
    v["i37"] = 0.100000*np.tanh(np.minimum(((0.0)), (((((((-3.0) - (((data["pred_2"]) * (((-3.0) + (data["t_median"]))))))) + (data["t_sum_log"]))/2.0))))) 
    v["i38"] = 0.100000*np.tanh(((np.where(((-3.0) + (data["t_sum_log"]))<0, ((((-3.0) + (data["t_sum_log"]))) + (data["t_mean"])), 0.0 )) * 2.0)) 
    v["i39"] = 0.100000*np.tanh(np.minimum(((((-3.0) + (data["t_mean"])))), ((np.where(data["trafficSource_source"]<0, -1.0, ((3.0) - (data["visitNumber"])) ))))) 
    v["i40"] = 0.100000*np.tanh(((((-3.0) * 2.0)) * ((((np.tanh((((1.0) * 2.0)))) > (data["t_sum_log"]))*1.)))) 
    v["i41"] = 0.099951*np.tanh(np.where(((data["t_mean"]) + (data["pred_2"]))>0, (((data["totals_hits"]) < (data["totals_pageviews"]))*1.), ((data["totals_pageviews"]) + (((-3.0) * 2.0))) )) 
    v["i42"] = 0.100000*np.tanh(((((((((data["t_sum_log"]) + (((data["t_sum_log"]) + (-2.0))))) * ((((data["t_mean"]) < (2.0))*1.)))) * 2.0)) * 2.0)) 
    v["i43"] = 0.100000*np.tanh(np.minimum(((((2.0) - (np.where(data["t_mean"]<0, (8.0), data["visitNumber"] ))))), (((((data["totals_pageviews"]) > (data["totals_hits"]))*1.))))) 
    v["i44"] = 0.100000*np.tanh(np.minimum(((((data["t_sum_act"]) + (((-3.0) + (data["t_sum_act"])))))), (((((((data["t_mean"]) < (data["t_sum_act"]))*1.)) - (data["trafficSource_source"])))))) 
    v["i45"] = 0.100000*np.tanh(np.where(data["t_mean"]<0, -3.0, np.where(data["geoNetwork_networkDomain"]>0, ((np.tanh((data["geoNetwork_networkDomain"]))) - (data["geoNetwork_networkDomain"])), (0.20732049643993378) ) )) 
    v["i46"] = 0.100000*np.tanh((((((((((1.0)) + (data["t_sum_log"]))/2.0)) > (data["t_sum_act"]))*1.)) * (((-3.0) * 2.0)))) 
    v["i47"] = 0.100000*np.tanh(((-3.0) * ((((np.tanh((((-1.0) + (data["t_sum_log"]))))) < ((((data["geoNetwork_networkDomain"]) < (np.tanh((-1.0))))*1.)))*1.)))) 
    v["i48"] = 0.100000*np.tanh(np.minimum(((((((data["t_sum_act"]) - (3.0))) + (data["t_sum_act"])))), ((((((((data["pred_0"]) + (data["sess_date_hours"]))/2.0)) > (data["t_sum_log"]))*1.))))) 
    v["i49"] = 0.100000*np.tanh(np.where(((data["t_median"]) + (data["pred_1"])) < -99998, data["pred_1"], np.where(data["t_mean"]<0, -3.0, (((data["pred_1"]) > (data["pred_1"]))*1.) ) )) 
    v["i50"] = 0.100000*np.tanh(((((-3.0) * ((((np.maximum(((data["visitNumber"])), ((np.tanh((np.tanh((np.tanh((data["t_mean"])))))))))) > (data["t_sum_log"]))*1.)))) * 2.0)) 
    v["i51"] = 0.099951*np.tanh(np.where(data["t_sum_log"]<0, ((-3.0) * 2.0), ((((((data["t_sum_log"]) < (data["pred_0"]))*1.)) < ((((-3.0) + (data["visitNumber"]))/2.0)))*1.) )) 
    v["i52"] = 0.099902*np.tanh(np.where(np.where(data["t_mean"]>0, (((data["geoNetwork_networkDomain"]) + ((1.0)))/2.0), -3.0 )>0, (((data["t_mean"]) < (data["totals_pageviews"]))*1.), -3.0 )) 
    v["i53"] = 0.100000*np.tanh(np.where(data["t_sum_log"]<0, -3.0, np.minimum(((data["pred_0"])), ((np.minimum(((data["t_median"])), ((((((9.0)) < (((data["pred_0"]) / 2.0)))*1.))))))) )) 
    v["i54"] = 0.100000*np.tanh(np.where(((((data["t_nb_sess"]) / 2.0)) + (data["t_sum_log"]))<0, ((-2.0) * 2.0), ((data["sess_date_dow"]) * (((data["t_nb_sess"]) / 2.0))) )) 
    v["i55"] = 0.100000*np.tanh(((-1.0) + (((np.tanh((((data["t_sum_act"]) * 2.0)))) - ((((data["totals_hits"]) > (data["totals_pageviews"]))*1.)))))) 
    v["i56"] = 0.099609*np.tanh(np.where(data["t_mean"]>0, (((data["t_mean"]) < (((((((3.0) + (data["t_sum_log"]))/2.0)) + (data["t_sum_log"]))/2.0)))*1.), -2.0 )) 
    v["i57"] = 0.100000*np.tanh(((((((((np.tanh((np.tanh((((data["totals_hits"]) / 2.0)))))) < (data["trafficSource_source"]))*1.)) > (np.tanh((data["t_sum_act"]))))*1.)) * (-3.0))) 
    v["i58"] = 0.099951*np.tanh(np.minimum(((((((((((-1.0) + (data["t_sum_log"]))) + (data["t_sum_log"]))) * 2.0)) * 2.0))), (((((-2.0) > (data["sess_date_hours"]))*1.))))) 
    v["i59"] = 0.100000*np.tanh(((np.minimum(((0.0)), ((np.minimum(((((data["t_sum_log"]) - (data["visitNumber"])))), ((((((data["t_sum_log"]) - (1.0))) * 2.0)))))))) * 2.0)) 
    v["i60"] = 0.100000*np.tanh(np.minimum(((np.where(data["t_sum_log"]<0, -3.0, ((((((((2.0)) > (data["totals_pageviews"]))*1.)) / 2.0)) / 2.0) ))), ((data["totals_pageviews"])))) 
    v["i61"] = 0.099951*np.tanh(np.where(data["t_sum_act"]<0, -2.0, ((np.minimum(((data["sess_date_dom"])), (((((data["pred_0"]) < (data["t_median"]))*1.))))) / 2.0) )) 
    v["i62"] = 0.099951*np.tanh(np.where(data["t_sum_act"]>0, np.minimum((((((data["visitNumber"]) < (np.tanh((data["sess_date_dom"]))))*1.))), (((((data["visitNumber"]) < (data["trafficSource_source"]))*1.)))), -3.0 )) 
    v["i63"] = 0.100000*np.tanh(np.minimum((((((((data["pred_1"]) * (((data["totals_hits"]) - (data["totals_pageviews"]))))) + (data["totals_pageviews"]))/2.0))), (((((data["sess_date_dom"]) < (-1.0))*1.))))) 
    v["i64"] = 0.099951*np.tanh(np.minimum(((np.where(data["t_sum_act"]<0, -3.0, 0.0 ))), ((((data["totals_hits"]) - (np.maximum(((data["totals_pageviews"])), ((data["sess_date_hours"]))))))))) 
    v["i65"] = 0.100000*np.tanh(np.minimum(((((-1.0) + (data["t_sum_log"])))), ((np.minimum(((data["totals_pageviews"])), (((((2.0) > (((data["geoNetwork_city"]) + (data["totals_pageviews"]))))*1.)))))))) 
    v["i66"] = 0.100000*np.tanh(((np.minimum(((data["totals_hits"])), ((np.minimum(((((data["totals_pageviews"]) - (data["totals_hits"])))), ((((data["totals_hits"]) - (data["totals_pageviews"]))))))))) * 2.0)) 
    v["i67"] = 0.099951*np.tanh(((((((((4.0)) < (data["totals_pageviews"]))*1.)) * (((data["sess_date_hours"]) + (((data["pred_0"]) - (data["totals_pageviews"]))))))) * (data["t_mean"]))) 
    v["i68"] = 0.099707*np.tanh(np.where(data["t_mean"]<0, ((-3.0) * 2.0), ((np.minimum((((((1.0) < (data["geoNetwork_region"]))*1.))), ((data["totals_hits"])))) / 2.0) )) 
    v["i69"] = 0.082218*np.tanh((((((np.where(data["sess_date_hours"]>0, data["sess_date_hours"], data["pred_1"] )) > (2.0))*1.)) - ((((data["sess_date_hours"]) < (((data["geoNetwork_city"]) / 2.0)))*1.)))) 
    v["i70"] = 0.078065*np.tanh(np.where(data["t_sum_log"]>0, np.tanh((((np.where(data["sess_date_hours"]>0, ((3.0) * 2.0), data["t_mean"] )) - (data["t_sum_act"])))), -3.0 )) 
    v["i71"] = 0.100000*np.tanh(np.where(data["geoNetwork_country"]>0, (((((1.0) < (((data["geoNetwork_city"]) * ((((data["geoNetwork_city"]) + (data["geoNetwork_region"]))/2.0)))))*1.)) / 2.0), -3.0 )) 
    v["i72"] = 0.100000*np.tanh(((np.where(data["geoNetwork_subContinent"]<0, -3.0, np.where((((data["geoNetwork_subContinent"]) > (data["geoNetwork_country"]))*1.)>0, 0.0, -3.0 ) )) * 2.0)) 
    v["i73"] = 0.099951*np.tanh((((11.99727344512939453)) * (np.minimum(((data["geoNetwork_subContinent"])), ((np.minimum(((data["geoNetwork_country"])), ((np.minimum(((data["t_sum_log"])), (((((data["pred_1"]) > (data["t_sum_log"]))*1.))))))))))))) 
    v["i74"] = 0.099951*np.tanh(np.minimum(((((data["geoNetwork_networkDomain"]) * (np.tanh((((((data["pred_1"]) + (data["sess_date_dom"]))) + (((data["geoNetwork_networkDomain"]) * 2.0))))))))), ((0.0)))) 
    v["i75"] = 0.099707*np.tanh(((data["totals_pageviews"]) * (np.maximum((((((data["totals_pageviews"]) > (data["t_mean"]))*1.))), (((((((data["t_sum_log"]) - (data["totals_pageviews"]))) > (data["t_mean"]))*1.))))))) 
    v["i76"] = 0.099902*np.tanh(np.minimum((((((data["geoNetwork_country"]) < (data["sess_date_hours"]))*1.))), ((((data["geoNetwork_country"]) - ((((data["geoNetwork_country"]) < ((((data["geoNetwork_subContinent"]) < (data["sess_date_hours"]))*1.)))*1.))))))) 
    v["i77"] = 0.099902*np.tanh(np.where(data["sess_date_dom"]>0, ((((0.05049348995089531)) > (data["sess_date_dom"]))*1.), (((((data["sess_date_hours"]) > (data["geoNetwork_country"]))*1.)) / 2.0) )) 
    v["i78"] = 0.100000*np.tanh(np.minimum(((np.where(data["geoNetwork_subContinent"]<0, -1.0, data["t_median"] ))), (((((data["sess_date_hours"]) < (((data["geoNetwork_networkDomain"]) - (2.0))))*1.))))) 
    v["i79"] = 0.100000*np.tanh(((np.minimum(((((data["t_mean"]) - (((((data["geoNetwork_region"]) + (data["geoNetwork_city"]))) + (((data["geoNetwork_region"]) * 2.0))))))), ((0.0)))) * 2.0)) 
    v["i80"] = 0.100000*np.tanh(np.minimum((((((((data["trafficSource_referralPath"]) * 2.0)) > (-1.0))*1.))), ((((((-2.0) * (((data["trafficSource_referralPath"]) * 2.0)))) + (data["geoNetwork_continent"])))))) 
    v["i81"] = 0.099951*np.tanh((((((((((data["trafficSource_source"]) > (((data["totals_hits"]) - (((data["trafficSource_referralPath"]) / 2.0)))))*1.)) * (data["sess_date_hours"]))) * (data["t_mean"]))) * 2.0)) 
    v["i82"] = 0.099853*np.tanh(((((((np.where(data["geoNetwork_subContinent"]<0, data["geoNetwork_city"], (((3.0) < ((((data["geoNetwork_region"]) + (data["totals_pageviews"]))/2.0)))*1.) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i83"] = 0.099951*np.tanh(((np.minimum(((data["totals_hits"])), (((((data["t_median"]) < ((((data["totals_hits"]) + (((-2.0) - (np.tanh((data["geoNetwork_country"]))))))/2.0)))*1.))))) * 2.0)) 
    v["i84"] = 0.099902*np.tanh(((((((((((data["trafficSource_referralPath"]) < (data["totals_pageviews"]))*1.)) - (data["totals_pageviews"]))) > (data["trafficSource_source"]))*1.)) - ((((1.0) < (data["geoNetwork_country"]))*1.)))) 
    v["i85"] = 0.081974*np.tanh((((((data["totals_pageviews"]) > (data["t_sum_log"]))*1.)) - (np.where(data["trafficSource_referralPath"]<0, (((((3.0) / 2.0)) > (data["totals_pageviews"]))*1.), data["geoNetwork_city"] )))) 
    v["i86"] = 0.070982*np.tanh(((np.minimum(((((data["geoNetwork_city"]) * (data["trafficSource_medium"])))), ((((((((data["trafficSource_source"]) < (((data["trafficSource_medium"]) * 2.0)))*1.)) < (data["trafficSource_source"]))*1.))))) * 2.0)) 
    v["i87"] = 0.099951*np.tanh((((((np.maximum(((((data["totals_pageviews"]) * 2.0))), ((data["t_mean"])))) > (np.maximum(((((data["totals_pageviews"]) + (data["totals_hits"])))), ((data["pred_0"])))))*1.)) / 2.0)) 
    v["i88"] = 0.076160*np.tanh((((((data["pred_0"]) < (np.minimum(((data["t_sum_log"])), ((2.0)))))*1.)) * (((data["t_median"]) * (np.minimum(((data["pred_0"])), ((data["geoNetwork_region"])))))))) 
    v["i89"] = 0.099951*np.tanh(np.where(((data["channelGrouping"]) + (data["t_sum_act"]))>0, ((((np.minimum(((data["geoNetwork_city"])), ((data["sess_date_dow"])))) / 2.0)) * (data["channelGrouping"])), -3.0 )) 
    v["i90"] = 0.085344*np.tanh(((np.minimum(((data["trafficSource_source"])), ((np.where(data["trafficSource_source"]<0, -1.0, ((((((data["geoNetwork_city"]) > (data["geoNetwork_region"]))*1.)) > (data["trafficSource_source"]))*1.) ))))) / 2.0)) 
    v["i91"] = 0.099951*np.tanh(((((((((((data["sess_date_dom"]) > (data["geoNetwork_country"]))*1.)) > ((((-1.0) + (data["t_mean"]))/2.0)))*1.)) * 2.0)) * (data["t_sum_log"]))) 
    v["i92"] = 0.099853*np.tanh(np.where(data["t_mean"]<0, -3.0, np.minimum(((data["totals_hits"])), (((((data["trafficSource_referralPath"]) > (data["totals_hits"]))*1.)))) )) 
    v["i93"] = 0.099511*np.tanh(((((((data["sess_date_hours"]) < (data["sess_date_dow"]))*1.)) < (np.tanh((np.where(data["sess_date_dom"]<0, (((data["sess_date_dow"]) < (-1.0))*1.), data["sess_date_dow"] )))))*1.)) 
    v["i94"] = 0.099853*np.tanh(np.where(np.minimum(((data["geoNetwork_country"])), ((data["channelGrouping"])))>0, ((data["channelGrouping"]) - (1.0)), ((((-1.0) - (data["channelGrouping"]))) / 2.0) )) 
    v["i95"] = 0.067220*np.tanh((((np.minimum(((data["t_median"])), ((np.where(data["sess_date_hours"]>0, np.where(data["geoNetwork_city"]>0, data["pred_0"], data["t_median"] ), 0.0 ))))) > (data["pred_0"]))*1.)) 
    v["i96"] = 0.100000*np.tanh(((((data["t_median"]) * ((((((((data["geoNetwork_subContinent"]) < ((((data["geoNetwork_subContinent"]) < (data["geoNetwork_country"]))*1.)))*1.)) * (data["geoNetwork_region"]))) * 2.0)))) * 2.0)) 
    v["i97"] = 0.099951*np.tanh(np.minimum((((((data["sess_date_dow"]) > (np.tanh((data["geoNetwork_networkDomain"]))))*1.))), ((np.where(data["geoNetwork_subContinent"]<0, -3.0, (((data["totals_hits"]) < (data["geoNetwork_city"]))*1.) ))))) 
    v["i98"] = 0.099951*np.tanh(np.minimum(((data["t_mean"])), (((((((((2.0) - ((((data["t_mean"]) < (data["totals_hits"]))*1.)))) < (data["geoNetwork_city"]))*1.)) * (-3.0)))))) 
    v["i99"] = 0.075574*np.tanh(np.where((((data["geoNetwork_networkDomain"]) > (data["geoNetwork_subContinent"]))*1.)>0, data["geoNetwork_city"], (((((data["geoNetwork_country"]) * (data["geoNetwork_city"]))) > (data["geoNetwork_subContinent"]))*1.) ))

    return Output(v.sum(axis=1))


# In[ ]:


a = maxval*GP1(gp_trn_users)
b = maxval*GP2(gp_trn_users)
c = maxval*GP3(gp_trn_users)
d = maxval*GP4(gp_trn_users)
e = maxval*GP5(gp_trn_users)
f = maxval*GP6(gp_trn_users)
print(mean_squared_error((maxval*gp_trn_users['target']), a) ** .5)
print(mean_squared_error((maxval*gp_trn_users['target']), b) ** .5)
print(mean_squared_error((maxval*gp_trn_users['target']), c) ** .5)
print(mean_squared_error((maxval*gp_trn_users['target']), d) ** .5)
print(mean_squared_error((maxval*gp_trn_users['target']), e) ** .5)
print(mean_squared_error((maxval*gp_trn_users['target']), f) ** .5)
print(mean_squared_error((maxval*gp_trn_users['target']), (a+b+c+d+e+f)/6) ** .5)


# In[ ]:


a = maxval*GP1(gp_sub_users)
b = maxval*GP2(gp_sub_users)
c = maxval*GP3(gp_sub_users)
d = maxval*GP4(gp_sub_users)
e = maxval*GP5(gp_sub_users)
f = maxval*GP6(gp_sub_users)


# In[ ]:


gp_sub_users['PredictedLogRevenue'] = a.values 
gp_sub_users[['PredictedLogRevenue']].to_csv('gp1_test.csv', index=True) #1.4381
gp_sub_users['PredictedLogRevenue'] = b.values
gp_sub_users[['PredictedLogRevenue']].to_csv('gp2_test.csv', index=True) #1.4376
gp_sub_users['PredictedLogRevenue'] = c.values
gp_sub_users[['PredictedLogRevenue']].to_csv('gp3_test.csv', index=True) #1.4379
gp_sub_users['PredictedLogRevenue'] = d.values
gp_sub_users[['PredictedLogRevenue']].to_csv('gp4_test.csv', index=True) #1.4382
gp_sub_users['PredictedLogRevenue'] = e.values
gp_sub_users[['PredictedLogRevenue']].to_csv('gp5_test.csv', index=True) #1.4325
gp_sub_users['PredictedLogRevenue'] = f.values
gp_sub_users[['PredictedLogRevenue']].to_csv('gp6_test.csv', index=True) #1.4324
gp_sub_users['PredictedLogRevenue'] = (a+b+c+d+e+f).values/6
gp_sub_users[['PredictedLogRevenue']].to_csv('gpari_test.csv', index=True) 

