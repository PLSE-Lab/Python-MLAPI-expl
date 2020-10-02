#!/usr/bin/env python
# coding: utf-8

# ## Reference kernel
# * https://www.kaggle.com/cdeotte/one-feature-model-0-930

# In[ ]:


from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import acf, pacf, graphics
from typing import List, Tuple, Union, NoReturn
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
import cufflinks as cf
import plotly
from statsmodels.robust import mad
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy import signal
import seaborn as sns
from sklearn import *
import pandas as pd 
import numpy as np
import warnings
import scipy
import pywt
import os
import gc
from sklearn.metrics import cohen_kappa_score,f1_score,confusion_matrix

cf.go_offline()
py.init_notebook_mode()
cf.getThemes()
cf.set_config_file(theme='ggplot')
warnings.simplefilter('ignore')
pd.plotting.register_matplotlib_converters()
sns.mpl.rc('figure',figsize=(16, 6))
plt.style.use('ggplot')
sns.set_style('darkgrid')


# In[ ]:


base = os.path.abspath('/kaggle/input/liverpool-ion-switching/')
train = pd.read_csv(os.path.join(base + '/train.csv'))
test  = pd.read_csv(os.path.join(base + '/test.csv'))


# ## Remove Training/Test Data Drift

# In[ ]:


train2 = train.copy()

a=500000; b=600000 # CLEAN TRAIN BATCH 2
train2.loc[train.index[a:b],'signal2'] = train2.signal[a:b].values - 3*(train2.time.values[a:b] - 50)/10.

def f(x,low,high,mid): return -((-low+high)/625)*(x-mid)**2+high -low

# CLEAN TRAIN BATCH 7
batch = 7; a = 500000*(batch-1); b = 500000*batch
train2.loc[train2.index[a:b],'signal2'] = train.signal.values[a:b] - f(train.time[a:b].values,-1.817,3.186,325)
# CLEAN TRAIN BATCH 8
batch = 8; a = 500000*(batch-1); b = 500000*batch
train2.loc[train2.index[a:b],'signal2'] = train.signal.values[a:b] - f(train.time[a:b].values,-0.094,4.936,375)
# CLEAN TRAIN BATCH 9
batch = 9; a = 500000*(batch-1); b = 500000*batch
train2.loc[train2.index[a:b],'signal2'] = train.signal.values[a:b] - f(train.time[a:b].values,1.715,6.689,425)
# CLEAN TRAIN BATCH 10
batch = 10; a = 500000*(batch-1); b = 500000*batch
train2.loc[train2.index[a:b],'signal2'] = train.signal.values[a:b] - f(train.time[a:b].values,3.361,8.45,475)


# In[ ]:


test2 = test.copy()

# REMOVE BATCH 1 DRIFT
start=500
a = 0; b = 100000
test2.loc[test2.index[a:b],'signal2'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.
start=510
a = 100000; b = 200000
test2.loc[test2.index[a:b],'signal2'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.
start=540
a = 400000; b = 500000
test2.loc[test2.index[a:b],'signal2'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.

# REMOVE BATCH 2 DRIFT
start=560
a = 600000; b = 700000
test2.loc[test2.index[a:b],'signal2'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.
start=570
a = 700000; b = 800000
test2.loc[test2.index[a:b],'signal2'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.
start=580
a = 800000; b = 900000
test2.loc[test2.index[a:b],'signal2'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.

# REMOVE BATCH 3 DRIFT
def f(x):
    return -(0.00788)*(x-625)**2+2.345 +2.58
a = 1000000; b = 1500000
test2.loc[test2.index[a:b],'signal2'] = test2.signal.values[a:b] - f(test2.time[a:b].values)


# ## Create feature

# In[ ]:


def features(df):
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df.index = ((df.time * 10_000) - 1).values
    df['batch'] = df.index // 25_000
    df['batch_index'] = df.index  - (df.batch * 25_000)
    df['batch_slices'] = df['batch_index']  // 2500
    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)
    
    for j in ['signal']:
        for c in ['batch','batch_slices2']:
            d = {}
            d['mean'+c+j] = df.groupby([c])[j].mean()
            d['median'+c+j] = df.groupby([c])[j].median()
            d['max'+c+j] = df.groupby([c])[j].max()
            d['min'+c+j] = df.groupby([c])[j].min()
            d['std'+c+j] = df.groupby([c])[j].std()
            d['mean_abs_chg'+c+j] = df.groupby([c])[j].apply(lambda x: np.mean(np.abs(np.diff(x))))
            d['abs_max'+c+j] = df.groupby([c])[j].apply(lambda x: np.max(np.abs(x)))
            d['abs_min'+c+j] = df.groupby([c])[j].apply(lambda x: np.min(np.abs(x)))
            d['range'+c+j] = d['max'+c+j] - d['min'+c+j]
            d['maxtomin'+c+j] = d['max'+c+j] / d['min'+c+j]
            d['abs_avg'+c+j] = (d['abs_min'+c+j] + d['abs_max'+c+j]) / 2
            for v in d:
                df[v] = df[c].map(d[v].to_dict())
    #add shifts
        df['signal_shift_+1'+j] = df[j].shift(1)
        df['signal_shift_-1'+j] = df[j].shift(-1)
        
            
    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:
        df[c+'_msignal'] = df[c] - df['signal']
        
   # df['binning']=pd.cut(df['signal'], [-np.inf] + list(np.linspace(df['signal'].min(),df['signal'].max(),12)) + [np.inf], labels = range(int(df['signal'].min()),int(df['signal'].max())))
    
    return df
    
train2 = features(train2)
test = features(test2)


# In[ ]:


col = [c for c in train2.columns if c not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]


# In[ ]:


def MacroF1Metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = metrics.f1_score(labels, preds, average = 'macro')
    return ('MacroF1Metric', score, True)


# In[ ]:


model_batch = [[1,2],[3,7],[4,8],[6,9],[5,10]]


# In[ ]:


sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')


# In[ ]:


valid_predict_LGBM =[]
valid_target_LGBM =[]
valid_predict_XGB =[]
valid_target_XGB =[]
valid_predict_CAT =[]
valid_target_CAT =[]


# In[ ]:


sub['LGBM_preds']=sub.open_channels
sub['XGB_preds']=sub.open_channels
sub['CAT_preds']=sub.open_channels


# ## Model 1 Slow Open Channel

# In[ ]:


for j,idx in enumerate(model_batch[0]):
    if j ==0:
        batch = idx; a = 500000*(batch-1); b = 500000*batch
        train_model_split = train2[col].loc[train2.index[a:b]]
        train_model_target = train2.loc[train2.index[a:b],'open_channels']
    else:
        batch = idx; a = 500000*(batch-1); b = 500000*batch
        train_model_split1 = train2[col].loc[train2.index[a:b]]
        train_model_target1 = train2.loc[train2.index[a:b],'open_channels']
        train_model_split = np.concatenate((train_model_split,train_model_split1),axis = 0)
        train_model_target = np.concatenate((train_model_target,train_model_target1),axis = 0)

x1, x2, y1, y2 = model_selection.train_test_split(train_model_split, train_model_target , test_size=0.2, random_state=49)


print('============================ start training LGBM ==================================')

#First model
import lightgbm as lgb
params = {'learning_rate': 0.1, 'max_depth': -1, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1, 'sample_fraction':0.33} 
model = lgb.train(params, lgb.Dataset(x1, y1), 22222,  lgb.Dataset(x2, y2), verbose_eval=100, early_stopping_rounds=250, feval=MacroF1Metric)
#preds_lgb = (model.predict(test[col], num_iteration=model.best_iteration)).astype(np.float16)
oof_valid_lgb = (model.predict(x2, num_iteration=model.best_iteration)).astype(np.float16)
print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_lgb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
valid_predict_LGBM = valid_predict_LGBM + oof_valid_lgb.tolist()
valid_target_LGBM = valid_target_LGBM + np.array(y2).tolist()

print('============================ start LGBM predict==================================')
a = 0 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),2] = (model.predict(test_batch, num_iteration=model.best_iteration)).astype(np.float16)

a = 3
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),2] = (model.predict(test_batch, num_iteration=model.best_iteration)).astype(np.float16)

a = 8
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),2] = (model.predict(test_batch, num_iteration=model.best_iteration)).astype(np.float16)

test_batch = test[col].loc[test.index[1000000:2000000]]
print ('start test last subsample'+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[1000000:2000000,2] = (model.predict(test_batch, num_iteration=model.best_iteration)).astype(np.float16)

print (gc.collect())


print('============================ start training XGBBOOST ==================================')
import xgboost as xgb
params = {'colsample_bytree': 0.375,'learning_rate': 0.1,'max_depth': 10, 'subsample': 1, 'objective':'reg:squarederror',
          'eval_metric':'rmse', 'n_estimators':22222,   'tree_method':'gpu_hist',}
train_set = xgb.DMatrix(x1, y1)
val_set = xgb.DMatrix(x2, y2)
model = xgb.train(params, train_set, num_boost_round=2222, evals=[(train_set, 'train'), (val_set, 'val')], 
                         verbose_eval=100, early_stopping_rounds=250)
#preds_xgb = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit)
oof_valid_xgb = (model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit)).astype(np.float16)
print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_xgb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
del train_set, val_set; gc.collect()
valid_predict_XGB = valid_predict_XGB + oof_valid_xgb.tolist()
valid_target_XGB = valid_target_XGB + np.array(y2).tolist()

print('============================ start XGB predict==================================')
a = 0 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),3] = (model.predict(xgb.DMatrix(np.array(test_batch)), ntree_limit=model.best_ntree_limit)).astype(np.float16)

a = 3
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),3] = (model.predict(xgb.DMatrix(np.array(test_batch)), ntree_limit=model.best_ntree_limit)).astype(np.float16)

a = 8
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),3] = (model.predict(xgb.DMatrix(np.array(test_batch)), ntree_limit=model.best_ntree_limit)).astype(np.float16)

test_batch = test[col].loc[test.index[1000000:2000000]]
print ('start test last subsample'+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[1000000:2000000,3] = (model.predict(xgb.DMatrix(np.array(test_batch)), ntree_limit=model.best_ntree_limit)).astype(np.float16)

print('============================ start training Catboost ==================================')
from catboost import Pool,CatBoostRegressor
model = CatBoostRegressor(task_type = 'GPU', iterations=22222, learning_rate=0.1, random_seed = 7, depth=7, eval_metric='RMSE')
train_dataset = Pool(x1,  y1)          
eval_dataset = Pool(x2,  y2)
model.fit(train_dataset, eval_set=eval_dataset, verbose=100, early_stopping_rounds=250)
#preds_cb = (model.predict(test[col])).astype(np.float16)
oof_valid_cb = (model.predict(x2)).astype(np.float16)
print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_cb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
del train_dataset, eval_dataset 
print (gc.collect())
valid_predict_CAT = valid_predict_CAT + oof_valid_cb.tolist()
valid_target_CAT = valid_target_CAT + np.array(y2).tolist()

print('============================ start Catboost predict==================================')
a = 0 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),4] = (model.predict(test_batch)).astype(np.float16)

a = 3
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),4] = (model.predict(test_batch)).astype(np.float16)

a = 8
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),4] = (model.predict(test_batch)).astype(np.float16)

test_batch = test[col].loc[test.index[1000000:2000000]]
print ('start test last subsample'+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[1000000:2000000,4] = (model.predict(test_batch)).astype(np.float16)


# ## Model 1 Fast Open Channel

# In[ ]:


for j,idx in enumerate(model_batch[1]):
    if j ==0:
        batch = idx; a = 500000*(batch-1); b = 500000*batch
        train_model_split = train2[col].loc[train2.index[a:b]]
        train_model_target = train2.loc[train2.index[a:b],'open_channels']
    else:
        batch = idx; a = 500000*(batch-1); b = 500000*batch
        train_model_split1 = train2[col].loc[train2.index[a:b]]
        train_model_target1 = train2.loc[train2.index[a:b],'open_channels']
        train_model_split = np.concatenate((train_model_split,train_model_split1),axis = 0)
        train_model_target = np.concatenate((train_model_target,train_model_target1),axis = 0)

x1, x2, y1, y2 = model_selection.train_test_split(train_model_split, train_model_target , test_size=0.2, random_state=49)

print('============================ start training LGBM ==================================')

#Second model
import lightgbm as lgb
params = {'learning_rate': 0.1, 'max_depth': -1, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1, 'sample_fraction':0.33} 
model = lgb.train(params, lgb.Dataset(x1, y1), 22222,  lgb.Dataset(x2, y2), verbose_eval=100, early_stopping_rounds=250, feval=MacroF1Metric)
#preds_lgb = (model.predict(test[col], num_iteration=model.best_iteration)).astype(np.float16)
oof_valid_lgb = (model.predict(x2, num_iteration=model.best_iteration)).astype(np.float16)

print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_lgb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
valid_predict_LGBM = valid_predict_LGBM + oof_valid_lgb.tolist()
valid_target_LGBM = valid_target_LGBM + np.array(y2).tolist()

print('============================ start LGBM predict==================================')
a = 4 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),2] = (model.predict(test_batch, num_iteration=model.best_iteration)).astype(np.float16)

gc.collect()


print('============================ start training XGBBOOST ==================================')
import xgboost as xgb
params = {'colsample_bytree': 0.375,'learning_rate': 0.1,'max_depth': 10, 'subsample': 1, 'objective':'reg:squarederror',
          'eval_metric':'rmse', 'n_estimators':22222,   'tree_method':'gpu_hist',}
train_set = xgb.DMatrix(x1, y1)
val_set = xgb.DMatrix(x2, y2)
model = xgb.train(params, train_set, num_boost_round=2222, evals=[(train_set, 'train'), (val_set, 'val')], 
                         verbose_eval=100, early_stopping_rounds=250)
#preds_xgb = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit)
oof_valid_xgb = (model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit)).astype(np.float16)
print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_xgb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
del train_set, val_set; gc.collect()
valid_predict_XGB = valid_predict_XGB + oof_valid_xgb.tolist()
valid_target_XGB = valid_target_XGB + np.array(y2).tolist()

print('============================ start XGB predict==================================')
a = 4 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),3] = (model.predict(xgb.DMatrix(np.array(test_batch)), ntree_limit=model.best_ntree_limit)).astype(np.float16)

print('============================ start training Catboost ==================================')
from catboost import Pool,CatBoostRegressor
model = CatBoostRegressor(task_type = 'GPU', iterations=22222, learning_rate=0.1, random_seed = 7, depth=7, eval_metric='RMSE')
train_dataset = Pool(x1,  y1)          
eval_dataset = Pool(x2,  y2)
model.fit(train_dataset, eval_set=eval_dataset, verbose=100, early_stopping_rounds=250)
#preds_cb = (model.predict(test[col])).astype(np.float16)
oof_valid_cb = (model.predict(x2)).astype(np.float16)
print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_cb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
del train_dataset, eval_dataset; gc.collect()
valid_predict_CAT = valid_predict_CAT + oof_valid_cb.tolist()
valid_target_CAT = valid_target_CAT + np.array(y2).tolist()

print('============================ start Catboost predict==================================')
a = 4 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),4] = (model.predict(test_batch)).astype(np.float16)


# ## Model 3 Open Channel

# In[ ]:


for j,idx in enumerate(model_batch[2]):
    if j ==0:
        batch = idx; a = 500000*(batch-1); b = 500000*batch
        train_model_split = train2[col].loc[train2.index[a:b]]
        train_model_target = train2.loc[train2.index[a:b],'open_channels']
    else:
        batch = idx; a = 500000*(batch-1); b = 500000*batch
        train_model_split1 = train2[col].loc[train2.index[a:b]]
        train_model_target1 = train2.loc[train2.index[a:b],'open_channels']
        train_model_split = np.concatenate((train_model_split,train_model_split1),axis = 0)
        train_model_target = np.concatenate((train_model_target,train_model_target1),axis = 0)

x1, x2, y1, y2 = model_selection.train_test_split(train_model_split, train_model_target , test_size=0.2, random_state=49)

print('============================ start training LGBM ==================================')
#Third model
import lightgbm as lgb
params = {'learning_rate': 0.1, 'max_depth': -1, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1, 'sample_fraction':0.33} 
model = lgb.train(params, lgb.Dataset(x1, y1), 22222,  lgb.Dataset(x2, y2), verbose_eval=100, early_stopping_rounds=250, feval=MacroF1Metric)
#preds_lgb = (model.predict(test[col], num_iteration=model.best_iteration)).astype(np.float16)
oof_valid_lgb = (model.predict(x2, num_iteration=model.best_iteration)).astype(np.float16)

print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_lgb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
valid_predict_LGBM = valid_predict_LGBM + oof_valid_lgb.tolist()
valid_target_LGBM = valid_target_LGBM + np.array(y2).tolist()

print('============================ start LGBM predict==================================')
a = 1 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),2] = (model.predict(test_batch, num_iteration=model.best_iteration)).astype(np.float16)

a = 9 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),2] = (model.predict(test_batch, num_iteration=model.best_iteration)).astype(np.float16)


gc.collect()

print('============================ start training XGBBOOST ==================================')
import xgboost as xgb
params = {'colsample_bytree': 0.375,'learning_rate': 0.1,'max_depth': 10, 'subsample': 1, 'objective':'reg:squarederror',
          'eval_metric':'rmse', 'n_estimators':22222,   'tree_method':'gpu_hist',}
train_set = xgb.DMatrix(x1, y1)
val_set = xgb.DMatrix(x2, y2)
model = xgb.train(params, train_set, num_boost_round=2222, evals=[(train_set, 'train'), (val_set, 'val')], 
                         verbose_eval=100, early_stopping_rounds=250)
#preds_xgb = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit)
oof_valid_xgb = (model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit)).astype(np.float16)
print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_xgb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
del train_set, val_set; gc.collect()
valid_predict_XGB = valid_predict_XGB + oof_valid_xgb.tolist()
valid_target_XGB = valid_target_XGB + np.array(y2).tolist()

print('============================ start XGB predict==================================')
a = 1 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),3] = (model.predict(xgb.DMatrix(np.array(test_batch)), ntree_limit=model.best_ntree_limit)).astype(np.float16)

a = 9 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),3] = (model.predict(xgb.DMatrix(np.array(test_batch)), ntree_limit=model.best_ntree_limit)).astype(np.float16)

print('============================ start training Catboost ==================================')
from catboost import Pool,CatBoostRegressor
model = CatBoostRegressor(task_type = 'GPU', iterations=22222, learning_rate=0.1, random_seed = 7, depth=7, eval_metric='RMSE')
train_dataset = Pool(x1,  y1)          
eval_dataset = Pool(x2,  y2)
model.fit(train_dataset, eval_set=eval_dataset, verbose=100, early_stopping_rounds=250)
#preds_cb = (model.predict(test[col])).astype(np.float16)
oof_valid_cb = (model.predict(x2)).astype(np.float16)
print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_cb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
del train_dataset, eval_dataset; gc.collect()
valid_predict_CAT = valid_predict_CAT + oof_valid_cb.tolist()
valid_target_CAT = valid_target_CAT + np.array(y2).tolist()

print('============================ start Catboost predict==================================')
a = 1 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),4] = (model.predict(test_batch)).astype(np.float16)

a = 9
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),4] = (model.predict(test_batch)).astype(np.float16)


# ## Model 5 Open Channel

# In[ ]:


for j,idx in enumerate(model_batch[3]):
    if j ==0:
        batch = idx; a = 500000*(batch-1); b = 500000*batch
        train_model_split = train2[col].loc[train2.index[a:b]]
        train_model_target = train2.loc[train2.index[a:b],'open_channels']
    else:
        batch = idx; a = 500000*(batch-1); b = 500000*batch
        train_model_split1 = train2[col].loc[train2.index[a:b]]
        train_model_target1 = train2.loc[train2.index[a:b],'open_channels']
        train_model_split = np.concatenate((train_model_split,train_model_split1),axis = 0)
        train_model_target = np.concatenate((train_model_target,train_model_target1),axis = 0)

x1, x2, y1, y2 = model_selection.train_test_split(train_model_split, train_model_target , test_size=0.2, random_state=49)

print('============================ start training LGBM ==================================')
#Third model
import lightgbm as lgb
params = {'learning_rate': 0.1, 'max_depth': -1, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1, 'sample_fraction':0.33} 
model = lgb.train(params, lgb.Dataset(x1, y1), 22222,  lgb.Dataset(x2, y2), verbose_eval=100, early_stopping_rounds=250, feval=MacroF1Metric)
#preds_lgb = (model.predict(test[col], num_iteration=model.best_iteration)).astype(np.float16)
oof_valid_lgb = (model.predict(x2, num_iteration=model.best_iteration)).astype(np.float16)

print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_lgb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
valid_predict_LGBM = valid_predict_LGBM + oof_valid_lgb.tolist()
valid_target_LGBM = valid_target_LGBM + np.array(y2).tolist()

print('============================ start LGBM predict==================================')
a = 2 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),2] = (model.predict(test_batch, num_iteration=model.best_iteration)).astype(np.float16)

a = 6 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),2] = (model.predict(test_batch, num_iteration=model.best_iteration)).astype(np.float16)


gc.collect()

print('============================ start training XGBBOOST ==================================')
import xgboost as xgb
params = {'colsample_bytree': 0.375,'learning_rate': 0.1,'max_depth': 10, 'subsample': 1, 'objective':'reg:squarederror',
          'eval_metric':'rmse', 'n_estimators':22222,   'tree_method':'gpu_hist',}
train_set = xgb.DMatrix(x1, y1)
val_set = xgb.DMatrix(x2, y2)
model = xgb.train(params, train_set, num_boost_round=2222, evals=[(train_set, 'train'), (val_set, 'val')], 
                         verbose_eval=100, early_stopping_rounds=250)
#preds_xgb = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit)
oof_valid_xgb = (model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit)).astype(np.float16)
print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_xgb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
del train_set, val_set; gc.collect()
valid_predict_XGB = valid_predict_XGB + oof_valid_xgb.tolist()
valid_target_XGB = valid_target_XGB + np.array(y2).tolist()

print('============================ start XGB predict==================================')
a = 2 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),3] = (model.predict(xgb.DMatrix(np.array(test_batch)), ntree_limit=model.best_ntree_limit)).astype(np.float16)

a = 6 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),3] = (model.predict(xgb.DMatrix(np.array(test_batch)), ntree_limit=model.best_ntree_limit)).astype(np.float16)

print('============================ start training Catboost ==================================')
from catboost import Pool,CatBoostRegressor
model = CatBoostRegressor(task_type = 'GPU', iterations=22222, learning_rate=0.1, random_seed = 7, depth=7, eval_metric='RMSE')
train_dataset = Pool(x1,  y1)
eval_dataset = Pool(x2,  y2)
model.fit(train_dataset, eval_set=eval_dataset, verbose=100, early_stopping_rounds=250)
#preds_cb = (model.predict(test[col])).astype(np.float16)
oof_valid_cb = (model.predict(x2)).astype(np.float16)
print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_cb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
del train_dataset, eval_dataset; gc.collect()
valid_predict_CAT = valid_predict_CAT + oof_valid_cb.tolist()
valid_target_CAT = valid_target_CAT + np.array(y2).tolist()

print('============================ start Catboost predict==================================')
a = 2 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),4] = (model.predict(test_batch)).astype(np.float16)

a = 6
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),4] = (model.predict(test_batch)).astype(np.float16)


# ## Model 10 Open Channel

# In[ ]:


for j,idx in enumerate(model_batch[4]):
    if j ==0:
        batch = idx; a = 500000*(batch-1); b = 500000*batch
        train_model_split = train2[col].loc[train2.index[a:b]]
        train_model_target = train2.loc[train2.index[a:b],'open_channels']
    else:
        batch = idx; a = 500000*(batch-1); b = 500000*batch
        train_model_split1 = train2[col].loc[train2.index[a:b]]
        train_model_target1 = train2.loc[train2.index[a:b],'open_channels']
        train_model_split = np.concatenate((train_model_split,train_model_split1),axis = 0)
        train_model_target = np.concatenate((train_model_target,train_model_target1),axis = 0)

x1, x2, y1, y2 = model_selection.train_test_split(train_model_split, train_model_target , test_size=0.2, random_state=49)

print('============================ start training LGBM ==================================')
#Fifth model
import lightgbm as lgb
params = {'learning_rate': 0.1, 'max_depth': -1, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1, 'sample_fraction':0.33} 
model = lgb.train(params, lgb.Dataset(x1, y1), 22222,  lgb.Dataset(x2, y2), verbose_eval=100, early_stopping_rounds=250, feval=MacroF1Metric)
#preds_lgb = (model.predict(test[col], num_iteration=model.best_iteration)).astype(np.float16)
oof_valid_lgb = (model.predict(x2, num_iteration=model.best_iteration)).astype(np.float16)

print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_lgb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
valid_predict_LGBM = valid_predict_LGBM + oof_valid_lgb.tolist()
valid_target_LGBM = valid_target_LGBM + np.array(y2).tolist()

print('============================ start LGBM predict==================================')
a = 5 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),2] = (model.predict(test_batch, num_iteration=model.best_iteration)).astype(np.float16)

a = 7 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),2] = (model.predict(test_batch, num_iteration=model.best_iteration)).astype(np.float16)


gc.collect()

print('============================ start training XGBBOOST ==================================')
import xgboost as xgb
params = {'colsample_bytree': 0.375,'learning_rate': 0.1,'max_depth': 10, 'subsample': 1, 'objective':'reg:squarederror',
          'eval_metric':'rmse', 'n_estimators':22222,   'tree_method':'gpu_hist',}
train_set = xgb.DMatrix(x1, y1)
val_set = xgb.DMatrix(x2, y2)
model = xgb.train(params, train_set, num_boost_round=2222, evals=[(train_set, 'train'), (val_set, 'val')], 
                         verbose_eval=100, early_stopping_rounds=250)
#preds_xgb = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit)
oof_valid_xgb = (model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit)).astype(np.float16)
print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_xgb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
del train_set, val_set; gc.collect()
valid_predict_XGB = valid_predict_XGB + oof_valid_xgb.tolist()
valid_target_XGB = valid_target_XGB + np.array(y2).tolist()

print('============================ start XGB predict==================================')
a = 5 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),3] = (model.predict(xgb.DMatrix(np.array(test_batch)), ntree_limit=model.best_ntree_limit)).astype(np.float16)

a = 7 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),3] = (model.predict(xgb.DMatrix(np.array(test_batch)), ntree_limit=model.best_ntree_limit)).astype(np.float16)

print('============================ start training Catboost ==================================')
from catboost import Pool,CatBoostRegressor
model = CatBoostRegressor(task_type = 'GPU', iterations=22222, learning_rate=0.1, random_seed = 7, depth=7, eval_metric='RMSE')
train_dataset = Pool(x1,  y1)          
eval_dataset = Pool(x2,  y2)
model.fit(train_dataset, eval_set=eval_dataset, verbose=100, early_stopping_rounds=250)
#preds_cb = (model.predict(test[col])).astype(np.float16)
oof_valid_cb = (model.predict(x2)).astype(np.float16)
print("BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(oof_valid_cb, 0, 10)).astype(int),np.array(y2) ,average = 'macro'))
del train_dataset, eval_dataset; gc.collect()
valid_predict_CAT = valid_predict_CAT + oof_valid_cb.tolist()
valid_target_CAT = valid_target_CAT + np.array(y2).tolist()

print('============================ start Catboost predict==================================')
a = 5 
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),4] = (model.predict(test_batch)).astype(np.float16)

a = 7
test_batch = test[col].loc[test.index[100000*a:100000*(a+1)]]
print ('start test subsample {}'.format(a)+', test_batch shape= ' + str(test_batch.shape))
sub.iloc[100000*a:100000*(a+1),4] = (model.predict(test_batch)).astype(np.float16)


# In[ ]:


print("LGBM BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(valid_predict_LGBM, 0, 10)).astype(int),np.array(valid_target_LGBM) ,average = 'macro'))
print("XGB BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(valid_predict_XGB, 0, 10)).astype(int),np.array(valid_target_XGB) ,average = 'macro'))
print("XGB BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(valid_predict_CAT, 0, 10)).astype(int),np.array(valid_target_CAT) ,average = 'macro'))


# In[ ]:


Ensemble_predict =np.array(valid_predict_LGBM)*0.4+np.array(valid_predict_XGB)*0.4+np.array(valid_predict_CAT)*0.2
print("Ensemble BEST VALIDATION_SCORE (F1): ", f1_score(np.round(np.clip(Ensemble_predict, 0, 10)).astype(int),np.array(valid_target_XGB) ,average = 'macro'))


# ## Optimize the thresholds

# In[ ]:


from functools import partial
import scipy as sp
class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        return -f1_score(y, X_p,average = 'macro')

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


    def coefficients(self):
        return self.coef_['x']


# In[ ]:


sub.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'optR = OptimizedRounder()\noptR.fit(np.array(Ensemble_predict).reshape(-1,), np.array(valid_target_XGB))\ncoefficients = optR.coefficients()\nprint(coefficients)')


# In[ ]:


opt_preds = optR.predict(np.array(Ensemble_predict).reshape(-1, ), coefficients)
print('f1', metrics.f1_score(np.array(valid_target_XGB), opt_preds, average = 'macro'))


# In[ ]:


predict_logic = sub.LGBM_preds*0.4+sub.XGB_preds*0.4+sub.CAT_preds*0.2
opt_test_preds = optR.predict(np.array(predict_logic).reshape(-1, ), coefficients)


# In[ ]:


sub.head()


# In[ ]:


submit = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')
submit.open_channels = opt_test_preds.astype(int)
submit.to_csv('submission_lgb.csv', index=False, float_format='%.4f')


# ## Show all test target and signal

# In[ ]:


def plot_time_channel_data(data_df, title="Time variation data"):
    plt.figure(figsize=(18,8))
    plt.plot(data_df["time"], data_df["signal"], color='b', label='Signal')
    plt.plot(data_df["time"], data_df["open_channels"], color='r', label='Open channel')
    plt.title(title, fontsize=24)
    plt.xlabel("Time [sec]", fontsize=20)
    plt.ylabel("Signal & Open channel data", fontsize=20)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


# In[ ]:


test2['open_channels'] = submit['open_channels'].astype(int)


# In[ ]:


test2.open_channels.head()


# In[ ]:


plot_time_channel_data(test2[:100000],'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:


plot_time_channel_data(test2[100000:200000],'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:


plot_time_channel_data(test2[200000:300000],'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:


plot_time_channel_data(test2[300000:400000],'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:


plot_time_channel_data(test2[400000:500000],'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:


plot_time_channel_data(test2[500000:600000],'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:


plot_time_channel_data(test2[600000:700000],'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:


plot_time_channel_data(test2[700000:800000],'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:


plot_time_channel_data(test2[800000:900000],'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:


plot_time_channel_data(test2[900000:1000000],'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:


plot_time_channel_data(test2[1000000:2000000],'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:


plot_time_channel_data(test2,'Train data: signal & open channel data (0.7-0.75 sec.)')


# In[ ]:




